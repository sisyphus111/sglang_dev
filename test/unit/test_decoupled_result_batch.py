from __future__ import annotations

import time
import unittest

try:
    import zmq

    from sglang.srt.speculative.decoupled_spec_io import (
        DraftClose,
        DraftControlBatch,
        DraftMeshMessage,
        DraftMeshMessageType,
        DraftSync,
        DraftTailStreamOutput,
        DraftTailStreamOutputBatch,
        RequestTerminateReason,
        VerifyCommit,
    )
    from sglang.srt.speculative.draft_adapter import DraftAdapterThread
    from sglang.srt.speculative.draft_proxy import DraftProxyThread
    from sglang.srt.speculative.draft_tail_buffer import DraftTailBuffer, DraftTailSnapshot

    _HAS_DECOUPLED_BATCH_DEPS = True
except ModuleNotFoundError:
    _HAS_DECOUPLED_BATCH_DEPS = False


class FakeSocket:
    def __init__(self):
        self.messages = []
        self.closed = False

    def send_pyobj(self, message):
        self.messages.append(message)

    def close(self, linger=0):
        del linger
        self.closed = True


class FakeRecvSocket:
    def __init__(self, messages):
        self.messages = list(messages)

    def recv_pyobj(self, flags=0):
        del flags
        if not self.messages:
            raise zmq.ZMQError()
        return self.messages.pop(0)


def wait_for(predicate, *, timeout_s: float = 1.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def make_stream_output(
    request_id: str,
    committed_len: int,
    token_id: int,
    *,
    dst_verifier_rank: int = 0,
    token_pos: int | None = None,
) -> DraftTailStreamOutput:
    if token_pos is None:
        token_pos = committed_len
    return DraftTailStreamOutput(
        request_id=request_id,
        src_drafter_rank=0,
        dst_verifier_rank=dst_verifier_rank,
        base_committed_len=committed_len,
        new_token_pos=token_pos,
        new_token_id=token_id,
    )


@unittest.skipUnless(
    _HAS_DECOUPLED_BATCH_DEPS,
    "decoupled result batch dependencies are not importable",
)
class TestDecoupledResultBatch(unittest.TestCase):
    def test_submit_draft_results_batches_by_verifier_rank(self):
        socket0 = FakeSocket()
        socket1 = FakeSocket()
        adapter = DraftAdapterThread()
        adapter.result_send_sockets = {0: socket0, 1: socket1}
        adapter.start()

        update0 = make_stream_output("req-0", 3, 11, dst_verifier_rank=0)
        update1 = make_stream_output("req-1", 4, 33, dst_verifier_rank=0)
        update2 = make_stream_output("req-2", 5, 55, dst_verifier_rank=1)

        try:
            adapter.submit_draft_results([update0, update1, update2])
            self.assertTrue(
                wait_for(
                    lambda: len(socket0.messages) == 1
                    and len(socket1.messages) == 1
                )
            )
        finally:
            adapter.close()

        self.assertEqual(len(socket0.messages), 1)
        self.assertEqual(
            socket0.messages[0].message_type,
            DraftMeshMessageType.TAIL_STREAM_OUTPUT_BATCH,
        )
        self.assertEqual(
            socket0.messages[0].tail_stream_output_batch.dst_verifier_rank,
            0,
        )
        self.assertEqual(
            socket0.messages[0].tail_stream_output_batch.stream_outputs,
            [update0, update1],
        )

        self.assertEqual(len(socket1.messages), 1)
        self.assertEqual(
            socket1.messages[0].message_type,
            DraftMeshMessageType.TAIL_STREAM_OUTPUT_BATCH,
        )
        self.assertEqual(
            socket1.messages[0].tail_stream_output_batch.dst_verifier_rank,
            1,
        )
        self.assertEqual(
            socket1.messages[0].tail_stream_output_batch.stream_outputs,
            [update2],
        )

    def test_adapter_drain_sync_keeps_commit_and_close_pending(self):
        adapter = DraftAdapterThread()
        sync = DraftSync(
            request_id="req-0",
            src_verifier_rank=0,
            dst_drafter_rank=0,
            committed_output_ids=[1, 2, 3],
        )
        commit = VerifyCommit(
            request_id="req-0",
            src_verifier_rank=0,
            dst_drafter_rank=0,
            pre_verify_committed_len=3,
            bonus_token_pos=3,
            bonus_token_id=11,
        )
        close = DraftClose(
            request_id="req-0",
            src_verifier_rank=0,
            dst_drafter_rank=0,
            reason=RequestTerminateReason.FINISHED,
        )
        adapter.control_recv_socket = FakeRecvSocket(
            [
                DraftMeshMessage.from_control_batch(
                    DraftControlBatch(
                        dst_drafter_rank=0,
                        sync_messages=[sync],
                        verify_commit_messages=[commit],
                        close_messages=[close],
                    )
                )
            ]
        )
        self.assertTrue(adapter._drain_control_socket())

        self.assertEqual(adapter.drain_sync_messages(), [sync])
        self.assertEqual(adapter.drain_sync_messages(), [])
        self.assertEqual(adapter.drain_post_result_messages(), [commit, close])

    def test_adapter_close_stops_thread_and_closes_sockets(self):
        socket = FakeSocket()
        adapter = DraftAdapterThread()
        adapter.result_send_sockets = {0: socket}
        adapter.start()

        adapter.close()

        self.assertFalse(adapter._thread.is_alive())
        self.assertTrue(socket.closed)

    def test_proxy_merges_stream_outputs_and_waits_by_requested_order(self):
        proxy = DraftProxyThread.__new__(DraftProxyThread)
        proxy.verifier_rank = 0
        proxy.draft_tail_buffer = DraftTailBuffer(verifier_rank=0, required_tail_len=0)
        proxy.draft_tail_buffer.open_requests(
            [
                DraftSync(
                    request_id="req-0",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    committed_output_ids=[1, 2, 3],
                ),
                DraftSync(
                    request_id="req-1",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    committed_output_ids=[1, 2, 3, 4],
                ),
            ]
        )

        update0 = make_stream_output("req-0", 3, 11)
        update1 = make_stream_output("req-1", 4, 33)
        update2 = make_stream_output("req-0", 3, 22, token_pos=4)
        update3 = make_stream_output("req-1", 4, 44, token_pos=5)
        update4 = make_stream_output("req-0", 3, 99, token_pos=5)
        update5 = make_stream_output("req-1", 4, 100, token_pos=6)
        message = DraftMeshMessage.from_tail_stream_output_batch(
            DraftTailStreamOutputBatch(
                dst_verifier_rank=0,
                stream_outputs=[update1, update3, update5, update0, update2, update4],
            )
        )

        proxy.result_recv_socket = FakeRecvSocket([message])
        proxy._recv_tail_stream_output()

        snapshots = proxy.draft_tail_buffer.get_draft_snapshots(
            [
                type("_Req", (), {"rid": "req-0"})(),
                type("_Req", (), {"rid": "req-1"})(),
            ],
            allow_partial=False,
        )

        self.assertEqual(
            snapshots,
            [
                DraftTailSnapshot("req-0", 3, [11, 22]),
                DraftTailSnapshot("req-1", 4, [33, 44]),
            ],
        )

    def test_proxy_verify_commit_preserves_suffix_on_full_accept(self):
        proxy = DraftProxyThread.__new__(DraftProxyThread)
        proxy.verifier_rank = 0
        proxy.draft_tail_buffer = DraftTailBuffer(verifier_rank=0, required_tail_len=0)

        proxy.draft_tail_buffer.open_requests(
            [
                DraftSync(
                    request_id="req-0",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    committed_output_ids=[1, 2, 3],
                )
            ]
        )
        proxy.draft_tail_buffer.append_draft_stream(
            [
                make_stream_output("req-0", 3, 11),
                make_stream_output("req-0", 3, 22, token_pos=4),
                make_stream_output("req-0", 3, 33, token_pos=5),
                make_stream_output("req-0", 3, 44, token_pos=6),
            ]
        )

        proxy.draft_tail_buffer.apply_verify_commits(
            [
                VerifyCommit(
                    request_id="req-0",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    pre_verify_committed_len=3,
                    bonus_token_pos=5,
                    bonus_token_id=33,
                )
            ]
        )

        proxy.draft_tail_buffer.append_draft_stream(
            [make_stream_output("req-0", 3, 55, token_pos=7)]
        )

        snapshots = proxy.draft_tail_buffer.get_draft_snapshots(
            [type("_Req", (), {"rid": "req-0"})()],
            allow_partial=False,
        )
        self.assertEqual(snapshots, [DraftTailSnapshot("req-0", 6, [44])])


if __name__ == "__main__":
    unittest.main()
