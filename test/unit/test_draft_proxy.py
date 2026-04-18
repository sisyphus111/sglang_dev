from __future__ import annotations

import queue
import unittest

try:
    from sglang.srt.speculative.decoupled_spec_io import (
        DraftClose,
        DraftControlBatch,
        DraftSync,
        DraftTailStreamOutput,
        RequestTerminateReason,
        VerifyCommit,
    )
    from sglang.srt.speculative.draft_proxy import DraftProxyThread
    from sglang.srt.speculative.draft_tail_buffer import DraftTailBuffer, DraftTailSnapshot

    _HAS_DECOUPLED_PROXY_DEPS = True
except ModuleNotFoundError:
    _HAS_DECOUPLED_PROXY_DEPS = False


def make_proxy() -> DraftProxyThread:
    proxy = DraftProxyThread.__new__(DraftProxyThread)
    proxy.verifier_rank = 0
    proxy.draft_tail_buffer = DraftTailBuffer(verifier_rank=0, required_tail_len=0)
    proxy._send_queue = queue.SimpleQueue()
    return proxy


def make_sync(request_id: str = "req-0", committed_len: int = 3):
    return DraftSync(
        request_id=request_id,
        src_verifier_rank=0,
        dst_drafter_rank=0,
        committed_output_ids=[100 + i for i in range(committed_len)],
    )


def make_stream_output(
    token_id: int,
    *,
    request_id: str = "req-0",
    committed_len: int = 3,
    token_pos: int | None = None,
    dst_verifier_rank: int = 0,
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
    _HAS_DECOUPLED_PROXY_DEPS,
    "decoupled proxy dependencies are not importable",
)
class TestDraftProxyThread(unittest.TestCase):
    def test_get_draft_snapshots_returns_tail_snapshot_in_request_order(self):
        proxy = make_proxy()
        proxy.draft_tail_buffer.open_requests(
            [make_sync("req-0", 3), make_sync("req-1", 4)]
        )
        proxy.draft_tail_buffer.append_draft_stream(
            [
                make_stream_output(33, request_id="req-1", committed_len=4),
                make_stream_output(44, request_id="req-1", committed_len=4, token_pos=5),
                make_stream_output(55, request_id="req-1", committed_len=4, token_pos=6),
                make_stream_output(11, request_id="req-0", committed_len=3),
                make_stream_output(22, request_id="req-0", committed_len=3, token_pos=4),
                make_stream_output(33, request_id="req-0", committed_len=3, token_pos=5),
            ]
        )

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

    def test_merge_stream_outputs_accepts_duplicate_and_rejects_gaps(self):
        proxy = make_proxy()
        proxy.draft_tail_buffer.open_requests([make_sync("req-0", 3)])

        proxy.draft_tail_buffer.append_draft_stream(
            [
                make_stream_output(11),
                make_stream_output(22, token_pos=4),
                make_stream_output(22, token_pos=4),
                make_stream_output(33, token_pos=5),
                make_stream_output(55, token_pos=7),
            ]
        )

        snapshots = proxy.draft_tail_buffer.get_draft_snapshots(
            [type("_Req", (), {"rid": "req-0"})()],
            allow_partial=False,
        )
        self.assertEqual(snapshots, [DraftTailSnapshot("req-0", 3, [11, 22])])

    def test_verify_commit_preserves_unconsumed_suffix_after_bonus_match(self):
        proxy = make_proxy()
        proxy.draft_tail_buffer.open_requests([make_sync("req-0", 3)])
        proxy.draft_tail_buffer.append_draft_stream(
            [
                make_stream_output(11),
                make_stream_output(22, token_pos=4),
                make_stream_output(33, token_pos=5),
                make_stream_output(44, token_pos=6),
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
            [make_stream_output(55, committed_len=3, token_pos=7)]
        )

        snapshots = proxy.draft_tail_buffer.get_draft_snapshots(
            [type("_Req", (), {"rid": "req-0"})()],
            allow_partial=False,
        )
        self.assertEqual(snapshots, [DraftTailSnapshot("req-0", 6, [44])])

    def test_merge_accepts_contiguous_stale_base_after_local_commit(self):
        proxy = make_proxy()
        proxy.draft_tail_buffer.open_requests([make_sync("req-0", 3)])
        proxy.draft_tail_buffer.append_draft_stream(
            [
                make_stream_output(11),
                make_stream_output(22, token_pos=4),
                make_stream_output(33, token_pos=5),
            ]
        )

        proxy.draft_tail_buffer.apply_verify_commits(
            [
                VerifyCommit(
                    request_id="req-0",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    pre_verify_committed_len=3,
                    bonus_token_pos=4,
                    bonus_token_id=22,
                )
            ]
        )
        proxy.draft_tail_buffer.append_draft_stream(
            [
                make_stream_output(44, committed_len=3, token_pos=6),
                make_stream_output(55, committed_len=3, token_pos=7),
            ]
        )

        snapshots = proxy.draft_tail_buffer.get_draft_snapshots(
            [type("_Req", (), {"rid": "req-0"})()],
            allow_partial=False,
        )
        self.assertEqual(snapshots, [DraftTailSnapshot("req-0", 5, [33, 44])])

    def test_merge_rejects_stale_base_after_bonus_mismatch(self):
        proxy = make_proxy()
        proxy.draft_tail_buffer.open_requests([make_sync("req-0", 3)])
        proxy.draft_tail_buffer.append_draft_stream(
            [
                make_stream_output(11),
                make_stream_output(99, token_pos=4),
            ]
        )

        proxy.draft_tail_buffer.apply_verify_commits(
            [
                VerifyCommit(
                    request_id="req-0",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    pre_verify_committed_len=3,
                    bonus_token_pos=4,
                    bonus_token_id=22,
                )
            ]
        )
        proxy.draft_tail_buffer.append_draft_stream(
            [make_stream_output(44, committed_len=3, token_pos=5)]
        )

        snapshots = proxy.draft_tail_buffer.get_draft_snapshots(
            [type("_Req", (), {"rid": "req-0"})()],
            allow_partial=True,
        )
        self.assertEqual(snapshots, [DraftTailSnapshot("req-0", 5, [])])

    def test_close_clears_state_and_unblocks_wait_with_error(self):
        proxy = make_proxy()
        proxy.draft_tail_buffer.open_requests([make_sync("req-0", 3)])
        proxy.draft_tail_buffer.close_requests(
            [
                DraftClose(
                    request_id="req-0",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    reason=RequestTerminateReason.ABORT,
                )
            ]
        )
        proxy.draft_tail_buffer.close()

        self.assertFalse(proxy.draft_tail_buffer.has_request("req-0"))

    def test_submit_control_batch_updates_local_state_and_queues_payload(self):
        proxy = make_proxy()
        batch = DraftControlBatch(
            dst_drafter_rank=0,
            close_messages=[
                DraftClose(
                    request_id="req-0",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    reason=RequestTerminateReason.ABORT,
                )
            ],
        )
        proxy.draft_tail_buffer.open_requests([make_sync("req-0", 3)])

        proxy.submit_control_batch(batch)

        self.assertFalse(proxy.draft_tail_buffer.has_request("req-0"))
        queued = proxy._send_queue.get_nowait()
        self.assertIs(queued.control_batch, batch)


if __name__ == "__main__":
    unittest.main()
