from __future__ import annotations

import types
import unittest

try:
    from sglang.srt.speculative.decoupled_spec_io import (
        DraftClose,
        DraftSync,
        DraftTailStreamOutput,
        RequestTerminateReason,
        VerifyCommit,
    )
    from sglang.srt.speculative.draft_tail_buffer import (
        DraftTailBuffer,
        DraftTailSnapshot,
    )

    _HAS_DRAFT_TAIL_BUFFER_DEPS = True
except ModuleNotFoundError:
    _HAS_DRAFT_TAIL_BUFFER_DEPS = False


def make_sync(request_id: str = "req-0", committed_len: int = 3) -> DraftSync:
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
) -> DraftTailStreamOutput:
    if token_pos is None:
        token_pos = committed_len
    return DraftTailStreamOutput(
        request_id=request_id,
        src_drafter_rank=0,
        dst_verifier_rank=0,
        base_committed_len=committed_len,
        new_token_pos=token_pos,
        new_token_id=token_id,
    )


@unittest.skipUnless(
    _HAS_DRAFT_TAIL_BUFFER_DEPS,
    "draft tail buffer dependencies are not importable",
)
class TestDraftTailBuffer(unittest.TestCase):
    def test_push_batch_accepts_duplicate_and_rejects_gap(self):
        buffer = DraftTailBuffer(verifier_rank=0, required_tail_len=2)
        buffer.open_requests([make_sync()])

        buffer.append_draft_stream(
            [
                make_stream_output(11),
                make_stream_output(22, token_pos=4),
                make_stream_output(22, token_pos=4),
                make_stream_output(33, token_pos=5),
                make_stream_output(55, token_pos=7),
            ]
        )

        req = types.SimpleNamespace(rid="req-0")
        self.assertEqual(
            buffer.get_draft_snapshots([req], allow_partial=False),
            [DraftTailSnapshot("req-0", 3, [11, 22])],
        )

    def test_apply_commit_preserves_suffix_when_bonus_matches(self):
        buffer = DraftTailBuffer(verifier_rank=0, required_tail_len=1)
        buffer.open_requests([make_sync()])
        buffer.append_draft_stream(
            [
                make_stream_output(11),
                make_stream_output(22, token_pos=4),
                make_stream_output(33, token_pos=5),
            ]
        )

        buffer.apply_verify_commits(
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

        self.assertEqual(
            buffer.get_draft_snapshots(
                [types.SimpleNamespace(rid="req-0")],
                allow_partial=True,
            ),
            [DraftTailSnapshot("req-0", 5, [])],
        )

        buffer.append_draft_stream([make_stream_output(44, committed_len=3, token_pos=6)])
        self.assertEqual(
            buffer.get_draft_snapshots(
                [types.SimpleNamespace(rid="req-0")],
                allow_partial=True,
            ),
            [DraftTailSnapshot("req-0", 5, [33])],
        )

    def test_apply_commit_rejects_consuming_all_buffered_tail_tokens(self):
        buffer = DraftTailBuffer(verifier_rank=0, required_tail_len=1)
        buffer.open_requests([make_sync()])
        buffer.append_draft_stream(
            [
                make_stream_output(11),
                make_stream_output(22, token_pos=4),
            ]
        )

        with self.assertRaises(RuntimeError):
            buffer.apply_verify_commits(
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

    def test_stale_base_can_continue_after_local_commit_match(self):
        buffer = DraftTailBuffer(verifier_rank=0, required_tail_len=2)
        buffer.open_requests([make_sync()])
        buffer.append_draft_stream(
            [
                make_stream_output(11),
                make_stream_output(22, token_pos=4),
                make_stream_output(33, token_pos=5),
            ]
        )
        buffer.apply_verify_commits(
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

        buffer.append_draft_stream(
            [
                make_stream_output(44, committed_len=3, token_pos=6),
                make_stream_output(55, committed_len=3, token_pos=7),
            ]
        )

        self.assertEqual(
            buffer.get_draft_snapshots(
                [types.SimpleNamespace(rid="req-0")],
                allow_partial=False,
            ),
            [DraftTailSnapshot("req-0", 5, [33, 44])],
        )

    def test_stale_base_survives_multiple_bonus_matches(self):
        buffer = DraftTailBuffer(verifier_rank=0, required_tail_len=1)
        buffer.open_requests([make_sync()])
        buffer.append_draft_stream(
            [
                make_stream_output(11),
                make_stream_output(22, token_pos=4),
                make_stream_output(33, token_pos=5),
            ]
        )
        buffer.apply_verify_commits(
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
        buffer.append_draft_stream(
            [
                make_stream_output(44, committed_len=3, token_pos=6),
                make_stream_output(55, committed_len=3, token_pos=7),
            ]
        )
        buffer.apply_verify_commits(
            [
                VerifyCommit(
                    request_id="req-0",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    pre_verify_committed_len=5,
                    bonus_token_pos=6,
                    bonus_token_id=44,
                )
            ]
        )

        buffer.append_draft_stream([make_stream_output(66, committed_len=3, token_pos=8)])

        self.assertEqual(
            buffer.get_draft_snapshots(
                [types.SimpleNamespace(rid="req-0")],
                allow_partial=False,
            ),
            [DraftTailSnapshot("req-0", 7, [55])],
        )

    def test_close_clears_state_and_ignores_late_stream_outputs(self):
        buffer = DraftTailBuffer(verifier_rank=0, required_tail_len=1)
        buffer.open_requests([make_sync()])
        buffer.close_requests(
            [
                DraftClose(
                    request_id="req-0",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    reason=RequestTerminateReason.ABORT,
                )
            ]
        )

        self.assertFalse(buffer.has_request("req-0"))

        buffer.append_draft_stream([make_stream_output(11)])
        self.assertFalse(buffer.has_request("req-0"))

    def test_stream_output_does_not_create_unknown_request_state(self):
        buffer = DraftTailBuffer(verifier_rank=0, required_tail_len=1)

        buffer.append_draft_stream([make_stream_output(11)])

        self.assertFalse(buffer.has_request("req-0"))

    def test_snapshot_for_verify_returns_copied_partial_tail(self):
        buffer = DraftTailBuffer(verifier_rank=0, required_tail_len=4)
        buffer.open_requests([make_sync()])
        buffer.append_draft_stream(
            [
                make_stream_output(11),
                make_stream_output(22, token_pos=4),
            ]
        )

        req = types.SimpleNamespace(rid="req-0")
        snapshots = buffer.get_draft_snapshots([req], allow_partial=True)
        self.assertEqual(
            snapshots,
            [DraftTailSnapshot("req-0", 3, [11])],
        )

        snapshots[0].tail_tokens.append(99)
        buffer.append_draft_stream([make_stream_output(33, token_pos=5)])

        next_snapshots = buffer.get_draft_snapshots([req], allow_partial=True)
        self.assertEqual(next_snapshots[0].tail_tokens, [11, 22])

    def test_snapshot_for_verify_can_publish_empty_partial_tail(self):
        buffer = DraftTailBuffer(verifier_rank=0, required_tail_len=2)
        buffer.open_requests([make_sync()])

        snapshots = buffer.get_draft_snapshots(
            [types.SimpleNamespace(rid="req-0")],
            allow_partial=True,
        )

        self.assertEqual(snapshots, [DraftTailSnapshot("req-0", 3, [])])

    def test_snapshot_for_verify_can_require_full_tail(self):
        buffer = DraftTailBuffer(verifier_rank=0, required_tail_len=2)
        buffer.open_requests([make_sync()])
        buffer.append_draft_stream([make_stream_output(11)])

        req = types.SimpleNamespace(rid="req-0")
        self.assertEqual(buffer.get_draft_snapshots([req], allow_partial=False), [])

        buffer.append_draft_stream([make_stream_output(22, token_pos=4)])
        self.assertEqual(buffer.get_draft_snapshots([req], allow_partial=False), [])

        buffer.append_draft_stream([make_stream_output(33, token_pos=5)])
        snapshots = buffer.get_draft_snapshots([req], allow_partial=False)
        self.assertEqual(snapshots, [DraftTailSnapshot("req-0", 3, [11, 22])])


if __name__ == "__main__":
    unittest.main()
