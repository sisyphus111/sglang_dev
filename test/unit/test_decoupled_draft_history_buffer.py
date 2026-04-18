from __future__ import annotations

from collections import deque
import types
import unittest

import torch

from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.speculative.decoupled_spec_io import (
    DraftReqKey,
    DraftSync,
    DraftTailStreamOutput,
    VerifyCommit,
)
from sglang.srt.speculative.draft_tail_buffer import DraftTailBuffer


class _FakeReqToTokenPool:
    def __init__(self):
        self.req_to_token = torch.tensor(
            [[1000, 1001, 1002, 1003, 1004, 1005, 1006]], dtype=torch.int64
        )


class _FakeTokenAllocator:
    def __init__(self):
        self.freed: list[torch.Tensor] = []

    def free(self, indices: torch.Tensor):
        self.freed.append(indices.clone())


_DEFAULT_DRAFT_KEY = object()


class _FakeReq:
    def __init__(self, *, output_ids: list[int], draft_key=_DEFAULT_DRAFT_KEY):
        self.rid = "draft-req-0"
        self.draft_key = (
            DraftReqKey(src_verifier_rank=0, request_id="req-0")
            if draft_key is _DEFAULT_DRAFT_KEY
            else draft_key
        )
        self.origin_input_ids = [7]
        self.output_ids = list(output_ids)
        self.fill_ids = self.origin_input_ids + self.output_ids
        self.decoded_text = ""
        self.finished_reason = None
        self.finished_len = None
        self.finished_output = None
        self.to_finish = None
        self.return_logprob = False
        self.hidden_states = []
        self.grammar = None
        self.req_pool_idx = 0
        self.kv_committed_freed = False
        self.kv_overallocated_freed = False
        self.kv_allocated_len = len(self.origin_input_ids) + len(self.output_ids)
        self.kv_committed_len = self.kv_allocated_len
        self.cache_protected_len = self.kv_allocated_len
        self.verifier_committed_prefix_len = 0
        self.draft_pending_verify_commits = deque()
        self.is_retracted = False
        self.spec_verify_ct = 0
        self.spec_accepted_tokens = 0

    def finished(self):
        return False

    def update_spec_acceptance_histogram(self, accepted_draft_tokens: int):
        del accepted_draft_tokens


class _NoIterOutputIds:
    def __init__(self, values: list[int]):
        self._values = list(values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, index):
        return self._values[index]

    def __iter__(self):
        raise AssertionError("output_ids iteration should not be needed")


class _FakeSpecAlgorithm:
    def is_decoupled_draft(self):
        return True


class _FakeVerifySpecAlgorithm:
    def is_none(self):
        return False

    def is_decoupled_draft(self):
        return False

    def is_decoupled_verify(self):
        return True


class _FakeServerArgs:
    speculative_num_draft_tokens: int = 4
    disaggregation_decode_enable_offload_kvcache: bool = False
    decode_log_interval: int = 1
    enable_dp_attention: bool = False


class _FakeDecodeTokenAllocator:
    def __init__(self):
        self.group_begin_count = 0
        self.group_end_count = 0

    def free_group_begin(self):
        self.group_begin_count += 1

    def free_group_end(self):
        self.group_end_count += 1


class _FakeScheduler(SchedulerOutputProcessorMixin):
    def __init__(self):
        self.req_to_token_pool = _FakeReqToTokenPool()
        self.token_to_kv_pool_allocator = _FakeTokenAllocator()
        self.spec_algorithm = _FakeSpecAlgorithm()
        self.server_args = _FakeServerArgs()
        self.draft_adapter_thread = object()
        self.draft_req_table = {}
        self.waiting_queue = []
        self.running_batch = types.SimpleNamespace(reqs=[])
        self.page_size = 1
        self.disaggregation_mode = None
        self.pp_rank = 0
        self.attn_tp_rank = 0
        self.attn_tp_size = 1
        self.attn_cp_rank = 0
        self.attn_cp_size = 1
        self.tp_size = 1
        self.dp_rank = 0


class _FakeDecodeScheduler(SchedulerOutputProcessorMixin):
    def __init__(self):
        self.num_generated_tokens = 0
        self.enable_metrics = False
        self.metrics_collector = None
        self.token_to_kv_pool_allocator = _FakeDecodeTokenAllocator()
        self.enable_overlap = False
        self.server_args = _FakeServerArgs()
        self.decode_offload_manager = None
        self.tree_cache = None
        self.forward_ct_decode = 0
        self.current_scheduler_metrics_enabled = False
        self.spec_metric_calls: list[tuple[int, int]] = []
        self.streamed_reqs = None

    def update_spec_metrics(self, batch_size: int, num_accepted_tokens: int):
        self.spec_metric_calls.append((batch_size, num_accepted_tokens))

    def _is_decoupled_verify_entry_rank(self):
        return True

    def _mamba_prefix_cache_update(self, req, batch, result, i):
        del req, batch, result, i

    def maybe_collect_routed_experts(self, req):
        del req

    def maybe_collect_customized_info(self, i, req, logits_output):
        del i, req, logits_output

    def stream_output(self, reqs, return_logprob):
        self.streamed_reqs = (list(reqs), return_logprob)


class _ListTensor:
    def __init__(self, values: list[int]):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


class _FakeDecodeReq:
    def __init__(self, *, rid: str, output_ids: list[int], draft_buffer: list[int]):
        self.rid = rid
        self.output_ids = list(output_ids)
        self.draft_buffer = list(draft_buffer)
        self.is_retracted = False
        self.kv_committed_len = len(output_ids)
        self.kv_allocated_len = len(output_ids)
        self.spec_verify_ct = 0
        self.spec_accepted_tokens = 0
        self.spec_acceptance_histogram: dict[int, int] = {}
        self.return_logprob = False
        self.return_hidden_states = False
        self.hidden_states = []
        self.grammar = None
        self.finished_reason = None
        self.time_stats = types.SimpleNamespace(completion_time=None)
        self.checked_accept_lengths: list[int] = []

    def finished(self):
        return False

    def check_finished(self, new_accepted_len: int):
        self.checked_accept_lengths.append(new_accepted_len)

    def update_spec_acceptance_histogram(self, accepted_draft_tokens: int):
        self.spec_acceptance_histogram[accepted_draft_tokens] = (
            self.spec_acceptance_histogram.get(accepted_draft_tokens, 0) + 1
        )


def _make_verify_decode_batch(reqs: list[_FakeDecodeReq]):
    return types.SimpleNamespace(
        reqs=reqs,
        spec_algorithm=_FakeVerifySpecAlgorithm(),
        is_spec_v2=False,
        return_logprob=False,
        batch_size=lambda: len(reqs),
    )


def _make_verify_decode_result(
    *,
    verified_token_ids: list[int],
    accept_lengths: list[int],
    num_accepted_tokens: int,
):
    return types.SimpleNamespace(
        copy_done=None,
        logits_output=types.SimpleNamespace(hidden_states=None),
        next_token_ids=_ListTensor(verified_token_ids),
        can_run_cuda_graph=False,
        accept_length_per_req_cpu=accept_lengths,
        num_accepted_tokens=num_accepted_tokens,
    )


def _make_draft_decode_batch(req: _FakeReq):
    seq_len = len(req.origin_input_ids) + len(req.output_ids)
    return types.SimpleNamespace(
        seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int64),
        seq_lens=torch.tensor([seq_len], dtype=torch.int64),
        orig_seq_lens=torch.tensor([seq_len], dtype=torch.int64),
        output_ids=[
            req.output_ids[-1] if req.output_ids else req.origin_input_ids[-1]
        ],
        seq_lens_sum=seq_len,
    )


class TestDecoupledDraftState(unittest.TestCase):
    def test_verify_commit_preserves_matching_suffix(self):
        scheduler = _FakeScheduler()
        req = _FakeReq(output_ids=[11, 22, 33, 44])

        batch = types.SimpleNamespace(
            seq_lens_cpu=torch.tensor([5], dtype=torch.int64),
            seq_lens=torch.tensor([5], dtype=torch.int64),
            orig_seq_lens=torch.tensor([5], dtype=torch.int64),
            output_ids=[44],
            seq_lens_sum=5,
        )

        scheduler._draft_apply_verify_commit(
            req,
            VerifyCommit(
                request_id="req-0",
                src_verifier_rank=0,
                dst_drafter_rank=0,
                pre_verify_committed_len=0,
                bonus_token_pos=1,
                bonus_token_id=22,
            ),
            batch=batch,
            req_batch_idx=0,
        )

        self.assertEqual(req.output_ids, [11, 22, 33, 44])
        self.assertEqual(batch.seq_lens_cpu.tolist(), [5])
        self.assertEqual(batch.seq_lens.tolist(), [5])
        self.assertEqual(batch.orig_seq_lens.tolist(), [5])
        self.assertEqual(batch.output_ids, [44])
        self.assertEqual(batch.seq_lens_sum, 5)
        self.assertEqual(len(scheduler.token_to_kv_pool_allocator.freed), 0)

    def test_verify_commit_mismatch_rewrites_without_history_restore(self):
        scheduler = _FakeScheduler()
        req = _FakeReq(output_ids=[11, 99, 33, 44])

        batch = types.SimpleNamespace(
            seq_lens_cpu=torch.tensor([4], dtype=torch.int64),
            seq_lens=torch.tensor([4], dtype=torch.int64),
            orig_seq_lens=torch.tensor([4], dtype=torch.int64),
            output_ids=[44],
            seq_lens_sum=4,
        )

        scheduler._draft_apply_verify_commit(
            req,
            VerifyCommit(
                request_id="req-0",
                src_verifier_rank=0,
                dst_drafter_rank=0,
                pre_verify_committed_len=0,
                bonus_token_pos=1,
                bonus_token_id=22,
            ),
            batch=batch,
            req_batch_idx=0,
        )

        self.assertEqual(req.output_ids, [11, 22])
        self.assertEqual(batch.seq_lens_cpu.tolist(), [2])
        self.assertEqual(batch.seq_lens.tolist(), [2])
        self.assertEqual(batch.orig_seq_lens.tolist(), [2])
        self.assertEqual(batch.output_ids, [22])
        self.assertEqual(batch.seq_lens_sum, 2)

    def test_verify_commit_beyond_decoded_tail_raises(self):
        scheduler = _FakeScheduler()
        req = _FakeReq(output_ids=[11, 22])

        with self.assertRaisesRegex(RuntimeError, "beyond its decoded tail"):
            scheduler._draft_apply_verify_commit(
                req,
                VerifyCommit(
                    request_id="req-0",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    pre_verify_committed_len=0,
                    bonus_token_pos=4,
                    bonus_token_id=55,
                ),
            )

    def test_verify_commit_build_skips_output_ids_copy(self):
        buffer = DraftTailBuffer(verifier_rank=0, required_tail_len=4)
        self.assertEqual(buffer.required_tail_len, 4)
        req = types.SimpleNamespace(
            rid="req-0",
            output_ids=_NoIterOutputIds([11, 22, 33]),
        )

        bonus_token_pos = len(req.output_ids) - 1
        message = VerifyCommit(
            request_id=req.rid,
            src_verifier_rank=0,
            dst_drafter_rank=0,
            pre_verify_committed_len=0,
            bonus_token_pos=bonus_token_pos,
            bonus_token_id=req.output_ids[bonus_token_pos],
        )

        self.assertEqual(message.pre_verify_committed_len, 0)
        self.assertEqual(message.bonus_token_pos, 2)
        self.assertEqual(message.bonus_token_id, 33)

    def test_collect_stream_output_publishes_current_decode_token(self):
        scheduler = _FakeScheduler()
        req = _FakeReq(output_ids=[11, 22, 33, 44, 55])
        req.verifier_committed_prefix_len = 2

        stream_output = scheduler._draft_apply_commits_and_maybe_emit(
            req,
            batch=_make_draft_decode_batch(req),
            req_batch_idx=0,
            decoded_token=(4, 55),
        )

        self.assertEqual(
            (
                stream_output.base_committed_len,
                stream_output.new_token_pos,
                stream_output.new_token_id,
            ),
            (2, 4, 55),
        )

    def test_collect_stream_output_applies_commit_before_publish(self):
        scheduler = _FakeScheduler()
        req = _FakeReq(output_ids=[11, 22, 33, 44, 55])
        req.verifier_committed_prefix_len = 0
        req.draft_pending_verify_commits.append(
            VerifyCommit(
                request_id="req-0",
                src_verifier_rank=0,
                dst_drafter_rank=0,
                pre_verify_committed_len=0,
                bonus_token_pos=1,
                bonus_token_id=22,
            )
        )

        stream_output = scheduler._draft_apply_commits_and_maybe_emit(
            req,
            batch=_make_draft_decode_batch(req),
            req_batch_idx=0,
            decoded_token=(4, 55),
        )

        self.assertEqual(req.verifier_committed_prefix_len, 2)
        self.assertEqual(
            (
                stream_output.base_committed_len,
                stream_output.new_token_pos,
                stream_output.new_token_id,
            ),
            (2, 4, 55),
        )

    def test_collect_stream_output_skips_invalidated_decode_token(self):
        scheduler = _FakeScheduler()
        req = _FakeReq(output_ids=[11, 99, 33, 44, 55])
        req.verifier_committed_prefix_len = 0
        req.draft_pending_verify_commits.append(
            VerifyCommit(
                request_id="req-0",
                src_verifier_rank=0,
                dst_drafter_rank=0,
                pre_verify_committed_len=0,
                bonus_token_pos=1,
                bonus_token_id=22,
            )
        )

        stream_output = scheduler._draft_apply_commits_and_maybe_emit(
            req,
            batch=_make_draft_decode_batch(req),
            req_batch_idx=0,
            decoded_token=(4, 55),
        )

        self.assertEqual(req.output_ids, [11, 22])
        self.assertIsNone(stream_output)

    def test_collect_stream_output_applies_all_pending_commits_in_order(self):
        scheduler = _FakeScheduler()
        req = _FakeReq(output_ids=[11, 99, 33, 44, 55])
        req.verifier_committed_prefix_len = 0
        req.draft_pending_verify_commits.extend(
            [
                VerifyCommit(
                    request_id="req-0",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    pre_verify_committed_len=0,
                    bonus_token_pos=1,
                    bonus_token_id=22,
                ),
                VerifyCommit(
                    request_id="req-0",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    pre_verify_committed_len=2,
                    bonus_token_pos=2,
                    bonus_token_id=33,
                ),
            ]
        )

        stream_output = scheduler._draft_apply_commits_and_maybe_emit(
            req,
            batch=_make_draft_decode_batch(req),
            req_batch_idx=0,
            decoded_token=(4, 55),
        )

        self.assertEqual(req.output_ids, [11, 22, 33])
        self.assertEqual(req.verifier_committed_prefix_len, 3)
        self.assertEqual(list(req.draft_pending_verify_commits), [])
        self.assertIsNone(stream_output)

    def test_non_entry_rank_does_not_emit_current_decode_token(self):
        scheduler = _FakeScheduler()
        scheduler.attn_tp_rank = 1
        scheduler.tp_size = 2
        req = _FakeReq(output_ids=[11, 22, 33])
        req.verifier_committed_prefix_len = 1

        stream_output = scheduler._draft_apply_commits_and_maybe_emit(
            req,
            batch=_make_draft_decode_batch(req),
            req_batch_idx=0,
            decoded_token=(2, 33),
        )

        self.assertIsNone(stream_output)

    def test_apply_sync_sets_committed_prefix(self):
        scheduler = _FakeScheduler()
        req = _FakeReq(output_ids=[11, 22], draft_key=None)
        message = DraftSync(
            request_id="req-0",
            src_verifier_rank=0,
            dst_drafter_rank=0,
            prompt_token_ids=[7],
            committed_output_ids=[11, 22],
        )

        scheduler._draft_apply_sync(req, message)

        self.assertEqual(req.output_ids, [11, 22])
        self.assertEqual(req.verifier_committed_prefix_len, 2)
        self.assertEqual(req.draft_key, message.draft_key)
        self.assertIs(scheduler.draft_req_table[message.draft_key], req)

    def test_coordinator_stream_output_merge_and_stale_drop(self):
        buffer = DraftTailBuffer(verifier_rank=0, required_tail_len=4)
        req = types.SimpleNamespace(
            rid="req-0",
            origin_input_ids=[7],
            output_ids=[11, 22, 33],
        )

        buffer.open_requests(
            [
                DraftSync(
                    request_id=req.rid,
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    prompt_token_ids=list(req.origin_input_ids),
                    committed_output_ids=list(req.output_ids),
                )
            ]
        )
        buffer.append_draft_stream(
            [
                DraftTailStreamOutput(
                    request_id="req-0",
                    src_drafter_rank=0,
                    dst_verifier_rank=0,
                    base_committed_len=3,
                    new_token_pos=3,
                    new_token_id=44,
                ),
                DraftTailStreamOutput(
                    request_id="req-0",
                    src_drafter_rank=0,
                    dst_verifier_rank=0,
                    base_committed_len=3,
                    new_token_pos=4,
                    new_token_id=55,
                ),
                DraftTailStreamOutput(
                    request_id="req-0",
                    src_drafter_rank=0,
                    dst_verifier_rank=0,
                    base_committed_len=2,
                    new_token_pos=2,
                    new_token_id=99,
                ),
                *[
                    DraftTailStreamOutput(
                        request_id="req-0",
                        src_drafter_rank=0,
                        dst_verifier_rank=0,
                        base_committed_len=3,
                        new_token_pos=token_pos,
                        new_token_id=token_id,
                    )
                    for token_pos, token_id in [(5, 66), (6, 77), (7, 88)]
                ],
            ]
        )

        snapshots = buffer.get_draft_snapshots(
            [types.SimpleNamespace(rid="req-0")],
            allow_partial=True,
        )
        self.assertEqual(snapshots[0].tail_tokens, [44, 55, 66, 77])

    def test_decode_output_validates_flattened_verified_ids_without_double_append(self):
        scheduler = _FakeDecodeScheduler()
        req0 = _FakeDecodeReq(
            rid="req-0",
            output_ids=[100, 11, 12, 99],
            draft_buffer=[11, 12, 13],
        )
        req1 = _FakeDecodeReq(rid="req-1", output_ids=[200, 88], draft_buffer=[21])
        req0.kv_committed_len = req0.kv_allocated_len = 4
        req1.kv_committed_len = req1.kv_allocated_len = 2
        req0.spec_verify_ct = 1
        req1.spec_verify_ct = 1
        req0.spec_accepted_tokens = 2
        req0.spec_acceptance_histogram = {2: 1}
        req1.spec_acceptance_histogram = {0: 1}
        batch = _make_verify_decode_batch([req0, req1])
        result = _make_verify_decode_result(
            verified_token_ids=[11, 12, 99, 88],
            accept_lengths=[2, 0],
            num_accepted_tokens=2,
        )

        scheduler.process_batch_result_decode(batch, result)

        self.assertEqual(req0.output_ids, [100, 11, 12, 99])
        self.assertEqual(req1.output_ids, [200, 88])
        self.assertEqual(req0.kv_committed_len, 4)
        self.assertEqual(req1.kv_committed_len, 2)
        self.assertEqual(req0.spec_verify_ct, 1)
        self.assertEqual(req1.spec_verify_ct, 1)
        self.assertEqual(req0.spec_accepted_tokens, 2)
        self.assertEqual(req1.spec_accepted_tokens, 0)
        self.assertEqual(req0.spec_acceptance_histogram, {2: 1})
        self.assertEqual(req1.spec_acceptance_histogram, {0: 1})
        self.assertEqual(req0.checked_accept_lengths, [])
        self.assertEqual(req1.checked_accept_lengths, [])
        self.assertEqual(scheduler.num_generated_tokens, 2)
        self.assertEqual(scheduler.spec_metric_calls, [(2, 2)])
        self.assertEqual(scheduler.token_to_kv_pool_allocator.group_begin_count, 1)
        self.assertEqual(scheduler.token_to_kv_pool_allocator.group_end_count, 1)

    def test_decode_output_rejects_mismatched_accepted_prefix(self):
        scheduler = _FakeDecodeScheduler()
        req = _FakeDecodeReq(rid="req-0", output_ids=[100], draft_buffer=[11])
        batch = _make_verify_decode_batch([req])
        result = _make_verify_decode_result(
            verified_token_ids=[12, 99],
            accept_lengths=[1],
            num_accepted_tokens=1,
        )

        with self.assertRaises(RuntimeError):
            scheduler.process_batch_result_decode(batch, result)


if __name__ == "__main__":
    unittest.main()
