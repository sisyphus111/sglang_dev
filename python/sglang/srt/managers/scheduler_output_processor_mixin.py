from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.routed_experts_capturer import get_global_experts_capturer
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOutput,
    BatchTokenIDOutput,
)
from sglang.srt.managers.schedule_batch import (
    BaseFinishReason,
    Req,
    RequestStage,
    ScheduleBatch,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.decoupled_spec_io import (
    DraftClose,
    DraftControlMessage,
    DraftReqKey,
    DraftSync,
    DraftTailStreamOutput,
    VerifyCommit,
    build_draft_scheduler_rid,
)
from sglang.srt.tracing.trace import trace_slice, trace_slice_batch, trace_slice_end
from sglang.srt.utils import broadcast_pyobj

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import (
        EmbeddingBatchResult,
        GenerationBatchResult,
        ScheduleBatch,
        Scheduler,
    )

logger = logging.getLogger(__name__)

DEFAULT_FORCE_STREAM_INTERVAL = 50


class SchedulerOutputProcessorMixin:
    """
    This class implements the output processing logic for Scheduler.
    We put them into a separate file to make the `scheduler.py` shorter.
    """

    def _is_decoupled_draft_entry_rank(self: Scheduler) -> bool:
        return (
            self.spec_algorithm.is_decoupled_draft()
            and self.pp_rank == 0
            and self.attn_tp_rank == 0
            and self.attn_cp_rank == 0
        )

    def _broadcast_draft_control_messages(
        self: Scheduler,
        messages: list[DraftControlMessage] | None,
    ) -> list[DraftControlMessage]:
        """
        Broadcast draft control messages among all ranks:
        DraftSync: build a new draft request based on its prompt token_ids
        VerifyCommit: overwrite the bonus token and truncate the suffix if needed
        """
        if getattr(self.server_args, "enable_dp_attention", False):
            if self.attn_tp_size != 1:
                messages = broadcast_pyobj(
                    messages,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )
            if self.attn_cp_size != 1:
                messages = broadcast_pyobj(
                    messages,
                    self.attn_cp_group.rank,
                    self.attn_cp_cpu_group,
                    src=self.attn_cp_group.ranks[0],
                )
            return list(messages or [])

        if self.tp_size != 1:
            messages = broadcast_pyobj(
                messages,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )
        return list(messages or [])

    def _get_draft_adapter_thread(self: Scheduler):
        adapter = self.draft_adapter_thread
        if adapter is None:
            raise RuntimeError("Decoupled draft entry rank has no draft adapter thread")
        return adapter

    def _submit_draft_tokens_stream(
        self: Scheduler,
        stream_outputs: list[DraftTailStreamOutput],
    ) -> None:
        """
        Submit new draft tokens produced by a decode round to the verifier.
        """
        if not stream_outputs:
            return
        if not self._is_decoupled_draft_entry_rank():
            stream_outputs.clear()
            return
        self._get_draft_adapter_thread().submit_draft_results(stream_outputs)
        stream_outputs.clear()


    def _draft_apply_verify_commit(
        self: Scheduler,
        req: Req,
        message: VerifyCommit,
        *,
        batch: Optional[ScheduleBatch] = None,
        req_batch_idx: Optional[int] = None,
    ) -> None:
        """
        apply the verify result (pre_verify_committed_len, bonus_token_pos,
        bonus_token_id) to the draft request:
        1. overwrite the bonus token, update the related states, including output_ids, grammar, kv cache, stream output... if needed
        2. update the req's verifier_committed_prefix_len to (bonus_token_pos + 1)
        """
        pre_verify_committed_len = int(message.pre_verify_committed_len)
        bonus_token_pos = int(message.bonus_token_pos)
        bonus_token_id = int(message.bonus_token_id)


        assert (
            pre_verify_committed_len <= req.verifier_committed_prefix_len
            and bonus_token_pos >= pre_verify_committed_len
        ), f"drafter must push forward verifier_committed_prefix_len based on previous committed prefix, but got pre_verify_committed_len > verifier_committed_prefix_len: {pre_verify_committed_len} > {req.verifier_committed_prefix_len}"

        assert bonus_token_pos + 1 >= req.verifier_committed_prefix_len, "VerifyCommit must arrive in order"

        if bonus_token_pos > len(req.output_ids):
            if req.draft_key is not None:
                request_id = req.draft_key.request_id
            else:
                request_id = req.rid
            raise RuntimeError(
                "Decoupled draft received a verify commit beyond its decoded tail: "
                f"request_id={request_id} "
                f"bonus_token_pos={bonus_token_pos} "
                f"output_len={len(req.output_ids)} "
                f"committed_prefix_len={req.verifier_committed_prefix_len}"
            )


        bonus_token_matches = (
            bonus_token_pos < len(req.output_ids)
            and req.output_ids[bonus_token_pos] == bonus_token_id
        )

        if bonus_token_matches:
            # if the bonus token matches, only need to push forward the req's verifier_committed_prefix_len
            req.verifier_committed_prefix_len = bonus_token_pos + 1
            return

        # The verifier-selected bonus token replaces the drafter suffix starting at
        # `bonus_token_pos`.
        #
        # Positions here are in req.output_ids, not in the full prompt+output
        # sequence. The kept output range is [0, truncate_from), and the removed
        # output range is [truncate_from, len(req.output_ids)). In other words,
        # `truncate_from` itself is removed. After the removal, `bonus_token_id`
        # is appended at exactly that position.
        truncate_from = max(0, min(bonus_token_pos, len(req.output_ids)))

        # Number of output tokens removed from the drafter suffix:
        # len(req.output_ids[truncate_from:]).
        removed = len(req.output_ids) - truncate_from

        # KV positions are in the full sequence coordinate system:
        # [0, prompt_len) are prompt tokens, and output_ids[i] corresponds to
        # full-sequence position prompt_len + i. Therefore the KV entries to
        # discard start at `kv_truncate_from`, inclusive.
        prompt_len = len(req.origin_input_ids)
        kv_truncate_from = prompt_len + truncate_from

        if removed > 0:
            if req.grammar is not None:
                try:
                    req.grammar.rollback(removed)
                except Exception:
                    logger.debug("Draft grammar rollback failed for req %s", req.rid)

            if req.req_pool_idx is not None and not req.kv_committed_freed:
                # Only free KV slots that are currently allocated for this req.
                # `trimmed_end` is exclusive. The freed full-sequence KV range is
                # [kv_truncate_from, trimmed_end). If kv_truncate_from ==
                # trimmed_end, there is nothing to free.
                trimmed_end = min(
                    req.kv_allocated_len, prompt_len + len(req.output_ids)
                )
                if kv_truncate_from < trimmed_end:
                    indices_to_free = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, kv_truncate_from:trimmed_end
                    ]
                    if len(indices_to_free) > 0:
                        self.token_to_kv_pool_allocator.free(indices_to_free)
                req.kv_committed_len = min(req.kv_committed_len, kv_truncate_from)
                req.kv_allocated_len = min(req.kv_allocated_len, kv_truncate_from)
                req.cache_protected_len = min(
                    req.cache_protected_len, kv_truncate_from
                )

            # Truncate per-output arrays with the same output-index interval:
            # delete [truncate_from, old_output_len).
            del req.output_ids[truncate_from:]
            if req.return_logprob:
                del req.output_token_logprobs_val[truncate_from:]
                del req.output_token_logprobs_idx[truncate_from:]
                del req.output_top_logprobs_val[truncate_from:]
                del req.output_top_logprobs_idx[truncate_from:]
                del req.output_token_ids_logprobs_val[truncate_from:]
                del req.output_token_ids_logprobs_idx[truncate_from:]
            if req.hidden_states:
                del req.hidden_states[truncate_from:]

        req.output_ids.append(bonus_token_id)
        if req.grammar is not None:
            try:
                req.grammar.accept_token(bonus_token_id)
            except Exception:
                logger.debug(
                    "Draft grammar accept failed during bonus token update for req %s",
                    req.rid,
                )
        req.finished_reason = None
        req.finished_len = None
        req.finished_output = None
        req.to_finish = None
        req.decoded_text = ""

        req.verifier_committed_prefix_len = bonus_token_pos + 1

        if batch is not None and req_batch_idx is not None:
            # Keep the in-flight decode batch consistent with the rewritten request
            # state. This block is only needed when the verifier bonus token changed
            # req.output_ids above: either an existing suffix was truncated and
            # replaced, or the bonus token was appended at the current tail.
            #
            # Decode seq_len is the number of tokens **already present in KV** before the
            # next tail token is consumed. For a drafter request, output_ids[-1] is the
            # current tail token used as the next decode input, so the KV-backed prefix
            # is origin_input_ids plus output_ids[0:-1]. The slice [0, -1) excludes the
            # tail token itself, hence len(origin_input_ids) + max(len(output_ids)-1, 0).
            new_seq_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)

            # batch.output_ids[req_batch_idx] stores the single tail token that the
            # decode worker will consume next. Prefer the last output token. If no
            # output token exists yet, fall back to the last prompt token. In both
            # cases the selected token is included as the decode input, but excluded
            # from new_seq_len above.
            if req.output_ids:
                new_tail_token_id = int(req.output_ids[-1])
            elif req.origin_input_ids:
                new_tail_token_id = int(req.origin_input_ids[-1])
            else:
                raise AssertionError(
                    f"Draft request {req.rid} has no token to decode from"
                )

            old_seq_len = None
            if batch.seq_lens_cpu is not None:
                # Save the old per-request seq_len so seq_lens_sum can be adjusted by
                # delta. req_batch_idx is the inclusive index of this req in batch.reqs.
                old_seq_len = int(batch.seq_lens_cpu[req_batch_idx].item())
                batch.seq_lens_cpu[req_batch_idx] = new_seq_len

            # Mirror the same new_seq_len into every per-request seq_len buffer.
            if batch.seq_lens is not None:
                batch.seq_lens[req_batch_idx] = new_seq_len
            if batch.orig_seq_lens is not None:
                batch.orig_seq_lens[req_batch_idx] = new_seq_len

            # The batch-level output_ids entry is not the whole output sequence. It is
            # exactly the one tail token for this request's next decode step.
            if batch.output_ids is not None:
                batch.output_ids[req_batch_idx] = new_tail_token_id

            if batch.seq_lens_sum is not None:
                if old_seq_len is not None:
                    # Incrementally maintain sum(seq_lens). This is equivalent to
                    # replacing one element: sum' = sum - old_seq_len + new_seq_len.
                    batch.seq_lens_sum += new_seq_len - old_seq_len
                elif batch.seq_lens_cpu is not None:
                    # Fallback when the old value was unavailable: recompute the full
                    # sum over all requests in the batch.
                    batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())



    def _draft_release_req(self: Scheduler, req: Req) -> None:
        """
        release a draft request only when it has completed at the verifier side:
        1. evict the req from waiting_queue or running_batch
        2. remove the req from the draft request table, and clear the draft related states in the req, including draft_key, pending verify commits, verifier_committed_prefix_len, etc.
        3. release its kvcache
        """
        assert req.draft_key is not None, "Only draft requests with draft_key should be released with _draft_release_req"

        draft_key = req.draft_key

        # remove the req from waiting_queue or running_batch
        self.waiting_queue = [
            queued_req for queued_req in self.waiting_queue if queued_req is not req
        ]
        if getattr(self, "running_batch", None) is not None and self.running_batch.reqs:
            keep_indices = [
                i
                for i, running_req in enumerate(self.running_batch.reqs)
                if running_req is not req
            ]
            self.running_batch.filter_batch(keep_indices=keep_indices)
        self.draft_req_table.pop(draft_key, None)
        req.draft_pending_verify_commits.clear()
        req.draft_key = None
        req.verifier_committed_prefix_len = 0
        release_kv_cache(req, self.tree_cache, is_insert=False)


    def _draft_apply_sync(
        self: Scheduler,
        req: Req,
        message: DraftSync,
    ) -> None:
        if req.draft_key is not None:
            raise RuntimeError(
                "Decoupled draft sync only supports creating a new draft request: "
                f"request_id={message.request_id}"
            )
        req.draft_key = message.draft_key
        # drafter must greedily sampling draft tokens, till recv DraftClose message from verifier
        req.sampling_params.temperature = 0.0
        req.sampling_params.top_k = 1
        req.sampling_params.ignore_eos = True

        req.verifier_committed_prefix_len = len(req.output_ids)
        req.fill_ids = req.origin_input_ids + req.output_ids
        req.draft_pending_verify_commits.clear()
        self.draft_req_table[message.draft_key] = req

    def _draft_create_req(
        self: Scheduler,
        message: DraftSync,
    ) -> Req:
        """
        Create a new request based on DraftSync message
        """
        sampling_params = SamplingParams(
            max_new_tokens=1 << 30, # a very large number to ensure the drafter keeps sampling until receiving DraftClose
            temperature=0.0,
            top_k=1,
            ignore_eos=True,
        )
        sampling_params.normalize(self.tokenizer)
        sampling_params.verify(self.model_config.vocab_size)

        req = Req(
            build_draft_scheduler_rid(message.request_id),
            "",
            list(message.prompt_token_ids),
            sampling_params,
            return_logprob=False,
            stream=False,
            eos_token_ids=self.model_config.hf_eos_token_id,
            vocab_size=self.model_config.vocab_size,
            metrics_collector=(self.metrics_collector if self.enable_metrics else None),
        )
        req.tokenizer = self.tokenizer
        req.output_ids = list(message.committed_output_ids)
        req.fill_ids = req.origin_input_ids + req.output_ids
        self.init_req_max_new_tokens(req)
        return req


    def _drain_post_decode_draft_control_messages(
        self: Scheduler,
    ) -> list[VerifyCommit | DraftClose]:
        """
        (called by decoupled drafter)
        Drain all VerifyCommit and DraftClose messages from draft adapter thread
        """
        messages: list[DraftControlMessage] | None = None
        if self._is_decoupled_draft_entry_rank():
            messages = self._get_draft_adapter_thread().drain_post_result_messages()

        return [
            message
            for message in self._broadcast_draft_control_messages(messages)
            if isinstance(message, (VerifyCommit, DraftClose))
        ]


    def _handle_draft_sync_messages(self: Scheduler) -> None:
        """
        (called by decoupled drafter)
        Drain DraftSync messages from draft adapter thread, and handle them.
        DraftSync creates a new drafter-side request from verifier state.
        """

        messages: list[DraftControlMessage] | None = None
        if self._is_decoupled_draft_entry_rank():
            messages = self._get_draft_adapter_thread().drain_sync_messages()

        messages = [
            message
            for message in self._broadcast_draft_control_messages(messages)
            if isinstance(message, DraftSync)
        ]
        if getattr(self.decoupled_spec_tracer, "enabled", False):
            self.decoupled_spec_tracer.record(
                "drafter",
                "recv_sync_batch",
                batch_size=len(messages),
                num_sync=len(messages),
                request_ids=[message.request_id for message in messages],
                committed_lens_by_req=[
                    len(message.committed_output_ids) for message in messages
                ],
                output_lens_by_req=[
                    len(message.committed_output_ids) for message in messages
                ],
            )
    
        if not messages:
            return

        created_reqs: list[Req] = []
        for message in messages:
            draft_key = message.draft_key
            req = self.draft_req_table.get(draft_key)
            if req is not None:
                raise RuntimeError(
                    "Received DraftSync for an existing decoupled draft request: "
                    f"request_id={message.request_id}"
                )
            req = self._draft_create_req(message)
            self._draft_apply_sync(req, message)
            running_batch = self.running_batch
            if req not in self.waiting_queue and req not in running_batch.reqs:
                self._add_request_to_queue(req)
            created_reqs.append(req)
        if getattr(self.decoupled_spec_tracer, "enabled", False):
            self.decoupled_spec_tracer.record(
                "drafter",
                "create_draft_req_batch",
                batch_size=len(created_reqs),
                num_sync=len(messages),
                request_ids=[req.draft_key.request_id for req in created_reqs],
                committed_lens_by_req=[
                    int(req.verifier_committed_prefix_len) for req in created_reqs
                ],
                output_lens_by_req=[len(req.output_ids) for req in created_reqs],
            )


    def _draft_apply_commits_and_maybe_emit(
        self: Scheduler,
        req: Req,
        *,
        commits: Optional[list[VerifyCommit]] = None,
        batch: ScheduleBatch,
        req_batch_idx: int,
        decoded_token: Optional[tuple[int, int]] = None,
    ) -> DraftTailStreamOutput | None:
        """
        1. apply the pending verify commits to the req,
          and update its verifier_committed_prefix_len and other states(if needed)
        2. if the bonus token is not overwritten,
          this req is considered having decoded a new valid draft token,
          therefore, drafter will send this draft token to verifier,
          as streaming draft token output
        """
        assert req.draft_key is not None, "Only draft requests with draft_key should be applied with _draft_apply_commits_and_maybe_emit"

        commits_to_apply: list[VerifyCommit] = []
        if req.draft_pending_verify_commits:
            # check the pending VerifyCommits: received when req is not in running batch
            commits_to_apply.extend(req.draft_pending_verify_commits)
            req.draft_pending_verify_commits.clear()
        
        if commits:
            commits_to_apply.extend(commits)

        for verify_commit in commits_to_apply:
            self._draft_apply_verify_commit(
                req,
                verify_commit,
                batch=batch,
                req_batch_idx=req_batch_idx,
            )

        if not self._is_decoupled_draft_entry_rank() or decoded_token is None:
            return None

        token_pos, token_id = (int(decoded_token[0]), int(decoded_token[1]))
        committed_len = int(req.verifier_committed_prefix_len)
        if (
            token_pos >= committed_len
            and token_pos < len(req.output_ids)
            and int(req.output_ids[token_pos]) == token_id
        ):
            return DraftTailStreamOutput(
                request_id=req.draft_key.request_id,
                src_drafter_rank=int(getattr(self, "dp_rank", 0) or 0),
                dst_verifier_rank=req.draft_key.src_verifier_rank,
                base_committed_len=committed_len,
                new_token_pos=token_pos,
                new_token_id=token_id,
            )
        return None


    def _draft_process_post_decode_controls(
        self: Scheduler,
        batch: ScheduleBatch,
        decoded_tokens: list[Optional[tuple[int, int]]],
        control_messages: list[VerifyCommit | DraftClose],
    ) -> list[DraftTailStreamOutput]:
        """
        args:
          decoded_tokens: the list of (token_pos, token_id) for each new decoded token
          control_messages: VerifyCommit and DraftClose message received from verifier

        called by decoupled drafter, during `process_batch_result_decode()`:
        1. apply DraftClose message: release the draft req
        2. apply VerifyCommit message to the req, and collect & send new draft token
        """
        trace_enabled = getattr(self.decoupled_spec_tracer, "enabled", False)
        trace_start_ns = time.perf_counter_ns() if trace_enabled else 0
        # build draft_key -> req mapping
        current_req_by_key: dict[DraftReqKey, Req] = {}
        for req in batch.reqs:
            assert (
                req.draft_key is not None
            ), "Decoupled drafter batch should only contain draft requests"
            current_req_by_key[req.draft_key] = req

        # collect each req's VerifyCommit message
        commits_by_key: dict[DraftReqKey, list[VerifyCommit]] = {}
        closed_keys: set[DraftReqKey] = set()
        for message in control_messages:
            draft_key = message.draft_key
            req = self.draft_req_table.get(draft_key)
            if isinstance(message, DraftClose):
                # this req will be release upon recv DraftClose
                # discard its pending VerifyCommits
                closed_keys.add(draft_key)
                commits_by_key.pop(draft_key, None)
                if req is not None and req.draft_key is not None:
                    self._draft_release_req(req)
                continue

            if draft_key in closed_keys:
                continue

            if req is None:
                raise RuntimeError(
                    "Received VerifyCommit for an unknown decoupled draft request: "
                    f"request_id={message.request_id} "
                    f"src_verifier_rank={message.src_verifier_rank}"
                )
            
            assert (
                req.draft_key == draft_key
            ), "draft_req_table contains a request under a mismatched draft_key"
            if draft_key in current_req_by_key:
                commits_by_key.setdefault(draft_key, []).append(message)
            else:
                # if the req is not in current batch, cache the pending VerifyCommit message
                req.draft_pending_verify_commits.append(message)

        if trace_enabled:
            commit_messages = [
                message
                for messages in commits_by_key.values()
                for message in messages
            ]
            close_messages = [
                message
                for message in control_messages
                if isinstance(message, DraftClose)
            ]
            self.decoupled_spec_tracer.record(
                "drafter",
                "apply_commit_batch",
                forward_mode=str(batch.forward_mode),
                batch_size=len(batch.reqs),
                num_commit=len(commit_messages),
                request_ids=[message.request_id for message in commit_messages],
                committed_lens_by_req=[
                    int(message.bonus_token_pos) + 1
                    for message in commit_messages
                ],
            )
            self.decoupled_spec_tracer.record(
                "drafter",
                "post_decode_control_batch",
                forward_mode=str(batch.forward_mode),
                batch_size=len(batch.reqs),
                num_commit=len(commit_messages),
                num_close=len(close_messages),
                request_ids=[message.request_id for message in control_messages],
            )

        # apply VerifyCommit and send new draft token
        stream_outputs: list[DraftTailStreamOutput] = []
        for req_batch_idx, req in enumerate(batch.reqs):
            draft_key = req.draft_key
            assert draft_key is not None
            decoded_token = (
                decoded_tokens[req_batch_idx]
                if req_batch_idx < len(decoded_tokens)
                else None
            )
            stream_output = self._draft_apply_commits_and_maybe_emit(
                req,
                commits=commits_by_key.get(draft_key),
                batch=batch,
                req_batch_idx=req_batch_idx,
                decoded_token=decoded_token,
            )
            if stream_output is not None:
                stream_outputs.append(stream_output)
        if trace_enabled:
            counts_by_request: dict[str, int] = {}
            for output in stream_outputs:
                counts_by_request[output.request_id] = (
                    counts_by_request.get(output.request_id, 0) + 1
                )
            request_ids = list(counts_by_request.keys())
            self.decoupled_spec_tracer.record(
                "drafter",
                "emit_tail_batch",
                forward_mode=str(batch.forward_mode),
                duration_ms=(time.perf_counter_ns() - trace_start_ns) / 1_000_000,
                batch_size=len(batch.reqs),
                num_stream_outputs=len(stream_outputs),
                request_ids=request_ids,
                emitted_token_lens_by_req=[
                    counts_by_request[request_id] for request_id in request_ids
                ],
                committed_lens_by_req=[
                    int(req.verifier_committed_prefix_len) for req in batch.reqs
                ],
                output_lens_by_req=[len(req.output_ids) for req in batch.reqs],
            )
        return stream_outputs

    def _get_storage_backend_type(self) -> str:
        """Get storage backend type from tree_cache."""
        storage_backend_type = "none"
        cache_controller = getattr(self.tree_cache, "cache_controller", None)
        if cache_controller and hasattr(cache_controller, "storage_backend"):
            storage_backend = cache_controller.storage_backend
            if storage_backend is not None:
                storage_backend_type = type(storage_backend).__name__
        return storage_backend_type

    def _get_cached_tokens_details(self, req: Req) -> Optional[dict]:
        """Get detailed cache breakdown for a request, if available.

        Returns:
            - None if HiCache is not enabled
            - {"device": X, "host": Y} if HiCache enabled but L3 storage is not
            - {"device": X, "host": Y, "storage": Z, "storage_backend": "..."} if L3 enabled
        """
        # Only show details if HiCache is enabled
        if not getattr(self, "enable_hierarchical_cache", False):
            return None

        # Only show if there are any cached tokens
        if (
            req.cached_tokens_device > 0
            or req.cached_tokens_host > 0
            or req.cached_tokens_storage > 0
        ):
            details = {
                "device": req.cached_tokens_device,
                "host": req.cached_tokens_host,
            }
            # Only include storage fields if L3 storage is enabled
            if getattr(self, "enable_hicache_storage", False):
                details["storage"] = req.cached_tokens_storage
                details["storage_backend"] = self._get_storage_backend_type()
            return details
        return None

    def process_batch_result_prebuilt(self: Scheduler, batch: ScheduleBatch):
        assert self.disaggregation_mode == DisaggregationMode.DECODE
        for req in batch.reqs:
            req.check_finished()
            if req.finished():
                req.time_stats.forward_entry_time = req.time_stats.completion_time = (
                    time.perf_counter()
                )
                trace_slice_end(
                    RequestStage.DECODE_QUICK_FINISH,
                    req.rid,
                    thread_finish_flag=True,
                )
                release_kv_cache(req, self.tree_cache)

        # Note: Logprobs should be handled on the prefill engine.
        trace_slice_batch(RequestStage.DECODE_FAKE_OUTPUT, batch.reqs)
        self.stream_output(batch.reqs, batch.return_logprob)

    def maybe_collect_routed_experts(self: Scheduler, req: Req):
        """Collect routed experts for a finished request."""
        req.routed_experts = get_global_experts_capturer().get_routed_experts(
            req_pool_idx=req.req_pool_idx,
            seqlen=req.seqlen,
            req_to_token_pool=self.req_to_token_pool,
        )

    def maybe_collect_customized_info(
        self: Scheduler, i: int, req: Req, logits_output: LogitsProcessorOutput
    ):
        if logits_output is not None and logits_output.customized_info is not None:
            if req.customized_info is None:
                req.customized_info = {}
            for k, v in logits_output.customized_info.items():
                if k not in req.customized_info:
                    req.customized_info[k] = []
                req.customized_info[k].append(v[i])

    def process_batch_result_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        skip_stream_req = None

        if self.is_generation:
            if result.copy_done is not None:
                result.copy_done.synchronize()

            (
                logits_output,
                next_token_ids,
                extend_input_len_per_req,
                extend_logprob_start_len_per_req,
            ) = (
                result.logits_output,
                result.next_token_ids,
                result.extend_input_len_per_req,
                result.extend_logprob_start_len_per_req,
            )

            # Move next_token_ids and logprobs to cpu
            next_token_ids = next_token_ids.tolist()
            if batch.return_logprob:
                if logits_output.next_token_logprobs is not None:
                    logits_output.next_token_logprobs = (
                        logits_output.next_token_logprobs.tolist()
                    )
                if logits_output.input_token_logprobs is not None:
                    logits_output.input_token_logprobs = tuple(
                        logits_output.input_token_logprobs.tolist()
                    )

            hidden_state_offset = 0

            # Check finish conditions
            logprob_pt = 0

            for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
                if req.finished() or req.is_retracted:
                    # decode req in mixed batch or retracted req
                    continue

                if req.is_chunked <= 0:
                    if req.time_stats.prefill_finished_ts == 0.0:
                        req.time_stats.prefill_finished_ts = time.time()

                    # req output_ids are set here
                    req.output_ids.append(next_token_id)
                    req.check_finished()

                    if req.finished():
                        self.maybe_collect_routed_experts(req)
                        release_kv_cache(req, self.tree_cache)
                        req.time_stats.completion_time = time.perf_counter()
                    elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                        # This updates radix so others can match
                        self.tree_cache.cache_unfinished_req(req)

                    self.maybe_collect_customized_info(i, req, logits_output)

                    if batch.return_logprob:
                        assert extend_logprob_start_len_per_req is not None
                        assert extend_input_len_per_req is not None
                        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                        extend_input_len = extend_input_len_per_req[i]

                        num_input_logprobs = self._calculate_num_input_logprobs(
                            req, extend_input_len, extend_logprob_start_len
                        )

                        if req.return_logprob:
                            self.add_logprob_return_values(
                                i,
                                req,
                                logprob_pt,
                                next_token_ids,
                                num_input_logprobs,
                                logits_output,
                            )
                        logprob_pt += num_input_logprobs

                    if (
                        req.return_hidden_states
                        and logits_output.hidden_states is not None
                    ):
                        req.hidden_states.append(
                            logits_output.hidden_states[
                                hidden_state_offset : (
                                    hidden_state_offset := hidden_state_offset
                                    + len(req.origin_input_ids)
                                )
                            ]
                            .cpu()
                            .clone()
                            .tolist()
                        )

                    if req.grammar is not None:
                        # FIXME: this try-except block is for handling unexpected xgrammar issue.
                        try:
                            req.grammar.accept_token(next_token_id)
                        except ValueError as e:
                            # Grammar accept_token can raise ValueError if the token is not in the grammar.
                            # This can happen if the grammar is not set correctly or the token is invalid.
                            logger.error(
                                f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                            )
                            self.abort_request(AbortReq(rid=req.rid))
                        req.grammar.finished = req.finished()

                    trace_slice(
                        RequestStage.PREFILL_FORWARD,
                        req.rid,
                        auto_next_anon=not req.finished(),
                        thread_finish_flag=req.finished(),
                    )

                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1
                    # There is only at most one request being currently chunked.
                    # Because this request does not finish prefill,
                    # we don't want to stream the request currently being chunked.
                    skip_stream_req = req

                    # Incrementally update input logprobs.
                    if batch.return_logprob:
                        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                        extend_input_len = extend_input_len_per_req[i]
                        if extend_logprob_start_len < extend_input_len:
                            # Update input logprobs.
                            num_input_logprobs = self._calculate_num_input_logprobs(
                                req, extend_input_len, extend_logprob_start_len
                            )
                            if req.return_logprob:
                                self.add_input_logprob_return_values(
                                    i,
                                    req,
                                    logits_output,
                                    logprob_pt,
                                    num_input_logprobs,
                                    last_prefill_chunk=False,
                                )
                            logprob_pt += num_input_logprobs

                    trace_slice(
                        RequestStage.PREFILL_CHUNKED_FORWARD,
                        req.rid,
                        auto_next_anon=True,
                    )

        else:  # embedding or reward model
            if result.copy_done is not None:
                result.copy_done.synchronize()

            is_sparse = envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set()

            embeddings = result.embeddings

            if is_sparse:
                batch_ids, token_ids = embeddings.indices()
                values = embeddings.values()

                embeddings = [{} for _ in range(embeddings.size(0))]
                for i in range(batch_ids.shape[0]):
                    embeddings[batch_ids[i].item()][token_ids[i].item()] = values[
                        i
                    ].item()
            else:
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.tolist()
                else:
                    embeddings = [tensor.tolist() for tensor in embeddings]

            # Check finish conditions
            for i, req in enumerate(batch.reqs):
                if req.is_retracted:
                    continue

                req.embedding = embeddings[i]
                if req.is_chunked <= 0:
                    # Dummy output token for embedding models
                    req.output_ids.append(0)
                    req.check_finished()

                    if req.finished():
                        release_kv_cache(req, self.tree_cache)
                    else:
                        self.tree_cache.cache_unfinished_req(req)
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1

                trace_slice(
                    RequestStage.PREFILL_FORWARD,
                    req.rid,
                    auto_next_anon=not req.finished(),
                    thread_finish_flag=req.finished(),
                )

        self.stream_output(batch.reqs, batch.return_logprob, skip_stream_req)

        if self.current_scheduler_metrics_enabled:
            can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
            self.log_prefill_stats(
                prefill_stats=batch.prefill_stats,
                can_run_cuda_graph=can_run_cuda_graph,
                dp_cooperation_info=batch.dp_cooperation_info,
            )

    def _resolve_spec_overlap_token_ids(
        self: Scheduler, result: GenerationBatchResult, batch: ScheduleBatch
    ) -> List[List[int]]:
        """Resolve the padding next token ids for speculative decoding with overlap."""
        assert result.next_token_ids.is_cpu
        assert result.accept_lens.is_cpu

        next_token_ids = result.next_token_ids.tolist()
        accept_lens = result.accept_lens.tolist()
        result.num_accepted_tokens = sum(accept_lens) - len(batch.reqs)
        result.accept_length_per_req_cpu = [x - 1 for x in accept_lens]

        predict_tokens = []
        stride = self.draft_worker.speculative_num_draft_tokens

        for i, req in enumerate(batch.reqs):
            req.kv_committed_len += accept_lens[i]
            predict_tokens.append(
                next_token_ids[i * stride : i * stride + accept_lens[i]]
            )
            req.spec_verify_ct += 1

            accepted_draft_tokens = result.accept_length_per_req_cpu[i]
            req.spec_accepted_tokens += accepted_draft_tokens
            req.update_spec_acceptance_histogram(accepted_draft_tokens)

        return predict_tokens

    def process_batch_result_idle(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        self.stream_output_generation(
            batch.reqs, batch.return_logprob, is_idle_batch=True
        )

    def process_batch_result_dllm(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        self.token_to_kv_pool_allocator.free_group_begin()

        for idx in range(batch.batch_size()):
            # If no new tokens generated, meaning the prefilling stage
            if not result.next_token_ids:
                break

            req = batch.reqs[idx]
            next_token_ids = result.next_token_ids[idx].tolist()
            self.num_generated_tokens += len(next_token_ids)

            for _token_idx, next_token_id in enumerate(next_token_ids):
                req.output_ids.append(next_token_id)
                req.check_finished()
                if req.finished():
                    release_kv_cache(req, self.tree_cache)
                    req.time_stats.completion_time = time.perf_counter()
                    break

                self.tree_cache.cache_unfinished_req(req)

        self.stream_output(batch.reqs, batch.return_logprob)
        self.token_to_kv_pool_allocator.free_group_end()

        if self.current_scheduler_metrics_enabled:
            can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
            self.log_prefill_stats(
                prefill_stats=batch.prefill_stats,
                can_run_cuda_graph=can_run_cuda_graph,
                dp_cooperation_info=batch.dp_cooperation_info,
            )

    def process_batch_result_decode(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        is_decoupled_draft = bool(batch.spec_algorithm.is_decoupled_draft())
        is_decoupled_verify = bool(batch.spec_algorithm.is_decoupled_verify())
        if result.copy_done is not None:
            result.copy_done.synchronize()

        logits_output, next_token_ids, can_run_cuda_graph = (
            result.logits_output,
            result.next_token_ids,
            result.can_run_cuda_graph,
        )

        if batch.spec_algorithm.is_none() or is_decoupled_draft:
            next_token_ids = next_token_ids.tolist()
            if batch.return_logprob:
                next_token_logprobs = logits_output.next_token_logprobs.tolist()
        elif batch.is_spec_v2:
            next_token_ids = self._resolve_spec_overlap_token_ids(result, batch)
        elif is_decoupled_verify:
            # Decoupled verify reuses the EAGLE/spec-v1 verify path, which
            # mutates req.output_ids and checks finish inside the worker.
            # Keep scheduler-side handling aligned with the v0.5.9 spec-v1
            # contract: do not append the returned verified ids a second time.
            next_token_ids = [None] * len(batch.reqs)

        self.num_generated_tokens += len(batch.reqs)
        if not batch.spec_algorithm.is_none() or is_decoupled_draft:
            self.update_spec_metrics(batch.batch_size(), result.num_accepted_tokens)

        if self.enable_metrics:
            self.metrics_collector.increment_cuda_graph_pass(value=can_run_cuda_graph)

        self.token_to_kv_pool_allocator.free_group_begin()

        # NOTE: in any case, we should check finish here
        # if finished, also clean up committed kv cache and over-allocated kv cache here

        # Check finish condition
        req_iter = (
            (i, req, next_token_id)
            for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids))
        )

        # newly decoded (token_pos, token_id) for all reqs in this batch
        decoded_draft_tokens: list[Optional[tuple[int, int]]] = (
            [None] * len(batch.reqs) if is_decoupled_draft else []
        )

        for i, req, next_token_id in req_iter:
            req: Req

            if self.enable_overlap and (req.finished() or req.is_retracted):
                # NOTE: This (req.finished() or req.is_retracted) should only happen when overlap scheduling is enabled.
                # (currently not, e.g. Eagle V1 still check finish during forward)
                # And all the over-allocated tokens will be freed in `release_kv_cache`.
                continue

            new_accepted_len = 1
            if (
                batch.spec_algorithm.is_none()
                or batch.spec_algorithm.is_decoupled_draft()
            ):
                if is_decoupled_draft:
                    decoded_draft_tokens[i] = (len(req.output_ids), int(next_token_id))
                req.output_ids.append(next_token_id)
            elif batch.is_spec_v2:
                # Only spec v2's output_ids are updated here.
                req.output_ids.extend(next_token_id)
                new_accepted_len = len(next_token_id)
            elif is_decoupled_verify:
                # Output ids were already committed by EAGLE/spec-v1 verify.
                pass

            # Update Mamba last track seqlen
            self._mamba_prefix_cache_update(req, batch, result, i)

            # External decoupled drafter must not finish locally based on draft
            # tokens. The verifier still owns finished state, matching the
            # v0.5.9 spec-v1 post-processing contract.
            if not is_decoupled_draft:
                req.check_finished(new_accepted_len)

            if req.finished():
                self.maybe_collect_routed_experts(req)

                if self.server_args.disaggregation_decode_enable_offload_kvcache:
                    # Asynchronously offload KV cache; release_kv_cache will be called after Device->Host transfer completes
                    if not self.decode_offload_manager.offload_kv_cache(req):
                        release_kv_cache(req, self.tree_cache)
                else:
                    release_kv_cache(req, self.tree_cache)

                req.time_stats.completion_time = time.perf_counter()

            self.maybe_collect_customized_info(i, req, logits_output)

            if req.return_logprob and (
                batch.spec_algorithm.is_none()
                or batch.spec_algorithm.is_decoupled_draft()
            ):
                # speculative worker handles logprob in speculative decoding
                req.output_token_logprobs_val.append(next_token_logprobs[i])
                req.output_token_logprobs_idx.append(next_token_id)
                if req.top_logprobs_num > 0:
                    req.output_top_logprobs_val.append(
                        logits_output.next_token_top_logprobs_val[i]
                    )
                    req.output_top_logprobs_idx.append(
                        logits_output.next_token_top_logprobs_idx[i]
                    )
                if req.token_ids_logprob is not None:
                    req.output_token_ids_logprobs_val.append(
                        logits_output.next_token_token_ids_logprobs_val[i]
                    )
                    req.output_token_ids_logprobs_idx.append(
                        logits_output.next_token_token_ids_logprobs_idx[i]
                    )

            if req.return_hidden_states and logits_output.hidden_states is not None:
                req.hidden_states.append(
                    logits_output.hidden_states[i].cpu().clone().tolist()
                )

            if req.grammar is not None and not is_decoupled_verify:
                # FIXME: this try-except block is for handling unexpected xgrammar issue.
                try:
                    if (
                        batch.spec_algorithm.is_none()
                        or batch.spec_algorithm.is_decoupled_draft()
                    ):
                        # Normal decode: single token
                        req.grammar.accept_token(next_token_id)
                    elif batch.is_spec_v2:
                        # Speculative decode: next_token_id is a list of accepted tokens
                        for token_id in next_token_id:
                            req.grammar.accept_token(token_id)
                except ValueError as e:
                    # Grammar accept_token can raise ValueError if the token is not in the grammar.
                    # This can happen if the grammar is not set correctly or the token is invalid.
                    logger.error(
                        f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                    )
                    self.abort_request(AbortReq(rid=req.rid))
                req.grammar.finished = req.finished()

        if is_decoupled_draft:
            post_decode_messages = self._drain_post_decode_draft_control_messages()
            stream_outputs = (
                self._draft_process_post_decode_controls(
                    batch,
                    decoded_draft_tokens,
                    post_decode_messages,
                )
            )
            self._submit_draft_tokens_stream(stream_outputs)

        self.stream_output(batch.reqs, batch.return_logprob)
        self.token_to_kv_pool_allocator.free_group_end()

        self.forward_ct_decode = (self.forward_ct_decode + 1) % (1 << 30)
        if self.current_scheduler_metrics_enabled:
            if self.forward_ct_decode % self.server_args.decode_log_interval == 0:
                self.log_decode_stats(can_run_cuda_graph, running_batch=batch)
            self.log_decode_stats_every_iteration(
                batch, num_accepted_tokens=result.num_accepted_tokens
            )

    def _mamba_prefix_cache_update(
        self, req: Req, batch: ScheduleBatch, result: GenerationBatchResult, i: int
    ) -> None:
        seq_len = len(req.origin_input_ids) + len(req.output_ids) - 1
        if req.mamba_ping_pong_track_buffer is not None:
            mamba_track_interval = get_global_server_args().mamba_track_interval
            if (
                (
                    batch.spec_algorithm.is_none()
                    or batch.spec_algorithm.is_decoupled_draft()
                )
                and seq_len % mamba_track_interval == 0
            ):
                # for non-spec decode, we update mamba_last_track_seqlen at the end of each track interval
                req.mamba_next_track_idx = 1 - req.mamba_next_track_idx
                req.mamba_last_track_seqlen = seq_len
            elif (
                not batch.spec_algorithm.is_none()
                and not batch.spec_algorithm.is_decoupled_draft()
                and result.accept_length_per_req_cpu is not None
            ):
                # for spec decode, update mamba_last_track_seqlen if this iteration crosses a track interval
                actual_seq_len = req.seqlen - 1
                if (
                    actual_seq_len // mamba_track_interval
                    != (actual_seq_len - result.accept_length_per_req_cpu[i])
                    // mamba_track_interval
                ):
                    req.mamba_last_track_seqlen = (
                        actual_seq_len // mamba_track_interval * mamba_track_interval
                    )

    def _process_input_token_logprobs(
        self, req: Req, input_token_logprobs: List
    ) -> None:
        """Process input token logprobs values and indices."""
        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Process logprob values - handle multi-item scoring vs regular requests
        if is_multi_item_scoring:
            # Multi-item scoring: use all logprobs as-is
            req.input_token_logprobs_val = input_token_logprobs
        else:
            # Regular request: add None at start, remove last (sampling token)
            req.input_token_logprobs_val = [None] + input_token_logprobs[:-1]

        # Process logprob indices based on scoring type
        if is_multi_item_scoring:
            # Multi-item scoring: only include delimiter token positions
            relevant_tokens = req.origin_input_ids[req.logprob_start_len :]
            input_token_logprobs_idx = [
                token_id
                for token_id in relevant_tokens
                if token_id == self.server_args.multi_item_scoring_delimiter
            ]
        else:
            # Regular request: include all tokens from logprob_start_len onwards
            input_token_logprobs_idx = req.origin_input_ids[req.logprob_start_len :]

        # Clip padded hash values from image tokens to prevent detokenization errors
        req.input_token_logprobs_idx = [
            x if x < self.model_config.vocab_size - 1 else 0
            for x in input_token_logprobs_idx
        ]

    def _process_input_top_logprobs(self, req: Req) -> None:
        """Process input top logprobs."""
        if req.top_logprobs_num <= 0:
            return

        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Initialize arrays - multi-item scoring starts empty, others start with None
        req.input_top_logprobs_val = [] if is_multi_item_scoring else [None]
        req.input_top_logprobs_idx = [] if is_multi_item_scoring else [None]

        # Extend arrays with temp values
        for val, idx in zip(
            req.temp_input_top_logprobs_val,
            req.temp_input_top_logprobs_idx,
            strict=True,
        ):
            req.input_top_logprobs_val.extend(val)
            req.input_top_logprobs_idx.extend(idx)

        # Remove last token (sampling token) for non multi-item scoring requests
        if not is_multi_item_scoring:
            req.input_top_logprobs_val.pop()
            req.input_top_logprobs_idx.pop()

        # Clean up temp storage
        req.temp_input_top_logprobs_idx = None
        req.temp_input_top_logprobs_val = None

    def _process_input_token_ids_logprobs(self, req: Req) -> None:
        """Process input token IDs logprobs."""
        if req.token_ids_logprob is None:
            return

        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Initialize arrays - multi-item scoring starts empty, others start with None
        req.input_token_ids_logprobs_val = [] if is_multi_item_scoring else [None]
        req.input_token_ids_logprobs_idx = [] if is_multi_item_scoring else [None]

        # Process temp values - convert tensors to lists and extend arrays
        for val, idx in zip(
            req.temp_input_token_ids_logprobs_val,
            req.temp_input_token_ids_logprobs_idx,
            strict=True,
        ):
            val_list = val.tolist() if isinstance(val, torch.Tensor) else val
            req.input_token_ids_logprobs_val.extend(
                val_list if isinstance(val_list, list) else [val_list]
            )
            req.input_token_ids_logprobs_idx.extend(idx)

        # Remove last token (sampling token) for non multi-item scoring requests
        if not is_multi_item_scoring:
            req.input_token_ids_logprobs_val.pop()
            req.input_token_ids_logprobs_idx.pop()

        # Clean up temp storage
        req.temp_input_token_ids_logprobs_idx = None
        req.temp_input_token_ids_logprobs_val = None

    def _calculate_relevant_tokens_len(self, req: Req) -> int:
        """Calculate the expected length of logprob arrays based on whether multi-item scoring is enabled.

        For multi-item scoring, only delimiter positions have logprobs.
        For regular requests, all positions from logprob_start_len onwards have logprobs.
        """
        is_multi_item_scoring = self._is_multi_item_scoring(req)
        relevant_tokens = req.origin_input_ids[req.logprob_start_len :]

        if is_multi_item_scoring:
            # Multi-item scoring: count delimiter tokens from logprob_start_len onwards
            return sum(
                1
                for token_id in relevant_tokens
                if token_id == self.server_args.multi_item_scoring_delimiter
            )
        else:
            # Regular request: all tokens from logprob_start_len onwards
            return len(relevant_tokens)

    def _calculate_num_input_logprobs(
        self, req: Req, extend_input_len: int, extend_logprob_start_len: int
    ) -> int:
        """Calculate the number of input logprobs based on whether multi-item scoring is enabled.

        For multi-item scoring, only delimiter positions have logprobs.
        For regular requests, all positions in the range have logprobs.
        """
        is_multi_item_scoring = self._is_multi_item_scoring(req)

        if is_multi_item_scoring:
            # Multi-item scoring: count delimiter tokens in the relevant portion
            relevant_tokens = req.origin_input_ids[
                extend_logprob_start_len:extend_input_len
            ]
            return sum(
                1
                for token_id in relevant_tokens
                if token_id == self.server_args.multi_item_scoring_delimiter
            )
        else:
            # Regular request: all tokens in the range
            return extend_input_len - extend_logprob_start_len

    def _is_multi_item_scoring(self, req: Req) -> bool:
        """Check if request uses multi-item scoring.

        Multi-item scoring applies to prefill-only requests when a delimiter
        token is configured. In this mode, only positions containing the
        delimiter token receive logprobs.
        """
        return req.is_prefill_only and self.server_args.multi_item_scoring_delimiter

    def add_input_logprob_return_values(
        self: Scheduler,
        i: int,
        req: Req,
        output: LogitsProcessorOutput,
        logprob_pt: int,
        num_input_logprobs: int,
        last_prefill_chunk: bool,  # If True, it means prefill is finished.
    ):
        """Incrementally add input logprobs to `req`.

        Args:
            i: The request index in a batch.
            req: The request. Input logprobs inside req are modified as a
                consequence of the API
            fill_ids: The prefill ids processed.
            output: Logit processor output that's used to compute input logprobs
            last_prefill_chunk: True if it is the last prefill (when chunked).
                Some of input logprob operation should only happen at the last
                prefill (e.g., computing input token logprobs).
        """
        assert output.input_token_logprobs is not None
        if req.input_token_logprobs is None:
            req.input_token_logprobs = []
        if req.temp_input_top_logprobs_val is None:
            req.temp_input_top_logprobs_val = []
        if req.temp_input_top_logprobs_idx is None:
            req.temp_input_top_logprobs_idx = []
        if req.temp_input_token_ids_logprobs_val is None:
            req.temp_input_token_ids_logprobs_val = []
        if req.temp_input_token_ids_logprobs_idx is None:
            req.temp_input_token_ids_logprobs_idx = []

        if req.input_token_logprobs_val is not None:
            # The input logprob has been already computed. It only happens
            # upon retract.
            if req.top_logprobs_num > 0:
                assert req.input_token_logprobs_val is not None
            return

        # Important for the performance.
        assert isinstance(output.input_token_logprobs, tuple)
        input_token_logprobs: Tuple[int] = output.input_token_logprobs
        input_token_logprobs = input_token_logprobs[
            logprob_pt : logprob_pt + num_input_logprobs
        ]
        req.input_token_logprobs.extend(input_token_logprobs)

        if req.top_logprobs_num > 0:
            req.temp_input_top_logprobs_val.append(output.input_top_logprobs_val[i])
            req.temp_input_top_logprobs_idx.append(output.input_top_logprobs_idx[i])

        if req.token_ids_logprob is not None:
            req.temp_input_token_ids_logprobs_val.append(
                output.input_token_ids_logprobs_val[i]
            )
            req.temp_input_token_ids_logprobs_idx.append(
                output.input_token_ids_logprobs_idx[i]
            )

        if last_prefill_chunk:
            input_token_logprobs = req.input_token_logprobs
            req.input_token_logprobs = None
            assert req.input_token_logprobs_val is None
            assert req.input_token_logprobs_idx is None
            assert req.input_top_logprobs_val is None
            assert req.input_top_logprobs_idx is None

            # Process all input logprob types using helper functions
            self._process_input_token_logprobs(req, input_token_logprobs)
            self._process_input_top_logprobs(req)

            self._process_input_token_ids_logprobs(req)

            if req.return_logprob:
                relevant_tokens_len = self._calculate_relevant_tokens_len(req)
                assert len(req.input_token_logprobs_val) == relevant_tokens_len
                assert len(req.input_token_logprobs_idx) == relevant_tokens_len
                if req.top_logprobs_num > 0:
                    assert len(req.input_top_logprobs_val) == relevant_tokens_len
                    assert len(req.input_top_logprobs_idx) == relevant_tokens_len
                if req.token_ids_logprob is not None:
                    assert len(req.input_token_ids_logprobs_val) == relevant_tokens_len
                    assert len(req.input_token_ids_logprobs_idx) == relevant_tokens_len

    def add_logprob_return_values(
        self: Scheduler,
        i: int,
        req: Req,
        pt: int,
        next_token_ids: List[int],
        num_input_logprobs: int,
        output: LogitsProcessorOutput,
    ):
        """Attach logprobs to the return values."""
        if output.next_token_logprobs is not None:
            req.output_token_logprobs_val.append(output.next_token_logprobs[i])
            req.output_token_logprobs_idx.append(next_token_ids[i])

        # Only add input logprobs if there are input tokens to process
        # Note: For prefill-only requests with default logprob_start_len, this will be 0,
        # meaning we only compute output logprobs (which is the intended behavior)
        if num_input_logprobs > 0:
            self.add_input_logprob_return_values(
                i, req, output, pt, num_input_logprobs, last_prefill_chunk=True
            )
        else:
            self._initialize_empty_logprob_containers(req)

        if req.top_logprobs_num > 0:
            req.output_top_logprobs_val.append(output.next_token_top_logprobs_val[i])
            req.output_top_logprobs_idx.append(output.next_token_top_logprobs_idx[i])

        if (
            req.token_ids_logprob is not None
            and output.next_token_token_ids_logprobs_val is not None
        ):
            # Convert GPU tensor to list if needed
            logprobs_val = output.next_token_token_ids_logprobs_val[i]
            if isinstance(logprobs_val, torch.Tensor):
                logprobs_val = logprobs_val.tolist()
            req.output_token_ids_logprobs_val.append(logprobs_val)
            req.output_token_ids_logprobs_idx.append(
                output.next_token_token_ids_logprobs_idx[i]
            )

        return num_input_logprobs

    def _initialize_empty_logprob_containers(self, req: Req) -> None:
        """
        Initialize logprob fields to empty lists if unset.

        This is needed for prefill-only requests where the normal initialization
        flow might be bypassed, but downstream code expects these fields to be lists.
        """
        if req.input_token_logprobs_val is None:
            req.input_token_logprobs_val = []
        if req.input_token_logprobs_idx is None:
            req.input_token_logprobs_idx = []
        if req.input_top_logprobs_val is None:
            req.input_top_logprobs_val = []
        if req.input_top_logprobs_idx is None:
            req.input_top_logprobs_idx = []
        if req.input_token_ids_logprobs_val is None:
            req.input_token_ids_logprobs_val = []
        if req.input_token_ids_logprobs_idx is None:
            req.input_token_ids_logprobs_idx = []

    def stream_output(
        self: Scheduler,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
    ):
        """Stream the output to detokenizer."""
        if self.spec_algorithm.is_decoupled_draft():
            return
        if self.is_generation:
            self.stream_output_generation(reqs, return_logprob, skip_req)
        else:  # embedding or reward model
            self.stream_output_embedding(reqs)

        if envs.SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS.get() > 0:
            self._trigger_crash_for_tests(
                envs.SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS.get()
            )

    def _trigger_crash_for_tests(self, crash_threshold: int):
        # Crash trigger: crash after stream_output is called N times
        # This is used for testing purposes.
        if not hasattr(self, "_test_stream_output_count"):
            self._test_stream_output_count = 0
        self._test_stream_output_count += 1
        if self._test_stream_output_count >= crash_threshold:
            raise RuntimeError(
                f"Test crash after stream_output called {self._test_stream_output_count} times"
            )

    def stream_output_generation(
        self: Scheduler,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
        is_idle_batch: bool = False,
    ):
        rids = []
        http_worker_ipcs = []
        finished_reasons: List[BaseFinishReason] = []

        decoded_texts = []
        decode_ids_list = []
        read_offsets = []
        output_ids = []

        skip_special_tokens = []
        spaces_between_special_tokens = []
        no_stop_trim = []
        prompt_tokens = []
        completion_tokens = []
        cached_tokens = []
        cached_tokens_details = []  # Detailed breakdown by cache source
        spec_verify_ct = []
        spec_accepted_tokens = []
        spec_acceptance_histogram = []
        retraction_counts = []
        output_hidden_states = None
        load = self.get_load()
        routed_experts = None
        customized_info = {}

        queue_times = []
        forward_entry_times = []
        prefill_launch_delays = []
        prefill_launch_latencies = []
        prefill_finished_timestamps = []

        if return_logprob:
            input_token_logprobs_val = []
            input_token_logprobs_idx = []
            output_token_logprobs_val = []
            output_token_logprobs_idx = []
            input_top_logprobs_val = []
            input_top_logprobs_idx = []
            output_top_logprobs_val = []
            output_top_logprobs_idx = []
            input_token_ids_logprobs_val = []
            input_token_ids_logprobs_idx = []
            output_token_ids_logprobs_val = []
            output_token_ids_logprobs_idx = []
        else:
            input_token_logprobs_val = input_token_logprobs_idx = (
                output_token_logprobs_val
            ) = output_token_logprobs_idx = input_top_logprobs_val = (
                input_top_logprobs_idx
            ) = output_top_logprobs_val = output_top_logprobs_idx = (
                input_token_ids_logprobs_val
            ) = input_token_ids_logprobs_idx = output_token_ids_logprobs_val = (
                output_token_ids_logprobs_idx
            ) = None

        for req in reqs:
            if req is skip_req:
                continue

            # Multimodal partial stream chunks break the detokenizer, so drop aborted requests here.
            if self.model_config.is_multimodal_gen and req.to_finish:
                continue

            if req.finished():
                if req.finished_output:
                    # With the overlap schedule, a request will try to output twice and hit this line twice
                    # because of the one additional delayed token. This "continue" prevented the dummy output.
                    continue
                req.finished_output = True
                if req.finished_len is None:
                    req.finished_len = len(req.output_ids)
                should_output = True
            else:
                if req.stream:
                    stream_interval = (
                        req.sampling_params.stream_interval or self.stream_interval
                    )

                    # origin stream_interval logic
                    should_output = (
                        len(req.output_ids) % stream_interval == 1
                        if not self.model_config.is_multimodal_gen
                        and stream_interval > 1
                        else len(req.output_ids) % stream_interval == 0
                    )

                    if should_output:
                        # check_match_stop_str_prefix if  tail_str's suffix match stop_str prefix
                        should_output &= not req.check_match_stop_str_prefix()
                else:
                    should_output = (
                        len(req.output_ids) % DEFAULT_FORCE_STREAM_INTERVAL == 0
                        if not self.model_config.is_multimodal_gen
                        else False
                    )

            if should_output:
                send_token_offset = req.send_token_offset
                send_output_token_logprobs_offset = (
                    req.send_output_token_logprobs_offset
                )
                rids.append(req.rid)
                http_worker_ipcs.append(req.http_worker_ipc)
                finished_reasons.append(
                    req.finished_reason.to_json() if req.finished_reason else None
                )
                decoded_texts.append(req.decoded_text)
                decode_ids, read_offset = req.init_incremental_detokenize()

                if self.model_config.is_multimodal_gen:
                    decode_ids_list.append(decode_ids)
                else:
                    decode_ids_list.append(decode_ids[req.send_decode_id_offset :])

                # Exclude the tokens after stop condition
                output_ids_ = req.output_ids_through_stop

                req.send_decode_id_offset = len(decode_ids)
                read_offsets.append(read_offset)
                output_ids.append(output_ids_[send_token_offset:])
                req.send_token_offset = len(output_ids_)
                skip_special_tokens.append(req.sampling_params.skip_special_tokens)
                spaces_between_special_tokens.append(
                    req.sampling_params.spaces_between_special_tokens
                )
                no_stop_trim.append(req.sampling_params.no_stop_trim)
                prompt_tokens.append(len(req.origin_input_ids))
                completion_tokens.append(len(output_ids_))
                cached_tokens.append(req.cached_tokens)

                # Collect detailed cache breakdown if available
                cached_tokens_details.append(self._get_cached_tokens_details(req))

                retraction_counts.append(req.retraction_count)

                queue_times.append(req.time_stats.get_queueing_time())
                forward_entry_times.append(req.time_stats.forward_entry_time)

                prefill_launch_delays.append(req.time_stats.get_prefill_launch_delay())
                prefill_launch_latencies.append(
                    req.time_stats.get_prefill_launch_latency()
                )
                prefill_finished_timestamps.append(
                    req.time_stats.get_prefill_finished_ts()
                )

                if not self.spec_algorithm.is_none() and not self.spec_algorithm.is_decoupled_draft():
                    spec_verify_ct.append(req.spec_verify_ct)
                    spec_accepted_tokens.append(req.spec_accepted_tokens)
                    spec_acceptance_histogram.append(req.spec_acceptance_histogram)

                if return_logprob:
                    if (
                        req.return_logprob
                        and not req.input_logprob_sent
                        # Decode server does not send input logprobs
                        and self.disaggregation_mode != DisaggregationMode.DECODE
                    ):
                        input_token_logprobs_val.append(req.input_token_logprobs_val)
                        input_token_logprobs_idx.append(req.input_token_logprobs_idx)
                        input_top_logprobs_val.append(req.input_top_logprobs_val)
                        input_top_logprobs_idx.append(req.input_top_logprobs_idx)
                        input_token_ids_logprobs_val.append(
                            req.input_token_ids_logprobs_val
                        )
                        input_token_ids_logprobs_idx.append(
                            req.input_token_ids_logprobs_idx
                        )
                        req.input_logprob_sent = True
                    else:
                        input_token_logprobs_val.append([])
                        input_token_logprobs_idx.append([])
                        input_top_logprobs_val.append([])
                        input_top_logprobs_idx.append([])
                        input_token_ids_logprobs_val.append([])
                        input_token_ids_logprobs_idx.append([])

                    if req.return_logprob:
                        output_token_logprobs_val.append(
                            req.output_token_logprobs_val[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_token_logprobs_idx.append(
                            req.output_token_logprobs_idx[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_top_logprobs_val.append(
                            req.output_top_logprobs_val[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_top_logprobs_idx.append(
                            req.output_top_logprobs_idx[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_token_ids_logprobs_val.append(
                            req.output_token_ids_logprobs_val[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_token_ids_logprobs_idx.append(
                            req.output_token_ids_logprobs_idx[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        req.send_output_token_logprobs_offset = len(
                            req.output_token_logprobs_val
                        )
                    else:
                        output_token_logprobs_val.append([])
                        output_token_logprobs_idx.append([])
                        output_top_logprobs_val.append([])
                        output_top_logprobs_idx.append([])
                        output_token_ids_logprobs_val.append([])
                        output_token_ids_logprobs_idx.append([])

                if req.return_hidden_states:
                    if output_hidden_states is None:
                        output_hidden_states = []
                    output_hidden_states.append(req.hidden_states)
                if req.return_routed_experts:
                    if routed_experts is None:
                        routed_experts = []
                    routed_experts.append(req.routed_experts)

                if req.customized_info is not None:
                    for k, v in req.customized_info.items():
                        if k not in customized_info:
                            customized_info[k] = []
                        customized_info[k].append(v[send_token_offset:])

            if (
                req.finished()
                and self.attn_tp_rank == 0
                and self.server_args.enable_request_time_stats_logging
            ):
                req.log_time_stats()

        # Send to detokenizer
        if reqs or is_idle_batch:
            if self.model_config.is_multimodal_gen:
                return
            self.send_to_detokenizer.send_output(
                BatchTokenIDOutput(
                    rids=rids,
                    http_worker_ipcs=http_worker_ipcs,
                    spec_verify_ct=spec_verify_ct,
                    spec_accepted_tokens=spec_accepted_tokens,
                    spec_acceptance_histogram=spec_acceptance_histogram,
                    queue_time=queue_times,
                    forward_entry_time=forward_entry_times,
                    prefill_launch_delay=prefill_launch_delays,
                    prefill_launch_latency=prefill_launch_latencies,
                    prefill_finished_ts=prefill_finished_timestamps,
                    finished_reasons=finished_reasons,
                    decoded_texts=decoded_texts,
                    decode_ids=decode_ids_list,
                    read_offsets=read_offsets,
                    output_ids=output_ids,
                    skip_special_tokens=skip_special_tokens,
                    spaces_between_special_tokens=spaces_between_special_tokens,
                    no_stop_trim=no_stop_trim,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                    cached_tokens_details=cached_tokens_details,
                    input_token_logprobs_val=input_token_logprobs_val,
                    input_token_logprobs_idx=input_token_logprobs_idx,
                    output_token_logprobs_val=output_token_logprobs_val,
                    output_token_logprobs_idx=output_token_logprobs_idx,
                    input_top_logprobs_val=input_top_logprobs_val,
                    input_top_logprobs_idx=input_top_logprobs_idx,
                    output_top_logprobs_val=output_top_logprobs_val,
                    output_top_logprobs_idx=output_top_logprobs_idx,
                    input_token_ids_logprobs_val=input_token_ids_logprobs_val,
                    input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
                    output_token_ids_logprobs_val=output_token_ids_logprobs_val,
                    output_token_ids_logprobs_idx=output_token_ids_logprobs_idx,
                    output_token_entropy_val=None,
                    output_hidden_states=output_hidden_states,
                    routed_experts=routed_experts,
                    customized_info=customized_info,
                    placeholder_tokens_idx=None,
                    placeholder_tokens_val=None,
                    retraction_counts=retraction_counts,
                    load=load,
                )
            )

    def stream_output_embedding(self: Scheduler, reqs: List[Req]):
        rids = []
        http_worker_ipcs = []
        finished_reasons: List[BaseFinishReason] = []

        embeddings = []
        prompt_tokens = []
        cached_tokens = []
        cached_tokens_details = []  # Detailed breakdown by cache source
        queue_times = []
        forward_entry_times = []
        prefill_launch_delays = []
        prefill_launch_latencies = []
        prefill_finished_timestamps = []
        retraction_counts = []
        for req in reqs:
            if req.finished():
                rids.append(req.rid)
                http_worker_ipcs.append(req.http_worker_ipc)
                finished_reasons.append(req.finished_reason.to_json())
                embeddings.append(req.embedding)
                prompt_tokens.append(len(req.origin_input_ids))
                cached_tokens.append(req.cached_tokens)

                # Collect detailed cache breakdown if available
                cached_tokens_details.append(self._get_cached_tokens_details(req))

                queue_times.append(req.time_stats.get_queueing_time())
                forward_entry_times.append(req.time_stats.forward_entry_time)

                prefill_launch_delays.append(req.time_stats.get_prefill_launch_delay())
                prefill_launch_latencies.append(
                    req.time_stats.get_prefill_launch_latency()
                )
                prefill_finished_timestamps.append(
                    req.time_stats.get_prefill_finished_ts()
                )
                retraction_counts.append(req.retraction_count)
        self.send_to_detokenizer.send_output(
            BatchEmbeddingOutput(
                rids=rids,
                http_worker_ipcs=http_worker_ipcs,
                queue_time=queue_times,
                forward_entry_time=forward_entry_times,
                prefill_launch_delay=prefill_launch_delays,
                prefill_launch_latency=prefill_launch_latencies,
                prefill_finished_ts=prefill_finished_timestamps,
                finished_reasons=finished_reasons,
                embeddings=embeddings,
                prompt_tokens=prompt_tokens,
                cached_tokens=cached_tokens,
                cached_tokens_details=cached_tokens_details,
                placeholder_tokens_idx=None,
                placeholder_tokens_val=None,
                retraction_counts=retraction_counts,
            )
        )
