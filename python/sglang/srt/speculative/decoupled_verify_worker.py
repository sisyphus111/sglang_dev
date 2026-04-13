from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.decoupled_spec_io import (
    DraftResult,
    build_draft_scheduler_rid,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_utils import TreeMaskMode, build_tree_kernel_efficient
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.utils.csv_debug_utils import emit_csv_event, emit_summary

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput

logger = logging.getLogger(__name__)


def _get_req_tail_token_id(req) -> int:
    if req.output_ids:
        return int(req.output_ids[-1])
    if req.origin_input_ids:
        return int(req.origin_input_ids[-1])
    raise RuntimeError(
        f"Request {req.rid} has no committed token to anchor external draft verification."
    )


def _slice_tensor_head_or_empty(
    value: torch.Tensor | None,
    live_count: int,
    *,
    empty_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        return torch.empty(empty_shape, dtype=dtype, device=device)
    return value[:live_count]


def _fit_verify_window_tokens(
    token_ids: list[int],
    verify_window_len: int,
    pad_token_id: int,
) -> list[int]:
    if len(token_ids) >= verify_window_len:
        return token_ids[:verify_window_len]
    return token_ids + [pad_token_id] * (verify_window_len - len(token_ids))


def _normalize_token_id(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, (list, tuple, set)):
        for item in value:
            normalized = _normalize_token_id(item)
            if normalized is not None:
                return normalized
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_cuda_device(device: str | torch.device) -> bool:
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return str(device).startswith("cuda")


def _build_linear_topk1_tree_metadata(
    batch_size: int,
    spec_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_index = torch.arange(
        spec_steps,
        dtype=torch.long,
        device=device,
    ).expand(batch_size, -1).contiguous()

    if spec_steps <= 1:
        parent_list = torch.empty((batch_size, 0), dtype=torch.long, device=device)
    else:
        parent_list = torch.arange(
            -1,
            spec_steps - 1,
            dtype=torch.long,
            device=device,
        ).expand(batch_size, -1).contiguous()

    return selected_index, parent_list


def normalize_external_draft_batch_spec_info(batch: ScheduleBatch) -> None:
    spec_info = getattr(batch, "spec_info", None)
    if not isinstance(spec_info, EagleDraftInput):
        return

    seq_lens = getattr(batch, "seq_lens", None)
    seq_lens_cpu = getattr(batch, "seq_lens_cpu", None)
    req_pool_indices = getattr(batch, "req_pool_indices", None)
    seq_lens_dtype = seq_lens.dtype if isinstance(seq_lens, torch.Tensor) else torch.int32
    seq_lens_cpu_dtype = (
        seq_lens_cpu.dtype if isinstance(seq_lens_cpu, torch.Tensor) else torch.int32
    )
    req_pool_indices_dtype = (
        req_pool_indices.dtype if isinstance(req_pool_indices, torch.Tensor) else torch.int32
    )

    live_count = sum(1 for req in batch.reqs if not req.is_retracted and not req.finished())
    if live_count == 0:
        batch.spec_info = EagleDraftInput.create_idle_input(
            device=batch.device,
            hidden_size=batch.model_config.hidden_size,
            dtype=batch.model_config.dtype,
            topk=1,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )
        return

    hidden_states = _slice_tensor_head_or_empty(
        spec_info.hidden_states,
        live_count,
        empty_shape=(live_count, batch.model_config.hidden_size),
        dtype=batch.model_config.dtype,
        device=batch.device,
    )
    verified_dtype = (
        spec_info.verified_id.dtype
        if isinstance(spec_info.verified_id, torch.Tensor)
        else torch.int32
    )
    capture_hidden_mode = getattr(
        spec_info, "capture_hidden_mode", CaptureHiddenMode.LAST
    )

    batch.spec_info = EagleDraftInput(
        hidden_states=hidden_states,
        verified_id=_slice_tensor_head_or_empty(
            spec_info.verified_id,
            live_count,
            empty_shape=(live_count,),
            dtype=verified_dtype,
            device=batch.device,
        ),
        topk_p=torch.empty((live_count, 1), dtype=torch.float32, device=batch.device),
        topk_index=torch.empty((live_count, 1), dtype=torch.int64, device=batch.device),
        capture_hidden_mode=capture_hidden_mode,
        accept_length=torch.zeros((live_count,), dtype=torch.int32, device=batch.device),
        accept_length_cpu=[0] * live_count,
        seq_lens_for_draft_extend=_slice_tensor_head_or_empty(
            getattr(spec_info, "seq_lens_for_draft_extend", seq_lens),
            live_count,
            empty_shape=(live_count,),
            dtype=seq_lens_dtype,
            device=batch.device,
        ),
        seq_lens_for_draft_extend_cpu=getattr(
            spec_info, "seq_lens_for_draft_extend_cpu", seq_lens_cpu
        )[:live_count]
        if getattr(spec_info, "seq_lens_for_draft_extend_cpu", seq_lens_cpu) is not None
        else torch.empty((0,), dtype=seq_lens_cpu_dtype),
        req_pool_indices_for_draft_extend=_slice_tensor_head_or_empty(
            getattr(spec_info, "req_pool_indices_for_draft_extend", req_pool_indices),
            live_count,
            empty_shape=(live_count,),
            dtype=req_pool_indices_dtype,
            device=batch.device,
        ),
    )


class VerifyWorker:
    verify = EAGLEWorker.verify
    _mamba_verify_update = EAGLEWorker._mamba_verify_update

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int | None,
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ) -> None:
        del gpu_id, moe_ep_rank, moe_dp_rank, nccl_port
        self.server_args = server_args
        self.target_worker = target_worker
        self.tp_rank = int(tp_rank)
        self.attn_cp_rank = int(attn_cp_rank)
        self.dp_rank = 0 if dp_rank is None else int(dp_rank)
        self.pp_rank = int(getattr(target_worker, "pp_rank", 0))
        self.model_runner = target_worker.model_runner
        self.model_config = target_worker.model_config
        self.page_size = server_args.page_size
        self.topk = 1
        self.speculative_num_steps = int(server_args.speculative_num_steps)
        self.speculative_num_draft_tokens = int(server_args.speculative_num_draft_tokens)
        self.enable_nan_detection = bool(server_args.enable_nan_detection)
        self.device = self.model_runner.device
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        self.total_accept_length = 0
        self.total_num_verified_reqs = 0
        self.total_round_forward_time_ms = 0.0
        self.total_round_forward_ct = 0
        self._last_logged_avg_req_len_bucket = 0
        self._last_verify_end_ts: float | None = None

    def clear_cache_pool(self):
        return

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        return self.target_worker.update_weights_from_tensor(recv_req)

    def _get_verify_buffers(self, draft_token_num: int):
        if draft_token_num != self.speculative_num_draft_tokens:
            return None, None

        attn_backend = getattr(self.target_worker.model_runner, "attn_backend", None)
        if attn_backend is None:
            return None, None

        get_buffers = getattr(
            attn_backend, "get_verify_buffers_to_fill_after_draft", None
        )
        if get_buffers is None:
            return None, None

        try:
            return get_buffers()
        except Exception as exc:
            logger.debug("Falling back to eager verify buffers: %s", exc)
            return None, None

    def _get_pad_token_id(self) -> int:
        hf_config = getattr(self.model_config, "hf_config", None)
        pad_token_id = _normalize_token_id(getattr(hf_config, "pad_token_id", None))
        if pad_token_id is not None:
            return pad_token_id

        eos_token_id = _normalize_token_id(getattr(hf_config, "eos_token_id", None))
        if eos_token_id is not None:
            return eos_token_id

        return 0

    def _build_req_verify_tokens(self, req, pad_token_id: int) -> list[int]:
        tail_token = _get_req_tail_token_id(req)
        draft_result = getattr(req, "draft_result", None)
        is_warmup_decode = bool(getattr(req, "needs_warmup_decode", False))
        verify_window_len = self.speculative_num_draft_tokens
        verify_replica_rank = self.dp_rank

        if draft_result is None:
            if is_warmup_decode:
                return [tail_token] + [pad_token_id] * (verify_window_len - 1)
            raise AssertionError("draft_result is None")

        if not draft_result.draft_token_ids:
            return [tail_token] + [pad_token_id] * (verify_window_len - 1)

        expected_rid = build_draft_scheduler_rid(req.rid)
        if draft_result.rid != expected_rid:
            raise RuntimeError(
                "Draft result rid mismatched verifier request: "
                f"{draft_result.rid!r} != expected {expected_rid!r}"
            )

        request_prompt_length = draft_result.request_prompt_length
        request_total_length = len(req.origin_input_ids) + len(req.output_ids)
        lower_bound = request_total_length - len(draft_result.draft_token_ids)
        if not (request_prompt_length < request_total_length and request_prompt_length >= lower_bound):
            raise RuntimeError(
                "Draft result prompt window mismatched verifier request tail: "
                f"{request_prompt_length=} {request_total_length=} "
                f"draft_token_count={len(draft_result.draft_token_ids)}"
            )

        if tail_token == draft_result.draft_token_ids[0]:
            return _fit_verify_window_tokens(
                draft_result.draft_token_ids,
                verify_window_len,
                pad_token_id,
            )

        return [tail_token] + [pad_token_id] * (verify_window_len - 1)

    def _build_verify_input(self, batch: ScheduleBatch) -> EagleVerifyInput:
        draft_token_num = self.speculative_num_draft_tokens
        if draft_token_num < 2:
            raise RuntimeError(
                "External draft verification requires at least one draft token per request."
            )

        pad_token_id = self._get_pad_token_id()
        full_draft_tokens_by_req = [
            self._build_req_verify_tokens(req, pad_token_id) for req in batch.reqs
        ]
        spec_steps = draft_token_num - 1
        verified_id = torch.tensor(
            [tokens[0] for tokens in full_draft_tokens_by_req],
            dtype=torch.long,
            device=batch.device,
        )
        draft_tokens = torch.tensor(
            [tokens[1:] for tokens in full_draft_tokens_by_req],
            dtype=torch.long,
            device=batch.device,
        )

        batch_size = batch.batch_size()
        seq_lens_sum = int(torch.sum(batch.seq_lens).item())
        selected_index, parent_list = _build_linear_topk1_tree_metadata(
            batch_size,
            spec_steps,
            batch.device,
        )

        tree_mask_buf, position_buf = self._get_verify_buffers(draft_token_num)
        (
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            flat_draft_tokens,
        ) = build_tree_kernel_efficient(
            verified_id=verified_id,
            parent_list=parent_list,
            top_scores_index=selected_index,
            draft_tokens=draft_tokens,
            seq_lens=batch.seq_lens,
            seq_lens_sum=seq_lens_sum,
            topk=1,
            spec_steps=spec_steps,
            num_verify_tokens=draft_token_num,
            tree_mask_mode=TreeMaskMode.FULL_MASK,
            tree_mask_buf=tree_mask_buf,
            position_buf=position_buf,
        )

        return EagleVerifyInput(
            draft_token=flat_draft_tokens,
            custom_mask=tree_mask,
            positions=positions,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=spec_steps,
            topk=1,
            draft_token_num=draft_token_num,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
        )

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        forward_start = time.perf_counter()
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = batch.get_model_worker_batch()
            result = self.target_worker.forward_batch_generation(model_worker_batch)
            if _is_cuda_device(batch.device):
                torch.cuda.synchronize()
            extend_forward_time_ms = (time.perf_counter() - forward_start) * 1000
            # if self._is_entry_rank():
            #     print(
            #         "[decoupled-verify] "
            #         f"extend_forward_time_ms={extend_forward_time_ms:.3f}",
            #         flush=True,
            #     )
            return result

        verify_input_start = time.perf_counter()
        spec_info = self._build_verify_input(batch)
        verify_input_duration_ms = (time.perf_counter() - verify_input_start) * 1000
        can_use_full_graph_path = (
            spec_info.draft_token_num == self.speculative_num_draft_tokens
        )
        emit_csv_event(
            "verify",
            "verify_input_built",
            server_role="verify",
            verify_replica_rank=self.dp_rank,
            batch_size=batch.batch_size(),
            live_req_count=batch.batch_size(),
            duration_ms=round(verify_input_duration_ms, 3),
            draft_token_num=int(spec_info.draft_token_num),
            spec_steps=int(spec_info.spec_steps),
            seq_lens_sum=int(spec_info.seq_lens_sum),
            message="verify worker built EagleVerifyInput",
        )
        verify_start = time.perf_counter()
        logits_output, verify_output, _, can_run_cuda_graph = self.verify(batch, spec_info)
        if _is_cuda_device(batch.device):
            torch.cuda.synchronize()
        verify_duration_ms = (time.perf_counter() - verify_start) * 1000
        round_forward_time_ms = (time.perf_counter() - forward_start) * 1000
        batch_size = batch.batch_size()
        avg_req_len = (
            float(batch.seq_lens.float().mean().item()) if batch_size > 0 else 0.0
        )
        avg_req_len_bucket = int(avg_req_len // 500)
        if avg_req_len_bucket > self._last_logged_avg_req_len_bucket:
            self._last_logged_avg_req_len_bucket = avg_req_len_bucket
            # if self._is_entry_rank():
            #     print(
            #         "[decoupled-verify] "
            #         f"bs={batch_size} "
            #         f"verify_ms={verify_duration_ms:.3f} "
            #         f"avg_req_len={avg_req_len:.3f}",
            #         flush=True,
            #     )

        normalize_external_draft_batch_spec_info(batch)
        result = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verify_output.verified_id,
            num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
            accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph and can_use_full_graph_path,
        )
        num_verified_reqs = len(verify_output.accept_length_per_req_cpu)
        self.total_accept_length += int(result.num_accepted_tokens) + num_verified_reqs
        self.total_num_verified_reqs += num_verified_reqs
        avg_accept_length = (
            self.total_accept_length / self.total_num_verified_reqs
            if self.total_num_verified_reqs > 0
            else 0.0
        )
        self.total_round_forward_time_ms += round_forward_time_ms
        self.total_round_forward_ct += 1
        avg_forward_time_ms = (
            self.total_round_forward_time_ms / self.total_round_forward_ct
            if self.total_round_forward_ct > 0
            else 0.0
        )
        target_disable_cuda_graph = bool(
            getattr(self.target_worker.model_runner.server_args, "disable_cuda_graph", False)
        )
        target_has_graph_runner = (
            getattr(self.target_worker.model_runner, "graph_runner", None) is not None
        )
        target_graph_can_run = bool(can_run_cuda_graph)
        verify_replica_rank = self.dp_rank
        emit_csv_event(
            "verify",
            "verify_batch_result",
            server_role="verify",
            verify_replica_rank=verify_replica_rank,
            batch_size=num_verified_reqs,
            live_req_count=num_verified_reqs,
            duration_ms=round(verify_duration_ms, 3),
            avg_accept_len=round(avg_accept_length, 6),
            message="verify batch finished",
            verify_cuda_graph=bool(can_run_cuda_graph and can_use_full_graph_path),
            can_use_full_graph_path=can_use_full_graph_path,
            target_disable_cuda_graph=target_disable_cuda_graph,
            target_has_graph_runner=target_has_graph_runner,
            target_graph_can_run=target_graph_can_run,
            avg_round_forward_time_ms=round(avg_forward_time_ms, 3),
        )
        emit_summary(
            logger,
            key=f"verify.stats.{verify_replica_rank}",
            component="verify",
            event="verify_stats_summary",
            message=(
                "verify stats updated "
                f"avg_accept_length={avg_accept_length:.6f} "
                f"avg_forward_time_ms={avg_forward_time_ms:.3f}"
            ),
            server_role="verify",
            verify_replica_rank=verify_replica_rank,
            batch_size=batch_size,
            avg_accept_length=round(avg_accept_length, 6),
            avg_forward_time_ms=round(avg_forward_time_ms, 3),
        )
        verify_end_ts = time.perf_counter()
        if self._last_verify_end_ts is not None:
            verify_to_verify_ms = (verify_end_ts - self._last_verify_end_ts) * 1000
            # print(
            #     "[decoupled-verify-gap] "
            #     f"dp_rank={verify_replica_rank} "
            #     f"bs={batch_size} "
            #     f"verify_to_verify_ms={verify_to_verify_ms:.3f}",
            #     flush=True,
            # )
        self._last_verify_end_ts = verify_end_ts
        return result
