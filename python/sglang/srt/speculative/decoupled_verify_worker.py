from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.decoupled_spec_io import (
    DraftBackendClient,
    DraftBackendMessage,
    DraftBackendMessageType,
    DraftLookupKey,
    DraftRequest,
    DraftResult,
    PollDraftResultsRequest,
    RequestTerminateMessage,
    RequestTerminateReason,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_utils import TreeMaskMode, build_tree_kernel_efficient
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.utils import broadcast_pyobj

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler
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


def _iter_live_batch_reqs(batch) -> list:
    return [req for req in batch.reqs if not req.is_retracted and not req.finished()]


def _build_sampling_params_dict(sampling_params) -> dict:
    params = {
        "max_new_tokens": sampling_params.max_new_tokens,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "top_k": sampling_params.top_k,
        "min_p": sampling_params.min_p,
        "frequency_penalty": sampling_params.frequency_penalty,
        "presence_penalty": sampling_params.presence_penalty,
        "repetition_penalty": sampling_params.repetition_penalty,
        "min_new_tokens": sampling_params.min_new_tokens,
        "n": sampling_params.n,
        "ignore_eos": sampling_params.ignore_eos,
        "skip_special_tokens": sampling_params.skip_special_tokens,
        "spaces_between_special_tokens": sampling_params.spaces_between_special_tokens,
        "no_stop_trim": sampling_params.no_stop_trim,
        "custom_params": sampling_params.custom_params,
        "stream_interval": sampling_params.stream_interval,
        "logit_bias": sampling_params.logit_bias,
        "sampling_seed": sampling_params.sampling_seed,
    }
    if sampling_params.stop_strs is not None:
        params["stop"] = sampling_params.stop_strs
    if sampling_params.stop_token_ids is not None:
        params["stop_token_ids"] = list(sampling_params.stop_token_ids)
    if sampling_params.stop_regex_strs is not None:
        params["stop_regex"] = sampling_params.stop_regex_strs
    if sampling_params.json_schema is not None:
        params["json_schema"] = sampling_params.json_schema
    if sampling_params.regex is not None:
        params["regex"] = sampling_params.regex
    if sampling_params.ebnf is not None:
        params["ebnf"] = sampling_params.ebnf
    if sampling_params.structural_tag is not None:
        params["structural_tag"] = sampling_params.structural_tag
    return params


@dataclass
class VerifyDraftSessionState:
    request_id: str
    next_draft_round_id: int = 0
    waiting_keys: deque[DraftLookupKey] = field(default_factory=deque)
    needs_warmup_decode: bool = False

    def alloc_next_round_id(self) -> int:
        round_id = int(self.next_draft_round_id)
        self.next_draft_round_id = round_id + 1
        return round_id

    def peek_waiting_key(self) -> DraftLookupKey | None:
        if not self.waiting_keys:
            return None
        return self.waiting_keys[0]

    def append_waiting_key(self, waiting_key: DraftLookupKey) -> None:
        self.waiting_keys.append(waiting_key)

    def pop_waiting_key(self) -> DraftLookupKey | None:
        if not self.waiting_keys:
            return None
        return self.waiting_keys.popleft()


@dataclass
class VerifyCoordinatorState:
    sessions: dict[str, VerifyDraftSessionState] = field(default_factory=dict)
    submit_times_by_key: dict[DraftLookupKey, float] = field(default_factory=dict)


@dataclass
class VerifyBatchActions:
    draft_requests: list[DraftRequest] = field(default_factory=list)
    terminate_messages: list[RequestTerminateMessage] = field(default_factory=list)


class VerifyCoordinator:
    def __init__(
        self,
        *,
        scheduler_dp_rank: int,
        num_speculative_steps: int,
    ) -> None:
        self.scheduler_dp_rank = int(scheduler_dp_rank)
        self.num_speculative_steps = int(num_speculative_steps)
        self.state = VerifyCoordinatorState()

    def get_or_create_session(self, request_id: str) -> VerifyDraftSessionState:
        session = self.state.sessions.get(request_id)
        if session is None:
            session = VerifyDraftSessionState(request_id=request_id)
            self.state.sessions[request_id] = session
        return session

    def get_session(self, request_id: str) -> VerifyDraftSessionState | None:
        return self.state.sessions.get(request_id)

    def alloc_next_round_id(self, request_id: str) -> int:
        return self.get_or_create_session(request_id).alloc_next_round_id()

    def peek_waiting_key(self, request_id: str) -> DraftLookupKey | None:
        session = self.get_session(request_id)
        if session is None:
            return None
        return session.peek_waiting_key()

    def append_waiting_key(
        self, request_id: str, waiting_key: DraftLookupKey
    ) -> None:
        self.get_or_create_session(request_id).append_waiting_key(waiting_key)

    def pop_waiting_key(self, request_id: str) -> DraftLookupKey | None:
        session = self.get_session(request_id)
        if session is None:
            return None
        waiting_key = session.pop_waiting_key()
        if not session.waiting_keys and not session.needs_warmup_decode:
            self.state.sessions.pop(request_id, None)
        return waiting_key

    def mark_warmup_decode(self, request_id: str) -> None:
        self.get_or_create_session(request_id).needs_warmup_decode = True

    def clear_warmup_decode(self, request_id: str) -> None:
        session = self.get_session(request_id)
        if session is None:
            return
        session.needs_warmup_decode = False
        if not session.waiting_keys:
            self.state.sessions.pop(request_id, None)

    def needs_warmup_decode(self, request_id: str) -> bool:
        session = self.get_session(request_id)
        return bool(session is not None and session.needs_warmup_decode)

    def clear_request(self, request_id: str) -> None:
        self.state.sessions.pop(request_id, None)
        for key in list(self.state.submit_times_by_key):
            if key.request_id == request_id:
                self.state.submit_times_by_key.pop(key, None)

    def record_submit_time(self, waiting_key: DraftLookupKey, submit_ts: float) -> None:
        self.state.submit_times_by_key[waiting_key] = float(submit_ts)

    def pop_submit_time(self, waiting_key: DraftLookupKey) -> float | None:
        return self.state.submit_times_by_key.pop(waiting_key, None)

    def build_draft_request(self, req, draft_round_id: int) -> DraftRequest:
        return DraftRequest(
            request_id=req.rid,
            draft_round_id=draft_round_id,
            scheduler_dp_rank=self.scheduler_dp_rank,
            prompt_token_ids=list(req.origin_input_ids),
            committed_token_ids=list(req.output_ids),
            num_speculative_steps=self.num_speculative_steps,
            sampling_params=_build_sampling_params_dict(req.sampling_params),
        )

    def register_submitted_request(
        self,
        req,
        draft_request: DraftRequest,
        needs_warmup_decode: bool,
        submit_ts: float,
    ) -> None:
        self.append_waiting_key(req.rid, draft_request.key)
        self.record_submit_time(draft_request.key, submit_ts)
        if needs_warmup_decode:
            self.mark_warmup_decode(req.rid)
        setattr(req, "needs_warmup_decode", needs_warmup_decode)
        setattr(req, "draft_result", None)
        self.get_or_create_session(req.rid)

    def build_submit_batch(
        self,
        reqs: list,
        warmup_request_ids: set[str] | None = None,
    ) -> list[DraftRequest]:
        draft_requests: list[DraftRequest] = []
        submit_ts = time.perf_counter()
        for req in reqs:
            draft_round_id = self.alloc_next_round_id(req.rid)
            draft_request = self.build_draft_request(req, draft_round_id)
            needs_warmup_decode = bool(
                warmup_request_ids and req.rid in warmup_request_ids
            )
            self.register_submitted_request(
                req,
                draft_request,
                needs_warmup_decode=needs_warmup_decode,
                submit_ts=submit_ts,
            )
            draft_requests.append(draft_request)
        return draft_requests

    def collect_missing_poll_keys(self, live_reqs) -> list[DraftLookupKey]:
        missing_keys: list[DraftLookupKey] = []
        for req in live_reqs:
            session = self.get_session(req.rid)
            if session is None:
                continue
            waiting_key = session.peek_waiting_key()
            if waiting_key is not None:
                missing_keys.append(waiting_key)
        return missing_keys

    def bind_polled_results_to_live_reqs(
        self,
        live_reqs,
        results: list[DraftResult],
    ) -> list[tuple[DraftResult, float | None]]:
        bind_records: list[tuple[DraftResult, float | None]] = []
        results_by_key = {result.key: result for result in results}
        for req in live_reqs:
            waiting_key = self.peek_waiting_key(req.rid)
            if waiting_key is None:
                setattr(req, "draft_result", None)
                continue

            draft_result = results_by_key.get(waiting_key)
            if draft_result is None:
                raise RuntimeError(
                    f"Draft result missing for request {req.rid} round {waiting_key.draft_round_id}"
                )
            session = self.get_session(req.rid)
            if session is None:
                raise RuntimeError(
                    f"Draft session missing while binding request {req.rid}"
                )
            expected_round_id = session.next_draft_round_id - 2
            if draft_result.key != waiting_key:
                raise RuntimeError(
                    f"Draft result key mismatch for request {req.rid}: "
                    f"{draft_result.key!r} != {waiting_key!r}"
                )
            if draft_result.draft_round_id != expected_round_id:
                raise RuntimeError(
                    "Draft result round mismatch: "
                    f"{draft_result.draft_round_id} != expected {expected_round_id}"
                )
            submit_ts = self.pop_submit_time(waiting_key)
            self.pop_waiting_key(req.rid)
            setattr(req, "draft_result", draft_result)
            submit_to_result_ms = None
            if submit_ts is not None:
                submit_to_result_ms = (time.perf_counter() - submit_ts) * 1000
            bind_records.append((draft_result, submit_to_result_ms))
        return bind_records
    def build_post_batch_actions(self, batch_reqs) -> VerifyBatchActions:
        actions = VerifyBatchActions()
        requests_to_submit = []
        for req in batch_reqs:
            if req.is_retracted or req.finished():
                terminate_reason = (
                    RequestTerminateReason.ABORT
                    if req.is_retracted
                    else RequestTerminateReason.FINISHED
                )
                actions.terminate_messages.append(
                    RequestTerminateMessage(
                        request_id=req.rid,
                        reason=terminate_reason,
                    )
                )
                self.clear_request(req.rid)
                continue

            if self.needs_warmup_decode(req.rid):
                self.clear_warmup_decode(req.rid)
                setattr(req, "needs_warmup_decode", False)

            requests_to_submit.append(req)

        actions.draft_requests = self.build_submit_batch(requests_to_submit)
        return actions


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
        draft_backend_client: DraftBackendClient | None = None,
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
        self.total_accepted_draft_tokens = 0
        self.total_verified_tokens = 0
        self.total_verified_reqs = 0
        self.total_round_forward_time_ms = 0.0
        self.total_round_forward_ct = 0
        self._last_logged_avg_req_len_bucket = 0
        self.coordinator = VerifyCoordinator(
            scheduler_dp_rank=self.dp_rank,
            num_speculative_steps=self.speculative_num_steps,
        )
        self.draft_backend_client = draft_backend_client
        if self._is_entry_rank() and self.draft_backend_client is None:
            raise RuntimeError(
                "Draft backend client is required on decoupled_verify entry rank"
            )

    def _is_entry_rank(self) -> bool:
        return self.pp_rank == 0 and self.tp_rank == 0 and self.attn_cp_rank == 0

    def clear_cache_pool(self):
        return

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        return self.target_worker.update_weights_from_tensor(recv_req)

    def _send_backend_message(self, message: DraftBackendMessage) -> None:
        if not self._is_entry_rank():
            return
        if self.draft_backend_client is None:
            raise RuntimeError("Draft backend client is not initialized on entry rank")
        self.draft_backend_client.send_message(message)

    def _recv_poll_response(self) -> list[DraftResult]:
        if not self._is_entry_rank():
            return []
        if self.draft_backend_client is None:
            raise RuntimeError("Draft backend client is not initialized on entry rank")

        while True:
            response = self.draft_backend_client.recv_message()
            if (
                response.message_type != DraftBackendMessageType.POLL_RESPONSE
                or response.poll_response is None
            ):
                raise RuntimeError(
                    f"Unexpected draft backend response: {response.message_type}"
                )
            return list(response.poll_response.results)

    def _sync_polled_results(
        self,
        scheduler: Scheduler,
        local_results: list[DraftResult] | None,
    ) -> list[DraftResult]:
        source_payload = list(local_results or []) if self._is_entry_rank() else []
        if scheduler.tp_size == 1:
            return source_payload
        synced_results = broadcast_pyobj(
            source_payload,
            scheduler.tp_group.rank,
            scheduler.tp_cpu_group,
            src=scheduler.tp_group.ranks[0],
        )
        return list(synced_results)

    def prepare_verify_batch(self, batch: ScheduleBatch, scheduler: Scheduler) -> None:
        if batch is None or not batch.forward_mode.is_decode():
            return

        live_reqs = _iter_live_batch_reqs(batch)
        for req in live_reqs:
            setattr(req, "draft_result", None)
            setattr(
                req,
                "needs_warmup_decode",
                self.coordinator.needs_warmup_decode(req.rid),
            )

        target_reqs = [
            req
            for req in live_reqs
            if not self.coordinator.needs_warmup_decode(req.rid)
        ]
        if not target_reqs:
            return

        missing_keys = self.coordinator.collect_missing_poll_keys(target_reqs)
        polled_results = None
        if self._is_entry_rank() and missing_keys:
            poll_wait_start = time.perf_counter()
            self._send_backend_message(
                DraftBackendMessage.from_poll_request(
                    PollDraftResultsRequest(keys=missing_keys)
                )
            )
            polled_results = self._recv_poll_response()
            poll_wait_ms = (time.perf_counter() - poll_wait_start) * 1000
            # print(
            #     "[decoupled-verify] "
            #     f"poll_wait_ms={poll_wait_ms:.3f} "
            #     f"missing_keys={len(missing_keys)} "
            #     f"ready_results={len(polled_results)}",
            #     flush=True,
            # )

        synced_results = self._sync_polled_results(scheduler, polled_results)
        bind_records = self.coordinator.bind_polled_results_to_live_reqs(
            target_reqs, synced_results
        )
        if self._is_entry_rank():
            for draft_result, submit_to_result_ms in bind_records:
                if submit_to_result_ms is None:
                    continue
                # print(
                #     "[decoupled-verify] "
                #     f"submit_draft_to_result_ms={submit_to_result_ms:.3f} "
                #     f"rid={draft_result.request_id} "
                #     f"round={draft_result.draft_round_id}",
                #     flush=True,
                # )

    def _submit_draft_requests(self, requests: list[DraftRequest]) -> None:
        if requests and self._is_entry_rank():
            self._send_backend_message(DraftBackendMessage.from_submit_draft(requests))

    def _terminate_request(self, message: RequestTerminateMessage) -> None:
        if self._is_entry_rank():
            self._send_backend_message(DraftBackendMessage.from_request_terminate(message))

    def abort_request_state(self, request_id: str) -> None:
        self.coordinator.clear_request(request_id)
        self._terminate_request(
            RequestTerminateMessage(
                request_id=request_id,
                reason=RequestTerminateReason.ABORT,
            )
        )

    def after_process_batch(
        self,
        batch: ScheduleBatch,
        scheduler: Scheduler,
    ) -> None:
        if batch.forward_mode.is_extend() and not batch.is_dllm():
            warmup_request_ids = _get_prefill_batch_warmup_request_ids(batch)
            reqs_to_submit = [
                req for req in _iter_live_batch_reqs(batch) if req.is_chunked <= 0
            ]
            draft_requests = self.coordinator.build_submit_batch(
                reqs_to_submit,
                warmup_request_ids=warmup_request_ids,
            )
            self._submit_draft_requests(draft_requests)
            return

        if batch.forward_mode.is_decode():
            actions = self.coordinator.build_post_batch_actions(batch.reqs)
            self._submit_draft_requests(actions.draft_requests)
            for terminate_message in actions.terminate_messages:
                self._terminate_request(terminate_message)

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

        if draft_result is None:
            if is_warmup_decode:
                return [tail_token] + [pad_token_id] * (verify_window_len - 1)
            raise AssertionError("draft_result is None")

        if not draft_result.draft_token_ids:
            return [tail_token] + [pad_token_id] * (verify_window_len - 1)

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

        # if self._is_entry_rank():
        #     print(
        #         "[decoupled-diagnose] verify anchor mismatch "
        #         f"rid={req.rid} round={draft_result.draft_round_id} "
        #         f"tail_token={tail_token} "
        #         f"draft_anchor_token={draft_result.draft_token_ids[0]} "
        #         f"prompt_len={len(req.origin_input_ids)} "
        #         f"committed_len={len(req.output_ids)} "
        #         f"draft_token_count={len(draft_result.draft_token_ids)} "
        #         "draft_tokens_head="
        #         f"{draft_result.draft_token_ids[: min(4, len(draft_result.draft_token_ids))]}",
        #         flush=True,
        #     )
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

        spec_info = self._build_verify_input(batch)
        can_use_full_graph_path = (
            spec_info.draft_token_num == self.speculative_num_draft_tokens
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

        verified_reqs = len(verify_output.accept_length_per_req_cpu)
        accepted_draft_tokens = int(result.num_accepted_tokens)
        verified_tokens = accepted_draft_tokens + verified_reqs
        self.total_accepted_draft_tokens += accepted_draft_tokens
        self.total_verified_tokens += verified_tokens
        self.total_verified_reqs += verified_reqs
        avg_tokens_per_round = (
            self.total_verified_tokens / self.total_verified_reqs
            if self.total_verified_reqs > 0
            else 0.0
        )
        self.total_round_forward_time_ms += round_forward_time_ms
        self.total_round_forward_ct += 1
        avg_round_forward_time_ms = (
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
        # if self._is_entry_rank():
        #     print(
        #         "[decoupled-verify] "
        #         f"accepted_this_round={accepted_draft_tokens} "
        #         f"avg_tokens_per_round={avg_tokens_per_round:.3f} "
        #         f"target_disable_cuda_graph={target_disable_cuda_graph} "
        #         f"target_has_graph_runner={target_has_graph_runner} "
        #         f"target_graph_can_run={target_graph_can_run} "
        #         f"verify_can_run_cuda_graph={bool(can_run_cuda_graph)} "
        #         f"verify_use_full_graph_path={bool(can_use_full_graph_path)} "
        #         f"avg_round_forward_time_ms={avg_round_forward_time_ms:.3f}",
        #         flush=True,
        #     )
        return result


def _get_prefill_batch_warmup_request_ids(batch) -> set[str]:
    decoding_rids = {req.rid for req in (batch.decoding_reqs or [])}
    warmup_request_ids = set()
    for req in _iter_live_batch_reqs(batch):
        if req.rid in decoding_rids or req.is_chunked > 0:
            continue
        warmup_request_ids.add(req.rid)
    return warmup_request_ids
