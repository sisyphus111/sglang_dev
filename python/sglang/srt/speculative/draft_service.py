from __future__ import annotations

from copy import deepcopy
import logging
import time
from collections import defaultdict

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.speculative.decoupled_spec_io import (
    DraftRequest,
    DraftResult,
    build_draft_round_scheduler_rid,
    build_draft_scheduler_rid,
)
from sglang.srt.utils.csv_debug_utils import emit_summary

logger = logging.getLogger(__name__)

def _build_draft_custom_labels(
    draft_request: DraftRequest,
    *,
    sglang_rid: str,
) -> dict[str, str]:
    return {
        "draft_trace": "1",
        "draft_external_rid": str(draft_request.request_id),
        "draft_round_id": str(draft_request.draft_round_id),
        "verify_replica_rank": str(draft_request.scheduler_dp_rank),
        "sglang_rid": str(sglang_rid),
    }


def build_generate_req_from_draft_request(
    draft_request: DraftRequest,
    *,
    request_id: str,
    max_new_tokens: int,
) -> GenerateReqInput:
    if not request_id:
        raise AssertionError(
            "Draft round scheduler rid must be non-empty while building GenerateReqInput"
        )
    sampling_params = deepcopy(draft_request.sampling_params)
    sampling_params.pop("max_tokens", None)
    sampling_params["max_new_tokens"] = max_new_tokens
    sampling_params["ignore_eos"] = True
    sampling_params["temperature"] = 0.0
    sampling_params["top_k"] = -1
    sampling_params["top_p"] = 1.0

    return GenerateReqInput(
        rid=request_id,
        input_ids=draft_request.full_token_ids,
        sampling_params=sampling_params,
        return_logprob=False,
        stream=False,
        custom_labels=_build_draft_custom_labels(
            draft_request,
            sglang_rid=request_id,
        ),
    )


class LocalDrafterService:
    def __init__(self, tokenizer_manager, max_model_len: int):
        self.tokenizer_manager = tokenizer_manager
        self.max_model_len = max_model_len
        self.active_submission_rids: dict[str, set[str]] = defaultdict(set)

    async def handle_draft_request(self, draft_request: DraftRequest) -> DraftResult:
        return (await self.handle_draft_requests([draft_request]))[0]

    async def handle_draft_requests(
        self, draft_requests: list[DraftRequest]
    ) -> list[DraftResult]:
        if not draft_requests:
            return []

        request_start = time.perf_counter()
        request_prompt_lengths = {
            request.key: len(request.full_token_ids) for request in draft_requests
        }
        stable_rids = {
            request.key: build_draft_scheduler_rid(request.request_id)
            for request in draft_requests
        }
        internal_rids = {
            request.key: build_draft_round_scheduler_rid(
                request.request_id, request.draft_round_id
            )
            for request in draft_requests
        }

        immediate_results: list[DraftResult | None] = [None] * len(draft_requests)
        batched_requests: list[DraftRequest] = []
        batched_internal_rids: list[str] = []
        batched_input_ids: list[list[int]] = []
        batched_sampling_params: list[dict] = []
        batched_custom_labels: list[dict[str, str]] = []

        for index, draft_request in enumerate(draft_requests):
            stable_rid = stable_rids[draft_request.key]
            internal_rid = internal_rids[draft_request.key]
            request_prompt_length = request_prompt_lengths[draft_request.key]
            if draft_request.rid != stable_rid:
                raise AssertionError(
                    "DraftRequest rid mismatch in LocalDrafterService: "
                    f"{draft_request.rid!r} != {stable_rid!r}"
                )

            max_possible_tokens = self.max_model_len - request_prompt_length
            if max_possible_tokens <= 0:
                immediate_results[index] = DraftResult(
                    request_id=draft_request.request_id,
                    rid=stable_rid,
                    draft_round_id=draft_request.draft_round_id,
                    request_prompt_length=request_prompt_length,
                    draft_token_ids=[],
                )
                continue

            max_new_tokens = max(
                0,
                min(draft_request.num_speculative_steps + 1, max_possible_tokens),
            )
            build_start = time.perf_counter()
            sampling_params = deepcopy(draft_request.sampling_params)
            sampling_params.pop("max_tokens", None)
            sampling_params["max_new_tokens"] = max_new_tokens
            sampling_params["ignore_eos"] = True
            sampling_params["temperature"] = 0.0
            sampling_params["top_k"] = -1
            sampling_params["top_p"] = 1.0
            batched_requests.append(draft_request)
            batched_internal_rids.append(internal_rid)
            batched_input_ids.append(draft_request.full_token_ids)
            batched_sampling_params.append(sampling_params)
            batched_custom_labels.append(
                _build_draft_custom_labels(
                    draft_request,
                    sglang_rid=internal_rid,
                )
            )

        if batched_requests:
            rounds_by_request_id: dict[str, list[int]] = defaultdict(list)
            for draft_request in batched_requests:
                rounds_by_request_id[draft_request.request_id].append(
                    draft_request.draft_round_id
                )
            duplicate_rounds = {
                request_id: rounds
                for request_id, rounds in rounds_by_request_id.items()
                if len(rounds) > 1
            }
            
            assert not duplicate_rounds, f"Duplicate draft rounds detected in batch: {duplicate_rounds}"

            for draft_request, internal_rid in zip(
                batched_requests, batched_internal_rids, strict=True
            ):
                self.active_submission_rids[draft_request.request_id].add(internal_rid)

            generate_start = time.perf_counter()
            try:
                output = await self.tokenizer_manager.generate_request(
                    GenerateReqInput(
                        rid=batched_internal_rids,
                        input_ids=batched_input_ids,
                        sampling_params=batched_sampling_params,
                        return_logprob=False,
                        stream=False,
                        custom_labels=batched_custom_labels,
                    ),
                    None,
                ).__anext__()
            except Exception:
                for draft_request, internal_rid in zip(
                    batched_requests, batched_internal_rids, strict=True
                ):
                    self.active_submission_rids[draft_request.request_id].discard(
                        internal_rid
                    )
                    if not self.active_submission_rids[draft_request.request_id]:
                        self.active_submission_rids.pop(draft_request.request_id, None)
                raise

            generate_return_ts = time.perf_counter()

            normalize_output_start = time.perf_counter()
            outputs = output if isinstance(output, list) else [output]
            if len(outputs) != len(batched_requests):
                raise RuntimeError(
                    "Draft batch generate_request returned unexpected number of outputs: "
                    f"{len(outputs)} != {len(batched_requests)}"
                )
            normalize_output_duration_ms = round(
                (time.perf_counter() - normalize_output_start) * 1000, 3
            )

            results_by_key: dict[tuple[str, int], DraftResult] = {}
            build_results_start = time.perf_counter()
            for draft_request, internal_rid, single_output in zip(
                batched_requests, batched_internal_rids, outputs, strict=True
            ):
                single_result_start = time.perf_counter()
                self.active_submission_rids[draft_request.request_id].discard(
                    internal_rid
                )
                if not self.active_submission_rids[draft_request.request_id]:
                    self.active_submission_rids.pop(draft_request.request_id, None)

                finish_reason = single_output.get("meta_info", {}).get("finish_reason")
                if finish_reason is None:
                    raise RuntimeError(
                        "Draft generate_request finished without finish_reason metadata"
                    )

                draft_token_ids = list(single_output.get("output_ids", []))
                single_result_build_duration_ms = round(
                    (time.perf_counter() - single_result_start) * 1000, 3
                )
                emit_summary(
                    logger,
                    key=f"draft_server.summary.{draft_request.scheduler_dp_rank}",
                    component="draft_server",
                    event="drafter_request_summary",
                    message=(
                        "draft request completed "
                        f"request_prompt_length={request_prompt_lengths[draft_request.key]} "
                        f"draft_token_count={len(draft_token_ids)}"
                    ),
                    server_role="draft",
                    verify_replica_rank=draft_request.scheduler_dp_rank,
                    duration_ms=round(
                        (time.perf_counter() - request_start) * 1000, 3
                    ),
                )
                results_by_key[(draft_request.request_id, draft_request.draft_round_id)] = (
                    DraftResult(
                        request_id=draft_request.request_id,
                        rid=stable_rids[draft_request.key],
                        draft_round_id=draft_request.draft_round_id,
                        request_prompt_length=request_prompt_lengths[draft_request.key],
                        draft_token_ids=draft_token_ids,
                    )
                )

            batched_iter = iter(batched_requests)
            for index, result in enumerate(immediate_results):
                if result is not None:
                    continue
                request = next(batched_iter)
                immediate_results[index] = results_by_key[
                    (request.request_id, request.draft_round_id)
                ]

        return [result for result in immediate_results if result is not None]

    async def terminate_draft_request(
        self,
        request_id: str,
    ) -> None:
        active_rids = sorted(self.active_submission_rids.pop(request_id, set()))
        for internal_rid in active_rids:
            self.tokenizer_manager.abort_request(internal_rid)

    async def release_draft_session(self, request_id: str) -> None:
        await self.tokenizer_manager.release_draft_session(
            build_draft_scheduler_rid(request_id)
        )
