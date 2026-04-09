from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
import time

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.speculative.decoupled_spec_io import DraftRequest, DraftResult
from sglang.srt.speculative.draft_session import DraftServerSession, DraftSessionManager


def _build_scheduler_session_id(request_id: str) -> str:
    return f"draft-session-{request_id}"


def _build_scheduler_rid(request_id: str) -> str:
    return f"draft-{request_id}"


def build_generate_req_from_draft_request(
    draft_request: DraftRequest,
    *,
    request_id: str,
    max_new_tokens: int,
    draft_session_id: str,
) -> GenerateReqInput:
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
        draft_session_id=draft_session_id,
        draft_stateful_mode=True,
        custom_labels={
            "draft_request_id": draft_request.request_id,
            "draft_round_id": str(draft_request.draft_round_id),
        },
    )


class DrafterServiceApi(ABC):
    @abstractmethod
    async def handle_draft_request(self, draft_request: DraftRequest) -> DraftResult:
        raise NotImplementedError

    @abstractmethod
    async def terminate_draft_request(
        self,
        request_id: str,
        draft_round_id_upper_bound: int | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def release_draft_session(self, request_id: str) -> None:
        raise NotImplementedError


class LocalDrafterService(DrafterServiceApi):
    def __init__(self, tokenizer_manager, max_model_len: int):
        self.tokenizer_manager = tokenizer_manager
        self.max_model_len = max_model_len
        self.session_manager = DraftSessionManager(
            execute_round=self._execute_round,
            release_session=self._release_session,
        )

    async def handle_draft_request(self, draft_request: DraftRequest) -> DraftResult:
        return await self.session_manager.submit(
            request_id=draft_request.request_id,
            scheduler_session_id=_build_scheduler_session_id(draft_request.request_id),
            scheduler_rid=_build_scheduler_rid(draft_request.request_id),
            draft_request=draft_request,
        )

    async def terminate_draft_request(
        self,
        request_id: str,
        draft_round_id_upper_bound: int | None = None,
    ) -> None:
        session = self.session_manager.get(request_id)
        if session is not None and session.running:
            self.tokenizer_manager.abort_request(session.scheduler_rid)
        await self.session_manager.terminate(
            request_id,
            draft_round_id_upper_bound=draft_round_id_upper_bound,
        )

    async def release_draft_session(self, request_id: str) -> None:
        await self.session_manager.release(request_id)

    async def _release_session(self, session: DraftServerSession) -> None:
        await self.tokenizer_manager.release_draft_session(session.scheduler_session_id)

    async def _execute_round(
        self,
        session: DraftServerSession,
        draft_request: DraftRequest,
    ) -> DraftResult:
        round_start = time.perf_counter()
        prompt_ids = draft_request.full_token_ids
        request_prompt_length = len(prompt_ids)
        max_possible_tokens = self.max_model_len - request_prompt_length
        if max_possible_tokens <= 0:
            return DraftResult(
                request_id=draft_request.request_id,
                draft_round_id=draft_request.draft_round_id,
                request_prompt_length=request_prompt_length,
                draft_token_ids=[],
            )

        max_new_tokens = max(
            0,
            min(draft_request.num_speculative_steps + 1, max_possible_tokens),
        )
        generate_request = build_generate_req_from_draft_request(
            draft_request,
            request_id=session.scheduler_rid,
            max_new_tokens=max_new_tokens,
            draft_session_id=session.scheduler_session_id,
        )
        output = await self.tokenizer_manager.generate_request(
            generate_request, None
        ).__anext__()
        finish_reason = output.get("meta_info", {}).get("finish_reason")
        if finish_reason is None:
            raise RuntimeError(
                "Draft generate_request finished without finish_reason metadata"
            )

        draft_token_ids = list(output.get("output_ids", []))
        round_e2e_ms = (time.perf_counter() - round_start) * 1000.0

        # print(
        #     "[decoupled-draft] "
        #     f"rid={draft_request.request_id} "
        #     f"round={draft_request.draft_round_id} "
        #     f"prompt_len={request_prompt_length} "
        #     f"draft_tokens={len(draft_token_ids)} "
        #     f"draft_round_e2e_ms={round_e2e_ms:.3f} "
        #     f"finish_reason={finish_reason}",
        #     flush=True,
        # )
        return DraftResult(
            request_id=draft_request.request_id,
            draft_round_id=draft_request.draft_round_id,
            request_prompt_length=request_prompt_length,
            draft_token_ids=draft_token_ids,
        )
