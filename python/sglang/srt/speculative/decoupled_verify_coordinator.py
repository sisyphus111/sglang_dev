from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from sglang.srt.speculative.decoupled_spec_io import (
    DraftLookupKey,
    DraftRequest,
    DraftResult,
    RequestTerminateMessage,
    RequestTerminateReason,
    build_draft_scheduler_rid,
)
from sglang.srt.utils.csv_debug_utils import emit_csv_event


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
        self.scheduler_dp_rank = 0 if scheduler_dp_rank is None else int(scheduler_dp_rank)
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

    def get_waiting_key_depth(self, request_id: str) -> int:
        session = self.get_session(request_id)
        if session is None:
            return 0
        return len(session.waiting_keys)

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
            rid=build_draft_scheduler_rid(req.rid),
            draft_round_id=draft_round_id,
            scheduler_dp_rank=self.scheduler_dp_rank,
            prompt_token_ids=list(req.origin_input_ids),
            committed_token_ids=list(req.output_ids),
            num_speculative_steps=self.num_speculative_steps,
            sampling_params=_build_sampling_params_dict(req.sampling_params),
        )

    def assert_can_submit_request(self, request_id: str, draft_round_id: int) -> None:
        session = self.get_or_create_session(request_id)
        waiting_depth = len(session.waiting_keys)
        if waiting_depth >= 2:
            waiting_rounds = [key.draft_round_id for key in session.waiting_keys]
            raise AssertionError(
                "Verify coordinator submit backlog exceeded limit: "
                f"request_id={request_id}, draft_round_id={draft_round_id}, "
                f"waiting_depth={waiting_depth}, waiting_rounds={waiting_rounds}, "
                f"next_draft_round_id={session.next_draft_round_id}"
            )

        if session.waiting_keys:
            last_waiting_round_id = session.waiting_keys[-1].draft_round_id
            if draft_round_id <= last_waiting_round_id:
                raise AssertionError(
                    "Verify coordinator draft round order violation: "
                    f"request_id={request_id}, draft_round_id={draft_round_id}, "
                    f"last_waiting_round_id={last_waiting_round_id}, "
                    f"next_draft_round_id={session.next_draft_round_id}"
                )

    def register_submitted_request(
        self,
        req,
        draft_request: DraftRequest,
        needs_warmup_decode: bool,
        submit_ts: float,
    ) -> None:
        self.assert_can_submit_request(req.rid, draft_request.draft_round_id)
        self.append_waiting_key(req.rid, draft_request.key)
        self.record_submit_time(draft_request.key, submit_ts)
        if needs_warmup_decode:
            self.mark_warmup_decode(req.rid)
        setattr(req, "needs_warmup_decode", needs_warmup_decode)
        setattr(req, "draft_result", None)
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
        if draft_requests:
            emit_csv_event(
                "verify_coordinator",
                "verify_submit_batch_built",
                server_role="verify",
                verify_replica_rank=self.scheduler_dp_rank,
                batch_size=len(draft_requests),
                live_req_count=len(draft_requests),
                message="verify coordinator built draft submit batch",
            )
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
                raise AssertionError(
                    f"Draft result missing for request {req.rid} round {waiting_key.draft_round_id}"
                )
            session = self.get_session(req.rid)
            if session is None:
                raise AssertionError(
                    f"Draft session missing while binding request {req.rid}"
                )
            expected_round_id = session.next_draft_round_id - 2
            if draft_result.key != waiting_key:
                raise AssertionError(
                    f"Draft result key mismatch for request {req.rid}: "
                    f"{draft_result.key!r} != {waiting_key!r}"
                )
            if draft_result.draft_round_id != expected_round_id:
                raise AssertionError(
                    "Draft result round mismatch: "
                    f"{draft_result.draft_round_id} != expected {expected_round_id}; "
                    f"waiting_head={waiting_key.draft_round_id}, "
                    f"next_draft_round_id={session.next_draft_round_id}, "
                    f"waiting_queue_depth={len(session.waiting_keys)}"
                )
            expected_rid = build_draft_scheduler_rid(req.rid)
            if draft_result.rid != expected_rid:
                raise AssertionError(
                    "Draft result rid mismatch for request "
                    f"{req.rid}: {draft_result.rid!r} != expected {expected_rid!r}"
                )
            submit_ts = self.pop_submit_time(waiting_key)
            self.pop_waiting_key(req.rid)
            setattr(req, "draft_result", draft_result)
            submit_to_result_ms = None
            if submit_ts is not None:
                submit_to_result_ms = (time.perf_counter() - submit_ts) * 1000
            bind_records.append((draft_result, submit_to_result_ms))
        return bind_records

    def build_after_batch_actions(self, batch_reqs) -> VerifyBatchActions:
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

            waiting_key_depth = self.get_waiting_key_depth(req.rid)
            if self.needs_warmup_decode(req.rid):
                self.clear_warmup_decode(req.rid)
                setattr(req, "needs_warmup_decode", False)

            # Decoupled verify is designed to keep at most two outstanding rounds
            # per request: the current round being waited on plus the next round
            # already submitted by the verifier pipeline.
            if waiting_key_depth >= 2:
                continue

            requests_to_submit.append(req)

        actions.draft_requests = self.build_submit_batch(requests_to_submit)
        return actions
