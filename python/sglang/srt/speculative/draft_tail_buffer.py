from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from sglang.srt.speculative.decoupled_spec_io import (
    DraftClose,
    DraftControlBatch,
    DraftSync,
    DraftTailStreamOutput,
    VerifyCommit,
    iter_control_batch_messages,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DraftTailSnapshot:
    request_id: str
    committed_len: int
    tail_tokens: list[int]


@dataclass
class RequestDraftTailState:
    """Store verifier-visible rolling draft tail metadata for one request."""

    drafter_rank: int
    # The verifier committed output length tracked by DraftSync/VerifyCommit.
    committed_len: int = 0
    # The only stale base allowed to append when bonus matches preserve a suffix.
    can_accept_prefix_len: int = 0
    tail_tokens: list[int] = field(default_factory=list)

    def consumable_tail_tokens(self) -> list[int]:
        return list(self.tail_tokens[:-1])

    def consumable_tail_len(self) -> int:
        return max(0, len(self.tail_tokens) - 1)


class DraftTailBuffer:
    """Verifier-side rolling draft tail state shared by scheduler and proxy."""

    def __init__(
        self,
        *,
        verifier_rank: int,
        required_tail_len: int,
        enable_debug_prints: bool = False,
    ) -> None:
        self.verifier_rank = int(verifier_rank)
        self.required_tail_len = max(0, int(required_tail_len))
        self.enable_debug_prints = bool(enable_debug_prints)
        self._lock = threading.Lock()
        self._states: dict[str, RequestDraftTailState] = {}

    def close(self) -> None:
        with self._lock:
            self._states.clear()

    def has_request(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._states

    def get_committed_len(self, request_id: str) -> int | None:
        with self._lock:
            state = self._states.get(request_id)
            if state is None:
                return None
            return int(state.committed_len)

    def open_requests(self, messages: list[DraftSync]) -> None:
        if not messages:
            return
        with self._lock:
            for message in messages:
                self._open_request_locked(message)

    def _open_request_locked(self, message: DraftSync) -> None:
        committed_len = len(message.committed_output_ids)
        self._states[message.request_id] = RequestDraftTailState(
            drafter_rank=int(message.dst_drafter_rank),
            committed_len=committed_len,
            can_accept_prefix_len=committed_len,
        )

    def apply_verify_commits(self, messages: list[VerifyCommit]) -> None:
        if not messages:
            return
        with self._lock:
            for message in messages:
                self._apply_commit_locked(message)

    def _apply_commit_locked(self, message: VerifyCommit) -> None:
        state = self._states.get(message.request_id)
        if state is None:
            return

        # old_committed_len is the verifier output length before the forward
        # pass that produced this VerifyCommit. tail_tokens[0] corresponds to
        # absolute output position old_committed_len, tail_tokens[1] to
        # old_committed_len + 1, and so on.
        old_committed_len = int(message.pre_verify_committed_len)

        # bonus_token_pos is an absolute output index in req.output_ids. The new
        # committed length is therefore index + 1.
        new_committed_len = int(message.bonus_token_pos) + 1

        # Number of draft tail tokens accepted before the verifier-generated
        # bonus token. If old_committed_len=3 and bonus_token_pos=5, then
        # positions 3 and 4 came from draft tail, so accepted_tail_len=2.
        # The bonus token itself should match tail_tokens[accepted_tail_len].
        accepted_tail_len = max(0, int(message.bonus_token_pos) - old_committed_len)

        remaining: list[int] = []
        can_accept_prefix_len = new_committed_len

        if state.tail_tokens and not (0 <= accepted_tail_len < len(state.tail_tokens)):

            if self.enable_debug_prints:
                print(
                    "[decoupled_verify][apply_commit_invalid] "
                    f"request_id={message.request_id} "
                    f"drafter_rank={state.drafter_rank} "
                    f"state_committed_len={state.committed_len} "
                    f"state_can_accept_prefix_len={state.can_accept_prefix_len} "
                    f"state_tail_len={len(state.tail_tokens)} "
                    f"state_tail_tokens={list(state.tail_tokens)} "
                    f"message_pre_verify_committed_len={old_committed_len} "
                    f"message_bonus_token_pos={int(message.bonus_token_pos)} "
                    f"message_bonus_token_id={int(message.bonus_token_id)} "
                    f"accepted_tail_len={accepted_tail_len}",
                    flush=True,
                )
                
            raise RuntimeError(
                "Decoupled verify consumed all buffered draft tokens without a "
                "reserved bonus-token anchor: "
                f"request_id={message.request_id} "
                f"pre_verify_committed_len={old_committed_len} "
                f"bonus_token_pos={int(message.bonus_token_pos)} "
                f"accepted_tail_len={accepted_tail_len} "
                f"buffered_tail_len={len(state.tail_tokens)}"
            )

        if 0 <= accepted_tail_len < len(state.tail_tokens):
            # If the verifier's bonus token equals the next buffered draft
            # token, the buffered suffix after that token is still valid under
            # the new committed prefix. Example:
            #   old_committed_len=3, tail=[11,22,33,44], bonus_pos=5,
            #   bonus_id=33 -> accepted_tail_len=2, remaining=[44].
            if int(state.tail_tokens[accepted_tail_len]) == int(message.bonus_token_id):
                remaining = state.tail_tokens[accepted_tail_len + 1 :]
                # Keep the previously accepted stale base. The preserved suffix
                # may itself have come from an older base, and in-flight stream
                # outputs from that base can still append contiguously.
                can_accept_prefix_len = int(state.can_accept_prefix_len)

        state.committed_len = new_committed_len
        state.can_accept_prefix_len = can_accept_prefix_len
        state.tail_tokens = remaining

    def close_requests(self, messages: list[DraftClose]) -> None:
        if not messages:
            return
        with self._lock:
            for message in messages:
                self._close_request_locked(message)

    def _close_request_locked(self, message: DraftClose) -> None:
        self._states.pop(message.request_id, None)

    def apply_control_batch(self, batch: DraftControlBatch) -> None:
        with self._lock:
            for message in iter_control_batch_messages(batch):
                if isinstance(message, DraftSync):
                    self._open_request_locked(message)
                elif isinstance(message, VerifyCommit):
                    self._apply_commit_locked(message)
                elif isinstance(message, DraftClose):
                    self._close_request_locked(message)

    def append_draft_stream(self, outputs: list[DraftTailStreamOutput]) -> None:
        if not outputs:
            return
        with self._lock:
            for output in outputs:
                self._push_one_locked(output)

    def _log_invalid_tail_stream_output(
        self,
        *,
        output: DraftTailStreamOutput,
        reason: str,
        detail: str,
    ) -> None:
        if not (self.enable_debug_prints or logger.isEnabledFor(logging.DEBUG)):
            return
        logger.warning(
            "Rejecting draft tail stream output for request %s: %s (%s) "
            "base_committed_len=%s new_token_pos=%s",
            output.request_id,
            reason,
            detail,
            int(output.base_committed_len),
            int(output.new_token_pos),
        )

    def _push_one_locked(self, output: DraftTailStreamOutput) -> bool:

        if int(output.dst_verifier_rank) != self.verifier_rank:
            return False

        request_id = output.request_id
        src_drafter_rank = int(output.src_drafter_rank)
        # base_committed_len is the committed output length that the drafter
        # used as the prefix when it generated this token.
        base_committed_len = int(output.base_committed_len)
        # token_pos is the absolute output position of this token. It is not an
        # index into tail_tokens until we subtract state.committed_len.
        token_pos = int(output.new_token_pos)
        token_id = int(output.new_token_id)

        state = self._states.get(request_id)
        if state is None:
            self._log_invalid_tail_stream_output(
                output=output,
                reason="unknown_request",
                detail="draft tail state must be created by DraftSync",
            )
            return False

        if src_drafter_rank != int(state.drafter_rank):
            self._log_invalid_tail_stream_output(
                output=output,
                reason="unexpected_drafter_rank",
                detail=f"expected={state.drafter_rank} got={src_drafter_rank}",
            )
            return False

        # tail_tokens[0] corresponds to absolute position state.committed_len.
        # buffer_end_len is the first absolute position not yet present in the
        # buffer, so a normal append must use token_pos == buffer_end_len.
        buffer_end_len = int(state.committed_len) + len(state.tail_tokens)

        if base_committed_len != int(state.committed_len):
            # A stale-base output is allowed only when bonus matching preserved
            # a suffix and recorded that base in can_accept_prefix_len. Even
            # then, the stale stream must append exactly at buffer_end_len.
            if (
                base_committed_len >= int(state.can_accept_prefix_len)
                and token_pos == buffer_end_len
            ):
                state.tail_tokens.append(token_id)
                return True
            self._log_invalid_tail_stream_output(
                output=output,
                reason="stale_base",
                detail=(
                    f"expected={state.committed_len} "
                    f"or stale_accept={state.can_accept_prefix_len} "
                    f"got={base_committed_len}"
                ),
            )
            return False

        if token_pos < buffer_end_len:
            # This output overlaps with tokens already buffered. Convert the
            # absolute output position back to a tail_tokens index.
            token_index = token_pos - int(state.committed_len)
            if token_index >= 0 and int(state.tail_tokens[token_index]) == token_id:
                return True
            self._log_invalid_tail_stream_output(
                output=output,
                reason="overlap_mismatch",
                detail=f"buffer_end_len={buffer_end_len}",
            )
            return False
        if token_pos > buffer_end_len:
            # Gap: the stream skipped one or more positions, so appending would
            # make the tail ambiguous.
            self._log_invalid_tail_stream_output(
                output=output,
                reason="gap_update",
                detail=f"buffer_end_len={buffer_end_len}",
            )
            return False

        state.tail_tokens.append(token_id)
        return True

    def get_draft_snapshots(
        self,
        reqs: list,
        *,
        allow_partial: bool = True,
    ) -> list[DraftTailSnapshot]:
        with self._lock:
            snapshots: list[DraftTailSnapshot] = []
            for req in reqs:
                state = self._states.get(req.rid)
                if state is None:
                    continue
                if (
                    not allow_partial
                    and state.consumable_tail_len() < self.required_tail_len
                ):
                    continue
                snapshots.append(
                    DraftTailSnapshot(
                        request_id=req.rid,
                        committed_len=int(state.committed_len),
                        tail_tokens=state.consumable_tail_tokens(),
                    )
                )
            return snapshots
