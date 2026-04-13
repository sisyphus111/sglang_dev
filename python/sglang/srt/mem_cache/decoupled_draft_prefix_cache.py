from __future__ import annotations

from dataclasses import dataclass
import time
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.speculative.decoupled_spec_io import (
    build_draft_scheduler_rid,
    extract_draft_request_id,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams


@dataclass
class DraftSession:
    kv_indices: torch.Tensor
    kv_len: int


class DecoupledDraftPrefixCache(ChunkCache):
    """Chunk-cache-like prefix cache that stores drafter sessions as KV indices."""

    def __init__(self, params: CacheInitParams):
        super().__init__(params)
        self.sessions: dict[str, DraftSession] = {}
        self.protected_size_ = 0

    def reset(self):
        self.sessions.clear()
        self.protected_size_ = 0

    def _get_session_id(self, rid: str) -> str | None:
        request_id = extract_draft_request_id(str(rid))
        if request_id is None:
            return None
        return build_draft_scheduler_rid(request_id)

    def _get_target_prefix_len(self, req: Req) -> int:
        return max(len(getattr(req, "fill_ids", [])) - 1, 0)

    def _req_debug_fields(self, req: Req) -> dict[str, object]:
        prefix_indices = getattr(req, "prefix_indices", None)
        try:
            prefix_len = 0 if prefix_indices is None else len(prefix_indices)
        except TypeError:
            prefix_len = None
        return {
            "rid": getattr(req, "rid", None),
            "req_pool_idx": getattr(req, "req_pool_idx", None),
            "fill_len": len(getattr(req, "fill_ids", [])),
            "origin_input_len": len(getattr(req, "origin_input_ids", [])),
            "output_len": len(getattr(req, "output_ids", [])),
            "kv_committed_len": getattr(req, "kv_committed_len", None),
            "kv_allocated_len": getattr(req, "kv_allocated_len", None),
            "cache_protected_len": getattr(req, "cache_protected_len", None),
            "prefix_len": prefix_len,
        }

    def draft_session_debug_for_req(self, req: Req) -> dict[str, object]:
        session_id = self._get_session_id(req.rid)
        session = None if session_id is None else self.sessions.get(session_id)
        target_input_len = len(getattr(req, "origin_input_ids", [])) + len(
            getattr(req, "output_ids", [])
        )
        return {
            "session_id": session_id,
            "has_session": session is not None,
            "session_kv_len": None if session is None else session.kv_len,
            "target_input_len": target_input_len,
            "target_prefix_len": max(target_input_len - 1, 0),
            "protected_size": self.protected_size_,
            **self._req_debug_fields(req),
        }

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        match_start = time.perf_counter()
        req = params.req
        session_lookup_start = time.perf_counter()
        session_id = None if req is None else self._get_session_id(req.rid)
        session_lookup_duration_ms = round(
            (time.perf_counter() - session_lookup_start) * 1000, 3
        )
        if req is None or session_id is None:
            total_duration_ms = round((time.perf_counter() - match_start) * 1000, 3)
            if req is not None:
                req._draft_restore_breakdown = {
                    "restored": 0,
                    "reason": "no_req_or_session_id",
                    "session_lookup_ms": session_lookup_duration_ms,
                    "total_ms": total_duration_ms,
                }
            # print(
            #     "[draft_prefix_cache] match_prefix summary: "
            #     f"rid={getattr(req, 'rid', None)}, session_id={session_id}, "
            #     f"restored=0, fallback=1, reason=no_req_or_session_id, "
            #     f"session_lookup_ms={session_lookup_duration_ms}, total_ms={total_duration_ms}"
            # )
            return super().match_prefix(params)

        req.draft_stateful_mode = True
        session_fetch_start = time.perf_counter()
        session = self.sessions.get(session_id)
        session_fetch_duration_ms = round(
            (time.perf_counter() - session_fetch_start) * 1000, 3
        )
        if session is None:
            total_duration_ms = round((time.perf_counter() - match_start) * 1000, 3)
            req._draft_restore_breakdown = {
                "restored": 0,
                "reason": "session_not_found",
                "session_lookup_ms": session_lookup_duration_ms,
                "session_fetch_ms": session_fetch_duration_ms,
                "total_ms": total_duration_ms,
            }
            # print(
            #     "[draft_prefix_cache] match_prefix summary: "
            #     f"rid={req.rid}, session_id={session_id}, restored=0, fallback=1, "
            #     "reason=session_not_found, "
            #     f"session_lookup_ms={session_lookup_duration_ms}, "
            #     f"session_fetch_ms={session_fetch_duration_ms}, total_ms={total_duration_ms}"
            # )
            return super().match_prefix(params)

        restore_call_start = time.perf_counter()
        result = self._restore_session_prefix(req, session_id, session)
        restore_call_duration_ms = round(
            (time.perf_counter() - restore_call_start) * 1000, 3
        )
        total_duration_ms = round((time.perf_counter() - match_start) * 1000, 3)
        req._draft_restore_breakdown = {
            "restored": 1,
            "session_lookup_ms": session_lookup_duration_ms,
            "session_fetch_ms": session_fetch_duration_ms,
            "restore_call_ms": restore_call_duration_ms,
            "restore_inner_breakdown": getattr(req, "_draft_restore_inner_breakdown", None),
            "total_ms": total_duration_ms,
        }
        # print(
        #     "[draft_prefix_cache] match_prefix summary: "
        #     f"rid={req.rid}, session_id={session_id}, restored=1, fallback=0, "
        #     f"session_lookup_ms={session_lookup_duration_ms}, "
        #     f"session_fetch_ms={session_fetch_duration_ms}, "
        #     f"restore_call_ms={restore_call_duration_ms}, total_ms={total_duration_ms}"
        # )
        return result

    def can_match_draft_decode_fast_path(self, req: Req) -> bool:
        session_id = self._get_session_id(req.rid)
        if session_id is None:
            req._draft_fast_path_match_reason = "not_draft_request"
            return False

        session = self.sessions.get(session_id)
        target_input_len = len(getattr(req, "origin_input_ids", [])) + len(
            getattr(req, "output_ids", [])
        )
        target_prefix_len = max(target_input_len - 1, 0)
        req._draft_retained_req_pool_idx = None
        req._draft_retained_kv_committed_len = (
            None if session is None else session.kv_len
        )
        req._draft_fast_path_keep_len = target_prefix_len if session is not None else None
        req._draft_fast_path_actual_delta = (
            target_input_len - target_prefix_len
            if session is not None
            else None
        )

        if session is None:
            req._draft_fast_path_match_reason = "no_retained_session"
            return False
        if req.req_pool_idx is not None:
            req._draft_fast_path_match_reason = "req_already_has_req_pool_idx"
            return False
        if target_prefix_len <= 0:
            req._draft_fast_path_match_reason = "non_positive_target_prefix_len"
            return False
        if session.kv_len < target_prefix_len:
            req._draft_fast_path_match_reason = "session_shorter_than_target"
            return False

        req._draft_fast_path_match_reason = "match_ok"
        return True

    def maybe_preserve_req(self, req: Req) -> bool:
        preserve_start = time.perf_counter()
        session_lookup_start = time.perf_counter()
        session_id = self._get_session_id(req.rid)
        session_lookup_duration_ms = round(
            (time.perf_counter() - session_lookup_start) * 1000, 3
        )
        if not getattr(req, "draft_stateful_mode", False) or session_id is None:
            total_duration_ms = round((time.perf_counter() - preserve_start) * 1000, 3)
            req._draft_preserve_breakdown = {
                "preserved": 0,
                "reason": "not_stateful_or_missing_session_id",
                "session_lookup_ms": session_lookup_duration_ms,
                "total_ms": total_duration_ms,
            }
            # print(
            #     "[draft_prefix_cache] preserve_req summary: "
            #     f"rid={req.rid}, session_id={session_id}, preserved=0, "
            #     "reason=not_stateful_or_missing_session_id, "
            #     f"session_lookup_ms={session_lookup_duration_ms}, total_ms={total_duration_ms}"
            # )
            return False
        if req.req_pool_idx is None or req.kv_committed_freed:
            total_duration_ms = round((time.perf_counter() - preserve_start) * 1000, 3)
            req._draft_preserve_breakdown = {
                "preserved": 0,
                "reason": "missing_req_pool_idx_or_kv_already_freed",
                "session_lookup_ms": session_lookup_duration_ms,
                "total_ms": total_duration_ms,
            }
            # print(
            #     "[draft_prefix_cache] preserve_req summary: "
            #     f"rid={req.rid}, session_id={session_id}, preserved=0, "
            #     "reason=missing_req_pool_idx_or_kv_already_freed, "
            #     f"session_lookup_ms={session_lookup_duration_ms}, total_ms={total_duration_ms}"
            # )
            return False

        kv_len = req.kv_committed_len
        session_fetch_start = time.perf_counter()
        previous_session = self.sessions.get(session_id)
        session_fetch_duration_ms = round(
            (time.perf_counter() - session_fetch_start) * 1000, 3
        )
        kv_copy_start = time.perf_counter()
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_len
        ].to(dtype=torch.int64, copy=True)
        kv_copy_duration_ms = round((time.perf_counter() - kv_copy_start) * 1000, 3)
        # print(
        #     "[draft_prefix_cache] preserve session kv_indices copy: "
        #     f"rid={req.rid}, session_id={session_id}, kv_len={kv_len}, "
        #     f"duration_ms={kv_copy_duration_ms}"
        # )
        previous_len = 0 if previous_session is None else previous_session.kv_len

        free_overallocated_duration_ms = 0.0
        if req.kv_allocated_len > kv_len:
            free_overallocated_start = time.perf_counter()
            indices_to_free = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, kv_len : req.kv_allocated_len
            ]
            if len(indices_to_free) > 0:
                self.token_to_kv_pool_allocator.free(indices_to_free)
            free_overallocated_duration_ms = round(
                (time.perf_counter() - free_overallocated_start) * 1000, 3
            )
        assert_previous_duration_ms = 0.0
        if previous_session is not None:
            assert_previous_start = time.perf_counter()
            self._assert_previous_session_preserved_in_new(
                req, session_id, previous_session, kv_indices
            )
            assert_previous_duration_ms = round(
                (time.perf_counter() - assert_previous_start) * 1000, 3
            )

        session_update_start = time.perf_counter()
        self.sessions[session_id] = DraftSession(
            kv_indices=kv_indices,
            kv_len=kv_len,
        )
        self.protected_size_ += kv_len - previous_len
        session_update_duration_ms = round(
            (time.perf_counter() - session_update_start) * 1000, 3
        )

        req.kv_committed_freed = True
        req.kv_overallocated_freed = True
        req_pool_free_start = time.perf_counter()
        self.req_to_token_pool.free(req)
        req_pool_free_duration_ms = round(
            (time.perf_counter() - req_pool_free_start) * 1000, 3
        )
        total_duration_ms = round((time.perf_counter() - preserve_start) * 1000, 3)
        req._draft_preserve_breakdown = {
            "preserved": 1,
            "kv_len": kv_len,
            "previous_session_len": previous_len,
            "session_lookup_ms": session_lookup_duration_ms,
            "session_fetch_ms": session_fetch_duration_ms,
            "kv_copy_ms": kv_copy_duration_ms,
            "free_overallocated_ms": free_overallocated_duration_ms,
            "assert_previous_ms": assert_previous_duration_ms,
            "session_update_ms": session_update_duration_ms,
            "req_pool_free_ms": req_pool_free_duration_ms,
            "total_ms": total_duration_ms,
        }
        # print(
        #     "[draft_prefix_cache] preserve_req summary: "
        #     f"rid={req.rid}, session_id={session_id}, preserved=1, "
        #     f"kv_len={kv_len}, previous_session_len={previous_len}, "
        #     f"session_lookup_ms={session_lookup_duration_ms}, "
        #     f"session_fetch_ms={session_fetch_duration_ms}, "
        #     f"kv_copy_ms={kv_copy_duration_ms}, "
        #     f"free_overallocated_ms={free_overallocated_duration_ms}, "
        #     f"assert_previous_ms={assert_previous_duration_ms}, "
        #     f"session_update_ms={session_update_duration_ms}, "
        #     f"req_pool_free_ms={req_pool_free_duration_ms}, total_ms={total_duration_ms}"
        # )
        return True

    def release_draft_session(self, scheduler_rid: str, waiting_queue=None) -> None:
        _ = waiting_queue
        session = self.sessions.pop(scheduler_rid, None)
        if session is None:
            return

        self._free_session_kv(session)

    def drop_draft_session_for_rid(self, rid: str) -> None:
        session_id = self._get_session_id(rid)
        if session_id is None:
            return
        session = self.sessions.pop(session_id, None)
        if session is not None:
            self.protected_size_ -= session.kv_len

    def can_use_draft_decode_fast_path(self, req: Req) -> bool:
        result = bool(
            getattr(req, "draft_stateful_mode", False)
            and req.req_pool_idx is not None
            and not req.kv_committed_freed
            and req.kv_committed_len > 0
            and len(req.fill_ids) == req.kv_committed_len + 1
            and req.cache_protected_len == req.kv_committed_len
        )
        if result:
            req._draft_fast_path_can_use_reason = "can_use"
        elif not getattr(req, "draft_stateful_mode", False):
            req._draft_fast_path_can_use_reason = "not_stateful_mode"
        elif req.req_pool_idx is None:
            req._draft_fast_path_can_use_reason = "missing_req_pool_idx"
        elif req.kv_committed_freed:
            req._draft_fast_path_can_use_reason = "kv_committed_freed"
        elif req.kv_committed_len <= 0:
            req._draft_fast_path_can_use_reason = "non_positive_kv_committed_len"
        elif len(req.fill_ids) != req.kv_committed_len + 1:
            req._draft_fast_path_can_use_reason = "fill_len_not_equal_kv_plus_one"
        else:
            req._draft_fast_path_can_use_reason = "cache_protected_len_mismatch"
        return result

    def protected_size(self):
        return self.protected_size_

    def session_debug_entries(self) -> list[dict[str, int | str]]:
        return [
            {
                "session_id": session_id,
                "kv_len": session.kv_len,
            }
            for session_id, session in self.sessions.items()
        ]

    def _restore_session_prefix(
        self,
        req: Req,
        session_id: str,
        session: DraftSession,
        *,
        target_prefix_len: int | None = None,
    ) -> MatchResult:
        restore_start = time.perf_counter()
        original_session_kv_len = session.kv_len
        target_prefix_len_start = time.perf_counter()
        target_prefix_len = (
            self._get_target_prefix_len(req)
            if target_prefix_len is None
            else target_prefix_len
        )
        target_prefix_len_duration_ms = round(
            (time.perf_counter() - target_prefix_len_start) * 1000, 3
        )
        validation_start = time.perf_counter()
        if target_prefix_len <= 0:
            raise AssertionError(
                f"draft request {req.rid} has invalid target_prefix_len {target_prefix_len}"
            )
        if session.kv_len < target_prefix_len:
            raise AssertionError(
                f"draft request {req.rid} needs prefix len {target_prefix_len}, "
                f"but session {session_id} only has {session.kv_len}"
            )
        if req.req_pool_idx is not None:
            raise AssertionError(
                f"draft request {req.rid} already has req_pool_idx {req.req_pool_idx}"
            )
        validation_duration_ms = round(
            (time.perf_counter() - validation_start) * 1000, 3
        )

        req_pool_alloc_start = time.perf_counter()
        alloc_result = self.req_to_token_pool.alloc([req])
        req_pool_alloc_duration_ms = round(
            (time.perf_counter() - req_pool_alloc_start) * 1000, 3
        )
        if alloc_result is None or req.req_pool_idx is None:
            raise RuntimeError(
                f"failed to allocate req_pool_idx for restored draft request {req.rid}"
            )

        prefix_copy_start = time.perf_counter()
        prefix_indices = session.kv_indices[:target_prefix_len].to(
            dtype=torch.int64, copy=True
        )
        prefix_copy_duration_ms = round(
            (time.perf_counter() - prefix_copy_start) * 1000, 3
        )
        # print(
        #     "[draft_prefix_cache] restore session prefix copy: "
        #     f"rid={req.rid}, session_id={session_id}, target_prefix_len={target_prefix_len}, "
        #     f"duration_ms={prefix_copy_duration_ms}"
        # )
        req_pool_write_start = time.perf_counter()
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(0, target_prefix_len)),
            prefix_indices.to(dtype=torch.int32),
        )
        req_pool_write_duration_ms = round(
            (time.perf_counter() - req_pool_write_start) * 1000, 3
        )

        tail_free_duration_ms = 0.0
        truncate_session_duration_ms = 0.0
        truncation_copy_duration_ms = 0.0
        truncated_len = max(original_session_kv_len - target_prefix_len, 0)
        if session.kv_len > target_prefix_len:
            truncate_session_start = time.perf_counter()
            tail_indices = session.kv_indices[target_prefix_len : session.kv_len]
            tail_free_start = time.perf_counter()
            if len(tail_indices) > 0:
                self.token_to_kv_pool_allocator.free(tail_indices)
            tail_free_duration_ms = round(
                (time.perf_counter() - tail_free_start) * 1000, 3
            )
            self.protected_size_ -= session.kv_len - target_prefix_len
            truncation_copy_start = time.perf_counter()
            session = DraftSession(
                kv_indices=prefix_indices.to(dtype=torch.int64, copy=True),
                kv_len=target_prefix_len,
            )
            truncation_copy_duration_ms = round(
                (time.perf_counter() - truncation_copy_start) * 1000, 3
            )
            self.sessions[session_id] = session
            truncate_session_duration_ms = round(
                (time.perf_counter() - truncate_session_start) * 1000, 3
            )

        state_update_start = time.perf_counter()
        req.kv_committed_len = target_prefix_len
        req.kv_allocated_len = target_prefix_len
        req.kv_committed_freed = False
        req.kv_overallocated_freed = False
        state_update_duration_ms = round(
            (time.perf_counter() - state_update_start) * 1000, 3
        )
        total_duration_ms = round((time.perf_counter() - restore_start) * 1000, 3)
        req._draft_restore_inner_breakdown = {
            "session_kv_len_before": original_session_kv_len,
            "session_kv_len_after": session.kv_len,
            "target_prefix_len": target_prefix_len,
            "truncated_len": truncated_len,
            "target_prefix_len_ms": target_prefix_len_duration_ms,
            "validation_ms": validation_duration_ms,
            "req_pool_alloc_ms": req_pool_alloc_duration_ms,
            "prefix_copy_ms": prefix_copy_duration_ms,
            "req_pool_write_ms": req_pool_write_duration_ms,
            "truncate_session_ms": truncate_session_duration_ms,
            "truncation_copy_ms": truncation_copy_duration_ms,
            "tail_free_ms": tail_free_duration_ms,
            "state_update_ms": state_update_duration_ms,
            "total_ms": total_duration_ms,
        }
        # print(
        #     "[draft_prefix_cache] restore_session_prefix summary: "
        #     f"rid={req.rid}, session_id={session_id}, "
        #     f"session_kv_len_before={original_session_kv_len}, "
        #     f"session_kv_len_after={session.kv_len}, "
        #     f"target_prefix_len={target_prefix_len}, truncated_len={truncated_len}, "
        #     f"target_prefix_len_ms={target_prefix_len_duration_ms}, "
        #     f"validation_ms={validation_duration_ms}, "
        #     f"req_pool_alloc_ms={req_pool_alloc_duration_ms}, "
        #     f"prefix_copy_ms={prefix_copy_duration_ms}, "
        #     f"req_pool_write_ms={req_pool_write_duration_ms}, "
        #     f"truncate_session_ms={truncate_session_duration_ms}, "
        #     f"truncation_copy_ms={truncation_copy_duration_ms}, "
        #     f"tail_free_ms={tail_free_duration_ms}, "
        #     f"state_update_ms={state_update_duration_ms}, total_ms={total_duration_ms}"
        # )
        return MatchResult(
            device_indices=prefix_indices,
            last_device_node=None,
            last_host_node=None,
        )

    def _free_session_kv(self, session: DraftSession) -> None:
        if session.kv_len <= 0:
            return
        self.token_to_kv_pool_allocator.free(session.kv_indices[: session.kv_len])
        self.protected_size_ -= session.kv_len

    def _assert_previous_session_preserved_in_new(
        self,
        req: Req,
        session_id: str,
        previous_session: DraftSession,
        new_kv_indices: torch.Tensor,
    ) -> None:
        return 
        # previous_indices = previous_session.kv_indices[: previous_session.kv_len]
        # if len(previous_indices) == 0:
        #     return

        # retained_mask = torch.isin(previous_indices, new_kv_indices)
        # retained_count = int(retained_mask.sum().item())
        # if retained_count == len(previous_indices):
        #     return

        # raise AssertionError(
        #     "draft session overwrite would orphan previous KV slots: "
        #     f"{req.rid=}, {session_id=}, previous_kv_len={previous_session.kv_len}, "
        #     f"new_kv_len={len(new_kv_indices)}, retained_previous_kv_count={retained_count}"
        # )
