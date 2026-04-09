from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Deque, Optional

from sglang.srt.speculative.decoupled_spec_io import DraftRequest, DraftResult


@dataclass
class QueuedDraftWorkItem:
    draft_request: DraftRequest
    result_future: asyncio.Future[DraftResult]


@dataclass
class DraftServerSession:
    request_id: str
    scheduler_session_id: str
    scheduler_rid: str
    queue: Deque[QueuedDraftWorkItem] = field(default_factory=deque)
    running: bool = False
    terminated: bool = False
    release_sent: bool = False
    last_enqueued_round_id: int = -1
    last_completed_round_id: int = -1
    terminate_upper_bound: Optional[int] = None


class DraftSessionManager:
    def __init__(
        self,
        *,
        execute_round: Callable[[DraftServerSession, DraftRequest], Awaitable[DraftResult]],
        release_session: Callable[[DraftServerSession], Awaitable[None]],
    ):
        self._execute_round = execute_round
        self._release_session = release_session
        self._sessions: dict[str, DraftServerSession] = {}

    def get(self, request_id: str) -> Optional[DraftServerSession]:
        return self._sessions.get(request_id)

    def get_or_create(
        self,
        *,
        request_id: str,
        scheduler_session_id: str,
        scheduler_rid: str,
    ) -> DraftServerSession:
        session = self._sessions.get(request_id)
        if session is None:
            session = DraftServerSession(
                request_id=request_id,
                scheduler_session_id=scheduler_session_id,
                scheduler_rid=scheduler_rid,
            )
            self._sessions[request_id] = session
        return session

    async def submit(
        self,
        *,
        request_id: str,
        scheduler_session_id: str,
        scheduler_rid: str,
        draft_request: DraftRequest,
    ) -> DraftResult:
        session = self.get_or_create(
            request_id=request_id,
            scheduler_session_id=scheduler_session_id,
            scheduler_rid=scheduler_rid,
        )
        if session.terminated:
            raise RuntimeError(f"Draft session {request_id} has already terminated")
        if (
            session.terminate_upper_bound is not None
            and draft_request.draft_round_id <= session.terminate_upper_bound
        ):
            raise RuntimeError(
                "Draft session already terminated for round_id <= "
                f"{session.terminate_upper_bound}"
            )
        if draft_request.draft_round_id <= session.last_enqueued_round_id:
            raise ValueError(
                "Draft rounds must be strictly increasing per request: "
                f"{draft_request.draft_round_id} <= {session.last_enqueued_round_id}"
            )

        loop = asyncio.get_running_loop()
        result_future: asyncio.Future[DraftResult] = loop.create_future()
        session.last_enqueued_round_id = draft_request.draft_round_id
        session.queue.append(
            QueuedDraftWorkItem(
                draft_request=draft_request,
                result_future=result_future,
            )
        )
        if not session.running:
            session.running = True
            loop.create_task(self._run_session(session))
        return await result_future

    async def terminate(
        self,
        request_id: str,
        draft_round_id_upper_bound: Optional[int] = None,
    ) -> None:
        session = self._sessions.get(request_id)
        if session is None:
            return

        if draft_round_id_upper_bound is None:
            session.terminated = True
        else:
            current_upper_bound = session.terminate_upper_bound
            session.terminate_upper_bound = max(
                draft_round_id_upper_bound,
                current_upper_bound if current_upper_bound is not None else -1,
            )

        retained_queue: Deque[QueuedDraftWorkItem] = deque()
        while session.queue:
            work_item = session.queue.popleft()
            terminate_item = session.terminated or (
                session.terminate_upper_bound is not None
                and work_item.draft_request.draft_round_id <= session.terminate_upper_bound
            )
            if terminate_item:
                if not work_item.result_future.done():
                    work_item.result_future.set_exception(
                        RuntimeError(
                            f"Draft session {request_id} terminated before execution"
                        )
                    )
                continue
            retained_queue.append(work_item)
        session.queue = retained_queue

        if session.terminated and not session.running:
            await self._finalize_session(session)

    async def release(self, request_id: str) -> None:
        session = self._sessions.get(request_id)
        if session is None:
            return
        session.terminated = True
        while session.queue:
            work_item = session.queue.popleft()
            if not work_item.result_future.done():
                work_item.result_future.set_exception(
                    RuntimeError(f"Draft session {request_id} released before execution")
                )
        if not session.running:
            await self._finalize_session(session)

    async def _run_session(self, session: DraftServerSession) -> None:
        try:
            while session.queue:
                work_item = session.queue.popleft()
                if session.terminated:
                    if not work_item.result_future.done():
                        work_item.result_future.set_exception(
                            RuntimeError(
                                f"Draft session {session.request_id} terminated before execution"
                            )
                        )
                    continue
                if (
                    session.terminate_upper_bound is not None
                    and work_item.draft_request.draft_round_id
                    <= session.terminate_upper_bound
                ):
                    if not work_item.result_future.done():
                        work_item.result_future.set_exception(
                            RuntimeError(
                                f"Draft session {session.request_id} terminated for round "
                                f"{work_item.draft_request.draft_round_id}"
                            )
                        )
                    continue

                try:
                    result = await self._execute_round(session, work_item.draft_request)
                except Exception as exc:
                    session.terminated = True
                    if not work_item.result_future.done():
                        work_item.result_future.set_exception(exc)
                else:
                    if result.request_id != work_item.draft_request.request_id:
                        raise RuntimeError(
                            "Draft result request_id mismatch: "
                            f"{result.request_id} != {work_item.draft_request.request_id}"
                        )
                    if result.draft_round_id != work_item.draft_request.draft_round_id:
                        raise RuntimeError(
                            "Draft result round mismatch: "
                            f"{result.draft_round_id} != {work_item.draft_request.draft_round_id}"
                        )
                    session.last_completed_round_id = result.draft_round_id
                    if not work_item.result_future.done():
                        work_item.result_future.set_result(result)
        finally:
            session.running = False
            if session.terminated:
                await self._finalize_session(session)

    async def _finalize_session(self, session: DraftServerSession) -> None:
        if session.release_sent:
            self._sessions.pop(session.request_id, None)
            return
        session.release_sent = True
        try:
            await self._release_session(session)
        finally:
            self._sessions.pop(session.request_id, None)
