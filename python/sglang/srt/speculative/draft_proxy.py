from __future__ import annotations

import faulthandler
import logging
import signal
from dataclasses import dataclass, field
from typing import Any, Optional

import psutil
import setproctitle
import zmq

from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.decoupled_spec_io import (
    DraftBackendIpcConfig,
    DraftBackendMessage,
    DraftBackendMessageType,
    DraftLookupKey,
    DraftRequest,
    DraftResult,
    DraftRoute,
    PollDraftResultsResponse,
    RequestTerminateMessage,
)
from sglang.srt.utils import configure_logger, get_zmq_socket, kill_itself_when_parent_died
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


def _get_ray() -> Any:
    import ray

    return ray


def _maybe_init_ray() -> None:
    ray = _get_ray()
    if ray.is_initialized():
        return
    ray.init(
        address="auto",
        ignore_reinit_error=True,
        log_to_driver=False,
        logging_level=logging.ERROR,
    )


def _resolve_actor_handles(
    actor_names: list[str],
    namespace: str | None = None,
) -> list[Any]:
    ray = _get_ray()
    return [ray.get_actor(actor_name, namespace=namespace) for actor_name in actor_names]


@dataclass
class InflightDraft:
    draft_index: int
    object_ref: Any


@dataclass
class DraftProxy:
    draft_actor_handles: list[Any] = field(default_factory=list)
    request_routes: dict[str, DraftRoute] = field(default_factory=dict)
    inflight_requests: dict[DraftLookupKey, InflightDraft] = field(default_factory=dict)
    inflight_per_index: list[int] = field(default_factory=list)
    ready_results: dict[DraftLookupKey, DraftResult] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.register_draft_handles(self.draft_actor_handles)

    def register_draft_handles(self, handles: list[Any]) -> None:
        self.draft_actor_handles = list(handles)
        self.inflight_per_index = [0] * len(self.draft_actor_handles)

    def acquire_route(self, request_id: str) -> DraftRoute:
        route = self.request_routes.get(request_id)
        if route is not None:
            return route
        if not self.draft_actor_handles:
            raise ValueError("DraftProxy has no registered draft actor handles")

        best_idx = min(
            range(len(self.draft_actor_handles)),
            key=lambda i: (self.inflight_per_index[i], i),
        )
        route = DraftRoute(request_id=request_id, draft_index=best_idx)
        self.request_routes[request_id] = route
        return route

    def _release_inflight(self, key: DraftLookupKey) -> None:
        inflight = self.inflight_requests.pop(key, None)
        if inflight is None:
            return
        draft_index = int(inflight.draft_index)
        if 0 <= draft_index < len(self.inflight_per_index):
            self.inflight_per_index[draft_index] = max(
                0, self.inflight_per_index[draft_index] - 1
            )

    def submit_request(self, request: DraftRequest, object_ref: Any) -> DraftRoute:
        route = self.acquire_route(request.request_id)
        self.inflight_requests[request.key] = InflightDraft(
            draft_index=route.draft_index,
            object_ref=object_ref,
        )
        self.inflight_per_index[route.draft_index] += 1
        return route

    def peek_ready_results(
        self,
        keys: list[DraftLookupKey],
    ) -> tuple[list[DraftResult], list[DraftLookupKey]]:
        ready_results: list[DraftResult] = []
        missing_keys: list[DraftLookupKey] = []
        for key in keys:
            result = self.ready_results.get(key)
            if result is None:
                missing_keys.append(key)
            else:
                ready_results.append(result)
        return ready_results, missing_keys

    def pop_ready_results(self, keys: list[DraftLookupKey]) -> list[DraftResult]:
        popped_results: list[DraftResult] = []
        for key in keys:
            result = self.ready_results.pop(key, None)
            if result is not None:
                popped_results.append(result)
        return popped_results

    def release_request(self, request_id: str) -> None:
        self.request_routes.pop(request_id, None)
        for key in list(self.ready_results):
            if key.request_id == request_id:
                self.ready_results.pop(key, None)

    def terminate_request(self, message: RequestTerminateMessage) -> None:
        upper_bound = message.draft_round_id_upper_bound
        for key in list(self.inflight_requests):
            if key.request_id != message.request_id:
                continue
            if upper_bound is not None and key.draft_round_id > upper_bound:
                continue
            self._release_inflight(key)

        for key in list(self.ready_results):
            if key.request_id != message.request_id:
                continue
            if upper_bound is not None and key.draft_round_id > upper_bound:
                continue
            self.ready_results.pop(key, None)

        has_newer_inflight = (
            any(
                key.request_id == message.request_id
                and key.draft_round_id > upper_bound
                for key in self.inflight_requests
            )
            if upper_bound is not None
            else False
        )
        has_newer_ready = (
            any(
                key.request_id == message.request_id
                and key.draft_round_id > upper_bound
                for key in self.ready_results
            )
            if upper_bound is not None
            else False
        )
        if upper_bound is None or (not has_newer_inflight and not has_newer_ready):
            self.release_request(message.request_id)

    def complete_request(
        self,
        key: DraftLookupKey,
        result: DraftResult,
    ) -> Optional[DraftResult]:
        self._release_inflight(key)
        self.ready_results[key] = result
        return result


class DraftBackendManager:
    def __init__(
        self,
        *,
        draft_actor_handles: list[Any],
        ipc_config: DraftBackendIpcConfig,
    ) -> None:
        self.proxy = DraftProxy(
            draft_actor_handles=draft_actor_handles,
        )
        self.ipc_config = ipc_config
        self.pending_poll_keys: dict[int, list[DraftLookupKey]] = {}
        self.init_ipc_channels()

    def init_ipc_channels(self) -> None:
        self.context = zmq.Context(1 + 2 * len(self.ipc_config.dp_ipc_endpoints))
        self.recv_from_scheduler: dict[int, zmq.Socket] = {}
        self.send_to_scheduler: dict[int, zmq.Socket] = {}
        self.poller = zmq.Poller()

        for dp_rank, endpoints in sorted(self.ipc_config.dp_ipc_endpoints.items()):
            recv_socket = get_zmq_socket(
                self.context,
                zmq.PULL,
                endpoints.scheduler_to_backend_ipc_name,
                True,
            )
            send_socket = get_zmq_socket(
                self.context,
                zmq.PUSH,
                endpoints.backend_to_scheduler_ipc_name,
                True,
            )
            self.recv_from_scheduler[dp_rank] = recv_socket
            self.send_to_scheduler[dp_rank] = send_socket
            self.poller.register(recv_socket, zmq.POLLIN)

    def close(self) -> None:
        for socket_map_name in ("recv_from_scheduler", "send_to_scheduler"):
            socket_map = getattr(self, socket_map_name, None) or {}
            for socket in socket_map.values():
                socket.close(linger=0)
        if getattr(self, "context", None) is not None:
            self.context.term()

    def _submit_draft_requests(self, requests: list[DraftRequest]) -> None:
        for request in requests:
            route = self.proxy.acquire_route(request.request_id)
            actor_handle = self.proxy.draft_actor_handles[route.draft_index]
            object_ref = actor_handle.handle_draft_request.remote(request)
            self.proxy.submit_request(request, object_ref)

    def _collect_completed_drafts(self) -> None:
        inflight_items = list(self.proxy.inflight_requests.items())
        if not inflight_items:
            return

        ray = _get_ray()
        object_refs = [inflight.object_ref for _, inflight in inflight_items]
        ready_refs, _ = ray.wait(
            object_refs,
            num_returns=len(object_refs),
            timeout=0,
        )
        if not ready_refs:
            return

        ref_to_key = {inflight.object_ref: key for key, inflight in inflight_items}
        for object_ref in ready_refs:
            key = ref_to_key.get(object_ref)
            if key is None:
                continue

            result = ray.get(object_ref)
            if not isinstance(result, DraftResult):
                raise RuntimeError(
                    f"Draft backend received unexpected result type: {type(result)!r}"
                )
            if result.request_id != key.request_id:
                raise RuntimeError(
                    "Draft backend result request_id mismatch: "
                    f"{result.request_id} != {key.request_id}"
                )
            if result.draft_round_id != key.draft_round_id:
                raise RuntimeError(
                    "Draft backend result round mismatch: "
                    f"{result.draft_round_id} != {key.draft_round_id}"
                )

            self.proxy.complete_request(key, result)

    def _send_poll_response(
        self,
        target_dp_rank: int,
        keys: list[DraftLookupKey],
    ) -> None:
        ready_results, _ = self.proxy.peek_ready_results(keys)
        keys_to_pop = [result.key for result in ready_results]
        popped_results = self.proxy.pop_ready_results(keys_to_pop)
        response = PollDraftResultsResponse(results=popped_results)
        target_socket = self.send_to_scheduler.get(target_dp_rank)
        if target_socket is None:
            raise RuntimeError(
                f"Missing draft backend response socket for dp_rank={target_dp_rank}"
            )
        target_socket.send_pyobj(DraftBackendMessage.from_poll_response(response))

    def _flush_pending_polls(self) -> None:
        next_pending_poll_keys: dict[int, list[DraftLookupKey]] = {}
        for target_dp_rank, keys in self.pending_poll_keys.items():
            _, missing_keys = self.proxy.peek_ready_results(keys)
            if not missing_keys:
                self._send_poll_response(target_dp_rank, keys)
                continue
            next_pending_poll_keys[target_dp_rank] = keys
        self.pending_poll_keys = next_pending_poll_keys

    def _handle_poll_request(
        self,
        poll_request,
        source_dp_rank: int,
    ) -> None:
        _, missing_keys = self.proxy.peek_ready_results(poll_request.keys)
        if not missing_keys:
            self._send_poll_response(source_dp_rank, poll_request.keys)
            return
        self.pending_poll_keys[source_dp_rank] = list(poll_request.keys)

    def _handle_proxy_message(
        self,
        message: DraftBackendMessage,
        source_dp_rank: int,
    ) -> None:
        if (
            message.message_type == DraftBackendMessageType.SUBMIT_DRAFT
            and message.requests is not None
        ):
            self._submit_draft_requests(message.requests)
            return
        if (
            message.message_type == DraftBackendMessageType.POLL_DRAFT_RESULTS
            and message.poll_request is not None
        ):
            self._handle_poll_request(message.poll_request, source_dp_rank)
            return
        if (
            message.message_type == DraftBackendMessageType.REQUEST_TERMINATE
            and message.terminate is not None
        ):
            route = self.proxy.request_routes.get(message.terminate.request_id)
            if route is not None:
                actor_handle = self.proxy.draft_actor_handles[route.draft_index]
                actor_handle.terminate_draft_request.remote(message.terminate.request_id)
            self.proxy.terminate_request(message.terminate)
            return
        raise RuntimeError(
            f"Unsupported draft backend message type: {message.message_type}"
        )

    def event_loop(self) -> None:
        try:
            while True:
                self._collect_completed_drafts()
                self._flush_pending_polls()

                events = dict(self.poller.poll(timeout=1))
                for dp_rank, recv_socket in self.recv_from_scheduler.items():
                    if events.get(recv_socket) != zmq.POLLIN:
                        continue
                    message = recv_socket.recv_pyobj()
                    self._handle_proxy_message(message, dp_rank)

                self._collect_completed_drafts()
                self._flush_pending_polls()
        finally:
            self.close()


def run_draft_backend_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer=None,
    *,
    draft_actor_names: list[str] | None = None,
    draft_actor_namespace: str | None = None,
    draft_backend_manager_class=DraftBackendManager,
) -> None:
    _ = port_args
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::draft_backend")
    faulthandler.enable()
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        _maybe_init_ray()
        ipc_config = port_args.draft_backend_ipc_config
        if ipc_config is None:
            raise ValueError("Draft backend IPC config is not configured")
        if not draft_actor_names:
            raise ValueError(
                "decoupled_verify requires non-empty draft_actor_names "
                "for the default DraftProxy backend"
            )
        draft_actor_handles = _resolve_actor_handles(
            draft_actor_names,
            namespace=draft_actor_namespace,
        )
        manager = draft_backend_manager_class(
            draft_actor_handles=draft_actor_handles,
            ipc_config=ipc_config,
        )
        if pipe_writer is not None:
            pipe_writer.send({"status": "ready"})
        manager.event_loop()
    except Exception:
        if pipe_writer is not None:
            try:
                pipe_writer.send({"status": "failed"})
            except Exception:
                pass
        traceback = get_exception_traceback()
        logger.error("Draft backend hit an exception: %s", traceback)
        parent_process.send_signal(signal.SIGQUIT)
