from __future__ import annotations

import faulthandler
import logging
import signal
import time
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
from sglang.srt.utils.csv_debug_utils import (
    build_batch_trace_fields,
    build_process_trace_fields,
    emit_csv_event,
    emit_summary,
)
from sglang.srt.utils import configure_logger, get_zmq_socket, kill_itself_when_parent_died
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


def _summarize_lookup_keys(
    keys: list[DraftLookupKey],
    *,
    max_count: int = 16,
) -> list[str]:
    summaries = [
        f"{key.request_id}:{key.draft_round_id}"
        for key in keys[:max_count]
    ]
    if len(keys) > max_count:
        summaries.append(f"...(+{len(keys) - max_count})")
    return summaries


def _summarize_requests(
    requests: list[DraftRequest],
    *,
    max_count: int = 16,
) -> list[dict[str, int | str]]:
    items = []
    for request in requests[:max_count]:
        items.append(
            {
                "request_id": request.request_id,
                "round_id": int(request.draft_round_id),
                "rid": request.rid,
                "prompt_len": len(request.prompt_token_ids),
                "committed_len": len(request.committed_token_ids),
                "scheduler_dp_rank": int(request.scheduler_dp_rank),
            }
        )
    if len(requests) > max_count:
        items.append({"request_id": f"...(+{len(requests) - max_count})"})
    return items


def _summarize_results(
    results: list[DraftResult],
    *,
    max_count: int = 16,
) -> list[dict[str, int | str]]:
    items = []
    for result in results[:max_count]:
        items.append(
            {
                "request_id": result.request_id,
                "round_id": int(result.draft_round_id),
                "rid": result.rid,
                "draft_len": len(result.draft_token_ids),
            }
        )
    if len(results) > max_count:
        items.append({"request_id": f"...(+{len(results) - max_count})"})
    return items


def _emit_draftproxy_batch_event(
    event: str,
    *,
    verify_replica_rank: int | None = None,
    batch_size: int | None = None,
    live_req_count: int | None = None,
    status: str | None = None,
    message: str | None = None,
    **extra,
) -> None:
    emit_csv_event(
        "draftproxy",
        event,
        server_role="draftproxy",
        verify_replica_rank=verify_replica_rank,
        status=status,
        message=message,
        batch_size=batch_size,
        live_req_count=live_req_count,
        **build_batch_trace_fields(trace_key=None),
        **extra,
    )


def _get_ray() -> Any:
    import ray

    return ray


def _maybe_init_ray(ray_init_kwargs: Optional[dict[str, Any]] = None) -> None:
    ray = _get_ray()
    if ray.is_initialized():
        return
    init_kwargs = dict(ray_init_kwargs or {})
    init_kwargs.setdefault("address", "auto")
    init_kwargs.setdefault("ignore_reinit_error", True)
    init_kwargs.setdefault("log_to_driver", False)
    init_kwargs.setdefault("logging_level", logging.ERROR)
    ray.init(**init_kwargs)


def _resolve_actor_handles(
    actor_names: list[str],
    namespace: str | None = None,
    *,
    timeout_s: float = 10.0,
    poll_interval_s: float = 0.5,
) -> list[Any]:
    ray = _get_ray()
    deadline = time.monotonic() + timeout_s
    last_error = None

    while True:
        try:
            return [
                ray.get_actor(actor_name, namespace=namespace)
                for actor_name in actor_names
            ]
        except ValueError as exc:
            last_error = exc
            if time.monotonic() >= deadline:
                raise
            time.sleep(poll_interval_s)

    raise last_error


@dataclass
class InflightDraft:
    draft_index: int
    object_ref: Any
    submit_ts: float
    rid: str


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

    def _release_inflight(self, key: DraftLookupKey) -> InflightDraft | None:
        inflight = self.inflight_requests.pop(key, None)
        if inflight is None:
            return None
        draft_index = int(inflight.draft_index)
        if 0 <= draft_index < len(self.inflight_per_index):
            self.inflight_per_index[draft_index] = max(
                0, self.inflight_per_index[draft_index] - 1
            )
        return inflight

    def plan_routes(
        self, requests: list[DraftRequest]
    ) -> list[tuple[DraftRequest, DraftRoute]]:
        if not requests:
            return []
        if not self.draft_actor_handles:
            raise ValueError("DraftProxy has no registered draft actor handles")

        planned_load = list(self.inflight_per_index)
        routed_requests: list[tuple[DraftRequest, DraftRoute]] = []
        for request in requests:
            route = self.request_routes.get(request.request_id)
            if route is None:
                best_idx = min(
                    range(len(self.draft_actor_handles)),
                    key=lambda i: (planned_load[i], i),
                )
                route = DraftRoute(request_id=request.request_id, draft_index=best_idx)
                self.request_routes[request.request_id] = route
            planned_load[route.draft_index] += 1
            routed_requests.append((request, route))
        return routed_requests

    def dispatch_batch(
        self,
        route: DraftRoute,
        requests: list[DraftRequest],
    ) -> None:
        if not requests:
            return
        actor_handle = self.draft_actor_handles[route.draft_index]
        submit_ts = time.perf_counter()
        object_ref = actor_handle.handle_draft_requests.remote(requests)
        for request in requests:
            self.inflight_requests[request.key] = InflightDraft(
                draft_index=route.draft_index,
                object_ref=object_ref,
                submit_ts=submit_ts,
                rid=request.rid,
            )
            self.inflight_per_index[route.draft_index] += 1

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
        route = self.request_routes.get(message.request_id)
        if route is not None:
            actor_handle = self.draft_actor_handles[route.draft_index]
            actor_handle.terminate_draft_request.remote(message.request_id)

        for key in list(self.inflight_requests):
            if key.request_id == message.request_id:
                self._release_inflight(key)

        for key in list(self.ready_results):
            if key.request_id == message.request_id:
                self.ready_results.pop(key, None)

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
        if requests:
            _emit_draftproxy_batch_event(
                "draftproxy_submit_batch_received",
                verify_replica_rank=requests[0].scheduler_dp_rank,
                batch_size=len(requests),
                live_req_count=len(requests),
                status="received",
                message="draftproxy received a submit batch from verify scheduler",
                inflight_total=len(self.proxy.inflight_requests),
                ready_total=len(self.proxy.ready_results),
                pending_poll_ranks=sorted(self.pending_poll_keys),
                requests=_summarize_requests(requests),
            )
        routed_requests = self.proxy.plan_routes(requests)
        requests_by_index: dict[int, list[DraftRequest]] = {}
        routes_by_index: dict[int, DraftRoute] = {}
        for request, route in routed_requests:
            requests_by_index.setdefault(route.draft_index, []).append(request)
            routes_by_index[route.draft_index] = route
        for draft_index, grouped_requests in sorted(requests_by_index.items()):
            self.proxy.dispatch_batch(routes_by_index[draft_index], grouped_requests)
        if requests:
            _emit_draftproxy_batch_event(
                "draftproxy_submit_batch_finished",
                verify_replica_rank=requests[0].scheduler_dp_rank,
                batch_size=len(requests),
                live_req_count=len(self.proxy.inflight_requests),
                status="submitted",
                message="draftproxy finished routing a submit batch",
                inflight_total=len(self.proxy.inflight_requests),
                ready_total=len(self.proxy.ready_results),
                inflight_per_actor=list(self.proxy.inflight_per_index),
                requests=_summarize_requests(requests),
            )
            emit_summary(
                logger,
                key=f"draftproxy.submit.{requests[0].scheduler_dp_rank}",
                component="draftproxy",
                event="draftproxy_submit_summary",
                message="draftproxy submitted batch",
                server_role="draftproxy",
                verify_replica_rank=requests[0].scheduler_dp_rank,
                batch_size=len(requests),
            )

    def _collect_completed_drafts(self) -> None:
        inflight_items = list(self.proxy.inflight_requests.items())
        if not inflight_items:
            return

        ray = _get_ray()
        ref_to_entries: dict[Any, list[tuple[DraftLookupKey, InflightDraft]]] = {}
        for key, inflight in inflight_items:
            ref_to_entries.setdefault(inflight.object_ref, []).append((key, inflight))
        object_refs = list(ref_to_entries)
        _emit_draftproxy_batch_event(
            "draftproxy_collect_completed_started",
            batch_size=len(object_refs),
            live_req_count=len(inflight_items),
            status="polling",
            message="draftproxy started polling inflight draft object refs",
            inflight_total=len(self.proxy.inflight_requests),
            inflight_per_actor=list(self.proxy.inflight_per_index),
            inflight_keys=_summarize_lookup_keys([key for key, _ in inflight_items]),
        )
        ready_refs, _ = ray.wait(
            object_refs,
            num_returns=len(object_refs),
            timeout=0,
        )
        if not ready_refs:
            return

        _emit_draftproxy_batch_event(
            "draftproxy_collect_completed_ready",
            batch_size=len(ready_refs),
            live_req_count=sum(len(ref_to_entries[ref]) for ref in ready_refs),
            status="ready",
            message="draftproxy found completed draft object refs",
            inflight_total=len(self.proxy.inflight_requests),
            inflight_per_actor=list(self.proxy.inflight_per_index),
        )

        completed_results: list[DraftResult] = []
        for object_ref in ready_refs:
            entries = ref_to_entries.get(object_ref, [])
            if not entries:
                continue

            raw_results = ray.get(object_ref)
            if not isinstance(raw_results, list):
                raise RuntimeError(
                    "Draft backend received unexpected batch result type: "
                    f"{type(raw_results)!r}"
                )
            results = [result for result in raw_results if isinstance(result, DraftResult)]
            if len(results) != len(raw_results):
                raise RuntimeError(
                    "Draft backend received unexpected item inside batch result"
                )
            results_by_key = {result.key: result for result in results}
            if len(results_by_key) != len(results):
                raise RuntimeError(
                    "Draft backend batch returned duplicate DraftLookupKey values"
                )
            for key, inflight in entries:
                result = results_by_key.get(key)
                if result is None:
                    raise RuntimeError(
                        "Draft backend batch result missing key: "
                        f"{key.request_id}:{key.draft_round_id}"
                    )
                expected_rid = inflight.rid
                if result.rid != expected_rid:
                    raise RuntimeError(
                        "Draft backend result rid mismatch: "
                        f"{result.rid!r} != {expected_rid!r}"
                    )
                submit_to_ready_ms = (time.perf_counter() - inflight.submit_ts) * 1000
                self.proxy.complete_request(key, result)
                completed_results.append(result)
                # print(
                #     "[decoupled-draft-collect] "
                #     f"request_id={key.request_id} "
                #     f"draft_round_id={key.draft_round_id} "
                #     f"draft_index={inflight.draft_index} "
                #     f"submit_to_collect_ms={submit_to_ready_ms:.3f}",
                #     flush=True,
                # )
        if completed_results:
            _emit_draftproxy_batch_event(
                "draftproxy_collect_completed_finished",
                batch_size=len(completed_results),
                live_req_count=len(completed_results),
                status="completed",
                message="draftproxy finished handling completed draft results",
                inflight_total=len(self.proxy.inflight_requests),
                ready_total=len(self.proxy.ready_results),
                inflight_per_actor=list(self.proxy.inflight_per_index),
                results=_summarize_results(completed_results),
            )

    def _send_poll_response(
        self,
        target_dp_rank: int,
        keys: list[DraftLookupKey],
    ) -> None:
        ready_results, _ = self.proxy.peek_ready_results(keys)
        keys_to_pop = [result.key for result in ready_results]
        popped_results = self.proxy.pop_ready_results(keys_to_pop)
        response = PollDraftResultsResponse(results=popped_results)
        if popped_results:
            emit_csv_event(
                "draftproxy",
                "draftproxy_poll_response",
                server_role="draftproxy",
                verify_replica_rank=target_dp_rank,
                batch_size=len(popped_results),
                live_req_count=len(popped_results),
                message="draftproxy sent poll response",
                requested_keys=_summarize_lookup_keys(keys),
                returned_results=_summarize_results(popped_results),
            )
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
            _emit_draftproxy_batch_event(
                "draftproxy_pending_poll_still_waiting",
                verify_replica_rank=target_dp_rank,
                batch_size=len(keys),
                live_req_count=len(missing_keys),
                status="waiting",
                message="draftproxy pending poll is still waiting for draft results",
                requested_keys=_summarize_lookup_keys(keys),
                missing_keys=_summarize_lookup_keys(missing_keys),
                ready_total=len(self.proxy.ready_results),
                inflight_total=len(self.proxy.inflight_requests),
            )
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
        emit_csv_event(
            "draftproxy",
            "draftproxy_poll_waiting",
            server_role="draftproxy",
            verify_replica_rank=source_dp_rank,
            batch_size=len(poll_request.keys),
            live_req_count=len(missing_keys),
            message="draftproxy is waiting for missing draft results",
            requested_keys=_summarize_lookup_keys(poll_request.keys),
            missing_keys=_summarize_lookup_keys(missing_keys),
            ready_total=len(self.proxy.ready_results),
            inflight_total=len(self.proxy.inflight_requests),
        )
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
    ray_init_kwargs: dict[str, Any] | None = None,
    actor_lookup_timeout_s: float = 10.0,
    draft_backend_manager_class=DraftBackendManager,
) -> None:
    _ = port_args
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::draft_backend")
    faulthandler.enable()
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        _maybe_init_ray(ray_init_kwargs=ray_init_kwargs)
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
            timeout_s=actor_lookup_timeout_s,
        )
        manager = draft_backend_manager_class(
            draft_actor_handles=draft_actor_handles,
            ipc_config=ipc_config,
        )
        emit_summary(
            logger,
            key="draftproxy.start",
            component="draftproxy",
            event="draftproxy_started",
            message="draftproxy process started",
            server_role="draftproxy",
            **build_process_trace_fields(trace_key="draftproxy:start"),
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
        emit_csv_event(
            "draftproxy",
            "draftproxy_exception",
            server_role="draftproxy",
            status="error",
            message="draft backend process hit an exception",
            **build_process_trace_fields(trace_key=None),
            traceback=traceback,
        )
        parent_process.send_signal(signal.SIGQUIT)
