from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import zmq

from sglang.srt.utils import get_zmq_socket


class DraftBackendMessageType(str, Enum):
    SUBMIT_DRAFT = "submit_draft"
    POLL_DRAFT_RESULTS = "poll_draft_results"
    POLL_RESPONSE = "poll_response"
    REQUEST_TERMINATE = "request_terminate"


@dataclass(frozen=True)
class DraftLookupKey:
    request_id: str
    draft_round_id: int


@dataclass
class DraftRoute:
    request_id: str
    draft_index: int


@dataclass
class DraftRequest:
    request_id: str
    draft_round_id: int = 0
    scheduler_dp_rank: int = 0
    prompt_token_ids: list[int] = field(default_factory=list)
    committed_token_ids: list[int] = field(default_factory=list)
    num_speculative_steps: int = 0
    sampling_params: dict[str, Any] = field(default_factory=dict)

    @property
    def full_token_ids(self) -> list[int]:
        return list(self.prompt_token_ids) + list(self.committed_token_ids)

    @property
    def key(self) -> DraftLookupKey:
        return DraftLookupKey(
            request_id=self.request_id,
            draft_round_id=self.draft_round_id,
        )


@dataclass
class DraftResult:
    request_id: str
    draft_round_id: int = 0
    request_prompt_length: int = 0
    draft_token_ids: list[int] = field(default_factory=list)

    @property
    def key(self) -> DraftLookupKey:
        return DraftLookupKey(
            request_id=self.request_id,
            draft_round_id=self.draft_round_id,
        )


class RequestTerminateReason(str, Enum):
    FINISHED = "finished"
    ABORT = "abort"


@dataclass
class RequestTerminateMessage:
    request_id: str
    reason: RequestTerminateReason
    draft_round_id_upper_bound: Optional[int] = None


@dataclass
class PollDraftResultsRequest:
    keys: list[DraftLookupKey] = field(default_factory=list)


@dataclass
class PollDraftResultsResponse:
    results: list[DraftResult] = field(default_factory=list)


@dataclass(frozen=True)
class DraftBackendDpIpcEndpoints:
    scheduler_to_backend_ipc_name: str
    backend_to_scheduler_ipc_name: str

    @staticmethod
    def init_new() -> DraftBackendDpIpcEndpoints:
        return DraftBackendDpIpcEndpoints(
            scheduler_to_backend_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            backend_to_scheduler_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
        )


@dataclass(frozen=True)
class DraftBackendIpcConfig:
    dp_ipc_endpoints: dict[int, DraftBackendDpIpcEndpoints]

    @staticmethod
    def init_new(dp_size: int = 1) -> DraftBackendIpcConfig:
        normalized_dp_size = max(1, int(dp_size))
        return DraftBackendIpcConfig(
            dp_ipc_endpoints={
                dp_rank: DraftBackendDpIpcEndpoints.init_new()
                for dp_rank in range(normalized_dp_size)
            }
        )

    def get_endpoints(
        self, dp_rank: Optional[int]
    ) -> DraftBackendDpIpcEndpoints:
        normalized_dp_rank = 0 if dp_rank is None else int(dp_rank)
        endpoints = self.dp_ipc_endpoints.get(normalized_dp_rank)
        if endpoints is not None:
            return endpoints
        if normalized_dp_rank != 0 and 0 in self.dp_ipc_endpoints:
            return self.dp_ipc_endpoints[0]
        raise KeyError(
            f"Draft backend IPC config missing endpoints for dp_rank={normalized_dp_rank}"
        )


@dataclass
class DraftBackendClient:
    send_socket: zmq.Socket
    recv_socket: zmq.Socket

    @classmethod
    def create(
        cls,
        context: zmq.Context,
        endpoints: DraftBackendDpIpcEndpoints,
    ) -> DraftBackendClient:
        return cls(
            send_socket=get_zmq_socket(
                context,
                zmq.PUSH,
                endpoints.scheduler_to_backend_ipc_name,
                False,
            ),
            recv_socket=get_zmq_socket(
                context,
                zmq.PULL,
                endpoints.backend_to_scheduler_ipc_name,
                False,
            ),
        )

    def send_message(self, message: DraftBackendMessage) -> None:
        self.send_socket.send_pyobj(message)

    def recv_message(self) -> DraftBackendMessage:
        return self.recv_socket.recv_pyobj()

    def close(self) -> None:
        self.send_socket.close(linger=0)
        self.recv_socket.close(linger=0)


@dataclass
class DraftBackendMessage:
    message_type: DraftBackendMessageType
    requests: Optional[list[DraftRequest]] = None
    poll_request: Optional[PollDraftResultsRequest] = None
    poll_response: Optional[PollDraftResultsResponse] = None
    terminate: Optional[RequestTerminateMessage] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_submit_draft(
        requests: list[DraftRequest],
    ) -> DraftBackendMessage:
        return DraftBackendMessage(
            message_type=DraftBackendMessageType.SUBMIT_DRAFT,
            requests=requests,
        )

    @staticmethod
    def from_poll_request(
        poll_request: PollDraftResultsRequest,
    ) -> DraftBackendMessage:
        return DraftBackendMessage(
            message_type=DraftBackendMessageType.POLL_DRAFT_RESULTS,
            poll_request=poll_request,
        )

    @staticmethod
    def from_poll_response(
        poll_response: PollDraftResultsResponse,
    ) -> DraftBackendMessage:
        return DraftBackendMessage(
            message_type=DraftBackendMessageType.POLL_RESPONSE,
            poll_response=poll_response,
        )

    @staticmethod
    def from_request_terminate(
        terminate: RequestTerminateMessage,
    ) -> DraftBackendMessage:
        return DraftBackendMessage(
            message_type=DraftBackendMessageType.REQUEST_TERMINATE,
            terminate=terminate,
        )
