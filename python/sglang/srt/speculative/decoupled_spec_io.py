from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DraftMeshMessageType(str, Enum):
    SYNC = "sync"
    VERIFY_COMMIT = "verify_commit"
    CLOSE = "close"
    CONTROL_BATCH = "control_batch"
    TAIL_STREAM_OUTPUT = "tail_stream_output"
    TAIL_STREAM_OUTPUT_BATCH = "tail_stream_output_batch"


def build_draft_scheduler_rid(request_id: str) -> str:
    return f"draft-{request_id}"


@dataclass(frozen=True)
class DraftReqKey:
    src_verifier_rank: int
    request_id: str


@dataclass
class DraftSync:
    request_id: str
    src_verifier_rank: int
    dst_drafter_rank: int
    prompt_token_ids: list[int] = field(default_factory=list)
    committed_output_ids: list[int] = field(default_factory=list)

    @property
    def draft_key(self) -> DraftReqKey:
        return DraftReqKey(
            src_verifier_rank=int(self.src_verifier_rank),
            request_id=self.request_id,
        )


@dataclass
class VerifyCommit:
    """
    Sent from verifier to drafter to commit a portion of the draft outputs.

    Drafter relies on pre_verify_committed_len and bonus_token_pos to
    correctly push forward the verifier_committed_prefix_len.
    """
    request_id: str
    src_verifier_rank: int
    dst_drafter_rank: int
    pre_verify_committed_len: int
    bonus_token_id: int
    bonus_token_pos: int

    @property
    def draft_key(self) -> DraftReqKey:
        return DraftReqKey(
            src_verifier_rank=int(self.src_verifier_rank),
            request_id=self.request_id,
        )


class RequestTerminateReason(str, Enum):
    FINISHED = "finished"
    ABORT = "abort"


@dataclass
class DraftClose:
    request_id: str
    src_verifier_rank: int
    dst_drafter_rank: int
    reason: RequestTerminateReason

    @property
    def draft_key(self) -> DraftReqKey:
        return DraftReqKey(
            src_verifier_rank=int(self.src_verifier_rank),
            request_id=self.request_id,
        )


@dataclass
class DraftTailStreamOutput:
    """
    Drafter sends a stream output to verifier whenever it decodes a new token.
    """
    request_id: str
    src_drafter_rank: int
    dst_verifier_rank: int
    base_committed_len: int
    new_token_pos: int
    new_token_id: int


@dataclass
class DraftTailStreamOutputBatch:
    dst_verifier_rank: int
    stream_outputs: list[DraftTailStreamOutput] = field(default_factory=list)


DraftControlMessage = DraftSync | VerifyCommit | DraftClose


@dataclass
class DraftControlBatch:
    dst_drafter_rank: int
    sync_messages: list[DraftSync] = field(default_factory=list)
    verify_commit_messages: list[VerifyCommit] = field(default_factory=list)
    close_messages: list[DraftClose] = field(default_factory=list)


def iter_control_batch_messages(batch: DraftControlBatch) -> list[DraftControlMessage]:
    return [
        *batch.sync_messages,
        *batch.verify_commit_messages,
        *batch.close_messages,
    ]


@dataclass
class DraftMeshMessage:
    message_type: DraftMeshMessageType
    sync: Optional[DraftSync] = None
    verify_commit: Optional[VerifyCommit] = None
    close: Optional[DraftClose] = None
    control_batch: Optional[DraftControlBatch] = None
    tail_stream_output: Optional[DraftTailStreamOutput] = None
    tail_stream_output_batch: Optional[DraftTailStreamOutputBatch] = None

    @staticmethod
    def from_sync(message: DraftSync) -> "DraftMeshMessage":
        return DraftMeshMessage(
            message_type=DraftMeshMessageType.SYNC,
            sync=message,
        )

    @staticmethod
    def from_verify_commit(message: VerifyCommit) -> "DraftMeshMessage":
        return DraftMeshMessage(
            message_type=DraftMeshMessageType.VERIFY_COMMIT,
            verify_commit=message,
        )

    @staticmethod
    def from_close(message: DraftClose) -> "DraftMeshMessage":
        return DraftMeshMessage(
            message_type=DraftMeshMessageType.CLOSE,
            close=message,
        )

    @staticmethod
    def from_control_batch(message: DraftControlBatch) -> "DraftMeshMessage":
        return DraftMeshMessage(
            message_type=DraftMeshMessageType.CONTROL_BATCH,
            control_batch=message,
        )

    @staticmethod
    def from_tail_stream_output(message: DraftTailStreamOutput) -> "DraftMeshMessage":
        return DraftMeshMessage(
            message_type=DraftMeshMessageType.TAIL_STREAM_OUTPUT,
            tail_stream_output=message,
        )

    @staticmethod
    def from_tail_stream_output_batch(
        message: DraftTailStreamOutputBatch,
    ) -> "DraftMeshMessage":
        return DraftMeshMessage(
            message_type=DraftMeshMessageType.TAIL_STREAM_OUTPUT_BATCH,
            tail_stream_output_batch=message,
        )


@dataclass(frozen=True)
class DraftMeshDpIpcEndpoints:
    verifier_to_drafter_control_ipc_name: str
    drafter_to_verifier_result_ipc_name: str

    @staticmethod
    def init_new() -> "DraftMeshDpIpcEndpoints":
        return DraftMeshDpIpcEndpoints(
            verifier_to_drafter_control_ipc_name=(
                f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
            ),
            drafter_to_verifier_result_ipc_name=(
                f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
            ),
        )


@dataclass(frozen=True)
class DraftMeshIpcConfig:
    control_endpoints: dict[int, str]
    result_endpoints: dict[int, str]

    @staticmethod
    def init_new(dp_size: int = 1) -> "DraftMeshIpcConfig":
        normalized_dp_size = max(1, int(dp_size))
        endpoints = {
            rank: DraftMeshDpIpcEndpoints.init_new()
            for rank in range(normalized_dp_size)
        }
        return DraftMeshIpcConfig(
            control_endpoints={
                rank: endpoint.verifier_to_drafter_control_ipc_name
                for rank, endpoint in endpoints.items()
            },
            result_endpoints={
                rank: endpoint.drafter_to_verifier_result_ipc_name
                for rank, endpoint in endpoints.items()
            },
        )

    @staticmethod
    def from_endpoint_lists(
        control_endpoints: list[str],
        result_endpoints: list[str],
    ) -> "DraftMeshIpcConfig":
        return DraftMeshIpcConfig(
            control_endpoints=dict(enumerate(control_endpoints)),
            result_endpoints=dict(enumerate(result_endpoints)),
        )

    def get_control_endpoint(self, drafter_rank: Optional[int]) -> str:
        normalized_rank = 0 if drafter_rank is None else int(drafter_rank)
        endpoint = self.control_endpoints.get(normalized_rank)
        if endpoint is not None:
            return endpoint
        if normalized_rank != 0 and 0 in self.control_endpoints:
            return self.control_endpoints[0]
        raise KeyError(
            f"Draft mesh IPC config missing control endpoint for drafter_rank={normalized_rank}"
        )

    def get_result_endpoint(self, verifier_rank: Optional[int]) -> str:
        normalized_rank = 0 if verifier_rank is None else int(verifier_rank)
        endpoint = self.result_endpoints.get(normalized_rank)
        if endpoint is not None:
            return endpoint
        if normalized_rank != 0 and 0 in self.result_endpoints:
            return self.result_endpoints[0]
        raise KeyError(
            f"Draft mesh IPC config missing result endpoint for verifier_rank={normalized_rank}"
        )

    def get_endpoints(self, dp_rank: Optional[int]) -> DraftMeshDpIpcEndpoints:
        normalized_dp_rank = 0 if dp_rank is None else int(dp_rank)
        return DraftMeshDpIpcEndpoints(
            verifier_to_drafter_control_ipc_name=self.get_control_endpoint(
                normalized_dp_rank
            ),
            drafter_to_verifier_result_ipc_name=self.get_result_endpoint(
                normalized_dp_rank
            ),
        )
