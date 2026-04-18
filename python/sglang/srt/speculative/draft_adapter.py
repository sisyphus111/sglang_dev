from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass, field

import zmq

from sglang.srt.speculative.decoupled_spec_io import (
    DraftClose,
    DraftControlBatch,
    DraftMeshIpcConfig,
    DraftMeshMessage,
    DraftMeshMessageType,
    DraftSync,
    DraftTailStreamOutput,
    DraftTailStreamOutputBatch,
    VerifyCommit,
    iter_control_batch_messages,
)
from sglang.srt.utils import get_zmq_socket


DraftControlMessage = DraftSync | VerifyCommit | DraftClose


@dataclass
class DraftAdapterThread:
    """Drafter-side adapter thread for decoupled speculation IPC."""

    context: zmq.Context | None = None
    ipc_config: DraftMeshIpcConfig | None = None
    drafter_rank: int = 0
    _pending_controls: deque[DraftControlMessage] = field(default_factory=deque)
    control_recv_socket: zmq.Socket | None = None # verifier -> drafter recv control messages
    result_send_sockets: dict[int, zmq.Socket] = field(default_factory=dict) # drafter -> verifier send draft tokens
    _pending_lock: threading.Lock = field(default_factory=threading.Lock) # used to protect _pending_controls
    _outgoing_results: queue.SimpleQueue[list[DraftTailStreamOutput]] = field(
        default_factory=queue.SimpleQueue
    )
    _closed: threading.Event = field(default_factory=threading.Event)
    _wakeup: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def __post_init__(self) -> None:
        if self.context is None or self.ipc_config is None:
            self._thread = threading.Thread(
                target=self._run,
                name="sglang-draft-adapter",
                daemon=True,
            )
            return
        self.control_recv_socket = get_zmq_socket(
            self.context,
            zmq.PULL,
            self.ipc_config.get_control_endpoint(self.drafter_rank),
            True,
        )
        self.result_send_sockets = {
            verifier_rank: get_zmq_socket(
                self.context,
                zmq.PUSH,
                endpoint,
                False,
            )
            for verifier_rank, endpoint in sorted(self.ipc_config.result_endpoints.items())
        }
        self._thread = threading.Thread(
            target=self._run,
            name="sglang-draft-adapter",
            daemon=True,
        )

    def start(self) -> None:
        if self._thread is None:
            return
        if not self._thread.is_alive():
            self._thread.start()

    def close(self) -> None:
        self._closed.set()
        self._wakeup.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.control_recv_socket is not None:
            self.control_recv_socket.close(linger=0)
        for socket in self.result_send_sockets.values():
            socket.close(linger=0)

    def _drain_control_socket(self) -> bool:
        did_work = False
        if self.control_recv_socket is None:
            return did_work

        while True:
            try:
                message = self.control_recv_socket.recv_pyobj(zmq.NOBLOCK)
            except zmq.error.ContextTerminated:
                raise
            except zmq.ZMQError:
                break
            did_work = True
            if not isinstance(message, DraftMeshMessage):
                raise RuntimeError(f"Unexpected draft control message: {message}")
            if (
                message.message_type != DraftMeshMessageType.CONTROL_BATCH
                or message.control_batch is None
            ):
                raise RuntimeError(f"Unexpected draft control message: {message}")
            control_batch = message.control_batch
            if int(control_batch.dst_drafter_rank) != int(self.drafter_rank):
                continue
            control_messages = iter_control_batch_messages(control_batch)
            if control_messages:
                with self._pending_lock:
                    self._pending_controls.extend(control_messages)
        return did_work

    def drain_sync_messages(self) -> list[DraftSync]:
        sync_messages: list[DraftSync] = []
        remaining_controls: deque[DraftControlMessage] = deque()
        with self._pending_lock:
            while self._pending_controls:
                message = self._pending_controls.popleft()
                if isinstance(message, DraftSync):
                    sync_messages.append(message)
                else:
                    remaining_controls.append(message)
            self._pending_controls = remaining_controls
        return sync_messages

    def drain_post_result_messages(self) -> list[VerifyCommit | DraftClose]:
        post_result_messages: list[VerifyCommit | DraftClose] = []
        remaining_controls: deque[DraftControlMessage] = deque()
        with self._pending_lock:
            while self._pending_controls:
                message = self._pending_controls.popleft()
                if isinstance(message, (VerifyCommit, DraftClose)):
                    post_result_messages.append(message)
                else:
                    remaining_controls.append(message)
            self._pending_controls = remaining_controls
        return post_result_messages

    def submit_draft_results(self, results: list[DraftTailStreamOutput]) -> None:
        if not results:
            return
        self._outgoing_results.put(list(results))
        self._wakeup.set()

    def _drain_outgoing_results(self) -> bool:
        did_work = False
        while True:
            try:
                results = self._outgoing_results.get_nowait()
            except queue.Empty:
                break
            did_work = True
            self._send_draft_results(results)
        return did_work

    def _send_draft_results(self, results: list[DraftTailStreamOutput]) -> None:
        if not results:
            return

        grouped_results: dict[int, list[DraftTailStreamOutput]] = {}
        for result in results:
            grouped_results.setdefault(int(result.dst_verifier_rank), []).append(result)

        for dst_verifier_rank, group in grouped_results.items():
            socket = self.result_send_sockets.get(dst_verifier_rank)
            if socket is None:
                raise RuntimeError(
                    f"Missing result socket for dst_verifier_rank={dst_verifier_rank}"
                )

            socket.send_pyobj(
                DraftMeshMessage.from_tail_stream_output_batch(
                    DraftTailStreamOutputBatch(
                        dst_verifier_rank=dst_verifier_rank,
                        stream_outputs=group,
                    )
                )
            )

    def _run(self) -> None:
        while not self._closed.is_set():
            did_work = False
            try:
                did_work = self._drain_outgoing_results() or did_work
                did_work = self._drain_control_socket() or did_work
            except zmq.error.ContextTerminated:
                break

            if not did_work:
                self._wakeup.wait(timeout=0.0005) # 0.5ms
                self._wakeup.clear()
