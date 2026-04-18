from __future__ import annotations

import queue
import threading

import zmq

from sglang.srt.speculative.decoupled_spec_io import (
    DraftClose,
    DraftControlBatch,
    DraftMeshIpcConfig,
    DraftMeshMessage,
    DraftMeshMessageType,
    DraftSync,
    VerifyCommit,
)
from sglang.srt.speculative.draft_tail_buffer import DraftTailBuffer
from sglang.srt.utils import get_zmq_socket


class DraftProxyThread:
    """
    Verifier-side proxy thread for decoupled speculation.

    Control batches from the verifier are first applied to the local
    DraftTailBuffer, then forwarded to the drafter. Draft tail stream batches
    from the drafter are appended to the same buffer.
    """

    def __init__(
        self,
        *,
        context: zmq.Context,
        ipc_config: DraftMeshIpcConfig,
        verifier_rank: int,
        draft_tail_buffer: DraftTailBuffer,
    ) -> None:
        self.verifier_rank = int(verifier_rank)
        self.draft_tail_buffer = draft_tail_buffer
        # verifier -> drafter send control messages
        self.control_send_sockets: dict[int, zmq.Socket] = {
            drafter_rank: get_zmq_socket(
                context,
                zmq.PUSH,
                endpoint,
                False,
            )
            for drafter_rank, endpoint in sorted(ipc_config.control_endpoints.items())
        }
        self.result_recv_socket = get_zmq_socket(
            context,
            zmq.PULL,
            ipc_config.get_result_endpoint(self.verifier_rank),
            True,
        )
        self._send_queue: queue.SimpleQueue[DraftMeshMessage] = queue.SimpleQueue()
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="sglang-draft-proxy",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._closed.set()
        self.draft_tail_buffer.close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        for socket in self.control_send_sockets.values():
            socket.close(linger=0)
        self.result_recv_socket.close(linger=0)

    def submit_sync(self, message: DraftSync) -> None:
        self.submit_control_batch(
            DraftControlBatch(
                dst_drafter_rank=int(message.dst_drafter_rank),
                sync_messages=[message],
            )
        )

    def submit_verify_commit(self, message: VerifyCommit) -> None:
        self.submit_control_batch(
            DraftControlBatch(
                dst_drafter_rank=int(message.dst_drafter_rank),
                verify_commit_messages=[message],
            )
        )

    def submit_close(self, message: DraftClose) -> None:
        self.submit_control_batch(
            DraftControlBatch(
                dst_drafter_rank=int(message.dst_drafter_rank),
                close_messages=[message],
            )
        )

    def submit_control_batch(self, message: DraftControlBatch) -> None:
        self.draft_tail_buffer.apply_control_batch(message)
        self._send_queue.put(DraftMeshMessage.from_control_batch(message))

    def _recv_tail_stream_output(self) -> None:
        message = self.result_recv_socket.recv_pyobj()
        if not isinstance(message, DraftMeshMessage):
            raise RuntimeError(f"Unexpected draft proxy message: {message}")
        if (
            message.message_type != DraftMeshMessageType.TAIL_STREAM_OUTPUT_BATCH
            or message.tail_stream_output_batch is None
        ):
            raise RuntimeError(f"Unexpected draft proxy message: {message}")

        output_batch = message.tail_stream_output_batch
        if int(output_batch.dst_verifier_rank) != self.verifier_rank:
            return
        self.draft_tail_buffer.append_draft_stream(list(output_batch.stream_outputs))

    def _dst_drafter_rank(self, message: DraftMeshMessage) -> int:
        if message.control_batch is not None:
            return int(message.control_batch.dst_drafter_rank)
        raise RuntimeError(f"Draft control message missing destination: {message}")

    def _send_control_message(self, message: DraftMeshMessage) -> None:
        dst_drafter_rank = self._dst_drafter_rank(message)
        socket = self.control_send_sockets.get(dst_drafter_rank)
        if socket is None:
            raise RuntimeError(
                f"Missing control socket for dst_drafter_rank={dst_drafter_rank}"
            )
        socket.send_pyobj(message)

    def _run(self) -> None:
        while not self._closed.is_set():
            while True:
                try:
                    message = self._send_queue.get_nowait()
                except queue.Empty:
                    break
                self._send_control_message(message)

            try:
                if self.result_recv_socket.poll(timeout=1):
                    self._recv_tail_stream_output()
            except zmq.error.ContextTerminated:
                break
