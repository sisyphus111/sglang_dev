from __future__ import annotations

import queue
import threading
import time
from typing import Any

import zmq

from sglang.srt.speculative.decoupled_spec_io import (
    DraftClose,
    DraftControlBatch,
    DraftMeshIpcConfig,
    DraftMeshMessage,
    DraftMeshMessageType,
    DraftSync,
    DraftTailStreamOutput,
    VerifyCommit,
    iter_control_batch_messages,
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
        tracer: Any = None,
    ) -> None:
        self.verifier_rank = int(verifier_rank)
        self.draft_tail_buffer = draft_tail_buffer
        self.tracer = tracer
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
        trace_enabled = getattr(getattr(self, "tracer", None), "enabled", False)
        start_ns = time.perf_counter_ns() if trace_enabled else 0
        self.draft_tail_buffer.apply_control_batch(message)
        self._send_queue.put(DraftMeshMessage.from_control_batch(message))
        if trace_enabled:
            self._record_control_batch(
                "enqueue_control_batch",
                message,
                duration_ms=(time.perf_counter_ns() - start_ns) / 1_000_000,
            )

    def _recv_tail_stream_output(self) -> None:
        trace_enabled = getattr(getattr(self, "tracer", None), "enabled", False)
        recv_start_ns = time.perf_counter_ns() if trace_enabled else 0
        message = self.result_recv_socket.recv_pyobj()
        recv_duration_ms = (
            (time.perf_counter_ns() - recv_start_ns) / 1_000_000
            if trace_enabled
            else 0
        )
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
        outputs = list(output_batch.stream_outputs)
        if trace_enabled:
            self._record_tail_stream_batch(
                "recv_tail_stream_batch",
                outputs,
                duration_ms=recv_duration_ms,
            )
        append_start_ns = time.perf_counter_ns() if trace_enabled else 0
        self.draft_tail_buffer.append_draft_stream(outputs)
        if trace_enabled:
            self._record_tail_stream_batch(
                "append_tail_stream_batch",
                outputs,
                duration_ms=(time.perf_counter_ns() - append_start_ns) / 1_000_000,
            )

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
        trace_enabled = getattr(getattr(self, "tracer", None), "enabled", False)
        start_ns = time.perf_counter_ns() if trace_enabled else 0
        socket.send_pyobj(message)
        if trace_enabled and message.control_batch is not None:
            self._record_control_batch(
                "send_control_batch",
                message.control_batch,
                duration_ms=(time.perf_counter_ns() - start_ns) / 1_000_000,
            )

    def _record_control_batch(
        self,
        op: str,
        batch: DraftControlBatch,
        *,
        duration_ms: float,
    ) -> None:
        if not getattr(getattr(self, "tracer", None), "enabled", False):
            return
        messages = iter_control_batch_messages(batch)
        self.tracer.record(
            "draft_proxy",
            op,
            duration_ms=duration_ms,
            verifier_rank=self.verifier_rank,
            dst_drafter_rank=int(batch.dst_drafter_rank),
            batch_size=len(messages),
            num_sync=len(batch.sync_messages),
            num_commit=len(batch.verify_commit_messages),
            num_close=len(batch.close_messages),
            request_ids=[message.request_id for message in messages],
        )

    def _record_tail_stream_batch(
        self,
        op: str,
        outputs: list[DraftTailStreamOutput],
        *,
        duration_ms: float,
    ) -> None:
        if not getattr(getattr(self, "tracer", None), "enabled", False):
            return
        counts_by_request: dict[str, int] = {}
        for output in outputs:
            counts_by_request[output.request_id] = (
                counts_by_request.get(output.request_id, 0) + 1
            )
        request_ids = list(counts_by_request.keys())
        self.tracer.record(
            "draft_proxy",
            op,
            duration_ms=duration_ms,
            verifier_rank=self.verifier_rank,
            batch_size=len(request_ids),
            num_stream_outputs=len(outputs),
            request_ids=request_ids,
            draft_token_lens_by_req=[
                counts_by_request[request_id] for request_id in request_ids
            ],
        )

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
