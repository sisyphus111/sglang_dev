from __future__ import annotations

import atexit
import csv
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


COMPONENT_FIELDNAMES: dict[str, list[str]] = {
    "verifier": [
        "wall_time_ns",
        "op",
        "duration_ms",
        "forward_mode",
        "batch_size",
        "num_sync",
        "num_commit",
        "num_close",
        "request_ids",
        "draft_token_lens_by_req",
        "accepted_tail_lens_by_req",
        "committed_lens_by_req",
        "output_lens_by_req",
        "dst_drafter_ranks",
    ],
    "drafter": [
        "wall_time_ns",
        "op",
        "duration_ms",
        "forward_mode",
        "batch_size",
        "num_sync",
        "num_commit",
        "num_close",
        "num_stream_outputs",
        "request_ids",
        "emitted_token_lens_by_req",
        "committed_lens_by_req",
        "output_lens_by_req",
    ],
    "draft_proxy": [
        "wall_time_ns",
        "op",
        "duration_ms",
        "verifier_rank",
        "dst_drafter_rank",
        "batch_size",
        "num_sync",
        "num_commit",
        "num_close",
        "num_stream_outputs",
        "request_ids",
        "draft_token_lens_by_req",
    ],
    "draft_adapter": [
        "wall_time_ns",
        "op",
        "duration_ms",
        "drafter_rank",
        "dst_verifier_rank",
        "batch_size",
        "num_sync",
        "num_commit",
        "num_close",
        "num_stream_outputs",
        "request_ids",
        "emitted_token_lens_by_req",
    ],
}


@dataclass
class _TraceEvent:
    component: str
    row: dict[str, Any]


class NullDecoupledSpecTracer:
    enabled = False

    def record(self, component: str, op: str, **fields: Any) -> None:
        return

    def close(self) -> None:
        return


class DecoupledSpecCsvTracer:
    enabled = True

    def __init__(self, *, output_dir: str | Path, file_names: dict[str, str]) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.file_names = dict(file_names)
        self._queue: queue.Queue[_TraceEvent | None] = queue.Queue(maxsize=0)
        self._closed = threading.Event()
        self._writers: dict[str, csv.DictWriter] = {}
        self._files: dict[str, Any] = {}
        self._thread = threading.Thread(
            target=self._run,
            name="sglang-decoupled-spec-trace-writer",
            daemon=True,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        atexit.register(self.close)

    def record(self, component: str, op: str, **fields: Any) -> None:
        if self._closed.is_set():
            return
        row = {"wall_time_ns": time.time_ns(), "op": op}
        row.update(fields)
        self._queue.put(_TraceEvent(component=component, row=row))

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._queue.put(None)
        if self._thread.is_alive():
            self._thread.join()

    def _run(self) -> None:
        try:
            while True:
                event = self._queue.get()
                if event is None:
                    break
                self._write_event(event)
            self._write_trace_complete_rows()
        except Exception:
            logger.exception("Decoupled spec trace writer failed")
        finally:
            for file in self._files.values():
                try:
                    file.flush()
                    file.close()
                except Exception:
                    logger.exception("Failed to close decoupled spec trace file")

    def _write_event(self, event: _TraceEvent) -> None:
        writer = self._get_writer(event.component)
        fieldnames = COMPONENT_FIELDNAMES[event.component]
        row = {key: "" for key in fieldnames}
        for key, value in event.row.items():
            if key in row:
                row[key] = self._serialize_value(value)
        writer.writerow(row)

    def _write_trace_complete_rows(self) -> None:
        for component in list(self._writers.keys()):
            writer = self._writers[component]
            fieldnames = COMPONENT_FIELDNAMES[component]
            row = {key: "" for key in fieldnames}
            row["wall_time_ns"] = str(time.time_ns())
            row["op"] = "trace_complete"
            writer.writerow(row)

    def _get_writer(self, component: str) -> csv.DictWriter:
        if component not in COMPONENT_FIELDNAMES:
            raise ValueError(f"Unknown decoupled spec trace component: {component}")
        writer = self._writers.get(component)
        if writer is not None:
            return writer

        file_name = self.file_names.get(component, f"{component}.csv")
        path = self.output_dir / file_name
        file = path.open("w", newline="", buffering=1024 * 1024)
        writer = csv.DictWriter(file, fieldnames=COMPONENT_FIELDNAMES[component])
        writer.writeheader()
        self._files[component] = file
        self._writers[component] = writer
        return writer

    def _serialize_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def build_decoupled_spec_tracer(
    *,
    enabled: bool,
    output_dir: str | None,
    file_names: dict[str, str],
) -> NullDecoupledSpecTracer | DecoupledSpecCsvTracer:
    if not enabled:
        return NullDecoupledSpecTracer()
    return DecoupledSpecCsvTracer(
        output_dir=output_dir or "decoupled_spec_trace",
        file_names=file_names,
    )
