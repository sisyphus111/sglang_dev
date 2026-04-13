from __future__ import annotations

import csv
import json
import logging
import os
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRACE_SCOPE_BATCH = "batch"
TRACE_SCOPE_PROCESS = "process"

CSV_FIELDNAMES = [
    "timestamp",
    "source",
    "component",
    "event",
    "hostname",
    "pid",
    "process_name",
    "server_role",
    "status",
    "message",
    "trace_scope",
    "trace_key",
    "verify_request_id",
    "draft_round_id",
    "draft_sglang_rid",
    "verify_replica_rank",
    "replica_rank",
    "node_rank",
    "local_rank",
    "tp_rank",
    "pp_rank",
    "global_steps",
    "batch_size",
    "live_req_count",
    "param_count",
    "bucket_index",
    "route_draft_index",
    "duration_ms",
    "accepted_draft_tokens",
    "verified_tokens",
    "avg_accepted_draft_len",
    "avg_verified_len",
    "details_json",
]

_LOGGER_LOCK = threading.Lock()
_LOGGER_CACHE: dict[str, "_CSVEmitter"] = {}
_SUMMARY_LOCK = threading.Lock()
_LAST_SUMMARY_TS: dict[str, float] = {}


def normalize_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def normalize_int(value: Any) -> int | None:
    text = normalize_str(value)
    if text is None:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def sanitize_message(value: Any) -> str | None:
    text = normalize_str(value)
    if text is None:
        return None
    prefix = "[decoupled-spec]"
    while text.startswith(prefix):
        text = text[len(prefix) :].lstrip()
    return text or None


def build_batch_trace_fields(*, trace_key: str | None = None) -> dict[str, Any]:
    return {
        "trace_scope": TRACE_SCOPE_BATCH,
        "trace_key": normalize_str(trace_key),
        "verify_request_id": None,
        "draft_round_id": None,
        "draft_sglang_rid": None,
    }


def build_process_trace_fields(*, trace_key: str | None = None) -> dict[str, Any]:
    return {
        "trace_scope": TRACE_SCOPE_PROCESS,
        "trace_key": normalize_str(trace_key),
        "verify_request_id": None,
        "draft_round_id": None,
        "draft_sglang_rid": None,
    }


def _serialize_details(
    *,
    details_json: Any,
    extra_json: Any,
    remaining_kwargs: dict[str, Any],
) -> str | None:
    details: dict[str, Any] = {}
    if isinstance(details_json, dict):
        details.update(details_json)
    elif details_json not in (None, ""):
        details["_details_json"] = details_json
    if isinstance(extra_json, dict):
        details.update(extra_json)
    elif extra_json not in (None, ""):
        details["_legacy_extra_json"] = extra_json
    if remaining_kwargs:
        details.update(remaining_kwargs)
    if not details:
        return None
    return json.dumps(details, ensure_ascii=False, sort_keys=True)


def build_log_row(source: str, component: str, event: str, **kwargs: Any) -> dict[str, Any]:
    details_json = kwargs.pop("details_json", None)
    extra_json = kwargs.pop("extra_json", None)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "component": component,
        "event": event,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "process_name": kwargs.pop("process_name", None),
        "server_role": kwargs.pop("server_role", None),
        "status": kwargs.pop("status", None),
        "message": kwargs.pop("message", None),
        "trace_scope": kwargs.pop("trace_scope", None),
        "trace_key": kwargs.pop("trace_key", None),
        "verify_request_id": kwargs.pop("verify_request_id", None),
        "draft_round_id": kwargs.pop("draft_round_id", None),
        "draft_sglang_rid": kwargs.pop("draft_sglang_rid", None),
        "verify_replica_rank": kwargs.pop("verify_replica_rank", None),
        "replica_rank": kwargs.pop("replica_rank", None),
        "node_rank": kwargs.pop("node_rank", None),
        "local_rank": kwargs.pop("local_rank", None),
        "tp_rank": kwargs.pop("tp_rank", None),
        "pp_rank": kwargs.pop("pp_rank", None),
        "global_steps": kwargs.pop("global_steps", None),
        "batch_size": kwargs.pop("batch_size", None),
        "live_req_count": kwargs.pop("live_req_count", None),
        "param_count": kwargs.pop("param_count", None),
        "bucket_index": kwargs.pop("bucket_index", None),
        "route_draft_index": kwargs.pop("route_draft_index", None),
        "duration_ms": kwargs.pop("duration_ms", None),
        "accepted_draft_tokens": kwargs.pop("accepted_draft_tokens", None),
        "verified_tokens": kwargs.pop("verified_tokens", None),
        "avg_accepted_draft_len": kwargs.pop("avg_accepted_draft_len", None),
        "avg_verified_len": kwargs.pop("avg_verified_len", None),
        "details_json": None,
    }
    kwargs.pop("scheduler_dp_rank", None)
    kwargs.pop("dp_rank", None)

    row["verify_request_id"] = normalize_str(row["verify_request_id"])
    row["draft_round_id"] = normalize_int(row["draft_round_id"])
    row["draft_sglang_rid"] = normalize_str(row["draft_sglang_rid"])
    row["trace_scope"] = normalize_str(row["trace_scope"])
    row["trace_key"] = normalize_str(row["trace_key"])
    row["message"] = sanitize_message(row["message"])
    if row["trace_scope"] is None:
        row["trace_scope"] = (
            TRACE_SCOPE_BATCH if row["batch_size"] is not None else TRACE_SCOPE_PROCESS
        )
    row["details_json"] = _serialize_details(
        details_json=details_json,
        extra_json=extra_json,
        remaining_kwargs=kwargs,
    )
    return row


def format_summary_line(row: dict[str, Any]) -> str:
    parts = [
        "[decoupled-spec]",
        f"source={row.get('source')}",
        f"component={row.get('component')}",
        f"event={row.get('event')}",
        f"trace_scope={row.get('trace_scope')}",
    ]
    for key in (
        "server_role",
        "status",
        "trace_key",
        "verify_request_id",
        "draft_round_id",
        "draft_sglang_rid",
        "verify_replica_rank",
        "batch_size",
        "duration_ms",
    ):
        value = row.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    message = normalize_str(row.get("message"))
    if message is not None:
        parts.append(f"message={message}")
    return " ".join(parts)


def _get_debug_root() -> Path | None:
    directory = os.getenv("DECOUPLED_SPEC_DEBUG_CSV_DIR")
    if not directory:
        return None
    return Path(directory).expanduser().resolve() / "sglang"


def debug_csv_enabled() -> bool:
    return _get_debug_root() is not None


def get_summary_interval(default: float = 10.0) -> float:
    raw_value = os.getenv("DECOUPLED_SPEC_DEBUG_SUMMARY_INTERVAL_SEC")
    if not raw_value:
        return default
    try:
        return max(float(raw_value), 0.0)
    except ValueError:
        return default


class _CSVEmitter:
    def __init__(self, component: str):
        root = _get_debug_root()
        assert root is not None
        root.mkdir(parents=True, exist_ok=True)
        hostname = socket.gethostname()
        pid = os.getpid()
        self.path = root / f"{component}_{hostname}_{pid}.csv"
        self._lock = threading.Lock()
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=CSV_FIELDNAMES)
                writer.writeheader()

    def emit(self, row: dict[str, Any]) -> None:
        with self._lock:
            with self.path.open("a", newline="", encoding="utf-8") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=CSV_FIELDNAMES)
                writer.writerow(row)


def _get_emitter(component: str) -> _CSVEmitter | None:
    root = _get_debug_root()
    if root is None:
        return None
    with _LOGGER_LOCK:
        emitter = _LOGGER_CACHE.get(component)
        if emitter is None:
            emitter = _CSVEmitter(component)
            _LOGGER_CACHE[component] = emitter
        return emitter


def emit_csv_event(component: str, event: str, **kwargs: Any) -> None:
    emitter = _get_emitter(component)
    if emitter is None:
        return
    row = build_log_row("sglang", component, event, **kwargs)
    emitter.emit(row)


def emit_summary(
    logger: logging.Logger,
    *,
    key: str,
    component: str,
    event: str,
    message: str,
    interval_s: float | None = None,
    level: int = logging.INFO,
    **kwargs: Any,
) -> bool:
    if interval_s is None:
        interval_s = get_summary_interval()
    now = time.monotonic()
    if interval_s > 0:
        with _SUMMARY_LOCK:
            last_ts = _LAST_SUMMARY_TS.get(key, 0.0)
            if now - last_ts < interval_s:
                return False
            _LAST_SUMMARY_TS[key] = now
    _ = logger, level
    emitter = _get_emitter(component)
    if emitter is None:
        return False
    row = build_log_row("sglang", component, event, message=message, **kwargs)
    print(format_summary_line(row), flush=True)
    emitter.emit(row)
    return True
