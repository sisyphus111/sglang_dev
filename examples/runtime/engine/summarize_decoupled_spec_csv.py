from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROUND_EVENT_FIELDS = {
    "verify_submit_ts": "verify_draft_request_registered",
    "draft_start_ts": "drafter_handle_draft_request_start",
    "draft_return_ts": "drafter_draft_result_returned",
    "draft_ready_ts": "draft_result_ready",
    "bind_ts": "verify_draft_result_bound",
}

TRACE_KEY_PATTERN = re.compile(r"^(?P<request_id>.*?):(?P<round_id>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize decoupled-spec CSV traces into per-round and aggregate reports."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Trace root directory. Can be DECOUPLED_SPEC_DEBUG_CSV_DIR or its sglang subdir.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <input-dir>/summary.",
    )
    return parser.parse_args()


def _resolve_trace_dir(input_dir: str) -> Path:
    root = Path(input_dir).expanduser().resolve()
    if (root / "sglang").is_dir():
        return root / "sglang"
    return root


def _parse_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trace_key_sort_key(trace_key: str) -> tuple[int, str, int, str]:
    match = TRACE_KEY_PATTERN.match(trace_key)
    if match is None:
        return (1, trace_key, -1, trace_key)
    return (
        0,
        match.group("request_id"),
        int(match.group("round_id")),
        trace_key,
    )


def _percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def load_rows(trace_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(trace_dir.glob("*.csv")):
        with path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            for row in reader:
                row["_path"] = str(path)
                rows.append(row)
    return rows


def summarize_rounds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rounds: dict[str, dict[str, Any]] = {}
    for row in rows:
        trace_key = row.get("trace_key")
        if not trace_key:
            continue
        event = row.get("event")
        item = rounds.setdefault(
            trace_key,
            {
                "trace_key": trace_key,
                "verify_request_id": row.get("verify_request_id"),
                "draft_round_id": row.get("draft_round_id"),
                "verify_submit_ts": None,
                "draft_start_ts": None,
                "draft_return_ts": None,
                "draft_ready_ts": None,
                "bind_ts": None,
                "submit_to_ready_ms": None,
                "draft_exec_ms": None,
                "ready_to_bind_ms": None,
                "end_to_end_round_ms": None,
                "error_event_count": 0,
                "empty_event_count": 0,
                "mismatch_event_count": 0,
            },
        )
        timestamp = _parse_timestamp(row.get("timestamp"))
        for field_name, expected_event in ROUND_EVENT_FIELDS.items():
            if event == expected_event and item[field_name] is None:
                item[field_name] = timestamp
        if row.get("status") == "error" or "failed" in (event or ""):
            item["error_event_count"] += 1
        if row.get("status") == "empty" or event == "empty_draft_result":
            item["empty_event_count"] += 1
        if event == "tail_token_mismatch":
            item["mismatch_event_count"] += 1

    summaries: list[dict[str, Any]] = []
    for item in rounds.values():
        submit_ts = item["verify_submit_ts"]
        ready_ts = item["draft_ready_ts"]
        draft_start_ts = item["draft_start_ts"]
        draft_return_ts = item["draft_return_ts"]
        bind_ts = item["bind_ts"]
        if submit_ts is not None and ready_ts is not None:
            item["submit_to_ready_ms"] = round((ready_ts - submit_ts) * 1000, 3)
        if draft_start_ts is not None and draft_return_ts is not None:
            item["draft_exec_ms"] = round((draft_return_ts - draft_start_ts) * 1000, 3)
        if ready_ts is not None and bind_ts is not None:
            item["ready_to_bind_ms"] = round((bind_ts - ready_ts) * 1000, 3)
        if submit_ts is not None and bind_ts is not None:
            item["end_to_end_round_ms"] = round((bind_ts - submit_ts) * 1000, 3)
        summaries.append(item)

    summaries.sort(key=lambda row: _trace_key_sort_key(row["trace_key"]))
    return summaries


def summarize_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_counts: Counter[tuple[str, str]] = Counter()
    grouped_durations: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row.get("component") or "", row.get("event") or "")
        grouped_counts[key] += 1
        duration_ms = _parse_float(row.get("duration_ms"))
        if duration_ms is not None:
            grouped_durations[key].append(duration_ms)

    summaries: list[dict[str, Any]] = []
    for key in sorted(grouped_counts):
        durations = sorted(grouped_durations.get(key, []))
        summaries.append(
            {
                "component": key[0],
                "event": key[1],
                "count": grouped_counts[key],
                "duration_count": len(durations),
                "duration_mean_ms": (
                    round(sum(durations) / len(durations), 3) if durations else None
                ),
                "duration_p50_ms": (
                    round(_percentile(durations, 0.5), 3) if durations else None
                ),
                "duration_p95_ms": (
                    round(_percentile(durations, 0.95), 3) if durations else None
                ),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    trace_dir = _resolve_trace_dir(args.input_dir)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else trace_dir / "summary"
    )
    rows = load_rows(trace_dir)
    round_summary = summarize_rounds(rows)
    event_summary = summarize_events(rows)

    write_csv(output_dir / "request_round_summary.csv", round_summary)
    write_csv(output_dir / "event_summary.csv", event_summary)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "trace_dir": str(trace_dir),
                "row_count": len(rows),
                "request_round_count": len(round_summary),
                "event_summary_count": len(event_summary),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[decoupled-spec-summary] trace_dir={trace_dir}")
    print(f"[decoupled-spec-summary] round_summary={output_dir / 'request_round_summary.csv'}")
    print(f"[decoupled-spec-summary] event_summary={output_dir / 'event_summary.csv'}")
    print(f"[decoupled-spec-summary] summary_json={output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
