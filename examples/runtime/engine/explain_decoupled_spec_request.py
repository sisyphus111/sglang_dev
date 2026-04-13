from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


LENGTH_FIELD_PATTERN = re.compile(
    r"(^|_)(len|length|count|tokens|token_count|seq_len|seq_lens|seq_lens_sum)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explain the full operation timeline for one decoupled-spec request round."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Trace root directory. Can be DECOUPLED_SPEC_DEBUG_CSV_DIR or its sglang subdir.",
    )
    parser.add_argument(
        "--verify-request-id",
        required=True,
        help="verify_request_id to inspect.",
    )
    parser.add_argument(
        "--round-id",
        required=True,
        type=int,
        help="draft round id to inspect.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <input-dir>/request_explanations.",
    )
    parser.add_argument(
        "--time-range",
        action="store_true",
        help=(
            "When set, export all events whose timestamps fall within the "
            "matched request-round time window."
        ),
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


def _load_details(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"_raw_details_json": value}
    return parsed if isinstance(parsed, dict) else {"_raw_details_json": parsed}


def _normalize_length_fields(row: dict[str, Any], details: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for source in (row, details):
        for key, value in source.items():
            if key in ("details_json", "_path"):
                continue
            if key in (
                "batch_size",
                "live_req_count",
                "accepted_draft_tokens",
                "verified_tokens",
                "avg_accepted_draft_len",
                "avg_verified_len",
            ) or LENGTH_FIELD_PATTERN.search(key):
                if value not in (None, "", [], {}):
                    fields[key] = value
    return dict(sorted(fields.items()))


def load_rows(trace_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(trace_dir.glob("*.csv")):
        with path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            for row in reader:
                row["_path"] = str(path)
                rows.append(row)
    return rows


def _is_target_row(
    row: dict[str, Any],
    *,
    verify_request_id: str,
    round_id: int,
) -> bool:
    trace_key = f"{verify_request_id}:{round_id}"
    row_trace_key = row.get("trace_key")
    row_request_id = row.get("verify_request_id")
    row_round_id = row.get("draft_round_id")
    return row_trace_key == trace_key or (
        row_request_id == verify_request_id and str(row_round_id) == str(round_id)
    )


def _get_time_range(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    timestamps = [
        ts for ts in (_parse_timestamp(row.get("timestamp")) for row in rows) if ts is not None
    ]
    if not timestamps:
        return None, None
    return min(timestamps), max(timestamps)


def _row_mentions_target_trace(
    row: dict[str, Any],
    *,
    verify_request_id: str,
    round_id: int,
) -> bool:
    details = _load_details(row.get("details_json"))
    trace_key = f"{verify_request_id}:{round_id}"
    if trace_key in (details.get("batch_trace_keys") or []):
        return True
    verify_request_ids = details.get("verify_request_ids") or []
    draft_round_ids = details.get("draft_round_ids") or []
    return any(
        request_id == verify_request_id and str(draft_round_id) == str(round_id)
        for request_id, draft_round_id in zip(
            verify_request_ids, draft_round_ids, strict=False
        )
    )


def collect_rows(
    all_rows: list[dict[str, Any]],
    *,
    verify_request_id: str,
    round_id: int,
    include_time_range: bool,
) -> tuple[list[dict[str, Any]], tuple[float | None, float | None]]:
    matched_rows = [
        row
        for row in all_rows
        if _is_target_row(
            row,
            verify_request_id=verify_request_id,
            round_id=round_id,
        )
    ]
    range_start, range_end = _get_time_range(matched_rows)
    if range_start is None or range_end is None:
        return matched_rows, (range_start, range_end)

    if not include_time_range:
        return [
            row
            for row in all_rows
            if _is_target_row(
                row,
                verify_request_id=verify_request_id,
                round_id=round_id,
            )
            or _row_mentions_target_trace(
                row,
                verify_request_id=verify_request_id,
                round_id=round_id,
            )
        ], (range_start, range_end)

    ranged_rows: list[dict[str, Any]] = []
    for row in all_rows:
        timestamp = _parse_timestamp(row.get("timestamp"))
        if timestamp is None:
            continue
        if range_start <= timestamp <= range_end:
            ranged_rows.append(row)
    return ranged_rows, (range_start, range_end)


def build_timeline(
    rows: list[dict[str, Any]],
    *,
    verify_request_id: str,
    round_id: int,
) -> list[dict[str, Any]]:
    base_ts, _ = _get_time_range(rows)
    timeline: list[dict[str, Any]] = []
    for row in rows:
        details = _load_details(row.get("details_json"))
        timestamp = _parse_timestamp(row.get("timestamp"))
        event_lengths = _normalize_length_fields(row, details)
        timeline.append(
            {
                "timestamp": row.get("timestamp"),
                "relative_ms": (
                    round((timestamp - base_ts) * 1000, 3)
                    if timestamp is not None and base_ts is not None
                    else None
                ),
                "source": row.get("source"),
                "component": row.get("component"),
                "event": row.get("event"),
                "status": row.get("status"),
                "message": row.get("message"),
                "duration_ms": row.get("duration_ms") or None,
                "trace_scope": row.get("trace_scope"),
                "trace_key": row.get("trace_key"),
                "verify_request_id": row.get("verify_request_id"),
                "draft_round_id": row.get("draft_round_id"),
                "draft_sglang_rid": row.get("draft_sglang_rid"),
                "is_target_request_round": int(
                    _is_target_row(
                        row,
                        verify_request_id=verify_request_id,
                        round_id=round_id,
                    )
                ),
                "path": row.get("_path"),
                "lengths_json": json.dumps(event_lengths, ensure_ascii=False, sort_keys=True),
            }
        )
    timeline.sort(
        key=lambda row: (
            _parse_timestamp(row["timestamp"]) or float("inf"),
            row["component"] or "",
            row["event"] or "",
            row["path"] or "",
        )
    )
    return timeline


def build_summary(
    timeline: list[dict[str, Any]],
    *,
    verify_request_id: str,
    round_id: int,
    include_time_range: bool,
    range_start: float | None,
    range_end: float | None,
) -> dict[str, Any]:
    component_events: dict[str, list[str]] = {}
    for row in timeline:
        component = row.get("component") or "unknown"
        component_events.setdefault(component, []).append(row.get("event") or "")

    return {
        "verify_request_id": verify_request_id,
        "round_id": round_id,
        "trace_key": f"{verify_request_id}:{round_id}",
        "include_time_range": include_time_range,
        "event_count": len(timeline),
        "target_event_count": sum(int(row["is_target_request_round"]) for row in timeline),
        "components": sorted(component_events),
        "events_by_component": {
            component: events for component, events in sorted(component_events.items())
        },
        "first_timestamp": timeline[0]["timestamp"] if timeline else None,
        "last_timestamp": timeline[-1]["timestamp"] if timeline else None,
        "matched_range_start": (
            datetime.fromtimestamp(range_start).isoformat() if range_start is not None else None
        ),
        "matched_range_end": (
            datetime.fromtimestamp(range_end).isoformat() if range_end is not None else None
        ),
    }


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
        else trace_dir / "request_explanations"
    )
    all_rows = load_rows(trace_dir)
    selected_rows, (range_start, range_end) = collect_rows(
        all_rows,
        verify_request_id=args.verify_request_id,
        round_id=args.round_id,
        include_time_range=args.time_range,
    )
    timeline = build_timeline(
        selected_rows,
        verify_request_id=args.verify_request_id,
        round_id=args.round_id,
    )
    summary = build_summary(
        timeline,
        verify_request_id=args.verify_request_id,
        round_id=args.round_id,
        include_time_range=args.time_range,
        range_start=range_start,
        range_end=range_end,
    )

    mode_stem = "time_range" if args.time_range else "target_only"
    trace_stem = f"{args.verify_request_id}_round_{args.round_id}_{mode_stem}"
    timeline_path = output_dir / f"{trace_stem}_timeline.csv"
    summary_path = output_dir / f"{trace_stem}_summary.json"
    write_csv(timeline_path, timeline)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[decoupled-spec-explain] timeline_csv={timeline_path}")
    print(f"[decoupled-spec-explain] summary_json={summary_path}")


if __name__ == "__main__":
    main()
