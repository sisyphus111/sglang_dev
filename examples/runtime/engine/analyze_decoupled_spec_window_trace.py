#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


CORE_EVENT_ORDER = [
    "window_send",
    "window_enqueue",
    "window_head",
    "window_done",
    "window_result_send",
    "window_result_recv",
]

DISPLAY_COLUMNS = [
    "kind",
    "role",
    "begin_pos",
    "end_pos",
    "bs",
    "latency_ms",
    "send_to_enqueue_ms",
    "send_to_head_ms",
    "enqueue_to_head_ms",
    "send_to_done_ms",
    "enqueue_to_done_ms",
    "head_to_done_ms",
    "done_to_send_ms",
    "send_to_recv_ms",
    "enqueue_to_recv_ms",
    "head_to_recv_ms",
    "done_to_recv_ms",
    "result_send_to_recv_ms",
    "reason",
    "overwrite_from",
    "rollback_tokens",
    "append_tokens",
    "new_output_len",
    "extra",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one decoupled-spec draft window timeline from CSV traces."
    )
    parser.add_argument(
        "--trace-csv",
        action="append",
        required=True,
        help="Trace CSV path. Repeat this flag to merge multiple shard files.",
    )
    parser.add_argument("--start-pos", type=int, required=True)
    parser.add_argument("--end-pos", type=int, required=True)
    parser.add_argument(
        "--rid",
        default=None,
        help="Optional request id. Required when multiple requests share one window.",
    )
    return parser.parse_args()


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return int(text)


def _load_rows(paths: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path_text in paths:
        path = Path(path_text)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for line_index, raw_row in enumerate(reader, start=2):
                wall_time_ns = _parse_int(raw_row.get("wall_time_ns"))
                if wall_time_ns is None:
                    continue
                row = dict(raw_row)
                row["_wall_time_ns"] = wall_time_ns
                row["_source"] = str(path)
                row["_line"] = line_index
                rows.append(row)
    rows.sort(
        key=lambda row: (
            int(row["_wall_time_ns"]),
            str(row.get("event") or ""),
            str(row.get("role") or ""),
        )
    )
    return rows


def _matching_window_rows(
    rows: list[dict[str, object]], start_pos: int, end_pos: int
) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    for row in rows:
        if _parse_int(row.get("begin_pos")) != start_pos:
            continue
        if _parse_int(row.get("end_pos")) != end_pos:
            continue
        matches.append(row)
    return matches


def _resolve_target_rid(
    matches: list[dict[str, object]], requested_rid: str | None
) -> str:
    if requested_rid is not None:
        return requested_rid

    candidate_rids = sorted(
        {
            str(row.get("rid") or "").strip()
            for row in matches
            if str(row.get("rid") or "").strip()
        }
    )
    if not candidate_rids:
        raise SystemExit("No matching rows contain rid, cannot disambiguate window.")
    if len(candidate_rids) == 1:
        return candidate_rids[0]

    print("Multiple request ids match this window. Please rerun with --rid.")
    print("Candidates:")
    for rid in candidate_rids:
        print(rid)
    raise SystemExit(1)


def _filter_rows_for_rid(
    rows: list[dict[str, object]],
    *,
    rid: str,
    start_pos: int,
    end_pos: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    core_rows = [
        row
        for row in rows
        if str(row.get("rid") or "") == rid
        and _parse_int(row.get("begin_pos")) == start_pos
        and _parse_int(row.get("end_pos")) == end_pos
        and str(row.get("event") or "") in CORE_EVENT_ORDER
    ]
    if not core_rows:
        raise SystemExit("No core timeline events found for the requested window.")

    start_ns = min(int(row["_wall_time_ns"]) for row in core_rows)
    end_ns = max(int(row["_wall_time_ns"]) for row in core_rows)

    rewrite_rows = [
        row
        for row in rows
        if str(row.get("rid") or "") == rid
        and str(row.get("event") or "") == "output_rewrite"
    ]
    forward_rows = [
        row
        for row in rows
        if str(row.get("event") or "") == "forward"
        and start_ns <= int(row["_wall_time_ns"]) <= end_ns
    ]

    return (
        sorted(core_rows, key=lambda row: int(row["_wall_time_ns"])),
        sorted(rewrite_rows, key=lambda row: int(row["_wall_time_ns"])),
        sorted(forward_rows, key=lambda row: int(row["_wall_time_ns"])),
    )


def _format_row_details(row: dict[str, object]) -> str:
    parts = []
    for key in DISPLAY_COLUMNS:
        value = str(row.get(key) or "").strip()
        if value:
            parts.append(f"{key}={value}")
    parts.append(f"source={row['_source']}:{row['_line']}")
    return " ".join(parts)


def _print_core_timeline(core_rows: list[dict[str, object]]) -> None:
    print("Core timeline")
    first_by_event: dict[str, dict[str, object]] = {}
    for event_name in CORE_EVENT_ORDER:
        event_rows = [
            row for row in core_rows if str(row.get("event") or "") == event_name
        ]
        if not event_rows:
            print(f"- {event_name}: <missing>")
            continue
        first_by_event[event_name] = event_rows[0]
        for row in event_rows:
            print(
                f"- {event_name}: wall_time_ns={row['_wall_time_ns']} "
                f"{_format_row_details(row)}"
            )

    print("Stage gaps")
    for prev_event, next_event in zip(CORE_EVENT_ORDER, CORE_EVENT_ORDER[1:]):
        prev_row = first_by_event.get(prev_event)
        next_row = first_by_event.get(next_event)
        if prev_row is None or next_row is None:
            continue
        gap_ms = (
            int(next_row["_wall_time_ns"]) - int(prev_row["_wall_time_ns"])
        ) / 1_000_000.0
        print(f"- {prev_event} -> {next_event}: {gap_ms:.3f} ms")


def _print_related_events(
    rewrite_rows: list[dict[str, object]], forward_rows: list[dict[str, object]]
) -> None:
    print("Related events")
    if not rewrite_rows and not forward_rows:
        print("- <none>")
        return

    for row in rewrite_rows:
        print(
            f"- output_rewrite: wall_time_ns={row['_wall_time_ns']} "
            f"{_format_row_details(row)}"
        )
    for row in forward_rows:
        print(
            f"- forward: wall_time_ns={row['_wall_time_ns']} "
            f"{_format_row_details(row)}"
        )


def main() -> int:
    args = parse_args()
    rows = _load_rows(args.trace_csv)
    matches = _matching_window_rows(rows, args.start_pos, args.end_pos)
    if not matches:
        print(
            f"No rows found for window ({args.start_pos}, {args.end_pos}).",
            file=sys.stderr,
        )
        return 1

    rid = _resolve_target_rid(matches, args.rid)
    core_rows, rewrite_rows, forward_rows = _filter_rows_for_rid(
        rows,
        rid=rid,
        start_pos=args.start_pos,
        end_pos=args.end_pos,
    )

    print(f"RID: {rid}")
    print(f"Window: ({args.start_pos}, {args.end_pos})")
    _print_core_timeline(core_rows)
    _print_related_events(rewrite_rows, forward_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
