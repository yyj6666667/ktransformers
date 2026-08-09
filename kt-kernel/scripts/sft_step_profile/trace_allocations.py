#!/usr/bin/env python3
"""Align PyTorch memory events to the raw-clock phase ledger.

Deep traces contain markers named ``kt.clock_sync.raw_ns=<value>``.  This tool
fits their trace timestamps to ``CLOCK_MONOTONIC_RAW`` and attributes each
allocator event to the narrowest containing host phase.  The allocator owns
the byte delta; phase attribution is explicitly temporal correlation.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


ANCHOR = re.compile(r"^kt\.clock_sync\.raw_ns=(\d+)$")
MEMORY_NAMES = {"[memory]", "[OutOfMemory]", "Memory Alloc", "Memory Free"}


def _read_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else path.open
    with opener("rt", encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{number}") from exc
    return rows


def _anchors(events: Iterable[dict[str, Any]]) -> list[tuple[float, int]]:
    anchors: list[tuple[float, int]] = []
    for event in events:
        match = ANCHOR.match(str(event.get("name", "")))
        if match and event.get("ts") is not None:
            anchors.append((float(event["ts"]) * 1_000.0, int(match.group(1))))
    return anchors


def fit_clock(anchors: list[tuple[float, int]]) -> tuple[float, float, float]:
    """Return ``raw_ns = scale * trace_ns + offset`` and max anchor error."""

    if not anchors:
        raise ValueError("deep trace has no kt.clock_sync.raw_ns anchor")
    if len(anchors) == 1:
        x, y = anchors[0]
        return 1.0, y - x, 0.0
    mean_x = sum(x for x, _ in anchors) / len(anchors)
    mean_y = sum(y for _, y in anchors) / len(anchors)
    denominator = sum((x - mean_x) ** 2 for x, _ in anchors)
    scale = (
        1.0
        if denominator == 0
        else sum((x - mean_x) * (y - mean_y) for x, y in anchors) / denominator
    )
    offset = mean_y - scale * mean_x
    error = max(abs((scale * x + offset) - y) for x, y in anchors)
    return scale, offset, error


def _phase_at(scopes: list[dict[str, Any]], timestamp_ns: int) -> dict[str, Any] | None:
    candidates = [
        row
        for row in scopes
        if int(row.get("start_ns", -1)) <= timestamp_ns < int(row.get("end_ns", -1))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: int(row.get("duration_ns", 0)))


def extract_allocations(
    trace: dict[str, Any], phase_rows: list[dict[str, Any]], *, source: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace_events = list(trace.get("traceEvents") or [])
    anchors = _anchors(trace_events)
    scale, offset, max_error = fit_clock(anchors)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid trace clock scale: {scale}")
    scopes = [row for row in phase_rows if row.get("record_type") == "scope"]
    identity = next((row for row in phase_rows if row.get("run") is not None), {})
    allocations: list[dict[str, Any]] = []
    for event in trace_events:
        name = str(event.get("name", ""))
        args = event.get("args") or {}
        if name not in MEMORY_NAMES and not (
            isinstance(args, dict)
            and any(
                key in args for key in ("Bytes", "Total Allocated", "Total Reserved")
            )
        ):
            continue
        if event.get("ts") is None:
            continue
        trace_ns = float(event["ts"]) * 1_000.0
        raw_ns = int(round(scale * trace_ns + offset))
        phase = _phase_at(scopes, raw_ns)
        row: dict[str, Any] = {
            "schema_version": 1,
            "record_type": "torch_allocation",
            "clock_name": "CLOCK_MONOTONIC_RAW",
            "monotonic_raw_ns": raw_ns,
            "trace_timestamp_ns": int(round(trace_ns)),
            "name": name,
            "source": source,
            "run": identity.get("run"),
            "rank": identity.get("rank"),
            "world_size": identity.get("world_size"),
            "allocator": "pytorch",
            "attribution_kind": "temporal_containment" if phase else "unattributed",
            "phase": phase.get("name") if phase else None,
            "scope_id": phase.get("scope_id") if phase else None,
            "step": phase.get("step") if phase else None,
            "microbatch": phase.get("microbatch") if phase else None,
            "pid": event.get("pid"),
            "tid": event.get("tid"),
            "args": args,
        }
        if isinstance(args, dict):
            for source_key, output_key in (
                ("Bytes", "delta_bytes"),
                ("Total Allocated", "total_allocated_bytes"),
                ("Total Reserved", "total_reserved_bytes"),
                ("Device Id", "device_id"),
                ("Device Type", "device_type"),
            ):
                if source_key in args:
                    row[output_key] = args[source_key]
        allocations.append(row)
    metadata = {
        "schema_version": 1,
        "record_type": "torch_allocation_metadata",
        "clock_name": "CLOCK_MONOTONIC_RAW",
        "source": source,
        "status": "ok",
        "anchor_count": len(anchors),
        "run": identity.get("run"),
        "rank": identity.get("rank"),
        "world_size": identity.get("world_size"),
        "clock_scale": scale,
        "clock_offset_ns": offset,
        "max_anchor_error_ns": max_error,
        "allocation_records": len(allocations),
        "attribution_note": "allocator byte ownership is measured; phase ownership is temporal containment",
    }
    if len(anchors) == 1:
        metadata["clock_fit_status"] = "single_anchor_assumed_unit_scale"
        metadata["max_anchor_error_ns"] = None
    else:
        metadata["clock_fit_status"] = "affine_fit"
    return metadata, allocations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--phases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        metadata, rows = extract_allocations(
            _read_json(args.trace), _read_jsonl(args.phases), source=str(args.trace)
        )
    except Exception as exc:
        metadata = {
            "schema_version": 1,
            "record_type": "torch_allocation_metadata",
            "clock_name": "CLOCK_MONOTONIC_RAW",
            "source": str(args.trace),
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }
        rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in (metadata, *rows):
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
