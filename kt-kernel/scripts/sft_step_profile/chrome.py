"""Convert rank-local timeline JSONL records to Chrome trace JSON."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
    return records


def _resolution_map(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    patches: dict[str, dict[str, Any]] = {}
    for record in records:
        if (
            record.get("record_type") == "cuda_resolution"
            and record.get("scope_id") is not None
        ):
            patches[str(record["scope_id"])] = record
    return patches


def _event_args(record: dict[str, Any], patch: dict[str, Any] | None) -> dict[str, Any]:
    excluded = {
        "name",
        "pid",
        "tid",
        "start_ns",
        "end_ns",
        "duration_ns",
        "record_type",
    }
    args = {key: value for key, value in record.items() if key not in excluded}
    if patch is not None:
        for key in (
            "cuda_elapsed_ns",
            "cuda_clock_regression_ns",
            "cuda_resolve_error",
        ):
            if patch.get(key) is not None:
                args[key] = patch[key]
        args["cuda_pending"] = False
    return args


def build_chrome_trace(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build a Chrome trace payload while preserving raw timestamps in args."""
    materialized = list(records)
    timed = [record for record in materialized if record.get("start_ns") is not None]
    origin_ns = min((int(record["start_ns"]) for record in timed), default=0)
    resolutions = _resolution_map(materialized)
    events: list[dict[str, Any]] = []
    process_metadata: set[tuple[int, int, str]] = set()
    thread_metadata: set[tuple[int, int]] = set()

    for record in timed:
        if record.get("record_type") == "cuda_resolution":
            continue
        pid = int(record.get("pid", 0))
        tid = int(record.get("tid", 0))
        rank = int(record.get("rank", 0))
        run = str(record.get("run", "unknown"))
        process_metadata.add((pid, rank, run))
        thread_metadata.add((pid, tid))
        patch = resolutions.get(str(record.get("scope_id")))
        common = {
            "name": str(record.get("name", record.get("record_type", "event"))),
            "cat": f"sft_step_profile.{record.get('record_type', 'event')}",
            "pid": pid,
            "tid": tid,
            "ts": (int(record["start_ns"]) - origin_ns) / 1_000.0,
            "args": {
                "raw_start_ns": int(record["start_ns"]),
                "raw_end_ns": int(record.get("end_ns", record["start_ns"])),
                **_event_args(record, patch),
            },
        }
        if record.get("record_type") == "scope":
            common.update(
                {"ph": "X", "dur": int(record.get("duration_ns", 0)) / 1_000.0}
            )
        else:
            common.update({"ph": "i", "s": "t"})
        events.append(common)

    for pid, rank, run in sorted(process_metadata):
        events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {"name": f"{run} rank {rank}"},
            }
        )
        events.append(
            {
                "name": "process_sort_index",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {"sort_index": rank},
            }
        )
    for pid, tid in sorted(thread_metadata):
        events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": tid,
                "args": {"name": f"tid {tid}"},
            }
        )
    return {
        "displayTimeUnit": "ms",
        "clock": "CLOCK_MONOTONIC_RAW",
        "clock_origin_ns": origin_ns,
        "traceEvents": events,
    }


def convert_jsonl_to_chrome_trace(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Convert one rank-local JSONL stream and atomically write its trace."""
    source = Path(input_path)
    destination = (
        Path(output_path)
        if output_path is not None
        else source.with_suffix(".trace.json")
    )
    payload = build_chrome_trace(read_jsonl(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)
    return destination


__all__ = ["build_chrome_trace", "convert_jsonl_to_chrome_trace", "read_jsonl"]
