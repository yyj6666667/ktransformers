#!/usr/bin/env python3
"""Summarize phase memory deltas and add memory counters to the timeline."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable


SCALAR_METRICS = (
    "process_rss_bytes",
    "process_anon_bytes",
    "process_file_bytes",
    "process_swap_bytes",
    "cgroup_current_bytes",
    "cgroup_anon_bytes",
    "cgroup_file_bytes",
    "cgroup_swap_bytes",
)
MAP_METRICS = (
    "torch_cuda_allocated_bytes_by_device",
    "torch_cuda_reserved_bytes_by_device",
    "nvml_used_bytes_by_device",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{number}") from exc
    return rows


def metric_values(row: dict[str, Any]) -> dict[str, int]:
    values: dict[str, int] = {}
    for key in SCALAR_METRICS:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values[key] = int(value)
    for key in MAP_METRICS:
        mapping = row.get(key)
        if not isinstance(mapping, dict):
            continue
        total = 0
        for device, value in mapping.items():
            if isinstance(value, (int, float)):
                values[f"{key}.device_{device}"] = int(value)
                total += int(value)
        if mapping:
            values[f"{key}.total"] = total
    return values


def summarize_memory(
    phases: list[dict[str, Any]], samples: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    boundaries: dict[str, dict[str, dict[str, Any]]] = {}
    periodic: list[dict[str, Any]] = []
    for row in samples:
        boundary_id = row.get("boundary_id")
        boundary = row.get("boundary")
        if boundary_id is not None and boundary in ("begin", "end"):
            boundaries.setdefault(str(boundary_id), {})[str(boundary)] = row
        elif (
            row.get("record_type") == "memory_sample"
            and row.get("monotonic_raw_ns") is not None
        ):
            periodic.append(row)

    output: list[dict[str, Any]] = []
    for phase in phases:
        if phase.get("record_type") != "scope" or not phase.get("boundary_id"):
            continue
        pair = boundaries.get(str(phase["boundary_id"]), {})
        begin = pair.get("begin")
        end = pair.get("end")
        if begin is None or end is None:
            status = "partial"
        else:
            status = "ok"
        start_ns = int(phase.get("start_ns", 0))
        end_ns = int(phase.get("end_ns", start_ns))
        inside = [
            sample
            for sample in periodic
            if start_ns <= int(sample.get("monotonic_raw_ns", -1)) <= end_ns
        ]
        begin_values = metric_values(begin or {})
        end_values = metric_values(end or {})
        sampled_values = [metric_values(sample) for sample in inside]
        keys = set(begin_values) | set(end_values)
        for values in sampled_values:
            keys.update(values)
        metrics: dict[str, dict[str, Any]] = {}
        for key in sorted(keys):
            initial = begin_values.get(key)
            final = end_values.get(key)
            observed = [values[key] for values in sampled_values if key in values]
            peak_candidates = [
                value for value in (initial, final) if value is not None
            ] + observed
            metrics[key] = {
                "begin_bytes": initial,
                "end_bytes": final,
                "delta_bytes": final - initial
                if initial is not None and final is not None
                else None,
                "sampled_peak_bytes": max(peak_candidates) if peak_candidates else None,
                "samples_inside": len(observed),
                "measurement": "state_counter",
                "attribution": "phase_boundary_correlation",
            }
        output.append(
            {
                "schema_version": 1,
                "record_type": "phase_memory_summary",
                "scope_id": phase.get("scope_id"),
                "boundary_id": phase.get("boundary_id"),
                "step": phase.get("step"),
                "microbatch": phase.get("microbatch"),
                "phase": phase.get("name"),
                "start_ns": start_ns,
                "end_ns": end_ns,
                "status": status,
                "metrics": metrics,
            }
        )
    return output


def _counter_events(
    samples: Iterable[dict[str, Any]], *, origin_ns: int
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in samples:
        timestamp = row.get("monotonic_raw_ns")
        if timestamp is None or row.get("record_type") != "memory_sample":
            continue
        groups: dict[str, dict[str, float]] = {}
        values = metric_values(row)
        for key, value in values.items():
            if key.startswith("process_"):
                group = "CPU process memory"
            elif key.startswith("cgroup_"):
                group = "cgroup memory"
            elif key.startswith("torch_cuda_"):
                group = "PyTorch CUDA allocator"
            else:
                group = "NVML device memory"
            groups.setdefault(group, {})[key.replace("_bytes", "") + "_GiB"] = value / (
                1024**3
            )
        for name, args in groups.items():
            events.append(
                {
                    "name": name,
                    "cat": "sft_step_profile.memory",
                    "ph": "C",
                    "pid": int(row.get("pid", 0)),
                    "tid": 0,
                    "ts": (int(timestamp) - origin_ns) / 1_000.0,
                    "args": args,
                }
            )
    return events


def write_reports(rank_dir: Path) -> None:
    phases = read_jsonl(rank_dir / "phase_events.jsonl")
    samples = read_jsonl(rank_dir / "memory_samples.jsonl")
    summaries = summarize_memory(phases, samples)
    payload = {
        "schema_version": 1,
        "record_type": "phase_memory_summary_collection",
        "rank_dir": str(rank_dir),
        "attribution_note": "RSS/cgroup/NVML and allocator state deltas are temporally correlated, not allocation ownership",
        "scopes": summaries,
    }
    destination = rank_dir / "memory_phase_summary.json"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, destination)

    fields = [
        "step",
        "microbatch",
        "phase",
        "scope_id",
        "status",
        "duration_ns",
        "process_rss_delta_bytes",
        "cgroup_current_delta_bytes",
        "torch_cuda_allocated_delta_bytes",
        "nvml_used_sampled_peak_bytes",
    ]
    csv_path = rank_dir / "memory_phase_summary.csv"
    csv_tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with csv_tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            metrics = row["metrics"]
            writer.writerow(
                {
                    "step": row["step"],
                    "microbatch": row["microbatch"],
                    "phase": row["phase"],
                    "scope_id": row["scope_id"],
                    "status": row["status"],
                    "duration_ns": row["end_ns"] - row["start_ns"],
                    "process_rss_delta_bytes": (
                        metrics.get("process_rss_bytes") or {}
                    ).get("delta_bytes"),
                    "cgroup_current_delta_bytes": (
                        metrics.get("cgroup_current_bytes") or {}
                    ).get("delta_bytes"),
                    "torch_cuda_allocated_delta_bytes": (
                        metrics.get("torch_cuda_allocated_bytes_by_device.total") or {}
                    ).get("delta_bytes"),
                    "nvml_used_sampled_peak_bytes": (
                        metrics.get("nvml_used_bytes_by_device.total") or {}
                    ).get("sampled_peak_bytes"),
                }
            )
    os.replace(csv_tmp, csv_path)

    trace_path = rank_dir / "timeline.trace.json"
    if trace_path.exists():
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        origin_ns = int(trace.get("clock_origin_ns", 0))
        trace.setdefault("traceEvents", []).extend(
            _counter_events(samples, origin_ns=origin_ns)
        )
        trace["memory_counter_note"] = (
            "state samples; timestamps use CLOCK_MONOTONIC_RAW"
        )
        trace_tmp = trace_path.with_suffix(trace_path.suffix + ".tmp")
        trace_tmp.write_text(json.dumps(trace, separators=(",", ":")) + "\n")
        os.replace(trace_tmp, trace_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rank_dir", type=Path)
    args = parser.parse_args()
    write_reports(args.rank_dir)


if __name__ == "__main__":
    main()
