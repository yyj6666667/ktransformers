"""Rank-isolated JSONL artifact writer for the SFT step timeline."""

from __future__ import annotations

import json
import os
import threading
import csv
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


class JsonlTimelineWriter:
    """Write events and step summaries without sharing files across ranks."""

    def __init__(
        self, out_dir: str | os.PathLike[str], run_id: str, rank: int, pid: int
    ) -> None:
        self.out_dir = Path(out_dir) / f"rank_{rank}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.out_dir / "phase_events.jsonl"
        self.steps_path = self.out_dir / "step_summary.jsonl"
        self.steps_json_path = self.out_dir / "step_summary.json"
        self.steps_csv_path = self.out_dir / "step_summary.csv"
        self.index_path = self.out_dir / "step_index.json"
        self.chrome_path = self.out_dir / "timeline.trace.json"
        self._events = self.events_path.open("x", encoding="utf-8", buffering=1)
        self._steps = self.steps_path.open("x", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._closed = False
        self._event_records = 0
        self._step_records = 0
        self._step_index: list[dict[str, Any]] = []
        self._step_summaries: list[dict[str, Any]] = []
        self.run_id = run_id
        self.rank = int(rank)
        self.pid = int(pid)

    @staticmethod
    def _write_line(stream: Any, record: dict[str, Any]) -> int:
        offset = stream.tell()
        stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
        return offset

    def write_event(self, record: dict[str, Any]) -> int:
        with self._lock:
            if self._closed:
                raise RuntimeError("timeline writer is closed")
            offset = self._write_line(self._events, record)
            self._event_records += 1
            return offset

    def write_step(self, record: dict[str, Any], event_offsets: list[int]) -> int:
        with self._lock:
            if self._closed:
                raise RuntimeError("timeline writer is closed")
            offset = self._write_line(self._steps, record)
            self._step_records += 1
            self._step_summaries.append(record)
            self._step_index.append(
                {
                    "step": record.get("step"),
                    "status": record.get("status"),
                    "warmup": record.get("warmup"),
                    "events_begin_offset": min(event_offsets)
                    if event_offsets
                    else None,
                    "events_end_offset": max(event_offsets) if event_offsets else None,
                    "summary_offset": offset,
                }
            )
            return offset

    def flush(self, *, pending_cuda: int = 0) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                return self.index_payload(pending_cuda=pending_cuda)
            self._events.flush()
            self._steps.flush()
            self._write_step_exports()
            payload = self.index_payload(pending_cuda=pending_cuda)
            _atomic_json(self.index_path, payload)
            return payload

    def index_payload(self, *, pending_cuda: int = 0) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": "timeline_index",
            "run": self.run_id,
            "rank": self.rank,
            "pid": self.pid,
            "event_records": self._event_records,
            "step_records": self._step_records,
            "pending_cuda_events": int(pending_cuda),
            "artifacts": {
                "events": self.events_path.name,
                "steps_jsonl": self.steps_path.name,
                "steps_json": self.steps_json_path.name,
                "steps_csv": self.steps_csv_path.name,
                "chrome_trace": self.chrome_path.name,
            },
            "steps": list(self._step_index),
        }

    def _write_step_exports(self) -> None:
        _atomic_json(
            self.steps_json_path,
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "step_summary_collection",
                "run": self.run_id,
                "rank": self.rank,
                "pid": self.pid,
                "steps": self._step_summaries,
            },
        )
        phase_names = sorted(
            {
                phase
                for summary in self._step_summaries
                for phase in (summary.get("phases") or {})
            }
        )
        fixed = [
            "run",
            "rank",
            "pid",
            "step",
            "status",
            "warmup",
            "start_ns",
            "end_ns",
            "duration_ns",
            "exclusive_ns",
            "microbatch_count",
            "tokens",
            "tokens_per_second",
            "scope_count",
            "accounted_ns",
            "leaf_accounted_ns",
            "other_ns",
            "accounting_error_ns",
            "clock_anomaly_count",
            "cuda_pending_count",
        ]
        phase_fields = [
            f"{phase}.{metric}"
            for phase in phase_names
            for metric in ("calls", "inclusive_ns", "exclusive_ns")
        ]
        temporary = self.steps_csv_path.with_suffix(self.steps_csv_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*fixed, *phase_fields])
            writer.writeheader()
            for summary in self._step_summaries:
                row = {key: summary.get(key) for key in fixed}
                for phase, stats in (summary.get("phases") or {}).items():
                    for metric in ("calls", "inclusive_ns", "exclusive_ns"):
                        row[f"{phase}.{metric}"] = stats.get(metric)
                writer.writerow(row)
        os.replace(temporary, self.steps_csv_path)

    def close(self, *, pending_cuda: int = 0) -> dict[str, Any]:
        payload = self.flush(pending_cuda=pending_cuda)
        with self._lock:
            if not self._closed:
                self._events.close()
                self._steps.close()
                self._closed = True
        return payload


__all__ = ["JsonlTimelineWriter", "SCHEMA_VERSION"]
