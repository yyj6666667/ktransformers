"""Read-only resctrl Memory Bandwidth Monitoring (MBM) snapshots.

This module only consumes an already configured resctrl hierarchy.  It never
mounts resctrl, creates control/monitor groups, writes task IDs, or changes
resource controls.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from .samplers import CLOCK_NAME, SCHEMA_VERSION, monotonic_raw_ns


def _exception_text(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _safe_relative_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise ValueError(f"resctrl group must be a safe relative path: {value!r}")
    return path


class ResctrlMbmReader:
    """Sample MBM counters from a pre-existing resctrl monitor group."""

    def __init__(
        self,
        *,
        root: str | os.PathLike[str] = "/sys/fs/resctrl",
        group: str | os.PathLike[str] = "",
        rank: int = 0,
        run_id: str = "",
        clock: Callable[[], int] = monotonic_raw_ns,
    ) -> None:
        self.root = Path(root)
        self.group = _safe_relative_path(group) if str(group) else Path()
        self.rank = int(rank)
        self.run_id = str(run_id)
        self._clock = clock
        self._previous_total: dict[str, int] = {}
        self._previous_local: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def group_path(self) -> Path:
        return self.root / self.group

    def _timestamp(self) -> tuple[int | None, str | None]:
        try:
            return int(self._clock()), None
        except Exception as exc:
            return None, _exception_text(exc)

    @staticmethod
    def _counter(path: Path) -> int:
        raw = path.read_text(encoding="ascii", errors="strict").strip()
        value = int(raw)
        if value < 0:
            raise ValueError(f"negative MBM counter: {value}")
        return value

    @staticmethod
    def _delta(
        current: dict[str, int], previous: dict[str, int]
    ) -> tuple[dict[str, int], list[str]]:
        deltas: dict[str, int] = {}
        reset_domains: list[str] = []
        for domain, value in current.items():
            if domain not in previous:
                continue
            prior = previous[domain]
            if value >= prior:
                deltas[domain] = value - prior
            else:
                reset_domains.append(domain)
        return deltas, reset_domains

    def snapshot(self) -> dict[str, Any]:
        """Return raw and since-last-sample MBM byte counters without raising."""

        with self._lock:
            timestamp, clock_error = self._timestamp()
            result: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "record_type": "resctrl_mbm_sample",
                "clock_name": CLOCK_NAME,
                "monotonic_raw_ns": timestamp,
                "run_id": self.run_id,
                "rank": self.rank,
                "resctrl_group_path": str(self.group_path),
                "resctrl_mbm_total_bytes_by_domain": {},
                "resctrl_mbm_local_bytes_by_domain": {},
                "resctrl_mbm_total_delta_bytes_by_domain": {},
                "resctrl_mbm_local_delta_bytes_by_domain": {},
                "status": "unavailable",
                "errors": {},
            }
            if clock_error is not None:
                result["errors"]["clock"] = clock_error

            mon_data = self.group_path / "mon_data"
            try:
                domain_paths = sorted(
                    path
                    for path in mon_data.iterdir()
                    if path.is_dir() and path.name.startswith("mon_L3_")
                )
            except OSError as exc:
                result["errors"]["resctrl"] = _exception_text(exc)
                return result
            if not domain_paths:
                result["errors"]["resctrl"] = "no mon_L3_* domains available"
                return result

            total: dict[str, int] = {}
            local: dict[str, int] = {}
            counter_errors: dict[str, str] = {}
            for domain_path in domain_paths:
                domain = domain_path.name.removeprefix("mon_L3_")
                for filename, output in (
                    ("mbm_total_bytes", total),
                    ("mbm_local_bytes", local),
                ):
                    try:
                        output[domain] = self._counter(domain_path / filename)
                    except (OSError, UnicodeError, ValueError) as exc:
                        counter_errors[f"{domain}:{filename}"] = _exception_text(exc)

            total_delta, total_resets = self._delta(total, self._previous_total)
            local_delta, local_resets = self._delta(local, self._previous_local)
            self._previous_total = dict(total)
            self._previous_local = dict(local)

            result["resctrl_mbm_total_bytes_by_domain"] = total
            result["resctrl_mbm_local_bytes_by_domain"] = local
            result["resctrl_mbm_total_delta_bytes_by_domain"] = total_delta
            result["resctrl_mbm_local_delta_bytes_by_domain"] = local_delta
            if total_resets:
                result["resctrl_mbm_total_reset_domains"] = total_resets
            if local_resets:
                result["resctrl_mbm_local_reset_domains"] = local_resets
            result["errors"].update(counter_errors)

            successful = len(total) + len(local)
            expected = len(domain_paths) * 2
            if successful == 0:
                result["status"] = "unavailable"
            elif successful < expected or counter_errors or clock_error is not None:
                result["status"] = "partial"
            else:
                result["status"] = "ok"
            return result

    def reset_baseline(self) -> None:
        with self._lock:
            self._previous_total.clear()
            self._previous_local.clear()


class HardwareCounterSampler:
    """Periodically write read-only resctrl MBM counters to JSONL.

    The sampler consumes a monitor group prepared by an administrator or job
    launcher.  It intentionally has no API for mounting resctrl or assigning
    tasks.  Existing ``perf stat -x`` output can be ingested separately with
    :class:`sft_step_profile.samplers.PerfStatReader`.
    """

    def __init__(
        self,
        out_path: str | os.PathLike[str],
        rank: int,
        run_id: str,
        resctrl_group: str | os.PathLike[str] | None = None,
        interval_ms: int = 100,
        *,
        resctrl_root: str | os.PathLike[str] | None = None,
        reader: ResctrlMbmReader | None = None,
        overwrite: bool = False,
    ) -> None:
        if interval_ms <= 0:
            raise ValueError("hardware counter interval must be positive")
        if resctrl_group is None:
            resctrl_group = os.environ.get("KT_STEP_PROFILE_RESCTRL_GROUP", "")
        if resctrl_root is None:
            resctrl_root = os.environ.get(
                "KT_STEP_PROFILE_RESCTRL_ROOT", "/sys/fs/resctrl"
            )
        self.out_path = Path(out_path)
        self.rank = int(rank)
        self.run_id = str(run_id)
        self.interval_ms = int(interval_ms)
        self.reader = reader or ResctrlMbmReader(
            root=resctrl_root,
            group=resctrl_group,
            rank=self.rank,
            run_id=self.run_id,
        )
        self.overwrite = bool(overwrite)
        self._stream: TextIO | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._running = False
        self.records_written = 0
        self.background_errors: list[str] = []

    def _emit(self, record: dict[str, Any]) -> None:
        record["run_id"] = self.run_id
        record["rank"] = self.rank
        line = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        with self._lock:
            if self._stream is None:
                return
            try:
                self._stream.write(line)
                self._stream.flush()
                self.records_written += 1
            except Exception as exc:
                self.background_errors.append(_exception_text(exc))
                self._stop_event.set()

    def _loop(self) -> None:
        interval_seconds = self.interval_ms / 1000.0
        while not self._stop_event.is_set():
            try:
                self._emit(self.reader.snapshot())
            except Exception as exc:
                try:
                    timestamp = monotonic_raw_ns()
                except Exception:
                    timestamp = None
                self._emit(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "record_type": "resctrl_mbm_sample",
                        "clock_name": CLOCK_NAME,
                        "monotonic_raw_ns": timestamp,
                        "status": "unavailable",
                        "errors": {"resctrl": _exception_text(exc)},
                    }
                )
            self._stop_event.wait(interval_seconds)

    def start(self) -> "HardwareCounterSampler":
        if self._running:
            return self
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.out_path.open(
            "w" if self.overwrite else "x", encoding="utf-8", buffering=1
        )
        self._stop_event.clear()
        self._running = True
        try:
            timestamp = monotonic_raw_ns()
            clock_error = None
        except Exception as exc:
            timestamp = None
            clock_error = _exception_text(exc)
        self._emit(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "hardware_counter_metadata",
                "clock_name": CLOCK_NAME,
                "monotonic_raw_ns": timestamp,
                "interval_ms": self.interval_ms,
                "resctrl_group_path": str(self.reader.group_path),
                "status": "ok" if clock_error is None else "unavailable",
                "errors": {} if clock_error is None else {"clock": clock_error},
            }
        )
        self._thread = threading.Thread(
            target=self._loop,
            name=f"sft-hardware-counters-rank{self.rank}",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> dict[str, Any]:
        if not self._running:
            return self.summary()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(max(0.0, timeout))
            if self._thread.is_alive():
                self.background_errors.append(
                    f"sampler thread did not stop: {self._thread.name}"
                )
        with self._lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
        self._thread = None
        self._running = False
        return self.summary()

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": "hardware_counter_summary",
            "run_id": self.run_id,
            "rank": self.rank,
            "records_written": self.records_written,
            "background_errors": list(self.background_errors),
            "status": "ok" if not self.background_errors else "partial",
        }

    def __enter__(self) -> "HardwareCounterSampler":
        return self.start()

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.stop()


__all__ = ["HardwareCounterSampler", "ResctrlMbmReader"]
