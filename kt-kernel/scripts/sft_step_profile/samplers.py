"""Read-only hardware and memory samplers for SFT step profiling.

The samplers in this module deliberately do not configure the host.  They do
not mount cgroup/resctrl filesystems, create monitor groups, invoke ``sudo``,
or change a CUDA driver setting.  Missing kernel files and optional Python
packages are represented as explicit ``unavailable`` records.
"""

from __future__ import annotations

import csv
import json
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, TextIO


SCHEMA_VERSION = 1
CLOCK_NAME = "CLOCK_MONOTONIC_RAW"
DEFAULT_HOST_INTERVAL_MS = 20
DEFAULT_NVML_INTERVAL_MS = 50
_AUTO = object()


def monotonic_raw_ns() -> int:
    """Return the process-independent Linux monotonic raw clock in ns."""

    clock_id = getattr(time, "CLOCK_MONOTONIC_RAW", None)
    if clock_id is None:
        raise RuntimeError("CLOCK_MONOTONIC_RAW is unavailable")
    return time.clock_gettime_ns(clock_id)


def _exception_text(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _read_text(path: Path) -> str:
    return path.read_text(encoding="ascii", errors="strict")


def _parse_integer(value: str) -> int:
    parsed = int(value.strip())
    if parsed < 0:
        raise ValueError(f"negative counter: {parsed}")
    return parsed


def _parse_proc_status(path: Path) -> tuple[dict[str, int], dict[str, str]]:
    wanted = {
        "VmRSS": "process_rss_bytes",
        "RssAnon": "process_anon_bytes",
        "RssFile": "process_file_bytes",
        "VmSwap": "process_swap_bytes",
    }
    values: dict[str, int] = {}
    errors: dict[str, str] = {}
    try:
        lines = _read_text(path).splitlines()
    except (OSError, UnicodeError) as exc:
        return values, {"process": _exception_text(exc)}

    for line in lines:
        key, separator, remainder = line.partition(":")
        if not separator or key not in wanted:
            continue
        fields = remainder.split()
        try:
            if not fields:
                raise ValueError("missing value")
            count = _parse_integer(fields[0])
            if len(fields) > 1 and fields[1].lower() != "kb":
                raise ValueError(f"unsupported unit: {fields[1]}")
            values[wanted[key]] = count * 1024
        except ValueError as exc:
            errors[wanted[key]] = _exception_text(exc)

    for output_key in wanted.values():
        if output_key not in values and output_key not in errors:
            errors[output_key] = "field unavailable"
    return values, errors


def _parse_key_value_file(path: Path) -> tuple[dict[str, int], str | None]:
    result: dict[str, int] = {}
    try:
        lines = _read_text(path).splitlines()
    except (OSError, UnicodeError) as exc:
        return result, _exception_text(exc)
    for line in lines:
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            result[fields[0]] = _parse_integer(fields[1])
        except ValueError:
            continue
    return result, None


def _source_status(
    value_count: int, errors: Mapping[str, str], expected: int | None = None
) -> str:
    if value_count == 0:
        return "unavailable"
    if errors or (expected is not None and value_count < expected):
        return "partial"
    return "ok"


def _overall_status(statuses: Iterable[str]) -> str:
    sampled = [status for status in statuses if status != "not_sampled"]
    if not sampled or all(status == "unavailable" for status in sampled):
        return "unavailable"
    if all(status == "ok" for status in sampled):
        return "ok"
    return "partial"


class MemoryReader:
    """Take JSON-safe process, cgroup, CUDA allocator, and NVML snapshots.

    ``snapshot`` never raises merely because a source is missing.  The caller
    can select a subset of sources for phase-boundary or periodic sampling.
    All timestamps use the same ``CLOCK_MONOTONIC_RAW`` domain as the timeline
    recorder.
    """

    def __init__(
        self,
        *,
        pid: int | None = None,
        rank: int = 0,
        run_id: str = "",
        proc_root: str | os.PathLike[str] = "/proc",
        cgroup_root: str | os.PathLike[str] = "/sys/fs/cgroup",
        torch_module: Any = _AUTO,
        torch_cuda_devices: Iterable[int] | None = None,
        nvml_module: Any = _AUTO,
        nvml_device_indices: Iterable[int] | None = None,
        clock: Callable[[], int] = monotonic_raw_ns,
    ) -> None:
        self.pid = int(os.getpid() if pid is None else pid)
        self.rank = int(rank)
        self.run_id = str(run_id)
        self.proc_root = Path(proc_root)
        self.cgroup_root = Path(cgroup_root)
        self._clock = clock
        self._torch_module_spec = torch_module
        self._torch_module: Any = None
        self._torch_checked = False
        self._torch_import_error: str | None = None
        self._torch_cuda_devices = (
            None
            if torch_cuda_devices is None
            else tuple(int(item) for item in torch_cuda_devices)
        )
        self._nvml_module_spec = nvml_module
        self._nvml_module: Any = None
        self._nvml_checked = False
        self._nvml_initialized = False
        self._nvml_error: str | None = None
        self._nvml_device_indices = (
            None
            if nvml_device_indices is None
            else tuple(int(item) for item in nvml_device_indices)
        )
        self._nvml_lock = threading.Lock()
        self._cgroup_path: Path | None = None
        self._cgroup_relative: str | None = None
        self._cgroup_error: str | None = None

    def _timestamp(self) -> tuple[int | None, str | None]:
        try:
            return int(self._clock()), None
        except Exception as exc:  # a fake clock is allowed in tests
            return None, _exception_text(exc)

    def _sample_process(self) -> tuple[dict[str, Any], dict[str, str], str]:
        values, errors = _parse_proc_status(self.proc_root / str(self.pid) / "status")
        status = _source_status(len(values), errors, expected=4)
        return values, errors, status

    def _resolve_cgroup(self) -> Path | None:
        if self._cgroup_path is not None or self._cgroup_error is not None:
            return self._cgroup_path
        cgroup_file = self.proc_root / str(self.pid) / "cgroup"
        try:
            lines = _read_text(cgroup_file).splitlines()
        except (OSError, UnicodeError) as exc:
            self._cgroup_error = _exception_text(exc)
            return None
        relative = None
        for line in lines:
            fields = line.split(":", 2)
            if len(fields) == 3 and fields[0] == "0" and fields[1] == "":
                relative = fields[2]
                break
        if relative is None:
            self._cgroup_error = "unified cgroup v2 entry unavailable"
            return None
        parts = Path(relative.lstrip("/")).parts
        if any(part in ("..", ".") for part in parts):
            self._cgroup_error = f"invalid cgroup path: {relative!r}"
            return None
        self._cgroup_relative = relative
        self._cgroup_path = self.cgroup_root.joinpath(*parts)
        return self._cgroup_path

    def _sample_cgroup(self) -> tuple[dict[str, Any], dict[str, str], str]:
        values: dict[str, Any] = {}
        errors: dict[str, str] = {}
        root = self._resolve_cgroup()
        if root is None:
            return (
                values,
                {"cgroup": self._cgroup_error or "cgroup unavailable"},
                "unavailable",
            )
        values["cgroup_path"] = self._cgroup_relative
        scalar_files = {
            "cgroup_current_bytes": root / "memory.current",
            "cgroup_swap_bytes": root / "memory.swap.current",
        }
        scalar_count = 0
        for key, path in scalar_files.items():
            try:
                values[key] = _parse_integer(_read_text(path))
                scalar_count += 1
            except (OSError, UnicodeError, ValueError) as exc:
                errors[key] = _exception_text(exc)

        stats, stat_error = _parse_key_value_file(root / "memory.stat")
        if stat_error is not None:
            errors["cgroup_memory_stat"] = stat_error
        for stat_key, output_key in (
            ("anon", "cgroup_anon_bytes"),
            ("file", "cgroup_file_bytes"),
        ):
            if stat_key in stats:
                values[output_key] = stats[stat_key]
                scalar_count += 1
            else:
                errors[output_key] = "field unavailable"
        status = _source_status(scalar_count, errors, expected=4)
        return values, errors, status

    def _ensure_torch(self) -> Any:
        if self._torch_checked:
            return self._torch_module
        self._torch_checked = True
        if self._torch_module_spec is None:
            self._torch_import_error = "torch sampling disabled"
            return None
        if self._torch_module_spec is not _AUTO:
            self._torch_module = self._torch_module_spec
            return self._torch_module
        try:
            import torch

            self._torch_module = torch
        except Exception as exc:
            self._torch_import_error = _exception_text(exc)
        return self._torch_module

    def _sample_torch_cuda(self) -> tuple[dict[str, Any], dict[str, str], str]:
        values: dict[str, Any] = {
            "torch_cuda_allocated_bytes_by_device": {},
            "torch_cuda_reserved_bytes_by_device": {},
        }
        errors: dict[str, str] = {}
        torch_module = self._ensure_torch()
        if torch_module is None:
            return (
                values,
                {"torch_cuda": self._torch_import_error or "torch unavailable"},
                "unavailable",
            )
        try:
            cuda = torch_module.cuda
            if not bool(cuda.is_available()):
                return values, {"torch_cuda": "CUDA unavailable"}, "unavailable"
            if self._torch_cuda_devices is None:
                devices = tuple(range(int(cuda.device_count())))
            else:
                devices = self._torch_cuda_devices
        except Exception as exc:
            return values, {"torch_cuda": _exception_text(exc)}, "unavailable"
        if not devices:
            return values, {"torch_cuda": "no CUDA devices"}, "unavailable"

        successful = 0
        for device in devices:
            key = str(device)
            try:
                allocated = int(cuda.memory_allocated(device))
                reserved = int(cuda.memory_reserved(device))
                values["torch_cuda_allocated_bytes_by_device"][key] = allocated
                values["torch_cuda_reserved_bytes_by_device"][key] = reserved
                successful += 1
            except Exception as exc:
                errors[f"torch_cuda_device_{device}"] = _exception_text(exc)
        return values, errors, _source_status(successful, errors, expected=len(devices))

    def _ensure_nvml_locked(self) -> Any:
        if self._nvml_checked:
            return self._nvml_module
        self._nvml_checked = True
        if self._nvml_module_spec is None:
            self._nvml_error = "NVML sampling disabled"
            return None
        if self._nvml_module_spec is not _AUTO:
            module = self._nvml_module_spec
        else:
            try:
                import pynvml as module
            except Exception as exc:
                self._nvml_error = _exception_text(exc)
                return None
        try:
            module.nvmlInit()
            self._nvml_initialized = True
            self._nvml_module = module
        except Exception as exc:
            self._nvml_error = _exception_text(exc)
        return self._nvml_module

    def _sample_nvml(self) -> tuple[dict[str, Any], dict[str, str], str]:
        values: dict[str, Any] = {
            "nvml_used_bytes_by_device": {},
            "nvml_total_bytes_by_device": {},
            "nvml_uuid_by_device": {},
        }
        errors: dict[str, str] = {}
        with self._nvml_lock:
            module = self._ensure_nvml_locked()
            if module is None:
                return (
                    values,
                    {"nvml": self._nvml_error or "NVML unavailable"},
                    "unavailable",
                )
            try:
                if self._nvml_device_indices is None:
                    devices = tuple(range(int(module.nvmlDeviceGetCount())))
                else:
                    devices = self._nvml_device_indices
            except Exception as exc:
                return values, {"nvml": _exception_text(exc)}, "unavailable"
            if not devices:
                return values, {"nvml": "no NVML devices"}, "unavailable"

            successful = 0
            for device in devices:
                key = str(device)
                try:
                    handle = module.nvmlDeviceGetHandleByIndex(device)
                    memory = module.nvmlDeviceGetMemoryInfo(handle)
                    values["nvml_used_bytes_by_device"][key] = int(memory.used)
                    values["nvml_total_bytes_by_device"][key] = int(memory.total)
                    successful += 1
                    try:
                        uuid = module.nvmlDeviceGetUUID(handle)
                        if isinstance(uuid, bytes):
                            uuid = uuid.decode("ascii", errors="replace")
                        values["nvml_uuid_by_device"][key] = str(uuid)
                    except Exception:
                        # UUID is useful provenance, but not required for a memory sample.
                        pass
                except Exception as exc:
                    errors[f"nvml_device_{device}"] = _exception_text(exc)
            return (
                values,
                errors,
                _source_status(successful, errors, expected=len(devices)),
            )

    def snapshot(
        self,
        *,
        include_process: bool = True,
        include_cgroup: bool = True,
        include_torch_cuda: bool = True,
        include_nvml: bool = True,
    ) -> dict[str, Any]:
        """Return one flat structured sample; unavailable sources never raise."""

        timestamp, clock_error = self._timestamp()
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "memory_sample",
            "clock_name": CLOCK_NAME,
            "monotonic_raw_ns": timestamp,
            "run_id": self.run_id,
            "rank": self.rank,
            "pid": self.pid,
            "process_status": "not_sampled",
            "cgroup_status": "not_sampled",
            "torch_cuda_status": "not_sampled",
            "nvml_status": "not_sampled",
            "errors": {},
        }
        if clock_error is not None:
            record["errors"]["clock"] = clock_error

        sources = (
            ("process", include_process, self._sample_process),
            ("cgroup", include_cgroup, self._sample_cgroup),
            ("torch_cuda", include_torch_cuda, self._sample_torch_cuda),
            ("nvml", include_nvml, self._sample_nvml),
        )
        for name, enabled, sample_fn in sources:
            if not enabled:
                continue
            try:
                values, errors, status = sample_fn()
            except Exception as exc:
                values, errors, status = {}, {name: _exception_text(exc)}, "unavailable"
            record.update(values)
            record[f"{name}_status"] = status
            record["errors"].update(errors)

        record["status"] = _overall_status(
            record[key]
            for key in (
                "process_status",
                "cgroup_status",
                "torch_cuda_status",
                "nvml_status",
            )
        )
        if clock_error is not None and record["status"] == "ok":
            record["status"] = "partial"
        return record

    def close(self) -> None:
        """Release this reader's NVML library reference, if it acquired one."""

        with self._nvml_lock:
            if not self._nvml_initialized or self._nvml_module is None:
                return
            try:
                self._nvml_module.nvmlShutdown()
            except Exception:
                pass
            finally:
                self._nvml_initialized = False

    def __enter__(self) -> "MemoryReader":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()


class MemorySampler:
    """Write periodic host/Torch and NVML snapshots to one JSONL stream."""

    def __init__(
        self,
        out_path: str | os.PathLike[str],
        rank: int,
        run_id: str,
        host_interval_ms: int = DEFAULT_HOST_INTERVAL_MS,
        nvml_interval_ms: int = DEFAULT_NVML_INTERVAL_MS,
        *,
        reader: MemoryReader | None = None,
        overwrite: bool = False,
        **reader_kwargs: Any,
    ) -> None:
        if host_interval_ms <= 0 or nvml_interval_ms <= 0:
            raise ValueError("sampling intervals must be positive")
        self.out_path = Path(out_path)
        self.rank = int(rank)
        self.run_id = str(run_id)
        self.host_interval_ms = int(host_interval_ms)
        self.nvml_interval_ms = int(nvml_interval_ms)
        self.reader = reader or MemoryReader(
            rank=self.rank, run_id=self.run_id, **reader_kwargs
        )
        self._owns_reader = reader is None
        self.overwrite = bool(overwrite)
        self._stream: TextIO | None = None
        self._stream_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._running = False
        self.records_written = 0
        self.background_errors: list[str] = []

    def _emit(self, record: Mapping[str, Any]) -> None:
        payload = dict(record)
        payload["run_id"] = self.run_id
        payload["rank"] = self.rank
        line = json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
        with self._stream_lock:
            if self._stream is None:
                return
            try:
                self._stream.write(line)
                self._stream.flush()
                self.records_written += 1
            except Exception as exc:
                self.background_errors.append(_exception_text(exc))
                self._stop_event.set()

    def emit(self, record: Mapping[str, Any]) -> None:
        """Write a boundary or annotation record through the sampler's FD.

        Periodic samples and caller-provided boundary records therefore share
        one stream offset and one lock; callers must not append to ``out_path``
        through a second file descriptor while the sampler is running.
        """

        self._emit(record)

    def _sample_host(self) -> dict[str, Any]:
        record = self.reader.snapshot(
            include_process=True,
            include_cgroup=True,
            include_torch_cuda=True,
            include_nvml=False,
        )
        record["sample_group"] = "host"
        return record

    def _sample_nvml(self) -> dict[str, Any]:
        record = self.reader.snapshot(
            include_process=False,
            include_cgroup=False,
            include_torch_cuda=False,
            include_nvml=True,
        )
        record["sample_group"] = "nvml"
        return record

    def _loop(
        self, name: str, interval_ms: int, sample_fn: Callable[[], dict[str, Any]]
    ) -> None:
        interval_seconds = interval_ms / 1000.0
        while not self._stop_event.is_set():
            try:
                self._emit(sample_fn())
            except Exception as exc:
                timestamp, _ = self.reader._timestamp()
                self._emit(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "record_type": "memory_sample",
                        "clock_name": CLOCK_NAME,
                        "monotonic_raw_ns": timestamp,
                        "sample_group": name,
                        "status": "unavailable",
                        "errors": {name: _exception_text(exc)},
                    }
                )
            self._stop_event.wait(interval_seconds)

    def start(self) -> "MemorySampler":
        with self._lifecycle_lock:
            if self._running:
                return self
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "w" if self.overwrite else "x"
            self._stream = self.out_path.open(mode, encoding="utf-8", buffering=1)
            self._stop_event.clear()
            self._running = True
            timestamp, clock_error = self.reader._timestamp()
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "record_type": "memory_sampler_metadata",
                "clock_name": CLOCK_NAME,
                "monotonic_raw_ns": timestamp,
                "run_id": self.run_id,
                "rank": self.rank,
                "pid": self.reader.pid,
                "host_interval_ms": self.host_interval_ms,
                "nvml_interval_ms": self.nvml_interval_ms,
                "status": "ok" if clock_error is None else "unavailable",
                "errors": {} if clock_error is None else {"clock": clock_error},
            }
            self._emit(metadata)
            self._threads = [
                threading.Thread(
                    target=self._loop,
                    args=("host", self.host_interval_ms, self._sample_host),
                    name=f"sft-memory-host-rank{self.rank}",
                    daemon=True,
                ),
                threading.Thread(
                    target=self._loop,
                    args=("nvml", self.nvml_interval_ms, self._sample_nvml),
                    name=f"sft-memory-nvml-rank{self.rank}",
                    daemon=True,
                ),
            ]
            for thread in self._threads:
                thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> dict[str, Any]:
        with self._lifecycle_lock:
            if not self._running:
                return self.summary()
            self._stop_event.set()
            threads = list(self._threads)
        deadline = time.monotonic() + max(0.0, timeout)
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        alive = [thread.name for thread in threads if thread.is_alive()]
        if alive:
            self.background_errors.append(f"sampler threads did not stop: {alive}")
        with self._lifecycle_lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            if self._owns_reader:
                self.reader.close()
            self._threads = []
            self._running = False
        return self.summary()

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": "memory_sampler_summary",
            "run_id": self.run_id,
            "rank": self.rank,
            "records_written": self.records_written,
            "background_errors": list(self.background_errors),
            "status": "ok" if not self.background_errors else "partial",
        }

    def __enter__(self) -> "MemorySampler":
        return self.start()

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.stop()


def _parse_perf_number(raw: str) -> int | float:
    value = raw.strip().replace(" ", "")
    if not value:
        raise ValueError("missing counter value")
    if any(character in value.lower() for character in (".", "e")):
        return float(value)
    return int(value)


def parse_perf_stat(
    text: str, *, delimiter: str = ",", timestamp_ns: int | None = None
) -> dict[str, Any]:
    """Parse ``perf stat -x<delimiter> --no-big-num`` output.

    Unsupported/not-counted events remain in the result with an explicit
    status.  Human-oriented (non-delimited) output is not guessed.
    """

    if len(delimiter) != 1:
        raise ValueError("perf delimiter must be one character")
    if timestamp_ns is None:
        try:
            timestamp_ns = monotonic_raw_ns()
            clock_error = None
        except Exception as exc:
            timestamp_ns = None
            clock_error = _exception_text(exc)
    else:
        clock_error = None

    events: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for line_number, fields in enumerate(
        csv.reader(text.splitlines(), delimiter=delimiter), start=1
    ):
        if not fields or not any(field.strip() for field in fields):
            continue
        if fields[0].lstrip().startswith("#"):
            continue
        fields = [field.strip() for field in fields]
        if len(fields) < 3 or not fields[2]:
            errors[f"line_{line_number}"] = "not perf stat delimited output"
            continue
        raw_value, unit, event = fields[:3]
        lowered = raw_value.lower().strip("<>")
        row: dict[str, Any] = {
            "event": event,
            "unit": unit or None,
            "raw_value": raw_value,
            "value": None,
            "status": "ok",
        }
        if lowered in ("not counted", "not supported"):
            row["status"] = "unavailable"
            row["reason"] = lowered
        else:
            try:
                row["value"] = _parse_perf_number(raw_value)
            except ValueError as exc:
                row["status"] = "unavailable"
                row["reason"] = _exception_text(exc)
        if len(fields) > 3 and fields[3]:
            row["run_time_raw"] = fields[3]
        if len(fields) > 4 and fields[4]:
            percent = fields[4].rstrip("%")
            try:
                row["percent_running"] = float(percent)
            except ValueError:
                row["percent_running_raw"] = fields[4]
        if len(fields) > 5 and fields[5]:
            row["metric_value_raw"] = fields[5]
        if len(fields) > 6 and fields[6]:
            row["metric_unit"] = fields[6]
        events.append(row)

    available = sum(event["status"] == "ok" for event in events)
    unavailable = len(events) - available
    if available == 0:
        status = "unavailable"
    elif unavailable or errors:
        status = "partial"
    else:
        status = "ok"
    if clock_error is not None:
        errors["clock"] = clock_error
        if status == "ok":
            status = "partial"
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "perf_stat",
        "clock_name": CLOCK_NAME,
        "monotonic_raw_ns": timestamp_ns,
        "status": status,
        "events": events,
        "errors": errors,
    }


class PerfStatReader:
    """Read and parse a pre-existing perf-stat output file without spawning perf."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        delimiter: str = ",",
        clock: Callable[[], int] = monotonic_raw_ns,
    ) -> None:
        self.path = Path(path)
        self.delimiter = delimiter
        self._clock = clock

    def snapshot(self) -> dict[str, Any]:
        try:
            timestamp = int(self._clock())
        except Exception as exc:
            timestamp = None
            clock_error = _exception_text(exc)
        else:
            clock_error = None
        try:
            contents = self.path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            errors = {"perf_stat": _exception_text(exc)}
            if clock_error is not None:
                errors["clock"] = clock_error
            return {
                "schema_version": SCHEMA_VERSION,
                "record_type": "perf_stat",
                "clock_name": CLOCK_NAME,
                "monotonic_raw_ns": timestamp,
                "status": "unavailable",
                "events": [],
                "errors": errors,
            }
        result = parse_perf_stat(
            contents, delimiter=self.delimiter, timestamp_ns=timestamp
        )
        result["path"] = str(self.path)
        if clock_error is not None:
            result["errors"]["clock"] = clock_error
            if result["status"] == "ok":
                result["status"] = "partial"
        return result


__all__ = [
    "CLOCK_NAME",
    "DEFAULT_HOST_INTERVAL_MS",
    "DEFAULT_NVML_INTERVAL_MS",
    "MemoryReader",
    "MemorySampler",
    "PerfStatReader",
    "monotonic_raw_ns",
    "parse_perf_stat",
]
