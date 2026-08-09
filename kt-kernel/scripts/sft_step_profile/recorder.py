"""Optimizer-step phase timeline recorder.

The recorder is deliberately independent of Trainer, Accelerate, and KT.  It
records host scopes on ``CLOCK_MONOTONIC_RAW`` and can attach non-blocking CUDA
event pairs.  Framework integration belongs in a separate adapter.
"""

from __future__ import annotations

import os
import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .clock import (
    CudaEventBackend,
    MonotonicRawClock,
    TimelineClock,
    TorchCudaEventBackend,
)
from .writer import SCHEMA_VERSION, JsonlTimelineWriter


class ProfileMode(str, Enum):
    OFF = "off"
    PHASE = "phase"
    DEEP = "deep"


_AUTO_MICROBATCH = object()


def monotonic_raw_ns() -> int:
    """Return one ``CLOCK_MONOTONIC_RAW`` timestamp."""
    return MonotonicRawClock().now_ns()


@dataclass
class _OpenScope:
    scope_id: str
    parent_id: str | None
    name: str
    step: int | None
    microbatch: int | None
    pid: int
    tid: int
    start_ns: int
    child_ns: int = 0
    cuda_start: Any | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class _PendingCuda:
    record: dict[str, Any]
    start: Any
    end: Any
    written: bool = False


class _NullScope(AbstractContextManager["_NullScope"]):
    scope_id = None

    def __enter__(self) -> "_NullScope":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False


class _ScopeContext(AbstractContextManager["_ScopeContext"]):
    def __init__(self, recorder: "TimelineRecorder", scope_id: str) -> None:
        self._recorder = recorder
        self.scope_id = scope_id

    def __enter__(self) -> "_ScopeContext":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        status = "error" if exc_type is not None else "ok"
        self._recorder._close_scope(self.scope_id, status=status, error=exc_value)
        return False


class TimelineRecorder:
    """Record exclusive, nested phase timings for optimizer-step windows.

    Args:
        out_dir: Artifact directory. No directory is created in ``off`` mode.
        run_id: Stable identifier shared by every rank in one training run.
        rank: Distributed rank. Each rank writes distinct files.
        world_size: Number of ranks, recorded as metadata only.
        mode: ``off``, ``phase``, or ``deep``.
        warmup_steps: Number of completed steps marked as warm-up in summaries.
        clock: Injectable clock for tests; production defaults to
            :class:`MonotonicRawClock`.
        cuda_backend: Optional non-blocking CUDA-event backend. CUDA timing is
            recorded only for scopes with ``cuda_timing=True`` in ``deep`` mode.
    """

    def __init__(
        self,
        out_dir: str | os.PathLike[str],
        run_id: str,
        rank: int = 0,
        world_size: int = 1,
        mode: str | ProfileMode = ProfileMode.OFF,
        warmup_steps: int = 0,
        *,
        clock: TimelineClock | None = None,
        cuda_backend: CudaEventBackend | None = None,
        enable_cuda_events: bool = True,
        pid: int | None = None,
    ) -> None:
        self.mode = (
            mode
            if isinstance(mode, ProfileMode)
            else ProfileMode(str(mode).strip().lower())
        )
        self.out_dir = Path(out_dir)
        self.run_id = str(run_id)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.warmup_steps = int(warmup_steps)
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        if self.world_size <= 0 or self.rank >= self.world_size:
            raise ValueError(
                f"invalid rank/world_size pair: {self.rank}/{self.world_size}"
            )
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")

        self.pid = int(os.getpid() if pid is None else pid)
        self.clock = clock if clock is not None else MonotonicRawClock()
        self.cuda_backend = cuda_backend
        self._enable_cuda_events = bool(enable_cuda_events)
        self._cuda_backend_attempted = cuda_backend is not None
        self._writer = (
            None
            if self.mode is ProfileMode.OFF
            else JsonlTimelineWriter(self.out_dir, self.run_id, self.rank, self.pid)
        )
        self._lock = threading.RLock()
        self._scope_counter = 0
        self._completed_steps = 0
        self._active_step: int | None = None
        self._step_root_id: str | None = None
        self._current_microbatch: int | None = None
        self._microbatches: list[dict[str, Any]] = []
        self._open: dict[str, _OpenScope] = {}
        self._open_order: list[str] = []
        self._stacks: dict[int, list[str]] = {}
        self._step_records: list[dict[str, Any]] = []
        self._step_aux_records: list[dict[str, Any]] = []
        self._step_cuda: list[_PendingCuda] = []
        self._pending_cuda: list[_PendingCuda] = []
        self._closed = False

    @property
    def enabled(self) -> bool:
        return self.mode is not ProfileMode.OFF

    @property
    def current_step(self) -> int | None:
        return self._active_step

    @property
    def current_microbatch(self) -> int | None:
        return self._current_microbatch

    def _identity(self, *, tid: int | None = None) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": self.mode.value,
            "clock": getattr(self.clock, "name", type(self.clock).__name__),
            "run": self.run_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "pid": self.pid,
            "tid": int(threading.get_native_id() if tid is None else tid),
        }

    def _next_scope_id(self) -> str:
        self._scope_counter += 1
        return f"{self.run_id}:{self.rank}:{self.pid}:{self._scope_counter}"

    def _stack(self, tid: int) -> list[str]:
        return self._stacks.setdefault(tid, [])

    def begin_step(self, step: int, **attributes: Any) -> str | None:
        """Begin one optimizer-step window, closing an old window as partial."""
        if not self.enabled:
            return None
        with self._lock:
            self._ensure_open()
            if self._active_step is not None:
                self.end_step(self._active_step, status="partial")
            self._resolve_pending_cuda()
            self._active_step = int(step)
            self._current_microbatch = None
            self._microbatches = []
            self._step_records = []
            self._step_aux_records = []
            self._step_cuda = []
            root = self._begin_scope(
                name="iteration_wall",
                step=self._active_step,
                microbatch=None,
                cuda_timing=False,
                parent_id=None,
                attributes={"kind": "optimizer_step", "overview": True, **attributes},
            )
            self._step_root_id = root
            return root

    def mark_microbatch(
        self,
        microbatch: int | None | object = _AUTO_MICROBATCH,
        *,
        tokens: int | None = None,
        **attributes: Any,
    ) -> int | None:
        """Select the current microbatch and emit a marker.

        Omitting ``microbatch`` advances a zero-based counter. Passing ``None``
        clears microbatch attribution for later optimizer-level scopes.
        """
        if not self.enabled:
            return None
        with self._lock:
            if (
                self._active_step is None
                and microbatch is _AUTO_MICROBATCH
                and tokens is None
            ):
                self._current_microbatch = None
                return None
            if self._active_step is None:
                raise RuntimeError("mark_microbatch requires an active step")
            if microbatch is _AUTO_MICROBATCH:
                index = (
                    0
                    if self._current_microbatch is None
                    else self._current_microbatch + 1
                )
            elif microbatch is None:
                self._current_microbatch = None
                return None
            else:
                index = int(microbatch)
                if index < 0:
                    raise ValueError("microbatch must be non-negative")
            if tokens is not None and int(tokens) < 0:
                raise ValueError("tokens must be non-negative")
            self._current_microbatch = index
            marker = {
                "microbatch": index,
                "tokens": None if tokens is None else int(tokens),
                **attributes,
            }
            self._microbatches.append(marker)
            self.emit("microbatch_marker", name="microbatch", **marker)
            return index

    def scope(
        self,
        name: str,
        step: int | None = None,
        microbatch: int | None = None,
        cuda_timing: bool = False,
        *,
        metadata: Mapping[str, Any] | None = None,
        **attributes: Any,
    ) -> AbstractContextManager[Any]:
        """Return a nested phase context manager.

        ``step=None`` and ``microbatch=None`` inherit the recorder's current
        values. Use ``mark_microbatch(None)`` before optimizer-level scopes.
        """
        if not self.enabled:
            return _NullScope()
        with self._lock:
            self._ensure_open()
            resolved_step = self._active_step if step is None else int(step)
            resolved_microbatch = (
                self._current_microbatch if microbatch is None else int(microbatch)
            )
            attrs = dict(metadata or {})
            attrs.update(attributes)
            scope_id = self._begin_scope(
                name=str(name),
                step=resolved_step,
                microbatch=resolved_microbatch,
                cuda_timing=bool(cuda_timing),
                parent_id=None,
                attributes=attrs,
            )
            return _ScopeContext(self, scope_id)

    def emit(self, record_type: str, **payload: Any) -> str | None:
        """Emit an instantaneous, schema-identifiable timeline record."""
        if not self.enabled:
            return None
        with self._lock:
            self._ensure_open()
            tid = threading.get_native_id()
            stack = self._stack(tid)
            timestamp = int(self.clock.now_ns())
            scope_id = str(payload.pop("scope_id", self._next_scope_id()))
            parent_id = payload.pop(
                "parent_id", stack[-1] if stack else self._step_root_id
            )
            step = payload.pop("step", self._active_step)
            microbatch = payload.pop("microbatch", self._current_microbatch)
            name = str(payload.pop("name", record_type))
            record = {
                **self._identity(tid=tid),
                "record_type": str(record_type),
                "scope_id": scope_id,
                "parent_id": parent_id,
                "name": name,
                "step": step,
                "microbatch": microbatch,
                "start_ns": timestamp,
                "end_ns": timestamp,
                "duration_ns": 0,
                "exclusive_ns": 0,
                "status": "ok",
                **payload,
            }
            if self._active_step is not None and step == self._active_step:
                self._step_aux_records.append(record)
            else:
                assert self._writer is not None
                self._writer.write_event(record)
            return scope_id

    def _begin_scope(
        self,
        *,
        name: str,
        step: int | None,
        microbatch: int | None,
        cuda_timing: bool,
        parent_id: str | None,
        attributes: dict[str, Any],
    ) -> str:
        tid = threading.get_native_id()
        stack = self._stack(tid)
        if parent_id is None and stack:
            parent_id = stack[-1]
        elif parent_id is None and self._step_root_id is not None and name != "step":
            parent_id = self._step_root_id
        scope_id = self._next_scope_id()
        start_ns = int(self.clock.now_ns())
        cuda_start = None
        if cuda_timing:
            self._ensure_cuda_backend()
        if cuda_timing and self.cuda_backend is not None:
            try:
                cuda_start = self.cuda_backend.record_event()
            except Exception as exc:
                attributes["cuda_record_error"] = f"{type(exc).__name__}: {exc}"
        self._open[scope_id] = _OpenScope(
            scope_id=scope_id,
            parent_id=parent_id,
            name=name,
            step=step,
            microbatch=microbatch,
            pid=self.pid,
            tid=tid,
            start_ns=start_ns,
            cuda_start=cuda_start,
            attributes=attributes,
        )
        self._open_order.append(scope_id)
        stack.append(scope_id)
        return scope_id

    def _ensure_cuda_backend(self) -> None:
        """Probe PyTorch CUDA once, lazily at the first CUDA-timed scope."""
        if (
            self.cuda_backend is not None
            or self._cuda_backend_attempted
            or not self._enable_cuda_events
        ):
            return
        self._cuda_backend_attempted = True
        try:
            self.cuda_backend = TorchCudaEventBackend()
        except Exception:
            self.cuda_backend = None

    def _close_scope(
        self,
        scope_id: str,
        *,
        status: str = "ok",
        error: BaseException | None = None,
        end_ns: int | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            opened = self._open.pop(scope_id, None)
            if opened is None:
                return None
            cuda_end = None
            if opened.cuda_start is not None and self.cuda_backend is not None:
                try:
                    cuda_end = self.cuda_backend.record_event()
                except Exception as exc:
                    opened.attributes["cuda_record_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
            ended = int(self.clock.now_ns() if end_ns is None else end_ns)
            raw_duration = ended - opened.start_ns
            duration = max(0, raw_duration)
            raw_exclusive = duration - opened.child_ns
            exclusive = max(0, raw_exclusive)
            record = {
                **self._identity(tid=opened.tid),
                "record_type": "scope",
                "scope_id": opened.scope_id,
                "parent_id": opened.parent_id,
                "name": opened.name,
                "step": opened.step,
                "microbatch": opened.microbatch,
                "start_ns": opened.start_ns,
                "end_ns": ended,
                "duration_ns": duration,
                "exclusive_ns": exclusive,
                "status": status,
                **opened.attributes,
            }
            if raw_duration < 0:
                record["clock_regression_ns"] = -raw_duration
            if raw_exclusive < 0:
                record["child_overrun_ns"] = -raw_exclusive
            if error is not None:
                record["error_type"] = type(error).__name__
                record["error_message"] = str(error)

            stack = self._stack(opened.tid)
            if scope_id in stack:
                stack.remove(scope_id)
            parent = self._open.get(opened.parent_id or "")
            if parent is not None:
                parent.child_ns += duration

            pending = None
            if opened.cuda_start is not None and cuda_end is not None:
                record["cuda_pending"] = True
                pending = _PendingCuda(
                    record=record, start=opened.cuda_start, end=cuda_end
                )

            if self._active_step is not None and opened.step == self._active_step:
                self._step_records.append(record)
                if pending is not None:
                    self._step_cuda.append(pending)
            else:
                if pending is not None:
                    self._attempt_cuda_resolution(pending)
                assert self._writer is not None
                self._writer.write_event(record)
                if pending is not None and record.get("cuda_pending"):
                    pending.written = True
                    self._pending_cuda.append(pending)
            return record

    def end_step(
        self,
        step: int | None = None,
        *,
        status: str = "ok",
        partial: bool = False,
        observed_tokens: int | None = None,
        **attributes: Any,
    ) -> dict[str, Any] | None:
        """End a step, auto-closing leaked scopes as ``partial``."""
        if not self.enabled:
            return None
        with self._lock:
            if self._active_step is None:
                return None
            if step is not None and int(step) != self._active_step:
                raise ValueError(
                    f"ending step {step}, but active step is {self._active_step}"
                )
            if partial:
                status = "partial"
            active_step = self._active_step
            end_ns = int(self.clock.now_ns())
            root_id = self._step_root_id
            for scope_id in reversed(self._open_order):
                opened = self._open.get(scope_id)
                if opened is None or opened.step != active_step or scope_id == root_id:
                    continue
                self._close_scope(scope_id, status="partial", end_ns=end_ns)
            if root_id is not None:
                self._close_scope(root_id, status=status, end_ns=end_ns)

            for pending in self._step_cuda:
                self._attempt_cuda_resolution(pending)
            ordered = sorted(
                [*self._step_records, *self._step_aux_records],
                key=lambda record: (
                    int(record["start_ns"]),
                    -int(record["end_ns"]),
                    str(record["scope_id"]),
                ),
            )
            assert self._writer is not None
            offsets = [self._writer.write_event(record) for record in ordered]
            for pending in self._step_cuda:
                if pending.record.get("cuda_pending"):
                    pending.written = True
                    self._pending_cuda.append(pending)

            summary = self._build_step_summary(active_step, status)
            if observed_tokens is not None:
                summary["observed_tokens"] = int(observed_tokens)
                summary["tokens"] = int(observed_tokens)
                if int(summary["duration_ns"]) > 0:
                    summary["tokens_per_second"] = (
                        int(observed_tokens) * 1e9 / int(summary["duration_ns"])
                    )
            summary.update(attributes)
            self._writer.write_step(summary, offsets)
            self._completed_steps += 1
            self._active_step = None
            self._step_root_id = None
            self._current_microbatch = None
            self._microbatches = []
            self._step_records = []
            self._step_aux_records = []
            self._step_cuda = []
            self._open_order = [
                scope_id for scope_id in self._open_order if scope_id in self._open
            ]
            self._stacks = {tid: stack for tid, stack in self._stacks.items() if stack}
            self._resolve_pending_cuda()
            return summary

    def _build_step_summary(self, step: int, requested_status: str) -> dict[str, Any]:
        scopes = [
            record
            for record in self._step_records
            if record.get("record_type") == "scope"
        ]
        root = next(
            (record for record in scopes if record["scope_id"] == self._step_root_id),
            None,
        )
        duration = int(root["duration_ns"]) if root is not None else 0
        overview_scopes = [
            record
            for record in scopes
            if record is root or bool(record.get("overview"))
        ]
        overview_ids = {str(record["scope_id"]) for record in overview_scopes}
        accounted_scopes = [
            record
            for record in scopes
            if record not in overview_scopes
            and str(record.get("parent_id")) in overview_ids
        ]
        accounted_ns = sum(int(record["duration_ns"]) for record in accounted_scopes)
        other_ns = sum(int(record["exclusive_ns"]) for record in overview_scopes)
        phases: dict[str, dict[str, int]] = {}
        for record in scopes:
            if record is root:
                continue
            stats = phases.setdefault(
                str(record["name"]), {"calls": 0, "inclusive_ns": 0, "exclusive_ns": 0}
            )
            stats["calls"] += 1
            stats["inclusive_ns"] += int(record["duration_ns"])
            stats["exclusive_ns"] += int(record["exclusive_ns"])
        partial = requested_status != "ok" or any(
            record.get("status") != "ok" for record in scopes
        )
        regressions = sum(
            1
            for record in scopes
            if record.get("clock_regression_ns") or record.get("child_overrun_ns")
        )
        token_values = [
            int(item["tokens"])
            for item in self._microbatches
            if item.get("tokens") is not None
        ]
        summary = {
            **self._identity(),
            "record_type": "step_summary",
            "scope_id": self._step_root_id,
            "parent_id": None,
            "name": "step_summary",
            "step": step,
            "microbatch": None,
            "start_ns": int(root["start_ns"]) if root is not None else None,
            "end_ns": int(root["end_ns"]) if root is not None else None,
            "duration_ns": duration,
            "exclusive_ns": int(root["exclusive_ns"]) if root is not None else 0,
            "status": "partial" if partial else "ok",
            "warmup": self._completed_steps < self.warmup_steps,
            "microbatch_count": len(self._microbatches),
            "tokens": sum(token_values) if token_values else None,
            "scope_count": len(scopes),
            "accounted_ns": accounted_ns,
            "leaf_accounted_ns": accounted_ns,
            "accounted_scope_ids": [record["scope_id"] for record in accounted_scopes],
            "other_ns": other_ns,
            "accounting_error_ns": duration - accounted_ns - other_ns,
            "clock_anomaly_count": regressions,
            "cuda_pending_count": sum(
                1 for item in self._step_cuda if item.record.get("cuda_pending")
            ),
            "phases": phases,
        }
        if summary["tokens"] is not None and duration > 0:
            summary["tokens_per_second"] = float(summary["tokens"]) * 1e9 / duration
        return summary

    def _attempt_cuda_resolution(self, pending: _PendingCuda) -> bool:
        if self.cuda_backend is None:
            return False
        try:
            if not self.cuda_backend.is_ready(
                pending.start
            ) or not self.cuda_backend.is_ready(pending.end):
                return False
            raw_ns = int(
                round(
                    self.cuda_backend.elapsed_ms(pending.start, pending.end)
                    * 1_000_000.0
                )
            )
            pending.record["cuda_elapsed_ns"] = max(0, raw_ns)
            if raw_ns < 0:
                pending.record["cuda_clock_regression_ns"] = -raw_ns
            pending.record["cuda_pending"] = False
        except Exception as exc:
            pending.record["cuda_pending"] = False
            pending.record["cuda_resolve_error"] = f"{type(exc).__name__}: {exc}"
        if pending.written:
            self._write_cuda_resolution(pending.record)
        return True

    def _write_cuda_resolution(self, scope: dict[str, Any]) -> None:
        assert self._writer is not None
        patch = {
            key: scope.get(key)
            for key in (
                "schema_version",
                "mode",
                "clock",
                "run",
                "rank",
                "world_size",
                "pid",
                "tid",
                "scope_id",
                "parent_id",
                "name",
                "step",
                "microbatch",
                "start_ns",
                "end_ns",
                "duration_ns",
                "exclusive_ns",
                "status",
                "cuda_elapsed_ns",
                "cuda_clock_regression_ns",
                "cuda_resolve_error",
            )
        }
        patch["record_type"] = "cuda_resolution"
        self._writer.write_event(patch)

    def _resolve_pending_cuda(self) -> None:
        remaining: list[_PendingCuda] = []
        for pending in self._pending_cuda:
            if not self._attempt_cuda_resolution(pending):
                remaining.append(pending)
        self._pending_cuda = remaining

    def flush(self, partial: bool = False) -> dict[str, Any] | None:
        """Flush artifacts; optionally close an active step as partial."""
        if not self.enabled:
            return None
        with self._lock:
            self._ensure_open()
            if partial and self._active_step is not None:
                self.end_step(self._active_step, status="partial")
            self._resolve_pending_cuda()
            assert self._writer is not None
            index = self._writer.flush(pending_cuda=len(self._pending_cuda))
            from .chrome import convert_jsonl_to_chrome_trace

            convert_jsonl_to_chrome_trace(
                self._writer.events_path, self._writer.chrome_path
            )
            return index

    def close(self, partial: bool = False) -> dict[str, Any] | None:
        if not self.enabled:
            self._closed = True
            return None
        with self._lock:
            if self._closed:
                return None
            if self._active_step is not None:
                self.end_step(self._active_step, status="partial" if partial else "ok")
            self.flush()
            assert self._writer is not None
            index = self._writer.close(pending_cuda=len(self._pending_cuda))
            self._closed = True
            return index

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("timeline recorder is closed")

    def __enter__(self) -> "TimelineRecorder":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close(partial=exc_type is not None)
        return False


StepTimelineRecorder = TimelineRecorder


__all__ = [
    "ProfileMode",
    "StepTimelineRecorder",
    "TimelineRecorder",
    "monotonic_raw_ns",
]
