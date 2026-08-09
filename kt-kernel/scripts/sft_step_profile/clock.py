"""Clock and optional CUDA-event adapters for the SFT step timeline."""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TimelineClock(Protocol):
    """Clock used for host timeline timestamps."""

    name: str

    def now_ns(self) -> int:
        """Return a timestamp in nanoseconds."""


class MonotonicRawClock:
    """Linux ``CLOCK_MONOTONIC_RAW`` clock, unaffected by NTP adjustments."""

    name = "CLOCK_MONOTONIC_RAW"

    def __init__(self) -> None:
        clock_id = getattr(time, "CLOCK_MONOTONIC_RAW", None)
        if clock_id is None:
            raise RuntimeError(
                "CLOCK_MONOTONIC_RAW is required by the SFT step profiler"
            )
        self._clock_id = clock_id

    def now_ns(self) -> int:
        return time.clock_gettime_ns(self._clock_id)


@runtime_checkable
class CudaEventBackend(Protocol):
    """Non-blocking CUDA-event interface used by :class:`TimelineRecorder`.

    Implementations must not synchronize a device or an event. Resolution is
    attempted only after a step has ended, and an event pair may remain pending.
    """

    def record_event(self) -> Any:
        """Record and return an event on the current stream."""

    def is_ready(self, event: Any) -> bool:
        """Return whether an event has completed without blocking."""

    def elapsed_ms(self, start: Any, end: Any) -> float:
        """Return elapsed device time for a completed event pair."""


class TorchCudaEventBackend:
    """Lazy PyTorch CUDA-event backend with no synchronization calls."""

    def __init__(self, torch_module: Any | None = None) -> None:
        if torch_module is None:
            import torch as torch_module  # type: ignore[no-redef]

        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        self._cuda = torch_module.cuda

    def record_event(self) -> Any:
        event = self._cuda.Event(enable_timing=True)
        event.record()
        return event

    def is_ready(self, event: Any) -> bool:
        return bool(event.query())

    def elapsed_ms(self, start: Any, end: Any) -> float:
        return float(start.elapsed_time(end))


__all__ = [
    "CudaEventBackend",
    "MonotonicRawClock",
    "TimelineClock",
    "TorchCudaEventBackend",
]
