"""Parse and aggregate opt-in KT SFT pool-allocation JSONL events."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO


SCHEMA = "kt.sft.pool_event.v1"
CLOCKS = frozenset(
    {"CLOCK_MONOTONIC_RAW", "CLOCK_MONOTONIC", "std::chrono::steady_clock"}
)
CAPACITY_EVENTS = frozenset({"allocate", "grow", "free"})
VALID_EVENTS = CAPACITY_EVENTS

_INTEGER_FIELDS = (
    "timestamp_ns",
    "rank",
    "pid",
    "tid",
    "layer",
    "numa",
    "qlen",
    "cache_depth",
    "requested_bytes",
    "old_capacity_bytes",
    "new_capacity_bytes",
)


class KTEventError(ValueError):
    """Raised when a KT pool event does not match the versioned schema."""


def _error(message: str, line_number: int | None) -> KTEventError:
    prefix = f"line {line_number}: " if line_number is not None else ""
    return KTEventError(prefix + message)


def parse_event_line(
    line: str | bytes, line_number: int | None = None
) -> dict[str, Any]:
    """Parse and strictly validate one ``kt.sft.pool_event.v1`` JSONL record."""
    if isinstance(line, bytes):
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError as error:
            raise _error("event is not valid UTF-8", line_number) from error
    if not line.strip():
        raise _error("empty event", line_number)

    try:
        event = json.loads(line)
    except json.JSONDecodeError as error:
        raise _error(f"invalid JSON: {error.msg}", line_number) from error
    if not isinstance(event, dict):
        raise _error("event must be a JSON object", line_number)
    if event.get("schema") != SCHEMA:
        raise _error(f"unsupported schema {event.get('schema')!r}", line_number)
    if event.get("clock") not in CLOCKS:
        raise _error(f"unsupported clock {event.get('clock')!r}", line_number)

    for field in _INTEGER_FIELDS:
        if type(event.get(field)) is not int:
            raise _error(f"{field} must be an integer", line_number)
    if type(event.get("shared")) is not bool:
        raise _error("shared must be a boolean", line_number)
    for field in ("pool", "event"):
        if not isinstance(event.get(field), str) or not event[field]:
            raise _error(f"{field} must be a non-empty string", line_number)

    if event["event"] not in VALID_EVENTS:
        raise _error(f"unsupported event {event['event']!r}", line_number)
    for field in (
        "timestamp_ns",
        "tid",
        "qlen",
        "cache_depth",
        "requested_bytes",
        "old_capacity_bytes",
        "new_capacity_bytes",
    ):
        if event[field] < 0:
            raise _error(f"{field} must be non-negative", line_number)
    for field in ("rank", "pid", "layer", "numa"):
        if event[field] < -1:
            raise _error(f"{field} must be -1 or non-negative", line_number)

    old_capacity = event["old_capacity_bytes"]
    new_capacity = event["new_capacity_bytes"]
    event_name = event["event"]
    if event_name == "allocate" and (old_capacity != 0 or new_capacity <= 0):
        raise _error(
            "allocate requires old_capacity_bytes=0 and new_capacity_bytes>0",
            line_number,
        )
    if event_name == "grow" and new_capacity <= old_capacity:
        raise _error("grow requires new_capacity_bytes>old_capacity_bytes", line_number)
    if event_name == "free" and new_capacity != 0:
        raise _error("free requires new_capacity_bytes=0", line_number)
    return event


def read_events(
    source: str | Path | TextIO | Iterable[str | bytes],
) -> list[dict[str, Any]]:
    """Read a JSONL path, text stream, or iterable and validate every non-empty line."""
    close_stream = False
    if isinstance(source, (str, Path)):
        stream: Iterable[str | bytes] = Path(source).open(encoding="utf-8")
        close_stream = True
    else:
        stream = source
    try:
        return [
            parse_event_line(line, line_number)
            for line_number, line in enumerate(stream, 1)
            if line.strip()
        ]
    finally:
        if close_stream:
            assert hasattr(stream, "close")
            stream.close()  # type: ignore[union-attr]


def pool_identity(event: dict[str, Any]) -> tuple[int, int, bool, str, int, int | None]:
    """Return the allocation identity; shared pools deliberately omit layer."""
    return (
        event["rank"],
        event["pid"],
        event["shared"],
        event["pool"],
        event["numa"],
        None if event["shared"] else event["layer"],
    )


def summarize_pool_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate unique pool capacities and their peaks across one capture run.

    Shared pools are keyed by process, pool name, and NUMA/TP index rather than
    layer, so a reuse event observed by another layer cannot double count the
    same allocation. Events from different ranks or processes remain distinct.
    """
    ordered = sorted(
        enumerate(events), key=lambda item: (item[1]["timestamp_ns"], item[0])
    )
    capacities: dict[tuple[int, int, bool, str, int, int | None], int] = {}
    pool_states: dict[tuple[int, int, bool, str, int, int | None], dict[str, Any]] = {}
    process_current: defaultdict[tuple[int, int], int] = defaultdict(int)
    process_peak: defaultdict[tuple[int, int], int] = defaultdict(int)
    process_event_count: Counter[tuple[int, int]] = Counter()
    process_first_timestamp: dict[tuple[int, int], int] = {}
    process_last_timestamp: dict[tuple[int, int], int] = {}
    current_by_pool: defaultdict[str, int] = defaultdict(int)
    peak_by_pool: defaultdict[str, int] = defaultdict(int)
    all_event_counts: Counter[str] = Counter()
    current_total = 0
    peak_total = 0

    for _, event in ordered:
        if event.get("schema") != SCHEMA:
            raise KTEventError("summarize_pool_events requires validated v1 events")
        process = (event["rank"], event["pid"])
        process_event_count[process] += 1
        process_first_timestamp.setdefault(process, event["timestamp_ns"])
        process_last_timestamp[process] = event["timestamp_ns"]
        all_event_counts[event["event"]] += 1
        identity = pool_identity(event)
        state = pool_states.setdefault(
            identity,
            {
                "rank": event["rank"],
                "pid": event["pid"],
                "pool": event["pool"],
                "numa": event["numa"],
                "shared": event["shared"],
                "layer": None if event["shared"] else event["layer"],
                "layers": set(),
                "event_counts": Counter(),
                "max_requested_bytes": 0,
                "current_capacity_bytes": 0,
                "peak_capacity_bytes": 0,
            },
        )
        state["layers"].add(event["layer"])
        state["event_counts"][event["event"]] += 1
        state["max_requested_bytes"] = max(
            state["max_requested_bytes"], event["requested_bytes"]
        )

        old_observed = capacities.get(identity)
        if old_observed is None:
            if event["old_capacity_bytes"] != 0:
                raise KTEventError(
                    f"first event for pool {identity!r} has non-zero old_capacity_bytes="
                    f"{event['old_capacity_bytes']}"
                )
            old_observed = 0
        elif event["old_capacity_bytes"] != old_observed:
            raise KTEventError(
                f"capacity discontinuity for pool {identity!r}: observed {old_observed}, "
                f"event reports {event['old_capacity_bytes']}"
            )
        new_observed = event["new_capacity_bytes"]
        capacities[identity] = new_observed
        delta = new_observed - old_observed
        current_total += delta
        process_current[process] += delta
        current_by_pool[event["pool"]] += delta
        peak_total = max(peak_total, current_total)
        process_peak[process] = max(process_peak[process], process_current[process])
        peak_by_pool[event["pool"]] = max(
            peak_by_pool[event["pool"]], current_by_pool[event["pool"]]
        )
        state["current_capacity_bytes"] = new_observed
        state["peak_capacity_bytes"] = max(state["peak_capacity_bytes"], new_observed)

    pools = []
    for identity in sorted(
        pool_states,
        key=lambda key: (key[0], key[1], key[3], key[4], key[2], key[5] or -1),
    ):
        state = pool_states[identity]
        pools.append(
            {
                **{
                    key: value
                    for key, value in state.items()
                    if key not in {"layers", "event_counts"}
                },
                "layers": sorted(state["layers"]),
                "event_counts": dict(sorted(state["event_counts"].items())),
            }
        )

    processes = []
    for process in sorted(process_event_count):
        rank, pid = process
        processes.append(
            {
                "rank": rank,
                "pid": pid,
                "event_count": process_event_count[process],
                "first_timestamp_ns": process_first_timestamp[process],
                "last_timestamp_ns": process_last_timestamp[process],
                "current_capacity_bytes": process_current[process],
                "peak_capacity_bytes": process_peak[process],
                "pool_count": sum(
                    1 for pool in pools if pool["rank"] == rank and pool["pid"] == pid
                ),
            }
        )

    return {
        "schema": SCHEMA,
        "event_count": sum(process_event_count.values()),
        "event_counts": dict(sorted(all_event_counts.items())),
        "process_count": len(processes),
        "pool_count": len(pools),
        "current_capacity_bytes": current_total,
        "peak_capacity_bytes": peak_total,
        "current_capacity_bytes_by_pool": dict(sorted(current_by_pool.items())),
        "peak_capacity_bytes_by_pool": dict(sorted(peak_by_pool.items())),
        "processes": processes,
        "pools": pools,
    }
