import json

import pytest

from scripts.sft_step_profile.kt_events import (
    KTEventError,
    parse_event_line,
    read_events,
    summarize_pool_events,
)


def _event(timestamp_ns=1, **overrides):
    event = {
        "schema": "kt.sft.pool_event.v1",
        "clock": "CLOCK_MONOTONIC_RAW",
        "timestamp_ns": timestamp_ns,
        "rank": 0,
        "pid": 101,
        "tid": 202,
        "layer": 1,
        "numa": 0,
        "pool": "forward_work",
        "event": "allocate",
        "shared": True,
        "qlen": 1024,
        "cache_depth": 1,
        "requested_bytes": 100,
        "old_capacity_bytes": 0,
        "new_capacity_bytes": 100,
    }
    event.update(overrides)
    return event


def _parsed(**overrides):
    return parse_event_line(json.dumps(_event(**overrides)))


def test_parse_v1_event_and_read_jsonl():
    event = _event()
    assert parse_event_line(json.dumps(event), 7) == event
    assert parse_event_line(json.dumps(event).encode()) == event
    assert read_events(["\n", json.dumps(event) + "\n"]) == [event]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema": "future"}, "unsupported schema"),
        ({"clock": "wall"}, "unsupported clock"),
        ({"timestamp_ns": True}, "timestamp_ns must be an integer"),
        ({"shared": 1}, "shared must be a boolean"),
        ({"event": "unknown"}, "unsupported event"),
        (
            {"event": "grow", "old_capacity_bytes": 100, "new_capacity_bytes": 100},
            "grow requires",
        ),
        (
            {"event": "reuse", "old_capacity_bytes": 100, "new_capacity_bytes": 100},
            "unsupported event",
        ),
        ({"event": "free", "new_capacity_bytes": 1}, "free requires"),
        (
            {"event": "release", "old_capacity_bytes": 100, "new_capacity_bytes": 100},
            "unsupported event",
        ),
        (
            {"event": "enter", "requested_bytes": 0, "new_capacity_bytes": 0},
            "unsupported event",
        ),
    ],
)
def test_parse_rejects_invalid_schema(overrides, message):
    with pytest.raises(KTEventError, match=message):
        parse_event_line(json.dumps(_event(**overrides)), 9)


def test_shared_pools_are_deduplicated_across_layers():
    events = [
        _parsed(timestamp_ns=10, layer=1),
        _parsed(
            timestamp_ns=30,
            layer=3,
            event="grow",
            requested_bytes=160,
            old_capacity_bytes=100,
            new_capacity_bytes=160,
        ),
    ]

    summary = summarize_pool_events(reversed(events))
    assert summary["pool_count"] == 1
    assert summary["current_capacity_bytes"] == 160
    assert summary["peak_capacity_bytes"] == 160
    assert summary["event_counts"] == {"allocate": 1, "grow": 1}
    assert summary["pools"][0]["layers"] == [1, 3]
    assert summary["pools"][0]["event_counts"] == {"allocate": 1, "grow": 1}


def test_local_layers_tp_indices_and_processes_remain_distinct():
    events = [
        _parsed(
            pool="cache",
            shared=False,
            layer=1,
            requested_bytes=40,
            new_capacity_bytes=40,
        ),
        _parsed(
            timestamp_ns=2,
            pool="cache",
            shared=False,
            layer=2,
            requested_bytes=40,
            new_capacity_bytes=40,
        ),
        _parsed(
            timestamp_ns=3,
            pool="tp_backward_temp",
            numa=0,
            requested_bytes=20,
            new_capacity_bytes=20,
        ),
        _parsed(
            timestamp_ns=4,
            pool="tp_backward_temp",
            numa=1,
            requested_bytes=30,
            new_capacity_bytes=30,
        ),
        _parsed(timestamp_ns=5, pid=303, requested_bytes=70, new_capacity_bytes=70),
        _parsed(
            timestamp_ns=6,
            pool="cache",
            shared=False,
            layer=1,
            event="free",
            requested_bytes=0,
            old_capacity_bytes=40,
            new_capacity_bytes=0,
        ),
    ]

    summary = summarize_pool_events(events)
    assert summary["process_count"] == 2
    assert summary["pool_count"] == 5
    assert summary["peak_capacity_bytes"] == 200
    assert summary["current_capacity_bytes"] == 160
    assert summary["peak_capacity_bytes_by_pool"] == {
        "cache": 80,
        "forward_work": 70,
        "tp_backward_temp": 50,
    }
    process_101 = next(
        process for process in summary["processes"] if process["pid"] == 101
    )
    assert process_101["peak_capacity_bytes"] == 130
    assert process_101["current_capacity_bytes"] == 90


def test_capacity_discontinuity_fails_fast():
    events = [
        _parsed(timestamp_ns=1),
        _parsed(
            timestamp_ns=2,
            event="grow",
            requested_bytes=160,
            old_capacity_bytes=90,
            new_capacity_bytes=160,
        ),
    ]
    with pytest.raises(KTEventError, match="capacity discontinuity"):
        summarize_pool_events(events)
