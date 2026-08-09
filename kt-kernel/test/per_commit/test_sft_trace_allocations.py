import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[2] / "scripts" / "sft_step_profile" / "trace_allocations.py"
)
SPEC = importlib.util.spec_from_file_location("sft_trace_allocations", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_clock_fit_and_narrowest_phase_attribution():
    trace = {
        "traceEvents": [
            {"name": "kt.clock_sync.raw_ns=1000000", "ts": 100.0},
            {"name": "kt.clock_sync.raw_ns=3000000", "ts": 2100.0},
            {
                "name": "[memory]",
                "ts": 600.0,
                "pid": 4,
                "tid": 8,
                "args": {"Bytes": 4096, "Total Allocated": 8192, "Device Id": 0},
            },
        ]
    }
    phases = [
        {
            "record_type": "scope",
            "name": "step",
            "scope_id": "root",
            "step": 1,
            "start_ns": 900000,
            "end_ns": 3100000,
            "duration_ns": 2200000,
        },
        {
            "record_type": "scope",
            "name": "forward",
            "scope_id": "forward",
            "step": 1,
            "microbatch": 0,
            "start_ns": 1400000,
            "end_ns": 1700000,
            "duration_ns": 300000,
        },
    ]

    metadata, allocations = MODULE.extract_allocations(
        trace, phases, source="fake.json"
    )

    assert metadata["anchor_count"] == 2
    assert metadata["max_anchor_error_ns"] == 0
    assert len(allocations) == 1
    assert allocations[0]["monotonic_raw_ns"] == 1500000
    assert allocations[0]["phase"] == "forward"
    assert allocations[0]["scope_id"] == "forward"
    assert allocations[0]["delta_bytes"] == 4096
    assert allocations[0]["attribution_kind"] == "temporal_containment"


def test_clock_fit_requires_explicit_anchor():
    try:
        MODULE.fit_clock([])
    except ValueError as exc:
        assert "clock_sync" in str(exc)
    else:
        raise AssertionError("fit_clock accepted a trace without a clock anchor")
