import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from sft_step_profile.chrome import read_jsonl  # noqa: E402
from sft_step_profile.clock import MonotonicRawClock  # noqa: E402
from sft_step_profile.recorder import TimelineRecorder  # noqa: E402


class FakeClock:
    name = "CLOCK_MONOTONIC_RAW"

    def __init__(self, now=0):
        self.now = now

    def now_ns(self):
        return self.now

    def advance(self, delta):
        self.now += delta


class FakeCudaBackend:
    def __init__(self, *, ready=True, elapsed_ms=1.25):
        self.ready = ready
        self.elapsed = elapsed_ms
        self.events = 0
        self.synchronize_calls = 0

    def record_event(self):
        self.events += 1
        return self.events

    def is_ready(self, event):
        return self.ready

    def elapsed_ms(self, start, end):
        return self.elapsed

    def synchronize(self):
        self.synchronize_calls += 1
        raise AssertionError("the timeline must never synchronize CUDA")


def _recorder(tmp_path, clock, *, rank=0, mode="phase", cuda_backend=None):
    return TimelineRecorder(
        tmp_path,
        "unit-run",
        rank=rank,
        world_size=2 if rank else 1,
        mode=mode,
        clock=clock,
        cuda_backend=cuda_backend,
        enable_cuda_events=False,
        pid=1234,
    )


def _step_summaries(tmp_path, rank=0):
    return read_jsonl(tmp_path / f"rank_{rank}" / "step_summary.jsonl")


def _events(tmp_path, rank=0):
    return read_jsonl(tmp_path / f"rank_{rank}" / "phase_events.jsonl")


def test_nested_scopes_report_inclusive_and_exclusive_time(tmp_path):
    clock = FakeClock()
    recorder = _recorder(tmp_path, clock)
    recorder.begin_step(7, trainer_id=4)
    clock.advance(10)
    with recorder.scope("forward"):
        clock.advance(20)
        with recorder.scope("expert"):
            clock.advance(30)
        clock.advance(20)
    clock.advance(20)
    summary = recorder.end_step(7)
    recorder.close()

    assert summary["duration_ns"] == 100
    assert summary["exclusive_ns"] == 30
    assert summary["accounted_ns"] == 70
    assert summary["leaf_accounted_ns"] == 70
    assert summary["other_ns"] == 30
    assert summary["accounting_error_ns"] == 0
    assert summary["phases"]["forward"] == {
        "calls": 1,
        "inclusive_ns": 70,
        "exclusive_ns": 40,
    }
    assert summary["phases"]["expert"] == {
        "calls": 1,
        "inclusive_ns": 30,
        "exclusive_ns": 30,
    }
    records = {
        row["name"]: row for row in _events(tmp_path) if row["record_type"] == "scope"
    }
    assert records["forward"]["parent_id"] == records["iteration_wall"]["scope_id"]
    assert records["expert"]["parent_id"] == records["forward"]["scope_id"]
    assert records["iteration_wall"]["trainer_id"] == 4
    required = {
        "run",
        "rank",
        "pid",
        "tid",
        "step",
        "microbatch",
        "scope_id",
        "parent_id",
        "start_ns",
        "end_ns",
        "duration_ns",
        "exclusive_ns",
    }
    assert required <= records["expert"].keys()


def test_gradient_accumulation_preserves_each_microbatch(tmp_path):
    clock = FakeClock(1_000)
    recorder = _recorder(tmp_path, clock)
    recorder.begin_step(1)
    for tokens in (101, 203):
        microbatch = recorder.mark_microbatch(tokens=tokens)
        with recorder.scope("forward"):
            clock.advance(10 + microbatch)
        with recorder.scope("backward"):
            clock.advance(20 + microbatch)
    recorder.mark_microbatch(None)
    with recorder.scope("optimizer"):
        clock.advance(5)
    summary = recorder.end_step(1)
    recorder.close()

    assert summary["microbatch_count"] == 2
    assert summary["tokens"] == 304
    events = _events(tmp_path)
    forwards = [row for row in events if row.get("name") == "forward"]
    backwards = [row for row in events if row.get("name") == "backward"]
    optimizer = next(row for row in events if row.get("name") == "optimizer")
    assert [row["microbatch"] for row in forwards] == [0, 1]
    assert [row["microbatch"] for row in backwards] == [0, 1]
    assert optimizer["microbatch"] is None
    assert summary["phases"]["forward"]["calls"] == 2
    assert summary["phases"]["backward"]["calls"] == 2


def test_observed_tokens_override_local_microbatch_estimate(tmp_path):
    clock = FakeClock()
    recorder = _recorder(tmp_path, clock)
    recorder.begin_step(1)
    recorder.mark_microbatch(tokens=17)
    clock.advance(10)
    summary = recorder.end_step(1, observed_tokens=31)
    recorder.close()

    assert summary["tokens"] == 31
    assert summary["observed_tokens"] == 31
    assert summary["tokens_per_second"] == pytest.approx(3.1e9)


def test_overview_accounting_does_not_double_count_nested_drilldown(tmp_path):
    clock = FakeClock()
    recorder = _recorder(tmp_path, clock)
    recorder.begin_step(1)
    with recorder.scope("dataloader"):
        clock.advance(10)
    clock.advance(5)
    with recorder.scope("train_core", metadata={"overview": True}):
        with recorder.scope("forward"):
            clock.advance(15)
            with recorder.scope("kt_repack"):
                clock.advance(20)
            clock.advance(15)
        clock.advance(10)
    clock.advance(5)
    summary = recorder.end_step(1)
    recorder.close()

    assert summary["duration_ns"] == 80
    assert (
        summary["accounted_ns"] == 60
    )  # dataloader + forward, not nested kt_repack again
    assert summary["other_ns"] == 20  # iteration_wall.exclusive + train_core.exclusive
    assert summary["accounting_error_ns"] == 0
    events = {
        row["name"]: row for row in _events(tmp_path) if row["record_type"] == "scope"
    }
    assert events["forward"]["duration_ns"] == 50
    assert events["kt_repack"]["duration_ns"] == 20
    assert events["kt_repack"]["scope_id"] not in summary["accounted_scope_ids"]


def test_partial_flush_closes_leaked_scope_and_off_mode_is_noop(tmp_path):
    clock = FakeClock()
    recorder = _recorder(tmp_path / "partial", clock)
    recorder.begin_step(3)
    leaked = recorder.scope("backward")
    leaked.__enter__()
    clock.advance(50)
    recorder.flush(partial=True)
    leaked.__exit__(None, None, None)
    recorder.close()

    summary = _step_summaries(tmp_path / "partial")[0]
    assert summary["status"] == "partial"
    backward = next(
        row for row in _events(tmp_path / "partial") if row.get("name") == "backward"
    )
    assert backward["status"] == "partial"
    assert (
        len(
            [
                row
                for row in _events(tmp_path / "partial")
                if row.get("name") == "backward"
            ]
        )
        == 1
    )

    disabled_root = tmp_path / "off"
    disabled = TimelineRecorder(disabled_root, "off-run", mode="off", clock=clock)
    assert disabled.begin_step(1) is None
    with disabled.scope("forward"):
        clock.advance(1)
    assert disabled.end_step(1) is None
    assert disabled.emit("marker") is None
    assert disabled.flush() is None
    disabled.close()
    assert not disabled_root.exists()


def test_rank_files_are_isolated_and_indexed(tmp_path):
    for rank in (0, 1):
        clock = FakeClock(rank * 1_000)
        recorder = TimelineRecorder(
            tmp_path,
            "distributed-run",
            rank=rank,
            world_size=2,
            mode="phase",
            clock=clock,
            enable_cuda_events=False,
            pid=99,
        )
        recorder.begin_step(1)
        clock.advance(10 + rank)
        recorder.end_step(1)
        recorder.close()

    assert (tmp_path / "rank_0" / "phase_events.jsonl").is_file()
    assert (tmp_path / "rank_1" / "phase_events.jsonl").is_file()
    assert {row["rank"] for row in _events(tmp_path, 0)} == {0}
    assert {row["rank"] for row in _events(tmp_path, 1)} == {1}
    for rank in (0, 1):
        rank_dir = tmp_path / f"rank_{rank}"
        index = json.loads((rank_dir / "step_index.json").read_text())
        assert index["rank"] == rank
        assert index["step_records"] == 1
        assert index["artifacts"]["events"] == "phase_events.jsonl"
        assert (rank_dir / "step_summary.json").is_file()
        assert (rank_dir / "step_summary.csv").is_file()
        assert (rank_dir / "timeline.trace.json").is_file()


def test_negative_host_delta_is_clamped_and_flagged(tmp_path):
    clock = FakeClock(100)
    recorder = _recorder(tmp_path, clock)
    recorder.begin_step(1)
    clock.now = 120
    context = recorder.scope("forward")
    context.__enter__()
    clock.now = 110
    context.__exit__(None, None, None)
    clock.now = 130
    summary = recorder.end_step(1)
    recorder.close()

    forward = next(row for row in _events(tmp_path) if row.get("name") == "forward")
    assert forward["duration_ns"] == 0
    assert forward["exclusive_ns"] == 0
    assert forward["clock_regression_ns"] == 10
    assert summary["clock_anomaly_count"] == 1


def test_cuda_events_resolve_later_without_any_synchronize(tmp_path):
    clock = FakeClock()
    cuda = FakeCudaBackend(ready=False, elapsed_ms=2.5)
    recorder = _recorder(tmp_path, clock, mode="deep", cuda_backend=cuda)
    recorder.begin_step(1)
    with recorder.scope("forward", cuda_timing=True):
        clock.advance(50)
    recorder.end_step(1)
    first = _events(tmp_path)
    forward = next(row for row in first if row.get("name") == "forward")
    assert forward["cuda_pending"] is True

    cuda.ready = True
    recorder.flush()
    recorder.close()
    records = _events(tmp_path)
    resolution = next(
        row for row in records if row.get("record_type") == "cuda_resolution"
    )
    assert resolution["scope_id"] == forward["scope_id"]
    assert resolution["cuda_elapsed_ns"] == 2_500_000
    assert cuda.synchronize_calls == 0
    assert (
        json.loads((tmp_path / "rank_0" / "step_index.json").read_text())[
            "pending_cuda_events"
        ]
        == 0
    )


def test_constructor_does_not_probe_or_import_cuda(tmp_path, monkeypatch):
    class FakeCuda:
        def __init__(self):
            self.probes = 0

        def is_available(self):
            self.probes += 1
            return False

    fake_cuda = FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    clock = FakeClock()
    recorder = TimelineRecorder(
        tmp_path,
        "lazy-cuda",
        mode="phase",
        clock=clock,
        enable_cuda_events=True,
        pid=1,
    )
    assert fake_cuda.probes == 0
    recorder.begin_step(1)
    clock.advance(1)
    recorder.end_step(1)
    recorder.close()
    assert fake_cuda.probes == 0


def test_monotonic_raw_clock_uses_required_clock_id(monkeypatch):
    calls = []

    def fake_clock_gettime_ns(clock_id):
        calls.append(clock_id)
        return 123

    import time

    monkeypatch.setattr(time, "clock_gettime_ns", fake_clock_gettime_ns)
    clock = MonotonicRawClock()
    assert clock.now_ns() == 123
    assert calls == [time.CLOCK_MONOTONIC_RAW]
