from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "sft_step_profile"


def _load_sampler_modules():
    # Load the two leaf modules without executing the package entrypoint.  This
    # keeps these hardware-reader tests independent of optional Trainer deps.
    package = types.ModuleType("_sft_step_profile_sampler_tests")
    package.__path__ = [str(PACKAGE_DIR)]
    sys.modules[package.__name__] = package
    loaded = {}
    for name in ("samplers", "resctrl", "append_perf_stat"):
        qualified = f"{package.__name__}.{name}"
        spec = importlib.util.spec_from_file_location(
            qualified, PACKAGE_DIR / f"{name}.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded["samplers"], loaded["resctrl"], loaded["append_perf_stat"]


samplers, resctrl, append_perf = _load_sampler_modules()
MemoryReader = samplers.MemoryReader
MemorySampler = samplers.MemorySampler
PerfStatReader = samplers.PerfStatReader
HardwareCounterSampler = resctrl.HardwareCounterSampler
ResctrlMbmReader = resctrl.ResctrlMbmReader
parse_perf_stat = samplers.parse_perf_stat
append_perf_stat = append_perf.append_perf_stat


class _FakeCuda:
    def __init__(self, available: bool = True) -> None:
        self.available = available

    def is_available(self):
        return self.available

    def device_count(self):
        return 2

    def memory_allocated(self, device):
        return 1_000 + int(device)

    def memory_reserved(self, device):
        return 2_000 + int(device)


class _FakeTorch:
    def __init__(self, available: bool = True) -> None:
        self.cuda = _FakeCuda(available)


class _MemoryInfo:
    def __init__(self, used: int, total: int) -> None:
        self.used = used
        self.total = total


class _FakeNvml:
    def __init__(self, *, fail_init: bool = False) -> None:
        self.fail_init = fail_init
        self.init_calls = 0
        self.shutdown_calls = 0

    def nvmlInit(self):
        self.init_calls += 1
        if self.fail_init:
            raise RuntimeError("driver absent")

    def nvmlShutdown(self):
        self.shutdown_calls += 1

    def nvmlDeviceGetCount(self):
        return 2

    def nvmlDeviceGetHandleByIndex(self, device):
        return int(device)

    def nvmlDeviceGetMemoryInfo(self, handle):
        return _MemoryInfo(used=3_000 + handle, total=10_000)

    def nvmlDeviceGetUUID(self, handle):
        return f"GPU-fake-{handle}".encode()


def _write_fake_memory_files(tmp_path: Path, pid: int = 123) -> tuple[Path, Path]:
    proc_root = tmp_path / "proc"
    cgroup_root = tmp_path / "sys" / "fs" / "cgroup"
    process = proc_root / str(pid)
    process.mkdir(parents=True)
    (process / "status").write_text(
        "VmRSS:\t100 kB\n"
        "RssAnon:\t60 kB\n"
        "RssFile:\t30 kB\n"
        "RssShmem:\t10 kB\n"
        "VmSwap:\t4 kB\n",
        encoding="ascii",
    )
    (process / "cgroup").write_text("0::/work.slice/train.scope\n", encoding="ascii")
    group = cgroup_root / "work.slice" / "train.scope"
    group.mkdir(parents=True)
    (group / "memory.current").write_text("1000000\n", encoding="ascii")
    (group / "memory.swap.current").write_text("4096\n", encoding="ascii")
    (group / "memory.stat").write_text(
        "anon 700000\nfile 250000\nkernel 50000\n", encoding="ascii"
    )
    return proc_root, cgroup_root


def test_memory_reader_reads_process_cgroup_torch_and_nvml(tmp_path):
    proc_root, cgroup_root = _write_fake_memory_files(tmp_path)
    fake_nvml = _FakeNvml()
    reader = MemoryReader(
        pid=123,
        rank=3,
        run_id="run-a",
        proc_root=proc_root,
        cgroup_root=cgroup_root,
        torch_module=_FakeTorch(),
        nvml_module=fake_nvml,
        clock=lambda: 987_654_321,
    )

    sample = reader.snapshot()

    assert sample["schema_version"] == 1
    assert sample["record_type"] == "memory_sample"
    assert sample["clock_name"] == "CLOCK_MONOTONIC_RAW"
    assert sample["monotonic_raw_ns"] == 987_654_321
    assert sample["status"] == "ok"
    assert sample["process_status"] == "ok"
    assert sample["process_rss_bytes"] == 100 * 1024
    assert sample["process_anon_bytes"] == 60 * 1024
    assert sample["process_file_bytes"] == 30 * 1024
    assert sample["process_swap_bytes"] == 4 * 1024
    assert sample["cgroup_status"] == "ok"
    assert sample["cgroup_path"] == "/work.slice/train.scope"
    assert sample["cgroup_current_bytes"] == 1_000_000
    assert sample["cgroup_anon_bytes"] == 700_000
    assert sample["cgroup_file_bytes"] == 250_000
    assert sample["cgroup_swap_bytes"] == 4_096
    assert sample["torch_cuda_allocated_bytes_by_device"] == {"0": 1_000, "1": 1_001}
    assert sample["torch_cuda_reserved_bytes_by_device"] == {"0": 2_000, "1": 2_001}
    assert sample["nvml_used_bytes_by_device"] == {"0": 3_000, "1": 3_001}
    assert sample["nvml_total_bytes_by_device"] == {"0": 10_000, "1": 10_000}
    assert sample["nvml_uuid_by_device"] == {"0": "GPU-fake-0", "1": "GPU-fake-1"}
    assert sample["errors"] == {}

    reader.close()
    assert fake_nvml.init_calls == 1
    assert fake_nvml.shutdown_calls == 1


def test_memory_reader_reports_missing_capabilities_without_raising(tmp_path):
    reader = MemoryReader(
        pid=999,
        proc_root=tmp_path / "missing-proc",
        cgroup_root=tmp_path / "missing-cgroup",
        torch_module=_FakeTorch(available=False),
        nvml_module=_FakeNvml(fail_init=True),
        clock=lambda: 42,
    )

    sample = reader.snapshot()

    assert sample["monotonic_raw_ns"] == 42
    assert sample["status"] == "unavailable"
    assert sample["process_status"] == "unavailable"
    assert sample["cgroup_status"] == "unavailable"
    assert sample["torch_cuda_status"] == "unavailable"
    assert sample["nvml_status"] == "unavailable"
    assert sample["torch_cuda_allocated_bytes_by_device"] == {}
    assert sample["nvml_used_bytes_by_device"] == {}
    assert {"process", "cgroup", "torch_cuda", "nvml"}.issubset(sample["errors"])


def test_memory_reader_marks_unselected_sources_not_sampled(tmp_path):
    proc_root, cgroup_root = _write_fake_memory_files(tmp_path)
    reader = MemoryReader(
        pid=123,
        proc_root=proc_root,
        cgroup_root=cgroup_root,
        torch_module=None,
        nvml_module=None,
        clock=lambda: 7,
    )

    sample = reader.snapshot(
        include_cgroup=False, include_torch_cuda=False, include_nvml=False
    )

    assert sample["status"] == "ok"
    assert sample["process_status"] == "ok"
    assert sample["cgroup_status"] == "not_sampled"
    assert sample["torch_cuda_status"] == "not_sampled"
    assert sample["nvml_status"] == "not_sampled"


def test_background_memory_sampler_writes_host_and_nvml_records(tmp_path):
    proc_root, cgroup_root = _write_fake_memory_files(tmp_path)
    reader = MemoryReader(
        pid=123,
        rank=1,
        run_id="background",
        proc_root=proc_root,
        cgroup_root=cgroup_root,
        torch_module=_FakeTorch(),
        nvml_module=_FakeNvml(),
    )
    output = tmp_path / "samples.jsonl"
    sampler = MemorySampler(
        output,
        rank=1,
        run_id="background",
        host_interval_ms=5,
        nvml_interval_ms=7,
        reader=reader,
    )

    assert (
        MemorySampler(
            tmp_path / "defaults.jsonl", rank=0, run_id="defaults"
        ).host_interval_ms
        == 20
    )
    assert (
        MemorySampler(
            tmp_path / "defaults2.jsonl", rank=0, run_id="defaults"
        ).nvml_interval_ms
        == 50
    )
    sampler.start()
    for boundary_index in range(25):
        sampler.emit(
            {
                "schema_version": 1,
                "record_type": "memory_sample",
                "sample_kind": "phase_boundary",
                "boundary_index": boundary_index,
                "monotonic_raw_ns": boundary_index,
            }
        )
    time.sleep(0.045)
    summary = sampler.stop()
    reader.close()

    rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["record_type"] == "memory_sampler_metadata"
    host = [row for row in rows if row.get("sample_group") == "host"]
    nvml = [row for row in rows if row.get("sample_group") == "nvml"]
    boundaries = [row for row in rows if row.get("sample_kind") == "phase_boundary"]
    assert len(host) >= 2
    assert len(nvml) >= 2
    assert [row["boundary_index"] for row in boundaries] == list(range(25))
    assert all(row["run_id"] == "background" and row["rank"] == 1 for row in rows)
    assert all(isinstance(row["monotonic_raw_ns"], int) for row in rows)
    assert all(row["nvml_status"] == "not_sampled" for row in host)
    assert all(row["process_status"] == "not_sampled" for row in nvml)
    assert summary["status"] == "ok"
    assert summary["records_written"] == len(rows)


def test_perf_stat_parser_preserves_unavailable_events_and_missing_file(tmp_path):
    result = parse_perf_stat(
        "1000,,cycles,1000000,100.00\n"
        "250,Joules,power/energy-pkg/,1000000,50.0\n"
        "<not supported>,,uncore_imc/data_reads/,0,0.00\n"
        "this is not delimited output\n",
        timestamp_ns=1234,
    )

    assert result["status"] == "partial"
    assert result["monotonic_raw_ns"] == 1234
    assert result["events"][0]["value"] == 1_000
    assert result["events"][1]["value"] == 250
    assert result["events"][2]["status"] == "unavailable"
    assert result["events"][2]["reason"] == "not supported"
    assert "line_4" in result["errors"]

    missing = PerfStatReader(
        tmp_path / "missing-perf.csv", clock=lambda: 999
    ).snapshot()
    assert missing["status"] == "unavailable"
    assert missing["monotonic_raw_ns"] == 999
    assert missing["events"] == []
    assert "perf_stat" in missing["errors"]


def test_completed_perf_stat_can_be_appended_to_hardware_ledger(tmp_path):
    perf_path = tmp_path / "perf.csv"
    perf_path.write_text(
        "1000,,cycles,1000000,100.00,,\n"
        "800,,instructions,1000000,100.00,0.80,insn per cycle\n",
        encoding="utf-8",
    )
    output = tmp_path / "hardware_counters.jsonl"

    record = append_perf_stat(perf_path, output, run_id="run-perf", rank=3)

    assert record["status"] == "ok"
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["run_id"] == "run-perf"
    assert saved["rank"] == 3
    assert [event["value"] for event in saved["events"]] == [1_000, 800]


def _write_resctrl_domain(root: Path, domain: str, total: int, local: int) -> None:
    path = root / "mon_data" / f"mon_L3_{domain}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "mbm_total_bytes").write_text(f"{total}\n", encoding="ascii")
    (path / "mbm_local_bytes").write_text(f"{local}\n", encoding="ascii")


def test_resctrl_mbm_reader_reports_raw_delta_and_counter_reset(tmp_path):
    root = tmp_path / "resctrl"
    _write_resctrl_domain(root, "00", total=1_000, local=700)
    _write_resctrl_domain(root, "01", total=2_000, local=1_200)
    reader = ResctrlMbmReader(root=root, rank=2, run_id="mbm", clock=lambda: 555)

    first = reader.snapshot()
    assert first["status"] == "ok"
    assert first["monotonic_raw_ns"] == 555
    assert first["resctrl_mbm_total_bytes_by_domain"] == {"00": 1_000, "01": 2_000}
    assert first["resctrl_mbm_total_delta_bytes_by_domain"] == {}

    _write_resctrl_domain(root, "00", total=1_150, local=790)
    _write_resctrl_domain(root, "01", total=100, local=50)
    second = reader.snapshot()
    assert second["resctrl_mbm_total_delta_bytes_by_domain"] == {"00": 150}
    assert second["resctrl_mbm_local_delta_bytes_by_domain"] == {"00": 90}
    assert second["resctrl_mbm_total_reset_domains"] == ["01"]
    assert second["resctrl_mbm_local_reset_domains"] == ["01"]

    reader.reset_baseline()
    assert reader.snapshot()["resctrl_mbm_total_delta_bytes_by_domain"] == {}


def test_resctrl_missing_hierarchy_is_explicitly_unavailable(tmp_path):
    sample = ResctrlMbmReader(root=tmp_path / "not-mounted", clock=lambda: 1).snapshot()
    assert sample["status"] == "unavailable"
    assert sample["resctrl_mbm_total_bytes_by_domain"] == {}
    assert "resctrl" in sample["errors"]


def test_hardware_counter_sampler_writes_available_and_unavailable_records(tmp_path):
    root = tmp_path / "resctrl"
    _write_resctrl_domain(root, "00", total=10_000, local=6_000)
    reader = ResctrlMbmReader(root=root, rank=4, run_id="counters")
    output = tmp_path / "hardware_counters.jsonl"
    sampler = HardwareCounterSampler(
        output,
        rank=4,
        run_id="counters",
        interval_ms=5,
        reader=reader,
    )
    sampler.start()
    time.sleep(0.025)
    summary = sampler.stop()

    rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["record_type"] == "hardware_counter_metadata"
    samples = [row for row in rows if row["record_type"] == "resctrl_mbm_sample"]
    assert len(samples) >= 2
    assert all(row["status"] == "ok" for row in samples)
    assert all(row["rank"] == 4 and row["run_id"] == "counters" for row in rows)
    assert summary["status"] == "ok"

    unavailable_output = tmp_path / "hardware_counters_unavailable.jsonl"
    unavailable = HardwareCounterSampler(
        unavailable_output,
        rank=0,
        run_id="missing",
        resctrl_root=tmp_path / "not-mounted",
        interval_ms=5,
    )
    unavailable.start()
    time.sleep(0.012)
    unavailable.stop()
    unavailable_rows = [
        json.loads(line)
        for line in unavailable_output.read_text(encoding="utf-8").splitlines()
    ]
    unavailable_samples = [
        row for row in unavailable_rows if row["record_type"] == "resctrl_mbm_sample"
    ]
    assert unavailable_samples
    assert all(row["status"] == "unavailable" for row in unavailable_samples)
    assert all("resctrl" in row["errors"] for row in unavailable_samples)
