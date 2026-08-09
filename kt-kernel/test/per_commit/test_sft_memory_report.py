import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[2] / "scripts" / "sft_step_profile" / "memory_report.py"
)
SPEC = importlib.util.spec_from_file_location("sft_memory_report", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_phase_memory_delta_peak_and_release():
    phases = [
        {
            "record_type": "scope",
            "scope_id": "fwd",
            "boundary_id": "b1",
            "name": "forward",
            "step": 3,
            "microbatch": 1,
            "start_ns": 100,
            "end_ns": 300,
        }
    ]
    samples = [
        {
            "record_type": "memory_sample",
            "boundary_id": "b1",
            "boundary": "begin",
            "process_rss_bytes": 1000,
            "torch_cuda_allocated_bytes_by_device": {"0": 800},
            "monotonic_raw_ns": 90,
        },
        {
            "record_type": "memory_sample",
            "sample_group": "host",
            "process_rss_bytes": 1600,
            "torch_cuda_allocated_bytes_by_device": {"0": 1200},
            "monotonic_raw_ns": 200,
        },
        {
            "record_type": "memory_sample",
            "boundary_id": "b1",
            "boundary": "end",
            "process_rss_bytes": 900,
            "torch_cuda_allocated_bytes_by_device": {"0": 700},
            "monotonic_raw_ns": 310,
        },
    ]

    rows = MODULE.summarize_memory(phases, samples)

    assert len(rows) == 1
    rss = rows[0]["metrics"]["process_rss_bytes"]
    assert rss["delta_bytes"] == -100
    assert rss["sampled_peak_bytes"] == 1600
    cuda = rows[0]["metrics"]["torch_cuda_allocated_bytes_by_device.total"]
    assert cuda["delta_bytes"] == -100
    assert cuda["sampled_peak_bytes"] == 1200
    assert rows[0]["status"] == "ok"


def test_missing_end_boundary_is_partial():
    phases = [
        {
            "record_type": "scope",
            "scope_id": "bwd",
            "boundary_id": "b2",
            "name": "backward",
            "start_ns": 10,
            "end_ns": 20,
        }
    ]
    samples = [
        {
            "record_type": "memory_sample",
            "boundary_id": "b2",
            "boundary": "begin",
            "process_rss_bytes": 1,
            "monotonic_raw_ns": 9,
        }
    ]

    assert MODULE.summarize_memory(phases, samples)[0]["status"] == "partial"
