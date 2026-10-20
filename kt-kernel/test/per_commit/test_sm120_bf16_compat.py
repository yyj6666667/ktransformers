"""Actual CPU backend versus the pinned CUDA arithmetic profile, without a model download."""
import importlib.metadata
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
kt_kernel = pytest.importorskip("kt_kernel")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from cpu_zero_operator import ExactOperator


@pytest.fixture(params=["0", "1"])
def fixture(request, monkeypatch):
    if not hasattr(kt_kernel.kt_kernel_ext.moe, "SM120BF16_MOE"):
        pytest.skip("requires GCC AVX512-BF16 SM120 compatibility backend")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("reference requires SM120")
    if torch.cuda.get_device_properties(0).multi_processor_count != 170:
        pytest.skip("reference requires the frozen RTX 5090 dispatch profile")
    try:
        cublas = importlib.metadata.version("nvidia-cublas-cu12")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("requires the pinned cuBLAS wheel")
    if cublas != "12.8.4.1" or torch.__version__ != "2.9.1+cu128":
        pytest.skip("reference dispatch is pinned to torch 2.9.1/cuBLAS 12.8.4.1")
    monkeypatch.setenv("KT_SM120_INTERVAL", request.param)
    torch.set_num_threads(2)
    old = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    generator = torch.Generator().manual_seed(90147)
    gu = torch.randn(2, 1024, 2048, generator=generator, dtype=torch.bfloat16).mul_(0.02)
    down = torch.randn(2, 2048, 512, generator=generator, dtype=torch.bfloat16).mul_(0.02)
    engine = ExactOperator(gu[:, :512], gu[:, 512:], down, topk=2, max_len=65,
                           pools=1, threads=4, mapping=torch.tensor([1, 0]))
    x = torch.randn(65, 2048, generator=generator, dtype=torch.bfloat16)
    route = torch.randn(65, 2, generator=generator).softmax(-1).bfloat16()
    gu_gpu, down_gpu = gu.cuda(), down.cuda()

    def reference(x, route):
        x, route = x.cuda(), route.cuda()
        result = torch.zeros_like(x)
        for expert in range(2):
            gate, up = torch.nn.functional.linear(x, gu_gpu[expert]).chunk(2, dim=-1)
            activated = torch.nn.functional.silu(gate) * up
            value = torch.nn.functional.linear(activated, down_gpu[expert])
            result += value * route[:, expert:expert+1]
        return result.cpu()

    yield engine, x, route, reference
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old


@pytest.mark.parametrize("rows", [1, 2, 17, 33, 65])
def test_native_forward(fixture, rows):
    engine, x, route, reference = fixture
    ids = torch.tensor([1, 0]).expand(rows, 2).contiguous()
    engine.prepare(x[:rows], ids, route[:rows])
    expected = reference(x[:rows], route[:rows])
    assert torch.equal(engine.forward().view(torch.int16), expected.view(torch.int16))


def test_separate_logical_decode_batches(fixture):
    engine, x, route, reference = fixture
    engine.moe.set_logical_batch_size(1)
    engine.prepare(x[:4], torch.tensor([1, 0]).expand(4, 2), route[:4])
    expected = torch.cat([reference(x[i:i+1], route[i:i+1]) for i in range(4)])
    assert torch.equal(engine.forward().view(torch.int16), expected.view(torch.int16))
