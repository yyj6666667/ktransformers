"""Exercise actual kernels with an independent dyadic-arithmetic reference.

One nonzero input channel makes gate/up exact. Down weights and routing weights
are dyadic, separating early BF16 rounding from GEMM accumulation differences.
Run on a BF16-capable CPU with at least two NUMA nodes for the TP2 cases.
"""

from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
kt_kernel = pytest.importorskip("kt_kernel")
pytestmark = pytest.mark.skipif(
    kt_kernel.__cpu_variant__ not in ("amx", "avx512_bf16"),
    reason="requires the BF16 AMX/AVX512 CPU backend",
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from bench_bf16_merge import Operator


def require_numa_nodes(pools):
    nodes = list(Path("/sys/devices/system/node").glob("node[0-9]*"))
    if len(nodes) < pools:
        pytest.skip(f"requires {pools} NUMA nodes")


def fixture(qlen):
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(17)
    experts, hidden, intermediate, topk = 4, 288, 128, 2
    gate = torch.zeros(experts, intermediate, hidden, dtype=torch.bfloat16)
    up = torch.zeros_like(gate)
    gate[:, :, 0] = 1
    up[:, :, 0] = 1
    down = (torch.randint(-16, 17, (experts, hidden, intermediate), generator=generator).float() / 512).bfloat16()
    x = torch.zeros(qlen, hidden, dtype=torch.bfloat16)
    x[:, 0] = (torch.arange(qlen) % 3 + 1).to(torch.bfloat16) / 2
    ids = torch.stack((torch.arange(qlen) % 4, (torch.arange(qlen) + 1) % 4), dim=-1)
    route = torch.tensor([0.375, 0.625]).expand(qlen, topk).contiguous()
    return (gate, up, down), x, ids, route


def reference(weights, x, ids, route, pools, *, early_round=False):
    # The KT activation has one BF16 boundary, preserved by this change.
    a = (torch.nn.functional.silu(x[:, :1].float()) * x[:, :1].float()).bfloat16().double()
    down = weights[2].double()
    expert_output = a[:, :, None] * down.sum(-1)[ids]
    out = torch.zeros_like(x, dtype=torch.float64)
    for part in down.chunk(pools, dim=-1):
        partial = a[:, :, None] * part.sum(-1)[ids]
        if early_round:
            partial = partial.bfloat16().double()
        out += (partial * route.double()[:, :, None]).sum(1)
    return out.bfloat16(), expert_output


@pytest.mark.parametrize("pools", [1, 2])
@pytest.mark.parametrize("qlen", [1, 4, 33, 65, 129])
@pytest.mark.parametrize("family", ["sft", "inference"])
def test_unrounded_down_forward(pools, qlen, family):
    require_numa_nodes(pools)
    weights, x, ids, route = fixture(qlen)
    engine = Operator(*weights, topk=2, max_len=qlen, pools=pools, threads=4, family=family)
    engine.prepare(x, ids, route)
    expected, _ = reference(weights, x, ids, route, pools)
    rounded, _ = reference(weights, x, ids, route, pools, early_round=True)
    assert not torch.equal(expected, rounded), "fixture must detect the old rounding boundary"
    torch.testing.assert_close(engine.forward(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("pools", [1, 2])
@pytest.mark.parametrize("full_weight_grad", [False, True])
def test_router_gradient_retains_fp32_across_workspace_reuse(pools, full_weight_grad):
    require_numa_nodes(pools)
    weights, x, ids, route = fixture(33)
    engine = Operator(*weights, topk=2, max_len=33, pools=pools, threads=4,
                      full_weight_grad=full_weight_grad, cache_depth=2)
    # Recompute overwrites this operator's live cache (the existing SFT API
    # contract); it does not create a second independent backward obligation.
    for scale in (1, 2):
        values = x * scale
        engine.prepare(values, ids, route)
        expected, expert_output = reference(weights, values, ids, route, pools)
        torch.testing.assert_close(engine.forward(save=True), expected, rtol=0, atol=0)
    # Overwrite shared scratch without changing the retained recompute cache.
    engine.prepare(x * 3, ids, route)
    engine.forward(save=False)
    dy = torch.full_like(x, 0.5)
    dx, droute = engine.backward(dy)
    expected = (expert_output * dy.double()[:, None, :]).sum(-1).float()
    torch.testing.assert_close(droute, expected, rtol=1e-6, atol=1e-6)
    assert torch.isfinite(dx).all()
    # Reallocate/reuse the cache after backward released its previous contents.
    engine.prepare(x, ids, route)
    engine.forward(save=True)
    _, expert_output = reference(weights, x, ids, route, pools)
    _, droute = engine.backward(dy)
    torch.testing.assert_close(droute, (expert_output * dy.double()[:, None, :]).sum(-1).float(),
                               rtol=1e-6, atol=1e-6)
