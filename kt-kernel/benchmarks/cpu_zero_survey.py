"""Survey the remaining CPU/golden arithmetic differences without changing either runtime."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from merge_replay_common import configure_torch, dump


def difference(a, b):
    finite = torch.isfinite(a) & torch.isfinite(b)
    delta = (a[finite].double() - b[finite].double()).abs()
    return {"elements": a.numel(), "different": int((a[finite] != b[finite]).sum()),
            "abs_mean": delta.mean().item(), "max_abs": delta.max().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_torch()
    torch.use_deterministic_algorithms(True)
    # Enumerate the entire BF16 domain, not values selected from the evaluation data.
    x = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    golden = F.silu(x.cuda()).cpu()
    modes = {"cpu_bf16": F.silu(x), "cpu_float": F.silu(x.float()).bfloat16(),
             "cpu_double": F.silu(x.double()).float().bfloat16()}
    activation = {}
    for name, value in modes.items():
        mask = torch.isfinite(golden) & torch.isfinite(value) & (value != golden)
        activation[name] = {**difference(value, golden), "mismatches": [
            {"bits": int(i), "x": float(x[i]), "cpu": float(value[i]), "golden": float(golden[i])}
            for i in mask.nonzero().flatten().tolist()]}
    dump(args.output / "activation.json", activation)
    torch.save({"input": x, "golden": golden, **modes}, args.output / "activation.pt")
    print("ACTIVATION", {name: row["different"] for name, row in activation.items()}, flush=True)
    results = []
    for component, n, k in (("gate_up", 1024, 2048), ("down", 2048, 512)):
        weight = torch.randn((n, k), dtype=torch.bfloat16)
        wg = weight.cuda()
        for m in (1, 2, 4, 8, 16, 32, 64, 128):
            x = torch.randn((m, k), dtype=torch.bfloat16)
            xg = x.cuda()
            for _ in range(2):
                golden = F.linear(xg, wg)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                golden = F.linear(xg, wg)
                torch.cuda.synchronize()
            kernels = sorted({event.name for event in prof.events()
                              if event.device_type == torch.autograd.DeviceType.CUDA})
            golden = golden.cpu()
            modes = {"cpu_bf16": F.linear(x, weight),
                     "cpu_float": F.linear(x.float(), weight.float()).bfloat16(),
                     "cpu_double": F.linear(x.double(), weight.double()).float().bfloat16()}
            result = {"component": component, "m": m, "n": n, "k": k, "kernels": kernels,
                      "differences": {name: difference(value, golden) for name, value in modes.items()}}
            results.append(result)
            torch.save({"input": x, "weight": weight, "golden": golden},
                       args.output / f"{component}-m{m}.pt")
            dump(args.output / "gemm.json", results)
            print("GEMM", json.dumps(result), flush=True)
    dump(args.output / "complete.json", {"success": True, "cases": len(results)})


if __name__ == "__main__":
    main()
