"""Record unchanged golden GEMM dispatch across the expert batch-size range."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from merge_replay_common import configure_torch, dump

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--max-rows", type=int, default=224)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
configure_torch()
torch.use_deterministic_algorithms(True)
rows = []
for component, n, k in (("gate_up", 1024, 2048), ("down", 2048, 512)):
    x = torch.randn((args.max_rows, k), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    last = None
    for m in range(1, args.max_rows+1):
        y = F.linear(x[:m], w)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            y = F.linear(x[:m], w)
            torch.cuda.synchronize()
        tracefile = args.output / "latest-trace.json"
        prof.export_chrome_trace(str(tracefile))
        events = json.loads(tracefile.read_text())["traceEvents"]
        kernels = [{"name": event["name"], "grid": event["args"]["grid"],
                    "block": event["args"]["block"]} for event in events if event.get("cat") == "kernel"]
        row = {"component": component, "m": m, "n": n, "k": k, "kernels": kernels}
        rows.append(row)
        signature = [(v["name"], v["grid"][2], v["block"]) for v in kernels]
        if signature != last:
            print("DISPATCH", json.dumps(row), flush=True)
            last = signature
    dump(args.output / "dispatch.json", rows)
dump(args.output / "complete.json", {"success": True, "rows": len(rows)})
