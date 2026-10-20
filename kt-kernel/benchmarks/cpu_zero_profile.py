"""Minimal unchanged cuBLAS call for inspecting its actual reduction kernel."""
import argparse
import json
import torch
import torch.nn.functional as F
from merge_replay_common import configure_torch

p = argparse.ArgumentParser()
p.add_argument("--component", choices=("gate_up", "down"), default="down")
p.add_argument("--rows", type=int, default=1)
p.add_argument("--trace")
args = p.parse_args()
configure_torch()
n, k = (1024, 2048) if args.component == "gate_up" else (2048, 512)
x = torch.randn((args.rows, k), dtype=torch.bfloat16, device="cuda")
w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
if args.trace:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        y = F.linear(x, w)
        torch.cuda.synchronize()
    prof.export_chrome_trace(args.trace)
    with open(args.trace) as stream:
        trace = json.load(stream)
    for event in trace["traceEvents"]:
        if event.get("cat") == "kernel":
            print("KERNEL", event["name"], event.get("args"), flush=True)
else:
    y = F.linear(x, w)
torch.cuda.synchronize()
print("shape", args.rows, n, k, "sum", y.float().sum().item(), flush=True)
