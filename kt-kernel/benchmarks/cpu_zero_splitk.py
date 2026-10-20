"""Identify split-K boundaries using the verified CPU MMA model."""
import argparse
import ctypes
import json
from pathlib import Path

import torch
from merge_replay_common import configure_torch, dump

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--library", type=Path, required=True)
p.add_argument("--survey", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
configure_torch()
lib = ctypes.CDLL(str(args.library))
lib.reference_mma.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*3
lib.reference_mma.restype = None
results = []
for m in (32, 64):
    v = torch.load(args.survey / f"gate_up-m{m}.pt", weights_only=False)
    x, w, golden = v["input"], v["weight"], v["golden"]
    n, k = w.shape
    rows = []
    for chunk in (128, 256, 384, 416, 448, 512, 640, 768, 1024, 2048):
        parts = []
        for begin in range(0, k, chunk):
            a, b = x[:, begin:begin+chunk].contiguous(), w[:, begin:begin+chunk].T.contiguous()
            out = torch.empty((m, n), dtype=torch.float32)
            lib.reference_mma(a.data_ptr(), b.data_ptr(), out.data_ptr(), None, m, n, a.shape[1])
            parts.append(out)
        total = torch.zeros_like(parts[0])
        for part in parts:
            total += part
        row = {"m": m, "chunk": chunk, "parts": len(parts),
               "different": int((total.bfloat16() != golden).sum()), "elements": golden.numel()}
        rows.append(row)
        print(json.dumps(row), flush=True)
    results += rows
dump(args.output / "results.json", results)
