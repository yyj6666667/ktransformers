"""Expose GPU MMA accumulation boundaries for subsequent CPU-only emulation."""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F
from merge_replay_common import configure_torch, dump, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--survey", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_torch()
    source = Path(__file__).with_name("cpu_zero_wmma_reference.cu")
    target = args.output / "wmma_reference.so"
    command = ["/usr/local/cuda-12.8/bin/nvcc", "-O3", "-shared", "-Xcompiler", "-fPIC",
               "-arch=sm_120", str(source), "-o", str(target)]
    subprocess.run(command, check=True)
    lib = ctypes.CDLL(str(target))
    lib.run_mma_chain.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*3 + [ctypes.c_void_p]
    lib.run_mma_chain.restype = ctypes.c_int
    results = []
    for file in sorted(args.survey.glob("*-m*.pt")):
        record = torch.load(file, weights_only=False)
        x, w, golden = record["input"], record["weight"], record["golden"]
        m, k = x.shape
        n = w.shape[0]
        mp = ((m+15)//16)*16
        a = F.pad(x, (0, 0, 0, mp-m)).cuda()
        b = w.cuda()
        out = torch.empty((mp, n), device="cuda", dtype=torch.float32)
        steps = torch.empty((k//16, mp, n), device="cuda", dtype=torch.float32) if m <= 2 else None
        code = lib.run_mma_chain(a.data_ptr(), b.data_ptr(), out.data_ptr(),
                                steps.data_ptr() if steps is not None else None,
                                mp, n, k, torch.cuda.current_stream().cuda_stream)
        assert code == 0, code
        out = out[:m].cpu()
        row = {"file": file.name, "elements": golden.numel(),
               "bf16_different": int((out.bfloat16() != golden).sum())}
        results.append(row)
        if steps is not None:
            torch.save({**record, "mma": out, "steps": steps[:, :m].cpu()}, args.output / file.name)
        print(json.dumps(row), flush=True)
    dump(args.output / "results.json", results)
    dump(args.output / "build.json", {"command": command, "source_sha256": sha(source), "library_sha256": sha(target)})


if __name__ == "__main__":
    main()
