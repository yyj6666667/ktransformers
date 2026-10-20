"""Compare CPU MMA emulation at every FP32 accumulation boundary."""
import argparse
import ctypes
from pathlib import Path
import subprocess
import json

import torch
from merge_replay_common import configure_torch, dump


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_torch()
    source = Path(__file__).with_name("cpu_zero_math.cpp")
    library = args.output / "cpu_zero_math.so"
    cmd = ["g++", "-O3", "-shared", "-fPIC", "-fopenmp", "-mavx512f", "-mavx512dq", "-mavx512vl",
           "-mfma", "-ffp-contract=off", str(source), "-o", str(library)]
    subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(library))
    lib.reference_mma.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*3
    lib.reference_mma.restype = None
    results = []
    for file in sorted(args.reference.glob("*-m*.pt")):
        v = torch.load(file, weights_only=False)
        x, wt = v["input"].contiguous(), v["weight"].T.contiguous()
        m, k = x.shape
        n = wt.shape[1]
        out = torch.empty((m, n), dtype=torch.float32)
        steps = torch.empty((k//16, m, n), dtype=torch.float32)
        lib.reference_mma(x.data_ptr(), wt.data_ptr(), out.data_ptr(), steps.data_ptr(), m, n, k)
        diff = steps != v["steps"]
        row = {"file": file.name, "fp32_elements": steps.numel(), "fp32_different": int(diff.sum()),
               "bf16_different_vs_mma": int((out.bfloat16() != v["mma"].bfloat16()).sum()),
               "bf16_different_vs_cublas": int((out.bfloat16() != v["golden"]).sum()),
               "first_mismatches": [{"index": index, "cpu": steps[tuple(index)].item(),
                                      "gpu": v["steps"][tuple(index)].item()}
                                     for index in diff.nonzero()[:8].tolist()]}
        results.append(row)
        print(json.dumps(row), flush=True)
    dump(args.output / "results.json", results)


if __name__ == "__main__":
    main()
