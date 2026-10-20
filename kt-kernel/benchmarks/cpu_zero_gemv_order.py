"""Validate SIMT reduction models on held real dot products, before kernel integration."""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess

import torch
from merge_replay_common import configure_torch, dump, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_torch()
    source = Path(__file__).with_name("cpu_zero_math.cpp")
    library = args.output / "cpu_zero_math.so"
    command = ["g++", "-O3", "-shared", "-fPIC", "-fopenmp", "-mavx512f", "-mavx512dq", "-mavx512vl", "-mfma", "-ffp-contract=off",
               str(source), "-o", str(library)]
    subprocess.run(command, check=True)
    lib = ctypes.CDLL(str(library))
    lib.gemv_dot_bank.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*4
    lib.gemv_dot_bank.restype = None
    results = {}
    for component in ("gate_up", "down"):
        bank = torch.load(args.bank / f"{component}.pt", weights_only=False)
        xs, ws, targets, raw_targets = [], [], [], []
        for row in bank:
            mask = ((row["fp64"].float().bfloat16() != row["golden"]) |
                    ((row["dpbf16"].bfloat16() != row["golden"]) & (row["golden"].abs() > 1e-30)) |
                    (row["columns"] % 257 == 0))
            n = int(mask.sum())
            xs.append(row["input"].expand(n, -1))
            ws.append(row["weight"][mask])
            targets.append(row["golden"][mask])
            raw_targets.append(row["gpu32"][mask])
        x, w = torch.cat(xs).float().contiguous(), torch.cat(ws).float().contiguous()
        target, raw_target = torch.cat(targets), torch.cat(raw_targets)
        values = []
        for lanes in (1, 2, 4, 8, 16, 32, 64, 128):
            for contiguous in (0, 1):
                value = torch.empty(len(x), dtype=torch.float32)
                lib.gemv_dot_bank(x.data_ptr(), w.data_ptr(), value.data_ptr(), len(x), x.shape[1], lanes, contiguous)
                different = int((value.bfloat16() != target).sum())
                raw_different = int((value != raw_target).sum())
                row = {"lanes": lanes, "contiguous": bool(contiguous), "elements": len(x),
                       "bf16_different": different, "fp32_different": raw_different}
                values.append(row)
        results[component] = sorted(values, key=lambda v: (v["bf16_different"], v["fp32_different"]))
        print(component, json.dumps(results[component][:6]), flush=True)
    dump(args.output / "results.json", results)
    dump(args.output / "build.json", {"command": command, "source_sha256": sha(source), "library_sha256": sha(library)})


if __name__ == "__main__":
    main()
