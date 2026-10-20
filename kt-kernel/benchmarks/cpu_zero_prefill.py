"""Validate all-CPU expert arithmetic on complete fixed golden prefill inputs."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from cpu_zero_local import CpuProbe
from kt_to_golden_ablation import LayerMath, STAGES, load_weights
from merge_replay_common import configure_torch, dump

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--fixtures", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--layers", default="0")
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
configure_torch()
probe = CpuProbe(args.output)
probe.mode = "exact"
fixtures = [(f.name, torch.load(f, weights_only=False)) for f in sorted(args.fixtures.glob("*.pt"))]
results = []
for layer in map(int, args.layers.split(",")):
    began = time.time()
    w = load_weights(layer)
    math = LayerMath(w["experts.gate_up_proj"], w["experts.down_proj"], probe, layer)
    total, count, different = 0., 0, 0
    rows = []
    for name, v in fixtures:
        v = v[layer]
        value = math.evaluate(v["input"], v["ids"], v["route"], {"cpu": STAGES["07_full_cpu_down_dot"]})["cpu"]
        delta = (value.double() - v["output"].double()).abs()
        row = {"case": name, "tokens": len(value), "different": int((value != v["output"]).sum()),
               "elements": value.numel(), "abs_mean": delta.mean().item()}
        rows.append(row)
        total += delta.sum().item()
        count += delta.numel()
        different += row["different"]
        if row["different"]:
            torch.save({"candidate": value, "golden": v["output"]}, args.output / f"layer-{layer}-{name}")
        print("CASE", layer, json.dumps(row), flush=True)
    result = {"layer": layer, "elements": count, "different": different,
              "abs_mean": total/count, "seconds": time.time()-began}
    results.append(result)
    dump(args.output / f"layer-{layer:02}.json", {**result, "rows": rows})
    dump(args.output / "progress.json", results)
    print("LAYER", json.dumps(result), flush=True)
    probe.clear()
    del w, math
    gc.collect()
dump(args.output / "results.json", results)
