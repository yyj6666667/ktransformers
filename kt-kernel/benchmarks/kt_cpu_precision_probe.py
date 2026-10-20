"""CPU FP64 dot-product controls for the residual after matching the golden graph.

This diagnostic keeps the operation on the CPU. It tests whether the remaining
CPU/CUDA gap can be reduced by more accurate accumulation, without claiming that
FP64 is a production-performance solution or that the BF16 golden is exact.
"""

import argparse
import gc
from pathlib import Path
import json
import time

import torch
import torch.nn.functional as F

from kt_to_golden_ablation import ArithmeticProbe, LayerMath, STAGES, load_weights
from merge_replay_common import ROOT, MODEL, configure_torch, dump


class PrecisionProbe(ArithmeticProbe):
    def __init__(self, folder):
        super().__init__(folder)
        self.precise = set()
        self.weights64 = {}

    def linear(self, x, weight, key):
        component = key[-1]
        if component not in self.precise:
            return super().linear(x, weight, key)
        if key not in self.weights64:
            self.weights64[key] = weight.cpu().double().contiguous()
        # Improve dot accumulation while retaining the FP32 -> BF16 output
        # boundary of the tested GEMMs. Make the FP32 cast explicit: this
        # PyTorch build's double-to-BF16 conversion also goes through FP32.
        return F.linear(x.cpu().double(), self.weights64[key]).float().bfloat16().float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0,10,20,30,39")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_torch()
    torch.use_deterministic_algorithms(True)
    probe = PrecisionProbe(args.output)
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True).text_config
    config._experts_implementation = "eager"
    layers = list(map(int, args.layers.split(",")))
    positions = (1, 2, 14, 21, 31)
    fixtures = []
    for file in sorted((ROOT / "outputs/native-step1-001/generated_cached").glob("*.cached.r0.pt")):
        hidden = torch.load(file, map_location="cpu", weights_only=False)["hidden"]
        fixtures.append((file.name, {k: v for k, v in hidden.items() if any(
            k == f"layer.{layer}.{suffix}" for layer in layers for suffix in
            ("post_attention_layernorm", "router_ids", "router_weights", "mlp"))}))
    variants = {
        "cpu_graph": (set(), "07_full_cpu_down_dot"),
        "cpu_fp64_gate_up": ({"gu"}, "07_full_cpu_down_dot"),
        "cpu_fp64_gate_up_cuda_silu": ({"gu"}, "08_cuda_silu"),
        "cpu_fp64_gate_up_down": ({"gu", "down_full"}, "07_full_cpu_down_dot"),
        "cpu_fp64_gate_up_down_cuda_silu": ({"gu", "down_full"}, "08_cuda_silu"),
    }
    dump(args.output / "controls.json", {"layers": layers, "positions": positions,
         "variants": {name: {"fp64_components": sorted(parts), "graph": stage}
                      for name, (parts, stage) in variants.items()},
         "scope": "CPU FP64 dot accumulations, explicit FP64 -> FP32 -> BF16 output; diagnostic only"})
    results = []
    with torch.no_grad():
        for layer in layers:
            began = time.time()
            weights = load_weights(layer)
            with torch.device("meta"):
                golden = Qwen3_5MoeSparseMoeBlock(config)
            golden.load_state_dict(weights, assign=True, strict=True)
            golden = golden.cuda().eval()
            math = LayerMath(weights["experts.gate_up_proj"], weights["experts.down_proj"], probe, layer)
            totals = {name: [0., 0] for name in variants}
            rows = []
            for label, hidden in fixtures:
                for pos in positions:
                    x = hidden[f"layer.{layer}.post_attention_layernorm"][pos:pos+1]
                    ids = hidden[f"layer.{layer}.router_ids"][pos:pos+1]
                    route = hidden[f"layer.{layer}.router_weights"][pos:pos+1]
                    official = golden.experts(x.cuda(), ids.cuda(), route.cuda()).cpu()
                    shared = (golden.shared_expert(x.cuda()) * torch.sigmoid(golden.shared_expert_gate(x.cuda()))).cpu()
                    assert torch.equal(official+shared, hidden[f"layer.{layer}.mlp"][pos:pos+1])
                    row = {"case": label, "position": pos, "abs_mean": {}}
                    for name, (parts, stage) in variants.items():
                        probe.precise = parts
                        value = math.evaluate(x, ids, route, {name: STAGES[stage]})[name]
                        delta = (value.double()-official.double()).abs()
                        totals[name][0] += delta.sum().item()
                        totals[name][1] += delta.numel()
                        row["abs_mean"][name] = delta.mean().item()
                    rows.append(row)
            result = {"layer": layer, "samples": len(rows), "seconds": time.time()-began,
                      "abs_mean": {name: total/count for name, (total, count) in totals.items()}}
            results.append(result)
            dump(args.output / f"layer-{layer:02}.json", {**result, "rows": rows})
            dump(args.output / "progress.json", results)
            print("LAYER", layer, json.dumps(result), flush=True)
            probe.packed.clear()
            probe.weights64.clear()
            del weights, math, golden
            gc.collect()
            torch.cuda.empty_cache()
    dump(args.output / "results.json", results)
    dump(args.output / "complete.json", {"exit": "success", "layers": layers})


if __name__ == "__main__":
    main()
