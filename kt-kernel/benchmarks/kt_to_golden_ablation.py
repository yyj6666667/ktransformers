"""Diagnostic KT-to-golden arithmetic ablation, not a production/training kernel.

The original golden is immutable. First require the independent AVX512 replay
to reproduce the real KT operator; then change only the replay's KT arithmetic.
All local probes use each layer's frozen golden inputs, IDs and routing weights.
"""

import argparse
import ctypes
from dataclasses import dataclass, replace
import gc
import json
from pathlib import Path
import subprocess
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

from merge_replay_common import ROOT, MODEL, configure_torch, dump, sha


@dataclass(frozen=True)
class Arithmetic:
    gate: str = "cpu"
    activation: str = "cpu_fused"
    down: str = "tp_bf16"
    merge: str = "tp_fp32"
    order: str = "router"


STAGES = {"00_original_kt": Arithmetic()}
STAGES["01_remove_partial_round"] = replace(STAGES["00_original_kt"], down="tp_fp32")
STAGES["02_round_complete_expert"] = replace(STAGES["01_remove_partial_round"], down="tp_sum_bf16", merge="expert_fp32")
STAGES["03_round_silu"] = replace(STAGES["02_round_complete_expert"], activation="cpu_split")
STAGES["04_round_weighted_product"] = replace(STAGES["03_round_silu"], merge="product_bf16")
STAGES["05_round_expert_accumulation"] = replace(STAGES["04_round_weighted_product"], merge="acc_bf16")
STAGES["06_expert_id_order"] = replace(STAGES["05_round_expert_accumulation"], order="expert")
STAGES["07_full_cpu_down_dot"] = replace(STAGES["06_expert_id_order"], down="cpu_full")
STAGES["08_cuda_silu"] = replace(STAGES["07_full_cpu_down_dot"], activation="cuda_split")
STAGES["09_cuda_gate_up"] = replace(STAGES["08_cuda_silu"], gate="cuda")
STAGES["10_cuda_down"] = replace(STAGES["09_cuda_gate_up"], down="cuda_full")
# Single-factor controls prevent attributing a cumulative effect to the last switch.
STAGES["only_round_silu"] = replace(STAGES["00_original_kt"], activation="cpu_split")
STAGES["only_full_cpu_down"] = replace(STAGES["00_original_kt"], down="cpu_full", merge="expert_fp32")
STAGES["only_golden_merge"] = replace(STAGES["02_round_complete_expert"], merge="acc_bf16", order="expert")
STAGES["golden_graph_cpu_fused"] = replace(STAGES["07_full_cpu_down_dot"], activation="cpu_fused")


def load_weights(layer):
    prefix = f"model.language_model.layers.{layer}.mlp."
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    result = {}
    for filename in sorted({v for k, v in index.items() if k.startswith(prefix)}):
        with safe_open(MODEL / filename, framework="pt", device="cpu") as handle:
            for key, source in index.items():
                if key.startswith(prefix) and source == filename:
                    result[key[len(prefix):]] = handle.get_tensor(key).to(torch.bfloat16)
    return result


class ArithmeticProbe:
    def __init__(self, folder):
        source = Path(__file__).with_name("kt_to_golden_math.cpp")
        target = folder / "kt_to_golden_math.so"
        command = ["g++", "-O3", "-shared", "-fPIC", "-fopenmp", "-mavx512f", "-mavx512bw",
                   "-mavx512bf16", "-mavx512vl", "-ffast-math", str(source), "-o", str(target)]
        subprocess.run(command, check=True)
        self.lib = ctypes.CDLL(str(target))
        self.lib.ordered_dpbf16_batch.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
        self.lib.ordered_dpbf16_batch.restype = None
        self.lib.kt_swiglu.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int]
        self.lib.kt_swiglu.restype = None
        self.lib.kt_merge_parts.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4
        self.lib.kt_merge_parts.restype = None
        self.packed = {}
        dump(folder / "probe-build.json", {"command": command, "source_sha256": sha(source),
             "binary_sha256": sha(target), "scope": "independent diagnostic arithmetic replay; no backward"})

    def linear(self, x, weight, key):
        x = x.cpu().bfloat16().contiguous()
        if key not in self.packed:
            weight = weight.cpu().bfloat16().contiguous()
            n, k = weight.shape
            self.packed[key] = weight.reshape(n, k // 2, 2).permute(1, 0, 2).contiguous()
        packed = self.packed[key]
        k2, n, _ = packed.shape
        assert x.shape[1] == 2 * k2
        out = torch.empty((len(x), n), dtype=torch.float32)
        self.lib.ordered_dpbf16_batch(x.data_ptr(), packed.data_ptr(), out.data_ptr(), len(x), n, 2 * k2)
        return out

    def activation(self, gate, up, split=False):
        gate, up = gate.cpu().float().contiguous(), up.cpu().float().contiguous()
        operand = torch.ones_like(up) if split else up
        out = torch.empty_like(gate)
        self.lib.kt_swiglu(gate.data_ptr(), operand.data_ptr(), out.data_ptr(), gate.numel())
        rounded = out.bfloat16()
        return rounded * up.bfloat16() if split else rounded

    def merge(self, values, route):
        values, route = values.float().cpu().contiguous(), route.float().cpu().contiguous()
        parts, tokens, topk, hidden = values.shape
        out = torch.empty((tokens, hidden), dtype=torch.float32)
        self.lib.kt_merge_parts(values.data_ptr(), route.data_ptr(), out.data_ptr(), parts, tokens, topk, hidden)
        return out.bfloat16()


def combine(values, ids, route, mode, order, probe):
    if mode == "tp_fp32":
        return probe.merge(values, route)
    assert values.shape[0] == 1
    values = values[0].cpu()
    ids, route = ids.cpu(), route.cpu().bfloat16()
    if mode == "expert_fp32":
        return probe.merge(values.unsqueeze(0), route)
    acc = torch.zeros((len(ids), values.shape[-1]), dtype=torch.bfloat16 if mode == "acc_bf16" else torch.float32)
    order_ids = ids.argsort(-1) if order == "expert" else torch.arange(ids.shape[1]).expand_as(ids)
    for j in range(ids.shape[1]):
        rank = order_ids[:, j]
        weighted = values[torch.arange(len(ids)), rank].bfloat16() * route.gather(1, rank[:, None])
        acc += weighted.to(acc.dtype)
    return acc.bfloat16()


class LayerMath:
    """Caches common intermediate work across variants, never alters the golden."""
    def __init__(self, gate_up, down, probe, layer):
        self.gu, self.down, self.probe, self.layer = gate_up, down, probe, layer

    def evaluate(self, x, ids, route, variants):
        device = self.gu.device if self.gu.is_cuda else torch.device("cuda", torch.cuda.current_device())
        all_cpu = all(cfg.gate == "cpu" and cfg.activation.startswith("cpu") and cfg.down != "cuda_full"
                      for cfg in variants.values())
        # One transfer per layer instead of one per active expert. This only
        # affects the diagnostic harness, not any arithmetic or KT timing run.
        x = x.to("cpu" if all_cpu else device)
        ids, route = ids.cpu(), route.cpu().bfloat16()
        mask = F.one_hot(ids, num_classes=256).permute(2, 1, 0)
        hit = (mask.sum((-1, -2)) > 0).nonzero().flatten().tolist()
        outputs = {name: torch.empty((2 if cfg.down in ("tp_bf16", "tp_fp32") else 1,
                                     len(x), ids.shape[1], x.shape[1]), dtype=torch.float32)
                   for name, cfg in variants.items()}
        for expert in hit:
            top, token = torch.where(mask[expert])
            current = x[token.to(x.device)]
            gates, acts, downs = {}, {}, {}
            for name, cfg in variants.items():
                if cfg.gate not in gates:
                    if cfg.gate == "cpu":
                        gu = self.probe.linear(current, self.gu[expert], (self.layer, expert, "gu")).bfloat16()
                    else:
                        gu = F.linear(current, self.gu[expert].to(current.device)).cpu()
                    gates[cfg.gate] = gu.chunk(2, dim=-1)
                act_key = (cfg.gate, cfg.activation)
                if act_key not in acts:
                    gate, up = gates[cfg.gate]
                    if cfg.activation.startswith("cpu"):
                        act = self.probe.activation(gate, up, split=cfg.activation == "cpu_split")
                    else:
                        act = (F.silu(gate.to(device)) * up.to(device)).cpu()
                    acts[act_key] = act
                key = (*act_key, cfg.down)
                if key not in downs:
                    act = acts[act_key]
                    if cfg.down.startswith("tp"):
                        raw = [self.probe.linear(act[:, start:start+256], self.down[expert, :, start:start+256],
                               (self.layer, expert, "down", part)) for part, start in enumerate((0, 256))]
                        if cfg.down == "tp_bf16":
                            value = [v.bfloat16().float() for v in raw]
                        elif cfg.down == "tp_fp32":
                            value = raw
                        else:
                            value = [(raw[0] + raw[1]).bfloat16().float()]
                    elif cfg.down == "cpu_full":
                        value = [self.probe.linear(act, self.down[expert], (self.layer, expert, "down_full")).bfloat16().float()]
                    else:
                        value = [F.linear(act.to(device), self.down[expert].to(device)).cpu().float()]
                    downs[key] = value
                for part, value in enumerate(downs[key]):
                    outputs[name][part, token, top] = value
        return {name: combine(outputs[name], ids, route, cfg.merge, cfg.order, self.probe)
                for name, cfg in variants.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0")
    parser.add_argument("--positions", default="1,2,14,21,31")
    parser.add_argument("--actual", choices=("before", "fp32"), default="before")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_torch()
    torch.use_deterministic_algorithms(True)
    probe = ArithmeticProbe(args.output)
    from bench_bf16_merge import Operator
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
    import kt_kernel
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True).text_config
    config._experts_implementation = "eager"
    layers, positions = list(map(int, args.layers.split(","))), list(map(int, args.positions.split(",")))
    assert all(0 < p < 32 for p in positions)
    fixtures = []
    for file in sorted((ROOT / "outputs/native-step1-001/generated_cached").glob("*.cached.r0.pt")):
        value = torch.load(file, map_location="cpu", weights_only=False)
        fixtures.append((file.name, {k: v for k, v in value["hidden"].items()
                         if any(k == f"layer.{layer}.{suffix}" for layer in layers for suffix in
                                ("post_attention_layernorm", "router_ids", "router_weights", "mlp"))}))
    dump(args.output / "controls.json", {"actual": args.actual, "extension_sha256": sha(kt_kernel.kt_kernel_ext.__file__),
         "reference": "unchanged original golden", "layers": layers, "positions": positions,
         "scope": "local decode MoE only; frozen per-layer input, weights and routing; no propagated error",
         "stages": {name: vars(cfg) for name, cfg in STAGES.items()}})
    summary = []
    with torch.no_grad():
        for layer in layers:
            began = time.time()
            weights = load_weights(layer)
            gu, down = weights["experts.gate_up_proj"], weights["experts.down_proj"]
            engine = Operator(gu[:, :512], gu[:, 512:], down, topk=8, max_len=4)
            with torch.device("meta"):
                golden = Qwen3_5MoeSparseMoeBlock(config)
            golden.load_state_dict(weights, assign=True, strict=True)
            golden = golden.cuda().eval()
            math = LayerMath(gu, down, probe, layer)
            rows, totals = [], {key: [0., 0] for key in STAGES}
            different_actual = compared_actual = 0
            for case, hidden in fixtures:
                for pos in positions:
                    x = hidden[f"layer.{layer}.post_attention_layernorm"][pos:pos+1]
                    ids = hidden[f"layer.{layer}.router_ids"][pos:pos+1]
                    route = hidden[f"layer.{layer}.router_weights"][pos:pos+1]
                    official = golden.experts(x.cuda(), ids.cuda(), route.cuda()).cpu()
                    shared = (golden.shared_expert(x.cuda()) * torch.sigmoid(golden.shared_expert_gate(x.cuda()))).cpu()
                    assert torch.equal(official + shared, hidden[f"layer.{layer}.mlp"][pos:pos+1])
                    engine.prepare(x.repeat(4, 1), ids.repeat(4, 1), route.repeat(4, 1))
                    actual = engine.forward().clone()[:1]
                    values = math.evaluate(x, ids, route, STAGES)
                    control = "00_original_kt" if args.actual == "before" else "01_remove_partial_round"
                    difference = int((actual != values[control]).sum())
                    different_actual += difference
                    compared_actual += actual.numel()
                    assert difference == 0, (layer, case, pos, "replay differs from actual KT", difference)
                    assert torch.equal(values["10_cuda_down"], official), (layer, case, pos, "golden endpoint mismatch")
                    row = {"case": case, "position": pos, "abs_mean": {}}
                    for name, value in values.items():
                        delta = (value.double() - official.double()).abs()
                        totals[name][0] += delta.sum().item()
                        totals[name][1] += delta.numel()
                        row["abs_mean"][name] = delta.mean().item()
                    rows.append(row)
            result = {"layer": layer, "samples": len(rows), "seconds": time.time() - began,
                      "actual_control": {"different": different_actual, "elements": compared_actual},
                      "abs_mean": {name: total / count for name, (total, count) in totals.items()}}
            dump(args.output / f"layer-{layer:02}.json", {**result, "rows": rows})
            summary.append(result)
            dump(args.output / "progress.json", summary)
            print("LAYER", layer, json.dumps(result), flush=True)
            probe.packed.clear()
            del engine, golden, math, weights, gu, down
            gc.collect()
            torch.cuda.empty_cache()
    dump(args.output / "results.json", summary)
    dump(args.output / "complete.json", {"exit": "success", "layers": len(summary), "samples": sum(s["samples"] for s in summary)})


if __name__ == "__main__":
    main()
