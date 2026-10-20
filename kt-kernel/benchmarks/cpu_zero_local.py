"""All-CPU MoE arithmetic probes against unchanged golden inputs and outputs."""
import argparse
import ctypes
import gc
import json
from pathlib import Path
import time
import subprocess

import torch
import torch.nn.functional as F

from kt_to_golden_ablation import ArithmeticProbe, LayerMath, STAGES, load_weights
from merge_replay_common import ROOT, MODEL, configure_torch, dump


class CpuProbe(ArithmeticProbe):
    def __init__(self, folder):
        super().__init__(folder)
        self.mode = "float64"
        self.converted = {}
        source = Path(__file__).with_name("cpu_zero_math.cpp")
        target = folder / "cpu_zero_math.so"
        command = ["g++", "-O3", "-shared", "-fPIC", "-fopenmp", "-mavx512f", "-mavx512dq", "-mavx512vl", "-mfma",
                   "-ffp-contract=off", str(source), "-o", str(target)]
        subprocess.run(command, check=True)
        self.cpu = ctypes.CDLL(str(target))
        self.cpu.reference_gemv.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*5
        self.cpu.reference_gemv.restype = None
        self.cpu.reference_mma.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*3
        self.cpu.reference_mma.restype = None
        dump(folder / "cpu-build.json", {"command": command})

    def linear(self, x, weight, key):
        if self.mode == "dpbf16":
            return super().linear(x, weight, key)
        if self.mode in ("simt", "exact") and len(x) == 1:
            cache_key = (self.mode, key)
            if cache_key not in self.converted:
                self.converted[cache_key] = weight.cpu().float().T.contiguous()
            w = self.converted[cache_key]
            x = x.cpu().float().contiguous()
            m, k = x.shape
            assert k in (512, 2048) and m == 1
            n = w.shape[1]
            out = torch.empty((m, n), dtype=torch.float32)
            lanes, contiguous = (16, 0) if k == 2048 else (32, 1)
            self.cpu.reference_gemv(x.data_ptr(), w.data_ptr(), out.data_ptr(), m, n, k, lanes, contiguous)
            return out
        if self.mode == "exact":
            cache_key = ("mma", key)
            if cache_key not in self.converted:
                self.converted[cache_key] = weight.cpu().bfloat16().T.contiguous()
            w = self.converted[cache_key]
            x = x.cpu().bfloat16().contiguous()
            m, k = x.shape
            n = w.shape[1]
            assert (n, k) in ((1024, 2048), (2048, 512)) and m > 1
            chunk = 448 if k == 2048 and 17 <= m <= 32 else 1024 if k == 2048 and 33 <= m <= 64 else k
            total = torch.zeros((m, n), dtype=torch.float32)
            for begin in range(0, k, chunk):
                a, b = x[:, begin:begin+chunk].contiguous(), w[begin:begin+chunk].contiguous()
                out = torch.empty_like(total)
                self.cpu.reference_mma(a.data_ptr(), b.data_ptr(), out.data_ptr(), None, m, n, a.shape[1])
                total += out
            return total
        dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32,
                 "float64": torch.float64}[self.mode]
        cache_key = (self.mode, key)
        if cache_key not in self.converted:
            self.converted[cache_key] = weight.cpu().to(dtype).contiguous()
        return F.linear(x.cpu().to(dtype), self.converted[cache_key]).float()

    def activation(self, gate, up, split=False):
        assert split
        return F.silu(gate.cpu().bfloat16()) * up.cpu().bfloat16()

    def clear(self):
        self.packed.clear()
        self.converted.clear()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layers", default="0")
    p.add_argument("--positions", default="1,2,14,21,31")
    p.add_argument("--modes", default="dpbf16,bfloat16,float32,float64")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_torch()
    torch.use_deterministic_algorithms(True)
    probe = CpuProbe(args.output)
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True).text_config
    config._experts_implementation = "eager"
    layers = list(map(int, args.layers.split(",")))
    positions = list(map(int, args.positions.split(",")))
    modes = args.modes.split(",")
    fixtures = []
    for file in sorted((ROOT / "outputs/native-step1-001/generated_cached").glob("*.cached.r0.pt")):
        hidden = torch.load(file, map_location="cpu", weights_only=False)["hidden"]
        fixtures.append((file.name, {k: v for k, v in hidden.items() if any(
            k == f"layer.{layer}.{suffix}" for layer in layers for suffix in
            ("post_attention_layernorm", "router_ids", "router_weights", "mlp"))}))
    dump(args.output / "controls.json", {"layers": layers, "positions": positions, "modes": modes,
         "activation": "torch CPU BF16 SiLU", "candidate_expert_compute": "CPU only",
         "scope": "local decode; fixed golden input and routing; native golden remains immutable"})
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
            totals = {mode: {"absolute_sum": 0., "elements": 0, "different": 0} for mode in modes}
            rows = []
            for label, hidden in fixtures:
                for pos in positions:
                    x = hidden[f"layer.{layer}.post_attention_layernorm"][pos:pos+1]
                    ids = hidden[f"layer.{layer}.router_ids"][pos:pos+1]
                    route = hidden[f"layer.{layer}.router_weights"][pos:pos+1]
                    official = golden.experts(x.cuda(), ids.cuda(), route.cuda()).cpu()
                    shared = (golden.shared_expert(x.cuda()) * torch.sigmoid(golden.shared_expert_gate(x.cuda()))).cpu()
                    assert torch.equal(official + shared, hidden[f"layer.{layer}.mlp"][pos:pos+1])
                    row = {"case": label, "position": pos, "modes": {}}
                    for mode in modes:
                        probe.mode = mode
                        value = math.evaluate(x, ids, route, {mode: STAGES["07_full_cpu_down_dot"]})[mode]
                        delta = (value.double() - official.double()).abs()
                        total = totals[mode]
                        total["absolute_sum"] += delta.sum().item()
                        total["elements"] += delta.numel()
                        total["different"] += int((value != official).sum())
                        row["modes"][mode] = {"abs_mean": delta.mean().item(),
                                              "different": int((value != official).sum())}
                    rows.append(row)
            for total in totals.values():
                total["abs_mean"] = total["absolute_sum"] / total["elements"]
            result = {"layer": layer, "samples": len(rows), "seconds": time.time() - began,
                      "modes": totals}
            results.append(result)
            dump(args.output / f"layer-{layer:02}.json", {**result, "rows": rows})
            dump(args.output / "progress.json", results)
            print("LAYER", json.dumps(result), flush=True)
            probe.clear()
            del weights, golden, math
            gc.collect()
            torch.cuda.empty_cache()
    dump(args.output / "results.json", results)
    dump(args.output / "complete.json", {"success": True, "layers": layers})


if __name__ == "__main__":
    main()
