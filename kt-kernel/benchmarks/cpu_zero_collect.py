"""Record isolated real GEMV residuals for CPU reduction-order diagnosis."""
import argparse
from pathlib import Path
import json

import torch
import torch.nn.functional as F

from kt_to_golden_ablation import ArithmeticProbe, load_weights
from merge_replay_common import ROOT, configure_torch, dump


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layer", type=int, default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_torch()
    torch.use_deterministic_algorithms(True)
    probe = ArithmeticProbe(args.output)
    weights = load_weights(args.layer)
    wg = weights["experts.gate_up_proj"].cuda()
    wd = weights["experts.down_proj"].cuda()
    banks = {name: [] for name in ("gate_up", "down")}
    totals = {name: {"elements": 0, "fp64_different": 0, "dpbf16_different": 0,
                     "float_output_cast_different": 0} for name in banks}
    for file in sorted((ROOT / "outputs/native-step1-001/generated_cached").glob("*.cached.r0.pt")):
        hidden = torch.load(file, map_location="cpu", weights_only=False)["hidden"]
        for pos in (1, 2, 14, 21, 31):
            x = hidden[f"layer.{args.layer}.post_attention_layernorm"][pos:pos+1].cuda()
            ids = hidden[f"layer.{args.layer}.router_ids"][pos].tolist()
            for expert in ids:
                gu = F.linear(x, wg[expert])
                g, u = gu.chunk(2, -1)
                h = F.silu(g) * u
                for component, a, w, golden in (("gate_up", x, wg[expert], gu),
                                               ("down", h, wd[expert], F.linear(h, wd[expert]))):
                    a_cpu, w_cpu, golden = a.cpu(), w.cpu(), golden.cpu()
                    fp64 = F.linear(a_cpu.double(), w_cpu.double())
                    raw = probe.linear(a_cpu, w_cpu, (component, expert))
                    gpu32 = torch.mm(a, w.T, out_dtype=torch.float32).cpu()
                    total = totals[component]
                    total["elements"] += golden.numel()
                    total["fp64_different"] += int((fp64.float().bfloat16() != golden).sum())
                    total["dpbf16_different"] += int((raw.bfloat16() != golden).sum())
                    total["float_output_cast_different"] += int((gpu32.bfloat16() != golden).sum())
                    mismatch = ((fp64.float().bfloat16() != golden) | (raw.bfloat16() != golden)).flatten()
                    # Include controls unrelated to a mismatch to detect overfitting.
                    mismatch[torch.arange(0, w.shape[0], 257)] = True
                    columns = mismatch.nonzero().flatten()
                    banks[component].append({"case": file.name, "position": pos, "expert": expert,
                        "columns": columns, "input": a_cpu[0], "weight": w_cpu[columns],
                        "golden": golden[0, columns], "gpu32": gpu32[0, columns],
                        "fp64": fp64[0, columns], "dpbf16": raw[0, columns]})
        print(file.name, json.dumps(totals), flush=True)
    for name, bank in banks.items():
        torch.save(bank, args.output / f"{name}.pt")
    dump(args.output / "results.json", {"layer": args.layer, "totals": totals,
         "scope": "same real golden input for each projection, no propagated candidate differences"})


if __name__ == "__main__":
    main()
