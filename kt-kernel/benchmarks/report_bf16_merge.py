"""Summarize operator timing and per-layer absolute error against frozen golden records."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import torch


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--original-kt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    performance = []
    raw = [json.loads(p.read_text()) for p in sorted((args.experiment / "performance").glob("*.json"))]
    assert len(raw) == 8
    assert all(not r["profiling"] and not r["save_for_backward"] for r in raw)
    assert len({tuple(r["weight_hashes"]) for r in raw}) == 1
    for family in ("sft", "inference"):
        runs = [r for r in raw if r["family"] == family]
        assert len(runs) == 4
        for index, tokens in enumerate((1, 4, 128, 512)):
            assert len({tuple(r["results"][index]["input_hashes"]) for r in runs}) == 1
            values, medians = {}, {}
            for variant in ("before", "after"):
                selected = [r["results"][index] for r in runs if r["tag"].endswith(variant)]
                assert len(selected) == 2 and all(r["tokens"] == tokens for r in selected)
                assert len({r["output_sha256"] for r in selected}) == 1
                values[variant] = statistics.median([v for r in selected for v in r["samples_ms"]])
                medians[variant] = [r["median_ms"] for r in selected]
            performance.append(dict(family=family, tokens=tokens,
                phase="decode" if tokens <= 4 else "prefill", before_ms=values["before"], after_ms=values["after"],
                latency_change_pct=(values["after"] / values["before"] - 1) * 100,
                before_run_median_min=min(medians["before"]), before_run_median_max=max(medians["before"]),
                after_run_median_min=min(medians["after"]), after_run_median_max=max(medians["after"])))
    write_csv(args.output / "operator-performance.csv", performance)

    sums = {(v, group, layer): [0.0, 0] for v in ("before", "after")
            for group in ("all", "prefill", "decode") for layer in range(40)}
    original_different = original_elements = positions = cases = 0
    greedy = {"before": 0, "after": 0}
    manifests = []
    golden_files = sorted(args.golden.glob("*.cached.r0.pt"))
    assert len(golden_files) == 8
    for file in golden_files:
        gold = torch.load(file, map_location="cpu", weights_only=False)
        original = torch.load(args.original_kt / file.name, map_location="cpu", weights_only=False)
        cases += 1
        positions += len(gold["response"])
        manifests.append(dict(file=file.name, sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        for variant in ("before", "after"):
            value = torch.load(args.experiment / f"replay-{variant}/reference_cached" / file.name,
                               map_location="cpu", weights_only=False)
            assert value["prompt"] == gold["prompt"] and value["response"] == gold["response"]
            if variant == "before":
                for key in original["hidden"].keys() & value["hidden"].keys():
                    original_different += int((original["hidden"][key] != value["hidden"][key]).sum())
                    original_elements += original["hidden"][key].numel()
                original_different += int((original["logits"] != value["logits"]).sum())
                original_elements += original["logits"].numel()
            greedy[variant] += int((value["logits"].argmax(-1) != gold["logits"].argmax(-1)).sum())
            for layer in range(40):
                key = f"layer.{layer}.boundary"
                delta = (value["hidden"][key].double() - gold["hidden"][key].double()).abs()
                assert torch.isfinite(delta).all()
                for group, item in (("all", delta), ("prefill", delta[:1]), ("decode", delta[1:])):
                    acc = sums[variant, group, layer]
                    acc[0] += item.sum().item()
                    acc[1] += item.numel()
    assert original_different == 0, "rebuilt baseline differs from retained KT"
    rows = []
    for layer in range(40):
        row = dict(layer=layer)
        for group in ("all", "prefill", "decode"):
            for variant in ("before", "after"):
                total, count = sums[variant, group, layer]
                row[f"{variant}_{group}"] = total / count
        row["change_pct"] = (row["after_all"] / row["before_all"] - 1) * 100
        rows.append(row)
    write_csv(args.output / "layer-0-to-39-abs-mean.csv", rows)
    summary = dict(cases=cases, positions=positions, prefill_positions=cases, decode_positions=positions-cases,
                   metric="mean(abs(KT - golden)) at each layer output including residual",
                   prefill_scope="last prompt position only; decode uses the fixed golden prefix",
                   before_vs_original_kt=dict(different_elements=original_different, elements=original_elements),
                   greedy_argmax_differences=greedy, improved_layers=sum(r["after_all"] < r["before_all"] for r in rows),
                   performance=performance, layer_errors=rows, golden_record_manifest=manifests)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["Qwen3.5-35B-A3B, LoRA off, BF16 CPU MoE TP2 (2 x 24 threads)",
             "Layer output mean(abs(KT - golden)); 8 cases x 32 fixed-prefix positions.",
             "Prefill = last prompt position (8); decode = following 31 positions/case (248).",
             "Performance is simulated single-operator timing, NOT model throughput.", "",
             f"{'Layer':>5} | {'Before':>14} | {'After':>14} | {'Change %':>10}", "-"*55]
    lines += [f"{r['layer']:5d} | {r['before_all']:14.7e} | {r['after_all']:14.7e} | {r['change_pct']:10.3f}" for r in rows]
    lines += ["", "Operator timing (median ms); latency change < 0 means faster."]
    lines += [f"{r['family']:9} tokens={r['tokens']:3} {r['before_ms']:.6f} -> {r['after_ms']:.6f} ms "
              f"({r['latency_change_pct']:+.2f}%)" for r in performance]
    lines += ["", f"Rebuilt before vs retained KT: {original_different}/{original_elements} elements differ.",
              f"Argmax differences vs golden before/after: {greedy}; positions={positions}.",
              "No complete-answer accuracy/reward or RL optimizer-step evaluation is claimed."]
    (args.output / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
