"""Report immutable-golden ablations without mixing local and propagated errors."""

import argparse
import csv
import json
from pathlib import Path


LABELS = {
    "00_original_kt": "Original KT",
    "01_remove_partial_round": "Remove TP-part BF16 rounding",
    "02_round_complete_expert": "Round each complete expert after TP sum",
    "03_round_silu": "Also round SiLU before multiplying up",
    "04_round_weighted_product": "Also round expert times routing weight",
    "05_round_expert_accumulation": "Also accumulate experts in BF16",
    "06_expert_id_order": "Also use ascending expert-ID order",
    "07_full_cpu_down_dot": "Also use one full CPU down dot",
    "08_cuda_silu": "Also use golden CUDA SiLU",
    "09_cuda_gate_up": "Also use golden CUDA gate/up GEMM",
    "10_cuda_down": "Also use golden CUDA down GEMM (endpoint control)",
}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--screen", type=Path)
    parser.add_argument("--full", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--merge-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    local = json.loads((args.local / "results.json").read_text())
    assert len(local) == 40 and all(v["actual_control"]["different"] == 0 for v in local)
    controls = json.loads((args.local / "controls.json").read_text())
    mean = {name: sum(v["abs_mean"][name] for v in local)/len(local) for name in local[0]["abs_mean"]}
    rows = [{"stage": name, "description": LABELS.get(name, name), "local_mean_abs": value,
             "local_reduction_pct": (1-value/mean["00_original_kt"])*100}
            for name, value in mean.items()]
    screen = json.loads((args.screen / "results.json").read_text()) if args.screen else []
    full = json.loads((args.full / "results.json").read_text()) if args.full else []
    routing = None
    if args.merge_reference:
        baseline = json.loads(args.merge_reference.read_text())
        for key, prefix in (("00_original_kt", "before"), ("01_remove_partial_round", "after")):
            existing = next((value for value in full if value["stage"] == key), None)
            if existing is not None:
                assert existing["positions"] == baseline["positions"]
                assert all(existing["abs_mean_by_layer"][str(v["layer"])] == v[prefix+"_all"]
                           for v in baseline["layer_errors"]), "actual KT control differs"
                continue
            full.append({"stage": key, "positions": baseline["positions"],
                "abs_mean_by_layer": {str(v["layer"]): v[prefix+"_all"] for v in baseline["layer_errors"]},
                "source": "Actual KT operator, retained before/after matched replay"})
    for label, records in (("screen", screen), ("full", full)):
        if not records:
            continue
        assert len({v["positions"] for v in records}) == 1
        by_name = {v["stage"]: v for v in records}
        baseline = by_name["00_original_kt"]["abs_mean_by_layer"]["39"]
        layer_rows = []
        for layer in range(40):
            layer_rows.append({"layer": layer, **{v["stage"]: v["abs_mean_by_layer"][str(layer)] for v in records}})
        write_csv(args.output / f"{label}-layer-0-to-39.csv", layer_rows)
        for row in rows:
            record = by_name.get(row["stage"])
            row[label+"_positions"] = record["positions"] if record else ""
            row[label+"_layer39_abs_mean"] = record["abs_mean_by_layer"]["39"] if record else ""
            row[label+"_layer39_reduction_pct"] = (1-record["abs_mean_by_layer"]["39"]/baseline)*100 if record else ""
    write_csv(args.output / "stage-summary.csv", rows)
    write_csv(args.output / "local-layer-0-to-39.csv",
              [{"layer": v["layer"], **v["abs_mean"]} for v in local])
    if args.frozen:
        frozen, = json.loads((args.frozen / "results.json").read_text())
        assert frozen["freeze_routing"]
        free = next(v for v in full if v["stage"] == frozen["stage"])
        assert free["positions"] == frozen["positions"]
        before, after = free["abs_mean_by_layer"]["39"], frozen["abs_mean_by_layer"]["39"]
        routing = {"stage": frozen["stage"], "positions": frozen["positions"],
                   "normal_routing_layer39": before, "frozen_routing_layer39": after,
                   "reduction_pct": (1-after/before)*100,
                   "intervention": "both expert IDs and route weights fixed from separately replayed golden",
                   "scope": "causal diagnostic, not a deployable fix"}
        write_csv(args.output / "routing-intervention-by-layer.csv", [
            {"layer": layer, "normal_routing": free["abs_mean_by_layer"][str(layer)],
             "frozen_golden_routing": frozen["abs_mean_by_layer"][str(layer)]} for layer in range(40)])
    summary = {"metric": "mean(abs(KT variant - unchanged original golden))", "local_cases_per_layer": local[0]["samples"],
               "local_actual_kt_control": {"different": 0, "elements": sum(v["actual_control"]["elements"] for v in local)},
               "local_input_protocol": controls["scope"], "stages": controls["stages"], "rows": rows,
               "screen": screen, "full": full, "routing_intervention": routing,
               "scope": "Diagnostic arithmetic replay, no production fix, no performance claim, no backward or reward evaluation"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    lines = ["KT -> unchanged original golden: arithmetic ablations", "",
             "LOCAL: 40 layers x 40 frozen decode inputs; each layer gets its own original golden input/routing.",
             "PROPAGATED: same fixed prefixes through the complete model; see position counts in separate columns.",
             "Local improvement does not imply the same propagated improvement.", "",
             f"{'Stage':35} {'Local abs mean':>16} {'Local decrease %':>18}"]
    lines.extend(f"{r['stage']:35} {r['local_mean_abs']:16.9e} {r['local_reduction_pct']:18.4f}" for r in rows)
    for label, records in (("SCREEN", screen), ("FULL", full)):
        if records:
            lines += ["", f"{label}: complete-model layer 39, positions per variant = {records[0]['positions']}"]
            lines.extend(f"{v['stage']:35} {v['abs_mean_by_layer']['39']:.9e}" for v in sorted(records, key=lambda x: x['stage']))
    if routing:
        lines += ["", f"ROUTING INTERVENTION: {routing['positions']} positions; only down GEMM remains CPU.",
                  f"Layer 39: {before:.9e} -> {after:.9e}; decrease {routing['reduction_pct']:.4f}%.",
                  "Both expert selection and routing weights are frozen jointly; this does not isolate them separately."]
    lines += ["", "Stage names 00..10 are cumulative. Other rows are non-cumulative controls.",
              "GPU replacements are diagnostic endpoint controls, not a CPU-kernel fix.",
              "The golden implementation/weights/data are fixed. New switches act only on the KT arithmetic replay."]
    (args.output / "RESULTS.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
