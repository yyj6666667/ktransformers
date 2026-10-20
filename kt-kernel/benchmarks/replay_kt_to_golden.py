"""Propagate KT-to-golden arithmetic ablations through a frozen HF model.

This is a diagnostic replay of KT arithmetic in the expert slot, not an optimized
KT implementation. Both endpoint controls are checked against retained raw runs.
The original golden implementation and records are never edited.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import time
import types

import torch

from merge_replay_common import ROOT, MODEL, Capture, configure_torch, dump, record_runtime, sha
from kt_to_golden_ablation import ArithmeticProbe, LayerMath, STAGES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stages", default=",".join(k for k in STAGES if k[:2].isdigit()))
    parser.add_argument("--case-limit", type=int, default=8)
    parser.add_argument("--max-positions", type=int, default=32)
    parser.add_argument("--freeze-routing", action="store_true",
                        help="diagnostic: use a separately replayed golden's full expert IDs and weights")
    parser.add_argument("--capture-prefill-experts", action="store_true",
                        help="save full expert inputs/outputs only in the immutable golden endpoint control")
    parser.add_argument("--kt-sm120", action="store_true", help="use the actual KT SM120BF16_MOE CPU operator")
    parser.add_argument("--cpu-exact", action="store_true",
                        help="diagnostic all-CPU expert arithmetic matched to the frozen SM120 reference")
    parser.add_argument("--cpu-prefill-reference", type=Path,
                        help="assert all prompt expert inputs/routing/outputs against separate golden captures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_torch()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    if args.kt_sm120:
        assert not args.cpu_exact
        STAGES["12_kt_sm120"] = STAGES["07_full_cpu_down_dot"]
    if args.cpu_exact:
        STAGES["11_cpu_exact"] = STAGES["07_full_cpu_down_dot"]
    stages = args.stages.split(",")
    assert all(name in STAGES for name in stages)
    if args.capture_prefill_experts:
        assert stages == ["10_cuda_down"] and not args.freeze_routing and args.max_positions == 1
    if args.cpu_prefill_reference:
        assert ((args.cpu_exact and stages == ["11_cpu_exact"]) or (args.kt_sm120 and stages == ["12_kt_sm120"])) and not args.freeze_routing
    if args.freeze_routing:
        assert len(stages) == 1 and stages[0] not in ("00_original_kt", "01_remove_partial_round", "10_cuda_down")
    assert 0 < args.max_positions <= 32 and 0 < args.case_limit <= 8
    assert torch.cuda.device_count() == 4
    from transformers import AutoConfig, Qwen3_5MoeForConditionalGeneration
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as implementation
    for name in ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule",
                 "fused_recurrent_gated_delta_rule", "FusedRMSNormGated"):
        setattr(implementation, name, None)
    implementation.is_fast_path_available = False
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    config._attn_implementation = config.text_config._attn_implementation = "eager"
    config._experts_implementation = config.text_config._experts_implementation = "eager"
    device_map = {"model.visual": 0, "model.language_model.embed_tokens": 0,
                  "model.language_model.rotary_emb": 0, "model.language_model.norm": 3, "lm_head": 3}
    for layer in range(40):
        device_map[f"model.language_model.layers.{layer}"] = layer // 10
    extra_backend = {}
    if args.kt_sm120:
        import kt_kernel
        extra_backend = {"extension": kt_kernel.kt_kernel_ext.__file__,
                         "extension_sha256": sha(kt_kernel.kt_kernel_ext.__file__),
                         "interval_certificate": os.environ.get("KT_SM120_INTERVAL", "0")}
    record_runtime(args.output, {"scope": ("actual KT CPU operator forward; immutable golden target" if args.kt_sm120
                                          else "KT arithmetic replay; immutable golden target"),
        "backend": extra_backend,
        "distribution": "four GPU layerwise; endpoint controls validate against original FSDP2 captures",
        "model_implementation_sha256": sha(implementation.__file__), "stages": stages,
        "arithmetic": {name: vars(STAGES[name]) for name in stages},
        "max_positions": args.max_positions, "case_limit": args.case_limit, "freeze_routing": args.freeze_routing,
        "kt_sm120": args.kt_sm120, "cpu_exact": args.cpu_exact, "cpu_prefill_reference": args.cpu_prefill_reference})
    model, loading = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        MODEL, config=config, dtype=torch.bfloat16, local_files_only=True,
        device_map=device_map, low_cpu_mem_usage=True, output_loading_info=True)
    assert not loading.get("missing_keys") and all("mtp" in k for k in loading.get("unexpected_keys", [])), loading
    model.eval().requires_grad_(False)
    assert not any("lora" in name.lower() for name, _ in model.named_parameters())
    text_model = model.model.language_model
    assert len(text_model.layers) == 40
    if args.kt_sm120:
        from cpu_zero_operator import ExactOperator, make_cpu
        shared_cpu = make_cpu()
        probe = None
    elif args.cpu_exact:
        from cpu_zero_local import CpuProbe
        probe = CpuProbe(args.output)
        probe.mode = "exact"
    else:
        probe = ArithmeticProbe(args.output)
    active = {"stage": stages[0], "phase": "candidate"}
    golden_routing = {}
    prefill_experts = {}
    full_prefill_elements = 0
    for layer_idx, layer in enumerate(text_model.layers):
        if args.kt_sm120:
            gu = layer.mlp.experts.gate_up_proj
            arithmetic = ExactOperator(gu[:, :512], gu[:, 512:], layer.mlp.experts.down_proj,
                                       max_len=224, cpu=shared_cpu)
            print("PACKED KT", layer_idx, flush=True)
        else:
            arithmetic = LayerMath(layer.mlp.experts.gate_up_proj, layer.mlp.experts.down_proj, probe, layer_idx)

        def forward(self, hidden, ids, route, _math=arithmetic, _layer=layer_idx):
            nonlocal full_prefill_elements
            name = active["stage"]
            if args.freeze_routing:
                if active["phase"] == "golden":
                    golden_routing[_layer] = (ids.clone(), route.clone())
                    name = "10_cuda_down"
                else:
                    ids, route = golden_routing[_layer]
            value = _math.evaluate(hidden, ids, route, {name: STAGES[name]})[name]
            reference = active.get("prefill_reference")
            if reference is not None:
                for field, candidate in (("input", hidden), ("ids", ids), ("route", route), ("output", value)):
                    target = reference[_layer][field]
                    assert torch.equal(candidate.cpu(), target), (_layer, field, "full prefill expert mismatch")
                    full_prefill_elements += candidate.numel()
            if args.capture_prefill_experts:
                prefill_experts[_layer] = {"input": hidden.cpu(), "ids": ids.cpu(),
                                           "route": route.cpu(), "output": value.cpu()}
            return value.to(hidden.device)

        layer.mlp.experts.forward = types.MethodType(forward, layer.mlp.experts)
    capture = Capture()
    capture.install_hf(text_model)
    fixtures = json.loads((ROOT / "outputs/native-step1-001/generated-cases.json").read_text())[:args.case_limit]
    controls = {"00_original_kt": ROOT / "outputs/kt-step1-002/reference_cached",
                "01_remove_partial_round": ROOT / "fp32-down-merge/compact-final/replay-after/reference_cached",
                "10_cuda_down": ROOT / "outputs/native-step1-001/generated_cached"}
    if args.kt_sm120:
        controls["12_kt_sm120"] = controls["10_cuda_down"]
    if args.cpu_exact:
        controls["11_cpu_exact"] = controls["10_cuda_down"]
    all_results = []
    with torch.no_grad():
        for name in stages:
            active["stage"] = name
            folder = args.output / name
            folder.mkdir()
            totals = {layer: [0., 0] for layer in range(40)}
            different = elements = positions = argmax = 0
            began = time.time()
            for case in fixtures:
                prompt, response = case["prompt"], case["response"][:args.max_positions]
                filename = case["label"] + ".cached.r0.pt"
                golden = torch.load(ROOT / "outputs/native-step1-001/generated_cached" / filename,
                                    map_location="cpu", weights_only=False)
                cached, hidden_parts, logits = None, {}, []
                golden_cache = None
                full_prefill = (torch.load(args.cpu_prefill_reference / filename, map_location="cpu", weights_only=False)
                                if args.cpu_prefill_reference else None)
                for pos in range(len(response)):
                    active["prefill_reference"] = full_prefill if pos == 0 else None
                    values = prompt if pos == 0 else [response[pos - 1]]
                    offset = 0 if pos == 0 else len(prompt) + pos - 1
                    ids = torch.tensor([values], device="cuda:0", dtype=torch.long)
                    if args.freeze_routing:
                        active["phase"] = "golden"
                        reference = model(input_ids=ids,
                            attention_mask=torch.ones((1, len(prompt)+pos), device="cuda:0", dtype=torch.long),
                            position_ids=torch.arange(offset, offset+len(values), device="cuda:0").unsqueeze(0),
                            use_cache=True, past_key_values=golden_cache, return_dict=True)
                        golden_cache = reference.past_key_values
                        assert torch.equal(reference.logits[0, -1].float().cpu(), golden["logits"][pos]), (
                            name, case["label"], pos, "routing donor differs from original golden")
                        for layer, (donor_ids, donor_weights) in golden_routing.items():
                            assert torch.equal(donor_ids[-1].cpu(), golden["hidden"][f"layer.{layer}.router_ids"][pos])
                            assert torch.equal(donor_weights[-1].cpu(), golden["hidden"][f"layer.{layer}.router_weights"][pos])
                        del reference
                        active["phase"] = "candidate"
                    capture.begin(len(values), len(values) - 1, 1)
                    output = model(input_ids=ids,
                        attention_mask=torch.ones((1, len(prompt)+pos), device="cuda:0", dtype=torch.long),
                        position_ids=torch.arange(offset, offset+len(values), device="cuda:0").unsqueeze(0),
                        use_cache=True, past_key_values=cached, return_dict=True)
                    cached = output.past_key_values
                    logits.append(output.logits[0, -1].float().cpu())
                    for key, value in capture.end().items():
                        hidden_parts.setdefault(key, []).append(value)
                    del output
                record = {"prompt": prompt, "response": response, "logits": torch.stack(logits),
                          "hidden": {key: torch.cat(value) for key, value in hidden_parts.items()}}
                torch.save(record, folder / filename)
                if args.capture_prefill_experts:
                    capture_folder = args.output / "prefill-experts"
                    capture_folder.mkdir(exist_ok=True)
                    torch.save(prefill_experts, capture_folder / filename)
                    prefill_experts.clear()
                for layer in range(40):
                    key = f"layer.{layer}.boundary"
                    delta = (record["hidden"][key].double() - golden["hidden"][key][:len(response)].double()).abs()
                    assert torch.isfinite(delta).all()
                    totals[layer][0] += delta.sum().item()
                    totals[layer][1] += delta.numel()
                argmax += int((record["logits"].argmax(-1) != golden["logits"][:len(response)].argmax(-1)).sum())
                positions += len(response)
                if name in controls:
                    target = torch.load(controls[name] / filename, map_location="cpu", weights_only=False)
                    pairs = [(record["logits"], target["logits"][:len(response)])]
                    pairs += [(record["hidden"][key], target["hidden"][key][:len(response)])
                              for key in record["hidden"].keys() & target["hidden"].keys()]
                    for a, b in pairs:
                        different += int((a != b).sum())
                        elements += a.numel()
                    assert different == 0, (name, case["label"], different, "endpoint control failed")
                    del target
                print("CASE", name, case["label"], "elapsed", round(time.time()-began, 2), flush=True)
                del cached, golden_cache, golden, record, logits, hidden_parts
            result = {"stage": name, "cases": len(fixtures), "positions": positions,
                      "seconds": time.time()-began, "argmax_differences": argmax,
                      "abs_mean_by_layer": {layer: total/count for layer, (total, count) in totals.items()},
                      "endpoint_control": {"different": different, "elements": elements} if name in controls else None,
                      "freeze_routing": args.freeze_routing}
            if args.cpu_prefill_reference:
                result["full_prefill_expert_control"] = {"different": 0, "elements": full_prefill_elements}
            dump(folder / "summary.json", result)
            all_results.append(result)
            dump(args.output / "progress.json", all_results)
            print("STAGE", name, "layer39", result["abs_mean_by_layer"][39], flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    dump(args.output / "results.json", all_results)
    dump(args.output / "complete.json", {"exit": "success", "stages": stages, "positions_per_stage": positions})


if __name__ == "__main__":
    main()
