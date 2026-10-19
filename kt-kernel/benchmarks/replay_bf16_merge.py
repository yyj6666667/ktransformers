"""Replay fixed golden tokens through the actual KT model; captures are not a timing benchmark.

KT_MERGE_FIXTURE_ROOT points to retained model/data fixtures. Use the same
script and fixtures for before/after builds, selected only through PYTHONPATH.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from merge_replay_common import ROOT, MODEL, Capture, cases, clone_cpu, configure_torch, dump, record_runtime, sha


def rng_digest():
    values = {
        "python": pickle.dumps(random.getstate()),
        "numpy": pickle.dumps(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().numpy().tobytes(),
        "torch_cuda_local": torch.cuda.get_rng_state().cpu().numpy().tobytes(),
    }
    return {key: hashlib.sha256(value).hexdigest() for key, value in values.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["native", "kt"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--case-limit", type=int, default=8)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == 4
    assert os.environ.get("PYTHONHASHSEED") == "42"
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    from datetime import timedelta
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    configure_torch()
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    selected = cases()[:args.case_limit]
    assert len(selected) == args.case_limit and 0 < args.max_new_tokens <= 256
    assert all(len(case["prompt"]) + args.max_new_tokens <= 512 for case in selected)

    from transformers import AutoConfig, AutoTokenizer, Qwen3_5MoeForConditionalGeneration
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as implementation
    for name in ["causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule",
                 "fused_recurrent_gated_delta_rule", "FusedRMSNormGated"]:
        setattr(implementation, name, None)
    implementation.is_fast_path_available = False
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    config._attn_implementation = config.text_config._attn_implementation = "eager"
    config._experts_implementation = config.text_config._experts_implementation = "eager"
    assert config.text_config.num_hidden_layers == 40
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    generation_config = json.loads((MODEL / "generation_config.json").read_text())
    eos = generation_config["eos_token_id"]
    eos = set(eos if isinstance(eos, list) else [eos])

    if args.backend == "native":
        for name in ("kt_kernel", "sglang", "fla"):
            assert importlib.util.find_spec(name) is None
        device_map = {"": "cpu"}
    else:
        from transformers.integrations.kt import HfTrainerKTConfig, unset_kt_config
        from kt_kernel.sft import KTConfig
        kt = KTConfig(kt_backend="AMXBF16_SkipLoRA", kt_num_gpu_experts=0,
            kt_num_threads=48, kt_tp_enabled=True, kt_threadpool_count=2,
            kt_max_cache_depth=1, kt_share_backward_bb=True, kt_weight_path=str(MODEL),
            kt_lora_rank=1, kt_lora_alpha=1.0, kt_model_max_length=512,
            kt_skip_expert_loading=True, kt_force_fused_expert_lora=False, kt_use_lora_experts=False)
        os.environ["ACCELERATE_USE_KT"] = "true"
        holder = HfTrainerKTConfig(kt)
        device_map = {"": local_rank}

    if rank == 0:
        record_runtime(args.output, {
            "backend": args.backend, "distribution": "fsdp2", "precision": "bf16",
            "world_size": world, "gdn_selection": "torch", "attention": "eager",
            "sampling": "greedy_argmax_no_temperature_or_logit_processors",
            "max_new_tokens": args.max_new_tokens, "case_limit": args.case_limit,
            "reference": str(args.reference) if args.reference else None,
            "scope": "same HF model with native/KT experts; not the SGLang serving backend",
            "numpy_seed": 42, "pythonhashseed": os.environ["PYTHONHASHSEED"],
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "model_implementation_sha256": sha(implementation.__file__),
            "generation_config_sha256": sha(MODEL / "generation_config.json"),
        })
    print("LOADING", args.backend, rank, flush=True)
    started = time.time()
    model, loading = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        MODEL, config=config, dtype=torch.bfloat16, local_files_only=True,
        device_map=device_map, low_cpu_mem_usage=True, output_loading_info=True)
    if args.backend == "kt":
        unset_kt_config()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert not any("lora" in name.lower() for name, _ in model.named_parameters())
    assert not loading.get("missing_keys"), loading
    assert all("mtp" in key for key in loading.get("unexpected_keys", [])), loading
    text_model = model.model.language_model
    assert len(text_model.layers) == 40
    audit = {"backend": args.backend, "rank": rank, "load_seconds": time.time() - started,
             "loading_info": loading, "lora_parameters": [], "optimizer_steps": 0,
             "gdn_implementations": sorted({layer.linear_attn.chunk_gated_delta_rule.__name__
                 for layer in text_model.layers if hasattr(layer, "linear_attn")}),
             "gdn_norm_classes": sorted({type(layer.linear_attn.norm).__name__
                 for layer in text_model.layers if hasattr(layer, "linear_attn")})}
    assert audit["gdn_implementations"] == ["torch_chunk_gated_delta_rule"]
    if args.backend == "kt":
        import kt_kernel
        audit["extension"] = kt_kernel.kt_kernel_ext.__file__
        audit["extension_sha256"] = sha(kt_kernel.kt_kernel_ext.__file__)
    ignored = set()
    if args.backend == "kt":
        assert len(model._kt_wrappers) == 40
        for wrapper in model._kt_wrappers:
            kernel = wrapper.wrapper
            # The distributed KT wrapper owns the CPU kernel on rank 0 only.
            if rank == 0:
                assert kernel._is_skip_lora and not kernel._lora_initialized
                for name in ("gate_lora_a", "gate_lora_b", "up_lora_a", "up_lora_b", "down_lora_a", "down_lora_b"):
                    assert getattr(kernel, name) is None
            else:
                assert kernel is None
            experts = getattr(wrapper, getattr(wrapper, "_experts_attr", "experts"), None)
            if experts is not None:
                ignored.update(experts.parameters())
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.device_mesh import init_device_mesh
    mesh = init_device_mesh("cuda", (world,))
    precision = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                     cast_forward_inputs=True)
    for layer in text_model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=precision, reshard_after_forward=True, ignored_params=ignored)
    fully_shard(model, mesh=mesh, mp_policy=precision, reshard_after_forward=True, ignored_params=ignored)
    audit["ignored_cpu_parameters"] = len(ignored)
    audit["fsdp_layers"] = sum("FSDP" in type(layer).__name__ for layer in text_model.layers)
    audit["rng_before_inference"] = rng_digest()
    dump(args.output / f"load-audit.rank{rank}.json", audit)
    device = torch.device("cuda", local_rank)
    capture = Capture()
    if rank == 0:
        capture.install_hf(text_model)
    rank_deltas = []

    def check_ranks(logits):
        reference = logits.clone()
        dist.broadcast(reference, src=0)
        delta = (reference - logits).abs().max()
        dist.all_reduce(delta, op=dist.ReduceOp.MAX)
        value = delta.item()
        rank_deltas.append(value)
        assert value == 0.0, f"Cross-rank logits differ: {value}"

    def save_record(folder, case, mode, response, logits, hidden, steps):
        assert logits.shape == (len(response), 248320) and torch.isfinite(logits).all()
        if rank == 0:
            folder.mkdir(parents=True, exist_ok=True)
            targets = torch.tensor(response, dtype=torch.long)
            logprobs = torch.log_softmax(logits.double(), dim=-1).gather(1, targets[:, None]).squeeze(1)
            torch.save({"prompt": case["prompt"], "response": response, "logits": logits,
                        "hidden": hidden, "logprobs_fp64": logprobs}, folder / f'{case["label"]}.{mode}.r0.pt')
            for i, step in enumerate(steps):
                step["target_logprob_fp64"] = logprobs[i].item()
            dump(folder / f'{case["label"]}.steps.json', steps)

    def cached(case, folder, generate):
        prompt = case["prompt"]
        n = len(prompt)
        limit = args.max_new_tokens if generate else len(case["response"])
        response, logits_parts, hidden_parts, steps = [], [], {}, []
        cache = None
        for position in range(limit):
            values = prompt if position == 0 else [response[-1]]
            offset = 0 if position == 0 else n + position - 1
            ids = torch.tensor([values], dtype=torch.long, device=device)
            capture.begin(len(values), len(values) - 1, 1)
            began = time.time()
            result = model(input_ids=ids,
                attention_mask=torch.ones((1, n + position), dtype=torch.long, device=device),
                position_ids=torch.arange(offset, offset + len(values), device=device).unsqueeze(0),
                use_cache=True, past_key_values=cache, return_dict=True)
            cache = result.past_key_values
            assert cache is not None and cache.get_seq_length() == n + position
            logits = result.logits[0, -1].detach().float()
            check_ranks(logits)
            top_values, top_ids = torch.topk(logits, 2)
            greedy_id = int(torch.argmax(logits).item())
            target = greedy_id if generate else case["response"][position]
            steps.append({"position": position, "phase": "prefill" if position == 0 else "decode",
                "input_tokens": len(values), "context_length": n + position,
                "context_ids": prompt + response, "target": target,
                "greedy_id": greedy_id, "top2_ids": top_ids.tolist(),
                "top2_logits": top_values.tolist(), "greedy_logit_margin": (top_values[0] - top_values[1]).item(),
                "generation": generate, "seconds_including_capture": time.time() - began})
            logits_parts.append(clone_cpu(logits))
            for key, value in capture.end().items():
                hidden_parts.setdefault(key, []).append(value)
            response.append(target)
            del result
            if rank == 0 and (position == 0 or (position + 1) % 8 == 0):
                print("TOKEN", folder.name, case["label"], position, target, flush=True)
            if generate and target in eos:
                break
        del cache
        hidden = {key: torch.cat(value, dim=0) for key, value in hidden_parts.items()}
        save_record(folder, case, "cached", response, torch.stack(logits_parts), hidden, steps)
        return {**case, "response": response, "response_text": tokenizer.decode(response, skip_special_tokens=False),
                "response_origin": "greedy_generation" if generate else "fixed_reference_replay",
                "stop_reason": "eos" if response[-1] in eos else ("max_new_tokens" if generate else "reference_end")}

    def finish_phase(folder, fixture):
        dist.barrier()
        if rank == 0:
            dump(folder / "cases.json", fixture)
            runtime = json.loads((args.output / "runtime.json").read_text())
            runtime["cases_sha256"] = sha(folder / "cases.json")
            runtime["phase"] = folder.name
            dump(folder / "runtime.json", runtime)
            dump(folder / "complete.json", {"cases": len(fixture), "response_positions": sum(len(c["response"]) for c in fixture),
                                           "max_rank_logit_difference": max(rank_deltas), "exit": "success"})

    with torch.no_grad():
        reference = json.loads(args.reference.read_text())
        assert [c["label"] for c in reference] == [c["label"] for c in selected]
        assert all(a["prompt"] == b["prompt"] for a, b in zip(reference, selected))
        for case in reference:
            cached(case, args.output / "reference_cached", False)
        finish_phase(args.output / "reference_cached", reference)
    audit["rng_after_inference"] = rng_digest()
    audit["rng_unchanged_during_inference"] = audit["rng_before_inference"] == audit["rng_after_inference"]
    dump(args.output / f"load-audit.rank{rank}.json", audit)
    dist.barrier()
    if rank == 0:
        dump(args.output / "complete.json", {"exit": "success", "cases": len(reference),
            "replayed_tokens": sum(len(c["response"]) for c in reference),
            "max_rank_logit_difference": max(rank_deltas), "no_lora": True, "optimizer_steps": 0,
            "seed": 42, "deterministic_algorithms": True,
            "note": "Fresh execution; no independent process restart verified by this run alone."})
        print("COMPLETE", args.backend, str(args.output), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
