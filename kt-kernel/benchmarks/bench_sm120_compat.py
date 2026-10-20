"""Standalone BF16 MoE operator benchmark; model loading and input copies excluded.

Run this identical script against both extension builds in separate processes.
The timed region includes task creation, CPUInfer submission, all TP workers,
expert merging, final BF16 conversion and sync. No GPU or full-model timing.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
import kt_kernel


class Operator:
    def __init__(self, gate, up, down, *, topk, max_len, pools=2, threads=48,
                 family="sft", full_weight_grad=False, cache_depth=1):
        ext = kt_kernel.kt_kernel_ext
        self.weights = (gate.contiguous(), up.contiguous(), down.contiguous())
        experts, intermediate, hidden = gate.shape
        self.topk, self.hidden, self.family = topk, hidden, family
        worker = ext.WorkerPoolConfig()
        worker.subpool_count = pools
        worker.subpool_numa_map = list(range(pools))
        worker.subpool_thread_count = [threads // pools] * pools
        assert threads % pools == 0
        self.cpu = ext.CPUInfer(worker)
        self.mapping = torch.arange(experts, dtype=torch.int64)
        if family == "inference":
            cfg = ext.moe.MOEConfig(experts, topk, hidden, intermediate, 0)
            cls = ext.moe.AMXBF16_MOE
        else:
            cfg = ext.moe.MOESFTConfig()
            cfg.expert_num, cfg.num_experts_per_tok = experts, topk
            cfg.hidden_size, cfg.intermediate_size = hidden, intermediate
            cfg.lora_rank = 0 if full_weight_grad else 1
            cfg.lora_alpha = 1.0
            cfg.max_cache_depth = cache_depth
            cfg.full_weight_grad = full_weight_grad
            cfg.authoritative_optimizer_grads = False
            cls = ext.moe.AMXBF16_SFT_MOE if full_weight_grad else ext.moe.AMXBF16_SFT_MOE_SkipLoRA
        cfg.max_len = (max_len + 31) // 32 * 32
        cfg.physical_to_logical_map = self.mapping.data_ptr()
        cfg.gate_proj, cfg.up_proj, cfg.down_proj = [w.data_ptr() for w in self.weights]
        cfg.pool = self.cpu.backend_
        self.moe = cls(cfg)
        self.cpu.submit(self.moe.load_weights_task())
        self.cpu.sync()
        self.full_weight_grad = full_weight_grad

    def prepare(self, x, ids, weights):
        self.x = x.contiguous()
        self.ids = ids.to(torch.int64).contiguous()
        self.route = weights.float().contiguous()
        self.qlen = torch.tensor([x.shape[0]], dtype=torch.int32)
        self.out = torch.empty_like(x)

    def forward(self, save=False):
        args = (self.qlen.data_ptr(), self.topk, self.ids.data_ptr(), self.route.data_ptr(),
                self.x.data_ptr(), self.out.data_ptr())
        task = (self.moe.forward_sft_task(*args, save) if self.family == "sft"
                else self.moe.forward_task(*args))
        self.cpu.submit(task)
        self.cpu.sync()
        return self.out

    def backward(self, dy):
        dy = dy.contiguous()
        dx = torch.empty_like(dy)
        dw = torch.zeros((len(dy), self.topk), dtype=torch.float32)
        base_grads = [torch.zeros_like(w) for w in self.weights] if self.full_weight_grad else []
        base_ptrs = [v.data_ptr() for v in base_grads] if base_grads else [0, 0, 0]
        self.cpu.submit(self.moe.backward_task(dy.data_ptr(), dx.data_ptr(), *([0] * 6),
                                              dw.data_ptr(), *base_ptrs))
        self.cpu.sync()
        return dx, dw


def tensor_hash(tensor):
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, help="reuse an identical retained synthetic weight/input fixture")
    parser.add_argument("--family", choices=["sft", "inference", "sm120"], default="sft")
    parser.add_argument("--tokens", default="1,4,128,512")
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--pools", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--tag", default="")
    parser.add_argument("--profile", action="store_true", help="diagnostic SFT stage timers; exclude from final timings")
    args = parser.parse_args()
    if args.profile and args.family != "sft":
        parser.error("--profile requires --family=sft")
    os.environ["KT_SFT_PROFILE"] = "1" if args.profile else "0"
    torch.set_num_threads(2)
    torch.manual_seed(42)
    tokens = list(map(int, args.tokens.split(",")))
    shape = dict(experts=256, hidden=2048, intermediate=512, topk=8)
    fixture = None
    if args.fixture and args.fixture.exists():
        fixture = torch.load(args.fixture, map_location="cpu", weights_only=True, mmap=True)
        gate, up, down = fixture["weights"]
    else:
        gate = torch.randn(256, 512, 2048, dtype=torch.bfloat16).mul_(0.02)
        up = torch.randn_like(gate).mul_(0.02)
        down = torch.randn(256, 2048, 512, dtype=torch.bfloat16).mul_(0.02)
        if args.fixture:
            fixture = {"weights": (gate, up, down), "inputs": {}}
            for qlen in tokens:
                x = torch.randn(qlen, 2048, dtype=torch.bfloat16)
                values, ids = torch.randn(qlen, 256).topk(8, dim=-1)
                route = values.softmax(-1).to(torch.bfloat16).float()
                fixture["inputs"][qlen] = (x, ids, route)
            args.fixture.parent.mkdir(parents=True, exist_ok=True)
            torch.save(fixture, args.fixture)
    if args.family == "sm120":
        from cpu_zero_operator import ExactOperator
        engine = ExactOperator(gate, up, down, topk=8, max_len=max(tokens), pools=args.pools, threads=args.threads)
    else:
        engine = Operator(gate, up, down, topk=8, max_len=max(tokens), pools=args.pools,
                          threads=args.threads, family=args.family)
    # Hash outside the timing region. Includes the complete synthetic weights.
    weight_hashes = [tensor_hash(w) for w in (gate, up, down)]
    results = []
    with torch.no_grad():
        for qlen in tokens:
            if fixture:
                x, ids, route = fixture["inputs"][qlen]
            else:
                x = torch.randn(qlen, 2048, dtype=torch.bfloat16)
                scores = torch.randn(qlen, 256)
                values, ids = scores.topk(8, dim=-1)
                route = values.softmax(-1).to(torch.bfloat16).float()
            engine.prepare(x, ids, route)
            for _ in range(args.warmup):
                engine.forward()
            if args.profile:
                engine.moe.reset_profile_stats()
            samples = []
            for _ in range(args.iterations):
                start = time.perf_counter_ns()
                engine.forward()
                samples.append((time.perf_counter_ns() - start) / 1e6)
            assert torch.isfinite(engine.out).all()
            row = dict(tokens=qlen, phase="decode" if qlen <= 4 else "prefill",
                       median_ms=statistics.median(samples), mean_ms=statistics.mean(samples),
                       samples_ms=samples, input_hashes=[tensor_hash(t) for t in (x, ids, route)],
                       output_sha256=tensor_hash(engine.out))
            if args.profile:
                row["stage_profile"] = dict(engine.moe.get_profile_stats())
            results.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "samples_ms"}), flush=True)
    extension = Path(kt_kernel.kt_kernel_ext.__file__)
    data = dict(tag=args.tag, family=args.family, shape=shape, threads=args.threads, pools=args.pools,
                fixture=str(args.fixture), interval_certificate=os.environ.get("KT_SM120_INTERVAL", "1"),
                seed=42, no_lora=True, save_for_backward=False, warmup=args.warmup,
                iterations=args.iterations, profiling=args.profile, torch=torch.__version__, cpu_variant=kt_kernel.__cpu_variant__,
                extension=str(extension), extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
                weight_hashes=weight_hashes, results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
