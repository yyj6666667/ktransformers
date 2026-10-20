"""Held-out native-HF comparison across every supported expert batch size."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
import kt_kernel
from cpu_zero_operator import ExactOperator
from merge_replay_common import MODEL, configure_torch, dump, sha

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--pools", default="1,2")
p.add_argument("--rows", default="all")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
configure_torch()
torch.use_deterministic_algorithms(True)
torch.manual_seed(80923)
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True).text_config
cfg.num_experts = cfg.num_experts_per_tok = 8
cfg._experts_implementation = "eager"
gu = torch.randn(8, 1024, 2048, dtype=torch.bfloat16).mul_(0.02)
down = torch.randn(8, 2048, 512, dtype=torch.bfloat16).mul_(0.02)
with torch.device("meta"):
    golden = Qwen3_5MoeSparseMoeBlock(cfg).experts
golden.load_state_dict({"gate_up_proj": gu, "down_proj": down}, assign=True, strict=True)
golden = golden.cuda().eval()
x = torch.randn(224, 2048, dtype=torch.bfloat16)
ids = torch.randn(224, 8).argsort(-1)
route = torch.randn(224, 8).softmax(-1).bfloat16()
mapping = torch.randperm(8)
physical_ids = torch.argsort(mapping)[ids]
sizes = list(range(1, 225)) if a.rows == "all" else list(map(int, a.rows.split(",")))
results = []
with torch.no_grad():
    reference = {m: golden(x[:m].cuda(), ids[:m].cuda(), route[:m].cuda()).cpu() for m in sizes}
    for pools in map(int, a.pools.split(",")):
        start = time.time()
        op = ExactOperator(gu[:, :512], gu[:, 512:], down, max_len=224,
                           pools=pools, threads=pools*24, mapping=mapping)
        elements = 0
        for m in sizes:
            op.prepare(x[:m], physical_ids[:m], route[:m])
            value = op.forward().clone()
            target = reference[m]
            different = int((value.view(torch.int16) != target.view(torch.int16)).sum())
            assert different == 0, (pools, m, different, "native HF mismatch")
            elements += value.numel()
        if hasattr(op.moe, "set_logical_batch_size"):
            grouped_target = torch.cat([golden(x[i:i+1].cuda(), ids[i:i+1].cuda(), route[i:i+1].cuda()).cpu() for i in range(4)])
            op.moe.set_logical_batch_size(1)
            op.prepare(x[:4], physical_ids[:4], route[:4])
            assert torch.equal(op.forward().view(torch.int16), grouped_target.view(torch.int16))
            op.moe.set_logical_batch_size(0)
        # Empty batches, reload of existing storage, and caller-side errors.
        op.prepare(x[:0], physical_ids[:0], route[:0])
        assert op.forward().numel() == 0
        op.cpu.submit(op.moe.load_weights_task())
        op.cpu.sync()
        op.prepare(x[:1], physical_ids[:1], route[:1])
        assert torch.equal(op.forward().view(torch.int16), reference[1].view(torch.int16))
        ptrs = (op.ids.data_ptr(), op.route.data_ptr(), op.x.data_ptr(), op.out.data_ptr())
        for qlen, topk, incremental in ((-1, 8, False), (225, 8, False), (1, 7, False), (1, 8, True)):
            q = torch.tensor([qlen], dtype=torch.int32)
            try:
                op.moe.forward(q.data_ptr(), topk, *ptrs, incremental)
            except ValueError:
                pass
            else:
                raise AssertionError((qlen, topk, incremental, "unsupported request accepted"))
        row = {"pools": pools, "rows": sizes, "bit_different": 0, "elements": elements,
               "seconds": time.time()-start, "mapping_permuted": True, "reload": "passed", "invalid_arguments": "passed"}
        results.append(row)
        print(json.dumps(row), flush=True)
        del op
        gc.collect()
dump(a.output / "results.json", {"seed": 80923, "results": results,
                               "extension_sha256": sha(kt_kernel.kt_kernel_ext.__file__)})
