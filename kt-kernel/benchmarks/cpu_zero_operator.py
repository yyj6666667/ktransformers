"""Actual KT C++/CPUInfer operator wrapper for the frozen SM120 profile."""
import torch
import kt_kernel


def make_cpu(pools=2, threads=48):
    cfg = kt_kernel.kt_kernel_ext.WorkerPoolConfig()
    cfg.subpool_count = pools
    cfg.subpool_numa_map = list(range(pools))
    assert threads % pools == 0
    cfg.subpool_thread_count = [threads // pools] * pools
    return kt_kernel.kt_kernel_ext.CPUInfer(cfg)


class ExactOperator:
    def __init__(self, gate, up, down, *, topk=8, max_len=512, pools=2, threads=48,
                 cpu=None, mapping=None):
        ext = kt_kernel.kt_kernel_ext
        self.weights = tuple(w.detach().cpu().contiguous() for w in (gate, up, down))
        experts, intermediate, hidden = gate.shape
        self.topk = topk
        self.cpu = cpu if cpu is not None else make_cpu(pools, threads)
        self.mapping = (torch.arange(experts, dtype=torch.int64) if mapping is None
                        else mapping.to(dtype=torch.int64, device="cpu").contiguous())
        cfg = ext.moe.MOEConfig(experts, topk, hidden, intermediate, 0)
        cfg.max_len = max_len
        cfg.physical_to_logical_map = self.mapping.data_ptr()
        cfg.gate_proj, cfg.up_proj, cfg.down_proj = [w.data_ptr() for w in self.weights]
        cfg.pool = self.cpu.backend_
        self.moe = ext.moe.SM120BF16_MOE(cfg)
        self.cpu.submit(self.moe.load_weights_task())
        self.cpu.sync()

    def prepare(self, x, ids, weights):
        self.x = x.detach().cpu().bfloat16().contiguous()
        self.ids = ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
        self.route = weights.detach().cpu().float().contiguous()
        self.qlen = torch.tensor([len(x)], dtype=torch.int32)
        self.out = torch.empty_like(self.x)

    def forward(self):
        self.cpu.submit(self.moe.forward_task(self.qlen.data_ptr(), self.topk,
                        self.ids.data_ptr(), self.route.data_ptr(), self.x.data_ptr(), self.out.data_ptr()))
        self.cpu.sync()
        return self.out

    def evaluate(self, x, ids, route, variants):
        assert len(variants) == 1
        self.prepare(x, ids, route)
        return {next(iter(variants)): self.forward()}
