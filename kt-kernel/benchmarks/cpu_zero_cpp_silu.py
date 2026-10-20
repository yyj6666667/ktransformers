"""Verify the CPU C++ activation over the complete BF16 input domain."""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import torch
p=argparse.ArgumentParser()
p.add_argument("--header", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
src=a.output/"silu.cpp"
src.write_text('#include "'+str(a.header)+'"\nextern "C" void table(uint16_t* y) { kt::sm120::silu_table(y); }\n')
lib=a.output/"silu.so"
subprocess.run(["g++","-std=c++20","-shared","-fPIC","-O3","-ffast-math","-mavx512f","-mavx512dq","-mavx512vl","-mfma",str(src),"-o",str(lib)],check=True)
c=ctypes.CDLL(str(lib)); c.table.argtypes=[ctypes.c_void_p]; c.table.restype=None
x=torch.arange(65536,dtype=torch.int32).short().view(torch.bfloat16)
y=torch.empty_like(x); c.table(y.data_ptr())
torch.set_flush_denormal(False)
r=torch.nn.functional.silu(x)
finite=torch.isfinite(r)&torch.isfinite(y)
neq=(r.view(torch.int16)!=y.view(torch.int16))&~(torch.isnan(r)&torch.isnan(y))
print({"bit_different_excluding_nan_payload":int(neq.sum()),"finite_different":int((neq&finite).sum()),"nan_class_different":int((torch.isnan(r)!=torch.isnan(y)).sum()),"indices":neq.nonzero().flatten().tolist()[:100]},flush=True)
print([(int(i),float(x[i]),int(y.view(torch.int16)[i]),int(r.view(torch.int16)[i])) for i in neq.nonzero().flatten()[::25]],flush=True)
g=torch.nn.functional.silu(x.cuda()).cpu()
result = {"gpu_bit_different_except_nan_payload": int(((g.view(torch.int16) != y.view(torch.int16)) & ~(torch.isnan(g) & torch.isnan(y))).sum()),
          "gpu_nan_class_different": int((torch.isnan(g) != torch.isnan(y)).sum()), "inputs": 65536}
print(result, flush=True)
(a.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
assert result["gpu_bit_different_except_nan_payload"] == result["gpu_nan_class_different"] == 0

torch.save({"input":x,"cpp":y,"cpu_torch":r,"different":neq},a.output/"table.pt")
