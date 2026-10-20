"""Check interval certificates and exact SIMD outputs against retained CUDA."""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import torch

p = argparse.ArgumentParser()
p.add_argument("--header", type=Path, required=True)
p.add_argument("--reference", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
src = a.output / "probe.cpp"
src.write_text('#include "' + str(a.header) + '"' + r'''
extern "C" int check(const uint16_t* x, const uint16_t* w, uint16_t* y, int k, int rows) {
  kt::sm120::GradualUnderflow scope;
  float exact[16], interval[16];
  kt::sm120::linear16_exact(x,w,exact,k,rows);
  bool accepted = rows > 1 && kt::sm120::linear16_interval(x,w,interval,k,rows);
  for (int j=0;j<16;++j) {
    y[j]=kt::sm120::bf16(exact[j]);
    if (accepted && y[j]!=kt::sm120::bf16(interval[j])) return -1;
  }
  return accepted;
}
''')
lib = a.output / "probe.so"
subprocess.run(["g++", "-std=c++20", "-shared", "-fPIC", "-O3", "-ffast-math",
                "-mavx512f", "-mavx512dq", "-mavx512vl", "-mfma", str(src), "-o", str(lib)], check=True)
c = ctypes.CDLL(str(lib))
c.check.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
c.check.restype = ctypes.c_int
torch.set_num_threads(1)
torch.set_flush_denormal(False)
torch.manual_seed(77301)
results = []
for path in sorted(a.reference.glob("*-m*.pt")):
    f = torch.load(path, weights_only=False, map_location="cpu")
    x, weight, golden = f["input"], f["weight"], f["golden"]
    rows, k = x.shape
    accepted = checked = 0
    for row in sorted({0, rows // 2, rows - 1}):
        for col in range(0, len(weight), 64):
            w = weight[col:col+16].T.contiguous()
            out = torch.empty(16, dtype=torch.bfloat16)
            status = c.check(x[row].data_ptr(), w.data_ptr(), out.data_ptr(), k, rows)
            assert status >= 0, (path.name, row, col, "unsound interval")
            assert torch.equal(out.view(torch.int16), golden[row, col:col+16].view(torch.int16)), (path.name, row, col)
            accepted += status
            checked += 1
    result = {"file": path.name, "checked_tiles": checked, "certified_tiles": accepted, "different": 0}
    results.append(result)
    print(json.dumps(result), flush=True)
for case in range(600):
    k = 512 if case % 4 == 0 else 2048
    rows = [2, 17, 33, 65][case % 4]
    x = torch.randn(k, dtype=torch.bfloat16)
    w = torch.randn(k, 16, dtype=torch.bfloat16)
    if case % 3 == 0:
        x = torch.ldexp(x.float(), torch.randint(-60, 61, x.shape)).bfloat16()
        w = torch.ldexp(w.float(), torch.randint(-60, 61, w.shape)).bfloat16()
    if case % 3 == 1:
        x.fill_(1.9921875)
        w.fill_(1.9921875)
        if case % 2:
            w[:, ::2].neg_()
    out = torch.empty(16, dtype=torch.bfloat16)
    assert c.check(x.data_ptr(), w.data_ptr(), out.data_ptr(), k, rows) >= 0, case
result = {"cuda_controls": results, "random_certificates_checked": 600}
(a.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
