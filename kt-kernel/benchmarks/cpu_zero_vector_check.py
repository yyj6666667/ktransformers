"""Cross-check SIMD optimization against the verified CPU MMA model."""
import argparse
import ctypes
from pathlib import Path
import subprocess
import torch

p = argparse.ArgumentParser()
p.add_argument("--header", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
src = a.output / "probe.cpp"
src.write_text('#include "' + str(a.header) + '"' + r'''
extern "C" void check(const uint16_t* x, const uint16_t* w, float* y, int blocks) {
  kt::sm120::GradualUnderflow scope;
  __m256 a = _mm256_setzero_ps(), b = a;
  __m512 c = _mm512_setzero_ps();
  for (int i = 0; i < blocks; ++i) {
    a = kt::sm120::mma16_cpu(x+i*16, w+i*256, 16, a);
    b = kt::sm120::mma16_cpu(x+i*16, w+i*256+8, 16, b);
    c = kt::sm120::mma16x16_cpu(x+i*16, w+i*256, c);
    _mm256_storeu_ps(y+i*32, a);
    _mm256_storeu_ps(y+i*32+8, b);
    _mm512_storeu_ps(y+i*32+16, c);
  }
}
''')
lib = a.output / "probe.so"
subprocess.run(["g++", "-std=c++20", "-shared", "-fPIC", "-O3", "-ffast-math",
                "-mavx512f", "-mavx512dq", "-mavx512vl", "-mfma", str(src), "-o", str(lib)], check=True)
c = ctypes.CDLL(str(lib))
c.check.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int]
c.check.restype = None
torch.set_num_threads(1)
torch.set_flush_denormal(False)
torch.manual_seed(77201)
elements = 0
for case in range(300):
    x = torch.randn(2048, dtype=torch.bfloat16)
    w = torch.randn(2048, 16, dtype=torch.bfloat16)
    if case % 3 == 0:
        x = torch.ldexp(x.float(), torch.randint(-60, 61, x.shape)).bfloat16()
        w = torch.ldexp(w.float(), torch.randint(-60, 61, w.shape)).bfloat16()
    if case % 3 == 1:
        x.fill_(1.9921875)
        w.fill_(1.9921875)
        if case % 2:
            w[:, ::2].neg_()
    out = torch.empty(128, 32)
    c.check(x.data_ptr(), w.data_ptr(), out.data_ptr(), 128)
    assert torch.equal(out[:, :16].view(torch.int32), out[:, 16:].view(torch.int32)), case
    elements += 128 * 16
print({"fp32_steps_bit_different": 0, "elements": elements}, flush=True)
