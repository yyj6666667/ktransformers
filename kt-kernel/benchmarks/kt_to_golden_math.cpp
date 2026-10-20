// Independent probes of the AVX512-BF16 reduction and KT SwiGLU arithmetic.
// The exponential polynomial follows kt-kernel/operators/amx/la/amx.hpp.
// No PyTorch/KT runtime is modified by this shared library.
#include <immintrin.h>
#include <cstdint>
#include <cstring>

extern "C" void ordered_dpbf16(const uint16_t* x, const uint16_t* packed_w,
                               float* out, int n, int k) {
  // packed_w layout: [k / 2, n, 2]. Each output accumulates consecutive
  // BF16 pairs in FP32, matching the AVX512 fallback's k traversal.
  for (int col = 0; col < n; col += 16) {
    __m512 acc = _mm512_setzero_ps();
    for (int pair = 0; pair < k / 2; ++pair) {
      uint32_t bits;
      std::memcpy(&bits, x + 2 * pair, sizeof(bits));
      const __m512bh a = (__m512bh)_mm512_set1_epi32(bits);
      const __m512bh b = (__m512bh)_mm512_loadu_si512(packed_w + (pair * n + col) * 2);
      acc = _mm512_dpbf16_ps(acc, a, b);
    }
    _mm512_storeu_ps(out + col, acc);
  }
}

extern "C" void ordered_dpbf16_batch(const uint16_t* x, const uint16_t* packed_w,
                                     float* out, int m, int n, int k) {
  #pragma omp parallel for collapse(2) num_threads(24) if(m > 8)
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; col += 16) {
      __m512 acc = _mm512_setzero_ps();
      for (int pair = 0; pair < k / 2; ++pair) {
        uint32_t bits;
        std::memcpy(&bits, x + row * k + 2 * pair, sizeof(bits));
        const __m512bh a = (__m512bh)_mm512_set1_epi32(bits);
        const __m512bh b = (__m512bh)_mm512_loadu_si512(packed_w + (pair * n + col) * 2);
        acc = _mm512_dpbf16_ps(acc, a, b);
      }
      _mm512_storeu_ps(out + row * n + col, acc);
    }
  }
}

static inline __m512 kt_exp(__m512 x) {
  const __m512 y = _mm512_mul_ps(x, _mm512_set1_ps(1.44269504089f));
  const __m512i whole = _mm512_cvtps_epi32(y);
  const __m512 part = _mm512_sub_ps(y, _mm512_cvtepi32_ps(whole));
  __m512 poly = _mm512_fmadd_ps(_mm512_set1_ps(0.0013333558f), part,
                               _mm512_set1_ps(0.0096181291f));
  poly = _mm512_fmadd_ps(poly, part, _mm512_set1_ps(0.0555041087f));
  poly = _mm512_fmadd_ps(poly, part, _mm512_set1_ps(0.2402265069f));
  poly = _mm512_fmadd_ps(poly, part, _mm512_set1_ps(0.6931471805f));
  poly = _mm512_fmadd_ps(poly, part, _mm512_set1_ps(0.9999999995f));
  return _mm512_mul_ps(_mm512_scalef_ps(_mm512_set1_ps(1.0f), _mm512_cvtepi32_ps(whole)), poly);
}

extern "C" void kt_swiglu(const float* gate, const float* up, float* out, int n) {
  for (int i = 0; i < n; i += 16) {
    const __m512 g = _mm512_loadu_ps(gate + i);
    const __m512 u = _mm512_loadu_ps(up + i);
    const __m512 neg = _mm512_min_ps(_mm512_sub_ps(_mm512_setzero_ps(), g), _mm512_set1_ps(88.0f));
    const __m512 denom = _mm512_add_ps(_mm512_set1_ps(1.0f), kt_exp(neg));
    _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_div_ps(g, denom), u));
  }
}

// Preserve the KT FMA order when down-projection values themselves are FP32.
extern "C" void kt_merge_parts(const float* values, const float* route,
                                float* out, int parts, int tokens, int topk, int hidden) {
  for (int token = 0; token < tokens; ++token) {
    for (int col = 0; col < hidden; col += 16) {
      __m512 total = _mm512_setzero_ps();
      for (int part = 0; part < parts; ++part) {
        __m512 acc = _mm512_setzero_ps();
        for (int j = 0; j < topk; ++j) {
          const float* src = values + ((part * tokens + token) * topk + j) * hidden + col;
          acc = _mm512_fmadd_ps(_mm512_loadu_ps(src), _mm512_set1_ps(route[token * topk + j]), acc);
        }
        total = _mm512_add_ps(total, acc);
      }
      _mm512_storeu_ps(out + token * hidden + col, total);
    }
  }
}
