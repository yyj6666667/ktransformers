// CPU-only arithmetic models used to identify the frozen CUDA reference's
// reduction. No compiler reassociation, approximate exp or GPU dispatch here.
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

extern "C" void gemv_dot_bank(const float* x, const float* w, float* out,
                              int rows, int k, int lanes, int contiguous) {
  #pragma omp parallel for num_threads(8)
  for (int row = 0; row < rows; ++row) {
    const unsigned saved = _mm_getcsr();
    _mm_setcsr(saved & ~0x8040u);  // Match FFMA without FTZ, also for tiny gate weights.
    float partial[128] = {};
    for (int lane = 0; lane < lanes; ++lane) {
      const int chunk = (k + lanes - 1) / lanes;
      const int begin = contiguous ? lane * chunk : lane;
      const int end = contiguous ? std::min(k, begin + chunk) : k;
      const int step = contiguous ? 1 : lanes;
      for (int j = begin; j < end; j += step)
        partial[lane] = std::fma(x[row*k+j], w[row*k+j], partial[lane]);
    }
    for (int stride = lanes / 2; stride; stride /= 2)
      for (int lane = 0; lane < stride; ++lane)
        partial[lane] = partial[lane] + partial[lane + stride];
    out[row] = partial[0] + 0.f;
    _mm_setcsr(saved);
  }
}

extern "C" void reference_gemv(const float* x, const float* transposed_w, float* out,
                               int m, int n, int k, int lanes, int contiguous) {
  // The frozen reference uses 16 strided chains for gate/up (K=2048),
  // and 32 contiguous chains for down (K=512), followed by a halving tree.
  #pragma omp parallel for collapse(2) num_threads(8) if(m*n > 16384)
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; col += 16) {
      const unsigned saved = _mm_getcsr();
      _mm_setcsr(saved & ~0x8040u);
      __m512 partial[32];
      for (int lane = 0; lane < lanes; ++lane) {
        __m512 acc = _mm512_setzero_ps();
        const int chunk = (k + lanes - 1) / lanes;
        const int begin = contiguous ? lane*chunk : lane;
        const int end = contiguous ? std::min(k, begin+chunk) : k;
        const int step = contiguous ? 1 : lanes;
        for (int j = begin; j < end; j += step)
          acc = _mm512_fmadd_ps(_mm512_set1_ps(x[row*k+j]),
                                _mm512_loadu_ps(transposed_w + j*n + col), acc);
        partial[lane] = acc;
      }
      for (int stride = lanes/2; stride; stride /= 2)
        for (int lane = 0; lane < stride; ++lane)
          partial[lane] = _mm512_add_ps(partial[lane], partial[lane+stride]);
      _mm512_storeu_ps(out + row*n + col, _mm512_add_ps(partial[0], _mm512_setzero_ps()));
      _mm_setcsr(saved);
    }
  }
}

static inline __m256i aligned_magnitude(__m256i mantissa, __m256i shift) {
  const __m256i zero = _mm256_setzero_si256();
  mantissa = _mm256_sllv_epi32(mantissa, _mm256_max_epi32(shift, zero));
  return _mm256_srlv_epi32(mantissa, _mm256_max_epi32(_mm256_sub_epi32(zero, shift), zero));
}

static inline __m256 mma16_cpu(const uint16_t* x, const uint16_t* wt, int n, __m256 carry) {
  // Measured SM120 BF16 MMA model: align the 16 exact products and the FP32
  // carry to a shared unnormalized exponent with 26 significant bits, truncate
  // aligned magnitudes, sum integers, then round toward zero to FP32.
  const __m256i zero = _mm256_setzero_si256();
  const __m256i cword = _mm256_castps_si256(carry);
  const __m256i cexp = _mm256_and_si256(_mm256_srli_epi32(cword, 23), _mm256_set1_epi32(255));
  const __m256i cnonzero = _mm256_cmpgt_epi32(_mm256_and_si256(cword, _mm256_set1_epi32(0x7fffffff)), zero);
  __m256i emax = _mm256_blendv_epi8(_mm256_set1_epi32(-1024),
                                   _mm256_add_epi32(cexp, _mm256_set1_epi32(127)), cnonzero);
  __m256i product[16], exponent[16], sign[16];
  for (int j = 0; j < 16; ++j) {
    const __m256i b = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(wt+j*n)));
    const int ae = (x[j] >> 7) & 255;
    const __m256i be = _mm256_and_si256(_mm256_srli_epi32(b, 7), _mm256_set1_epi32(255));
    const int am = (x[j] & 127) | (ae ? 128 : 0);
    __m256i bm = _mm256_and_si256(b, _mm256_set1_epi32(127));
    bm = _mm256_or_si256(bm, _mm256_and_si256(_mm256_cmpgt_epi32(be, zero), _mm256_set1_epi32(128)));
    product[j] = _mm256_mullo_epi32(bm, _mm256_set1_epi32(am));
    exponent[j] = _mm256_add_epi32(_mm256_max_epi32(be, _mm256_set1_epi32(1)),
                                   _mm256_set1_epi32(std::max(ae, 1)));
    const __m256i present = _mm256_cmpgt_epi32(product[j], zero);
    emax = _mm256_max_epi32(emax, _mm256_blendv_epi8(_mm256_set1_epi32(-1024), exponent[j], present));
    sign[j] = _mm256_srai_epi32(_mm256_slli_epi32(_mm256_xor_si256(b, _mm256_set1_epi32(x[j])), 16), 31);
  }
  __m256i cmant = _mm256_and_si256(cword, _mm256_set1_epi32(0x7fffff));
  cmant = _mm256_or_si256(cmant, _mm256_and_si256(_mm256_cmpgt_epi32(cexp, zero), _mm256_set1_epi32(0x800000)));
  cmant = aligned_magnitude(cmant, _mm256_sub_epi32(_mm256_add_epi32(cexp, _mm256_set1_epi32(129)), emax));
  const __m256i csign = _mm256_srai_epi32(cword, 31);
  cmant = _mm256_sub_epi32(_mm256_xor_si256(cmant, csign), csign);
  __m512i total = _mm512_cvtepi32_epi64(cmant);
  for (int j = 0; j < 16; ++j) {
    __m256i term = aligned_magnitude(product[j], _mm256_sub_epi32(_mm256_add_epi32(exponent[j], _mm256_set1_epi32(11)), emax));
    term = _mm256_sub_epi32(_mm256_xor_si256(term, sign[j]), sign[j]);
    total = _mm512_add_epi64(total, _mm512_cvtepi32_epi64(term));
  }
  const __m512d power = _mm512_cvtepi32_pd(_mm256_sub_epi32(emax, _mm256_set1_epi32(279)));
  const __m512d value = _mm512_scalef_pd(_mm512_cvtepi64_pd(total), power);
  return _mm512_cvt_roundpd_ps(value, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

extern "C" void reference_mma(const uint16_t* x, const uint16_t* wt, float* out,
                              float* steps, int m, int n, int k) {
  #pragma omp parallel for collapse(2) num_threads(8) if(m*n > 16384)
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; col += 8) {
      const unsigned saved = _mm_getcsr();
      _mm_setcsr(saved & ~0x8040u);
      __m256 acc = _mm256_setzero_ps();
      for (int start = 0; start < k; start += 16) {
        acc = mma16_cpu(x + row*k + start, wt + start*n + col, n, acc);
        if (steps) _mm256_storeu_ps(steps + (start/16)*m*n + row*n + col, acc);
      }
      _mm256_storeu_ps(out + row*n + col, acc);
      _mm_setcsr(saved);
    }
  }
}
