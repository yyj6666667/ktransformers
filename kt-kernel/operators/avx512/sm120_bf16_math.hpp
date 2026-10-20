#pragma once
// Opt-in arithmetic compatibility with the frozen SM120/cuBLAS 12.8 eager
// BF16 reference. This is not a general CUDA arithmetic contract.
#include <immintrin.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math", "fp-contract=off")
#endif
namespace kt::sm120 {
struct GradualUnderflow {
  unsigned saved = _mm_getcsr();
  GradualUnderflow() { _mm_setcsr(saved & ~0xe040u); }
  ~GradualUnderflow() { _mm_setcsr(saved); }
};
inline float fp32(uint16_t x) { return std::bit_cast<float>(uint32_t(x) << 16); }
inline uint16_t bf16(float x) {
  uint32_t u = std::bit_cast<uint32_t>(x);
  if ((u & 0x7fffffffU) > 0x7f800000U) return 0x7fc0;
  return uint16_t((u + 0x7fffU + ((u >> 16) & 1)) >> 16);
}
inline __m512 expand16(const uint16_t* x) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x))), 16));
}
inline __m512 round_bf16(__m512 value) {
  __m512i u = _mm512_castps_si512(value);
  __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1));
  __m512i rounded = _mm512_and_si512(_mm512_add_epi32(u, _mm512_add_epi32(lsb, _mm512_set1_epi32(0x7fff))),
                                    _mm512_set1_epi32(0xffff0000));
  __mmask16 nan = _mm512_cmpgt_epu32_mask(_mm512_and_si512(u, _mm512_set1_epi32(0x7fffffff)), _mm512_set1_epi32(0x7f800000));
  return _mm512_castsi512_ps(_mm512_mask_mov_epi32(rounded, nan, _mm512_set1_epi32(0x7fc00000)));
}
inline void store_bf16(uint16_t* output, __m512 rounded) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(output),
      _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(rounded), 16)));
}
inline void silu_table(uint16_t* table) {
  GradualUnderflow scope;
  for (int i = 0; i < 65536; ++i) {
    float v = fp32(uint16_t(i));
    table[i] = bf16(v / (1.f + std::exp(-v)));
  }
}
// One output tile is [K,16] BF16. Keeping a full K on one worker preserves
// the reference's reduction; NUMA shards split output columns, never K.
inline __m512 gemv16(const uint16_t* x, const uint16_t* w, int k) {
  __m512 partial[32];
  if (k == 2048) {
    for (int lane = 0; lane < 16; ++lane) partial[lane] = _mm512_setzero_ps();
    for (int base = 0; base < k; base += 16) {
      #pragma GCC unroll 16
      for (int lane = 0; lane < 16; ++lane)
        partial[lane] = _mm512_fmadd_ps(_mm512_set1_ps(fp32(x[base+lane])),
                                       expand16(w+(base+lane)*16), partial[lane]);
    }
  } else {
    for (int group = 0; group < 32; group += 8) {
      __m512 acc[8];
      for (int lane = 0; lane < 8; ++lane) acc[lane] = _mm512_setzero_ps();
      for (int j = 0; j < 16; ++j) {
        #pragma GCC unroll 8
        for (int lane = 0; lane < 8; ++lane) {
          int p = (group+lane)*16+j;
          acc[lane] = _mm512_fmadd_ps(_mm512_set1_ps(fp32(x[p])), expand16(w+p*16), acc[lane]);
        }
      }
      for (int lane = 0; lane < 8; ++lane) partial[group+lane] = acc[lane];
    }
  }
  for (int stride = (k == 2048 ? 8 : 16); stride; stride /= 2)
    for (int lane = 0; lane < stride; ++lane)
      partial[lane] = _mm512_add_ps(partial[lane], partial[lane+stride]);
  return _mm512_add_ps(partial[0], _mm512_setzero_ps());
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


// The aligned positive and negative totals each fit uint32: at most
// 16*(255*255*2^11) + (2^24-1)*4 = 2,197,848,060 < 2^32.
// Splitting their signs avoids signed overflow without 64-bit SIMD. RZ
// conversion followed by power-of-two RZ scaling is the same truncation as
// the exact integer-times-power-of-two model, including gradual underflow.
inline __m512 mma16x16_cpu(const uint16_t* x, const uint16_t* wt, __m512 carry) {
  const __m512i zero = _mm512_setzero_si512();
  const __m512i cword = _mm512_castps_si512(carry);
  const __m512i cexp = _mm512_and_si512(_mm512_srli_epi32(cword, 23), _mm512_set1_epi32(255));
  __mmask16 cnz = _mm512_cmpneq_epi32_mask(_mm512_and_si512(cword, _mm512_set1_epi32(0x7fffffff)), zero);
  __m512i emax = _mm512_mask_mov_epi32(_mm512_set1_epi32(-1024), cnz,
                                     _mm512_add_epi32(cexp, _mm512_set1_epi32(127)));
  for (int j = 0; j < 16; ++j) {
    if (!(x[j] & 0x7fff)) continue;
    __m512i b = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(wt+j*16)));
    __m512i be = _mm512_and_si512(_mm512_srli_epi32(b, 7), _mm512_set1_epi32(255));
    __m512i exponent = _mm512_add_epi32(_mm512_max_epi32(be, _mm512_set1_epi32(1)),
                                      _mm512_set1_epi32(std::max(int((x[j] >> 7) & 255), 1)));
    __mmask16 nz = _mm512_cmpneq_epi32_mask(_mm512_and_si512(b, _mm512_set1_epi32(0x7fff)), zero);
    emax = _mm512_mask_max_epi32(emax, nz, emax, exponent);
  }
  auto align = [zero](__m512i m, __m512i shift) {
    m = _mm512_sllv_epi32(m, _mm512_max_epi32(shift, zero));
    return _mm512_srlv_epi32(m, _mm512_max_epi32(_mm512_sub_epi32(zero, shift), zero));
  };
  __m512i cmant = _mm512_and_si512(cword, _mm512_set1_epi32(0x7fffff));
  cmant = _mm512_mask_or_epi32(cmant, _mm512_cmpneq_epi32_mask(cexp, zero), cmant, _mm512_set1_epi32(0x800000));
  cmant = align(cmant, _mm512_sub_epi32(_mm512_add_epi32(cexp, _mm512_set1_epi32(129)), emax));
  __mmask16 cneg = _mm512_cmp_epi32_mask(cword, zero, _MM_CMPINT_LT);
  __m512i positive = _mm512_maskz_mov_epi32(~cneg, cmant);
  __m512i negative = _mm512_maskz_mov_epi32(cneg, cmant);
  for (int j = 0; j < 16; ++j) {
    __m512i b = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(wt+j*16)));
    int ae = (x[j] >> 7) & 255;
    __m512i be = _mm512_and_si512(_mm512_srli_epi32(b, 7), _mm512_set1_epi32(255));
    __m512i bm = _mm512_and_si512(b, _mm512_set1_epi32(127));
    bm = _mm512_mask_or_epi32(bm, _mm512_cmpneq_epi32_mask(be, zero), bm, _mm512_set1_epi32(128));
    __m512i product = _mm512_mullo_epi32(bm, _mm512_set1_epi32((x[j]&127) | (ae ? 128 : 0)));
    __m512i exponent = _mm512_add_epi32(_mm512_max_epi32(be, _mm512_set1_epi32(1)), _mm512_set1_epi32(std::max(ae, 1)));
    __m512i term = align(product, _mm512_sub_epi32(_mm512_add_epi32(exponent, _mm512_set1_epi32(11)), emax));
    __mmask16 neg = _mm512_test_epi32_mask(_mm512_xor_si512(b, _mm512_set1_epi32(x[j])), _mm512_set1_epi32(0x8000));
    positive = _mm512_mask_add_epi32(positive, ~neg, positive, term);
    negative = _mm512_mask_add_epi32(negative, neg, negative, term);
  }
  __mmask16 neg = _mm512_cmplt_epu32_mask(positive, negative);
  __m512i magnitude = _mm512_mask_blend_epi32(neg, _mm512_sub_epi32(positive, negative), _mm512_sub_epi32(negative, positive));
  __m512 value = _mm512_cvt_roundepu32_ps(magnitude, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  value = _mm512_scalef_round_ps(value, _mm512_cvtepi32_ps(_mm512_sub_epi32(emax, _mm512_set1_epi32(279))),
                                _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  return _mm512_castsi512_ps(_mm512_mask_xor_epi32(_mm512_castps_si512(value), neg,
                                                _mm512_castps_si512(value), _mm512_set1_epi32(0x80000000U)));
}

inline void linear16_exact(const uint16_t* x, const uint16_t* w, float* out, int k, int rows) {
  if (rows == 1) {
    _mm512_storeu_ps(out, gemv16(x, w, k));
    return;
  }
  const int chunk = k == 2048 && rows >= 17 && rows <= 32 ? 448
                  : k == 2048 && rows >= 33 && rows <= 64 ? 1024 : k;
  __m512 total = _mm512_setzero_ps();
  for (int begin = 0; begin < k; begin += chunk) {
    __m512 acc = _mm512_setzero_ps();
    for (int j = begin; j < std::min(k, begin+chunk); j += 16)
      acc = mma16x16_cpu(x+j, w+j*16, acc);
    total = _mm512_add_ps(total, acc);
  }
  _mm512_storeu_ps(out, total);
}
// An interval certificate, not a fitted numerical tolerance. Directed FP32
// FMAs enclose the exact sum; each of the 16 aligned products plus the carry
// loses less than q=2^(emax-279) in the measured MMA. Expanding by 17q encloses
// that truncation. RZ is monotone and fixes the representable endpoints.
// Equal BF16-rounded endpoints therefore certify the final BF16 output.
inline bool linear16_interval(const uint16_t* x, const uint16_t* w, float* out, int k, int rows) {
  const int chunk = k == 2048 && rows >= 17 && rows <= 32 ? 448
                  : k == 2048 && rows >= 33 && rows <= 64 ? 1024 : k;
  __m512 lower_total = _mm512_setzero_ps(), upper_total = _mm512_setzero_ps();
  constexpr int rd = _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC;
  constexpr int ru = _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC;
  const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);
  for (int begin = 0; begin < k; begin += chunk) {
    __m512 lower = _mm512_setzero_ps(), upper = lower;
    for (int start = begin; start < std::min(k, begin+chunk); start += 16) {
      __m512 magnitude = _mm512_max_ps(_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(lower), abs_mask)),
                                      _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(upper), abs_mask)));
      __m512i emax = _mm512_add_epi32(_mm512_srli_epi32(_mm512_castps_si512(magnitude), 23), _mm512_set1_epi32(127));
      __m512 min_magnitude = _mm512_min_ps(_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(lower), abs_mask)),
                                          _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(upper), abs_mask)));
      __mmask16 cross_zero = _mm512_cmp_ps_mask(lower, _mm512_setzero_ps(), _CMP_LE_OQ) &
                            _mm512_cmp_ps_mask(upper, _mm512_setzero_ps(), _CMP_GE_OQ);
      min_magnitude = _mm512_mask_mov_ps(min_magnitude, cross_zero, _mm512_setzero_ps());
      __m512i min_carry_exp = _mm512_srli_epi32(_mm512_castps_si512(min_magnitude), 23);
      __m512i product_exp[16];
      __m512 block_lower[4], block_upper[4];
      for (int lane = 0; lane < 4; ++lane) block_lower[lane] = block_upper[lane] = _mm512_setzero_ps();
      for (int j = start; j < start+16; ++j) {
        __m512i b = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(w+j*16)));
        __m512i be = _mm512_max_epi32(_mm512_and_si512(_mm512_srli_epi32(b, 7), _mm512_set1_epi32(255)), _mm512_set1_epi32(1));
        product_exp[j-start] = _mm512_add_epi32(be, _mm512_set1_epi32(std::max(int((x[j]>>7)&255), 1)));
        emax = _mm512_max_epi32(emax, product_exp[j-start]);
        __m512 bv = _mm512_castsi512_ps(_mm512_slli_epi32(b, 16));
        __m512 av = _mm512_set1_ps(fp32(x[j]));
        block_lower[j&3] = _mm512_fmadd_round_ps(av, bv, block_lower[j&3], rd);
        block_upper[j&3] = _mm512_fmadd_round_ps(av, bv, block_upper[j&3], ru);
      }
      for (int stride = 2; stride; stride /= 2)
        for (int lane = 0; lane < stride; ++lane) {
          block_lower[lane] = _mm512_add_round_ps(block_lower[lane], block_lower[lane+stride], rd);
          block_upper[lane] = _mm512_add_round_ps(block_upper[lane], block_upper[lane+stride], ru);
        }
      lower = _mm512_add_round_ps(lower, block_lower[0], rd);
      upper = _mm512_add_round_ps(upper, block_upper[0], ru);
      // Products have only 16 mantissa bits. Alignment cannot discard bits
      // when product_exp+11 >= emax; likewise carry_exp+129 >= emax.
      // Charge q only for terms that can actually lose bits. Counting zero
      // terms is conservative and avoids an extra zero-detection pass.
      __m512i lossy = _mm512_maskz_set1_epi32(_mm512_cmpgt_epi32_mask(emax,
          _mm512_add_epi32(min_carry_exp, _mm512_set1_epi32(129))), 1);
      for (int j = 0; j < 16; ++j)
        lossy = _mm512_mask_add_epi32(lossy, _mm512_cmpgt_epi32_mask(emax,
            _mm512_add_epi32(product_exp[j], _mm512_set1_epi32(11))), lossy, _mm512_set1_epi32(1));
      __m512 radius = _mm512_scalef_round_ps(_mm512_cvtepi32_ps(lossy),
          _mm512_cvtepi32_ps(_mm512_sub_epi32(emax, _mm512_set1_epi32(279))), ru);
      lower = _mm512_sub_round_ps(lower, radius, rd);
      upper = _mm512_add_round_ps(upper, radius, ru);
    }
    lower_total = _mm512_add_ps(lower_total, lower);
    upper_total = _mm512_add_ps(upper_total, upper);
  }
  alignas(64) float lo[16], hi[16];
  _mm512_store_ps(lo, lower_total); _mm512_store_ps(hi, upper_total);
  for (int i = 0; i < 16; ++i)
    if (!std::isfinite(lo[i]) || !std::isfinite(hi[i]) || bf16(lo[i]) != bf16(hi[i])) return false;
  _mm512_storeu_ps(out, lower_total);
  return true;
}
inline void linear16(const uint16_t* x, const uint16_t* w, float* out, int k, int rows, bool guarded = false) {
  if (guarded && rows > 1 && linear16_interval(x, w, out, k, rows)) return;
  linear16_exact(x, w, out, k, rows);
}

} // namespace kt::sm120
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
