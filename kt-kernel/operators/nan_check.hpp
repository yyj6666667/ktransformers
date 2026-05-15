/**
 * @Description  : NaN/Inf detection utilities for MoE debugging
 * @Date         : 2026-05-15
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_NAN_CHECK_HPP
#define CPUINFER_NAN_CHECK_HPP

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "llama.cpp/ggml.h"

namespace nan_check {

inline bool is_nan(float v) { return v != v; }
inline bool is_inf(float v) { return (v > 0 && v * 2 == v) || (v < 0 && v * 2 == v); }

inline void throw_if_nan_fp32(const float* buf, int size, const char* label,
                               int layer_idx = -1, int expert_idx = -1) {
  for (int i = 0; i < size; i++) {
    if (is_nan(buf[i]) || is_inf(buf[i])) {
      fprintf(stderr, "[NaN ERROR] %s: idx=%d, val=%f, layer=%d, expert=%d\n",
              label, i, buf[i], layer_idx, expert_idx);
      throw std::runtime_error("NaN detected in " + std::string(label));
    }
  }
}

inline void throw_if_nan_bf16(const ggml_bf16_t* buf, int size, const char* label,
                               int layer_idx = -1, int expert_idx = -1) {
  for (int i = 0; i < size; i++) {
    float v = GGML_BF16_TO_FP32(buf[i]);
    if (is_nan(v) || is_inf(v)) {
      fprintf(stderr, "[NaN ERROR] %s: idx=%d, val=%f, layer=%d, expert=%d\n",
              label, i, v, layer_idx, expert_idx);
      throw std::runtime_error("NaN detected in " + std::string(label));
    }
  }
}

inline void throw_if_nan_single(float v, const char* label, int m_idx, int n_idx) {
  if (is_nan(v) || is_inf(v)) {
    fprintf(stderr, "[NaN ERROR] %s: m=%d, n=%d, val=%f\n", label, m_idx, n_idx, v);
    throw std::runtime_error("NaN detected in " + std::string(label));
  }
}

}  // namespace nan_check
#endif  // CPUINFER_NAN_CHECK_HPP
