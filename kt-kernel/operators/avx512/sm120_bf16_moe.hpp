#pragma once
// Explicit forward-only compatibility backend for the frozen Qwen3.5-35B
// SM120 eager BF16 reference. Existing AMXBF16/SFT backends are unchanged.
#include "../common.hpp"
#include "sm120_bf16_math.hpp"

#include <array>
#include <cstdlib>
#include <numeric>
#include <vector>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math", "fp-contract=off")
#endif
namespace kt::sm120 {

class Moe : public MoE_Interface {
 public:
  GeneralMOEConfig config;

  explicit Moe(GeneralMOEConfig cfg) : config(cfg) {
    if (!cfg.pool || cfg.hidden_size != 2048 || cfg.intermediate_size != 512 ||
        cfg.expert_num <= 0 || cfg.num_experts_per_tok <= 0 ||
        cfg.num_experts_per_tok > cfg.expert_num || cfg.max_possible_qlen() <= 0 ||
        cfg.swiglu_limit != 0 || cfg.swiglu_alpha != 0 || cfg.gpu_experts_mask ||
        !cfg.gate_projs.empty() || !cfg.up_projs.empty() || !cfg.down_projs.empty())
      throw std::invalid_argument("SM120BF16_MOE requires dense BF16 Qwen3.5-35B routed weights, H=2048 I=512, no GPU expert mask/clamp");
    parts_ = cfg.pool->config.subpool_count;
    if (parts_ <= 0 || 32 % parts_)
      throw std::invalid_argument("SM120BF16_MOE NUMA count must divide 32");
    max_len_ = cfg.max_possible_qlen();
    const size_t routes = size_t(max_len_) * cfg.num_experts_per_tok;
    activation_.resize(routes * 512);
    expert_output_.resize(routes * 2048);
    active_experts_.resize(cfg.expert_num);
    slots_.resize(routes);
    ordered_slots_.resize(routes);
    counts_.resize(cfg.expert_num);
    offsets_.resize(cfg.expert_num + 1);
    cursor_.resize(cfg.expert_num);
    logical_.resize(cfg.expert_num);
    shards_.resize(parts_);
    for (int part = 0; part < parts_; ++part)
      shards_[part].csr.resize(cfg.pool->config.subpool_thread_count.at(part));
    std::array<uint16_t, 65536> table;
    silu_table(table.data());
    for (int i = 0; i < 65536; ++i) silu_[i] = fp32(table[i]);
    const char* interval = std::getenv("KT_SM120_INTERVAL");
    guarded_ = !interval || std::string(interval) != "0";
  }

  Moe(const Moe&) = delete;
  Moe& operator=(const Moe&) = delete;

  void load_weights() {
    loaded_ = false;
    if (!config.gate_proj || !config.up_proj || !config.down_proj)
      throw std::invalid_argument("SM120BF16_MOE requires three contiguous BF16 weight pointers");
    const auto* mapping = static_cast<const int64_t*>(config.physical_to_logical_map);
    std::vector<bool> present(config.expert_num);
    for (int e = 0; e < config.expert_num; ++e) {
      int64_t logical = mapping ? mapping[e] : e;
      if (logical < 0 || logical >= config.expert_num || present[logical])
        throw std::invalid_argument("SM120BF16_MOE expert mapping must be a permutation");
      present[logical] = true;
      logical_[e] = int(logical);
    }
    // Each NUMA node first-touches its own BF16 output-column shards.
    config.pool->dispense_backend()->do_numa_job([&](int part) {
      auto& shard = shards_[part];
      int gu_n = 512 / parts_, down_n = 2048 / parts_;
      shard.gate.resize(size_t(config.expert_num) * gu_n * 2048);
      shard.up.resize(shard.gate.size());
      shard.down.resize(size_t(config.expert_num) * down_n * 512);
      auto pack = [&](const uint16_t* source, uint16_t* target, int n, int k, int local_n) {
        int tiles = local_n / 16;
        config.pool->get_subpool(part)->do_work_stealing_job(config.expert_num * tiles, [&](int job) {
          int e = job / tiles, col = (part * tiles + job % tiles) * 16;
          uint16_t* out = target + size_t(job) * k * 16;
          const uint16_t* in = source + size_t(logical_[e]) * n * k + size_t(col) * k;
          for (int j = 0; j < k; ++j)
            for (int lane = 0; lane < 16; ++lane) out[j*16+lane] = in[size_t(lane)*k+j];
        });
      };
      pack(static_cast<const uint16_t*>(config.gate_proj), shard.gate.data(), 512, 2048, gu_n);
      pack(static_cast<const uint16_t*>(config.up_proj), shard.up.data(), 512, 2048, gu_n);
      pack(static_cast<const uint16_t*>(config.down_proj), shard.down.data(), 2048, 512, down_n);
    });
    loaded_ = true;
  }

  void warm_up() {}  // Callers warm the exact input distribution before timing.

  void forward_binding(intptr_t qlen, int k, intptr_t ids, intptr_t route,
                       intptr_t input, intptr_t output, bool incremental = false) {
    forward(*reinterpret_cast<const int*>(qlen), k, reinterpret_cast<const int64_t*>(ids),
            reinterpret_cast<const float*>(route), reinterpret_cast<const void*>(input),
            reinterpret_cast<void*>(output), incremental);
  }

  // Grouped actor ranks are separate logical GEMMs. Concatenating them may
  // change cuBLAS's M-dependent arithmetic; explicitly retain their boundaries.
  void set_logical_batch_size(int rows) {
    if (rows < 0 || rows > max_len_) throw std::invalid_argument("invalid logical batch size");
    logical_rows_ = rows;
  }

  void forward(int qlen, int k, const int64_t* ids, const float* route, const void* input,
               void* output, bool incremental = false) override {
    if (!loaded_ || qlen < 0 || qlen > max_len_ || k != config.num_experts_per_tok || incremental)
      throw std::invalid_argument("SM120BF16_MOE invalid capacity/top-k/incremental request or unloaded weights");
    if (logical_rows_ && qlen > logical_rows_) {
      if (qlen > max_len_ || qlen % logical_rows_)
        throw std::invalid_argument("grouped SM120BF16_MOE requires complete logical batches within capacity");
      for (int begin = 0; begin < qlen; begin += logical_rows_)
        forward_one(logical_rows_, k, ids+size_t(begin)*k, route+size_t(begin)*k,
                    static_cast<const uint16_t*>(input)+size_t(begin)*2048,
                    static_cast<uint16_t*>(output)+size_t(begin)*2048, incremental);
    } else {
      forward_one(qlen, k, ids, route, input, output, incremental);
    }
  }

 private:
  void forward_one(int qlen, int k, const int64_t* ids, const float* route, const void* input,
                   void* output, bool incremental) {
    if (!loaded_ || qlen < 0 || qlen > max_len_ || k != config.num_experts_per_tok || incremental)
      throw std::invalid_argument("SM120BF16_MOE requires loaded weights, configured top-k/capacity, and a fresh output");
    if (!qlen) return;
    const int routes = qlen * k;
    std::fill(counts_.begin(), counts_.end(), 0);
    for (int slot = 0; slot < routes; ++slot) {
      if (ids[slot] < 0 || ids[slot] >= config.expert_num)
        throw std::invalid_argument("SM120BF16_MOE invalid physical expert ID");
      ++counts_[ids[slot]];
    }
    // Frozen cuBLAS dispatch has been exhaustively profiled for M=1..224.
    // Reject unmeasured shapes instead of silently claiming exactness.
    for (int count : counts_)
      if (count > 224) throw std::invalid_argument("SM120BF16_MOE supports at most 224 rows per expert in one logical batch");
    int active_count = 0;
    for (int e = 0; e < config.expert_num; ++e)
      if (counts_[e]) active_experts_[active_count++] = e;
    offsets_[0] = 0;
    for (int e = 0; e < config.expert_num; ++e) offsets_[e+1] = offsets_[e] + counts_[e];
    std::copy(offsets_.begin(), offsets_.end()-1, cursor_.begin());
    for (int slot = 0; slot < routes; ++slot) slots_[cursor_[ids[slot]]++] = slot;
    for (int token = 0; token < qlen; ++token) {
      auto first = ordered_slots_.begin() + token*k;
      std::iota(first, first+k, token*k);
      std::sort(first, first+k, [&](int a, int b) { return logical_[ids[a]] < logical_[ids[b]]; });
      for (int j = 1; j < k; ++j)
        if (ids[first[j-1]] == ids[first[j]]) throw std::invalid_argument("SM120BF16_MOE duplicate expert in a token");
    }
    const auto* x = static_cast<const uint16_t*>(input);
    auto* y = static_cast<uint16_t*>(output);
    config.pool->dispense_backend()->do_numa_job([&](int part) {
      const auto& shard = shards_[part];
      const int tiles = 32 / parts_;
      work(part, active_count * tiles, [&](int job) {
        int expert = active_experts_[job / tiles], tile = job % tiles;
        int col = (part*tiles+tile)*16;
        size_t base = (size_t(expert)*tiles+tile)*2048*16;
        // Reuse this worker's 16-column BF16 weight tile for every expert row.
        for (int position = offsets_[expert]; position < offsets_[expert+1]; ++position) {
          int slot = slots_[position];
          const uint16_t* row = x + size_t(slot/k)*2048;
          alignas(64) float gate[16], up[16];
          linear16(row, shard.gate.data()+base, gate, 2048, counts_[expert], guarded_);
          linear16(row, shard.up.data()+base, up, 2048, counts_[expert], guarded_);
          __m512 g = round_bf16(_mm512_load_ps(gate));
          __m512 u = round_bf16(_mm512_load_ps(up));
          __m512i index = _mm512_srli_epi32(_mm512_castps_si512(g), 16);
          __m512 activated = _mm512_i32gather_ps(index, silu_.data(), 4);
          store_bf16(activation_.data()+size_t(slot)*512+col, round_bf16(_mm512_mul_ps(activated, u)));
        }
      });
    });
    // The small BF16 activations are shared between NUMA nodes. Full-K down
    // dots avoid TP partial rounding/reassociation and need no output reduction.
    config.pool->dispense_backend()->do_numa_job([&](int part) {
      const auto& shard = shards_[part];
      const int tiles = 128 / parts_;
      if (qlen > 8) {
        // Prefill groups all rows of an expert around one resident weight tile.
        // Decode instead fuses the merge to minimize small-batch dispatch cost.
        work(part, active_count * tiles, [&](int job) {
          int expert = active_experts_[job / tiles], tile = job % tiles;
          int col = (part*tiles+tile)*16;
          size_t base = (size_t(expert)*tiles+tile)*512*16;
          for (int position = offsets_[expert]; position < offsets_[expert+1]; ++position) {
            int slot = slots_[position];
            alignas(64) float down[16];
            linear16(activation_.data()+size_t(slot)*512, shard.down.data()+base, down, 512, counts_[expert], guarded_);
            store_bf16(expert_output_.data()+size_t(slot)*2048+col, round_bf16(_mm512_load_ps(down)));
          }
        });
        work(part, qlen * tiles, [&](int job) {
          int token = job / tiles, col = (part*tiles+job%tiles)*16;
          __m512 sum = _mm512_setzero_ps();
          for (int j = 0; j < k; ++j) {
            int slot = ordered_slots_[token*k+j];
            __m512 value = expand16(expert_output_.data()+size_t(slot)*2048+col);
            __m512 weighted = round_bf16(_mm512_mul_ps(value, _mm512_set1_ps(route[slot])));
            sum = round_bf16(_mm512_add_ps(sum, weighted));
          }
          store_bf16(y+size_t(token)*2048+col, sum);
        });
        return;
      }
      // Each task owns an output tile and consumes experts in logical order.
      // Fuse BF16 down output, route weighting and merge; no expert-output
      // scratch, cross-NUMA output reduction or separate merge dispatch.
      work(part, qlen * tiles, [&](int job) {
        int token = job / tiles, tile = job % tiles;
        int col = (part*tiles+tile)*16;
        __m512 sum = _mm512_setzero_ps();
        for (int j = 0; j < k; ++j) {
          int slot = ordered_slots_[token*k+j], expert = int(ids[slot]);
          size_t base = (size_t(expert)*tiles+tile)*512*16;
          alignas(64) float down[16];
          linear16(activation_.data()+size_t(slot)*512, shard.down.data()+base, down, 512, counts_[expert], guarded_);
          __m512 value = round_bf16(_mm512_load_ps(down));
          __m512 weighted = round_bf16(_mm512_mul_ps(value, _mm512_set1_ps(route[slot])));
          sum = round_bf16(_mm512_add_ps(sum, weighted));
        }
        store_bf16(y+size_t(token)*2048+col, sum);
      });
    });
  }

 private:
  template<class Function>
  void work(int part, int count, Function&& fn) {
    auto& saved = shards_[part].csr;
    config.pool->get_subpool(part)->do_work_stealing_job(count,
        [&](int thread) { saved[thread] = _mm_getcsr(); _mm_setcsr(saved[thread] & ~0xe040u); },
        std::forward<Function>(fn), [&](int thread) { _mm_setcsr(saved[thread]); });
  }
  struct Shard { std::vector<uint16_t> gate, up, down; std::vector<unsigned> csr; };
  int parts_, max_len_, logical_rows_ = 0;
  bool loaded_ = false, guarded_ = false;
  std::vector<Shard> shards_;
  std::vector<uint16_t> activation_, expert_output_;
  std::vector<int> slots_, ordered_slots_, counts_, offsets_, cursor_, logical_, active_experts_;
  std::array<float, 65536> silu_;
};
} // namespace kt::sm120
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
