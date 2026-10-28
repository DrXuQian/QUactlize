#pragma once

// One pinned MoE routing fixture. Tokens are routed through top-k DISTINCT experts; per-expert row counts are the
// resulting histogram, never an asserted mean or Mmax.
//
// Distribution: 16 "hot" experts have lottery weight 4 and every other expert weight 1. Each token samples top-k
// experts without replacement by sequential weighted draws. The integer lottery and fixed SplitMix64 seed make the
// fixture bit-reproducible across host compilers. The modest hot set models the popularity skew that uniform routing
// omits: at L=256/top-k=8 its pinned Mmax ladder is 1,2,3,12,239,447 for T=1,2,4,64,2048,4096.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace moe_router_fixture {

inline constexpr char kName[] = "token-topk-hot16x4-wor-sm64-s44-v1";
inline constexpr uint64_t kSeed = UINT64_C(0x51554143544c0044);

struct Rows {
  std::vector<int> per_expert;
  int tokens = 0;
  int topk = 0;
  int total = 0;
  int active = 0;
  int zero = 0;
  int min = 0;
  int max = 0;
};

inline uint64_t splitmix64(uint64_t& state) {
  uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}

inline int expert_weight(int expert, int hot_experts) { return expert < hot_experts ? 4 : 1; }

inline bool route(int tokens, int topk, int experts, Rows& out, char* why = nullptr, size_t why_n = 0) {
  auto fail = [&](char const* message) {
    if (why && why_n) std::snprintf(why, why_n, "%s", message);
    return false;
  };
  if (tokens <= 0) return fail("tokens must be positive");
  if (experts <= 0) return fail("experts must be positive");
  if (topk <= 0 || topk > experts) return fail("top-k must be in [1, experts]");

  out = Rows{};
  out.tokens = tokens;
  out.topk = topk;
  out.per_expert.assign(experts, 0);
  std::vector<uint8_t> selected(size_t(experts), 0);
  int const hot_experts = std::min(16, experts);
  uint64_t rng = kSeed;

  for (int token = 0; token < tokens; ++token) {
    std::fill(selected.begin(), selected.end(), uint8_t(0));
    int remaining_weight = hot_experts * 4 + (experts - hot_experts);
    for (int pick = 0; pick < topk; ++pick) {
      uint64_t lottery = splitmix64(rng) % uint64_t(remaining_weight);
      int chosen = -1;
      for (int expert = 0; expert < experts; ++expert) {
        if (selected[size_t(expert)]) continue;
        int const weight = expert_weight(expert, hot_experts);
        if (lottery < uint64_t(weight)) { chosen = expert; break; }
        lottery -= uint64_t(weight);
      }
      if (chosen < 0) return fail("weighted draw did not select an expert");
      selected[size_t(chosen)] = 1;
      ++out.per_expert[size_t(chosen)];
      remaining_weight -= expert_weight(chosen, hot_experts);
    }
  }

  out.total = tokens * topk;
  out.min = out.total;
  for (int rows : out.per_expert) {
    if (rows == 0) ++out.zero;
    else ++out.active;
    out.min = std::min(out.min, rows);
    out.max = std::max(out.max, rows);
  }
  return out.max > 0;
}

}  // namespace moe_router_fixture
