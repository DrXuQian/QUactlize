#pragma once

#include <stdexcept>

namespace pdl_experiment {
inline constexpr int default_carveout = -1;
inline constexpr int occupancy_preserving_l1 = -2;

struct ResourceChoice {
  int requested, selected, default_blocks, selected_blocks;
};

// Queries are injected so the actual selection algorithm can be tested on CPU.
// This is a preference, not a promise about achieved occupancy or cache size.
// Call once per function before execution/capture; no process-global cache.
template <class Set, class Query>
ResourceChoice choose_resources(int request, int initial, Set set, Query query) {
  if (request != default_carveout && request != occupancy_preserving_l1 && request != 25)
    throw std::invalid_argument("expected default, 25 or occupancy-preserving L1");
  if (initial < -1 || initial > 100) throw std::invalid_argument("invalid original carveout");
  if (request == default_carveout) {
    int blocks = query();
    if (blocks <= 0) throw std::runtime_error("kernel has no resident block");
    return {request, initial, blocks, blocks};
  }
  try {
    set(default_carveout);
    const int baseline = query();
    if (baseline <= 0) throw std::runtime_error("default kernel occupancy is zero");
    if (request == 25) {
      set(25);
      const int blocks = query();
      if (blocks <= 0) throw std::runtime_error("fixed carveout has no resident block");
      return {request, 25, baseline, blocks};
    }
    for (int pct = 0; pct <= 100; ++pct) {
      set(pct);
      const int blocks = query();
      if (blocks >= baseline) return {request, pct, baseline, blocks};
    }
    throw std::runtime_error("no carveout preserves default occupancy");
  } catch (...) {
    set(initial);
    throw;
  }
}
} // namespace pdl_experiment
