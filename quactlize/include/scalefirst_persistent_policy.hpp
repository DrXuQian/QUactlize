#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace quactlize::scalefirst_policy {

struct GridChoice {
  int grid = 0;
  std::uint64_t capacity_b_mask = 0;
  std::uint64_t balanced_b_mask = 0;
};

constexpr std::uint64_t ceil_div(std::uint64_t value,
                                 std::uint64_t divisor) {
  return (value + divisor - 1) / divisor;
}

constexpr std::uint64_t capacity_grid(std::uint64_t q, int cu, int b) {
  std::uint64_t const wave = std::uint64_t(cu) * std::uint64_t(b);
  return q < wave ? q : wave;
}

constexpr std::uint64_t balanced_grid(std::uint64_t q, int cu, int b) {
  std::uint64_t const wave = std::uint64_t(cu) * std::uint64_t(b);
  return ceil_div(q, ceil_div(q, wave));
}

// The motivating 72-CU/Q=2048 case must retain BOTH policy witnesses.  These
// compile-time facts prevent an implementation refactor from collapsing the
// balanced grid back to the capacity grid.
static_assert(capacity_grid(2048, 72, 8) == 576);
static_assert(balanced_grid(2048, 72, 8) == 512);

inline std::vector<GridChoice> grid_space(std::uint64_t q, int cu,
                                          int occupancy) {
  if (q == 0 || cu <= 0 || occupancy <= 0 || occupancy > 63) return {};
  std::vector<GridChoice> out;
  auto add = [&](std::uint64_t grid, int b, bool capacity) {
    if (grid == 0 || grid > std::uint64_t(std::numeric_limits<int>::max())) {
      out.clear();
      return false;
    }
    auto it = std::find_if(out.begin(), out.end(), [&](auto const& choice) {
      return choice.grid == int(grid);
    });
    if (it == out.end()) {
      out.push_back(GridChoice{int(grid), 0, 0});
      it = out.end() - 1;
    }
    std::uint64_t const bit = std::uint64_t(1) << b;
    if (capacity) it->capacity_b_mask |= bit;
    else it->balanced_b_mask |= bit;
    return true;
  };
  for (int b = 1; b <= occupancy; ++b) {
    if (!add(capacity_grid(q, cu, b), b, true) ||
        !add(balanced_grid(q, cu, b), b, false)) return {};
  }
  std::sort(out.begin(), out.end(), [](auto const& lhs, auto const& rhs) {
    return lhs.grid < rhs.grid;
  });
  return out;
}

// One typed owner for the machine record.  Keeping the literal next to the
// typed arguments lets `-Wformat=2 -Werror=format` catch varargs drift; a
// source-side duplicated JSON key is caught by the independent local parser.
inline int format_q8_policy_cell(
    char* buffer, std::size_t size,
    int m, int n, int k, char const* algorithm, char const* policy, int grid,
    char const* config, int rep, double sample_us, double mfu_pct,
    double mbu_pct, std::uint64_t q, int cu, int occupancy,
    std::uint64_t capacity_b_mask, std::uint64_t balanced_b_mask,
    std::size_t candidate_denominator) {
  return std::snprintf(
      buffer, size,
      "Q8_POLICY_CELL {\"rec\":\"cell_sample\",\"shape\":\"%dx%dx%d\","
      "\"m\":%d,\"n\":%d,\"k\":%d,\"algorithm\":\"%s\","
      "\"policy\":\"%s\",\"grid\":%d,\"config\":\"%s\","
      "\"status\":\"MEASURED\",\"rep\":%d,\"sample_us\":%.9f,"
      "\"MFU_pct\":%.9f,\"distinct_MBU_model_pct\":%.9f,"
      "\"Q\":%llu,\"CU\":%d,\"occupancy\":%d,"
      "\"capacity_b_mask\":\"0x%llx\",\"balanced_b_mask\":\"0x%llx\","
      "\"candidate_denominator\":%zu}\n",
      m, n, k, m, n, k, algorithm, policy, grid, config, rep, sample_us,
      mfu_pct, mbu_pct, static_cast<unsigned long long>(q), cu, occupancy,
      static_cast<unsigned long long>(capacity_b_mask),
      static_cast<unsigned long long>(balanced_b_mask), candidate_denominator);
}

}  // namespace quactlize::scalefirst_policy
