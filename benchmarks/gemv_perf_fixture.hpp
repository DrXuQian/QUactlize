#pragma once

// Host-only authority seam for the grouped GEMV performance fixture.  The
// kernel consumes a dense expert-id address space [0,E), even when only a few
// experts own rows.  Keep that address space separate from the compact list of
// active slots: expert id selects W/S/Z, while row_offsets selects gathered A/D.

#include <algorithm>
#include <cstdint>
#include <vector>

#include "moe_router_fixture.hpp"

namespace gemv_perf_fixture {

struct Route {
  bool valid = false;
  int experts = 0;
  int tokens = 0;
  int topk = 0;
  int total_rows = 0;
  int max_rows = 0;
  std::vector<int> rows_per_expert;
  std::vector<int> row_offsets;
  std::vector<int> active_ids;
  std::vector<int> active_slot_for_expert;
};

inline Route make_route(int experts, int tokens, int topk) {
  Route out;
  out.experts = experts;
  out.tokens = tokens;
  out.topk = topk;
  moe_router_fixture::Rows routed;
  if (!moe_router_fixture::route(tokens, topk, experts, routed)) return out;

  out.rows_per_expert = routed.per_expert;
  out.row_offsets.assign(std::size_t(experts + 1), 0);
  out.active_slot_for_expert.assign(std::size_t(experts), -1);
  for (int e = 0; e < experts; ++e) {
    out.row_offsets[std::size_t(e + 1)] =
        out.row_offsets[std::size_t(e)] + out.rows_per_expert[std::size_t(e)];
    if (out.rows_per_expert[std::size_t(e)] > 0) {
      out.active_slot_for_expert[std::size_t(e)] = int(out.active_ids.size());
      out.active_ids.push_back(e);
    }
  }
  out.total_rows = out.row_offsets.back();
  out.max_rows = *std::max_element(out.rows_per_expert.begin(), out.rows_per_expert.end());
  out.valid = out.total_rows == routed.total &&
              int(out.active_ids.size()) == routed.active &&
              out.max_rows == routed.max;
  return out;
}

// BYTE UNIT IS LOAD-BEARING.  The grouped GEMV kernel adds this value to a
// uint8_t pointer.  Returning logical sub-byte codes here recreates the exact
// expert-pitch failure that identical-expert fixtures concealed.
inline std::uint64_t packed_plane_bytes(int n, int k, int bits) {
  return std::uint64_t(n) * std::uint64_t(k) * std::uint64_t(bits) / 8u;
}

inline std::uint64_t packed_plane_expert_offset(
    int expert, int n, int k, int bits) {
  return std::uint64_t(expert) * packed_plane_bytes(n, k, bits);
}

inline std::uint32_t plane_code(int bits, int n, int k, std::uint32_t seed) {
  std::uint32_t const mask = (UINT32_C(1) << bits) - UINT32_C(1);
  bool const active = (seed >> 24) == UINT32_C(0x51);
  if (!active) return mask;  // inactive poison: maximum code in every plane
  int const expert = int(seed & UINT32_C(0xff));
  int const high_plane = int((seed >> 16) & UINT32_C(1));
  // The expert salt changes code density, not merely the phase of an otherwise
  // equal-mean pseudorandom stream.  K multiples of 256 visit every ticket
  // equally because 29 is coprime to 256, so a wrong expert pitch changes the
  // GEMV sum by construction rather than by probability.
  int const ticket = (17 * n + 29 * k + 101 * high_plane) & 255;
  return ticket <= expert ? mask : 0u;
}

// Active expert data are distinct by real expert id, not compact active slot.
// Inactive experts remain allocated and receive a disjoint poison seed so an
// accidental active-slot/real-id substitution is observable rather than OOB.
inline std::uint32_t plane_seed(int expert, bool active, bool high_plane) {
  return (active ? UINT32_C(0x51000000) : UINT32_C(0xf0000000)) |
         (high_plane ? UINT32_C(0x00010000) : 0u) |
         std::uint32_t(expert & 255);
}

inline float activation_value(int expert, int row_in_expert) {
  return float(1 + ((expert * 3 + row_in_expert * 5) & 7)) / 16.0f;
}

inline float scale_value(int expert, int group, int column, bool active) {
  if (!active) return 16.0f + float((expert + group + column) & 3);
  return float(1 + ((expert * 5 + group * 3 + column) & 7)) / 64.0f;
}

inline float zero_value(int expert, int group, int column, bool active) {
  if (!active) return 32.0f + float((expert + 3 * group + column) & 7);
  return float(((expert * 7 + group * 5 + column) % 7) - 3) / 32.0f;
}

}  // namespace gemv_perf_fixture
