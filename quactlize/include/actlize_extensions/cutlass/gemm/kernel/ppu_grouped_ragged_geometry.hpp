// Copyright (c) 2026, quactlize contributors.
// SPDX-License-Identifier: BSD-3-Clause
//
// One host/device authority for flattening ragged grouped output tiles.  Both
// grouped persistent schedulers and their exhaustive host proof use these
// exact functions; a parallel hand-written q model is not evidence.
#pragma once

#include <cstdint>
#include <limits>

#include "cutlass/cutlass.h"

namespace cutlass::gemm::kernel::detail {

struct GroupedRaggedOutputTiles {
  CUTLASS_HOST_DEVICE static constexpr uint64_t ceil_div_u64(
      uint64_t x, uint64_t y) {
    return y == 0 ? 0 : x / y + uint64_t(x % y != 0);
  }

  CUTLASS_HOST_DEVICE static constexpr bool group_tile_count(
      int m, int n, int tile_m, int tile_n, uint64_t& out) {
    if (m < 0 || n <= 0 || tile_m <= 0 || tile_n <= 0) {
      out = 0;
      return false;
    }
    uint64_t const mt = ceil_div_u64(uint64_t(m), uint64_t(tile_m));
    uint64_t const nt = ceil_div_u64(uint64_t(n), uint64_t(tile_n));
    if (mt != 0 && nt > std::numeric_limits<uint64_t>::max() / mt) {
      out = 0;
      return false;
    }
    out = mt * nt;
    return true;
  }

  CUTLASS_HOST_DEVICE static constexpr bool append_group(
      uint64_t prefix, int m, int n, int tile_m, int tile_n,
      uint64_t& next) {
    uint64_t group = 0;
    if (!group_tile_count(m, n, tile_m, tile_n, group) ||
        prefix > std::numeric_limits<uint64_t>::max() - group) {
      next = 0;
      return false;
    }
    next = prefix + group;
    return true;
  }

  // upper_bound(prefix,q)-1.  Repeated entries from zero-row experts are
  // skipped, so no empty expert can acquire a global q.
  template <class Prefix>
  CUTLASS_HOST_DEVICE static constexpr int decode_expert(
      Prefix const& prefix, int groups, int q) {
    if (groups <= 0 || q < 0 || q >= prefix[groups]) {
      return -1;
    }
    int lo = 0;
    int hi = groups;
    while (lo < hi) {
      int const mid = (lo + hi) >> 1;
      if (prefix[mid + 1] <= q) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo < groups ? lo : -1;
  }

  CUTLASS_HOST_DEVICE static constexpr bool decode_local_mn(
      int q, int prefix_e, int m, int tile_m,
      int& m_idx, int& n_idx) {
    if (q < prefix_e || m <= 0 || tile_m <= 0) {
      m_idx = -1;
      n_idx = -1;
      return false;
    }
    int const mt = int(ceil_div_u64(uint64_t(m), uint64_t(tile_m)));
    int const local = q - prefix_e;
    m_idx = local % mt;
    n_idx = local / mt;
    return true;
  }
};

}  // namespace cutlass::gemm::kernel::detail
