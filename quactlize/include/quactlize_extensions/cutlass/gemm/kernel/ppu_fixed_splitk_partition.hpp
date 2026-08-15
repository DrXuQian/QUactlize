/***************************************************************************************************
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(CUTLASS_HOST_DEVICE)
#define QUACTLIZE_FIXED_SPLITK_HOST_DEVICE CUTLASS_HOST_DEVICE
#elif defined(__CUDACC__) || defined(__HGGCCC__)
#define QUACTLIZE_FIXED_SPLITK_HOST_DEVICE __host__ __device__
#else
#define QUACTLIZE_FIXED_SPLITK_HOST_DEVICE
#endif

namespace cutlass::gemm::kernel::fixed_splitk {

// One ABI owns both host decomposition and future device work assignment.
// K coordinates are tile ordinals, not elements or bytes.  Logical work is
// q-major with peer_idx as the fast dimension:
//
//   linear_work = q * splits + peer_idx
//
// This makes logical_work_id compact and unique.  It is deliberately not a
// physical FP32 workspace offset: v1 stores split-major [S][M][N] planes for
// coalesced reduction, whereas the work queue is q-major.  A future
// last-arriver protocol can use q as its completion-counter slot and
// peer_count as its terminal arrival value without changing this descriptor.
struct Work {
  static constexpr uint64_t InvalidQ = std::numeric_limits<uint64_t>::max();
  static constexpr uint64_t InvalidLogicalWorkId =
      std::numeric_limits<uint64_t>::max();

  uint64_t q = InvalidQ;
  uint32_t k_begin = 0;
  uint32_t k_count = 0;
  uint32_t peer_idx = 0;
  uint32_t peer_count = 0;
  uint64_t logical_work_id = InvalidLogicalWorkId;

  QUACTLIZE_FIXED_SPLITK_HOST_DEVICE constexpr bool is_valid() const {
    return q != InvalidQ && k_count != 0 && peer_count != 0 &&
           peer_idx < peer_count && logical_work_id != InvalidLogicalWorkId;
  }

  QUACTLIZE_FIXED_SPLITK_HOST_DEVICE constexpr bool is_first_peer() const {
    return is_valid() && peer_idx == 0;
  }

  // This is the final peer in canonical K order, not necessarily the CTA that
  // arrives last at a future completion counter.
  QUACTLIZE_FIXED_SPLITK_HOST_DEVICE constexpr bool is_final_peer() const {
    return is_valid() && peer_idx + 1 == peer_count;
  }

  QUACTLIZE_FIXED_SPLITK_HOST_DEVICE constexpr uint64_t completion_slot() const {
    return q;
  }

  QUACTLIZE_FIXED_SPLITK_HOST_DEVICE static constexpr Work invalid() {
    return {};
  }
};

static_assert(std::is_standard_layout_v<Work> &&
                  std::is_trivially_copyable_v<Work>,
              "fixed Split-K Work must cross the host/device ABI unchanged");
static_assert(sizeof(Work) == 32 && alignof(Work) == 8 &&
                  offsetof(Work, q) == 0 &&
                  offsetof(Work, k_begin) == 8 &&
                  offsetof(Work, k_count) == 12 &&
                  offsetof(Work, peer_idx) == 16 &&
                  offsetof(Work, peer_count) == 20 &&
                  offsetof(Work, logical_work_id) == 24,
              "fixed Split-K Work ABI changed");

struct Params {
  uint64_t output_tiles = 0;
  uint64_t work_units = 0;
  uint32_t k_tiles_per_output = 0;
  uint32_t splits = 0;
  uint32_t k_tiles_per_split = 0;
  uint32_t reserved = 0;

  QUACTLIZE_FIXED_SPLITK_HOST_DEVICE constexpr bool is_valid() const {
    bool const supported_splits =
        splits == 1 || splits == 2 || splits == 4 || splits == 8;
    return output_tiles != 0 && work_units != 0 &&
           k_tiles_per_output != 0 && supported_splits &&
           k_tiles_per_output % splits == 0 &&
           k_tiles_per_split == k_tiles_per_output / splits &&
           k_tiles_per_split != 0 && reserved == 0 &&
           output_tiles <= std::numeric_limits<uint64_t>::max() / splits &&
           work_units == output_tiles * uint64_t(splits);
  }

  QUACTLIZE_FIXED_SPLITK_HOST_DEVICE static constexpr Params invalid() {
    return {};
  }
};

static_assert(std::is_standard_layout_v<Params> &&
                  std::is_trivially_copyable_v<Params>,
              "fixed Split-K Params must cross the host/device ABI unchanged");
static_assert(sizeof(Params) == 32 && alignof(Params) == 8 &&
                  offsetof(Params, output_tiles) == 0 &&
                  offsetof(Params, work_units) == 8 &&
                  offsetof(Params, k_tiles_per_output) == 16 &&
                  offsetof(Params, splits) == 20 &&
                  offsetof(Params, k_tiles_per_split) == 24 &&
                  offsetof(Params, reserved) == 28,
              "fixed Split-K Params ABI changed");

QUACTLIZE_FIXED_SPLITK_HOST_DEVICE constexpr bool supported_split_count(
    uint32_t splits) {
  return splits == 1 || splits == 2 || splits == 4 || splits == 8;
}

// V1 deliberately rejects ragged K partitions.  This is stronger than a
// ceil-div decomposition: every peer owns the same positive contiguous range,
// and no mainloop can step beyond Kt on the final peer.
QUACTLIZE_FIXED_SPLITK_HOST_DEVICE constexpr Params make_params(
    uint64_t output_tiles, uint32_t k_tiles_per_output, uint32_t splits) {
  if (output_tiles == 0 || k_tiles_per_output == 0 ||
      !supported_split_count(splits) ||
      k_tiles_per_output % splits != 0 ||
      output_tiles > std::numeric_limits<uint64_t>::max() / splits) {
    return Params::invalid();
  }

  uint32_t const k_tiles_per_split = k_tiles_per_output / splits;
  if (k_tiles_per_split == 0) {
    return Params::invalid();
  }

  return Params{output_tiles,
                output_tiles * uint64_t(splits),
                k_tiles_per_output,
                splits,
                k_tiles_per_split,
                0};
}

QUACTLIZE_FIXED_SPLITK_HOST_DEVICE constexpr uint64_t linear_work_id(
    Params const& params, uint64_t q, uint32_t peer_idx) {
  if (!params.is_valid() || q >= params.output_tiles ||
      peer_idx >= params.splits) {
    return Work::InvalidLogicalWorkId;
  }
  return q * uint64_t(params.splits) + peer_idx;
}

QUACTLIZE_FIXED_SPLITK_HOST_DEVICE constexpr Work work_for(
    Params const& params, uint64_t q, uint32_t peer_idx) {
  uint64_t const linear = linear_work_id(params, q, peer_idx);
  if (linear == Work::InvalidLogicalWorkId) {
    return Work::invalid();
  }

  return Work{q,
              peer_idx * params.k_tiles_per_split,
              params.k_tiles_per_split,
              peer_idx,
              params.splits,
              linear};
}

QUACTLIZE_FIXED_SPLITK_HOST_DEVICE constexpr Work work_for_linear(
    Params const& params, uint64_t linear) {
  if (!params.is_valid() || linear >= params.work_units) {
    return Work::invalid();
  }
  uint64_t const q = linear / params.splits;
  uint32_t const peer_idx = uint32_t(linear % params.splits);
  return work_for(params, q, peer_idx);
}

// Validate a descriptor at the ABI seam before it is used for address
// arithmetic.  The oracle additionally proves global exact-once coverage; this
// predicate is the cheap per-work counterpart suitable for future kernels.
QUACTLIZE_FIXED_SPLITK_HOST_DEVICE constexpr bool work_matches_params(
    Params const& params, Work const& work) {
  if (!params.is_valid() || !work.is_valid() ||
      work.q >= params.output_tiles ||
      work.peer_count != params.splits ||
      work.peer_idx >= params.splits ||
      work.k_count != params.k_tiles_per_split ||
      work.k_begin != work.peer_idx * params.k_tiles_per_split ||
      uint64_t(work.k_begin) + uint64_t(work.k_count) >
          params.k_tiles_per_output) {
    return false;
  }
  return work.logical_work_id ==
         linear_work_id(params, work.q, work.peer_idx);
}

}  // namespace cutlass::gemm::kernel::fixed_splitk

#undef QUACTLIZE_FIXED_SPLITK_HOST_DEVICE
