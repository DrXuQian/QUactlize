/***************************************************************************************************
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Host/device-independent ABI arithmetic for fixed Split-K actual-last completion.
 **************************************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(CUTLASS_HOST_DEVICE)
#define QUACTLIZE_COMPLETION_HOST_DEVICE CUTLASS_HOST_DEVICE
#elif defined(__CUDACC__) || defined(__HGGCCC__)
#define QUACTLIZE_COMPLETION_HOST_DEVICE __host__ __device__
#else
#define QUACTLIZE_COMPLETION_HOST_DEVICE
#endif

namespace cutlass::gemm::kernel::fixed_splitk {

struct CompletionWorkspace {
  size_t partial_bytes = 0;
  size_t counter_offset = 0;
  size_t counter_bytes = 0;
  size_t total_bytes = 0;
  uint64_t output_tiles = 0;

  constexpr bool is_valid() const {
    return partial_bytes != 0 && counter_offset >= partial_bytes &&
        counter_bytes != 0 && total_bytes >= counter_offset &&
        total_bytes - counter_offset == counter_bytes && output_tiles != 0;
  }
};

constexpr size_t kCompletionAlignment = 128;

constexpr bool checked_align_up(
    size_t value, size_t alignment, size_t& result) {
  result = 0;
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return false;
  }
  size_t const mask = alignment - 1;
  if (value > (std::numeric_limits<size_t>::max)() - mask) {
    return false;
  }
  result = (value + mask) & ~mask;
  return true;
}

// The caller supplies partial_bytes so this protocol cannot silently change the
// established [S][M][N] FP32 ABI.  Counters are one int32 per global output
// tile q, aligned away from the partial planes.
constexpr CompletionWorkspace make_completion_workspace(
    size_t partial_bytes, uint64_t output_tiles) {
  size_t counter_offset = 0;
  if (partial_bytes == 0 || output_tiles == 0 ||
      output_tiles > uint64_t((std::numeric_limits<size_t>::max)() /
                              sizeof(int32_t)) ||
      !checked_align_up(partial_bytes, kCompletionAlignment, counter_offset)) {
    return {};
  }
  size_t const counter_bytes = size_t(output_tiles) * sizeof(int32_t);
  if (counter_offset > (std::numeric_limits<size_t>::max)() - counter_bytes) {
    return {};
  }
  return CompletionWorkspace{
      partial_bytes, counter_offset, counter_bytes,
      counter_offset + counter_bytes, output_tiles};
}

QUACTLIZE_COMPLETION_HOST_DEVICE constexpr bool completion_arrival_is_valid(
    uint32_t old_count, uint32_t peer_count) {
  return peer_count > 1 && old_count < peer_count;
}

// The returned value of fetch-old arrival decides which physical CTA performs
// final reduction.  It is intentionally independent of logical peer_idx.
QUACTLIZE_COMPLETION_HOST_DEVICE constexpr bool completion_arrival_is_last(
    uint32_t old_count, uint32_t peer_count) {
  return completion_arrival_is_valid(old_count, peer_count) &&
      old_count + 1 == peer_count;
}

}  // namespace cutlass::gemm::kernel::fixed_splitk

#undef QUACTLIZE_COMPLETION_HOST_DEVICE
