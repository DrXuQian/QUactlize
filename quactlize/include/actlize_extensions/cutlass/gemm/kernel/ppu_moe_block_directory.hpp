// Copyright (c) 2026, quactlize contributors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Compact, device-resident work directory for ragged grouped GEMM.
//
// The scheduling idea follows DeepGEMM-for-sail's fused grouped scheduler
// (`deep_gemm/include/deep_gemm/fused_scheduler.cuh`, audited at
// f89eae10c0e90c20630b50e4314448f01321bfba): one
// 16-byte record describes each real M tile, and a fixed resident CTA grid walks
// the exact M-tile x N-tile product with a grid-stride loop.  The data contract is
// deliberately native to quactlize: A is already concatenated in expert order,
// so no token scatter, inverse permutation, or alternate weight representation is
// introduced here.
#pragma once

#include <cstddef>
#include <cstdint>

#include "cutlass/cutlass.h"

namespace quactlize::moe_directory {

struct alignas(16) BlockEntry {
  int32_t expert;
  int32_t expert_rows;
  int32_t expert_block_begin;
  int32_t row_begin;
};
static_assert(sizeof(BlockEntry) == 16, "one MoE M-block record must be one 128-bit load");

enum class BuildStatus : int32_t {
  Success = 0,
  InvalidArgument = 1,
  NegativeRows = 2,
  CapacityExceeded = 3,
};

struct alignas(16) Header {
  int32_t num_m_blocks;
  int32_t status;
  int32_t tile_m;
  int32_t experts;
};
static_assert(sizeof(Header) == 16, "directory header ABI must remain one 128-bit record");

constexpr size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

constexpr int ceil_div_nonnegative(int value, int divisor) {
  return value <= 0 ? 0 : (value + divisor - 1) / divisor;
}

constexpr int maximum_entries(int max_rows, int experts, int tile_m) {
  return max_rows < 0 || experts < 0 || tile_m <= 0
      ? 0 : experts * ceil_div_nonnegative(max_rows, tile_m);
}

constexpr size_t entries_offset() {
  return align_up(sizeof(Header), alignof(BlockEntry));
}

constexpr size_t workspace_bytes(int max_rows, int experts, int tile_m) {
  return entries_offset() + size_t(maximum_entries(max_rows, experts, tile_m)) * sizeof(BlockEntry);
}

struct View {
  Header* header;
  BlockEntry* entries;
  int capacity;
};

inline View make_view(void* workspace, size_t bytes, int max_rows, int experts, int tile_m) {
  int const capacity = maximum_entries(max_rows, experts, tile_m);
  size_t const need = entries_offset() + size_t(capacity) * sizeof(BlockEntry);
  if (workspace == nullptr || bytes < need || capacity <= 0) return {nullptr, nullptr, 0};
  auto* base = static_cast<unsigned char*>(workspace);
  return {
      reinterpret_cast<Header*>(base),
      reinterpret_cast<BlockEntry*>(base + entries_offset()),
      capacity,
  };
}

CUTLASS_HOST_DEVICE
constexpr BlockEntry make_entry(
    int expert, int expert_rows, int expert_block_begin, int row_begin) {
  return {expert, expert_rows, expert_block_begin, row_begin};
}

struct DecodedWork {
  int m_tile;
  int n_tile;
};

// Decode the same two-M-block L2 swizzle used by DeepGEMM, but with runtime N.
// `linear_in_expert` is in the expert-local M-block x N-block product.
CUTLASS_HOST_DEVICE
constexpr DecodedWork decode_swizzled(
    int linear_in_expert, int expert_m_blocks, int n_blocks, int m_blocks_per_group) {
  int const group_span = n_blocks * m_blocks_per_group;
  int const group = linear_in_expert / group_span;
  int const first_m = group * m_blocks_per_group;
  int const in_group = linear_in_expert - group * group_span;
  int const remaining = expert_m_blocks - first_m;
  int const m_in_group = remaining < m_blocks_per_group ? remaining : m_blocks_per_group;
  return {
      first_m + in_group % m_in_group,
      in_group / m_in_group,
  };
}

#if defined(__CUDACC__) || defined(__HGGCCC__)

template <int TileM, int BlockThreads>
__global__ void build_kernel(
    int const* group_rows,
    int const* row_offsets,
    int uniform_rows,
    int experts,
    Header* header,
    BlockEntry* entries,
    int capacity) {
  static_assert(TileM > 0, "TileM must be positive");
  static_assert(BlockThreads >= 1 && BlockThreads <= 1024,
                "directory builder uses one legal CTA");

  __shared__ int scan[BlockThreads];
  __shared__ int status;
  int const tid = int(threadIdx.x);

  if (tid == 0) status = int(BuildStatus::Success);
  __syncthreads();

  int rows = 0;
  if (tid < experts) {
    rows = group_rows[tid];
    if (rows < 0) {
      atomicMax(&status, int(BuildStatus::NegativeRows));
      rows = 0;
    }
  }
  scan[tid] = ceil_div_nonnegative(rows, TileM);
  __syncthreads();

  // Deterministic inclusive scan.  The directory is tiny (one integer per
  // expert), and this runs once before a much larger GEMM, so avoiding a second
  // metadata kernel is more valuable than a sophisticated multi-CTA scan here.
  for (int offset = 1; offset < BlockThreads; offset <<= 1) {
    int add = tid >= offset ? scan[tid - offset] : 0;
    __syncthreads();
    scan[tid] += add;
    __syncthreads();
  }

  int const total = experts > 0 ? scan[experts - 1] : 0;
  if (tid == 0 && total > capacity) status = int(BuildStatus::CapacityExceeded);
  __syncthreads();

  if (tid == 0) {
    header->num_m_blocks = status == int(BuildStatus::Success) ? total : 0;
    header->status = status;
    header->tile_m = TileM;
    header->experts = experts;
  }
  if (status != int(BuildStatus::Success) || tid >= experts) return;

  int const begin = tid == 0 ? 0 : scan[tid - 1];
  int const count = scan[tid] - begin;
  int const row_begin = row_offsets != nullptr ? row_offsets[tid] : tid * uniform_rows;
  BlockEntry const entry = make_entry(tid, rows, begin, row_begin);
  for (int local_m = 0; local_m < count; ++local_m) entries[begin + local_m] = entry;
}

template <int TileM>
inline bool launch_build(
    int const* group_rows,
    int const* row_offsets,
    int uniform_rows,
    int experts,
    View view,
    hggcStream_t stream) {
  if (group_rows == nullptr || view.header == nullptr || view.entries == nullptr ||
      experts <= 0 || experts > 1024 || uniform_rows < 0 || view.capacity <= 0) {
    return false;
  }
  if (experts <= 128) {
    build_kernel<TileM, 128><<<1, 128, 0, stream>>>(
        group_rows, row_offsets, uniform_rows, experts, view.header, view.entries, view.capacity);
  } else if (experts <= 256) {
    build_kernel<TileM, 256><<<1, 256, 0, stream>>>(
        group_rows, row_offsets, uniform_rows, experts, view.header, view.entries, view.capacity);
  } else if (experts <= 512) {
    build_kernel<TileM, 512><<<1, 512, 0, stream>>>(
        group_rows, row_offsets, uniform_rows, experts, view.header, view.entries, view.capacity);
  } else {
    build_kernel<TileM, 1024><<<1, 1024, 0, stream>>>(
        group_rows, row_offsets, uniform_rows, experts, view.header, view.entries, view.capacity);
  }
  return true;
}

struct WorkTile {
  int expert = -1;
  int m_tile = 0;
  int n_tile = 0;
  int expert_rows = 0;
  int row_begin = 0;

  CUTLASS_DEVICE bool is_valid() const { return expert >= 0; }
};

template <int TileM, int MBlocksPerGroup = (TileM == 16 ? 1 : 2)>
class MoeBlockDirectoryScheduler {
 public:
  struct Params {
    Header const* header = nullptr;
    BlockEntry const* entries = nullptr;
    int n_blocks = 0;
  };

  CUTLASS_DEVICE explicit MoeBlockDirectoryScheduler(Params params)
      : entries_(params.entries), n_blocks_(params.n_blocks),
        num_m_blocks_(0), valid_(false), current_iter_(0) {
    // Match DeepGEMM's scheduler lifetime: header/count is a constructor load,
    // not a load repeated for every work item.  Every thread constructs the
    // same scheduler state and subsequently reads only one 16-byte entry.
    Header const header = *params.header;
    num_m_blocks_ = header.num_m_blocks;
    valid_ = header.status == int(BuildStatus::Success) &&
             header.tile_m == TileM && header.experts > 0 &&
             n_blocks_ > 0 && num_m_blocks_ >= 0;
  }

  CUTLASS_DEVICE WorkTile fetch_next() {
    WorkTile out{};
    if (!valid_) return out;

    uint64_t const linear = uint64_t(current_iter_++) * uint64_t(gridDim.x) + uint64_t(blockIdx.x);
    uint64_t const total = uint64_t(num_m_blocks_) * uint64_t(n_blocks_);
    if (linear >= total) return out;

    int const directory_index = int(linear / uint64_t(n_blocks_));
    BlockEntry const entry = entries_[directory_index];
    int const expert_m_blocks = ceil_div_nonnegative(entry.expert_rows, TileM);
    int const linear_in_expert = int(linear) - entry.expert_block_begin * n_blocks_;
    DecodedWork const decoded = decode_swizzled(
        linear_in_expert, expert_m_blocks, n_blocks_, MBlocksPerGroup);

    out.expert = entry.expert;
    out.m_tile = decoded.m_tile;
    out.n_tile = decoded.n_tile;
    out.expert_rows = entry.expert_rows;
    out.row_begin = entry.row_begin;
    return out;
  }

 private:
  BlockEntry const* entries_;
  int n_blocks_;
  int num_m_blocks_;
  bool valid_;
  int current_iter_;
};

#endif  // defined(__CUDACC__) || defined(__HGGCCC__)

}  // namespace quactlize::moe_directory
