/***************************************************************************************************
 * Copyright (c) 2026 quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * CUTLASS device_kernel-compatible wrapper for the fixed BF16
 * C64 / Dk128 / Dv128 / BV64 PPU chunked-GDN forward collective.
 **************************************************************************************************/
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "quactlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_collective.cuh"

#if defined(__CUDACC__) && !defined(__HGGCCC__)
#define QZ_PPU_GDN_KERNEL_DEVICE __device__ __forceinline__
#else
#define QZ_PPU_GDN_KERNEL_DEVICE CUTLASS_DEVICE
#endif

namespace cutlass::linear_attention {

template <
    class Arguments_ = PpuChunkedGdnArguments<cutlass::bfloat16_t,
                                              cutlass::bfloat16_t, float>,
    class Traits_ = PpuChunkedGdnTraits<64, 128, 128>>
struct PpuChunkedGdnKernel {
  using Arguments = Arguments_;
  using Params = Arguments;
  using Traits = Traits_;
  using Scheduler = PpuChunkedGdnScheduler<Traits>;
  using Collective =
      PpuChunkedGdnCollectiveBf16C64D128BV64<Traits, Arguments>;
  using SharedStorage = typename Collective::SharedStorage;

  static constexpr std::uint32_t MaxThreadsPerBlock = Collective::kThreadCount;
  static constexpr std::uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = int(sizeof(SharedStorage));

  static_assert(std::is_trivially_copyable_v<Params>,
                "device_kernel Params must be copied by value");
  static_assert(MaxThreadsPerBlock == 128,
                "the C64 global-dot collective owns exactly four warps");
  static_assert(SharedStorageSize == 139776,
                "kernel and collective shared-memory ledgers disagree");

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    (void)workspace;
    return args;
  }

  static bool can_implement(Arguments const& args) {
    return Collective::argument_status(args) == PpuChunkedGdnStatus::kSuccess;
  }

  static std::size_t get_workspace_size(Arguments const&) { return 0; }

  static cutlass::Status initialize_workspace(
      Arguments const& args, void* workspace = nullptr) {
    (void)workspace;
    return can_implement(args) ? cutlass::Status::kSuccess
                               : cutlass::Status::kErrorInvalidProblem;
  }

  static dim3 get_grid_shape(Params const& params) {
    return dim3(
        static_cast<unsigned>(Scheduler::grid_size(params.problem)), 1, 1);
  }

  static dim3 get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

 public:
  QZ_PPU_GDN_KERNEL_DEVICE void operator()(
      Params const& params, char* smem_buf) {
    PpuChunkedGdnWorkTileInfo const work =
        Scheduler::work(static_cast<int>(blockIdx.x), params.problem);
    // This predicate is uniform for the CTA, so it cannot strand peers before
    // a barrier in the collective.
    if (!work.valid) return;

    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem_buf);
    Collective::run(params, work, shared);
  }
};

}  // namespace cutlass::linear_attention

#undef QZ_PPU_GDN_KERNEL_DEVICE
