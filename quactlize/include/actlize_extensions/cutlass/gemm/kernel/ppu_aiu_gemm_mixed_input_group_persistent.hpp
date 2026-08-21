// Copyright (c) 2026, quactlize contributors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Persistent driver for a ragged grouped mixed-input collective.  This file is
// intentionally a new kernel rather than another compatibility branch in the
// shipping non-persistent GemmUniversal specialization: both kernels instantiate
// the exact same collective, and the old kernel remains a bit-exact control.
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/utils.h"
#include "cute/tensor.hpp"

#include "actlize_extensions/cutlass/gemm/kernel/ppu_moe_block_directory.hpp"

namespace cutlass::gemm::kernel {

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_>
class GroupPersistentMixedInputKernel {
 public:
  using ProblemShape = ProblemShape_;
  static_assert(isGroupProblemShape_v<ProblemShape>,
                "GroupPersistentMixedInputKernel requires GroupProblemShape");
  static_assert(cute::rank(typename ProblemShape::UnderlyingProblemShape{}) == 3 ||
                    cute::rank(typename ProblemShape::UnderlyingProblemShape{}) == 4,
                "underlying problem shape must be <M,N,K> or <M,N,K,L>");
  static_assert(cute::is_base_of_v<
                    KernelAiuMultistageMixedInput,
                    typename CollectiveMainloop_::DispatchPolicy::Schedule>,
                "persistent grouped driver requires a mixed-input collective");

  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma = typename CollectiveMainloop::TiledMma;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementA = typename CollectiveMainloop::ElementA;
  using StrideA = typename CollectiveMainloop::StrideA;
  using ElementB = typename CollectiveMainloop::ElementB;
  using StrideB = typename CollectiveMainloop::StrideB;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD = typename CollectiveEpilogue::StrideD;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static constexpr int TileM = int(cute::size<0>(TileShape{}));
  static constexpr int TileN = int(cute::size<1>(TileShape{}));
  using DirectoryScheduler = quactlize::moe_directory::MoeBlockDirectoryScheduler<TileM>;

  struct SharedStorage {
    union SharedTensorStorage {
      using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
      using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
      MainloopSharedStorage mainloop;
      EpilogueSharedStorage epilogue;
    } tensors;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static constexpr uint32_t MaxThreadsPerBlock = cute::size(TiledMma{});
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumMmaWarpGroups = 1;

  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    int representative_m = 0;
    int representative_n = 0;
    int representative_k = 0;
    quactlize::moe_directory::Header const* directory_header = nullptr;
    quactlize::moe_directory::BlockEntry const* directory_entries = nullptr;
    uint64_t logical_work_upper = 0;
    int ctas_per_cu = 0;
    uint32_t grid_ctas_override = 0;
    int splitk = 1;
  };

  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    KernelHardwareInfo hw_info{};
    int representative_m = 0;
    int representative_n = 0;
    int representative_k = 0;
    typename DirectoryScheduler::Params directory{};
    uint64_t logical_work_upper = 0;
    int ctas_per_cu = 0;
    uint32_t grid_ctas_override = 0;
  };

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    (void)workspace;
    int cu_count = args.hw_info.cu_count;
    if (cu_count <= 0) {
      cu_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }
    KernelHardwareInfo hw_info{args.hw_info.device_id, cu_count};
    auto const representative = cute::make_shape(
        args.representative_m, args.representative_n, args.representative_k,
        args.problem_shape.groups());
    return {
        args.mode,
        args.problem_shape,
        CollectiveMainloop::to_underlying_arguments(representative, args.mainloop, nullptr),
        CollectiveEpilogue::to_underlying_arguments(representative, args.epilogue, nullptr),
        hw_info,
        args.representative_m,
        args.representative_n,
        args.representative_k,
        {args.directory_header, args.directory_entries,
         int(cute::ceil_div(args.representative_n, TileN))},
        args.logical_work_upper,
        args.ctas_per_cu,
        args.grid_ctas_override,
    };
  }

  static bool can_implement(Arguments const& args) {
    if ((args.mode != GemmUniversalMode::kGrouped && args.mode != GemmUniversalMode::kArray) ||
        args.problem_shape.groups() <= 0 || args.representative_m <= 0 ||
        args.representative_n <= 0 || args.representative_k <= 0 ||
        args.directory_header == nullptr || args.directory_entries == nullptr ||
        args.logical_work_upper == 0 || args.ctas_per_cu <= 0 || args.splitk != 1) {
      return false;
    }
    int cu_count = args.hw_info.cu_count;
    if (cu_count <= 0) {
      cu_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }
    uint64_t const capacity = cu_count > 0
        ? uint64_t(cu_count) * uint64_t(args.ctas_per_cu) : 0;
    if (args.grid_ctas_override != 0 &&
        (uint64_t(args.grid_ctas_override) > capacity ||
         uint64_t(args.grid_ctas_override) > args.logical_work_upper)) {
      return false;
    }
    auto const representative = cute::make_shape(
        args.representative_m, args.representative_n, args.representative_k,
        args.problem_shape.groups());
    return CollectiveMainloop::can_implement(representative, args.mainloop);
  }

  static size_t get_workspace_size(Arguments const&) { return 0; }

  static cutlass::Status initialize_workspace(
      Arguments const&, void* = nullptr, hggcStream_t = nullptr,
      HostAdapter* = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    if (params.hw_info.cu_count <= 0 || params.ctas_per_cu <= 0 ||
        params.logical_work_upper == 0) {
      return dim3(0, 0, 0);
    }
    uint64_t const resident =
        uint64_t(params.hw_info.cu_count) * uint64_t(params.ctas_per_cu);
    uint64_t requested = params.grid_ctas_override != 0
        ? uint64_t(params.grid_ctas_override) : resident;
    if (requested > params.logical_work_upper) requested = params.logical_work_upper;
    return dim3(static_cast<unsigned>(requested), 1, 1);
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    CUTE_STATIC_ASSERT(is_static<TileShape>::value);
    static_assert(rank(StrideA{}) == 3, "StrideA must be rank-3 [M,K,L]");
    static_assert(rank(StrideB{}) == 3, "StrideB must be rank-3 [N,K,L]");

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    DirectoryScheduler scheduler{params.directory};
    int const thread_idx = int(threadIdx.x);
    int const groups = params.problem_shape.groups();
    auto const block_shape = TileShape{};

    while (true) {
      auto const work = scheduler.fetch_next();
      if (!work.is_valid()) break;

      int const M = work.expert_rows;
      int const N = params.representative_n;
      int const K = params.representative_k;
      int const expert = work.expert;
      auto const problem_shape_mnkl = make_shape(M, N, K, groups);
      auto const block_coord_mnkl = make_coord(work.m_tile, work.n_tile, _, expert);

      CollectiveMainloop collective_mainloop;
      auto load_inputs = collective_mainloop.load_init(
          problem_shape_mnkl, block_coord_mnkl, params.mainloop);
      static_assert(tuple_size_v<decltype(load_inputs)> >= 2,
                    "mixed-input load_init must return A and B tensors");
      Tensor gA = get<0>(load_inputs);

      auto const residue_mnk = make_tuple(
          M - size<0>(block_shape) * work.m_tile,
          N - size<1>(block_shape) * work.n_tile,
          K - size<1>(gA) * size<2>(gA));

      TiledMma tiled_mma;
      Tensor accumulators = make_fragment_like<ElementCompute>(
          partition_fragment_C(tiled_mma, take<0, 2>(block_shape)));
      clear(accumulators);
      auto k_tile_iter = make_coord_iterator(shape<2>(gA));
      int const k_tile_count = size<2>(gA);
      collective_mainloop(params.mainloop, load_inputs, accumulators,
                          k_tile_iter, k_tile_count, thread_idx, smem_buf);

      CollectiveEpilogue epilogue{params.epilogue, shared_storage.tensors.epilogue};
      #pragma hggc dislicm
      {
        epilogue(problem_shape_mnkl, block_shape, block_coord_mnkl,
                 accumulators, tiled_mma, residue_mnk, thread_idx,
                 reinterpret_cast<char*>(&shared_storage.tensors.epilogue));
      }

      // The next work item may select a different expert and immediately reuse
      // both halves of the shared-storage union.  Make that lifetime boundary
      // explicit rather than depending on an epilogue implementation detail.
      __syncthreads();
    }
  }
};

}  // namespace cutlass::gemm::kernel
