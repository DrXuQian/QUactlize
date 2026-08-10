// Copyright (c) 2026, quactlize contributors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Dense persistent driver for the mixed-input PPU collective.  The vendor mixed-input
// GemmUniversal specialization names a persistent scheduler type, but launches the full
// M-tile x N-tile grid and never constructs that scheduler.  This named kernel is kept
// separate from the vendor specialization so the non-persistent path remains an exact A/B
// control and actlize stays pristine.
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/utils.h"
#include "cute/tensor.hpp"

namespace cutlass::gemm::kernel {

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_>
class PersistentMixedInputKernel {
public:
  using ProblemShape = ProblemShape_;
  static_assert(cute::rank(ProblemShape{}) == 3 || cute::rank(ProblemShape{}) == 4,
                "ProblemShape must be <M,N,K> or <M,N,K,L>");
  static_assert(cute::is_base_of_v<KernelAiuMultistageMixedInput,
                                   typename CollectiveMainloop_::DispatchPolicy::Schedule>,
                "PersistentMixedInputKernel requires a mixed-input mainloop");
  static_assert(!isGroupProblemShape_v<ProblemShape>,
                "PersistentMixedInputKernel is dense-only");

  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma = typename CollectiveMainloop::TiledMma;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using ElementA = typename CollectiveMainloop::ElementA;
  using StrideA = typename CollectiveMainloop::StrideA;
  using ElementB = typename CollectiveMainloop::ElementB;
  using StrideB = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
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

  // The mixed-input kernel is not clustered.  Keep this explicit instead of assuming the
  // collective's dispatch policy exposes a ClusterShape member on every actlize revision.
  using ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;
  using TileSchedulerTag = PersistentScheduler;
  using TileScheduler = typename detail::TileSchedulerSelector<
      TileSchedulerTag, ArchTag, TileShape, ClusterShape>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Plain persistence is a serial mainloop -> epilogue -> next-tile loop.  Their shared
  // lifetimes do not overlap, so persistence does NOT require mainloop+epilogue bytes.
  // This is the same union used by actlize's existing persistent AIU kernel.
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
    TileSchedulerArguments scheduler{};
    // Filled from GemmUniversalAdapter::maximum_active_blocks() for this exact kernel.
    // Keep it separate from the real CU count: CU*occupancy is a physical worker count,
    // not a different piece of hardware.
    int ctas_per_cu = 0;
  };

  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
    int ctas_per_cu = 0;
  };

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    auto problem_shape = args.problem_shape;
    if constexpr (detail::Has_SwapAB_v<CollectiveMainloop>) {
      cute::get<0>(problem_shape) = cute::get<1>(args.problem_shape);
      cute::get<1>(problem_shape) = cute::get<0>(args.problem_shape);
    }
    auto problem_shape_mnkl = cute::append<4>(problem_shape, cute::Int<1>{});

    int cu_count = args.hw_info.cu_count;
    if (cu_count <= 0) {
      cu_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }
    KernelHardwareInfo hw_info{args.hw_info.device_id, cu_count};
    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(
        problem_shape_mnkl, TileShape{}, ClusterShape{}, hw_info, args.scheduler,
        /*workspace=*/nullptr, /*epilogue_subtile=*/1);

    (void)workspace;
    return {
        args.mode,
        problem_shape,
        CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, nullptr),
        CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, nullptr),
        hw_info,
        scheduler,
        args.ctas_per_cu,
    };
  }

  static bool can_implement(Arguments const& args) {
    return args.mode == GemmUniversalMode::kGemm && args.ctas_per_cu > 0 &&
           TileScheduler::can_implement(args.scheduler);
  }

  static size_t get_workspace_size(Arguments const&) { return 0; }

  static cutlass::Status initialize_workspace(
      Arguments const&, void* = nullptr, hggcStream_t = nullptr,
      HostAdapter* = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    if (params.hw_info.cu_count <= 0 || params.ctas_per_cu <= 0) {
      return dim3(0, 0, 0);
    }

    uint64_t const resident_workers =
        uint64_t(params.hw_info.cu_count) * uint64_t(params.ctas_per_cu);
    uint64_t const logical_tiles = params.scheduler.blocks_per_problem_;
    uint64_t const workers = resident_workers < logical_tiles ? resident_workers : logical_tiles;

    // Match StaticPersistentTileScheduler's linear block-id convention.  ClusterShape is 1,
    // so one long physical dimension is sufficient and no cluster rounding is involved.
    if (params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN) {
      return dim3(1, static_cast<unsigned>(workers), 1);
    }
    return dim3(static_cast<unsigned>(workers), 1, 1);
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace cute;

    TileScheduler scheduler{params.scheduler};
    auto work_tile_info = scheduler.get_current_work();

    CUTE_STATIC_ASSERT(is_static<TileShape>::value);
    static_assert(rank(StrideA{}) == 3, "StrideA must be rank-3 [M,K,L]");
    static_assert(rank(StrideB{}) == 3, "StrideB must be rank-3 [N,K,L]");
    static_assert(rank(StrideC{}) == 3, "StrideC must be rank-3 [M,N,L]");
    static_assert(rank(StrideD{}) == 3, "StrideD must be rank-3 [M,N,L]");

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    auto problem_shape_mnkl = append<4>(params.problem_shape, Int<1>{});
    auto [M, N, K, L] = problem_shape_mnkl;
    (void)L;
    int const thread_idx = int(threadIdx.x);
    auto const blk_shape = TileShape{};

    while (work_tile_info.is_valid()) {
      if (!TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
        auto next = scheduler.fetch_next_work(work_tile_info);
        work_tile_info = get<0>(next);
        continue;
      }

      int const m_coord = work_tile_info.M_idx;
      int const n_coord = work_tile_info.N_idx;
      int const l_coord = work_tile_info.L_idx;
      auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);

      // load_init carries the mixed-input B/scale/zero descriptors and tile coordinates,
      // so it must be rebuilt for every scheduler work item.
      CollectiveMainloop collective_mainloop;
      auto load_inputs = collective_mainloop.load_init(
          problem_shape_mnkl, blk_coord_mnkl, params.mainloop);
      static_assert(tuple_size_v<decltype(load_inputs)> >= 2,
                    "mixed-input load_init must return A and B tensors");
      Tensor gA = get<0>(load_inputs);
      Tensor gB = get<1>(load_inputs);

      auto m_max_coord = M - size<0>(gA) * m_coord;
      auto n_max_coord = N - size<0>(gB) * n_coord;
      auto k_residue = K - size<1>(gA) * size<2>(gA);
      auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);

      TiledMma tiled_mma;
      Tensor accumulators = make_fragment_like<ElementCompute>(
          partition_fragment_C(tiled_mma, take<0, 2>(blk_shape)));
      clear(accumulators);

      auto k_tile_iter = make_coord_iterator(shape<2>(gA));
      int const k_tile_count = size<2>(gA);
      collective_mainloop(params.mainloop, load_inputs, accumulators,
                          k_tile_iter, k_tile_count, thread_idx, smem_buf);

      CollectiveEpilogue epilogue{params.epilogue, shared_storage.tensors.epilogue};
      // Match actlize's existing persistent kernel: do not let the device compiler
      // hoist an epilogue whose tile coordinate changes on every work-loop iteration.
      #pragma hggc dislicm
      {
        epilogue(problem_shape_mnkl, blk_shape, blk_coord_mnkl, accumulators,
                 tiled_mma, residue_mnk, thread_idx,
                 reinterpret_cast<char*>(&shared_storage.tensors.epilogue));
      }

      auto next = scheduler.fetch_next_work(work_tile_info);
      work_tile_info = get<0>(next);
    }
  }
};

} // namespace cutlass::gemm::kernel
