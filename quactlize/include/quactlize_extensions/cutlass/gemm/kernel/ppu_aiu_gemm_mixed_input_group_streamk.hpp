// Copyright (c) 2026, quactlize contributors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Experimental grouped mixed-input Stream-K kernel.  The ordinary grouped
// kernel remains the shipping/control path.  This wrapper flattens ragged
// output tiles to one dense scheduler coordinate q, then decodes q separately
// for the grouped mainloop and epilogue.  Keeping those two coordinate systems
// separate is essential: Stream-K's global lock index is the scheduler output
// tile index, so replacing q with an expert-local m tile silently aliases locks
// across experts.
#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/ppu_tile_scheduler_stream_k.hpp"
#include "cutlass/utils.h"
#include "cute/tensor.hpp"

namespace cutlass::gemm::kernel {

template <class GroupProblemShape_, class CollectiveMainloop_,
          class CollectiveEpilogue_, uint32_t MinSkIters = 8u>
class GroupStreamKMixedInputKernel {
 public:
  using ProblemShape = GroupProblemShape_;
  using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;
  static_assert(isGroupProblemShape_v<ProblemShape>,
                "GroupStreamKMixedInputKernel requires GroupProblemShape");
  static_assert(cute::rank(UnderlyingProblemShape{}) == 3,
                "grouped Stream-K v1 requires per-expert <M,N,K>");
  static_assert(std::is_trivially_copyable_v<UnderlyingProblemShape>,
                "grouped Stream-K host shape mirror must be byte-copyable");

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

  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using SchedulerProblemShape = cute::Shape<int, int, int, int>;
  static constexpr uint32_t MaxThreadsPerBlock = cute::size(TiledMma{});
  using TileScheduler = detail::PersistentTileSchedulerPPUStreamK<
      TileShape, ClusterShape, MinSkIters, MaxThreadsPerBlock>;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;
  using ExpectedTileSchedulerParams =
      detail::PersistentTileSchedulerPPUStreamKParamsT<MinSkIters>;

  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumMmaWarpGroups = 1;
  static constexpr int MinWorkspaceAlignment = 16;
  static constexpr bool IsGroupedStreamK = true;

  static_assert(cute::is_base_of_v<KernelAiuMultistageMixedInput,
                                   typename DispatchPolicy::Schedule>,
                "grouped Stream-K requires the mixed-input mainloop");
  static_assert(!CollectiveMainloop::SwapAB,
                "grouped Stream-K v1 requires q to remain on scheduler M");
  static_assert(std::is_same_v<TileSchedulerParams,
                               ExpectedTileSchedulerParams>,
                "grouped Stream-K lost configured MinSkIters");
  static_assert(MinWorkspaceAlignment % alignof(UnderlyingProblemShape) == 0,
                "workspace alignment must cover the host shape mirror");
  static_assert(MinSkIters >= uint32_t(DispatchPolicy::Stages - 1),
                "Stream-K stripe is shorter than pipeline startup");
  static_assert(MaxThreadsPerBlock == 64u || MaxThreadsPerBlock == 128u,
                "grouped mixed-input Stream-K supports exactly 64- or 128-thread CTAs");
  static_assert(TileScheduler::FixupThreadCount == MaxThreadsPerBlock,
                "grouped Stream-K fixup cohort must equal the exact CTA thread count");
  static_assert(cute::is_same_v<ElementAccumulator, ElementCompute>,
                "Stream-K scratch and live accumulators must have one type");

  // Tile i executes mainloop -> fixup/epilogue before tile i+1 starts.  This
  // union remains legal only while those two tiles are not overlapped.
  struct SharedStorage {
    union SharedTensorStorage {
      typename CollectiveMainloop::SharedStorage mainloop;
      typename CollectiveEpilogue::SharedStorage epilogue;
    } tensors;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Gate-only census.  All pointers are null in a performance launch.
  // totals: [0] fixup work, [1] epilogue work, [2] reduction work,
  //         [3] fixup+epilogue work, [4] q out of range,
  //         [5] q decoded to an empty expert.
  struct Census {
    uint32_t* peer_count = nullptr;       // [Q]
    uint32_t* k_visit_count = nullptr;    // [Q * Kt]
    uint32_t* totals = nullptr;           // [6]
    uint32_t peer_capacity = 0;
    uint64_t k_visit_capacity = 0;
  };

  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};         // real device/CU count
    TileSchedulerArguments scheduler{};
    int ctas_per_cu = 0;                  // occupancy of this exact kernel
    int representative_m = 0;
    int representative_n = 0;
    int representative_k = 0;
    // Set by the owned type builder. It carries compile-time layout-domain
    // checks (for example interleaved N/K divisibility) into can_implement.
    bool domain_valid = false;
    Census census{};
  };

  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    SchedulerProblemShape scheduler_problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    KernelHardwareInfo real_hw_info{};
    KernelHardwareInfo scheduler_hw_info{};  // cu_count == physical workers
    TileSchedulerParams scheduler{};
    void* workspace_base = nullptr;
    size_t scheduler_workspace_bytes = 0;
    size_t scheduler_barrier_offset = 0;
    size_t scheduler_barrier_bytes = 0;
    int const* output_tile_prefix = nullptr; // [groups+1], after scheduler ws
    UnderlyingProblemShape const* host_shape_mirror = nullptr; // [groups]
    int output_tiles = 0;
    int k_tiles_per_output = 0;
    int ctas_per_cu = 0;
    Census census{};
  };

 private:
  struct HostGeometry {
    int groups = 0;
    int n = 0;
    int k = 0;
    int q = 0;
    bool valid = false;
  };

  struct WorkspaceLayout {
    size_t scheduler_bytes = 0;
    size_t prefix_offset = 0;
    size_t prefix_bytes = 0;
    size_t shape_offset = 0;
    size_t shape_bytes = 0;
    size_t epilogue_offset = 0;
    size_t epilogue_bytes = 0;
    size_t total_bytes = 0;
  };

  static HostGeometry host_geometry(Arguments const& args,
                                    std::vector<int>* prefix = nullptr) {
    HostGeometry out{};
    ProblemShape const& ps = args.problem_shape;
    if (ps.num_groups <= 0 || ps.problem_shapes == nullptr ||
        ps.host_problem_shapes == nullptr) {
      return out;
    }

    int64_t q = 0;
    int n0 = 0;
    int k0 = 0;
    if (prefix != nullptr) {
      prefix->assign(size_t(ps.num_groups) + 1, 0);
    }
    constexpr int TM = int(cute::size<0>(TileShape{}));
    constexpr int TN = int(cute::size<1>(TileShape{}));
    for (int e = 0; e < ps.num_groups; ++e) {
      auto shape = ps.host_problem_shapes[e];
      int const m = int(cute::get<0>(shape));
      int const n = int(cute::get<1>(shape));
      int const k = int(cute::get<2>(shape));
      if (m < 0 || n <= 0 || k <= 0) {
        return out;
      }
      if (e == 0) {
        n0 = n;
        k0 = k;
      } else if (n != n0 || k != k0) {
        return out;  // v1 supports ragged M, uniform N/K only
      }
      int64_t const mt = (int64_t(m) + TM - 1) / TM;
      int64_t const nt = (int64_t(n) + TN - 1) / TN;
      q += mt * nt;
      if (q > std::numeric_limits<int>::max()) {
        return out;
      }
      if (prefix != nullptr) {
        (*prefix)[size_t(e) + 1] = int(q);
      }
    }
    if (q <= 0 || q * int64_t(TM) > std::numeric_limits<int>::max()) {
      return out;
    }
    if ((args.representative_n > 0 && args.representative_n != n0) ||
        (args.representative_k > 0 && args.representative_k != k0)) {
      return out;
    }
    out.groups = ps.num_groups;
    out.n = n0;
    out.k = k0;
    out.q = int(q);
    out.valid = true;
    return out;
  }

  static SchedulerProblemShape scheduler_problem_shape(HostGeometry const& g) {
    return cute::make_shape(g.q * int(cute::size<0>(TileShape{})),
                            int(cute::size<1>(TileShape{})), g.k, 1);
  }

  static auto representative_problem_shape(Arguments const& args,
                                           HostGeometry const& g) {
    int const m = args.representative_m > 0 ? args.representative_m : 1;
    return cute::make_shape(m, g.n, g.k, g.groups);
  }

  static KernelHardwareInfo real_hw_info(Arguments const& args) {
    int cu_count = args.hw_info.cu_count;
    if (cu_count <= 0) {
      cu_count = KernelHardwareInfo::query_device_multiprocessor_count(
          args.hw_info.device_id);
    }
    return KernelHardwareInfo{args.hw_info.device_id, cu_count};
  }

  static KernelHardwareInfo scheduler_hw_info(Arguments const& args) {
    KernelHardwareInfo real = real_hw_info(args);
    int64_t const workers = int64_t(real.cu_count) * int64_t(args.ctas_per_cu);
    int const bounded = workers > 0 &&
                                workers <= std::numeric_limits<int>::max()
                            ? int(workers)
                            : 0;
    return KernelHardwareInfo{real.device_id, bounded};
  }

  static size_t scheduler_workspace_size(Arguments const& args,
                                         HostGeometry const& g) {
    if (!g.valid || args.ctas_per_cu <= 0) {
      return 0;
    }
    auto shape = scheduler_problem_shape(g);
    return TileScheduler::template get_workspace_size<SchedulerProblemShape,
                                                       ElementAccumulator>(
        args.scheduler, shape, scheduler_hw_info(args), NumMmaWarpGroups);
  }

  static WorkspaceLayout workspace_layout(Arguments const& args,
                                          HostGeometry const& g) {
    WorkspaceLayout out{};
    if (!g.valid) {
      return out;
    }
    out.scheduler_bytes = scheduler_workspace_size(args, g);
    out.prefix_offset = round_nearest(out.scheduler_bytes,
                                      MinWorkspaceAlignment);
    out.prefix_bytes = sizeof(int) * size_t(g.groups + 1);
    out.shape_offset = round_nearest(out.prefix_offset + out.prefix_bytes,
                                     MinWorkspaceAlignment);
    out.shape_bytes = sizeof(UnderlyingProblemShape) * size_t(g.groups);
    out.epilogue_offset = round_nearest(out.shape_offset + out.shape_bytes,
                                        MinWorkspaceAlignment);
    auto rep = representative_problem_shape(args, g);
    out.epilogue_bytes =
        CollectiveEpilogue::get_workspace_size(rep, args.epilogue);
    out.total_bytes = round_nearest(out.epilogue_offset + out.epilogue_bytes,
                                    MinWorkspaceAlignment);
    return out;
  }

 public:
  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    HostGeometry const g = host_geometry(args);
    WorkspaceLayout const layout = workspace_layout(args, g);
    auto scheduler_shape = scheduler_problem_shape(g);
    auto representative_shape = representative_problem_shape(args, g);
    KernelHardwareInfo real = real_hw_info(args);
    KernelHardwareInfo workers = scheduler_hw_info(args);
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);

    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(
        scheduler_shape, TileShape{}, ClusterShape{}, workers, args.scheduler,
        workspace, /*epilogue_subtile=*/1);
    size_t const barrier_bytes =
        TileSchedulerParams::get_barrier_workspace_size(
            scheduler.sk_tiles_, NumMmaWarpGroups,
            cutlass::sizeof_bits<typename TileScheduler::BarrierType>::value);
    size_t const barrier_offset =
        barrier_bytes <= layout.scheduler_bytes
            ? layout.scheduler_bytes - barrier_bytes
            : layout.scheduler_bytes;
    void* epilogue_workspace =
        workspace_ptr ? workspace_ptr + layout.epilogue_offset : nullptr;

    return {
        args.mode,
        args.problem_shape,
        scheduler_shape,
        CollectiveMainloop::to_underlying_arguments(
            representative_shape, args.mainloop, nullptr),
        CollectiveEpilogue::to_underlying_arguments(
            representative_shape, args.epilogue, epilogue_workspace),
        real,
        workers,
        scheduler,
        workspace,
        layout.scheduler_bytes,
        barrier_offset,
        barrier_bytes,
        workspace_ptr ? reinterpret_cast<int const*>(
                            workspace_ptr + layout.prefix_offset)
                      : nullptr,
        workspace_ptr ? reinterpret_cast<UnderlyingProblemShape const*>(
                            workspace_ptr + layout.shape_offset)
                      : nullptr,
        g.q,
        g.valid ? int(cute::ceil_div(g.k, int(cute::size<2>(TileShape{}))))
                : 0,
        args.ctas_per_cu,
        args.census,
    };
  }

  static bool can_implement(Arguments const& args) {
    HostGeometry const g = host_geometry(args);
    KernelHardwareInfo real = real_hw_info(args);
    KernelHardwareInfo workers = scheduler_hw_info(args);
    uint64_t const qk = uint64_t(g.q) *
                        uint64_t(g.valid ? cute::ceil_div(
                            g.k, int(cute::size<2>(TileShape{}))) : 0);
    bool const census_ok =
        (args.census.peer_count == nullptr &&
         args.census.k_visit_count == nullptr &&
         args.census.totals == nullptr) ||
        (args.census.peer_count != nullptr &&
         args.census.k_visit_count != nullptr &&
         args.census.totals != nullptr &&
         args.census.peer_capacity >= uint32_t(g.q) &&
         args.census.k_visit_capacity >= qk);
    return args.mode == GemmUniversalMode::kGrouped && g.valid &&
           real.cu_count > 0 && workers.cu_count > 0 &&
           args.ctas_per_cu > 0 && args.domain_valid && census_ok &&
           args.mainloop.group_row_offsets != nullptr &&
           args.scheduler.splits == 1 &&
           args.scheduler.max_swizzle_size == 1 &&
           args.scheduler.raster_order == TileScheduler::RasterOrderOptions::AlongN &&
           args.scheduler.reduction_mode ==
               TileSchedulerParams::ReductionMode::Deterministic &&
           args.scheduler.decomposition_mode ==
               TileSchedulerParams::DecompositionMode::StreamK &&
           TileScheduler::can_implement(args.scheduler);
  }

  static size_t get_workspace_size(Arguments const& args) {
    HostGeometry const g = host_geometry(args);
    return workspace_layout(args, g).total_bytes;
  }

  static cutlass::Status initialize_workspace(
      Arguments const& args, void* workspace = nullptr,
      hggcStream_t stream = nullptr, HostAdapter* host_adapter = nullptr) {
    if (!can_implement(args)) {
      return Status::kErrorInvalidProblem;
    }
    HostGeometry const g = host_geometry(args);
    WorkspaceLayout const layout = workspace_layout(args, g);
    if (layout.total_bytes > 0 && workspace == nullptr) {
      return Status::kErrorWorkspaceNull;
    }
    auto shape = scheduler_problem_shape(g);
    KernelHardwareInfo workers = scheduler_hw_info(args);
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);

    Status status =
        TileScheduler::template initialize_workspace<SchedulerProblemShape,
                                                     ElementAccumulator>(
            args.scheduler, workspace_ptr, stream, shape, workers,
            NumMmaWarpGroups, /*epilogue_subtile=*/1,
            /*num_accumulator_mtxs=*/1, host_adapter);
    if (status != Status::kSuccess) {
      return status;
    }

    // P and the shape mirror are caller-owned host data (P itself is a local
    // vector), so both copies remain blocking. They complete before the timed
    // batch. Timed launches call reset_scheduler_workspace_after_prefix_install
    // below, which resets the vendor barrier tail and cannot silently opt out
    // via a bool.
    std::vector<int> prefix;
    HostGeometry const checked = host_geometry(args, &prefix);
    if (!checked.valid || checked.q != g.q) {
      return Status::kErrorInvalidProblem;
    }
    hggcError_t copy_status = hggcMemcpy(
        workspace_ptr + layout.prefix_offset, prefix.data(),
        layout.prefix_bytes, hggcMemcpyHostToDevice);
    if (copy_status != hggcSuccess) {
      return Status::kErrorInternal;
    }
    copy_status = hggcMemcpy(
        workspace_ptr + layout.shape_offset,
        args.problem_shape.host_problem_shapes, layout.shape_bytes,
        hggcMemcpyHostToDevice);
    if (copy_status != hggcSuccess) {
      return Status::kErrorInternal;
    }

    auto rep = representative_problem_shape(args, g);
    void* epilogue_workspace = workspace_ptr + layout.epilogue_offset;
    return CollectiveEpilogue::initialize_workspace(
        rep, args.epilogue, epilogue_workspace, stream);
  }

  // Gate-only timed reset. It consumes the frozen Params produced by the
  // successful install and rejects a different workspace base; mutable host
  // shapes, scheduler arguments, and occupancy therefore cannot silently
  // reinterpret an older P/mirror allocation. It clears only the frozen
  // barrier tail, matching the vendor initializer: peer 0 overwrites partial
  // accumulators, so clearing that scratch would perturb the cache state and
  // add non-production wall work. No P/shape or epilogue setup occurs here.
  // GemmUniversalAdapter::update() is intentionally unsupported by this
  // isolated gate because update does not reinstall P/shape.
  static cutlass::Status reset_scheduler_workspace_after_prefix_install(
      Params const& params, void* workspace, hggcStream_t stream = nullptr) {
    if (workspace == nullptr || workspace != params.workspace_base ||
        params.output_tile_prefix == nullptr ||
        params.host_shape_mirror == nullptr) {
      return Status::kErrorInvalidProblem;
    }
    if (params.scheduler.reduction_workspace_ != params.workspace_base ||
        params.scheduler_barrier_bytes > params.scheduler_workspace_bytes ||
        params.scheduler_barrier_offset !=
            params.scheduler_workspace_bytes - params.scheduler_barrier_bytes ||
        (params.scheduler.sk_units_ > params.scheduler.sk_tiles_ &&
         params.scheduler_barrier_bytes == 0)) {
      return Status::kErrorInvalidProblem;
    }
    if (params.scheduler_barrier_bytes == 0) {
      return Status::kSuccess;
    }
    hggcError_t const status = hggcMemsetAsync(
        static_cast<uint8_t*>(workspace) + params.scheduler_barrier_offset, 0,
        params.scheduler_barrier_bytes, stream);
    return status == hggcSuccess ? Status::kSuccess : Status::kErrorInternal;
  }

  static dim3 get_grid_shape(Params const& params) {
    TileSchedulerArguments args{};
    args.max_swizzle_size = 1;
    args.raster_order = TileScheduler::RasterOrderOptions::AlongN;
    args.splits = 1;
    args.reduction_mode = TileSchedulerParams::ReductionMode::Deterministic;
    args.decomposition_mode = TileSchedulerParams::DecompositionMode::StreamK;
    // Do not truncate by Q: Stream-K may use more workers than output tiles.
    return TileScheduler::get_grid_shape(
        params.scheduler, params.scheduler_problem_shape, TileShape{},
        ClusterShape{}, params.scheduler_hw_info, args);
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    SharedStorage& shared_storage =
        *reinterpret_cast<SharedStorage*>(smem_buf);
    int const thread_idx = int(threadIdx.x);
    auto const blk_shape = TileShape{};
    int const groups = params.problem_shape.groups();

    TileScheduler scheduler{params.scheduler};
    auto sched_work = scheduler.get_current_work();
    while (sched_work.is_valid()) {
      if (sched_work.is_reduction_unit()) {
        if (params.census.totals != nullptr && thread_idx == 0) {
          atomicAdd(params.census.totals + 2, 1u);
        }
        auto next = scheduler.fetch_next_work(sched_work);
        sched_work = get<0>(next);
        continue;
      }
      if (!TileScheduler::valid_warpgroup_in_work_tile(sched_work)) {
        auto next = scheduler.fetch_next_work(sched_work);
        sched_work = get<0>(next);
        continue;
      }

      // Synthetic geometry is (Q*TM,TN,K,1), max_swizzle=1, AlongN, so
      // scheduler M is the globally unique output tile and lock id.
      int const q = sched_work.M_idx;
      bool const q_valid = q >= 0 && q < params.output_tiles;
      if (!q_valid) {
        if (params.census.totals != nullptr && thread_idx == 0) {
          atomicAdd(params.census.totals + 4, 1u);
        }
        auto next = scheduler.fetch_next_work(sched_work);
        sched_work = get<0>(next);
        continue;
      }

      // upper_bound(P,q)-1, written against P[e+1], is required because empty
      // experts create repeated prefix entries.
      int lo = 0;
      int hi = groups;
      while (lo < hi) {
        int const mid = (lo + hi) >> 1;
        if (params.output_tile_prefix[mid + 1] <= q) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      int const expert = lo;
      if (expert < 0 || expert >= groups) {
        if (params.census.totals != nullptr && thread_idx == 0) {
          atomicAdd(params.census.totals + 4, 1u);
        }
        auto next = scheduler.fetch_next_work(sched_work);
        sched_work = get<0>(next);
        continue;
      }
      auto expert_shape = params.host_shape_mirror[expert];
      int const M = int(get<0>(expert_shape));
      int const N = int(get<1>(expert_shape));
      int const K = int(get<2>(expert_shape));
      int const mt = int(ceil_div(M, int(size<0>(blk_shape))));
      int const local = q - params.output_tile_prefix[expert];
      int const m_idx = mt > 0 ? local % mt : 0;
      int const n_idx = mt > 0 ? local / mt : 0;
      if (mt <= 0) {
        if (params.census.totals != nullptr && thread_idx == 0) {
          atomicAdd(params.census.totals + 5, 1u);
        }
        auto next = scheduler.fetch_next_work(sched_work);
        sched_work = get<0>(next);
        continue;
      }

      auto const real_problem_shape = make_shape(M, N, K, groups);
      auto const real_blk_coord = make_coord(m_idx, n_idx, _, expert);

      CollectiveMainloop collective_mainloop;
      auto load_inputs = collective_mainloop.load_init(
          real_problem_shape, real_blk_coord, params.mainloop);
      Tensor gA = get<0>(load_inputs);
      Tensor gB = get<1>(load_inputs);
      auto residue_mnk = make_tuple(
          M - size<0>(gA) * m_idx,
          N - size<0>(gB) * n_idx,
          K - size<1>(gA) * size<2>(gA));

      TiledMma tiled_mma;
      Tensor accumulators = make_fragment_like<ElementCompute>(
          partition_fragment_C(tiled_mma, take<0, 2>(blk_shape)));
      clear(accumulators);

      uint32_t const k_tile_start =
          TileScheduler::get_work_k_tile_start(sched_work);
      uint32_t const k_tile_count = TileScheduler::get_work_k_tile_count(
          sched_work, params.scheduler_problem_shape, blk_shape);
      auto k_tile_iter = make_coord_iterator(
          idx2crd(k_tile_start, shape<2>(gA)), shape<2>(gA));
      collective_mainloop(params.mainloop, load_inputs, accumulators,
                          k_tile_iter, int(k_tile_count), thread_idx, smem_buf);

      bool const requires_fixup =
          TileScheduler::requires_fixup(params.scheduler, sched_work);
      bool const compute_epilogue =
          TileScheduler::compute_epilogue(sched_work, params.scheduler);
      if (params.census.totals != nullptr && thread_idx == 0) {
        if (uint32_t(q) < params.census.peer_capacity) {
          atomicAdd(params.census.peer_count + q, 1u);
        } else {
          atomicAdd(params.census.totals + 4, 1u);
        }
        if (requires_fixup) atomicAdd(params.census.totals + 0, 1u);
        if (compute_epilogue) atomicAdd(params.census.totals + 1, 1u);
        if (requires_fixup && compute_epilogue)
          atomicAdd(params.census.totals + 3, 1u);
        for (uint32_t kk = 0; kk < k_tile_count; ++kk) {
          uint64_t const visit = uint64_t(q) *
                                     uint64_t(params.k_tiles_per_output) +
                                 uint64_t(k_tile_start + kk);
          if (visit < params.census.k_visit_capacity) {
            atomicAdd(params.census.k_visit_count + visit, 1u);
          } else {
            atomicAdd(params.census.totals + 4, 1u);
          }
        }
      }

      // sched_work.M_idx remains q here.  Never substitute expert-local
      // coordinates: output_tile_index() derives the global lock from it.
      TileScheduler::fixup(params.scheduler, sched_work, accumulators,
                           NumMmaWarpGroups, 0);

      if (compute_epilogue) {
        CollectiveEpilogue epilogue{
            params.epilogue, shared_storage.tensors.epilogue};
        #pragma hggc dislicm
        {
          epilogue(real_problem_shape, blk_shape, real_blk_coord, accumulators,
                   tiled_mma, residue_mnk, thread_idx,
                   reinterpret_cast<char*>(&shared_storage.tensors.epilogue));
        }
      }

      auto next = scheduler.fetch_next_work(sched_work);
      sched_work = get<0>(next);
    }
  }
};

}  // namespace cutlass::gemm::kernel
