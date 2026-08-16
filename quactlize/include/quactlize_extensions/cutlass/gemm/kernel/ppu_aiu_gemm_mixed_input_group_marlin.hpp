// Copyright (c) 2026, quactlize contributors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Grouped mixed-input driver for Marlin's independent CTA stripe scheduler.
// Ragged expert output tiles are flattened through a host-built prefix into
// one global scheduler coordinate q.  The proven Marlin scheduler owns the
// K-fast stripe, default-B1 G=max(Q,CU) launch policy and cooperative; this wrapper
// only decodes q back to (expert,m,n).  sched_work itself must remain global-q
// when passed to fixup so locks cannot alias across experts.
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
#include "cutlass/gemm/kernel/ppu_tile_scheduler_marlin.hpp"
#include "cutlass/utils.h"
#include "cute/tensor.hpp"
#include "quactlize_extensions/cutlass/gemm/kernel/ppu_accumulator_residue_mask.hpp"
#include "quactlize_extensions/cutlass/gemm/kernel/ppu_grouped_ragged_geometry.hpp"

namespace cutlass::gemm::kernel {

template <class GroupProblemShape_, class CollectiveMainloop_,
          class CollectiveEpilogue_>
class GroupMarlinMixedInputKernel {
 public:
  using ProblemShape = GroupProblemShape_;
  using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;
  static_assert(isGroupProblemShape_v<ProblemShape>,
                "GroupMarlinMixedInputKernel requires GroupProblemShape");
  static_assert(cute::rank(UnderlyingProblemShape{}) == 3,
                "grouped Marlin v1 requires per-expert <M,N,K>");
  static_assert(std::is_trivially_copyable_v<UnderlyingProblemShape>,
                "grouped Marlin host shape mirror must be byte-copyable");

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
  using TileScheduler = detail::PersistentTileSchedulerPPUMarlin<
      TileShape, ClusterShape, MaxThreadsPerBlock>;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumMmaWarpGroups = 1;
  static constexpr int MinWorkspaceAlignment = 16;
  static constexpr bool IsGroupedMarlin = true;

  static_assert(cute::is_base_of_v<KernelAiuMultistageMixedInput,
                                   typename DispatchPolicy::Schedule>,
                "grouped Marlin requires the mixed-input mainloop");
  static_assert(!CollectiveMainloop::SwapAB,
                "grouped Marlin v1 requires q to remain on scheduler M");
  static_assert(MinWorkspaceAlignment % alignof(UnderlyingProblemShape) == 0,
                "workspace alignment must cover the host shape mirror");
  static_assert(TileScheduler::fixup_thread_count_capable(MaxThreadsPerBlock),
                "grouped Marlin requires a structurally capable CTA cohort");
  static_assert(TileScheduler::FixupThreadCount == MaxThreadsPerBlock,
                "grouped Marlin fixup cohort must equal the exact CTA thread count");
  static_assert(cute::is_same_v<ElementAccumulator, ElementCompute>,
                "Marlin scratch and live accumulators must have one type");

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
    TileSchedulerParams scheduler{};
    void* workspace_base = nullptr;
    size_t scheduler_workspace_bytes = 0;
    int const* output_tile_prefix = nullptr; // [groups+1], after scheduler ws
    UnderlyingProblemShape const* host_shape_mirror = nullptr; // [groups]
    int output_tiles = 0;
    int k_tiles_per_output = 0;
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
      uint64_t next_q = 0;
      if (!detail::GroupedRaggedOutputTiles::append_group(
              uint64_t(q), m, n, TM, TN, next_q) ||
          next_q > uint64_t(std::numeric_limits<int>::max())) {
        return out;
      }
      q = int64_t(next_q);
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

  static size_t scheduler_workspace_size(Arguments const& args,
                                         HostGeometry const& g) {
    if (!g.valid) {
      return 0;
    }
    auto shape = scheduler_problem_shape(g);
    return TileScheduler::template get_workspace_size<SchedulerProblemShape,
                                                       ElementAccumulator>(
        args.scheduler, shape, real_hw_info(args), NumMmaWarpGroups);
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
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);

    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(
        scheduler_shape, TileShape{}, ClusterShape{}, real, args.scheduler,
        workspace, /*epilogue_subtile=*/1);
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
        scheduler,
        workspace,
        layout.scheduler_bytes,
        workspace_ptr ? reinterpret_cast<int const*>(
                            workspace_ptr + layout.prefix_offset)
                      : nullptr,
        workspace_ptr ? reinterpret_cast<UnderlyingProblemShape const*>(
                            workspace_ptr + layout.shape_offset)
                      : nullptr,
        g.q,
        g.valid ? int(cute::ceil_div(g.k, int(cute::size<2>(TileShape{}))))
                : 0,
        args.census,
    };
  }

  static bool can_implement(Arguments const& args) {
    HostGeometry const g = host_geometry(args);
    KernelHardwareInfo real = real_hw_info(args);
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
           real.cu_count > 0 && args.domain_valid && census_ok &&
           // The blocks-per-CU experiment is intentionally dense-only.  A
           // grouped caller has no binding to its final kernel's
           // maximum_active_blocks(), so accepting B>1 here would silently
           // enable an occupancy-unbounded launch policy.
           args.scheduler.blocks_per_cu == 1 &&
           CollectiveMainloop::can_implement(
               representative_problem_shape(args, g), args.mainloop) &&
           args.mainloop.group_row_offsets != nullptr &&
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
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);

    Status status =
        TileScheduler::template initialize_workspace<SchedulerProblemShape,
                                                     ElementAccumulator>(
            args.scheduler, workspace_ptr, stream, shape, real_hw_info(args),
            NumMmaWarpGroups, /*epilogue_subtile=*/1,
            /*num_accumulator_mtxs=*/1, host_adapter);
    if (status != Status::kSuccess) {
      return status;
    }

    // P and the shape mirror are caller-owned host data (P itself is a local
    // vector), so both copies remain blocking and finish before timing.
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

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.scheduler);
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
      if (!TileScheduler::valid_warpgroup_in_work_tile(sched_work)) {
        auto next = scheduler.fetch_next_work(sched_work);
        sched_work = get<0>(next);
        continue;
      }

      // Synthetic geometry is (Q*TM,TN,K,1), so scheduler M is the globally
      // unique output tile and lock id.  No expert-local coordinate may
      // replace sched_work in the cooperative below.
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

      int const expert = detail::GroupedRaggedOutputTiles::decode_expert(
          params.output_tile_prefix, groups, q);
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
      int m_idx = -1;
      int n_idx = -1;
      if (!detail::GroupedRaggedOutputTiles::decode_local_mn(
              q, params.output_tile_prefix[expert], M,
              int(size<0>(blk_shape)), m_idx, n_idx)) {
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
      // gA/gB describe physical transfer shapes; output residue follows the logical CTA tile.
      auto residue_mnk = make_tuple(
          M - size<0>(blk_shape) * m_idx,
          N - size<1>(blk_shape) * n_idx,
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
      bool const full_output_tile =
          int(get<0>(residue_mnk)) >= int(size<0>(blk_shape)) &&
          int(get<1>(residue_mnk)) >= int(size<1>(blk_shape));
      if (!requires_fixup || full_output_tile) {
        TileScheduler::fixup(params.scheduler, sched_work, accumulators,
                             NumMmaWarpGroups, 0);
      }
      else {
        auto valid_accumulator = detail::make_accumulator_residue_mask(
            tiled_mma, accumulators, take<0, 2>(blk_shape),
            take<0, 2>(residue_mnk), thread_idx);
        TileScheduler::fixup(params.scheduler, sched_work, accumulators,
                             NumMmaWarpGroups, 0, valid_accumulator);
      }

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
