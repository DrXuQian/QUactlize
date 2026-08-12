// Copyright (c) 2026, quactlize contributors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Dense mixed-input driver for Marlin's CTA stripe scheduler.  The scheduler
// and cooperative are independent of Stream-K: they own Marlin's K-fast
// flattened decomposition, scheduler-local launch protection, reverse peer
// order, FP32 partial workspace, and per-output-tile lock lifecycle.  The
// mixed-input collective remains a template parameter, so converter, metadata,
// folding, B-chunk and two-plane semantics are exactly the existing ones.
#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/utils.h"
#include "cute/tensor.hpp"
#include "quactlize_extensions/cutlass/gemm/kernel/ppu_accumulator_residue_mask.hpp"

namespace cutlass::gemm::kernel {

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_>
class MarlinMixedInputKernel {
public:
  using ProblemShape = ProblemShape_;
  static_assert(cute::rank(ProblemShape{}) == 3 || cute::rank(ProblemShape{}) == 4,
                "ProblemShape must be <M,N,K> or <M,N,K,L>");
  static_assert(cute::is_base_of_v<KernelAiuMultistageMixedInput,
                                   typename CollectiveMainloop_::DispatchPolicy::Schedule>,
                "MarlinMixedInputKernel requires a mixed-input mainloop");
  static_assert(!isGroupProblemShape_v<ProblemShape>,
                "the first Marlin scheduler wiring is dense-only");

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

  // A K-tiled TiledMma gives every K cohort an isomorphic FP32 fragment for
  // the same output tile.  Only the K=0 cohort survives the CTA-local
  // reduction and participates in the cross-CTA cooperative/epilogue.  Derive
  // that exact cohort from the output tile and fragment instead of assuming
  // either the launch size or a historical 128-thread default.
  using AccumulatorFragment = decltype(cute::make_fragment_like<ElementCompute>(
      cute::partition_fragment_C(TiledMma{}, cute::take<0, 2>(TileShape{}))));
  static constexpr uint32_t OutputTileElements =
      uint32_t(cute::size<0>(TileShape{})) * uint32_t(cute::size<1>(TileShape{}));
  static constexpr uint32_t FragmentElements = uint32_t(cute::size(AccumulatorFragment{}));
  static constexpr uint32_t WarpKCohorts =
      uint32_t(cute::size<3>(typename TiledMma::ThrLayoutVMNK{}));
  static_assert(FragmentElements > 0 && OutputTileElements % FragmentElements == 0,
                "Marlin output fragment must divide one output tile exactly");
  static constexpr uint32_t OutputThreads = OutputTileElements / FragmentElements;
  static constexpr uint32_t ReductionScratchElements =
      (WarpKCohorts - 1) * OutputTileElements;

  using ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;
  using TileSchedulerTag = MarlinScheduler;
  static constexpr uint32_t MaxThreadsPerBlock = cute::size(TiledMma{});
  using TileScheduler = detail::PersistentTileSchedulerPPUMarlin<
      TileShape, ClusterShape, OutputThreads>;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumMmaWarpGroups = 1;
  static constexpr bool IsDenseMarlin = true;

  static_assert(MaxThreadsPerBlock == uint32_t(cute::size(TiledMma{})),
                "Marlin launch size must equal the exact TiledMma CTA size");
  static_assert(MaxThreadsPerBlock == OutputThreads * WarpKCohorts,
                "Marlin CTA must be an exact output-cohort x warp-K product");
  static_assert(TileScheduler::fixup_thread_count_capable(OutputThreads),
                "dense mixed-input Marlin requires a structurally capable output cohort");
  static_assert(TileScheduler::FixupThreadCount == OutputThreads,
                "Marlin cooperative cohort must equal the exact output cohort");
  static_assert(uint32_t(typename CollectiveEpilogue::TiledCopyS2R::TiledNumThr{}) ==
                    OutputThreads,
                "Marlin epilogue named-barrier cohort must equal the K0 output cohort");
  static_assert(cute::is_same_v<ElementAccumulator, ElementCompute> &&
                cute::is_same_v<ElementAccumulator, float>,
                "the first Marlin cooperative stores exact FP32 accumulator tiles");

  // Like the existing persistent and Stream-K drivers, tiles execute
  // mainloop -> cooperative/epilogue serially.  C-chain is deliberately not
  // used: CUTLASS C is a const beta source and D may have an arbitrary
  // epilogue/layout.  FP32 scratch preserves the existing C/D/beta ABI.
  struct SharedStorage {
    struct ReductionScratchStorage {
      // C++ does not admit zero-length arrays.  WK=1 never names this member;
      // retaining one inert scalar keeps that specialization well-formed and
      // does not enlarge the existing mainloop/epilogue union.
      ElementAccumulator values[ReductionScratchElements == 0
                                    ? 1
                                    : ReductionScratchElements];
    };
    union SharedTensorStorage {
      using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
      using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
      MainloopSharedStorage mainloop;
      EpilogueSharedStorage epilogue;
      ReductionScratchStorage reduction;
    } tensors;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
  };

  CUTLASS_HOST_DEVICE static constexpr ProblemShape scheduler_problem_shape(
      ProblemShape const& input) {
    auto shape = input;
    if constexpr (detail::Has_SwapAB_v<CollectiveMainloop>) {
      cute::get<0>(shape) = cute::get<1>(input);
      cute::get<1>(shape) = cute::get<0>(input);
    }
    return shape;
  }

  CUTLASS_HOST_DEVICE static constexpr auto scheduler_output_tile_coord(
      typename TileScheduler::WorkTileInfo const& work) {
    return cute::make_coord(work.M_idx, work.N_idx, work.L_idx);
  }

  template <class KTileShape>
  CUTLASS_HOST_DEVICE static constexpr auto scheduler_k_tile_coord(
      typename TileScheduler::WorkTileInfo const& work,
      KTileShape const& shape) {
    return TileScheduler::get_work_k_tile_coord(work, shape);
  }

private:

  static KernelHardwareInfo real_hw_info(Arguments const& args) {
    int cu_count = args.hw_info.cu_count;
    if (cu_count <= 0) {
      cu_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }
    return KernelHardwareInfo{args.hw_info.device_id, cu_count};
  }

  static size_t scheduler_workspace_size(Arguments const& args) {
    auto shape = scheduler_problem_shape(args.problem_shape);
    return TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
        args.scheduler, shape, real_hw_info(args), NumMmaWarpGroups);
  }

public:
  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    auto shape = scheduler_problem_shape(args.problem_shape);
    KernelHardwareInfo hw = real_hw_info(args);
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t offset = round_nearest(scheduler_workspace_size(args), MinWorkspaceAlignment);
    void* epilogue_workspace = workspace_ptr ? workspace_ptr + offset : nullptr;
    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(
        shape, TileShape{}, ClusterShape{}, hw, args.scheduler, workspace,
        /*epilogue_subtile=*/1, /*ktile alignment=*/1);

    return {
        args.mode,
        shape,
        CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, nullptr),
        CollectiveEpilogue::to_underlying_arguments(
            args.problem_shape, args.epilogue, epilogue_workspace),
        hw,
        scheduler,
    };
  }

  static bool can_implement(Arguments const& args) {
    auto shape = scheduler_problem_shape(args.problem_shape);
    KernelHardwareInfo hw = real_hw_info(args);
    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(
        shape, TileShape{}, ClusterShape{}, hw, args.scheduler,
        /*workspace=*/nullptr, /*epilogue_subtile=*/1,
        /*ktile alignment=*/1);
    return args.mode == GemmUniversalMode::kGemm && hw.cu_count > 0 &&
           CollectiveMainloop::can_implement(args.problem_shape, args.mainloop) &&
           scheduler.valid_ && TileScheduler::can_implement(args.scheduler);
  }

  static size_t get_workspace_size(Arguments const& args) {
    size_t bytes = round_nearest(scheduler_workspace_size(args), MinWorkspaceAlignment);
    bytes += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    return round_nearest(bytes, MinWorkspaceAlignment);
  }

  static cutlass::Status initialize_workspace(
      Arguments const& args, void* workspace = nullptr,
      hggcStream_t stream = nullptr, HostAdapter* host_adapter = nullptr) {
    if (!can_implement(args)) {
      return Status::kErrorInvalidProblem;
    }
    auto shape = scheduler_problem_shape(args.problem_shape);
    KernelHardwareInfo hw = real_hw_info(args);
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    Status status = TileScheduler::template initialize_workspace<ProblemShape, ElementAccumulator>(
        args.scheduler, workspace_ptr, stream, shape, hw, NumMmaWarpGroups,
        /*epilogue_subtile=*/1, /*num accumulators=*/1, host_adapter);
    if (status != Status::kSuccess) {
      return status;
    }
    size_t offset = round_nearest(scheduler_workspace_size(args), MinWorkspaceAlignment);
    void* epilogue_workspace = workspace_ptr ? workspace_ptr + offset : nullptr;
    return CollectiveEpilogue::initialize_workspace(
        args.problem_shape, args.epilogue, epilogue_workspace, stream);
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.scheduler);
  }
  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
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
      auto const output_tile_coord = scheduler_output_tile_coord(work_tile_info);
      int const m_coord = get<0>(output_tile_coord);
      int const n_coord = get<1>(output_tile_coord);
      int const l_coord = get<2>(output_tile_coord);
      auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);

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

      uint32_t const k_tile_count = TileScheduler::get_work_k_tile_count(
          work_tile_info, problem_shape_mnkl, blk_shape);
      auto k_tile_iter = make_coord_iterator(
          scheduler_k_tile_coord(work_tile_info, shape<2>(gA)), shape<2>(gA));
      collective_mainloop(params.mainloop, load_inputs, accumulators,
                          k_tile_iter, int(k_tile_count), thread_idx, smem_buf);

      if constexpr (WarpKCohorts == 1) {
        // Keep the shipping WK=1 path structurally unchanged.  In particular,
        // its cooperative cohort remains the full CTA and no extra barrier or
        // shared-memory instruction is emitted.
        bool const requires_fixup = TileScheduler::requires_fixup(
            params.scheduler, work_tile_info);
        bool const full_output_tile =
            int(get<0>(residue_mnk)) >= int(size<0>(blk_shape)) &&
            int(get<1>(residue_mnk)) >= int(size<1>(blk_shape));
        if (!requires_fixup || full_output_tile) {
          TileScheduler::fixup(
              params.scheduler, work_tile_info, accumulators, NumMmaWarpGroups, 0);
        }
        else {
          auto valid_accumulator = detail::make_accumulator_residue_mask(
              tiled_mma, accumulators, take<0, 2>(blk_shape),
              take<0, 2>(residue_mnk), thread_idx);
          TileScheduler::fixup(
              params.scheduler, work_tile_info, accumulators, NumMmaWarpGroups, 0,
              valid_accumulator);
        }

        if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler)) {
          CollectiveEpilogue epilogue{params.epilogue, shared_storage.tensors.epilogue};
          #pragma hggc dislicm
          {
            epilogue(problem_shape_mnkl, blk_shape, blk_coord_mnkl, accumulators,
                     tiled_mma, residue_mnk, thread_idx,
                     reinterpret_cast<char*>(&shared_storage.tensors.epilogue));
          }
        }
      }
      else {
        // Project the real VMNK thread coordinate onto K=0.  This is the
        // compact output-cohort ID proved by L139; unlike thread_idx % cohort,
        // it remains tied to the TiledMma layout if its physical ordering ever
        // changes.
        auto const thr_layout_vmnk = tiled_mma.get_thr_layout_vmnk();
        auto const vmnk = thr_layout_vmnk.get_flat_coord(thread_idx);
        int const warp_k = int(get<3>(vmnk));
        int const output_thread_idx = int(thr_layout_vmnk(make_coord(
            get<0>(vmnk), get<1>(vmnk), get<2>(vmnk), Int<0>{})));
        CUTLASS_ASSERT(0 <= warp_k && warp_k < int(WarpKCohorts));
        CUTLASS_ASSERT(0 <= output_thread_idx &&
                       output_thread_idx < int(OutputThreads));
        CUTLASS_ASSERT(warp_k != 0 || output_thread_idx == thread_idx);

        // The reduction scratch aliases mainloop and epilogue storage.  The
        // mainloop ends with a CTA barrier; the barrier after the stores below
        // is therefore the phase boundary that makes scratch visible before
        // any K0 load.  Avoiding a redundant leading barrier matters in this
        // latency-sensitive decode path.
        ElementAccumulator* reduction_scratch =
            shared_storage.tensors.reduction.values;
        if (warp_k != 0) {
          #pragma unroll
          for (uint32_t i = 0; i < FragmentElements; ++i) {
            uint32_t const stripe = i * OutputThreads +
                                    uint32_t(output_thread_idx);
            uint32_t const scratch_idx =
                (uint32_t(warp_k) - 1) * OutputTileElements + stripe;
            reduction_scratch[scratch_idx] = accumulators.data()[i];
          }
        }
        __syncthreads();

        if (warp_k == 0) {
          // Every K cohort has the same logical C fragment ordering for a
          // given compact output thread (L139).  Accumulate the other cohorts
          // in deterministic K order and keep the complete FP32 result in K0.
          #pragma unroll
          for (uint32_t i = 0; i < FragmentElements; ++i) {
            uint32_t const stripe = i * OutputThreads +
                                    uint32_t(output_thread_idx);
            #pragma unroll
            for (uint32_t peer_k = 1; peer_k < WarpKCohorts; ++peer_k) {
              accumulators.data()[i] += reduction_scratch[
                  (peer_k - 1) * OutputTileElements + stripe];
            }
          }
        }

        // All K0 loads must finish before the epilogue reuses the union.  Only
        // that compact cohort may enter the cross-CTA cooperative and output.
        __syncthreads();
        if (warp_k == 0) {
          bool const requires_fixup = TileScheduler::requires_fixup(
              params.scheduler, work_tile_info);
          bool const full_output_tile =
              int(get<0>(residue_mnk)) >= int(size<0>(blk_shape)) &&
              int(get<1>(residue_mnk)) >= int(size<1>(blk_shape));
          if (!requires_fixup || full_output_tile) {
            TileScheduler::fixup(
                params.scheduler, work_tile_info, accumulators, NumMmaWarpGroups, 0);
          }
          else {
            auto valid_accumulator = detail::make_accumulator_residue_mask(
                tiled_mma, accumulators, take<0, 2>(blk_shape),
                take<0, 2>(residue_mnk), thread_idx);
            TileScheduler::fixup(
                params.scheduler, work_tile_info, accumulators, NumMmaWarpGroups, 0,
                valid_accumulator);
          }

          if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler)) {
            CollectiveEpilogue epilogue{params.epilogue, shared_storage.tensors.epilogue};
            #pragma hggc dislicm
            {
              epilogue(problem_shape_mnkl, blk_shape, blk_coord_mnkl, accumulators,
                       tiled_mma, residue_mnk, thread_idx,
                       reinterpret_cast<char*>(&shared_storage.tensors.epilogue));
            }
          }
        }

        // No non-output K cohort may advance to the next persistent tile while
        // K0 still owns epilogue shared memory for the current tile.
        __syncthreads();
      }

      auto next = scheduler.fetch_next_work(work_tile_info);
      work_tile_info = get<0>(next);
    }
  }
};

} // namespace cutlass::gemm::kernel
