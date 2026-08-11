// Copyright (c) 2026, quactlize contributors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Dense Stream-K driver for quactlize's mixed-input PPU collective.  This is a
// named extension rather than a GemmUniversal specialization: the actlize
// Stream-K wrapper calls the old plain-A/B mainloop API, while mixed input must
// rebuild load_init (including scale/zero descriptors) for every work item.
#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include "cutlass/block_striped.h"
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
class StreamKMixedInputKernel {
public:
  using ProblemShape = ProblemShape_;
  static_assert(cute::rank(ProblemShape{}) == 3 || cute::rank(ProblemShape{}) == 4,
                "ProblemShape must be <M,N,K> or <M,N,K,L>");
  static_assert(cute::is_base_of_v<KernelAiuMultistageMixedInput,
                                   typename CollectiveMainloop_::DispatchPolicy::Schedule>,
                "StreamKMixedInputKernel requires a mixed-input mainloop");
  static_assert(!isGroupProblemShape_v<ProblemShape>,
                "StreamKMixedInputKernel is dense-only");

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

  using ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;
  using TileSchedulerTag = StreamKScheduler;
  static constexpr uint32_t MaxThreadsPerBlock = cute::size(TiledMma{});
  using TileScheduler = detail::PersistentTileSchedulerPPUStreamK<
      TileShape, ClusterShape, 8u, MaxThreadsPerBlock>;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumMmaWarpGroups = 1;
  static constexpr bool IsDenseStreamK = true;

  static_assert(MaxThreadsPerBlock == 64u || MaxThreadsPerBlock == 128u,
                "dense mixed-input Stream-K supports exactly 64- or 128-thread CTAs");
  static_assert(TileScheduler::FixupThreadCount == MaxThreadsPerBlock,
                "dense Stream-K fixup cohort must equal the exact CTA thread count");
  static_assert(cute::is_same_v<ElementAccumulator, ElementCompute>,
                "Stream-K scratch and the live accumulator fragment must have one type");
  static_assert(cute::is_same_v<ElementAccumulator, float>,
                "the gate-only pre-fixup capture ABI is an FP32 scalar stream");
  // The scheduler guarantees no fewer than eight K tiles per Stream-K unit.
  // The mixed pipeline has a Stages-1 startup schedule (short slices repeat a
  // final valid prefetch rather than reading past the slice).  Keep this first
  // wiring inside the reviewed <=8 startup envelope; deeper-stage partial work
  // belongs to a later tactic sweep, not this fixed s2 mechanism gate.
  static_assert(DispatchPolicy::Stages - 1 <= 8,
                "dense mixed-input Stream-K first wiring is limited to the reviewed <=8 startup envelope");

  // Keep the kernel-parameter ABI at one nullable diagnostic pointer.  The
  // previous gate used one uint32_t* witness pointer; growing Arguments/Params
  // by individual capture fields would perturb that layout.  This POD lives
  // in device memory only for correctness gates.  Null performs no diagnostic global
  // access; adding the dormant branch still changes codegen, so byte identity
  // with an older binary is deliberately not claimed.
  static constexpr uint32_t PreFixupCaptureMagic = 0x534b4650u; // "SKFP"
  struct DiagnosticState {
    uint32_t witness[3]{};
    uint32_t pre_fixup_capture_magic = 0;
    ElementAccumulator* pre_fixup_capture = nullptr;
    int32_t const* pre_fixup_capture_slot_map = nullptr;
    uint32_t* pre_fixup_capture_slot_visits = nullptr;
    uint32_t* pre_fixup_capture_slot_k_counts = nullptr;
    uint64_t pre_fixup_capture_slot_capacity = 0;
    uint64_t pre_fixup_capture_map_capacity = 0;
    uint32_t pre_fixup_capture_k_stride = 0;
    uint32_t pre_fixup_capture_error_count = 0;
  };
  static_assert(std::is_trivially_copyable_v<DiagnosticState> &&
                alignof(DiagnosticState) >= alignof(void*),
                "Stream-K diagnostic POD must cross the host/device ABI unchanged");

  // Tile i executes mainloop -> fixup/epilogue before tile i+1 starts.  The
  // union is valid only while nobody overlaps tile i's epilogue with the next
  // tile's mainloop; such an overlap would require the sum of both allocations.
  struct SharedStorage {
    union SharedTensorStorage {
      using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
      using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
      MainloopSharedStorage mainloop;
      EpilogueSharedStorage epilogue;
    } tensors;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};       // real device/CU count
    TileSchedulerArguments scheduler{};
    int ctas_per_cu = 0;                // occupancy of this exact kernel
    // Optional gate POD; null on every performance/shipping run. witness[] is
    // [requires_fixup, final-epilogue, separate-reduction].  Capture metadata
    // is active only when its magic is exact.
    DiagnosticState* diagnostics = nullptr;
  };

  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    KernelHardwareInfo real_hw_info{};
    KernelHardwareInfo scheduler_hw_info{}; // cu_count == physical workers
    TileSchedulerParams scheduler{};
    int ctas_per_cu = 0;
    DiagnosticState* diagnostics = nullptr;
  };

private:
  static ProblemShape scheduler_problem_shape(ProblemShape const& input) {
    auto shape = input;
    if constexpr (detail::Has_SwapAB_v<CollectiveMainloop>) {
      cute::get<0>(shape) = cute::get<1>(input);
      cute::get<1>(shape) = cute::get<0>(input);
    }
    return shape;
  }

  static KernelHardwareInfo real_hw_info(Arguments const& args) {
    int cu_count = args.hw_info.cu_count;
    if (cu_count <= 0) {
      cu_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }
    return KernelHardwareInfo{args.hw_info.device_id, cu_count};
  }

  static KernelHardwareInfo scheduler_hw_info(Arguments const& args) {
    KernelHardwareInfo real = real_hw_info(args);
    int64_t const workers = int64_t(real.cu_count) * int64_t(args.ctas_per_cu);
    int const bounded = workers > 0 && workers <= std::numeric_limits<int>::max()
        ? static_cast<int>(workers) : 0;
    return KernelHardwareInfo{real.device_id, bounded};
  }

  static size_t scheduler_workspace_size(Arguments const& args) {
    if (args.ctas_per_cu <= 0) {
      return 0;
    }
    auto shape = scheduler_problem_shape(args.problem_shape);
    return TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
        args.scheduler, shape, scheduler_hw_info(args), NumMmaWarpGroups);
  }

public:
  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    auto shape = scheduler_problem_shape(args.problem_shape);
    auto shape_mnkl = cute::append<4>(shape, cute::Int<1>{});
    KernelHardwareInfo real = real_hw_info(args);
    KernelHardwareInfo workers = scheduler_hw_info(args);

    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t offset = scheduler_workspace_size(args);
    offset = round_nearest(offset, MinWorkspaceAlignment);
    void* epilogue_workspace = workspace_ptr ? workspace_ptr + offset : nullptr;

    // The scheduler API accepts ktile_start_alignment_count, but this actlize
    // revision does not forward it into Params::initialize().  Current formats
    // require alignment 1; a future offline format with a stronger declaration
    // must fail closed or repair that vendor seam before using this kernel.
    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(
        shape_mnkl, TileShape{}, ClusterShape{}, workers, args.scheduler,
        workspace, /*epilogue_subtile=*/1);

    return {
        args.mode,
        shape,
        CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, nullptr),
        CollectiveEpilogue::to_underlying_arguments(
            args.problem_shape, args.epilogue, epilogue_workspace),
        real,
        workers,
        scheduler,
        args.ctas_per_cu,
        args.diagnostics,
    };
  }

  static bool can_implement(Arguments const& args) {
    KernelHardwareInfo real = real_hw_info(args);
    KernelHardwareInfo workers = scheduler_hw_info(args);
    return args.mode == GemmUniversalMode::kGemm &&
           real.cu_count > 0 && workers.cu_count > 0 &&
           args.ctas_per_cu > 0 &&
           args.scheduler.splits == 1 &&
           args.scheduler.reduction_mode == TileSchedulerParams::ReductionMode::Deterministic &&
           args.scheduler.decomposition_mode == TileSchedulerParams::DecompositionMode::StreamK &&
           TileScheduler::can_implement(args.scheduler);
  }

  static size_t get_workspace_size(Arguments const& args) {
    size_t bytes = scheduler_workspace_size(args);
    bytes = round_nearest(bytes, MinWorkspaceAlignment);
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
    KernelHardwareInfo workers = scheduler_hw_info(args);
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t offset = 0;

    Status status = TileScheduler::template initialize_workspace<ProblemShape, ElementAccumulator>(
        args.scheduler, workspace_ptr, stream, shape, workers, NumMmaWarpGroups,
        /*epilogue_subtile=*/1, /*num_accumulator_mtxs=*/1, host_adapter);
    if (status != Status::kSuccess) {
      return status;
    }
    offset += scheduler_workspace_size(args);
    offset = round_nearest(offset, MinWorkspaceAlignment);

    void* epilogue_workspace = workspace_ptr ? workspace_ptr + offset : nullptr;
    return CollectiveEpilogue::initialize_workspace(
        args.problem_shape, args.epilogue, epilogue_workspace, stream);
  }

  static dim3 get_grid_shape(Params const& params) {
    TileSchedulerArguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.scheduler.log_swizzle_size_;
    }
    args.raster_order =
        params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN
        ? TileScheduler::RasterOrderOptions::AlongN
        : TileScheduler::RasterOrderOptions::AlongM;
    // Stream-K deliberately does not truncate this grid by output-tile count:
    // a small MN problem can use many physical workers by striping K.
    return TileScheduler::get_grid_shape(
        params.scheduler, params.problem_shape, TileShape{}, ClusterShape{},
        params.scheduler_hw_info, args);
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
      if (params.diagnostics && thread_idx == 0 && work_tile_info.is_reduction_unit()) {
        atomicAdd(params.diagnostics->witness + 2, 1u);
      }
      if (!TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
        auto next = scheduler.fetch_next_work(work_tile_info);
        work_tile_info = get<0>(next);
        continue;
      }

      int const m_coord = work_tile_info.M_idx;
      int const n_coord = work_tile_info.N_idx;
      int const l_coord = work_tile_info.L_idx;
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

      uint32_t const k_tile_start = TileScheduler::get_work_k_tile_start(work_tile_info);
      uint32_t const k_tile_count = TileScheduler::get_work_k_tile_count(
          work_tile_info, problem_shape_mnkl, blk_shape);
      auto k_tile_iter = make_coord_iterator(
          idx2crd(k_tile_start, shape<2>(gA)), shape<2>(gA));
      collective_mainloop(params.mainloop, load_inputs, accumulators,
                          k_tile_iter, int(k_tile_count), thread_idx, smem_buf);

      bool const requires_fixup = TileScheduler::requires_fixup(
          params.scheduler, work_tile_info);
      if (params.diagnostics && thread_idx == 0 && requires_fixup) {
        atomicAdd(params.diagnostics->witness + 0, 1u);
      }
      // Capture the raw mainloop partial before fixup changes the live
      // accumulator.  This seam is intentionally all-or-none and gate-only:
      // the default nullptr configuration performs no map/capture global
      // access.  Invalid q/K/map/slot metadata increments one error per work
      // item (thread 0 only), while every CTA thread takes the same skip path.
      DiagnosticState* diagnostics = params.diagnostics;
      if (requires_fixup && diagnostics != nullptr &&
          diagnostics->pre_fixup_capture_magic == PreFixupCaptureMagic) {
        using AccumulatorArray =
            Array<ElementAccumulator, size(decltype(accumulators){})>;
        using CaptureStriped =
            BlockStripedReduce<TileScheduler::FixupThreadCount, AccumulatorArray>;
        static_assert(CaptureStriped::kStripes == size(decltype(accumulators){}),
                      "pre-fixup capture must use scalar FP32 stripes");
        static_assert(
            uint64_t(TileScheduler::FixupThreadCount) *
                uint64_t(size(decltype(accumulators){})) ==
              uint64_t(size<0>(TileShape{})) * uint64_t(size<1>(TileShape{})),
            "pre-fixup capture slot must cover one exact accumulator tile");

        uint64_t const q = TileScheduler::output_tile_index(
            params.scheduler, work_tile_info);
        int32_t const k_idx = work_tile_info.K_idx;
        uint64_t map_idx = 0;
        bool valid = diagnostics->pre_fixup_capture != nullptr &&
                     diagnostics->pre_fixup_capture_slot_map != nullptr &&
                     diagnostics->pre_fixup_capture_slot_visits != nullptr &&
                     diagnostics->pre_fixup_capture_slot_k_counts != nullptr &&
                     diagnostics->pre_fixup_capture_k_stride > 0 && k_idx >= 0 &&
                     uint64_t(k_idx) <
                         uint64_t(diagnostics->pre_fixup_capture_k_stride);
        if (valid) {
          uint64_t const k = uint64_t(k_idx);
          uint64_t const stride =
              uint64_t(diagnostics->pre_fixup_capture_k_stride);
          valid = q <= (std::numeric_limits<uint64_t>::max() - k) / stride;
          if (valid) {
            map_idx = q * stride + k;
            valid = map_idx < diagnostics->pre_fixup_capture_map_capacity;
          }
        }

        int32_t slot = -1;
        if (valid) {
          slot = diagnostics->pre_fixup_capture_slot_map[map_idx];
          valid = slot >= 0 &&
                  uint64_t(slot) <
                      diagnostics->pre_fixup_capture_slot_capacity;
        }

        constexpr uint64_t kCaptureElementsPerSlot =
            uint64_t(size<0>(TileShape{})) * uint64_t(size<1>(TileShape{}));
        if (valid) {
          valid = uint64_t(slot) <=
              std::numeric_limits<uint64_t>::max() / kCaptureElementsPerSlot;
        }
        if (!valid) {
          if (thread_idx == 0) {
            atomicAdd(&diagnostics->pre_fixup_capture_error_count, 1u);
          }
        }
        else {
          if (thread_idx == 0) {
            diagnostics->pre_fixup_capture_slot_k_counts[slot] =
                uint32_t(k_tile_count);
            atomicAdd(diagnostics->pre_fixup_capture_slot_visits + slot, 1u);
          }
          ElementAccumulator* slot_base = diagnostics->pre_fixup_capture +
              uint64_t(slot) * kCaptureElementsPerSlot;
          auto* capture_array = reinterpret_cast<AccumulatorArray*>(slot_base);
          auto const* accumulator_array =
              reinterpret_cast<AccumulatorArray const*>(accumulators.data());
          CaptureStriped::store(capture_array, *accumulator_array, thread_idx);
        }
      }
      bool const full_output_tile =
          int(get<0>(residue_mnk)) >= int(size<0>(blk_shape)) &&
          int(get<1>(residue_mnk)) >= int(size<1>(blk_shape));
      if (!requires_fixup || full_output_tile) {
        // Preserve the vendor's old unpredicated fast path byte-for-byte.
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
        if (params.diagnostics && thread_idx == 0) {
          atomicAdd(params.diagnostics->witness + 1, 1u);
        }
        CollectiveEpilogue epilogue{params.epilogue, shared_storage.tensors.epilogue};
        #pragma hggc dislicm
        {
          epilogue(problem_shape_mnkl, blk_shape, blk_coord_mnkl, accumulators,
                   tiled_mma, residue_mnk, thread_idx,
                   reinterpret_cast<char*>(&shared_storage.tensors.epilogue));
        }
      }

      auto next = scheduler.fetch_next_work(work_tile_info);
      work_tile_info = get<0>(next);
    }
  }
};

} // namespace cutlass::gemm::kernel
