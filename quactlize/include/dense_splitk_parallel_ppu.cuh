/***************************************************************************************************
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Dense mixed-input fixed Split-K parallel type authority.
 *
 * This file is intentionally separate from fpA_intB_ppu.cuh.  The shipping S=1 launcher continues
 * to instantiate its historical GemmUniversal<..., SplitKSerialScheduler> type.  An explicit
 * S>1 caller opts into this header, which reuses that shipping type's mainloop verbatim and changes
 * only the output phase: FP32 partial planes followed by one ordered reduction kernel.
 **************************************************************************************************/
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "fpA_intB_ppu.cuh"

#include "cutlass/epilogue/collective/ppu_epilogue_vectorized_parallel.hpp"
#include "cutlass/epilogue/thread/conversion_op.h"
#include "cutlass/gemm/config/gemm_operands.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "quactlize_extensions/cutlass/gemm/device/ppu_mixed_input_splitk_parallel.hpp"
#include "quactlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_splitk_parallel.hpp"

namespace dense_splitk_parallel_ppu {

struct WorkspacePlan {
  size_t partial_bytes = 0;
  size_t alignment = 16;
};

// Optional attribution seam for a prepared launch.  Ranking must use run(),
// which records no midpoint event.  run_with_events() exists only to split the
// winning end-to-end span into producer and reducer components after the
// winner has already been selected.
struct SplitKParallelSpanEvents {
  hggcEvent_t producer_start{};
  hggcEvent_t producer_stop{};
  hggcEvent_t reducer_stop{};
  bool recorded = false;

  bool valid() const {
    return producer_start != nullptr && producer_stop != nullptr &&
           reducer_stop != nullptr;
  }
};

// Public allocation authority for the new path.  The output D allocation is
// deliberately separate and is not included in partial_bytes.
inline bool query_workspace_plan(
    int64_t rows, int64_t columns, int split_k_slices, WorkspacePlan& plan) {
  plan = WorkspacePlan{};
  if (split_k_slices == 1) {
    return rows > 0 && columns > 0;
  }
  return cutlass::gemm::device::splitk_parallel::fp32_workspace_size(
      rows, columns, split_k_slices, plan.partial_bytes);
}

// actlize's EpilogueParallel predates the CUTLASS-3 adapter metadata aliases.  Do not edit the
// shared vendor epilogue merely to satisfy this one new handle: the thin owned wrapper supplies
// only the two descriptive aliases and inherits the exact store implementation unchanged.
template <class Base>
struct AdapterVisiblePartialEpilogue : Base {
  using Base::Base;
  using GmemTiledCopyC = void;
  using GmemTiledCopyD = typename Base::CopyAtomR2G;
};

// Builds only the partial-producing output type.  ShippingTypes remains the one source of truth
// for the mixed-input mainloop, including B/S/Z/B2, fold, packed metadata, and packed-A choices.
template <class ShippingTypes, class TileShape, class WarpShape>
struct KernelTypes {
  using ShippingKernel = typename ShippingTypes::GemmKernel;
  using CollectiveMainloop = typename ShippingTypes::CollectiveMainloop;
  using TiledMma = typename CollectiveMainloop::TiledMma;
  using ElementAccumulator = typename ShippingTypes::ElementAccumulator;

  static_assert(std::is_same_v<ElementAccumulator, float>,
                "dense fixed Split-K v1 requires FP32 mainloop accumulators");
  static_assert(cute::is_same_v<TiledMma, typename ShippingKernel::TiledMma>,
                "parallel path must reuse the exact shipping TiledMma");

  static constexpr int BlockM = int(cute::size<0>(TileShape{}));
  static constexpr int BlockN = int(cute::size<1>(TileShape{}));
  static constexpr int WarpM = int(cute::size<0>(WarpShape{}));
  static constexpr int WarpN = int(cute::size<1>(WarpShape{}));
  static constexpr int ThreadCount = int(cute::size(TiledMma{}));
  static constexpr int FragmentSize = BlockM * BlockN / ThreadCount;
  static constexpr int PartialAlignment = 128 / cutlass::sizeof_bits<float>::value;

  static_assert(BlockM > 0 && BlockN > 0 && WarpM > 0 && WarpN > 0);
  static_assert(BlockM % WarpM == 0 && BlockN % WarpN == 0);
  static_assert(BlockM * BlockN % ThreadCount == 0,
                "each output thread must own an integral FP32 fragment");

  using WarpOnM = cute::Int<BlockM / WarpM>;
  using InstM = cute::conditional_t<BlockM == 8 && WarpM == 8,
                                     cute::Int<8>, cute::Int<16>>;
  using PartialStride = cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>;
  // EpilogueParallel invokes its thread op one scalar at a time.  The 128-bit store width belongs
  // to the copy atoms/configuration below; putting it in Convert::Count makes the scalar call ill
  // formed (and putting FragmentSize there manufactures uint_bit<256> at this M8 shape).
  using PartialOp = cutlass::epilogue::thread::AcConvert<
      float, 1, ElementAccumulator>;
  using EpilogueCopyInst = cute::AutoVectorizingCopyWithAssumedAlignment<128>;
  using EpilogueConfiguration = cutlass::gemm::config::DefaultGemm_Epilogue_Configuration<
      EpilogueCopyInst, ElementAccumulator, PartialAlignment,
      cute::Int<BlockM>, cute::Int<BlockN>, WarpOnM, ThreadCount, InstM>;
  using PartialEpilogueBase = cutlass::epilogue::collective::EpilogueParallel<
      PartialStride,
      PartialOp,
      typename EpilogueConfiguration::SmemLayoutO,
      cute::Copy_Atom<EpilogueCopyInst, ElementAccumulator>,
      typename EpilogueConfiguration::GmemTiledCopyO,
      cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, float>>;
  using CollectivePartialEpilogue = AdapterVisiblePartialEpilogue<PartialEpilogueBase>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversalMixedInputSplitKParallel<
      cute::Shape<int, int, int, int>, CollectiveMainloop,
      CollectivePartialEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // Eight FP32 values (32 B) per reduction thread.  The custom checked reducer handles the N tail
  // explicitly; unlike the legacy CUTLASS2 device handle, it cannot silently vector-load past N.
  static constexpr int ReductionElementsPerAccess = 8;
  using Reduction = cutlass::gemm::device::splitk_parallel::
      PpuMixedInputSplitKParallelReduction<ReductionElementsPerAccess>;

  static_assert(std::is_same_v<typename CollectivePartialEpilogue::ElementD, float>);
  static_assert(std::is_same_v<typename GemmKernel::CollectiveMainloop,
                               typename ShippingKernel::CollectiveMainloop>,
                "fixed Split-K must not rebuild or substitute the shipping mainloop");
};

// A reusable one-plane handle whose initialization is deliberately separate
// from device timing.  The old generic_launcher below remains the source-
// compatible one-shot API.  Performance sweeps use this handle so host-side
// argument lowering and initialize() cannot be charged to a 7--17 us device
// span.  S==1 still instantiates and runs ShippingTypes::Gemm verbatim; S>1
// instantiates the same mainloop behind the FP32-partial kernel.
template <class ShippingTypes, class TileShape, class WarpShape>
class PreparedOnePlaneLauncher {
 public:
  using MainloopPolicy = typename ShippingTypes::MainloopPolicy;
  using ShippingKernel = typename ShippingTypes::GemmKernel;
  using ShippingGemm = typename ShippingTypes::Gemm;
  using SplitTypes = KernelTypes<ShippingTypes, TileShape, WarpShape>;
  using SplitKernel = typename SplitTypes::GemmKernel;
  using SplitGemm = typename SplitTypes::Gemm;
  using Reduction = typename SplitTypes::Reduction;
  using ElementB = typename ShippingTypes::CollectiveMainloop::ElementB;
  using StrideScale = typename ShippingTypes::CollectiveMainloop::StrideScale;

 private:
  ShippingGemm shipping_{};
  SplitGemm split_{};
  Reduction reduction_{};
  int splits_ = 0;
  bool initialized_ = false;

  static bool record(hggcEvent_t event, hggcStream_t stream) {
    return event != nullptr && hggcEventRecord(event, stream) == hggcSuccess;
  }

 public:
  static_assert(MainloopPolicy::HighBits == 0,
                "the first prepared fixed Split-K handle is one-plane only");
  static_assert(MainloopPolicy::PackedARows == 1,
                "the first prepared fixed Split-K handle is the M1 packed-A provider");
  static_assert(ShippingTypes::SharedStorageSize <=
                    ppu_tactics::kBlockSmemBytes &&
                    SplitKernel::SharedStorageSize <=
                    ppu_tactics::kBlockSmemBytes,
                "prepared shipping/partial kernels must fit the compiled PPU smem limit");
  static_assert(ppu_mixed_policy::kernel_policy_valid_v<
                    fpa_intb_ppu::TacticSpace, MainloopPolicy>,
                "prepared fixed Split-K must retain the shipping dense tactic guard");
  static_assert(std::is_same_v<typename SplitTypes::CollectiveMainloop,
                               typename ShippingTypes::CollectiveMainloop>,
                "prepared S==1 and S>1 handles must share the exact mainloop");

  bool initialize(
      cutlass::half_t const* A, ElementB const* B,
      cutlass::half_t const* scales, cutlass::half_t const* zeros,
      cutlass::half_t* D, int m, int n, int k, int group_size,
      int split_k_slices, char* workspace, size_t workspace_bytes,
      hggcStream_t stream = nullptr) {
    initialized_ = false;
    splits_ = 0;
    if (m != MainloopPolicy::PackedARows || n <= 0 || k <= 0 || group_size <= 0 ||
        A == nullptr || B == nullptr || scales == nullptr || D == nullptr) {
      return false;
    }

    constexpr int LowFold = MainloopPolicy::ArtifactLowFold;
    static_assert(MainloopPolicy::TacticTileK == int(cute::size<2>(TileShape{})),
                  "prepared launch TileK must match the shipping policy");
    int const scale_k = (k + group_size - 1) / group_size;

    using ShippingStrideA = typename ShippingKernel::StrideA;
    using ShippingStrideB = typename ShippingKernel::StrideB;
    using ShippingStrideC = typename ShippingKernel::StrideC;
    using ShippingStrideD = typename ShippingKernel::StrideD;
    ShippingStrideA shipping_sA = cutlass::make_cute_packed_stride(
        ShippingStrideA{}, cute::make_shape(m, k, 1));
    ShippingStrideB shipping_sB = cutlass::make_cute_packed_stride(
        ShippingStrideB{}, cute::make_shape(n / LowFold, k * LowFold, 1));
    ShippingStrideC shipping_sC = cutlass::make_cute_packed_stride(
        ShippingStrideC{}, cute::make_shape(m, n, 1));
    ShippingStrideD shipping_sD = cutlass::make_cute_packed_stride(
        ShippingStrideD{}, cute::make_shape(m, n, 1));
    StrideScale sS = cutlass::make_cute_packed_stride(
        StrideScale{}, cute::make_shape(n, scale_k, 1));

    if (split_k_slices == 1) {
      using ElementAccumulator = typename ShippingTypes::ElementAccumulator;
      using ElementC = typename ShippingTypes::ElementC;
      typename ShippingGemm::Arguments args{
          cutlass::gemm::GemmUniversalMode::kGemm,
          {m, n, k, 1},
          {A, shipping_sA, B, shipping_sB, scales, sS, group_size, zeros},
          {{ElementAccumulator(1.f), ElementAccumulator(0.f)},
           static_cast<ElementC*>(nullptr), shipping_sC, D, shipping_sD},
          1};
      if (ShippingGemm::can_implement(args) != cutlass::Status::kSuccess ||
          ShippingGemm::get_workspace_size(args) != 0 ||
          shipping_.initialize(args, nullptr, stream) !=
              cutlass::Status::kSuccess) {
        return false;
      }
      splits_ = 1;
      initialized_ = true;
      return true;
    }

    WorkspacePlan workspace_plan;
    if (!query_workspace_plan(m, n, split_k_slices, workspace_plan) ||
        workspace == nullptr || workspace_bytes < workspace_plan.partial_bytes) {
      return false;
    }

    using SplitStrideA = typename SplitKernel::StrideA;
    using SplitStrideB = typename SplitKernel::StrideB;
    using PartialStride = typename SplitKernel::StrideD;
    SplitStrideA split_sA = cutlass::make_cute_packed_stride(
        SplitStrideA{}, cute::make_shape(m, k, 1));
    SplitStrideB split_sB = cutlass::make_cute_packed_stride(
        SplitStrideB{}, cute::make_shape(n / LowFold, k * LowFold, 1));
    PartialStride sP = cutlass::make_cute_packed_stride(
        PartialStride{}, cute::make_shape(m, n, split_k_slices));
    float* partials = reinterpret_cast<float*>(workspace);
    typename SplitGemm::Arguments main_args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {A, split_sA, B, split_sB, scales, sS, group_size, zeros},
        {partials, sP, partials, sP},
        split_k_slices};
    typename Reduction::Arguments reduction_args{
        m, n, split_k_slices, partials, workspace_bytes, D, n};
    if (Reduction::can_implement(reduction_args) != cutlass::Status::kSuccess ||
        reduction_.initialize(reduction_args) != cutlass::Status::kSuccess ||
        SplitGemm::can_implement(main_args) != cutlass::Status::kSuccess ||
        SplitGemm::get_workspace_size(main_args) != 0 ||
        split_.initialize(main_args, nullptr, stream) !=
            cutlass::Status::kSuccess) {
      return false;
    }
    splits_ = split_k_slices;
    initialized_ = true;
    return true;
  }

  cutlass::Status run(hggcStream_t stream = nullptr) {
    if (!initialized_) return cutlass::Status::kErrorInvalidProblem;
    if (splits_ == 1) return shipping_.run(stream);
    cutlass::Status const main_status = split_.run(stream);
    if (main_status != cutlass::Status::kSuccess) return main_status;
    return reduction_.run(stream);
  }

  // Diagnostic seam used to compare the partial-producing phase with an
  // externally reshaped GEMM under one timing protocol.  Production callers
  // must keep using run(): deliberately exposing the producer here must not
  // turn a missing reduction into a plausible final result.
  cutlass::Status run_producer_only_for_diagnostics(
      hggcStream_t stream = nullptr) {
    if (!initialized_) return cutlass::Status::kErrorInvalidProblem;
    return splits_ == 1 ? shipping_.run(stream) : split_.run(stream);
  }

  cutlass::Status run_with_events(
      SplitKParallelSpanEvents& events, hggcStream_t stream = nullptr) {
    events.recorded = false;
    if (!initialized_ || !events.valid() ||
        !record(events.producer_start, stream)) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    cutlass::Status status = splits_ == 1
        ? shipping_.run(stream)
        : split_.run(stream);
    if (status != cutlass::Status::kSuccess ||
        !record(events.producer_stop, stream)) {
      return status == cutlass::Status::kSuccess
          ? cutlass::Status::kErrorInternal : status;
    }
    if (splits_ > 1) {
      status = reduction_.run(stream);
      if (status != cutlass::Status::kSuccess) return status;
    }
    if (!record(events.reducer_stop, stream)) {
      return cutlass::Status::kErrorInternal;
    }
    events.recorded = true;
    return cutlass::Status::kSuccess;
  }
};

// Explicit two-launch entry point.  Runtime S==1 is delegated to the historical shipping launcher
// verbatim; only S>1 constructs the new producer.  This makes the comparison arm an executable
// type identity rather than a promise that a rewritten S==1 branch is "equivalent".
template <fpa_intb_ppu::QuantMode QuantOp, class BaseSchedule,
          class TileShape, class ScaleTileShape, class WarpShape, int Stages,
          bool AiuInterleaved, class ElementB = cutlass::int4b_t,
          class PlaneB2 = void, int ArtifactTileK = 0,
          class ShippingTypesOverride = void>
bool generic_launcher(
    cutlass::half_t const* A, ElementB const* B,
    cutlass::half_t const* scales, cutlass::half_t const* zeros,
    cutlass::half_t* D, int m, int n, int k, int group_size,
    int split_k_slices, char* workspace, size_t workspace_bytes,
    hggcStream_t stream, PlaneB2 const* B2 = nullptr) {
  using DefaultShippingTypes = fpa_intb_ppu::DenseKernelTypes<
      QuantOp, BaseSchedule, TileShape, ScaleTileShape, WarpShape, Stages,
      AiuInterleaved, ElementB, PlaneB2, ArtifactTileK>;
  using ShippingTypes = std::conditional_t<
      std::is_void_v<ShippingTypesOverride>, DefaultShippingTypes,
      ShippingTypesOverride>;

  if (split_k_slices == 1) {
    return fpa_intb_ppu::generic_launcher<
        QuantOp, BaseSchedule, TileShape, ScaleTileShape, WarpShape, Stages,
        AiuInterleaved, ElementB, PlaneB2,
        false, false, false, ArtifactTileK, ShippingTypesOverride>(
            A, B, scales, zeros, D, m, n, k, group_size, 1,
            workspace, workspace_bytes, stream, B2);
  }
  // Preserve the historical S==1 authority above, but reject an invalid
  // divisor before the new path constructs its metadata shape.
  if (group_size <= 0) {
    return false;
  }
  WorkspacePlan workspace_plan;
  if (!query_workspace_plan(m, n, split_k_slices, workspace_plan) ||
      workspace == nullptr || workspace_bytes < workspace_plan.partial_bytes) {
    return false;
  }

  using MainloopPolicy = typename ShippingTypes::MainloopPolicy;
  using SplitTypes = KernelTypes<ShippingTypes, TileShape, WarpShape>;
  using Gemm = typename SplitTypes::Gemm;
  using GemmKernel = typename SplitTypes::GemmKernel;
  using CollectiveMainloop = typename SplitTypes::CollectiveMainloop;
  using Reduction = typename SplitTypes::Reduction;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using PartialStride = typename GemmKernel::StrideD;
  using StrideScale = typename CollectiveMainloop::StrideScale;

  static_assert(std::is_same_v<typename ShippingTypes::CollectiveMainloop,
                               CollectiveMainloop>,
                "parallel producer must consume the exact shipping mainloop");
  static_assert(MainloopPolicy::TacticTileK == int(cute::size<2>(TileShape{})),
                "the launch TileK must match the shipping mainloop policy");

  constexpr int LowFold = MainloopPolicy::ArtifactLowFold;
  int const scale_k = (k + group_size - 1) / group_size;
  StrideA sA = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(m, k, 1));
  StrideB sB = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape(n / LowFold, k * LowFold, 1));
  StrideScale sS = cutlass::make_cute_packed_stride(
      StrideScale{}, cute::make_shape(n, scale_k, 1));
  PartialStride sP = cutlass::make_cute_packed_stride(
      PartialStride{}, cute::make_shape(m, n, split_k_slices));
  float* partials = reinterpret_cast<float*>(workspace);

  typename Gemm::Arguments main_args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {A, sA, B, sB, scales, sS, group_size, zeros},
      {partials, sP, partials, sP},
      split_k_slices};
  if constexpr (!std::is_void_v<PlaneB2>) {
    main_args.mainloop.ptr_B2 = B2;
    constexpr int HighFold = MainloopPolicy::ArtifactHighFold;
    if constexpr (HighFold != LowFold) {
      main_args.mainloop.dB2 = cutlass::make_cute_packed_stride(
          StrideB{}, cute::make_shape(n / HighFold, k * HighFold, 1));
      main_args.mainloop.dB2_valid = true;
    }
  }

  typename Reduction::Arguments reduction_args{
      m, n, split_k_slices, partials, workspace_bytes, D, n};
  Reduction reduction;
  if (reduction.initialize(reduction_args) != cutlass::Status::kSuccess) {
    return false;
  }

  Gemm gemm;
  if (Gemm::can_implement(main_args) != cutlass::Status::kSuccess ||
      Gemm::get_workspace_size(main_args) != 0 ||
      gemm.initialize(main_args, nullptr, stream) != cutlass::Status::kSuccess) {
    return false;
  }
  return cutlass::gemm::device::splitk_parallel::
             launch_main_then_reduce_same_stream(
                 [&](hggcStream_t launch_stream) {
                   return gemm.run(launch_stream);
                 },
                 reduction, stream) == cutlass::Status::kSuccess;
}

}  // namespace dense_splitk_parallel_ppu
