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
