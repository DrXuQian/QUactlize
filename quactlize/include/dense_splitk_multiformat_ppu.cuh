/***************************************************************************************************
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Dense fixed Split-K support for the ordinary shipping mixed-input formats.
 *
 * This layer deliberately does not own a format collective.  ShippingTypes is the exact
 * fpa_intb_ppu::DenseKernelTypes instantiation used by S==1, including its low/high resident
 * layouts and its fp16-plane or packed-unit metadata channel.  S>1 changes only scheduling and
 * output: the same CollectiveMainloop writes compact FP32 planes and the existing fixed-order
 * reducer performs the shipping alpha=1, beta=0, FP16 conversion exactly once.
 *
 * "Fully quantized" in the current dense C ABI means packed GGUF scale/zero units.  Activation A
 * remains FP16.  Keep the assertions below explicit so a future quantized-A or richer epilogue
 * cannot enter this handle by typedef coincidence; such a path needs its own typed partial/final
 * epilogue ABI.
 **************************************************************************************************/
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "dense_splitk_parallel_ppu.cuh"
#include "actlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_fold.hpp"
#include "actlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_mixed_input_2plane.hpp"

namespace dense_splitk_parallel_ppu {

enum class MultiformatArgumentIssue {
  None,
  InvalidProblem,
  StaticGroupSizeMismatch,
  MissingActivation,
  MissingLowPlane,
  MissingMetadata,
  MissingDestination,
  MissingHighPlane,
  UnexpectedHighPlane,
  PackedMetadataHasSeparateZero,
  ScaleZeroMissingZero,
  ScaleOnlyHasZero,
  UnsupportedSplitCount,
  MissingPartialWorkspace,
  InsufficientPartialWorkspace,
};

constexpr char const* multiformat_argument_issue_name(
    MultiformatArgumentIssue issue) {
  switch (issue) {
    case MultiformatArgumentIssue::None: return "NONE";
    case MultiformatArgumentIssue::InvalidProblem: return "INVALID_PROBLEM";
    case MultiformatArgumentIssue::StaticGroupSizeMismatch:
      return "STATIC_GROUP_SIZE_MISMATCH";
    case MultiformatArgumentIssue::MissingActivation: return "MISSING_FP16_ACTIVATION";
    case MultiformatArgumentIssue::MissingLowPlane: return "MISSING_LOW_PLANE";
    case MultiformatArgumentIssue::MissingMetadata: return "MISSING_METADATA";
    case MultiformatArgumentIssue::MissingDestination: return "MISSING_FP16_DESTINATION";
    case MultiformatArgumentIssue::MissingHighPlane: return "MISSING_HIGH_PLANE";
    case MultiformatArgumentIssue::UnexpectedHighPlane: return "UNEXPECTED_HIGH_PLANE";
    case MultiformatArgumentIssue::PackedMetadataHasSeparateZero:
      return "PACKED_METADATA_HAS_SEPARATE_ZERO";
    case MultiformatArgumentIssue::ScaleZeroMissingZero: return "SCALE_ZERO_MISSING_ZERO";
    case MultiformatArgumentIssue::ScaleOnlyHasZero: return "SCALE_ONLY_HAS_ZERO";
    case MultiformatArgumentIssue::UnsupportedSplitCount: return "UNSUPPORTED_SPLIT_COUNT";
    case MultiformatArgumentIssue::MissingPartialWorkspace: return "MISSING_PARTIAL_WORKSPACE";
    case MultiformatArgumentIssue::InsufficientPartialWorkspace:
      return "INSUFFICIENT_PARTIAL_WORKSPACE";
  }
  return "UNKNOWN";
}

template <class Collective, class = void>
struct MainloopUsesPackedMetadata : std::false_type {};

template <class Collective>
struct MainloopUsesPackedMetadata<
    Collective, std::void_t<decltype(Collective::is_packed_scale)>>
    : std::bool_constant<Collective::is_packed_scale> {};

// Two-launch handle for ordinary A providers, folded one-plane providers and two-plane providers.
// PlaneB2 is explicit because omitting the high plane must be a typed admission failure, not a null
// pointer accepted by a broad CUTLASS Arguments aggregate.  ExpectPackedMetadata similarly binds the
// call site to the exact collective selected by the format-specific shipping build.
template <class ShippingTypes, class TileShape, class WarpShape,
          class PlaneB2 = void, bool ExpectPackedMetadata = false>
class PreparedMultiformatLauncher {
 public:
  using MainloopPolicy = typename ShippingTypes::MainloopPolicy;
  using ShippingKernel = typename ShippingTypes::GemmKernel;
  using ShippingGemm = typename ShippingTypes::Gemm;
  using SplitTypes = KernelTypes<ShippingTypes, TileShape, WarpShape>;
  using SplitKernel = typename SplitTypes::GemmKernel;
  using SplitGemm = typename SplitTypes::Gemm;
  using Reduction = typename SplitTypes::Reduction;
  using CollectiveMainloop = typename ShippingTypes::CollectiveMainloop;
  using ElementA = typename MainloopPolicy::ElementA;
  using ElementB = typename CollectiveMainloop::ElementB;
  using ElementScale = typename CollectiveMainloop::ElementScale;
  using ElementZero = typename CollectiveMainloop::ElementZero;
  using StrideScale = typename CollectiveMainloop::StrideScale;
  using MainloopArguments = typename CollectiveMainloop::Arguments;

  static constexpr bool UsesPackedMetadata =
      MainloopUsesPackedMetadata<CollectiveMainloop>::value;
  static constexpr bool HasHighPlane = !std::is_void_v<PlaneB2>;
  static constexpr bool HasSeparateZero =
      ppu_mixed_policy::has_zero(MainloopPolicy::Descriptor::quant_mode);

 private:
  ShippingGemm shipping_{};
  SplitGemm split_{};
  Reduction reduction_{};
  int splits_ = 0;
  bool initialized_ = false;

 public:
  static_assert(std::is_same_v<ElementA, cutlass::half_t>,
                "current dense fully-quantized shipping A is FP16; quantized A needs a new ABI");
  static_assert(std::is_same_v<typename ShippingTypes::ElementAccumulator, float>,
                "non-FP32 accumulators require a typed partial/reduction ABI");
  static_assert(std::is_same_v<typename ShippingTypes::ElementC, cutlass::half_t> &&
                    std::is_same_v<typename ShippingTypes::ElementD, cutlass::half_t>,
                "the shared reducer closes only the audited FP16 C/D shipping epilogue");
  static_assert(std::is_same_v<
                    typename ShippingTypes::EpilogueSchedule,
                    cutlass::epilogue::EpilogueSimtVectorizedWithoutEvt>,
                "a richer shipping epilogue needs an explicit post-reduction implementation");
  static_assert(UsesPackedMetadata == ExpectPackedMetadata,
                "packed-unit and fp16-plane metadata call sites must name the selected shipping collective");
  static_assert(MainloopPolicy::HighBits ==
                    ppu_mixed_policy::element_bits_v<PlaneB2>,
                "PlaneB2 must exactly match the shipping high-plane type");
  static_assert(std::is_same_v<typename SplitTypes::CollectiveMainloop,
                               CollectiveMainloop> &&
                    std::is_same_v<typename SplitKernel::CollectiveMainloop,
                                   CollectiveMainloop>,
                "S>1 must reuse the exact S==1 shipping collective");
  static_assert(std::is_same_v<typename SplitKernel::ElementAccumulator, float> &&
                    std::is_same_v<typename SplitKernel::ElementD, float>,
                "the audited multiformat partial ABI is compact FP32");
  static_assert(ShippingTypes::SharedStorageSize <= ppu_tactics::kBlockSmemBytes,
                "prepared S=1 shipping kernel must fit one compiled PPU block");
  static_assert(ppu_mixed_policy::kernel_policy_valid_v<
                    fpa_intb_ppu::TacticSpace, MainloopPolicy>,
                "multiformat Split-K must retain the shipping dense tactic guard");

  static MultiformatArgumentIssue inspect_arguments(
      cutlass::half_t const* A, ElementB const* B, void const* metadata,
      void const* zeros, cutlass::half_t* D, int m, int n, int k,
      int group_size, int split_k_slices, char* workspace,
      size_t workspace_bytes, PlaneB2 const* B2 = nullptr) {
    if (m <= 0 || n <= 0 || k <= 0 || group_size <= 0) {
      return MultiformatArgumentIssue::InvalidProblem;
    }
    if (!shipping_group_size_arguments_valid<ShippingTypes>(group_size)) {
      return MultiformatArgumentIssue::StaticGroupSizeMismatch;
    }
    if (A == nullptr) return MultiformatArgumentIssue::MissingActivation;
    if (B == nullptr) return MultiformatArgumentIssue::MissingLowPlane;
    if (metadata == nullptr) return MultiformatArgumentIssue::MissingMetadata;
    if (D == nullptr) return MultiformatArgumentIssue::MissingDestination;
    if constexpr (HasHighPlane) {
      if (B2 == nullptr) return MultiformatArgumentIssue::MissingHighPlane;
    } else {
      if (B2 != nullptr) return MultiformatArgumentIssue::UnexpectedHighPlane;
    }
    if constexpr (UsesPackedMetadata) {
      if (zeros != nullptr) {
        return MultiformatArgumentIssue::PackedMetadataHasSeparateZero;
      }
    } else if constexpr (HasSeparateZero) {
      if (zeros == nullptr) return MultiformatArgumentIssue::ScaleZeroMissingZero;
    } else {
      if (zeros != nullptr) return MultiformatArgumentIssue::ScaleOnlyHasZero;
    }
    if (!cutlass::gemm::kernel::fixed_splitk::supported_split_count(
            uint32_t(split_k_slices))) {
      return MultiformatArgumentIssue::UnsupportedSplitCount;
    }
    if (split_k_slices > 1) {
      WorkspacePlan plan;
      if (!query_workspace_plan(m, n, split_k_slices, plan)) {
        return MultiformatArgumentIssue::InvalidProblem;
      }
      if (workspace == nullptr) {
        return MultiformatArgumentIssue::MissingPartialWorkspace;
      }
      if (workspace_bytes < plan.partial_bytes) {
        return MultiformatArgumentIssue::InsufficientPartialWorkspace;
      }
    }
    return MultiformatArgumentIssue::None;
  }

  // Public for local ABI probes.  Production callers normally use initialize(); exposing the exact
  // aggregate lets a negative test change only dB2 and ask the real mainloop admission predicate.
  static MainloopArguments make_mainloop_arguments(
      cutlass::half_t const* A, ElementB const* B, void const* metadata,
      void const* zeros, int m, int n, int k, int group_size,
      PlaneB2 const* B2 = nullptr) {
    constexpr int LowFold = MainloopPolicy::ArtifactLowFold;
    using StrideA = typename CollectiveMainloop::StrideA;
    using StrideB = typename CollectiveMainloop::StrideB;
    StrideA sA = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(m, k, 1));
    StrideB sB = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape(n / LowFold, k * LowFold, 1));
    int const scale_k = (k + group_size - 1) / group_size;
    StrideScale sS = cutlass::make_cute_packed_stride(
        StrideScale{}, cute::make_shape(n, scale_k, 1));
    MainloopArguments args{
        A, sA, B, sB,
        reinterpret_cast<ElementScale const*>(metadata), sS, group_size,
        reinterpret_cast<ElementZero const*>(zeros)};
    if constexpr (HasHighPlane) {
      args.ptr_B2 = B2;
      constexpr int HighFold = MainloopPolicy::ArtifactHighFold;
      if constexpr (HighFold != LowFold) {
        args.dB2 = cutlass::make_cute_packed_stride(
            StrideB{}, cute::make_shape(n / HighFold, k * HighFold, 1));
        args.dB2_valid = true;
      }
    }
    return args;
  }

  bool initialize(
      cutlass::half_t const* A, ElementB const* B, void const* metadata,
      void const* zeros, cutlass::half_t* D, int m, int n, int k,
      int group_size, int split_k_slices, char* workspace,
      size_t workspace_bytes, hggcStream_t stream = nullptr,
      PlaneB2 const* B2 = nullptr) {
    initialized_ = false;
    splits_ = 0;
    if (inspect_arguments(A, B, metadata, zeros, D, m, n, k, group_size,
                          split_k_slices, workspace, workspace_bytes, B2) !=
        MultiformatArgumentIssue::None) {
      return false;
    }

    static_assert(MainloopPolicy::TacticTileK == int(cute::size<2>(TileShape{})),
                  "launch TileK must match the shipping mainloop policy");
    MainloopArguments mainloop = make_mainloop_arguments(
        A, B, metadata, zeros, m, n, k, group_size, B2);

    using ShippingStrideC = typename ShippingKernel::StrideC;
    using ShippingStrideD = typename ShippingKernel::StrideD;
    ShippingStrideC shipping_sC = cutlass::make_cute_packed_stride(
        ShippingStrideC{}, cute::make_shape(m, n, 1));
    ShippingStrideD shipping_sD = cutlass::make_cute_packed_stride(
        ShippingStrideD{}, cute::make_shape(m, n, 1));

    if (split_k_slices == 1) {
      using ElementAccumulator = typename ShippingTypes::ElementAccumulator;
      using ElementC = typename ShippingTypes::ElementC;
      typename ShippingGemm::Arguments args{
          cutlass::gemm::GemmUniversalMode::kGemm,
          {m, n, k, 1},
          mainloop,
          {{ElementAccumulator(1.f), ElementAccumulator(0.f)},
           static_cast<ElementC*>(nullptr), shipping_sC, D, shipping_sD},
          1};
      if (ShippingGemm::can_implement(args) != cutlass::Status::kSuccess ||
          ShippingGemm::get_workspace_size(args) != 0 ||
          shipping_.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) {
        return false;
      }
      splits_ = 1;
      initialized_ = true;
      return true;
    }

    if constexpr (SplitKernel::SharedStorageSize > ppu_tactics::kBlockSmemBytes) {
      // S==1 above remains the exact independently admitted shipping type.  A
      // partial epilogue may change the union maximum, so S>1 must check its
      // own compiled kernel size rather than inherit S==1's verdict.
      return false;
    }

    using PartialStride = typename SplitKernel::StrideD;
    PartialStride sP = cutlass::make_cute_packed_stride(
        PartialStride{}, cute::make_shape(m, n, split_k_slices));
    float* partials = reinterpret_cast<float*>(workspace);
    typename SplitGemm::Arguments main_args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        mainloop,
        {partials, sP, partials, sP},
        split_k_slices};
    typename Reduction::Arguments reduction_args{
        m, n, split_k_slices, partials, workspace_bytes, D, n};
    if (Reduction::can_implement(reduction_args) != cutlass::Status::kSuccess ||
        reduction_.initialize(reduction_args) != cutlass::Status::kSuccess ||
        SplitGemm::can_implement(main_args) != cutlass::Status::kSuccess ||
        SplitGemm::get_workspace_size(main_args) != 0 ||
        split_.initialize(main_args, nullptr, stream) != cutlass::Status::kSuccess) {
      return false;
    }
    splits_ = split_k_slices;
    initialized_ = true;
    return true;
  }

  cutlass::Status run(hggcStream_t stream = nullptr) {
    if (!initialized_) return cutlass::Status::kErrorInvalidProblem;
    if (splits_ == 1) return shipping_.run(stream);
    return cutlass::gemm::device::splitk_parallel::
        launch_main_then_reduce_same_stream(
            [&](hggcStream_t ordered_stream) { return split_.run(ordered_stream); },
            reduction_, stream);
  }

  cutlass::Status run_reducer_only_for_diagnostics(
      hggcStream_t stream = nullptr) {
    if (!initialized_ || splits_ <= 1) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    return reduction_.run(stream);
  }

  bool reduction_fast_path_selected_for_diagnostics() const {
    return initialized_ && splits_ > 1 &&
        reduction_.fast_path_selected_for_diagnostics();
  }
};

}  // namespace dense_splitk_parallel_ppu
