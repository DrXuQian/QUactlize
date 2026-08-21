/***************************************************************************************************
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Dense mixed-input fixed Split-K parallel type authority.
 *
 * This file is intentionally separate from fpA_intB_ppu.cuh.  The shipping S=1 launcher continues
 * to instantiate its historical GemmUniversal<..., SplitKSerialScheduler> type.  An explicit
 * S>1 caller opts into this header, which reuses that shipping type's mainloop verbatim and changes
 * only the output phase: FP32 partial planes followed either by the permanent ordered-reducer
 * oracle or by an explicit actual-last completion policy.
 **************************************************************************************************/
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "fpA_intB_ppu.cuh"

#include "cutlass/epilogue/collective/ppu_epilogue_vectorized_parallel.hpp"
#include "cutlass/epilogue/thread/conversion_op.h"
#include "cutlass/gemm/config/gemm_operands.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "actlize_extensions/cutlass/gemm/device/ppu_mixed_input_splitk_parallel.hpp"
#include "actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_splitk_parallel.hpp"

namespace dense_splitk_parallel_ppu {

struct WorkspacePlan {
  size_t partial_bytes = 0;
  size_t counter_offset = 0;
  size_t counter_bytes = 0;
  size_t total_bytes = 0;
  uint64_t output_tiles = 0;
  size_t alignment = 16;
  size_t preferred_fast_alignment = 128;
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
  bool const ok = cutlass::gemm::device::splitk_parallel::fp32_workspace_size(
      rows, columns, split_k_slices, plan.partial_bytes);
  plan.total_bytes = plan.partial_bytes;
  return ok;
}

// The fused policy appends one completion counter per global output tile.
// Tile geometry is explicit because q belongs to the producer scheduler, not
// merely to the logical M*N output tensor.
inline bool query_fused_workspace_plan(
    int64_t rows, int64_t columns, int split_k_slices,
    int tile_m, int tile_n, WorkspacePlan& plan) {
  if (!query_workspace_plan(rows, columns, split_k_slices, plan) ||
      split_k_slices <= 1 || tile_m <= 0 || tile_n <= 0) {
    return false;
  }
  // Avoid the usual (extent + tile - 1) spelling here: the public query is a
  // fail-close ABI seam and must remain defined even at INT64_MAX.
  uint64_t const m_tiles = uint64_t(1 + (rows - 1) / tile_m);
  uint64_t const n_tiles = uint64_t(1 + (columns - 1) / tile_n);
  if (m_tiles == 0 || n_tiles == 0 ||
      m_tiles > (std::numeric_limits<uint64_t>::max)() / n_tiles) {
    return false;
  }
  auto const completion = cutlass::gemm::kernel::fixed_splitk::
      make_completion_workspace(plan.partial_bytes, m_tiles * n_tiles);
  if (!completion.is_valid()) {
    return false;
  }
  plan.counter_offset = completion.counter_offset;
  plan.counter_bytes = completion.counter_bytes;
  plan.total_bytes = completion.total_bytes;
  plan.output_tiles = completion.output_tiles;
  plan.alignment = cutlass::gemm::kernel::fixed_splitk::kCompletionAlignment;
  return true;
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

  using FusedCompletion = cutlass::gemm::kernel::fixed_splitk::
      LastArriverM1Fp16Completion<2>;
  using FusedGemmKernel = cutlass::gemm::kernel::
      GemmUniversalMixedInputSplitKParallel<
          cute::Shape<int, int, int, int>, CollectiveMainloop,
          CollectivePartialEpilogue, FusedCompletion>;
  using FusedGemm = cutlass::gemm::device::GemmUniversalAdapter<
      FusedGemmKernel>;

  // M=1 uses one 32-thread CTA per 64-column stripe and compile-time S=2/4/8
  // chains.  Wider M, tails, custom D strides and weaker alignment retain the
  // checked generic reducer as a fail-closed fallback inside this handle.
  static constexpr int ReductionElementsPerAccess = 2;
  using Reduction = cutlass::gemm::device::splitk_parallel::
      PpuMixedInputSplitKParallelM1FastReduction<
          ReductionElementsPerAccess>;

  static_assert(std::is_same_v<typename CollectivePartialEpilogue::ElementD, float>);
  static_assert(std::is_same_v<typename GemmKernel::CollectiveMainloop,
                               typename ShippingKernel::CollectiveMainloop>,
                "fixed Split-K must not rebuild or substitute the shipping mainloop");
  static_assert(std::is_same_v<typename FusedGemmKernel::CollectiveMainloop,
                               typename ShippingKernel::CollectiveMainloop>,
                "last-arriver completion must not rebuild the shipping mainloop");
};

// The one-plane metadata pointer is part of the format ABI, not an optional
// value that Split-K may reinterpret.  Bind the check to the real shipping
// collective's conversion mode so the S==1 delegation and S>1 producer accept
// exactly the same ScaleOnly/ScaleZero argument shape.  In particular, a
// plausible all-zero result must not let a missing ScaleZero plane pass.
template <class ShippingTypes>
inline bool one_plane_metadata_arguments_valid(
    void const* zeros) {
  using MainloopPolicy = typename ShippingTypes::MainloopPolicy;
  using CollectiveMainloop = typename ShippingTypes::CollectiveMainloop;
  static_assert(MainloopPolicy::HighBits == 0,
                "one-plane metadata admission cannot classify a B2 tuple");
  static_assert(
      bool(CollectiveMainloop::has_zero_channel) ==
          ppu_mixed_policy::has_zero(
              MainloopPolicy::Descriptor::quant_mode),
      "shipping collective and descriptor disagree on the zero channel");
  return CollectiveMainloop::has_zero_channel ? zeros != nullptr
                                               : zeros == nullptr;
}

template <class ShippingTypes>
inline bool one_plane_metadata_arguments_valid(
    typename ShippingTypes::CollectiveMainloop::Arguments const& mainloop) {
  return one_plane_metadata_arguments_valid<ShippingTypes>(mainloop.ptr_Z);
}

// A shipping mainloop with a positive StaticGroupSize has already selected a
// concrete scale layout and conversion schedule.  The runtime argument may not
// silently describe a different metadata plane.  Zero/-1 retain their existing
// runtime/per-column meanings, while every launcher still rejects non-positive
// runtime group sizes.
template <class ShippingTypes>
inline bool shipping_group_size_arguments_valid(int group_size) {
  using DispatchPolicy =
      typename ShippingTypes::CollectiveMainloop::DispatchPolicy;
  constexpr int StaticGroupSize = DispatchPolicy::StaticGroupSize;
  return group_size > 0 &&
      (StaticGroupSize <= 0 || group_size == StaticGroupSize);
}

template <class ShippingTypes>
inline bool shipping_group_size_arguments_valid(
    typename ShippingTypes::CollectiveMainloop::Arguments const& mainloop) {
  return shipping_group_size_arguments_valid<ShippingTypes>(
      mainloop.group_size);
}

// A reusable one-plane handle whose initialization is deliberately separate
// from device timing.  The old generic_launcher below remains the source-
// compatible one-shot API.  Performance sweeps use this handle so host-side
// argument lowering and initialize() cannot be charged to a 7--17 us device
// span.  S==1 still instantiates and runs ShippingTypes::Gemm verbatim; S>1
// instantiates the same mainloop behind the FP32-partial kernel.  run() remains
// the two-launch arithmetic oracle; run_fused_last_arriver() is the one-launch
// device canary and cannot silently replace it.
template <class ShippingTypes, class TileShape, class WarpShape>
class PreparedOnePlaneLauncher {
 public:
  using MainloopPolicy = typename ShippingTypes::MainloopPolicy;
  using ShippingKernel = typename ShippingTypes::GemmKernel;
  using ShippingGemm = typename ShippingTypes::Gemm;
  using SplitTypes = KernelTypes<ShippingTypes, TileShape, WarpShape>;
  using SplitKernel = typename SplitTypes::GemmKernel;
  using SplitGemm = typename SplitTypes::Gemm;
  using FusedKernel = typename SplitTypes::FusedGemmKernel;
  using FusedGemm = typename SplitTypes::FusedGemm;
  using Reduction = typename SplitTypes::Reduction;
  using ElementB = typename ShippingTypes::CollectiveMainloop::ElementB;
  using StrideScale = typename ShippingTypes::CollectiveMainloop::StrideScale;

 private:
  ShippingGemm shipping_{};
  SplitGemm split_{};
  FusedGemm fused_{};
  // Same exact kernel type as fused_; only the runtime completion argument
  // differs, so the diagnostic cannot change registers, smem, or occupancy.
  FusedGemm publish_only_{};
  Reduction reduction_{};
  int splits_ = 0;
  bool initialized_ = false;
  bool fused_initialized_ = false;
  bool publish_only_initialized_ = false;
  int32_t* fused_counters_ = nullptr;
  size_t fused_counter_bytes_ = 0;
  hggcStream_t fused_stream_ = nullptr;

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
                    ppu_tactics::kBlockSmemBytes &&
                    FusedKernel::SharedStorageSize <=
                    ppu_tactics::kBlockSmemBytes,
                "prepared shipping/partial/completion kernels must fit the compiled PPU smem limit");
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
    fused_initialized_ = false;
    publish_only_initialized_ = false;
    fused_counters_ = nullptr;
    fused_counter_bytes_ = 0;
    fused_stream_ = nullptr;
    splits_ = 0;
    if (m != MainloopPolicy::PackedARows || n <= 0 || k <= 0 ||
        !shipping_group_size_arguments_valid<ShippingTypes>(group_size) ||
        A == nullptr || B == nullptr || scales == nullptr || D == nullptr) {
      return false;
    }
    if (!one_plane_metadata_arguments_valid<ShippingTypes>(zeros)) {
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
    WorkspacePlan fused_workspace_plan;
    bool const fused_workspace_available = query_fused_workspace_plan(
        m, n, split_k_slices,
        int(cute::size<0>(TileShape{})), int(cute::size<1>(TileShape{})),
        fused_workspace_plan) &&
        workspace_bytes >= fused_workspace_plan.total_bytes;

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
    if (fused_workspace_available) {
      int32_t* counters = reinterpret_cast<int32_t*>(
          workspace + fused_workspace_plan.counter_offset);
      typename FusedGemm::Arguments fused_args{
          cutlass::gemm::GemmUniversalMode::kGemm,
          {m, n, k, 1},
          {A, split_sA, B, split_sB, scales, sS, group_size, zeros},
          {partials, sP, partials, sP},
          split_k_slices};
      fused_args.completion = {
          partials, D, counters, m, n, n,
          fused_workspace_plan.output_tiles, split_k_slices, true};
      typename FusedGemm::Arguments publish_only_args = fused_args;
      publish_only_args.completion = fused_args.completion;
      publish_only_args.completion.perform_final_reduction = false;
      if (FusedGemm::can_implement(fused_args) == cutlass::Status::kSuccess &&
          FusedGemm::get_workspace_size(fused_args) == 0 &&
          fused_.initialize(fused_args, nullptr, stream) ==
              cutlass::Status::kSuccess &&
          FusedGemm::can_implement(publish_only_args) ==
              cutlass::Status::kSuccess &&
          FusedGemm::get_workspace_size(publish_only_args) == 0 &&
          publish_only_.initialize(publish_only_args, nullptr, stream) ==
              cutlass::Status::kSuccess &&
          hggcMemsetAsync(counters, 0, fused_workspace_plan.counter_bytes,
                          stream) == hggcSuccess) {
        fused_initialized_ = true;
        publish_only_initialized_ = true;
        fused_counters_ = counters;
        fused_counter_bytes_ = fused_workspace_plan.counter_bytes;
        fused_stream_ = stream;
      }
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

  // One-launch diagnostic counterfactual.  It shares the exact producer
  // mainloop and partial ABI with run(), but the actual last physical peer
  // performs the fixed-order reduction before retiring.  The PPU canary
  // closed correctness and rejected performance (3.26--7.52% slower), so
  // run() remains both the arithmetic oracle and the selected policy.
  cutlass::Status run_fused_last_arriver(hggcStream_t stream = nullptr) {
    if (!initialized_ || splits_ <= 1 || !fused_initialized_ ||
        stream != fused_stream_) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    return fused_.run(stream);
  }

  // Timing-only diagnostic.  It executes the same producer and complete
  // actual-last publication lifecycle, including terminal acquire and counter
  // reset, through the same compiled kernel type and resource allocation.  A
  // CTA-uniform diagnostic bit skips peer reduction and D stores.  Its output
  // is intentionally invalid and this seam is forbidden from production
  // ranking.
  cutlass::Status run_publish_protocol_only_for_diagnostics(
      hggcStream_t stream = nullptr) {
    if (!initialized_ || splits_ <= 1 || !publish_only_initialized_ ||
        stream != fused_stream_) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    return publish_only_.run(stream);
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

  // The workspace is immutable after the producer returns, so repeating this
  // launch measures the standalone reduction without manufacturing another
  // producer.  It is diagnostic-only: production correctness still belongs
  // to run(), which enqueues both kernels on one stream.
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

  bool fused_last_arriver_selected_for_diagnostics() const {
    return initialized_ && splits_ > 1 && fused_initialized_;
  }

  bool publish_protocol_only_selected_for_diagnostics() const {
    return initialized_ && splits_ > 1 && publish_only_initialized_;
  }

  cutlass::Status reset_fused_counters_for_diagnostics(
      hggcStream_t stream = nullptr) {
    if (!fused_initialized_ || fused_counters_ == nullptr ||
        fused_counter_bytes_ == 0 || stream != fused_stream_) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    return hggcMemsetAsync(
               fused_counters_, 0, fused_counter_bytes_, stream) == hggcSuccess
        ? cutlass::Status::kSuccess
        : cutlass::Status::kErrorInternal;
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

  using MainloopPolicy = typename ShippingTypes::MainloopPolicy;
  if (!shipping_group_size_arguments_valid<ShippingTypes>(group_size)) {
    return false;
  }
  if constexpr (MainloopPolicy::HighBits == 0) {
    if (!one_plane_metadata_arguments_valid<ShippingTypes>(zeros)) {
      return false;
    }
  }

  if (split_k_slices == 1) {
    return fpa_intb_ppu::generic_launcher<
        QuantOp, BaseSchedule, TileShape, ScaleTileShape, WarpShape, Stages,
        AiuInterleaved, ElementB, PlaneB2,
        false, false, false, ArtifactTileK, ShippingTypesOverride>(
            A, B, scales, zeros, D, m, n, k, group_size, 1,
            workspace, workspace_bytes, stream, B2);
  }
  WorkspacePlan workspace_plan;
  if (!query_workspace_plan(m, n, split_k_slices, workspace_plan) ||
      workspace == nullptr || workspace_bytes < workspace_plan.partial_bytes) {
    return false;
  }

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
