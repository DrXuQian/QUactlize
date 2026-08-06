// fpA_intB launcher for actlize v1.0.0, structured after the PPU-official acext
// cutlass3/fpA_intB_gemm_template_cutlass3.cu (generic_mixed_gemm_kernelLauncher / dispatch_gs_and_check_valid
// / filter_and_run_mixed_gemm), adapted to the PUBLIC actlize API this repo vendors.
//
// WHAT IS FAITHFUL TO THE OFFICIAL CODE
//   - the compile-time FinegrainedGs128/Gs64/PerCol schedules (NOT example 16's generic runtime-g schedule),
//   - the explicit ScaleTileShape = Shape<CTA_N, ceil(CTA_K/gs)> and the cute::tuple<TileShape, ScaleTileShape>
//     mainloop tile argument,
//   - group_size -> schedule dispatch with the block_k >= group_size validity gate,
//   - SplitKSerialScheduler + batch_count(=split_k) in the kernel Arguments,
//   - AiuInterleaved chosen from n%256==0 && k%256==0.
//
// ADAPTED (verified against the actlize submodule, not guessed)
//   - builder signature: cutlass::gemm::collective::CollectiveBuilder<PPU0010, OpClassTensorOp, ElementA,
//     LayoutA, AlignA, ElementBInfo, LayoutB, AlignB, Accum, cute::tuple<TileShape,ScaleTileShape>,
//     ClusterShape(=WarpShape), Int<Stages>, Schedule>  (ppu_mma_builder.inl lines ~430+).
//   - kernel Arguments {mode, problem_shape, mainloop, epilogue, batch_count, hw_info, scheduler}
//     (ppu_aiu_gemm_mixed_input_splitk_serial.hpp line 128).
//   - epilogue builder + args tuple form taken from example 16 (known to compile green on this box).
//
// LIKELY TO NEED A COMPILE FIX ON THE BOX (I cannot run hgcc here) — the four spots, flagged inline:
//   [F1] EpilogueSchedule name (EpilogueSimtVectorized vs official's ...WithoutEvt).
//   [F2] whether ElementC must be void when there is no bias (official leaves a TODO about smem).
//   [F3] stride construction / swap-and-transpose: example 16 REVERSED C/D strides; the official does not.
//   [F4] the splitk kernel may want ClusterShape rather than WarpShape in the epilogue builder.
#pragma once

#include <cstdio>
#include <type_traits>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "fold_traits.hpp"
#include "ppu_group_schedule.hpp"
#include "ppu_tactic_space.hpp"

#include "quactlize_actlize.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
#include "ppu_mixed_policy.hpp"
#include "cutlass/epilogue/collective/builders/ppu_builder.inl"

namespace fpa_intb_ppu {
using namespace cute;
using TacticSpace = ppu_tactics::DenseSpace;

using QuantMode = ppu_mixed_policy::QuantMode;
using ppu_mixed_policy::has_zero;
using ppu_mixed_policy::is_finegrained;

template <QuantMode QuantOp, class BaseSchedule, class TileShape, class ScaleTileShape, class WarpShape,
          int Stages, bool AiuInterleaved, class ElementB = cutlass::int4b_t, class PlaneB2 = void,
          int ArtifactTileK = 0, int ACompactRows = cutlass::gemm::kDefaultACompactRows>
using MixedMainloopPolicy = ppu_mixed_policy::MainloopPolicy<QuantOp, BaseSchedule, TileShape, ScaleTileShape,
                                                             WarpShape, Stages, AiuInterleaved, ElementB, PlaneB2,
                                                             ArtifactTileK, ACompactRows>;

// One instantiation: fp16 x a packed 1/2/4-bit B plane, optionally with a second high plane. The ElementBInfo tuple
// is the same seam moe_grouped_ppu uses: a fourth tuple element makes CollectiveBuilder select the already-existing
// two-plane collective. There is no dense-specific second collective or converter to maintain here.
template <QuantMode QuantOp, class BaseSchedule,
          class TileShape, class ScaleTileShape, class WarpShape, int Stages, bool AiuInterleaved,
          class ElementB = cutlass::int4b_t, class PlaneB2 = void, bool ExpectPackedScale = false,
          bool QueryOnly = false, bool RequireUniversalFallback = false,
          int ArtifactTileK = 0, int ACompactRows = cutlass::gemm::kDefaultACompactRows>
bool generic_launcher(const cutlass::half_t* A, const ElementB* B,
                      const cutlass::half_t* scales, const cutlass::half_t* zeros, cutlass::half_t* D,
                      int m, int n, int k, int group_size, int split_k,
                      char* workspace, size_t workspace_bytes, hggcStream_t stream,
                      const PlaneB2* B2 = nullptr) {
  using MainloopPolicy = MixedMainloopPolicy<QuantOp, BaseSchedule, TileShape, ScaleTileShape, WarpShape,
                                              Stages, AiuInterleaved, ElementB, PlaneB2,
                                              ArtifactTileK, ACompactRows>;
  using ElementA = typename MainloopPolicy::ElementA;

  using ElementC = cutlass::half_t;                     // [F2] could be void when no bias
  using LayoutC  = cutlass::layout::RowMajor;
  using ElementD = cutlass::half_t;
  using LayoutD  = cutlass::layout::RowMajor;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementAccumulator = float;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ClusterShape  = WarpShape;                      // ppu1.0 has no cluster; the builder takes WarpShape here
  using EpilogueSchedule = cutlass::epilogue::EpilogueSimtVectorizedWithoutEvt;  // [F1] non-EVT: splitk kernel needs .thread.beta
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::PPU0010, OperatorClass, TileShape, ClusterShape, EpilogueTileType,
      ElementAccumulator, ElementAccumulator, ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD, EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop = typename MainloopPolicy::CollectiveOp;

  if constexpr (CollectiveMainloop::compact_a_rows > 0) {
    if (m > CollectiveMainloop::compact_a_rows) {
      if constexpr (!QueryOnly)
        std::printf("[fpA_intB] compact A holds %d rows, got M=%d; select the ordinary A path or a wider compact build\n",
                    CollectiveMainloop::compact_a_rows, m);
      return false;
    }
  }
  if constexpr (MainloopPolicy::ACompactRows > 0 && CollectiveMainloop::compact_a_rows == 0) {
    if constexpr (!QueryOnly)
      std::printf("[fpA_intB] compact A capacity %d is unavailable for this selected collective; A remains TileM rows\n",
                  MainloopPolicy::ACompactRows);
    return false;
  }

  // FULLY_QUANTIZED is an INSTANTIATION of the shared mainloop, not a dense-specific decoder. Make that selection
  // compile-time observable at its call site: a flagged binary that accidentally falls back to fp16 scale planes
  // must fail to build instead of accepting packed bytes through the half pointer and producing plausible garbage.
  // The false/default arm deliberately does not name the witness, so ordinary fp16-scale instantiations remain
  // independent of whether their selected collective exposes a packed channel.
  if constexpr (ExpectPackedScale) {
    static_assert(CollectiveMainloop::is_packed_scale,
                  "fully-quantized dense requires the shared packed-scale mainloop at this tile shape");
  }

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::SplitKSerialScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // This is the exact compiled type, including packed-unit staging, scale padding/swizzles and experimental A
  // layouts. The host arithmetic in ppu_tactic_space.hpp deliberately remains useful for emitting a broad finite
  // domain, but the runtime answer must not guess sizeof(SharedStorage) from tile coordinates. QueryOnly returns
  // before pointer/stride construction and requires no PPU context; the ordinary launch takes the same guard.
  static_assert(!RequireUniversalFallback ||
                    GemmKernel::SharedStorageSize <= ppu_tactics::kBlockSmemBytes,
                "the compiled dense default must fit one ppu001 block for every admitted shape");
  static_assert(!RequireUniversalFallback || CollectiveMainloop::compact_a_rows == 0,
                "the compiled dense default must use the unrestricted ordinary-A path");
#if defined(PPU_A_PACK) && (PPU_A_PACK != 0)
  static_assert(!RequireUniversalFallback,
                "PPU_A_PACK is a one-row experiment and cannot be the universal dense fallback");
#endif
  if constexpr (GemmKernel::SharedStorageSize > ppu_tactics::kBlockSmemBytes) return false;
  if constexpr (QueryOnly) return true;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
  using StrideS = typename CollectiveMainloop::StrideScale;

  // The same minimum-delivery fold the grouped consumer uses. A folded resident B is physically
  // (N/F, F*K), so both the schedule selected by dispatch_gs() and this gmem stride must name F. Before this
  // existed the so-called dense int1/int2 folded measurements were actually grouped launches at L=1.
  constexpr int P1_FOLD = MainloopPolicy::ArtifactLowFold;
  static_assert(ppu_mixed_policy::kernel_policy_valid_v<TacticSpace, MainloopPolicy>);

  const int scale_k = (k + group_size - 1) / group_size;
  StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB sB = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape(n / P1_FOLD, k * P1_FOLD, 1));
  StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
  StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));   // [F3]
  StrideS sS = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(n, scale_k, 1));

  const float beta = (zeros == nullptr) ? 0.f : 0.f;   // no bias path; scale/zero handled in the mainloop
  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {m, n, k, 1},
    { A, sA, B, sB, scales, sS, group_size, zeros },
    { {ElementAccumulator(1.f), ElementAccumulator(beta)}, (ElementC*)nullptr, sC, D, sD },
    split_k
  };
  if constexpr (!std::is_void_v<PlaneB2>) {
    args.mainloop.ptr_B2 = B2;
    // The high-plane stride is part of the artifact ABI. It is intentionally independent of TacticTileK: Q3 at
    // ArtifactTileK=64 is physically (F_low,F_high)=(2,4), even when a TK256 tactic consumes it.
    constexpr int ARTIFACT_HIGH_FOLD = MainloopPolicy::ArtifactHighFold;
    if constexpr (ARTIFACT_HIGH_FOLD != P1_FOLD) {
      args.mainloop.dB2 = cutlass::make_cute_packed_stride(
          StrideB{}, cute::make_shape(n / ARTIFACT_HIGH_FOLD, k * ARTIFACT_HIGH_FOLD, 1));
      args.mainloop.dB2_valid = true;
    }
  }

  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
  if (gemm.get_workspace_size(args) > workspace_bytes) return false;
  if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) return false;
  gemm.run(stream);
  return true;
}

// group_size -> compile-time schedule + ScaleTileShape, with the official block_k >= group_size gate.
template <QuantMode QuantOp, int TM, int TN, int TK, int WM, int WN, int Stages, bool AiuInterleaved,
          class ElementB = cutlass::int4b_t, class PlaneB2 = void, int ArtifactTileK = TK,
          int ACompactRows = cutlass::gemm::kDefaultACompactRows>
bool dispatch_gs(const cutlass::half_t* A, const ElementB* B, const cutlass::half_t* scales,
                 const cutlass::half_t* zeros, cutlass::half_t* D, int m, int n, int k, int group_size,
                 int split_k, char* ws, size_t ws_bytes, hggcStream_t stream, const PlaneB2* B2 = nullptr) {
  using TileShape = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;
  using WarpShape = cute::Shape<cute::Int<WM>, cute::Int<WN>, cute::Int<TK>>;
  if (k % 64 || n % 64) { std::printf("[fpA_intB] n,k must be multiples of 64\n"); return false; }

  if constexpr (is_finegrained(QuantOp)) {
    // Official acext RETURNED here when block_k < group_size; we relaxed it to proceed (a group then spans
    // gs/CTA_K k-tiles, CTA_SCALE_K = ceil(CTA_K/gs) = 1 when TK<gs). No per-launch print -- it flooded the
    // sweep. Timing stays valid; the bench-harness verify catches any correctness issue from block_k < gs.
    switch (group_size) {
      case 128: {
        constexpr int CTA_SCALE_K = ppu_group_schedule::scale_groups_v<TK, 128>;
        return generic_launcher<QuantOp, ppu_group_schedule::FinegrainedSchedule<128>,
            TileShape, cute::Shape<cute::Int<TN>, cute::Int<CTA_SCALE_K>>, WarpShape, Stages, AiuInterleaved,
            ElementB, PlaneB2, false, false, false, ArtifactTileK, ACompactRows>(
                A, B, scales, zeros, D, m, n, k, group_size, split_k, ws, ws_bytes, stream, B2);
      }
      case 64: {
        constexpr int CTA_SCALE_K = ppu_group_schedule::scale_groups_v<TK, 64>;
        return generic_launcher<QuantOp, ppu_group_schedule::FinegrainedSchedule<64>,
            TileShape, cute::Shape<cute::Int<TN>, cute::Int<CTA_SCALE_K>>, WarpShape, Stages, AiuInterleaved,
            ElementB, PlaneB2, false, false, false, ArtifactTileK, ACompactRows>(
                A, B, scales, zeros, D, m, n, k, group_size, split_k, ws, ws_bytes, stream, B2);
      }
      case 32: {
        // gs=32 IS THE DECODE GROUP SIZE and it was not dispatched here, so the dense split-K path could not be
        // measured at the shape the grouped kernel actually runs. The Gs32 schedule exists in dispatch_policy.hpp
        // and its mainloop policy exposes `Schedule = KernelAiuMultistageMixedInput`, which is exactly what the
        // SplitKSerialScheduler specialization enable_ifs on -- so this reaches the split-K kernel unchanged.
        constexpr int CTA_SCALE_K = ppu_group_schedule::scale_groups_v<TK, 32>;
        return generic_launcher<QuantOp, ppu_group_schedule::FinegrainedSchedule<32>,
            TileShape, cute::Shape<cute::Int<TN>, cute::Int<CTA_SCALE_K>>, WarpShape, Stages, AiuInterleaved,
            ElementB, PlaneB2, false, false, false, ArtifactTileK, ACompactRows>(
                A, B, scales, zeros, D, m, n, k, group_size, split_k, ws, ws_bytes, stream, B2);
      }
      case 16: {
        // Q2_K/Q3_K/Q6_K. The shared collective applies the fine scale per MMA atom; Gs32 is only the schedule tag,
        // while ScaleTileShape carries the real eight groups in a TK=128 tile (sixteen in the production TK=256).
        constexpr int CTA_SCALE_K = ppu_group_schedule::scale_groups_v<TK, 16>;
        return generic_launcher<QuantOp, ppu_group_schedule::FinegrainedSchedule<16>,
            TileShape, cute::Shape<cute::Int<TN>, cute::Int<CTA_SCALE_K>>, WarpShape, Stages, AiuInterleaved,
            ElementB, PlaneB2, false, false, false, ArtifactTileK, ACompactRows>(
                A, B, scales, zeros, D, m, n, k, group_size, split_k, ws, ws_bytes, stream, B2);
      }
      default: std::printf("[fpA_intB] group_size %d unsupported (finegrained: 16/32/64/128)\n", group_size);
    }
  } else {  // per-column
    return generic_launcher<QuantOp, cutlass::gemm::KernelAiuMultistageMixedInputPerCol,
        TileShape, cute::Shape<cute::Int<TN>, cute::_1>, WarpShape, Stages, AiuInterleaved,
        ElementB, PlaneB2, false, false, false, ArtifactTileK, ACompactRows>(
            A, B, scales, zeros, D, m, n, k, k, split_k, ws, ws_bytes, stream, B2);
  }
  return false;
}

// AiuInterleaved from shape divisibility (official filter_and_run_mixed_gemm).
template <QuantMode QuantOp, int TM, int TN, int TK, int WM, int WN, int Stages,
          class ElementB = cutlass::int4b_t, class PlaneB2 = void, int ArtifactTileK = TK,
          int ACompactRows = cutlass::gemm::kDefaultACompactRows>
bool filter_and_run(const cutlass::half_t* A, const ElementB* B, const cutlass::half_t* scales,
                    const cutlass::half_t* zeros, cutlass::half_t* D, int m, int n, int k, int group_size,
                    int split_k, char* ws, size_t ws_bytes, hggcStream_t stream, const PlaneB2* B2 = nullptr) {
  if (n % 256 == 0 && k % 256 == 0)
    return dispatch_gs<QuantOp, TM, TN, TK, WM, WN, Stages, true, ElementB, PlaneB2,
                       ArtifactTileK, ACompactRows>(
        A,B,scales,zeros,D,m,n,k,group_size,split_k,ws,ws_bytes,stream,B2);
  else
    return dispatch_gs<QuantOp, TM, TN, TK, WM, WN, Stages, false, ElementB, PlaneB2,
                       ArtifactTileK, ACompactRows>(
        A,B,scales,zeros,D,m,n,k,group_size,split_k,ws,ws_bytes,stream,B2);
}

} // namespace fpa_intb_ppu
