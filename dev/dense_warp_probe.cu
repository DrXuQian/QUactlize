// Does the DENSE route actually reject a two-warp CTA?
//
// A four-warp minimum on the dense route (ppu_tactic_space.hpp) said every tested sub-four-warp instantiation
// aborted on device, and its own comment said to keep it "until the non-grouped kernel/epilogue assert site is
// identified". Nobody had identified it. This TU was the identification attempt -- instantiate the dense
// mixed-input kernel at the RECORDED WINNER's geometry and force the whole mainloop into existence -- and it
// found nothing, which is why that exclusion was deleted on 2026-08-05. It stays as the standing check.
//
//   PROBE_WM=64 PROBE_WN=32  ->  (64/64)*(64/32) = 2 warps   <- docs/BACKTEST.md A1, 211.33 us / 65.0% on grouped
//   PROBE_WM=32 PROBE_WN=32  ->  (64/32)*(64/32) = 4 warps   <- the control, known-good on dense
//
// Everything except WarpShape is held identical, so a difference between the two compiles is attributable.
// Forced instantiation copies moe_grouped_ppu.cuh's mechanism: taking the address of the __global__
// device_kernel<K> odr-uses it and drags the entire mainloop in. Without that the front end parses the
// collective and instantiates nothing, which is the gap that let two static errors reach the box.
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "ppu_group_schedule.hpp"
#include "quactlize_actlize.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
#include "ppu_mixed_policy.hpp"

using namespace cute;

// ---- element types, copied verbatim from benchmarks/test_lowbit_dense_bench.cu -------------------------------
using MmaType   = cutlass::half_t;
using QuantType = cutlass::int4b_t;
constexpr int TileShapeK = 128 * 8 / cutlass::sizeof_bits<MmaType>::value;   // 64

using ElementA = MmaType;
using LayoutA  = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB    = QuantType;
using LayoutB     = cutlass::layout::ColumnMajor;
using LayoutB_opt = cutlass::layout::ColumnMajorInterleaved<256>;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = MmaType;
using LayoutC  = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementD = ElementC;
using LayoutD  = LayoutC;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ArchTag            = cutlass::arch::PPU0010;
using OperatorClass      = cutlass::arch::OpClassTensorOp;

// ---- the geometry under test ---------------------------------------------------------------------------------
#ifndef PROBE_TM
#define PROBE_TM 64
#endif
#ifndef PROBE_TN
#define PROBE_TN 64
#endif
#ifndef PROBE_WM
#define PROBE_WM 64
#endif
#ifndef PROBE_WN
#define PROBE_WN 32
#endif
#ifndef PROBE_STAGES
#define PROBE_STAGES 3
#endif
#ifndef PROBE_GS
#define PROBE_GS 32
#endif

using TileShape = Shape<Int<PROBE_TM>, Int<PROBE_TN>, Int<TileShapeK>>;
using WarpShape = Shape<Int<PROBE_WM>, Int<PROBE_WN>, Int<TileShapeK>>;

using EpilogueSchedule = cutlass::epilogue::EpilogueSimtVectorized;
using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, WarpShape,
    EpilogueTileType,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule
  >::CollectiveOp;

using Schedule = ppu_group_schedule::FinegrainedSchedule<PROBE_GS>;
constexpr int ScaleK = ppu_group_schedule::scale_groups_v<TileShapeK, PROBE_GS>;
using ScaleTile = Shape<Int<PROBE_TN>, Int<ScaleK>>;

using ScaleOnlyPolicy = ppu_mixed_policy::MainloopPolicy<
    ppu_mixed_policy::QuantMode::FinegrainedScaleOnly, Schedule, TileShape, ScaleTile, WarpShape,
    PROBE_STAGES, true, ElementB>;
using ScaleOnlyMainloop = typename ScaleOnlyPolicy::CollectiveOp;

// THE DENSE KERNEL: plain ProblemShape, NOT the grouped GroupProblemShape.
using DenseKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, ScaleOnlyMainloop, CollectiveEpilogue>;
using DenseAdapter = cutlass::gemm::device::GemmUniversalAdapter<DenseKernel>;

// Report the two quantities the quarantine is about, so the compile also PRINTS what it thinks it built.
static constexpr int kCtaWarps    = (PROBE_TM / PROBE_WM) * (PROBE_TN / PROBE_WN);
static constexpr int kMmaThreads  = cute::size(typename ScaleOnlyMainloop::TiledMma{});
static_assert(kMmaThreads == 32 * kCtaWarps,
              "cta_warps(tm/wm * tn/wn) does not equal size(TiledMma)/32 -- the host predicate is re-derived, "
              "not read off the instantiated type");

void probe_shapes(int* out) {
  out[0] = kCtaWarps;
  out[1] = kMmaThreads;
  out[2] = (int) DenseKernel::SharedStorageSize;
  out[3] = (int) sizeof(typename DenseAdapter::Arguments);
}

// Force the WHOLE mainloop into existence. This is the point of the TU.
void force_instantiate() {
  [[maybe_unused]] void const* p = (void const*) cutlass::device_kernel<DenseKernel>;
}
