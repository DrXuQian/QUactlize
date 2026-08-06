// L100 -- IS THE FUSED (scale, zero) PATH ACTUALLY ON? A macro that silently does nothing is this task's most
// expensive failure mode, and it has now happened twice.
//
// The first time: PPU_PACKED_SCALE_FUSED was built, the correctness gate passed, the perf run measured -0.8% with a
// CI containing 1.0, and acu reported Shared Store bank conflicts 81,920 (+0.00%) -- byte-for-byte what pack has.
// The fix supposedly removes exactly 73,728 of those, so "no change at all" is not a small effect, it is the path
// being inactive. Nothing in the build, the gate or the bench could tell the difference, because a flag that does
// nothing produces a working binary and a plausible number.
//
// WHAT THIS FILE ASSERTS, and why each piece is separate. kFusedScaleZero is a conjunction of three conditions and
// the useless report is "false". So each conjunct is asserted on its own line: whichever line fires names the reason
// without a bisect. Two of the three come from OUTSIDE this feature -- kPackedScaleOn depends on Scale_TileK and the
// conversion mode is chosen by SK_QUANT -- so this doubles as a check that the pinned bench row still satisfies the
// preconditions it was chosen for.
//
// Built by ci/local_gates.py with -DPPU_PACKED_SCALE=1 -DPPU_PACKED_SCALE_FUSED=1. It needs no device and no data:
// every claim is a static_assert on the real CollectiveOp, built exactly the way l95_stub_vs_real builds it.
#include <hggc_fp16.h>
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "quactlize_actlize.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
#include <cstdio>
using namespace cute;

// THE PINNED PERF ROW, and it must stay this row. benchmarks/run_batch.sh times "16x128:256 w16x16 s2" at gs=32, and
// the whole store-conflict derivation (4 warps x 8 groups x 2 planes x 9 passes x 128 CTAs = 73,728) is that row's.
using TileShape      = Shape<_16,_128,_256>;
using ScaleTileShape = Shape<_128,_8>;                 // gs=32 with TileK=256 -> Scale_TileK = 8
using WarpShape      = Shape<_16,_16,_256>;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cute::tuple<cutlass::int4b_t, cutlass::half_t, cutlass::half_t>,
    cutlass::layout::ColumnMajorInterleaved<256>, 32,
    float, cute::tuple<TileShape, ScaleTileShape>, WarpShape, cute::Int<2>,
    cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs32>::CollectiveOp;

// (1) THE PRECONDITION FROM THE PACKED CHANNEL. kPackedScaleOn is `PPU_PACKED_SCALE && Scale_TileK == 8`, so this
//     firing means either the macro did not reach this translation unit or the row's Scale_TileK moved.
static_assert(Mainloop::is_packed_scale,
              "kPackedScaleOn is false: either -DPPU_PACKED_SCALE did not reach here, or Scale_TileK != 8");

// (2) THE PRECONDITION FROM THE CONVERSION MODE. Fusing needs a zero channel to fuse WITH; SK_QUANT=1 (ScaleOnly)
//     has none. This is the conjunct I added without ever checking it, so it gets its own line.
// ConversionMode is a MEMBER enum of the collective (declared at its line 127), not a namespace-scope one -- there
// are three unrelated definitions of that name in three collectives, so naming it through Mainloop is also the only
// way to be sure this is the one that governs THIS type.
static_assert(Mainloop::has_zero_channel,
              "conversion mode is not ConvertAndScaleWithZero, so there is no zero plane to fuse");

// (3) THE FEATURE ITSELF. If (1) and (2) hold and this still fires, the macro is not defined in this build.
static_assert(Mainloop::is_fused_scale_zero,
              "kFusedScaleZero is FALSE while its preconditions hold -- -DPPU_PACKED_SCALE_FUSED did not reach here");

// (4) THE ALLOCATION MUST HAVE MOVED, and by nothing. The fused tile takes both planes' elements and the zero tile
//     goes to zero, so the total is byte-identical -- which is deliberate and is also why NO counter in the bench or
//     in acu's Launch Statistics can reveal whether the path is on. Shared memory per block is the same either way.
//     That is exactly why this file exists: the property that makes the change safe also makes it invisible.
static constexpr int kScaleElems = int(Mainloop::SharedStorage::scale_elements);
static constexpr int kZeroElems  = int(Mainloop::SharedStorage::zero_elements);
static constexpr int kPlain      = int(cute::cosize_v<typename Mainloop::SmemLayoutScale>);
static_assert(kZeroElems == 0,           "fused: the zero tile must be gone, not merely smaller");
static_assert(kScaleElems == 2 * kPlain, "fused: the scale tile must carry both planes");

// (5) THE STRIDES THE WHOLE DERIVATION RESTS ON. The store is conflict-free only because 32 lanes with consecutive n
//     write 32 CONSECUTIVE WORDS: stride 1 in n on the word view. And the half views must interleave, i.e. stride 2.
using W = typename Mainloop::FusedScaleWordLayout;
using H = typename Mainloop::FusedScaleHalfLayout;
static_assert(int(cute::stride<0>(W{})) == 1,   "fused word view: n stride must be 1 or the store reconflicts");
static_assert(int(cute::stride<1>(W{})) == 128, "fused word view: group stride must be Scale_TileN");
static_assert(int(cute::stride<0>(H{})) == 2,   "fused half view: n stride must be 2 (scale and zero interleaved)");

int main() {
  printf("[l100] fused (scale,zero) ACTIVE on the pinned row: scale_elems=%d zero_elems=%d (plain %d)\n",
         kScaleElems, kZeroElems, kPlain);
  printf("       word view stride n=%d g=%d   half view stride n=%d\n",
         int(cute::stride<0>(W{})), int(cute::stride<1>(W{})), int(cute::stride<0>(H{})));
  return 0;
}
