// L103 -- IS THE PACKED PATH ACTUALLY ON FOR THE FORMAT THE BUILD SELECTED? A gate that compiles a disabled branch
// proves the branch parses and nothing else, and that is precisely what the five-format syntax gate was doing.
//
// kPackedScaleOn requires Scale_TileK == the format's group count -- 8 for Q4_K and Q5_K, 16 for Q2_K, Q3_K and
// Q6_K. Every row of tests/test_q4k_packed_gemm.cu has Scale_TileK of 8 or 2, so for the three 16-group formats the
// decoder, its staging and its partial-word register assembly were never instantiated as ACTIVE code, and the gate
// reported "clean" for all five. That is the same degenerate-verification shape as a comparison against an all-NaN
// buffer: the check runs, passes, and covers nothing.
//
// So this file builds the collective at a shape where the format's own condition CAN hold, and asserts it does.
// A format that cannot activate fails here instead of passing somewhere else.
//
//   nvcc ... -DPPU_PACKED_SCALE=1 -DPPU_PACKED_FORMAT=<0..4> -DL103_GS=<32|16> l103_packed_format_active.cu
#include <hggc_fp16.h>
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "ppu_include.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
#include <cstdio>
using namespace cute;

#ifndef PPU_PACKED_FORMAT
#define PPU_PACKED_FORMAT 0
#endif
// The group count the selected format has, and the ScaleTileShape that makes a k-tile exactly one superblock.
static constexpr int kGroups = cutlass::gguf_packed::Unit<cutlass::gguf_packed::Fmt(PPU_PACKED_FORMAT)>::kGroups;

using TileShape      = Shape<_16,_128, Int<32 * kGroups>>;      // gs=32 * groups -> Scale_TileK == kGroups
using ScaleTileShape = Shape<_128, Int<kGroups>>;
using WarpShape      = Shape<_16,_16, Int<32 * kGroups>>;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cute::tuple<cutlass::int4b_t, cutlass::half_t, cutlass::half_t>,
    cutlass::layout::ColumnMajorInterleaved<256>, 32,
    float, cute::tuple<TileShape, ScaleTileShape>, WarpShape, cute::Int<2>,
    cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs32>::CollectiveOp;

// THE POINT OF THE FILE. Without this the gate is satisfied by a branch that is switched off.
static_assert(Mainloop::is_packed_scale,
              "the packed path is NOT active for this format at a k-tile of one superblock -- so any gate that only "
              "compiles it is checking dead code");

int main() {
  printf("[l103] format %d: groups=%d packed_scale=%d unit=%d bytes\n", PPU_PACKED_FORMAT, kGroups,
         int(Mainloop::is_packed_scale), int(Mainloop::PackedUnit::kUnitBytes));
  return 0;
}
