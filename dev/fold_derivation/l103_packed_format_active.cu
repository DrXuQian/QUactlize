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
#include <type_traits>
using namespace cute;

#ifndef PPU_PACKED_FORMAT
#define PPU_PACKED_FORMAT 0
#endif
// THE WEIGHT TYPE IS PART OF THE FORMAT, and this file used to hardcode it. Every instantiation below said
// tuple<int4b_t, half, half> whatever PPU_PACKED_FORMAT selected -- so the format-2 run compiled Q2_K's GROUP
// COUNT onto Q4_K's WEIGHT TYPE, a combination production never builds, and reported "Q2_K is ACTIVE". The gate
// ran, passed, and covered a configuration that does not exist. codex's new type/format guard in the single-plane
// collective is what falsified it (080); before that, nothing could.
//
// So the format now names its own types AND its own tactic. Both matter: Q6_K's k-tile is deliberately HALF a
// superblock (TK=128, SK=8 at gs=16) because its weight placement is the box-validated TK128 one, and forcing
// TK=256 to make the k-tile a whole superblock is the tactic whose high-plane map produced conditioned error
// 8.76e-1. A gate that assumed "one k-tile == one superblock" could not express that, and would have demanded the
// wrong shape of the one format that must not have it.
template <int F> struct FmtTraits;
//   weight type / second plane (void = single) / TK / SK / the COPYABLE unit total.
//   kExpectTotal is DERIVED from the pairing rule -- supers = 2 when the format's per-superblock metadata is not
//   a multiple of 4 -- and not copied from anywhere, so a wrong pairing shows up here as a compile error rather
//   than as agreement between two transcriptions of the same mistake.
template <> struct FmtTraits<0> {        // Q4_K   meta 16, 16 % 4 == 0 -> one superblock
  using B = cutlass::int4b_t;  using B2 = void;
  static constexpr int kTK = 256, kSK = 8,  kExpectTotal = 16;
};
template <> struct FmtTraits<1> {        // Q5_K   meta 16 -> one; two-plane WEIGHT, same scale unit as Q4_K
  using B = cutlass::int4b_t;  using B2 = cutlass::uint1b_t;
  static constexpr int kTK = 256, kSK = 8,  kExpectTotal = 16;
};
template <> struct FmtTraits<2> {        // Q2_K   meta 20, 20 % 4 == 0 -> one superblock, staged as five 4 B copies
  using B = cutlass::uint2b_t; using B2 = void;
  static constexpr int kTK = 256, kSK = 16, kExpectTotal = 20;
};
template <> struct FmtTraits<3> {        // Q3_K   meta 14 -> 2 mod 4, UNMOVABLE alone; paired = 28
  using B = cutlass::uint2b_t; using B2 = cutlass::uint1b_t;
  static constexpr int kTK = 256, kSK = 16, kExpectTotal = 28;
};
template <> struct FmtTraits<4> {        // Q6_K   meta 18 -> 2 mod 4, paired = 36; TK128 kept, see above
  using B = cutlass::int4b_t;  using B2 = cutlass::uint2b_t;
  static constexpr int kTK = 128, kSK = 8,  kExpectTotal = 36;
};
using Fmt = FmtTraits<PPU_PACKED_FORMAT>;
static constexpr int kGroups = cutlass::gguf_packed::Unit<cutlass::gguf_packed::Fmt(PPU_PACKED_FORMAT)>::kGroups;

using TileShape      = Shape<_16,_128, Int<Fmt::kTK>>;
using ScaleTileShape = Shape<_128, Int<Fmt::kSK>>;
using WarpShape      = Shape<_16,_16, Int<Fmt::kTK>>;

// The B info tuple gains a fourth element for a two-plane format -- the same seam moe_grouped_ppu and
// fpA_intB_ppu use, so this instantiates the collective production selects rather than a lookalike.
using BInfo = std::conditional_t<
    std::is_void_v<typename Fmt::B2>,
    cute::tuple<typename Fmt::B, cutlass::half_t, cutlass::half_t>,
    cute::tuple<typename Fmt::B, cutlass::half_t, cutlass::half_t, typename Fmt::B2>>;

using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    BInfo,
    cutlass::layout::ColumnMajorInterleaved<256>, 32,
    float, cute::tuple<TileShape, ScaleTileShape>, WarpShape, cute::Int<2>,
    cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs32>::CollectiveOp;

// THE POINT OF THE FILE. Without this the gate is satisfied by a branch that is switched off.
static_assert(Mainloop::is_packed_scale,
              "the packed path is NOT active for this format at its own tactic -- so any gate that only compiles "
              "it is checking dead code");

// AND THE UNIT MUST BE THE FORMAT'S OWN. 16/20/28/36 bytes for Q4-Q5 / Q2 / Q3 / Q6. Asserting activation alone
// would still pass if the collective had selected some other format's staging, which is the exact way the old
// version of this file was wrong.
static_assert(Mainloop::PackedUnit::kUnitBytes
                  == cutlass::gguf_packed::Unit<cutlass::gguf_packed::Fmt(PPU_PACKED_FORMAT)>::kUnitBytes,
              "the collective activated a packed unit that is not this format's");

// AND THE PAIRING, which is the one thing Q3_K and Q6_K newly required and the only thing the assertion above
// cannot see. kUnitBytes is the PER-SUPERBLOCK metadata size (14 for Q3, 18 for Q6); kUnitTotal is what is
// actually copied, and for the two formats whose metadata is 2 mod 4 that is TWO superblocks -- 28 and 36 --
// because ppu.cp.async moves only 4, 8 or 16 bytes and 14 and 18 are movable at neither.
//
// The expected numbers are DERIVED from that rule here rather than copied from a message: supers = 2 when
// meta % 4 != 0. A build that paired wrongly, or not at all, satisfied every other assertion in this file.
static_assert(Mainloop::PackedUnit::kUnitTotal == Fmt::kExpectTotal,
              "the activated unit's COPYABLE size is not this format's paired total -- the staging is not doing "
              "the superblock pairing this format needs, or is doing it for a different format");

int main() {
  printf("[l103] format %d: groups=%d TK=%d SK=%d bits=%d%s packed_scale=%d unit=%d bytes\n",
         PPU_PACKED_FORMAT, kGroups, Fmt::kTK, Fmt::kSK, int(cutlass::sizeof_bits<typename Fmt::B>::value),
         std::is_void_v<typename Fmt::B2> ? "" : "+hi",
         int(Mainloop::is_packed_scale), int(Mainloop::PackedUnit::kUnitBytes));
  printf("[l103]   copyable unit total = %d (expected %d, paired = %d superblock(s))\n",
         int(Mainloop::PackedUnit::kUnitTotal), Fmt::kExpectTotal,
         Fmt::kExpectTotal / int(Mainloop::PackedUnit::kUnitBytes));
  return 0;
}
