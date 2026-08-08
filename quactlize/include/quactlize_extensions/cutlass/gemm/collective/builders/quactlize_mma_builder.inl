#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder_decl.hpp"

#include "cutlass/detail/collective.hpp"

#include "cutlass/gemm/collective/builders/tile_shape_infer.inl"
#include "cutlass/gemm/config/gemm_operands.hpp"

// quactlize's: the dispatch policies this builder selects, and the NoZero marker its ScaleOnly 2-plane arm parks
// in the zero slot. Both were inside actlize headers until 2026-08-06.
#include "quactlize_extensions/cutlass/gemm/quactlize_dispatch_policy.hpp"
#include "quactlize_extensions/cutlass/detail/quactlize_mixed_dtype.hpp"
#include "quactlize_extensions/cutlass/gemm/collective/detail/ppu_a_pack.hpp"

#define ENABLE_AIU 1

namespace cutlass::gemm::collective {

namespace quactlize_ppu_detail {

constexpr int ppu10000_smem_capacity_bytes = 262144;
// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template<int CapacityBytes, class ElementA, class ElementB, class TileShapeMNK, int stages>
constexpr int
compute_stage_count_or_override(StageCount<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template<int CapacityBytes, class ElementA, class ElementB, class TileShapeMNK, int stages>
constexpr int
compute_stage_count_or_override(cute::Int<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template<int CapacityBytes, class ElementA, class ElementB, class TileShapeMNK, int carveout_bytes>
constexpr int
compute_stage_count_or_override(StageCountAutoCarveout<carveout_bytes> stage_count) {
  static_assert(carveout_bytes < ppu10000_smem_capacity_bytes, "epilogue carved out shm size should be smaller than total shm size");
  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr int stage_bytes =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{}));
  constexpr int compute_stages = (CapacityBytes - carveout_bytes) / stage_bytes;
  constexpr int out_stages = min(compute_stages, Int<5>{});
  return out_stages;
}
} // namespace quactlize_ppu_detail
namespace quactlize_detail {

///////////////////////////////////////////////////////////////////////////////
#if ENABLE_AIU
// ContigShape_: describes WHAT MAKES UP the AIU's 32-byte contiguous run. void (default) = the run is pure K, i.e.
// Shape<Block_K> -- byte-identical to the original derivation for every existing config (proved for int4 TK64/TK128,
// int2 TK128/TK256, int1 TK256 x MN64/MN128). An N-FOLD instead passes Shape<Int<FoldF>, Int<TK>>, so the 32B rule is
// satisfied by (FoldF N-columns x TK) and every derived quantity (AiuContElemSize, InstNum, bits_per_aiu, swzl CUBE_W)
// follows automatically -- no manual Block_K/Block_MN doubling to keep in sync across four places.
template <
  typename Element,
  bool Trans,
  typename Block_MN,
  typename Block_K,
  bool Swap,
  typename ContigShape_ = void
> struct MixGemm_AIU_Operand;

namespace aiu_detail {
// contiguous element count of the run: size(ContigShape) when given, else Block_K
template <class Block_K, class ContigShape_> struct contig_elems {
  static constexpr int value = cute::size(ContigShape_{});
};
template <class Block_K> struct contig_elems<Block_K, void> {
  static constexpr int value = Block_K{};
};
} // namespace aiu_detail

template <
  typename Element,
  typename Block_MN,
  typename Block_K,
  bool Swap
> struct MixGemm_AIU_Operand<
  Element,
  false,
  Block_MN,
  Block_K,
  Swap
> {
  static constexpr int BlockContSize = Block_K{} * sizeof_bits<Element>::value / 8;
  static_assert(BlockContSize % 32 == 0, "aiu_trans: block contiguous size should be multiple of 32B");
  static_assert(BlockContSize > 128 ? (BlockContSize % 128 == 0) : (BlockContSize % 32 == 0), "aiu_trans: block contiguous size should be multiple of 128B or 32B");
  static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  using AiuContElemSize = Int<AiuContByteSize / sizeof_bits<Element>::value * 8>;
  static constexpr int InstNum = Block_K{} / AiuContElemSize{};

  // THE AIU WRITE AND THE SWZL READ ARE ONE MATCHED TRIPLE, all three legs derived from Block_MN:
  //
  //     write payload   bits_per_aiu = Block_MN * AiuContElemSize * bits    -> PPU0010_AIU_LOAD<C<16384>,...>
  //     write cube      GmemTiledCopy's value layout (Block_MN, AiuCont)    -> Tiler_MN, hence desc_.cube_h
  //     read cube       PPU0010_TSM_LD_SWZL<Element, Block_MN, AiuCont,...> -> CUBE_H
  //
  // and both asm forms carry .swzl, so write-then-read is a byte identity only while the three agree. An earlier
  // PPU_A_CUBE_H changed the READ leg alone: a 16-row swizzled cube was written and reinterpreted as a 1-row cube,
  // and the kernel returned NaN. That was a one-sided edit, not a hardware limit -- printed side by side in
  // fold_derivation/l81_aiu_pair.cu.
  //
  // AND MOVING ALL THREE TOGETHER IS STILL NOT ENOUGH -- tried, measured, removed. With all three at 1 the NaN
  // went away (the write and read agreed again) but every value was wrong: max_rel 868, |max| 10.4 against 21.72,
  // finite and self-consistent, i.e. a permutation rather than missing data.
  //
  // The reason is in Copy_Traits<PPU0010_TSM_LD_SWZL>. SrcLayout is (32 lanes, 128 bits) with no CUBE_H in it, and
  // LogicalTV resolves the row index to
  //
  //     row = lane/4 + 8*(v/2),   lane/4 in [0,8), v/2 in [0,2)   ->   row in [0,16)
  //
  // so ONE instruction's (thread, vreg) structure spans SIXTEEN ROWS by construction. That is the instruction's
  // shape, not a parameter: CUBE_H reframes the hardware cube, it does not shrink that register footprint. And the
  // .swzl cancellation that makes write-then-read a byte identity requires the WRITE to frame the same 16-row cube,
  // which is why cube_h = 1 corrupted row 0 as well as the padding rows.
  //
  // The traits comment states the boundary exactly: stock cute covers any cube WIDTH. Height is fixed at 16 by the
  // TV structure. It is also why A's read costs only 4 instructions per k-tile -- one instruction already covers
  // the whole 16-row tile, so that is the floor, and A's entire chain is 0.6% of the instruction stream.
  static constexpr int bits_per_aiu = Block_MN{} * AiuContElemSize{} * sizeof_bits<Element>::value;
  using CopyInst = PPU0010_AIU_LOAD<cute::C<bits_per_aiu>, Element, false>;

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<CopyInst, Element>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Block_MN, AiuContElemSize>>{}));

  // The read leg of the triple above. THIS is the struct that builds A's atom on the mixed-input path -- printing
  // InternalSmemCopyAtomA gives PPU0010_TSM_LD_SWZL<half_t, 16, 64, true, false, 4>, whose (Block_MN,
  // AiuContElemSize, InstNum) match here and not DefaultGemm_AIU_Operand's, which is why an override placed there
  // was inert.
  // PPU_A_PACK=R: pack the cube BASES together. Each of the first R rows owns 32 of a cube's 512 words, in four
  // 8-word runs (fold_derivation/l84, l86); the bytes between those runs are read only into accumulator rows the
  // epilogue masks. l85 derives the collision-free, 128-B-aligned pitch for R. Geometry, and therefore the swizzle
  // and the write/read pairing, are untouched -- only the distance between bases changes.
#if defined(PPU_A_PACK) && (PPU_A_PACK != 0)
  // Both the read atom and collective writer call the same constexpr function. Separate literals once diverged
  // (16 here, 64 there), making the kernel write at one spacing and read at another until the AIU load faulted.
  static constexpr int kCubePitchA = detail::aPackPitchForRows(PPU_A_PACK);
#else
  static constexpr int kCubePitchA = 0;    // 0 = natural CUBE_H * CUBE_W
#endif
  using SmemCopyOp = PPU0010_TSM_LD_SWZL<Element, Block_MN{}, AiuContElemSize{}, Swap, false, InstNum, kCubePitchA>;
  using SmemCopyAtom = Copy_Atom<SmemCopyOp, Element>;
  using SmemLayoutAtom = Layout<Shape<_8, AiuContElemSize>, Stride<AiuContElemSize, _1>>;
};

template <
  typename Block_MN,
  typename Block_K,
  bool Swap
> struct MixGemm_AIU_Operand<
  cutlass::int4b_t,
  false,
  Block_MN,
  Block_K,
  Swap    // NO trailing comma: a trailing comma in a template-argument-list is ill-formed C++. clang (hence hgcc)
          // accepts it as an extension; nvcc's EDG front end rejects it, which made this specialization -- and
          // therefore CollectiveMma and EVERY collective downstream of it -- fail to instantiate under the local
          // nvcc front-end gate. Two static errors reached the box behind that one character.
> {
  static constexpr int BlockContSize = Block_K{} * sizeof_bits<cutlass::int4b_t>::value / 8;
  static_assert(BlockContSize % 32 == 0, "aiu_no_trans: block_k must be multiple of 32B");
  static_assert(BlockContSize > 128 ? (BlockContSize % 128 == 0) : (BlockContSize % 32 == 0), "aiu_trans: block contiguous size should be multiple of 128B or 32B");
  static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  using AiuContElemSize = Int<AiuContByteSize / sizeof_bits<cutlass::int4b_t>::value * 8>;
  static constexpr int InstNum = Block_K{} / AiuContElemSize{};

  static constexpr int bits_per_aiu = Block_MN{} * AiuContByteSize * 8;
  using CopyInst = PPU0010_AIU_LOAD<cute::C<bits_per_aiu>, cutlass::int4b_t, false>;     // load as i8

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<CopyInst, cutlass::int4b_t>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Block_MN, AiuContElemSize>>{}));

  using SmemCopyOp = PPU0010_TSM_LD_SWZL<int8_t, Block_MN{}, AiuContElemSize{} / 2, Swap, false, InstNum>;
  using SmemCopyAtom = Copy_Atom<SmemCopyOp, int8_t>;
  using SmemLayoutAtom = Layout<Shape<_8, AiuContElemSize>, Stride<AiuContElemSize, _1>>;
};

// W2A16 base plane: uint2b_t swzl operand. Mirrors the int4b_t spec above; only difference is 4 int2/byte
// (int4 has 2/byte) -> the int8-typed swzl element count is AiuContElemSize/4 (int4 uses /2). The resulting
// smem->reg fragment order is NOT the same as int4's -> its converter reshuffle must be probed on ppu001
// (see fast_numeric_conversion_for_mix_gemm.h uint2b_t wide converter TODO).
template <
  typename Block_MN,
  typename Block_K,
  bool Swap
> struct MixGemm_AIU_Operand<
  cutlass::uint2b_t,
  false,
  Block_MN,
  Block_K,
  Swap
> {
  static constexpr int BlockContSize = Block_K{} * sizeof_bits<cutlass::uint2b_t>::value / 8;   // Block_K/4 bytes
  static_assert(BlockContSize % 32 == 0, "aiu w2: block_k*2/8 must be multiple of 32B (block_k % 128 == 0)");
  static_assert(BlockContSize > 128 ? (BlockContSize % 128 == 0) : (BlockContSize % 32 == 0), "aiu w2: 128B or 32B");
  static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  using AiuContElemSize = Int<AiuContByteSize / sizeof_bits<cutlass::uint2b_t>::value * 8>;      // AiuContByteSize*4
  static constexpr int InstNum = Block_K{} / AiuContElemSize{};

  static constexpr int bits_per_aiu = Block_MN{} * AiuContByteSize * 8;
  using CopyInst = PPU0010_AIU_LOAD<cute::C<bits_per_aiu>, cutlass::uint2b_t, false>;            // load as i8

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<CopyInst, cutlass::uint2b_t>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Block_MN, AiuContElemSize>>{}));

  using SmemCopyOp = PPU0010_TSM_LD_SWZL<int8_t, Block_MN{}, AiuContElemSize{} / 4, Swap, false, InstNum>;  // 4 int2/byte
  using SmemCopyAtom = Copy_Atom<SmemCopyOp, int8_t>;
  using SmemLayoutAtom = Layout<Shape<_8, AiuContElemSize>, Stride<AiuContElemSize, _1>>;
};

// W1A16 base plane: uint1b_t swzl operand. Mirrors uint2b_t; 8 int1/byte (int2 has 4/byte) -> the int8-typed
// swzl element count is AiuContElemSize/8. Block_K*1/8 must be %32==0 -> Block_K % 256 == 0.
template <
  typename Block_MN,
  typename Block_K,
  bool Swap
> struct MixGemm_AIU_Operand<
  cutlass::uint1b_t,
  false,
  Block_MN,
  Block_K,
  Swap
> {
  static constexpr int BlockContSize = Block_K{} * sizeof_bits<cutlass::uint1b_t>::value / 8;   // Block_K/8 bytes
  static_assert(BlockContSize % 32 == 0, "aiu w1: block_k*1/8 must be multiple of 32B (block_k % 256 == 0)");
  static_assert(BlockContSize > 128 ? (BlockContSize % 128 == 0) : (BlockContSize % 32 == 0), "aiu w1: 128B or 32B");
  static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  using AiuContElemSize = Int<AiuContByteSize / sizeof_bits<cutlass::uint1b_t>::value * 8>;      // AiuContByteSize*8
  static constexpr int InstNum = Block_K{} / AiuContElemSize{};

  static constexpr int bits_per_aiu = Block_MN{} * AiuContByteSize * 8;
  using CopyInst = PPU0010_AIU_LOAD<cute::C<bits_per_aiu>, cutlass::uint1b_t, false>;            // load as i8

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<CopyInst, cutlass::uint1b_t>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Block_MN, AiuContElemSize>>{}));

  using SmemCopyOp = PPU0010_TSM_LD_SWZL<int8_t, Block_MN{}, AiuContElemSize{} / 8, Swap, false, InstNum>;  // 8 int1/byte
  using SmemCopyAtom = Copy_Atom<SmemCopyOp, int8_t>;
  using SmemLayoutAtom = Layout<Shape<_8, AiuContElemSize>, Stride<AiuContElemSize, _1>>;
};

#endif

template <typename Arch,
          typename ElementA,
          typename ElementB,
          typename ElementAccumulator,
          typename TileShape_MNK,
          typename ClusterShape_MNK,
          typename PermutionK_ = void
          >
struct get_tiled_mma {
  using MmaInst = typename config::GetMmaInst<Arch, ElementA, ElementB, ElementAccumulator>::type;

  static constexpr int blockM = cute::get<0>(TileShape_MNK{});
  static constexpr int blockN = cute::get<1>(TileShape_MNK{});
  static constexpr int blockK = cute::get<2>(TileShape_MNK{});

  // User can configure custom warp tile shape through ClusterShape_MNK
  static constexpr bool CustomWarpShape = cute::get<0>(ClusterShape_MNK{}) != 1 ||
                                          cute::get<1>(ClusterShape_MNK{}) != 1 ||
                                          cute::get<2>(ClusterShape_MNK{}) != 1;
  static constexpr auto WarpShapeStage = MapBlockShapeToWarpShapeStage<Arch, ElementA, blockM, blockN, blockK>();

  static constexpr int InstM = cute::get<0>(typename MMA_Traits<MmaInst>::Shape_MNK{});
  static constexpr int InstN = cute::get<1>(typename MMA_Traits<MmaInst>::Shape_MNK{});
  static constexpr int InstK = cute::get<2>(typename MMA_Traits<MmaInst>::Shape_MNK{});

  static constexpr int warpM = max(Int<InstM>{}, CustomWarpShape ? cute::get<0>(ClusterShape_MNK{}) : cute::get<0>(WarpShapeStage));
  static constexpr int warpN = max(Int<InstN>{}, CustomWarpShape ? cute::get<1>(ClusterShape_MNK{}) : cute::get<1>(WarpShapeStage));

  using WarpOnM = Int<blockM / warpM>;
  using WarpOnN = Int<blockN / warpN>;

  using PermutionK = cute::conditional_t<cute::is_void_v<PermutionK_>, Int<InstK>, PermutionK_>;
  static_assert(PermutionK{} % InstK == 0, "PermutionK must be multiple of InstK.");

  using TiledMma = cute::conditional_t<cute::is_void_v<PermutionK_>,
                      TiledMMA<MMA_Atom<MmaInst>,
                              cute::Layout<Shape<WarpOnM, WarpOnN, _1>>>,
                      TiledMMA<MMA_Atom<MmaInst>,
                              cute::Layout<Shape<WarpOnM, WarpOnN, _1>>,
                              Tile<Int<blockM / warpM * InstM>, Int<blockN / warpN * InstN>, PermutionK>>>;
};

} // namespace quactlize_detail


// AIU GEMM

// AIU GEMM with scale (for a8w8 block-wise quant)

// AIU Mixed GEMM
template <
  typename Arch,
  class ElementPairA_,
  class GmemLayoutA_,
  int AlignmentA,
  class ElementPairB_,
  class GmemLayoutB_,
  int AlignmentB,
  class ElementAccumulator,
  class TileShapePair_,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    Arch,
    arch::OpClassTensorOp,
    ElementPairA_,
    GmemLayoutA_,
    AlignmentA,
    ElementPairB_,
    GmemLayoutB_,
    AlignmentB,
    ElementAccumulator,
    TileShapePair_,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    // THE COMPLEMENT OF actlize's MIXED-INPUT ARM, deliberately, and this is the line that makes quactlize an
    // extension rather than a fork. actlize's builder claims {KernelTmaWarpSpecialized*MixedInput, PerCol, Gs128,
    // Gs64}. This one used to claim those SIX TOO, plus Gs32 and the fold contract -- a strict superset, so the
    // two specialisations were ambiguous for every schedule they shared and the translation unit would not
    // compile with both present. It only ever "worked" because quactlize's copy replaced actlize's in the include
    // list rather than joining it.
    //
    // The two quactlize owns:
    //   KernelAiuFold<F, Base, H>  -- ArtifactLowFold > 0 for ANY F, including F=1
    //   Gs32                       -- a schedule actlize does not define
    //
    // WHY EVERY quactlize CALLER STILL LANDS HERE, including the unfolded gs=128 and gs=64 rows that name
    // actlize's own schedule tags: ppu_mixed_policy::ArtifactFoldedSchedule wraps unconditionally, so what
    // reaches KernelScheduleType is KernelAiuFold<1, Gs128, 0> rather than a bare Gs128. That wrapper is
    // routing-neutral by construction -- ArtifactLowFold=1 gives HasFold=false and BaseSchedule=Gs128, which is
    // exactly what the bare tag produced. Pass a bare Gs128 and you get actlize's collective, which is the
    // correct answer to that question and no longer an accident of include order.
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelAiuMultistageMixedInputFinegrainedGs32> ||
       (fold_schedule_traits<KernelScheduleType>::ArtifactLowFold > 0))>   // artifact-fold contract
> {
private:
  using ScaleA = detail::deduce_mixed_width_dtype_t<1, ElementPairA_>;
  using ScaleB = detail::deduce_mixed_width_dtype_t<1, ElementPairB_>;
  // strip_no_zero_t: the 2-plane ScaleOnly tuple parks detail::NoZero in the zero slot to keep the second plane
  // at index 3. The builder and the collective MUST agree on that mapping, so both call the same alias.
  using ZeroA = detail::strip_no_zero_t<detail::deduce_mixed_width_dtype_t<2, ElementPairA_>>;
  using ZeroB = detail::strip_no_zero_t<detail::deduce_mixed_width_dtype_t<2, ElementPairB_>>;
  static constexpr bool NeitherIsTuple = !cute::is_tuple<ElementPairA_>::value && !cute::is_tuple<ElementPairB_>::value;

public:
  using TileShape_MNK = detail::deduce_mixed_width_dtype_t<0, TileShapePair_>;
  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementPairA_>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementPairB_>;
  static_assert(cute::is_tuple<ElementPairA_>::value ^ cute::is_tuple<ElementPairB_>::value ||
               (NeitherIsTuple && (sizeof_bits<ElementA>::value != sizeof_bits<ElementB>::value)),
    "Either A OR B must be a tuple or the widths of A and B must be different.");

  static constexpr bool IsANarrow = sizeof_bits<ElementA>::value < sizeof_bits<ElementB>::value;

  using ElementPairA = cute::conditional_t<IsANarrow && NeitherIsTuple, cute::tuple<ElementA>, ElementPairA_>;
  using ElementPairB = cute::conditional_t<!IsANarrow && NeitherIsTuple, cute::tuple<ElementB>, ElementPairB_>;

  static constexpr bool IsATransformed = cute::is_tuple<ElementPairA>::value;
  using ElementScale = cute::conditional_t<IsATransformed, ScaleA, ScaleB>;
  using ElementZero = cute::conditional_t<IsATransformed, ZeroA, ZeroB>;

  using ElementMma = cute::conditional_t<IsATransformed, ElementB, ElementA>;
  using RealInternalElementA = cute::conditional_t<IsATransformed, ElementB, ElementA>;
  using RealInternalElementB = cute::conditional_t<IsATransformed, ElementA, ElementB>;

  // currently only support a16w8 / a16w4 mix gemm
  // static_assert(IsATransformed, "currently only A is supported for quantization.");
  static_assert(sizeof_bits<RealInternalElementA>::value == 16 && (sizeof_bits<RealInternalElementB>::value == 8 || sizeof_bits<RealInternalElementB>::value == 4 || sizeof_bits<RealInternalElementB>::value == 2 || sizeof_bits<RealInternalElementB>::value == 1),
    "currently only support a16w8 / a16w4 / a16w2 / a16w1 mix gemm");
  // For fp32 types, map to tf32 MMA value type
  // using MmaElementA = ElementA; //cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  // using MmaElementB = ElementB; //cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  // PermutionK = the K span one B swzl copy step delivers = 32B worth of the packed element (int4 64 / int2 128 /
  // int1 256). Under an N-FOLD that 32B run is FoldF N-cols x blockK each, so the K span the MMA sees is only
  // blockK -- using the unfolded value would exceed TileShape.K and index past the tile.
  // FOLD: the fragment must have ORDINARY N x K register semantics (the fold lives only in the load layer), so the
  // K-permutation is TileShape.K when folding -- NOT the 32B-run span, which would keep the fragment in the folded
  // (N/FoldF) x (FoldF*K) form and force a 2-pass mainloop.
  // (e) ONE definition, shared with the offline generators -- see MixGemmMmaPermK in
  // fast_numeric_conversion_for_mix_gemm.h for why restating it broke a folded plane.
  static constexpr int ArtifactLowFold =
      fold_schedule_traits<KernelScheduleType>::ArtifactLowFold > 0
          ? fold_schedule_traits<KernelScheduleType>::ArtifactLowFold : 1;
  static constexpr int ExplicitArtifactHighFold =
      fold_schedule_traits<KernelScheduleType>::ArtifactHighFold;
  // Carry-only in #37's first phase. ArtifactTileK is the resident delivery width; blockK below remains the tactic
  // TileShape.K until FullBlockK and CopyBlockK are split in the next reviewed phase. A zero value preserves direct
  // CollectiveBuilder callers whose legacy KernelAiuFold spelling predates an explicit artifact contract.
  static constexpr int ArtifactTileK =
      fold_schedule_traits<KernelScheduleType>::ArtifactTileK;
  static constexpr int MmaPermK =
      cutlass::MixGemmMmaPermK<sizeof_bits<RealInternalElementB>::value,
                               cute::get<2>(TileShape_MNK{}),
                               ArtifactLowFold>::value;
  using TiledMma = typename quactlize_detail::get_tiled_mma<
        Arch, ElementMma, ElementMma, ElementAccumulator, TileShape_MNK, ClusterShape_MNK,
        Int<MmaPermK>>::TiledMma;

  static constexpr int PipelineStages = quactlize_ppu_detail::compute_stage_count_or_override<quactlize_ppu_detail::ppu10000_smem_capacity_bytes,
      ElementMma, ElementMma, TileShape_MNK>(StageCountType{});

  // currently only support k-major
  static_assert(cute::is_same_v<GmemLayoutA_, cutlass::layout::RowMajor> || cute::is_same_v<GmemLayoutA_, cutlass::layout::RowMajorInterleaved<256>>,
      "invalid GmemLayoutA, currently only support k-major or k256-major");
  static_assert(cute::is_same_v<GmemLayoutB_, cutlass::layout::ColumnMajor> || cute::is_same_v<GmemLayoutB_, cutlass::layout::ColumnMajorInterleaved<256>>,
      "invalid GmemLayoutB, currently only support k-major or k256-major");

  using kContinousA = cute::conditional_t<cute::is_same_v<GmemLayoutA_, cutlass::layout::RowMajorInterleaved<256>>, Int<256>, Int<1>>;
  using kContinousB = cute::conditional_t<cute::is_same_v<GmemLayoutB_, cutlass::layout::ColumnMajorInterleaved<256>>, Int<256>, Int<1>>;
  using kContinous = cute::conditional_t<IsATransformed, kContinousA, kContinousB>;
  // ---- B BIT-PLANE CONCAT: a 4th member in the B element tuple (tuple<ElementB,Scale,Zero,PlaneB2>) routes to
  // the dedicated TWO-plane mainloop. Absent => void => everything below degenerates to the single-plane build
  // BIT-IDENTICALLY (same policy, same atoms, no BPlanes wrapper).
  // NOTE PermutionK further down uses sizeof_bits<RealInternalElementB>, i.e. the LOW plane -- which is exactly
  // right: the low plane drives the main swzl and tCrB_mma; the high plane only feeds extra bits to the converter.
  using PlaneB2 = detail::deduce_mixed_width_dtype_t<3, ElementPairB>;
  static constexpr bool HasPlane2 = !cute::is_void_v<PlaneB2>;

  // ArtifactLowFold comes from the resident byte layout. It must never be re-derived from blockK: blockK is the
  // tactic's TileK, and a larger-tile tactic deliberately reads the same folded artifact. A fold of one keeps the
  // ordinary low-plane collective even when an independently folded high plane supplied the wrapper.
  static constexpr int FoldF = ArtifactLowFold;  // compatibility inside the existing collective API
  static constexpr bool HasFold = ArtifactLowFold > 1;
  using BaseSchedule = typename fold_schedule_traits<KernelScheduleType>::Base;   // == KernelScheduleType if no fold

  // HasPlane2 must WIN over HasFold. It used to be the other way round, which meant a 2-plane build whose LOW plane
  // needs a fold (int2 at Block_K=64 -> F1=2) was routed to the single-plane fold collective and plane 2 was silently
  // dropped. Both artifact folds enter the operand atoms below; the 2-plane collective reads their ratio back from the
  // two layouts. BaseSchedule keeps the group-size schedule, while MmaPermK follows the low plane because that plane
  // owns the shared MMA fragment.
  using DispatchPolicy = cute::conditional_t<HasPlane2,
      MainloopPPUAiuMixedInput2Plane<PipelineStages, kContinous, BaseSchedule>,
      cute::conditional_t<HasFold,
          MainloopPPUAiuFold<PipelineStages, kContinous, (HasFold ? FoldF : 2), BaseSchedule>,
          MainloopQuactlizeMixedInput<PipelineStages, kContinous, BaseSchedule>>>;

  using GmemLayoutA = cutlass::layout::RowMajor;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

#if ENABLE_AIU
  static constexpr int blockM = cute::get<0>(TileShape_MNK{});
  static constexpr int blockN = cute::get<1>(TileShape_MNK{});
  static constexpr int blockK = cute::get<2>(TileShape_MNK{});

  // N-FOLD: the B plane's AIU contiguous run folds FoldF adjacent N-cols x blockK each, so its operand Block_K =
  // FoldF*blockK (=> AiuContElemSize = FoldF*blockK, reusing a validated config, e.g. int2 blockK=64 FoldF=2 => 128
  // == int2@TK128). A stays blockK. The collective's fold-in-N SmemLayoutB then presents this as (FoldF*Ng, blockK).
  static constexpr int BFoldBlockK = ArtifactLowFold * blockK;
  // ...and the PHYSICAL row count halves/quarters correspondingly: folding FoldF N-columns into one contiguous run
  // means the B tile physically has blockN/FoldF rows of FoldF*blockK each (same total bytes). Folding only K while
  // leaving Block_MN=blockN makes the swzl atom address a 2x-too-large tile per stage -> "TSM out of range" at
  // runtime (observed: tsm.ld.swzl stepping 0x800 through a 0x400-per-stage buffer).
  static_assert(blockN % ArtifactLowFold == 0, "artifact low fold must divide Block_N");
  static constexpr int BFoldBlockN = blockN / ArtifactLowFold;
  // (MOEG_FOLD_DEBUG dump removed -- it confirmed fold_dbg<64,64,64,2,32,128>: builder params are CORRECT,
  //  i.e. B operand gets Block_MN=32 / Block_K=128 -> swzl CUBE <32,32> -> 1024B/stage, matching SmemLayoutB.)
  using DefaultOperandA = quactlize_detail::MixGemm_AIU_Operand<RealInternalElementA, false, Int<blockM>, Int<blockK>, true>;
  using DefaultOperandB = quactlize_detail::MixGemm_AIU_Operand<RealInternalElementB, false, Int<BFoldBlockN>, Int<BFoldBlockK>, true>;
#elif 0 // async_cp not work now
  static_assert(false, "async_cp not work now");
  using DispatchPolicy = MainloopQuactlizeMixedInput<PipelineStages, kContinous, KernelScheduleType>;
  using DefaultOperandA = detail::DefaultGemm_TensorOpPPU_OperandA<
    RealInternalElementA, GmemLayoutA, cute::conditional_t<IsATransformed, AlignmentA, AlignmentB>, 32>;
  using DefaultOperandB = detail::DefaultGemm_TensorOpPPU_OperandB<
    RealInternalElementB, GmemLayoutB, cute::conditional_t<IsATransformed, AlignmentB, AlignmentA>, 32>;
#endif
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B plane 0 = the LOW plane
  using SmemLayoutAtomB0 = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB0 = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB0 = typename DefaultOperandB::GmemTiledCopy;

  // B plane 1 = the HIGH plane. Both planes share ONE tile (so the tile is bounded below by the SPARSEST plane's
  // AIU 32B minimum: int1 => Block_K>=256, int2 => >=128); only the element width differs, so plane 1's AIU/swzl
  // config comes out with the matching 2x/4x-smaller byte extent automatically. The fallback element keeps this
  // well-formed (and unused) in single-plane builds.
  // NOTE Block_K here must ALSO be the folded one (BFoldBlockK): even in single-plane builds this type gets
  // instantiated (it feeds the unused BPlanes fallback), so with a fold the plain blockK would give a sub-32B
  // contiguous run and trip MixGemm_AIU_Operand's `BlockContSize % 32 == 0` static_assert.
  // PER-PLANE ARTIFACT FOLD. BFoldBlockK already contains ArtifactLowFold. The second operand needs the remaining
  // ratio ArtifactHighFold/ArtifactLowFold, independent of the tactic blockK. Re-deriving it from blockK made a
  // large-TileK consumer silently reinterpret a small-TileK artifact, and was already wrong for Q3 at artifact TK64
  // where the two physical planes are F_low=2 and F_high=4.
  using P2Elem = cute::conditional_t<HasPlane2, PlaneB2, RealInternalElementB>;
  static constexpr int P2Contig = BFoldBlockK * cutlass::sizeof_bits<P2Elem>::value / 8;  // bytes AFTER plane 1's fold
  // Direct CollectiveBuilder users predating the shared policy carry no explicit high fold. Preserve their legacy
  // tactic-derived result; every quactlize two-plane policy supplies the explicit artifact value.
  static constexpr int LegacyP2Fold = P2Contig >= 32 ? 1 : 32 / P2Contig;
  static constexpr int ArtifactHighFold = ExplicitArtifactHighFold > 0
      ? ExplicitArtifactHighFold : ArtifactLowFold * LegacyP2Fold;
  static_assert(!HasPlane2 || ArtifactHighFold >= ArtifactLowFold,
                "artifact high fold cannot be smaller than the low fold");
  static_assert(!HasPlane2 || ArtifactHighFold % ArtifactLowFold == 0,
                "artifact plane folds must form an integral physical-tiler ratio");
  static constexpr int P2Fold = HasPlane2 ? ArtifactHighFold / ArtifactLowFold : 1;
  static_assert(P2Contig * P2Fold >= 32 || !HasPlane2,
      "artifact high fold cannot reach the AIU 32 B contiguous minimum at this tactic Block_K");
  static_assert(BFoldBlockN % P2Fold == 0 || !HasPlane2, "plane 2's fold must divide Block_N");
  using DefaultOperandB2 = quactlize_detail::MixGemm_AIU_Operand<
      P2Elem, false, Int<BFoldBlockN / P2Fold>, Int<BFoldBlockK * P2Fold>, true>;

  // Both planes' atoms ride the EXISTING single template params (CollectiveMma's parameter list is fixed by its
  // primary template). collective::BPlanes is the marker -- NOT cute::is_tuple, since a cute Layout is itself
  // tuple-like and would false-positive on SmemLayoutAtomB.
  using SmemLayoutAtomB = cute::conditional_t<HasPlane2,
      collective::BPlanes<SmemLayoutAtomB0, typename DefaultOperandB2::SmemLayoutAtom>, SmemLayoutAtomB0>;
  using SmemCopyAtomB = cute::conditional_t<HasPlane2,
      collective::BPlanes<SmemCopyAtomB0, typename DefaultOperandB2::SmemCopyAtom>, SmemCopyAtomB0>;
  using GmemTiledCopyB = cute::conditional_t<HasPlane2,
      collective::BPlanes<GmemTiledCopyB0, typename DefaultOperandB2::GmemTiledCopy>, GmemTiledCopyB0>;

  // Mainloop
  using CollectiveOp = collective::CollectiveMma<
    Arch, DispatchPolicy, TileShapePair_,
    ElementPairA, TagToStrideA_t<GmemLayoutA>,
    ElementPairB, TagToStrideB_t<GmemLayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

};

// AIU GEMM for Batch Array

} // namespace cutlass::gemm::collective
