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
#include "actlize_extensions/cutlass/gemm/quactlize_dispatch_policy.hpp"
#include "actlize_extensions/cutlass/detail/quactlize_mixed_dtype.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_a_pack.hpp"
#include "q4_kpack4_offline.hpp"
#include "kquant_kpack_offline.hpp"

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
// Block_K is the FULL physical K span covered by the tactic. ContigShape_ independently describes one resident
// artifact copy/swizzle quantum. void (default) keeps the historical rule CopyBlockK == FullBlockK. A folded artifact
// passes Shape<Int<FoldF>, Int<ArtifactTileK>>, so size(ContigShape_) is F*A even when Block_K is F*T. The operand can
// then tile the full tactic span with several identical artifact-sized cubes instead of silently rebuilding the cube
// from T. Every per-cube quantity (AiuContElemSize, bits_per_aiu, swzl CUBE_W) follows the shape; InstNum follows the
// full span divided by that quantum.
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
  bool Swap,
  typename ContigShape_
> struct MixGemm_AIU_Operand<
  Element,
  false,
  Block_MN,
  Block_K,
  Swap,
  ContigShape_
> {
  static constexpr int CopyContigElems = aiu_detail::contig_elems<Block_K, ContigShape_>::value;
  static_assert(CopyContigElems > 0 &&
                    (CopyContigElems * sizeof_bits<Element>::value) % 8 == 0,
                "artifact copy span must contain a positive whole-byte payload");
  static_assert(Block_K{} % CopyContigElems == 0,
                "artifact copy span must evenly tile the full tactic span");
  static constexpr int BlockContSize = CopyContigElems * sizeof_bits<Element>::value / 8;
  static_assert(BlockContSize % 32 == 0, "aiu_trans: block contiguous size should be multiple of 32B");
  static_assert(BlockContSize > 128 ? (BlockContSize % 128 == 0) : (BlockContSize % 32 == 0), "aiu_trans: block contiguous size should be multiple of 128B or 32B");
  static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  using AiuContElemSize = Int<AiuContByteSize / sizeof_bits<Element>::value * 8>;
  static_assert(Block_K{} % AiuContElemSize{} == 0,
                "AIU copy quantum must evenly tile the full tactic span");
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
  static constexpr int kCubePitchA = 0;    // 0 = natural CUBE_H * CUBE_W
  static constexpr int kStagePitchA = 0;   // 0 = CubePitch * InstNum
  using SmemCopyOp = PPU0010_TSM_LD_SWZL<Element, Block_MN{}, AiuContElemSize{}, Swap, false, InstNum, kCubePitchA>;
  using SmemCopyAtom = Copy_Atom<SmemCopyOp, Element>;
  // m8's A fragment has two registers per lane, but the physical swzl instruction always produces x4.  The
  // projected atom contains x4 in a private temporary and publishes v0/v1; selecting the ordinary atom for m8
  // would write two registers past the fragment even though the physical cube geometry is otherwise correct.
  using SmemCopyOpM8 = PPU0010_TSM_LD_SWZL_M8<
      Element, Block_MN{}, AiuContElemSize{}, Swap, false, InstNum,
      kCubePitchA, kStagePitchA>;
  using SmemCopyAtomM8 = Copy_Atom<SmemCopyOpM8, Element>;
  using SmemLayoutAtom = Layout<Shape<_8, AiuContElemSize>, Stride<AiuContElemSize, _1>>;
};

// Compile-time packed-row A variant used only by the independent dense M==1 schedule.  The physical write/read
// cube remains Block_MN=16 and the projected m8 read remains x4->x2. Cube bases stay compressed, while stage bases
// retain complete physical x4 footprints. Keeping this separate leaves every ordinary operand instantiation untouched.
template <
  int Rows,
  typename Element,
  typename Block_MN,
  typename Block_K,
  bool Swap
> struct MixGemm_AIU_OperandPackedA {
  using Ordinary = MixGemm_AIU_Operand<Element, false, Block_MN, Block_K, Swap>;
  static_assert(cute::is_same_v<Element, cutlass::half_t>,
                "packed-row A currently supports the shipping fp16 A operand only");
  static_assert(Rows == 1 && Block_MN{} == 16,
                "dense M==1 packed A retains the physical 16-row PPU0010 cube");

  using AiuContElemSize = typename Ordinary::AiuContElemSize;
  static constexpr int InstNum = Ordinary::InstNum;
  static constexpr int kCubePitchA = detail::aPackPitchForRows(Rows);
  static constexpr int kStagePitchA = detail::aPackStagePitchHalfs(
      kCubePitchA, InstNum, Block_MN{} * AiuContElemSize{});
  using GmemTiledCopy = typename Ordinary::GmemTiledCopy;
  using SmemCopyOp = PPU0010_TSM_LD_SWZL<
      Element, Block_MN{}, AiuContElemSize{}, Swap, false, InstNum, kCubePitchA>;
  using SmemCopyAtom = Copy_Atom<SmemCopyOp, Element>;
  using SmemCopyOpM8 = PPU0010_TSM_LD_SWZL_M8<
      Element, Block_MN{}, AiuContElemSize{}, Swap, false, InstNum,
      kCubePitchA, kStagePitchA>;
  using SmemCopyAtomM8 = Copy_Atom<SmemCopyOpM8, Element>;
  using SmemLayoutAtom = typename Ordinary::SmemLayoutAtom;
};

template <
  typename Block_MN,
  typename Block_K,
  bool Swap,
  typename ContigShape_
> struct MixGemm_AIU_Operand<
  cutlass::int4b_t,
  false,
  Block_MN,
  Block_K,
  Swap,
  ContigShape_
> {
  static constexpr int CopyContigElems = aiu_detail::contig_elems<Block_K, ContigShape_>::value;
  static_assert(CopyContigElems > 0 &&
                    (CopyContigElems * sizeof_bits<cutlass::int4b_t>::value) % 8 == 0,
                "int4 artifact copy span must contain a positive whole-byte payload");
  static_assert(Block_K{} % CopyContigElems == 0,
                "artifact copy span must evenly tile the full tactic span");
  static constexpr int BlockContSize = CopyContigElems * sizeof_bits<cutlass::int4b_t>::value / 8;
  static_assert(BlockContSize % 32 == 0, "aiu int4: artifact copy span must be a multiple of 32B");
  static_assert(BlockContSize > 128 ? (BlockContSize % 128 == 0) : (BlockContSize % 32 == 0), "aiu_trans: block contiguous size should be multiple of 128B or 32B");
  static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  using AiuContElemSize = Int<AiuContByteSize / sizeof_bits<cutlass::int4b_t>::value * 8>;
  static_assert(Block_K{} % AiuContElemSize{} == 0,
                "AIU copy quantum must evenly tile the full tactic span");
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
  bool Swap,
  typename ContigShape_
> struct MixGemm_AIU_Operand<
  cutlass::uint2b_t,
  false,
  Block_MN,
  Block_K,
  Swap,
  ContigShape_
> {
  static constexpr int CopyContigElems = aiu_detail::contig_elems<Block_K, ContigShape_>::value;
  static_assert(CopyContigElems > 0 &&
                    (CopyContigElems * sizeof_bits<cutlass::uint2b_t>::value) % 8 == 0,
                "int2 artifact copy span must contain a positive whole-byte payload");
  static_assert(Block_K{} % CopyContigElems == 0,
                "artifact copy span must evenly tile the full tactic span");
  static constexpr int BlockContSize = CopyContigElems * sizeof_bits<cutlass::uint2b_t>::value / 8;
  static_assert(BlockContSize % 32 == 0, "aiu int2: artifact copy span must be a multiple of 32B");
  static_assert(BlockContSize > 128 ? (BlockContSize % 128 == 0) : (BlockContSize % 32 == 0), "aiu w2: 128B or 32B");
  static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  using AiuContElemSize = Int<AiuContByteSize / sizeof_bits<cutlass::uint2b_t>::value * 8>;      // AiuContByteSize*4
  static_assert(Block_K{} % AiuContElemSize{} == 0,
                "AIU copy quantum must evenly tile the full tactic span");
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
// swzl element count is AiuContElemSize/8. The artifact copy span (not the full tactic span) must contain a whole
// number of 32 B runs; several such runs may tile one larger tactic.
template <
  typename Block_MN,
  typename Block_K,
  bool Swap,
  typename ContigShape_
> struct MixGemm_AIU_Operand<
  cutlass::uint1b_t,
  false,
  Block_MN,
  Block_K,
  Swap,
  ContigShape_
> {
  static constexpr int CopyContigElems = aiu_detail::contig_elems<Block_K, ContigShape_>::value;
  static_assert(CopyContigElems > 0 &&
                    (CopyContigElems * sizeof_bits<cutlass::uint1b_t>::value) % 8 == 0,
                "int1 artifact copy span must contain a positive whole-byte payload");
  static_assert(Block_K{} % CopyContigElems == 0,
                "artifact copy span must evenly tile the full tactic span");
  static constexpr int BlockContSize = CopyContigElems * sizeof_bits<cutlass::uint1b_t>::value / 8;
  static_assert(BlockContSize % 32 == 0, "aiu int1: artifact copy span must be a multiple of 32B");
  static_assert(BlockContSize > 128 ? (BlockContSize % 128 == 0) : (BlockContSize % 32 == 0), "aiu w1: 128B or 32B");
  static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  using AiuContElemSize = Int<AiuContByteSize / sizeof_bits<cutlass::uint1b_t>::value * 8>;      // AiuContByteSize*8
  static_assert(Block_K{} % AiuContElemSize{} == 0,
                "AIU copy quantum must evenly tile the full tactic span");
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

// One converter-native code plane transported as opaque b16 words through the
// already-shipping transposed fp16 AIU/TSM pair.  Pack is 16/plane_bits, so a
// logical TK maps to TK/Pack physical b16 rows.  The transport never performs
// fp16 arithmetic; the collective recasts the registers to the original
// sub-byte type before invoking the existing converter.
template <int Pack, typename Arch, typename Block_N, typename LogicalBlock_K,
          bool Swap, int ScheduledDeliveryN = 0>
struct MixGemm_KPack_Transpose_Operand {
  static_assert(cute::is_same_v<Arch, cutlass::arch::PPU0010>,
                "K-pack transport is bound to the proved PPU0010 b16 pair");
  static_assert(Pack == 4 || Pack == 8 || Pack == 16,
                "b16 K-pack supports int4/int2/int1 planes");
  static_assert(Block_N{} % kquant_kpack::kTransportN == 0,
                "K-pack transposed N must compose whole N16 reads");
  static_assert(LogicalBlock_K{} >= kquant_kpack::kReaderPhysicalK * Pack &&
                    LogicalBlock_K{} % (kquant_kpack::kReaderPhysicalK * Pack) == 0,
                "logical TK must contain whole physical K16 b16 reads");
  using TransportElement = cutlass::half_t;
  using PhysicalBlockK = Int<LogicalBlock_K{} / Pack>;
  static constexpr int AutoDeliveryN = Block_N{} < 64 ? Block_N{} : 64;
  static constexpr int DeliveryN =
      ScheduledDeliveryN > 0
          ? (Block_N{} < ScheduledDeliveryN ? Block_N{} : ScheduledDeliveryN)
          : AutoDeliveryN;
  static_assert(DeliveryN == 16 || DeliveryN == 32 || DeliveryN == 64,
                "K-pack resolved delivery N is 16, 32 or 64");
  static_assert(Block_N{} % DeliveryN == 0,
                "K-pack delivery N must exactly tile tactic N");
  static constexpr int InstNum = Block_N{} / DeliveryN;

  // The offline bytes remain canonical [K/Pack,N] b16 for every DeliveryN.
  // DeliveryN changes only the matched AIU/TSM resident cube:
  //   64 -> one 8 KiB cube, 128 B physical-K row pitch;
  //   32 -> two 4 KiB cubes,  64 B physical-K row pitch;
  //   16 -> four 2 KiB cubes, 32 B physical-K row pitch.
  // The complete stage remains Block_N*(TK/Pack)*sizeof(b16), so converter,
  // MMA, barriers and workspace ABI are independent of this scheduling axis.
  static constexpr int bits_per_aiu =
      DeliveryN * PhysicalBlockK{} * sizeof_bits<TransportElement>::value;
  using CopyInst = PPU0010_AIU_LOAD<
      cute::C<bits_per_aiu>, TransportElement, true>;
  using GmemTiledCopy = decltype(
      make_tiled_copy(Copy_Atom<CopyInst, TransportElement>{},
                      Layout<Shape<_1, _1>, Stride<_1, _1>>{},
                      Layout<Shape<Int<DeliveryN>, PhysicalBlockK>>{}));
  using SmemCopyOp = PPU0010_TSM_LD_SWZL<
      TransportElement, PhysicalBlockK{}, DeliveryN, Swap, true, InstNum>;
  using SmemCopyAtom = Copy_Atom<SmemCopyOp, TransportElement>;
  using SmemLayoutAtom = Layout<
      Shape<Int<DeliveryN>, PhysicalBlockK>,
      Stride<_1, Int<DeliveryN>>>;
  // This is the shipping transposed-b16 operand and deliberately keeps its
  // historical six-template-argument type identity.  Adding a provider
  // parameter here would change the type even when the default were selected.
  // A plain-shared/direct provider therefore owns a distinct physical operand;
  // the selector below may choose it only after its complete G2S/S2R chain is
  // admitted.  The legacy operand itself remains exactly bound to the matched
  // AIU-swzl/TSM-swzl endpoints.
  using BDeliveryTags =
      cutlass::gemm::collective::detail::quactlize_b_delivery::
          ProductionBDelivery;
  using StageSmemLayout = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Block_N{}, PhysicalBlockK{})));
  static constexpr std::size_t StageBytes =
      std::size_t(Block_N{}) * std::size_t(PhysicalBlockK{}) *
      sizeof_bits<TransportElement>::value / 8;
  using PhysicalSharedContract =
      cutlass::gemm::collective::detail::quactlize_b_delivery::
          PhysicalSharedContract<
              StageSmemLayout, TransportElement, StageBytes,
              kquant_kpack::kTransportN,
              kquant_kpack::kReaderPhysicalK * Pack, 32>;
  struct WriterBinding {
    using G2S = typename BDeliveryTags::G2S;
    using Shared = PhysicalSharedContract;
    using Copy = GmemTiledCopy;
  };
  struct ReaderBinding {
    using S2R = typename BDeliveryTags::S2R;
    using Shared = PhysicalSharedContract;
    using Copy = SmemCopyAtom;
  };
  using BDeliveryBinding =
      cutlass::gemm::collective::detail::quactlize_b_delivery::
          BoundBDelivery<WriterBinding, ReaderBinding>;
  static_assert(BDeliveryBinding::single_issuer,
                "the shipping K-pack AIU writer remains one opaque issuer");
  static_assert(cute::cosize_v<StageSmemLayout> *
                        sizeof_bits<TransportElement>::value / 8 ==
                    StageBytes,
                "the shipping K-pack shared layout must own exactly one stage");
  static constexpr int PhysicalK = PhysicalBlockK{};
  static constexpr int LogicalK = LogicalBlock_K{};
  static constexpr int CodesPerWord = Pack;
};

// Preserve the Q4 type name and all of its nested members as a source/type ABI.
template <typename Arch, typename Block_N, typename LogicalBlock_K, bool Swap,
          int ScheduledDeliveryN = 0>
struct MixGemm_Q4_KPack4_Transpose_Operand
    : MixGemm_KPack_Transpose_Operand<
          q4_kpack4::kPack, Arch, Block_N, LogicalBlock_K, Swap,
          ScheduledDeliveryN> {};

template <bool Enabled, int Pack, typename Arch, typename Block_N,
          typename LogicalBlock_K, bool Swap, int DeliveryN,
          typename LegacyOperand,
          typename BDeliveryTags =
              cutlass::gemm::collective::detail::quactlize_b_delivery::
                  ProductionBDelivery>
struct SelectKPackOperand {
  using type = LegacyOperand;
};
template <int Pack, typename Arch, typename Block_N,
          typename LogicalBlock_K, bool Swap, int DeliveryN,
          typename LegacyOperand, typename BDeliveryTags>
struct SelectKPackOperand<true, Pack, Arch, Block_N, LogicalBlock_K,
                          Swap, DeliveryN, LegacyOperand, BDeliveryTags> {
  static_assert(
      std::is_same_v<
          BDeliveryTags,
          cutlass::gemm::collective::detail::quactlize_b_delivery::
              ProductionBDelivery>,
      "plain/direct K-pack delivery requires its distinct physical operand "
      "selector; it cannot reinterpret the shipping transposed-b16 operand");
  using type = MixGemm_KPack_Transpose_Operand<
      Pack, Arch, Block_N, LogicalBlock_K, Swap, DeliveryN>;
};

template <bool Enabled, typename Arch, typename Block_N,
          typename LogicalBlock_K, bool Swap, int DeliveryN,
          typename LegacyOperand,
          typename BDeliveryTags =
              cutlass::gemm::collective::detail::quactlize_b_delivery::
                  ProductionBDelivery>
struct SelectQ4KPack4Operand {
  using type = LegacyOperand;
};
template <typename Arch, typename Block_N, typename LogicalBlock_K,
          bool Swap, int DeliveryN, typename LegacyOperand,
          typename BDeliveryTags>
struct SelectQ4KPack4Operand<true, Arch, Block_N, LogicalBlock_K,
                            Swap, DeliveryN, LegacyOperand, BDeliveryTags> {
  static_assert(
      std::is_same_v<
          BDeliveryTags,
          cutlass::gemm::collective::detail::quactlize_b_delivery::
              ProductionBDelivery>,
      "plain/direct Q4 delivery requires its distinct physical operand "
      "selector; it cannot reinterpret the shipping transposed-b16 operand");
  using type = MixGemm_Q4_KPack4_Transpose_Operand<
      Arch, Block_N, LogicalBlock_K, Swap, DeliveryN>;
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
  static constexpr int blockM = cute::get<0>(TileShape_MNK{});
  static constexpr int blockN = cute::get<1>(TileShape_MNK{});
  static constexpr int blockK = cute::get<2>(TileShape_MNK{});

  // User can configure custom warp tile shape through ClusterShape_MNK
  static constexpr bool CustomWarpShape = cute::get<0>(ClusterShape_MNK{}) != 1 ||
                                          cute::get<1>(ClusterShape_MNK{}) != 1 ||
                                          cute::get<2>(ClusterShape_MNK{}) != 1;
  static constexpr auto WarpShapeStage = MapBlockShapeToWarpShapeStage<Arch, ElementA, blockM, blockN, blockK>();
  static constexpr int requestedWarpM = CustomWarpShape
      ? cute::get<0>(ClusterShape_MNK{}) : cute::get<0>(WarpShapeStage);
  static constexpr int requestedWarpN = CustomWarpShape
      ? cute::get<1>(ClusterShape_MNK{}) : cute::get<1>(WarpShapeStage);

  // The m8 atom is a SHAPE refinement, not a dtype-wide replacement.  Resolve the requested warp before selecting
  // the instruction so the exact PPU0010 TM8/WM8 family can reach m8n16k16 while every established family remains
  // on m16n16k16.  This is the shipping CollectiveBuilder seam; #111 deliberately left it unwired while G0-G2
  // validated the raw atom and its physical x4 delivery.
  using MmaInst = typename config::GetMmaInstForShape<
      Arch, ElementA, ElementB, ElementAccumulator, blockM, requestedWarpM>::type;

  static constexpr int InstM = cute::get<0>(typename MMA_Traits<MmaInst>::Shape_MNK{});
  static constexpr int InstN = cute::get<1>(typename MMA_Traits<MmaInst>::Shape_MNK{});
  static constexpr int InstK = cute::get<2>(typename MMA_Traits<MmaInst>::Shape_MNK{});

  // Do not silently clamp an illegal WM8/m16 request to WM16.  The epilogue sees the requested WarpShape directly;
  // accepting different effective geometries on the two sides is a compile-success / wrong-result failure mode.
  static_assert(requestedWarpM >= InstM && requestedWarpM % InstM == 0,
                "requested WarpM must contain whole selected MMA atoms");
  static_assert(requestedWarpN >= InstN && requestedWarpN % InstN == 0,
                "requested WarpN must contain whole selected MMA atoms");
  static_assert(blockM % requestedWarpM == 0 && blockN % requestedWarpN == 0,
                "requested warp shape must evenly divide the CTA tile");
  static constexpr int warpM = requestedWarpM;
  static constexpr int warpN = requestedWarpN;

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
       cute::is_same_v<KernelScheduleType, KernelAiuMultistageMixedInputFinegrainedGs16> ||
       (fold_schedule_traits<KernelScheduleType>::ArtifactLowFold > 0) ||
       kpack_schedule_traits<KernelScheduleType>::Value)>   // physical provider contract
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

  // PermutionK controls the COMPLETE logical MMA fragment, not one artifact copy. Under an N-FOLD one resident 32B
  // run is FoldF N-cols x ArtifactTileK each, and a larger tactic concatenates T/ArtifactTileK such deliveries; the
  // fragment still sees the tactic's full TileShape.K.
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
  // ArtifactTileK is the resident delivery width. A zero value preserves direct CollectiveBuilder callers whose
  // legacy KernelAiuFold spelling predates an explicit artifact contract; for those callers A falls back to T.
  static constexpr int ArtifactTileK =
      fold_schedule_traits<KernelScheduleType>::ArtifactTileK;
  static constexpr int ScheduledAPackRows =
      a_provider_schedule_traits<KernelScheduleType>::Rows;
  static constexpr bool HasQ4KPack4 =
      q4_kpack4_schedule_traits<KernelScheduleType>::Value;
  static constexpr int Q4KPack4DeliveryN =
      q4_kpack4_schedule_traits<KernelScheduleType>::DeliveryN;
  static constexpr bool HasKPack =
      kpack_schedule_traits<KernelScheduleType>::Value;
  static constexpr int KPackLow =
      kpack_schedule_traits<KernelScheduleType>::LowPack;
  static constexpr int KPackHigh =
      kpack_schedule_traits<KernelScheduleType>::HighPack;
  static constexpr int KPackDeliveryN =
      kpack_schedule_traits<KernelScheduleType>::DeliveryN;
  using ScheduledBDelivery =
      typename kpack_schedule_traits<KernelScheduleType>::BDelivery;
  static constexpr int blockM = cute::get<0>(TileShape_MNK{});
  static constexpr int blockN = cute::get<1>(TileShape_MNK{});
  static constexpr int blockK = cute::get<2>(TileShape_MNK{});
  static constexpr int physicalBlockM = blockM < 16 ? 16 : blockM;
  static constexpr int MmaPermK =
      cutlass::MixGemmMmaPermK<sizeof_bits<RealInternalElementB>::value,
                               cute::get<2>(TileShape_MNK{}),
                               ArtifactLowFold>::value;
  using TiledMma = typename quactlize_detail::get_tiled_mma<
        Arch, ElementMma, ElementMma, ElementAccumulator, TileShape_MNK, ClusterShape_MNK,
        Int<MmaPermK>>::TiledMma;
  static constexpr int MmaInstM = size<0>(typename TiledMma::AtomShape_MNK{});
  static_assert(MmaInstM == 8 || MmaInstM == 16,
                "quactlize mixed-input builder supports only the ppu001 m8/m16 fp16 atom families");
  static_assert(MmaInstM != 8 || physicalBlockM == 16,
                "m8 must retain the physical 16-row A write/read cube");
  static_assert(ScheduledAPackRows == 0 ||
                    (ScheduledAPackRows == 1 && MmaInstM == 8 && physicalBlockM == 16),
                "typed packed-A schedule must be the exact physical16/logical8 M==1 provider");

  using PhysicalATileShape_MNK = Shape<Int<physicalBlockM>, Int<blockN>, Int<blockK>>;
  static constexpr int PipelineStages = quactlize_ppu_detail::compute_stage_count_or_override<quactlize_ppu_detail::ppu10000_smem_capacity_bytes,
      ElementMma, ElementMma, PhysicalATileShape_MNK>(StageCountType{});

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

  static constexpr int EffectiveArtifactTileK = ArtifactTileK > 0 ? ArtifactTileK : blockK;
  static_assert(EffectiveArtifactTileK > 0, "effective artifact TileK must be positive");
  static_assert(blockK % EffectiveArtifactTileK == 0,
                "ArtifactTileK must evenly tile the tactic TileK");

  // The high fold is an artifact property too. Legacy direct-builder spellings carry A=0/high-fold=0, so their
  // effective A is T and this reduces to the old tactic-derived fallback. Shipped policies bring both folds and A.
  using P2Elem = cute::conditional_t<HasPlane2, PlaneB2, RealInternalElementB>;
  static constexpr int P2ArtifactBytesBeforeExtraFold =
      ArtifactLowFold * EffectiveArtifactTileK * cutlass::sizeof_bits<P2Elem>::value / 8;
  static constexpr int LegacyP2ExtraFold =
      P2ArtifactBytesBeforeExtraFold >= 32 ? 1 : 32 / P2ArtifactBytesBeforeExtraFold;
  static constexpr int ArtifactHighFold = HasKPack ? 1 :
      (ExplicitArtifactHighFold > 0
          ? ExplicitArtifactHighFold : ArtifactLowFold * LegacyP2ExtraFold);
  static_assert(!HasPlane2 || ArtifactHighFold >= ArtifactLowFold,
                "artifact high fold cannot be smaller than the low fold");
  static_assert(!HasPlane2 || ArtifactHighFold % ArtifactLowFold == 0,
                "artifact plane folds must form an integral physical-tiler ratio");

  // ArtifactLowFold comes from the resident byte layout. It must never be re-derived from blockK: blockK is the
  // tactic's TileK, and a larger-tile tactic deliberately reads the same folded artifact. A fold of one keeps the
  // ordinary low-plane collective even when an independently folded high plane supplied the wrapper.
  static constexpr int FoldF = ArtifactLowFold;  // compatibility inside the existing collective API
  static constexpr bool HasFold = ArtifactLowFold > 1;
  static_assert(!HasKPack ||
                    (KPackLow * sizeof_bits<RealInternalElementB>::value == 16 &&
                     (!HasPlane2 ||
                        KPackHigh * sizeof_bits<P2Elem>::value == 16) &&
                     ArtifactLowFold == 1 && ArtifactHighFold == 1),
                "K-pack factors must exactly fill b16 and must not inherit an xplane fold");
  static_assert(!HasQ4KPack4 ||
                    (!HasPlane2 && KPackLow == 4 && KPackHigh == 0),
                "the historical Q4 wrapper remains the exact one-plane Pack4 path");
  using FoldBaseSchedule = typename fold_schedule_traits<KernelScheduleType>::Base;
  // The one-plane collective reads K-pack directly from its schedule wrapper.
  // The two-plane collective instead receives both pack factors explicitly in
  // its dispatch policy, so strip only that outer wrapper and retain the
  // underlying group-size (and optional A-provider) schedule.
  using BaseSchedule = cute::conditional_t<HasKPack && HasPlane2,
      typename kpack_schedule_traits<FoldBaseSchedule>::Wrapped,
      FoldBaseSchedule>;

  // HasPlane2 must WIN over HasFold. It used to be the other way round, which meant a 2-plane build whose LOW plane
  // needs a fold (int2 at Block_K=64 -> F1=2) was routed to the single-plane fold collective and plane 2 was silently
  // dropped. Both artifact folds enter the operand atoms below; the 2-plane collective reads their ratio back from the
  // two layouts. BaseSchedule keeps the group-size schedule, while MmaPermK follows the low plane because that plane
  // owns the shared MMA fragment.
  using DispatchPolicy = cute::conditional_t<HasPlane2,
      MainloopPPUAiuMixedInput2Plane<PipelineStages, kContinous, BaseSchedule,
                                    ArtifactLowFold, ArtifactHighFold, EffectiveArtifactTileK,
                                    (HasKPack ? KPackLow : 0),
                                    (HasKPack ? KPackHigh : 0),
                                    (HasKPack ? KPackDeliveryN : 0)>,
      cute::conditional_t<HasFold,
          MainloopPPUAiuFold<PipelineStages, kContinous, (HasFold ? FoldF : 2), BaseSchedule>,
          MainloopQuactlizeMixedInput<PipelineStages, kContinous, BaseSchedule>>>;

  using GmemLayoutA = cutlass::layout::RowMajor;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

#if ENABLE_AIU
  // TWO K SPANS, intentionally different when one resident artifact feeds a larger tactic:
  //   FullBlockK = F*T is the complete physical B extent consumed by this tactic tile.
  //   CopyBlockK = F*A is one resident artifact copy/swizzle quantum (32 B for the canonical folded artifacts).
  // Recomputing the second as F*T makes each delivery claim the entire tactic-wide cube. At T>A those deliveries
  // collide in the resident code-slot map instead of advancing through independent artifact cubes.
  static constexpr int FullBlockK = ArtifactLowFold * blockK;
  static constexpr int CopyBlockK = ArtifactLowFold * EffectiveArtifactTileK;
  static_assert(FullBlockK % CopyBlockK == 0,
                "artifact copy quantum must evenly tile the full folded tactic span");
  static constexpr int BFoldBlockK = FullBlockK;
  // ...and the PHYSICAL row count halves/quarters correspondingly: folding FoldF N-columns into one contiguous run
  // means the B tile physically has blockN/FoldF rows of FoldF*blockK each (same total bytes). Folding only K while
  // leaving Block_MN=blockN makes the swzl atom address a 2x-too-large tile per stage -> "TSM out of range" at
  // runtime (observed: tsm.ld.swzl stepping 0x800 through a 0x400-per-stage buffer).
  static_assert(blockN % ArtifactLowFold == 0, "artifact low fold must divide Block_N");
  static constexpr int BFoldBlockN = blockN / ArtifactLowFold;
  // (MOEG_FOLD_DEBUG dump removed -- it confirmed fold_dbg<64,64,64,2,32,128>: builder params are CORRECT,
  //  i.e. B operand gets Block_MN=32 / Block_K=128 -> swzl CUBE <32,32> -> 1024B/stage, matching SmemLayoutB.)
  // m8 is LOGICAL only.  PPU0010's AIU write and swzl read still operate on a physical 16-row cube; `.padz` fills
  // rows outside the real problem.  The collective exposes only the first logical eight rows to the MMA fragment,
  // but the descriptor, write payload and shared allocation all follow this physical operand.
  using OrdinaryOperandA = quactlize_detail::MixGemm_AIU_Operand<
      RealInternalElementA, false, Int<physicalBlockM>, Int<blockK>, true>;
  using DefaultOperandA = cute::conditional_t<ScheduledAPackRows == 0, OrdinaryOperandA,
      quactlize_detail::MixGemm_AIU_OperandPackedA<
          ScheduledAPackRows, RealInternalElementA, Int<physicalBlockM>, Int<blockK>, true>>;
  // ContigShape_ is the seam between the two spans: the operand keeps FullBlockK for its total extent and InstNum,
  // while size(ArtifactContigShape) supplies CopyBlockK to the per-cube AIU/swizzle derivation.
  using ArtifactContigShape = Shape<Int<ArtifactLowFold>, Int<EffectiveArtifactTileK>>;
  using LegacyOperandB = quactlize_detail::MixGemm_AIU_Operand<
      RealInternalElementB, false, Int<BFoldBlockN>, Int<FullBlockK>, true, ArtifactContigShape>;
  using DefaultOperandB = typename quactlize_detail::SelectKPackOperand<
      HasKPack, KPackLow, Arch, Int<blockN>, Int<blockK>, true,
      KPackDeliveryN, LegacyOperandB, ScheduledBDelivery>::type;
#elif 0 // async_cp not work now
  static_assert(false, "async_cp not work now");
  using DispatchPolicy = MainloopQuactlizeMixedInput<PipelineStages, kContinous, KernelScheduleType>;
  using DefaultOperandA = detail::DefaultGemm_TensorOpPPU_OperandA<
    RealInternalElementA, GmemLayoutA, cute::conditional_t<IsATransformed, AlignmentA, AlignmentB>, 32>;
  using DefaultOperandB = detail::DefaultGemm_TensorOpPPU_OperandB<
    RealInternalElementB, GmemLayoutB, cute::conditional_t<IsATransformed, AlignmentB, AlignmentA>, 32>;
#endif
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = cute::conditional_t<MmaInstM == 8,
      typename DefaultOperandA::SmemCopyAtomM8, typename DefaultOperandA::SmemCopyAtom>;
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
  static constexpr int P2Fold = HasPlane2 ? ArtifactHighFold / ArtifactLowFold : 1;
  // DefaultOperandB2 is instantiated as an unused fallback even for one-plane policies. In that case mirror the
  // already-valid low operand; an absent high plane has no independent high-fold contract to satisfy.
  static constexpr int OperandB2Fold = HasPlane2 ? ArtifactHighFold : ArtifactLowFold;
  static constexpr int FullBlockK2 = OperandB2Fold * blockK;
  static constexpr int CopyBlockK2 = OperandB2Fold * EffectiveArtifactTileK;
  static_assert(!HasPlane2 || (CopyBlockK2 * cutlass::sizeof_bits<P2Elem>::value / 8) % 32 == 0,
                "artifact high copy quantum must be an integral number of AIU 32 B deliveries");
  static_assert(!HasPlane2 || FullBlockK2 % CopyBlockK2 == 0,
                "artifact high copy quantum must evenly tile the full folded tactic span");
  static_assert(blockN % ArtifactHighFold == 0 || !HasPlane2,
                "artifact high fold must divide Block_N");
  using ArtifactContigShape2 = Shape<Int<OperandB2Fold>, Int<EffectiveArtifactTileK>>;
  using LegacyOperandB2 = quactlize_detail::MixGemm_AIU_Operand<
      P2Elem, false, Int<blockN / OperandB2Fold>, Int<FullBlockK2>, true, ArtifactContigShape2>;
  using DefaultOperandB2 = typename quactlize_detail::SelectKPackOperand<
      HasKPack && HasPlane2, (HasPlane2 ? KPackHigh : KPackLow),
      Arch, Int<blockN>, Int<blockK>, true, KPackDeliveryN,
      LegacyOperandB2, ScheduledBDelivery>::type;

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
