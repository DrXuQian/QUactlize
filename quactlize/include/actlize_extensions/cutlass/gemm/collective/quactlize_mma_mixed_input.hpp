/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once


#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
// quactlize's mainloop policies; this collective specialises CollectiveMma on one of them.
#include "actlize_extensions/cutlass/gemm/quactlize_dispatch_policy.hpp"
#include "actlize_extensions/cutlass/gguf_packed_scale.h"
#include "actlize_extensions/cutlass/quactlize_mix_gemm_convert.h"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

#include "cutlass/gemm/collective/collective_mma.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_metadata_policy.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_argument_contract.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_packed_metadata_ownership.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_a_schedule.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_pipeline.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_a_pack.hpp"
#include "q4_kpack4_offline.hpp"

#include "cutlass/detail/collective.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective::detail {
// Instantiates composition() ONLY when selected: naming both branches of a conditional_t instantiates both, and the
// unselected one fires cute's "Requires pow2 shape*stride".
// PER Scale_TileN, because the conflict map depends on it through the mma B operand's TV layout. A member template
// cannot be explicitly specialised inside the class, so the table lives here. A width with no entry gets the identity
// swizzle and keeps the plain map rather than pretending to be fixed.
// The pattern is closed-form -- Swizzle<2, 3, log2(TN) - 2> -- because the donor bits are always the group index's
// bits 1 and 2, and the group stride is TN halfs, so they sit at bit log2(TN)+1; MBase stays 3 so the low three bits
// are untouched and the 16 B cp.async keeps its contiguity. Written as explicit specialisations anyway, so the table
// claims exactly the three widths l98 measured and any other width gets the identity and keeps the plain map. A
// formula would have silently extended a result to widths nobody checked, which is how the first version of this
// table -- one entry, applied to all three widths -- left TN=32 at 4-way while advertising 1-way.
template <int TN> struct ScaleSwizzleFor      { using type = cute::Swizzle<0, 4, 4>; };   // identity
template <>       struct ScaleSwizzleFor<32>  { using type = cute::Swizzle<2, 3, 3>; };   // l98: 4-way -> 1-way
template <>       struct ScaleSwizzleFor<64>  { using type = cute::Swizzle<2, 3, 4>; };   // l98: 4-way -> 1-way
template <>       struct ScaleSwizzleFor<128> { using type = cute::Swizzle<2, 3, 5>; };   // l98: 4-way -> 1-way

template <bool On, class Swz, class L> struct MaybeScaleSwizzle { using type = L; };
template <class Swz, class L> struct MaybeScaleSwizzle<true, Swz, L> {
  using type = decltype(cute::composition(Swz{}, L{}));
};
}  // namespace cutlass::gemm::collective::detail

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Arch,
  int Stages,
  class kContinous,
  class KernelSchedule,
  class TileShapePair_,
  class ElementAOptionalTuple,
  class StrideA_,
  class ElementBOptionalTuple,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    Arch,
    MainloopQuactlizeMixedInput<Stages, kContinous, KernelSchedule>,
    TileShapePair_,
    ElementAOptionalTuple,
    StrideA_,
    ElementBOptionalTuple,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
private:
  enum class ConversionMode {
    DirectConvert,
    ConvertAndScale,
    ConvertAndScaleWithZero
  };
  using ScaleA = detail::deduce_mixed_width_dtype_t<1, ElementAOptionalTuple>;
  using ScaleB = detail::deduce_mixed_width_dtype_t<1, ElementBOptionalTuple>;
  using ZeroA = detail::deduce_mixed_width_dtype_t<2, ElementAOptionalTuple>;
  using ZeroB = detail::deduce_mixed_width_dtype_t<2, ElementBOptionalTuple>;
  using TileShape_Scale = detail::deduce_mixed_width_dtype_t<1, TileShapePair_>;
public:
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopQuactlizeMixedInput<Stages, kContinous, KernelSchedule>;
  static constexpr bool kQ4KPack4Transpose =
      q4_kpack4_schedule_traits<KernelSchedule>::Value;
  static constexpr int kQ4KPack4ScheduledDeliveryN =
      q4_kpack4_schedule_traits<KernelSchedule>::DeliveryN;
  using TileShape = detail::deduce_mixed_width_dtype_t<0, TileShapePair_>;
  using ScaleTileShape = cute::conditional_t<cute::is_void_v<TileShape_Scale>,
      decltype(make_shape(shape<1>(TileShape{}), Int<1>{})), TileShape_Scale>;
  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementAOptionalTuple>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementBOptionalTuple>;
  static constexpr bool IsATransformed = cute::is_tuple<ElementAOptionalTuple>::value;
  using ElementScale = cute::conditional_t<IsATransformed, ScaleA, ScaleB>;
  using ElementZero = cute::conditional_t<IsATransformed, ZeroA, ZeroB>;
  // For cases where we can't have a void type, we can use this to allow the code to compile when the scale / zero is void.
  using NonVoidElementScale = cute::conditional_t<cute::is_void_v<ElementScale>, float, ElementScale>;
  using NonVoidElementZero = cute::conditional_t<cute::is_void_v<ElementZero>, float, ElementZero>;

  using StrideA = StrideA_;
  using StrideB = StrideB_;
  // These are always MN major
  using StrideScale = cute::Stride<cute::Int<1>, int64_t, int64_t>;
  // For cases where we can't have a void scale, we can use this to allow the code to compile when the scale is void.
  using NonVoidStrideScale = cute::conditional_t<cute::is_void_v<StrideScale>, cute::Stride<_1, int64_t, int64_t>, StrideScale>;

  static_assert((IsATransformed && cutlass::gemm::detail::is_k_major<StrideA>()) ||
                (!IsATransformed && cutlass::gemm::detail::is_k_major<StrideB>()),
                "The transformed type must be K-major.");

  static_assert(( IsATransformed && (sizeof(ElementB) == 2)) ||
                (!IsATransformed && (sizeof(ElementA) == 2)) ||
                (cutlass::gemm::detail::is_k_major<StrideA>() &&
                 cutlass::gemm::detail::is_k_major<StrideB>()),
                "The unscaled element must be 2 bytes OR both inputs must be K-major");

  static_assert(cutlass::gemm::detail::is_mn_major<NonVoidStrideScale>(),
    "Scale must be MN major [Col Major if A is scaled, Row Major if B is scaled].");

  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;

  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;

  constexpr static int Scale_TileN = shape<0>(ScaleTileShape{});
  constexpr static int Scale_TileK = shape<1>(ScaleTileShape{});
  using MetadataPolicy = detail::MixedMetadataPolicy<ElementScale, Scale_TileK,
      !cute::is_void_v<ElementZero>, detail::FlatMetadataAddress>;
  using PipelineDriver = detail::MixedPipelineDriver;
  static_assert(Scale_TileK > 0,
                "ScaleTileShape.K must be positive; use ceil(TileK / group_size), not integer truncation");
  // Cap the copy's N-thread extent to the CTA and give each remaining slot more fixed-width atom iterations. The
  // shared plan asserts both slot capacity and complete-tile coverage; CuTe itself diagnoses neither truncation.
  constexpr static int Scale_NumThreads = size(TiledMma{});
  using ScaleCopyPlan = typename MetadataPolicy::template ScaleCopy<Scale_TileN, Scale_NumThreads>;
  using Scale_GmemCopyThrLayoutH = Int<ScaleCopyPlan::thread_layout_h>;
  using Scale_GmemCopyThrLayoutW = Int<ScaleCopyPlan::thread_layout_w>;
  using GmemTiledCopyScale = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, NonVoidElementScale>{},
                    Layout<Shape <Scale_GmemCopyThrLayoutH, Scale_GmemCopyThrLayoutW>>{},
                    Layout<Shape <Int<ScaleCopyPlan::values_per_thread>,_1>>{}));
  using GmemTiledCopyZero = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, NonVoidElementZero>{},
                    Layout<Shape <Scale_GmemCopyThrLayoutH, Scale_GmemCopyThrLayoutW>>{},
                    Layout<Shape <Int<ScaleCopyPlan::values_per_thread>,_1>>{}));

  // ---------------------------------------------------------------------------------------------------------------
  // THE NATIVE (PACKED) SCALE CHANNEL -- plan #20 option E. TYPES ONLY in this commit; nothing below consumes them yet,
  // and PPU_PACKED_SCALE off leaves every existing type byte-identical.
  //
  // The channel carries the gguf's own scale bytes instead of two pre-multiplied fp16 planes: for Q4_K one 16 B unit per
  // (superblock, column) holding d, dmin and the 8+8 six-bit codes, reordered offline into two self-contained 6-byte
  // halves so a k-tile covering half a superblock reads half the unit (`fold_derivation/l94`, PackBits layout).
  //
  // Every number here was measured locally, not assumed:
  //   * the tile is TN x 16 B = 2048 B at TN=128, i.e. THE SAME BYTES as today's scale tile alone -- the zero tile
  //     disappears, so the channel halves from 4.0 to 2.0 B per (group, column).
  //   * one thread per COLUMN, 16 B each: 16 B aligned, and consecutive threads on consecutive chunks, i.e. one fully
  //     coalesced burst, where today's (TN/8, SK) x (_8,_1) map is strided. l94 (6): contiguity/alignment/coalescing/
  //     coverage all 0 bad.
  //   * smem banks: 1-way on all 32, against 4-way on 4 today (l94 (2), counting DISTINCT addresses -- lanes sharing an
  //     address broadcast, and the scale fragment is deliberately k-broadcast). Config dependent: it holds because warp 0
  //     touches 8 CONSECUTIVE columns, so re-run l94 when the warp shape changes.
  //   * TileK=64 must stay on the fp16 path: 2 groups per tile would read 10 B, i.e. 5.0 B per (group, column), worse
  //     than fp16's 4.0. TileK >= 128 wins (2.5 B) and TileK == 256 is the best case (2.0 B).
#if defined(PPU_PACKED_FORMAT)
  static constexpr cutlass::gguf_packed::Fmt kPackedFmt = cutlass::gguf_packed::Fmt(PPU_PACKED_FORMAT);
#else
  static constexpr cutlass::gguf_packed::Fmt kPackedFmt = cutlass::gguf_packed::Fmt::Q4K;
#endif
  using PackedUnit = cutlass::gguf_packed::Unit<kPackedFmt>;
  // This collective has ONE weight plane. A format selector for Q3/Q5/Q6 must therefore leave every instantiation
  // here on fp16 metadata even when its low plane happens to have the same bit width as Q2/Q4. Besides preventing a
  // scale-pointer reinterpretation, the type-level match gives inactive formats a harmless legal staging type.
  static constexpr bool kPackedFormatMatchesElement =
      (kPackedFmt == cutlass::gguf_packed::Fmt::Q4K && std::is_same_v<ElementB, cutlass::int4b_t>) ||
      (kPackedFmt == cutlass::gguf_packed::Fmt::Q2K && std::is_same_v<ElementB, cutlass::uint2b_t>);
  static constexpr int kPackedScaleUnit = kPackedFormatMatchesElement ? PackedUnit::kUnitBytes : 16;
  using SmemLayoutScalePacked = Layout<Shape <Int<Scale_TileN>, Int<kPackedScaleUnit>>,
                                       Stride<Int<kPackedScaleUnit>, _1>>;
  // APPLICABILITY, NOT A REQUIREMENT. The ordinary Xplane path retains its historical one-superblock gate. K-pack4
  // has no fp16 metadata plane to fall back to, so its TK64/TK128 tactics consume integral two-/four-group runs from
  // the same native Q4_K unit. The unit is copied whole for this first closure; only the selected run is decoded into
  // the existing local (group,stage) fp16 planes. This changes neither the offline bytes nor the MMA read view.
  // THE FORMAT, AND EVERY CONSTANT DERIVED FROM IT. PPU_PACKED_FORMAT selects one of cutlass::gguf_packed::Fmt and
  // defaults to Q4_K, so a build that does not set it is byte-identical to what shipped. Before this the six numbers
  // below were literals -- 16 bytes per unit, Scale_TileK == 8, bias 0, has-min, ZMul 8, four 32-bit words -- and
  // schemes.py recorded the consequence: Q2_K is single-plane so the SHAPE fits and none of the constants did.
  //
  // The selected trait's unit size genuinely differs per format. Only Q4/Q2 can activate this one-plane collective;
  // a two-plane selector keeps an inert legal 16-byte staging type here and is consumed by the other collective.
  // Scale_TileK is the number of GROUPS a k-tile covers. The default path activates only for one complete
  // superblock. The K-pack4 schedule additionally admits a divisor of the superblock; tying that condition to
  // kQ4KPack4Transpose prevents this experiment from silently changing the established Xplane denominator.
  static constexpr bool kPackedTileDividesSb =
      int(Scale_TileK) > 0 && PackedUnit::kGroups % int(Scale_TileK) == 0;
  static constexpr bool kPackedKpack4Subtile =
      kQ4KPack4Transpose && kPackedFormatMatchesElement &&
      int(Scale_TileK) < PackedUnit::kGroups && kPackedTileDividesSb;
  static constexpr bool kPackedScaleOn =
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
      kPackedFormatMatchesElement &&
      ((int(Scale_TileK) == PackedUnit::kGroups) || kPackedKpack4Subtile);
#else
      false;
#endif
  static_assert(!kPackedScaleOn || kPackedTileDividesSb,
                "a packed single-plane K tile must cover an integral group run");
  static constexpr int kPackedTilesPerSb =
      kPackedKpack4Subtile ? PackedUnit::kGroups / int(Scale_TileK) : 1;
  static constexpr int kPackedTilesPerUnit =
      kPackedScaleOn ? PackedUnit::kSbPerUnit * kPackedTilesPerSb : 1;

  // THE STAGING TILE: the gguf's own bytes, cp.async'd in and decoded at the barrier. At Scale_TileK == 8 its size is
  // exactly one scale tile (TN*16 == TN*SK*2), so the channel goes from two smem tiles to three -- the price of keeping
  // the gmem side ASYNC. Without it the loader had load -> wait -> decode -> store serialised in one thread at the point
  // a cp.async used to be merely ISSUED, and that measured 24.17 us against a 20.2 baseline with an inner loop already
  // byte-identical to fp16.
  using SmemLayoutScaleRawStaged =
      Layout<Shape <Int<Scale_TileN>, Int<kPackedScaleUnit>, Int<DispatchPolicy::Stages>>,
             Stride<Int<kPackedScaleUnit>, _1, Int<Scale_TileN * kPackedScaleUnit>>>;

  // THE TRANSFER WIDTH FOLLOWS THE ACTIVE UNIT. A single uint128 per column is right for Q4; Q2's 20-byte unit is
  // five uint32 copies. Picking a width that does not divide would either drop bytes or read past the unit.
  // ppu.cp.async ACCEPTS 4, 8 OR 16 BYTES AND NOTHING ELSE (cute/arch/copy_ppu.hpp:262), so the width is the largest
  // of those that divides the unit: 16 for Q4_K and 4 for Q2_K. Q3/Q6's paired 28/36-byte transport lives in the
  // two-plane collective that owns their weight shapes, not in this file.
  using PackedMetadataOwnership = detail::PackedMetadataColumnOwnership<
      int(Scale_TileN), int(Scale_NumThreads)>;
  static constexpr int kPackedOwnerThreads = PackedMetadataOwnership::owner_threads;
  static constexpr int kPackedColsPerThread = PackedMetadataOwnership::columns_per_thread;
#if defined(PPU_MIXED_LEGACY_MODULO_METADATA_PUBLISHERS) && \
    (PPU_MIXED_LEGACY_MODULO_METADATA_PUBLISHERS != 0)
  // Must-red device control: reconstruct the historical all-thread modulo publication and omit its missing init edge.
  static constexpr bool kLegacyModuloMetadataPublishers = true;
#else
  static constexpr bool kLegacyModuloMetadataPublishers = false;
#endif
  // Keep one metadata column as the value tile. When TileN exceeds the owner
  // count, CuTe expresses the additional columns in the partition's rest
  // mode; widening the value tile would instead cross a column boundary
  // inside one copy atom and duplicate the rest-mode traversal.
  static constexpr int kPackedCopySpan = kPackedScaleUnit;
  static_assert(kPackedCopySpan % 4 == 0,
                "an active single-plane packed unit must divide into legal ppu.cp.async transfers");
  static_assert(int(Scale_TileN) % kPackedColsPerThread == 0,
                "the column tile must divide evenly among the copying threads");
  static constexpr int kPackedCopyBytes = (kPackedCopySpan % 16 == 0) ? 16
                                        : (kPackedCopySpan % 8  == 0) ? 8
                                        : (kPackedCopySpan % 4  == 0) ? 4 : 0;
  static_assert(kPackedCopyBytes != 0,
                "an active single-plane packed unit must divide into 4, 8 or 16-byte ppu.cp.async transfers");
  using PackedCopyElem =
      cute::conditional_t<kPackedCopyBytes == 16, cute::uint128_t,
      cute::conditional_t<kPackedCopyBytes == 8,  uint64_t, uint32_t>>;

  using GmemTiledCopyScalePacked = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<PackedCopyElem>, uint8_t>{},
                    Layout<Shape <Int<kPackedOwnerThreads>, _1>>{},  // the owner threads that physically exist
                    Layout<Shape <_1, Int<kPackedScaleUnit>>>{}));   // exactly one complete metadata column unit
  // The staged view, which is what SharedStorage actually holds: stage s starts at s * TN * 16 bytes.
  using SmemLayoutScalePackedStaged =
      Layout<Shape <Int<Scale_TileN>, Int<kPackedScaleUnit>, Int<DispatchPolicy::Stages>>,
             Stride<Int<kPackedScaleUnit>, _1, Int<Scale_TileN * kPackedScaleUnit>>>;

  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;

  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using SmemCopyAtomScale = Copy_Atom<cute::DefaultCopy, NonVoidElementScale>;

  // We must ensure the type to be scaled goes to RF
  static constexpr bool SwapAB = IsATransformed;
  using InternalSmemLayoutAtomA = cute::conditional_t<!SwapAB, SmemLayoutAtomA, SmemLayoutAtomB>;
  using InternalSmemLayoutAtomB = cute::conditional_t<!SwapAB, SmemLayoutAtomB, SmemLayoutAtomA>;
  using InternalSmemCopyAtomA   = cute::conditional_t<!SwapAB, SmemCopyAtomA, SmemCopyAtomB>;
  using InternalSmemCopyAtomB   = cute::conditional_t<!SwapAB, SmemCopyAtomB, SmemCopyAtomA>;
  static constexpr int kQ4KPack4ResolvedDeliveryN =
      kQ4KPack4Transpose ? int(size<0>(InternalSmemLayoutAtomB{})) : 0;
  static_assert(!kQ4KPack4Transpose ||
                    (kQ4KPack4ResolvedDeliveryN == 16 ||
                     kQ4KPack4ResolvedDeliveryN == 32 ||
                     kQ4KPack4ResolvedDeliveryN == 64),
                "K-pack4 collective must retain a named resident delivery N");
  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  // static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  // static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  // using ConvertedElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  // using ConvertedElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;
  // using InternalElementA = cute::conditional_t<!SwapAB, ConvertedElementA, ConvertedElementB>;
  // using InternalElementB = cute::conditional_t<!SwapAB, ConvertedElementB, ConvertedElementA>;

  using RealInternalElementA = cute::conditional_t<!SwapAB, ElementA, ElementB>;
  static constexpr int LogicalTileM = int(size<0>(TileShape{}));
  static constexpr int PhysicalATileM = LogicalTileM < 16 ? 16 : LogicalTileM;
  // Packed-A geometry, all read off fold_derivation/l84-l86 rather than extrapolated from row 0. Every real row
  // occupies four 16-half runs. l85 selects the first 128-B-aligned cube pitch with no live-writer collisions; l86
  // supplies the odd-cache-line half-run swap. The m8 CuTe atom publishes x2 but physically reads x4, so stages are
  // separated by the complete physical footprint instead of the compressed cube span. For logical m8, the authority
  // is the physical 16-row atom, never TileShape.M=8.
  static constexpr int kACubeH      = PhysicalATileM;                         // physical PPU0010 A cube height
  static constexpr int kACubeW      = 64;                                    // AiuContElemSize for fp16
  static constexpr int kASlices     = kACubeW / 16;                          // 8 words per slice
  static constexpr int kScheduledAPackRows =
      a_provider_schedule_traits<KernelSchedule>::Rows;
#if defined(PPU_A_PACK) && (PPU_A_PACK != 0)
  static constexpr int kLegacyAPackRows = PPU_A_PACK;
#else
  static constexpr int kLegacyAPackRows = 0;
#endif
  static_assert(kScheduledAPackRows == 0 || kLegacyAPackRows == 0,
                "typed packed-A schedule and legacy PPU_A_PACK cannot both own the A provider");
  static constexpr int kAPackRows =
      kScheduledAPackRows > 0 ? kScheduledAPackRows : kLegacyAPackRows;
  static constexpr bool kPackedA = kAPackRows > 0;
  static constexpr int kAPackGeometryRows = kPackedA ? kAPackRows : 1;
  // Every selected pitch is a 64-half/128-B multiple. Besides aligning every cube base, that makes the whole A span
  // a 128-B multiple so smem_b, which immediately follows it, retains the PPU0010 AIU load's required alignment.
  // The old tight row-0 pitch made smem_b start at a merely 16-B-aligned address and faulted on the box.
  static constexpr int kAPackPitch  = detail::aPackPitchForRows(kAPackGeometryRows); // halfs; model-derived R
  static constexpr int kACubes      = shape<2>(TileShape{}) / kACubeW;       // cubes per stage = InstNum
  static constexpr int kAPackStagePitch = detail::aPackStagePitchHalfs(
      kAPackPitch, kACubes, kACubeH * kACubeW);
  // Rounded up to 64 halfs so smem_b starts 128-B aligned whatever the cube geometry is.
  static constexpr int kAPackSpanRaw =
      kAPackStagePitch * (DispatchPolicy::Stages - 1) +
      kAPackPitch * (kACubes - 1) + kACubeH * kACubeW;
  static constexpr int kAPackSpan    = ((kAPackSpanRaw + 63) / 64) * 64;
  static constexpr int kAWrThreads   = kACubes * kAPackGeometryRows * kASlices * 2;
  // The run-start arithmetic is a single production authority in detail/ppu_a_pack.hpp. Odd cache lines swap the
  // two 4-word half-runs but do not change their union's start; copy_A_packed_rows applies that swap to h separately.
  // l85's live-writer collision check. The physical cross-stage read/write check lives in L186. Defined here,
  // ASSERTED in mma(): a static_assert in the class body calls a member of an
  // incomplete class, which EDG accepts and hgcc rejects with "no type named 'SharedStorage'".
  CUTLASS_HOST_DEVICE static constexpr bool aPackDisjoint() {
    int const n = kACubes * DispatchPolicy::Stages;
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j)
        for (int row_i = 0; row_i < kAPackGeometryRows; ++row_i)
          for (int row_j = 0; row_j < kAPackGeometryRows; ++row_j)
            for (int a = 0; a < kASlices; ++a)
              for (int b = 0; b < kASlices; ++b) {
                int const x = kAPackPitch * (i % kACubes) +
                    kAPackStagePitch * (i / kACubes) +
                    detail::aPackRunOffsetHalfs(kACubeH, row_i, a);
                int const y = kAPackPitch * (j % kACubes) +
                    kAPackStagePitch * (j / kACubes) +
                    detail::aPackRunOffsetHalfs(kACubeH, row_j, b);
                if (x < y + 16 && y < x + 16) return false;
              }
    return true;
  }
  using RealInternalElementB = cute::conditional_t<!SwapAB, ElementB, ElementA>;
  using BTransportElement = cute::conditional_t<
      kQ4KPack4Transpose, cutlass::half_t, RealInternalElementB>;
  static constexpr int PhysicalBTileK = kQ4KPack4Transpose
      ? int(size<2>(TileShape{})) / q4_kpack4::kPack
      : int(size<2>(TileShape{}));
  static_assert(!kQ4KPack4Transpose ||
                    (std::is_same_v<RealInternalElementB, cutlass::int4b_t> &&
                     int(size<2>(TileShape{})) % q4_kpack4::kTransportK == 0),
                "K-pack4 collective needs Q4 and a whole K64 transport");
  using InternalStrideA  = cute::conditional_t<!SwapAB, StrideA, StrideB>;
  using InternalStrideB  = cute::conditional_t<!SwapAB, StrideB, StrideA>;

  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using InternalTransformA  = cute::conditional_t<!SwapAB, TransformA, TransformB>;
  using InternalTransformB  = cute::conditional_t<!SwapAB, TransformB, TransformA>;
  using ArchTag = Arch;

  using SmemLayoutAtomScale = Layout<Shape<_8, _1>>;

  static_assert(rank(InternalSmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(InternalSmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(InternalSmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(InternalSmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(InternalSmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(InternalSmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomScale{}) == 2, "SmemLayoutAtomScale must be rank 2");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomScale{})) == 0, "SmemLayoutAtomScale must equal the tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomScale{})) == 0, "SmemLayoutAtomScale must evenly divide tile k shape.");

  // A's SMEM CANNOT BE MADE STRIDE-0 IN M -- TRIED, IT FAULTS. Recorded because the arithmetic is compelling
  // and someone (me) will otherwise try it again.
  //
  // The motivation: at decode every expert has ONE row while the original path used TileM=16, so 15/16 of that
  // tile was padding whose results the epilogue's residue mask discarded. SharedStorage sizes smem_a
  // by cosize_v<SmemLayoutA>, so a stride-0 M mode shrinks the allocation with no change there: 16,384 B ->
  // 1,024 B at (16,32,256) with 2 stages, block total 26,624 -> 11,264, 9 blocks/CU -> 23, 18 warps/CU -> 46,
  // theoretical occupancy 28% -> 72%.
  //
  // WHY IT FAILS, measured rather than inferred (fold_derivation/l74_swzl_coord_not_stride.cu). The mma-side read
  // is partition_S(make_mix_tensor_like(sA)), and a mix tensor carries (ptr, COORDINATE) rather than a resolved
  // pointer -- copy_unpack forwards src.data().coord_ to the asm, as the traits' own comment in
  // copy_traits_ppu0010_aiu.hpp states. Printing that coordinate for both layouts:
  //
  //     compact (16,256,2):(256,1,4096)   coord at (m,0,0) = (0,m,_0,0)   linear offsets 0 256 512 768
  //     bcast   (16,256,2):(  0,1, 256)   coord at (m,0,0) = (0,m,_0,0)   linear offsets 0   0   0   0
  //
  // The coordinates are IDENTICAL: m enters raw, unscaled by any stride. So the stride-0 layout altered exactly
  // the quantity this path does not use and left untouched the one it does, and the hardware still turned
  // coordinate m into base + m*(cube row pitch). On ppu001 that reads 16x past the 16x smaller allocation:
  //     tsm.ld.swzl.b32x4.s0.t1.trans0  vreg[64:67], [sreg63] @sreg27      (bases 512 B apart: 0xc00, 0xe00)
  // nvcc's front end accepted it with PPU_FORCE_INSTANTIATE and every static_assert passing, so the front end was
  // no evidence at all here.
  //
  // NO LAYOUT CHANGE CAN DO THIS. The fix has to pin the M COORDINATE, which is partition_S's output, not a
  // stride: copy(smem_tiled_copy_A, tCsA_p(_,0,k_block), tCrA_copy_view(_,i,k_block)) for each destination i.
  // Whether that is a saving at all depends on CPY_M = size<1>(tCrA_copy_view) exceeding 1 -- the box's three
  // tsm.ld.swzl at 512-byte-apart bases say it does, but that is an inference from three instructions.
  //
  // WHETHER IT IS WORTH DOING is now answerable, and the answer is probably not: the TileN ladder raised
  // warps/block 2 -> 8 at fixed total work and bought 1.066x within a single run (22.68 -> 21.28 us), so
  // occupancy is a weak lever for this kernel. 2.6x of theoretical occupancy would not be expected to behave
  // differently from 1.78x of it.
  // A's shared-memory tile retains a physical 16 x TileK x Stages floor even when the logical MMA tile is m8.
  // Thus the m8 path reduces issued work but does not halve A-smem. Two box faults and one
  // silent-NaN round established why: the allocation is sized by cosize_v<SmemLayoutA>, so a stride-0 M mode does
  // collapse it, but the swzl read is addressed by COORDINATE (fold_derivation/l74) and by its cube geometry, not
  // by these strides, so the instruction keeps sourcing TileM rows and reads out of bounds. Shrinking the cube to
  // match changes the permutation rather than the footprint and delivers the right bits to the wrong registers.
  //
  // Bypassing shared memory for A was also built and measured, and lost by 1.14-1.85x: the mma fragment needs 128
  // elements per thread per k-tile, which is ~64 vector loads against this atom's InstNum = 4, and A's 16x reuse
  // is BETWEEN threads, which only shared memory can serve. It has been removed again.
  //
  // Nor is there gmem traffic left to save. AiuDesc::init takes dim_h from the problem's M -- per expert in the
  // grouped kernel -- and the instruction is ...padz..., so at one row per expert the AIU already fetches exactly
  // one row per k-tile and zero-fills the rest of the cube.
  // THE LAYOUTS COME FROM THE SHARED HEADER. They are a pure function of (TileShape, Stages, atom) with no
  // reference to B, and all three collectives spelled the ordinary one identically -- so an A-side feature that
  // lives in one of them is an accident of where it was typed, not a property of the format.
  // m8 is a LOGICAL MMA tile over a PHYSICAL 16-row AIU cube.  Keep two views of the same storage:
  //   * SmemLayoutAPhysical owns/writes 16 x K x stages, so the raw x4 swzl read never crosses the allocation;
  //   * SmemLayoutA exposes only TileM rows to partition_fragment_A and the projected A2 copy atom.
  // The logical view's stage stride is the PHYSICAL stage span.  A compact 8-row staged layout would put stage i+1
  // halfway through stage i's cube -- a silent cross-stage alias even though its shape and cosize look plausible.
  using SmemLayoutACompact = decltype(tile_to_shape(
      InternalSmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
  using SmemLayoutAPhysicalStage = decltype(tile_to_shape(
      InternalSmemLayoutAtomA{},
      make_shape(Int<PhysicalATileM>{}, shape<2>(TileShape{}))));
  using SmemLayoutAPhysical = decltype(tile_to_shape(
      InternalSmemLayoutAtomA{},
      make_shape(Int<PhysicalATileM>{}, shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
  using SmemLayoutALogicalStage = decltype(tile_to_shape(
      InternalSmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}))));
  using SmemLayoutALogicalPitched = decltype(append(
      SmemLayoutALogicalStage{},
      make_layout(Int<DispatchPolicy::Stages>{}, Int<cute::cosize_v<SmemLayoutAPhysicalStage>>{})));
  using SmemLayoutA = cute::conditional_t<LogicalTileM == PhysicalATileM,
      SmemLayoutACompact, SmemLayoutALogicalPitched>;
  static_assert(size<0>(SmemLayoutA{}) == LogicalTileM,
                "A MMA view must expose exactly the logical TileM");
  static_assert(size<0>(SmemLayoutAPhysical{}) == PhysicalATileM,
                "A physical AIU view must retain the 16-row cube floor");
  static_assert(LogicalTileM != 8 ||
                    cute::cosize_v<SmemLayoutAPhysical> >= 16 * int(size<2>(TileShape{})) * DispatchPolicy::Stages,
                "TM8 must pay for complete 16-row A cubes in every stage");
  using SmemLayoutB = decltype(tile_to_shape(
      InternalSmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), Int<PhysicalBTileK>{},
                 Int<DispatchPolicy::Stages>{})));

  // It is assumed that the scales and zero-points share the same smem layout
  // PPU_SCALE_PAD: break the bank period on the GROUP stride.
  //
  // The natural layout is (Scale_TileN, Scale_TileK, Stages) : (1, Scale_TileN, Scale_TileN*Scale_TileK), and at
  // Scale_TileN = 64 halfs the group stride is 128 B -- exactly 32 banks x 4 B, so consecutive groups start on the
  // SAME bank. N is contiguous inside a group, so a single group covers the banks once and is fine; the conflicts
  // come from accesses that step in the group or stage direction. acu, sz against pc: +252k conflicts on +272k scale
  // reads, about 1.02 each, and they double the channel's transactions (+504k) while shared memory sits at 28% of
  // peak -- so this is transactions times latency, and padding is free apart from the extra bytes.
  //
  // Padding by 8 halfs (16 B) shifts each group by 4 banks. The data, the gmem->smem copy and every read all go
  // through this layout, so nothing else changes.
  // PER TileN, not one constant. The conflict map depends on Scale_TileN through the mma B operand's TV layout, and
  // fold_derivation/l98 sweeps each width against the collective's own layout: Swizzle<2,3,5> takes TN=128 from 4-way
  // to 1-way but leaves TN=32 at 4-way, i.e. it was overfit to the one width l98 originally hardcoded. The table below
  // is filled from that sweep; a width with no entry keeps the plain layout rather than pretending to be fixed.
  using ScaleSwizzleT = typename detail::ScaleSwizzleFor<int(shape<0>(ScaleTileShape{}))>::type;

#if defined(PPU_SCALE_PAD) && (PPU_SCALE_PAD > 0)
  static constexpr int kScalePad = PPU_SCALE_PAD;
  using SmemLayoutScale = decltype(make_layout(
    make_shape(shape<0>(ScaleTileShape{}), shape<1>(ScaleTileShape{}), Int<DispatchPolicy::Stages>{}),
    make_stride(_1{},
                Int<int(shape<0>(ScaleTileShape{})) + kScalePad>{},
                Int<(int(shape<0>(ScaleTileShape{})) + kScalePad) * int(shape<1>(ScaleTileShape{}))>{})));
#else
  // PPU_SCALE_SWIZZLE -- an XOR on the scale tile's address, chosen by sweeping the collective's OWN layout in
  // fold_derivation/l98 rather than derived here. Today's map is 4-way conflicted on 4 banks (l94 (2) and l98 agree);
  // Swizzle<2,3,5> takes it to 1-way on 16 banks. It moves two bits from position 8 -- inside the group field, since
  // the group stride is 128 halfs = bit 7 -- down to position 3, so it never touches the stage field and stays a
  // permutation of the allocation (cosize 2048 halfs is a power of two, which l98 checks).
  //
  // WHY A SWIZZLE AND NOT PADDING: PPU_SCALE_PAD added halfs to the group stride and LOST, because an additive pad
  // makes the address non-power-of-two and the multiply costs more than the conflict. An XOR is free.
  //
  // WHY IT IS WORTH TRYING, bounded by numbers already in TODO.md: SK_QUANT=0 prices the whole per-group scale reload
  // at 7.3%, and PPU_SCALE_PREFETCH -- which removes only the WAITING -- recovered 0.7%. So nine tenths of that
  // channel is work, not stall, and a 4-way conflict is work: four shared-pipe services for one instruction, which
  // prefetching provably cannot reach. This attacks the part prefetch left behind.
  //
  // IT MUST BE COMPOSED ON BOTH VIEWS. partition_extra_inputs builds sS with THIS layout while
  // partition_extra_mma_info builds a tensor also called sS with SmemCopyLayoutScale (n, 1, stage*Scale_TileK + g).
  // composition(Swz, L)(c) = Swz(L(c)), so the two stay equal iff they are equal unswizzled -- l98 (2) checks exactly
  // that, 0 bad over every coordinate, before and after. Swizzling one and not the other is the same class of bug as
  // the two-literals pitch that faulted with an invalid VA.
  //
  // APPLICABILITY, AND WHY IT IS A PARTIAL SPECIALISATION AND NOT conditional_t. cute requires a power-of-two
  // shape*stride to compose a swizzle, and this bench builds many units into one binary -- Stages = 3 alone makes
  // smem_scale_k = 3*Scale_TileK non-power-of-two. `conditional_t` does not help: BOTH branch types are instantiated
  // to be named, so the assert fires from the branch that was not taken. A partial specialisation instantiates only
  // the selected body. The local front-end check caught this on the first build; a static_assert here would instead
  // have failed the whole binary for the shapes that cannot carry it, which is the mistake PPU_SCALE_PREFETCH made.
  using PlainSmemLayoutScale = decltype(tile_to_shape(
    SmemLayoutAtomScale{},
    make_shape(shape<0>(ScaleTileShape{}), shape<1>(ScaleTileShape{}), Int<DispatchPolicy::Stages>{})));
  static constexpr bool kScaleSwizzleOkInner =
#if defined(PPU_SCALE_SWIZZLE) && (PPU_SCALE_SWIZZLE != 0)
      cute::is_static<PlainSmemLayoutScale>::value &&
      ((int(cute::cosize_v<PlainSmemLayoutScale>) & (int(cute::cosize_v<PlainSmemLayoutScale>) - 1)) == 0) &&
      ((int(DispatchPolicy::Stages) & (int(DispatchPolicy::Stages) - 1)) == 0) &&
      ((int(shape<0>(ScaleTileShape{})) & (int(shape<0>(ScaleTileShape{})) - 1)) == 0) &&
      ((int(shape<1>(ScaleTileShape{})) & (int(shape<1>(ScaleTileShape{})) - 1)) == 0);
#else
      false;
#endif
  using SmemLayoutScale = typename detail::MaybeScaleSwizzle<kScaleSwizzleOkInner, ScaleSwizzleT, PlainSmemLayoutScale>::type;
#endif
  // DECLARED IN BOTH BRANCHES, because partition_extra_mma_info uses them unconditionally. Putting them only in the
  // #else broke `PPU_SCALE_PAD=8` outright -- "identifier ScaleSwizzle is undefined" -- and the local front-end check
  // does not exercise that macro unless asked, so it shipped. Same shape as every other defect here: a definition and
  // its use governed by two different conditions.
  static constexpr bool kScaleSwizzleOk =
#if defined(PPU_SCALE_PAD) && (PPU_SCALE_PAD > 0)
      false;                       // padding already changes the map; the two are alternatives, not a stack
#else
      kScaleSwizzleOkInner;
#endif


  static_assert(DispatchPolicy::Stages >= 2, "CpAsync mainloop must have at least 2 stages in the pipeline.");

private:
  static constexpr ConversionMode
  get_conversion_mode() {
    if constexpr (cute::is_void_v<ElementScale>) {
      return ConversionMode::DirectConvert;
    }
    else if constexpr (cute::is_void_v<ElementZero>) {
      return ConversionMode::ConvertAndScale;
    }
    else {
      return ConversionMode::ConvertAndScaleWithZero;
    }
  }

  static constexpr ConversionMode KernelConversionMode = get_conversion_mode();
  static constexpr bool ModeHasScales = KernelConversionMode == ConversionMode::ConvertAndScale ||
                                        KernelConversionMode == ConversionMode::ConvertAndScaleWithZero;

  // PLACED HERE, AFTER KernelConversionMode, AND THAT IS NOT COSMETIC. This block first sat next to the swizzle
  // constants 70 lines above, where KernelConversionMode does not exist yet -- and because the offending conjunct
  // lives inside `#if defined(PPU_PACKED_SCALE_FUSED)`, EVERY build without the macro preprocessed it away and
  // compiled. The one configuration that used it was the only one that could not build, and nothing local covered
  // that configuration, so it reached the box as a define that quietly failed to apply. See ci/local_gates.py's
  // SYNTAX table, which now carries the macro, and l100_fused_active, which asserts the path is actually on.
  // -----------------------------------------------------------------------------------------------------------------
  // PPU_PACKED_SCALE_FUSED -- ONE INTERLEAVED (scale, zero) TILE INSTEAD OF TWO PLANES.
  //
  // WHAT IT IS FOR, and it is the STORE side only. The packed decoder writes `sS(n,G,st)` and `sZ(n,G,st)` as two
  // 16-bit stores; 32 lanes with consecutive n cover 32 adjacent 2-byte slots, which is 16 of the 32 four-byte banks
  // two deep, and stores cannot broadcast. acu measured the cost exactly: +73,728 conflicts against base, matching
  // `4 decoder warps x 8 groups x 2 planes x 9 passes x 128 CTAs` to the unit. Interleaving makes it ONE 32-bit store
  // whose 32 lanes hit all 32 banks once.
  //
  // WHY THIS ONE AND NOT THE OTHER TWO CANDIDATES, both already tried and both dead:
  //   * PAIRING ADJACENT COLUMNS to widen the store RACES. cp_async_wait is per thread, so between the wait and the
  //     publishing __syncthreads a thread may only read bytes it copied itself; the paired version read column 2p+1
  //     and rowC went to bad=128/4096, concentrated in odd columns. This is ownership-safe by construction: a thread
  //     derives BOTH halves from its OWN column's unit.
  //   * AN OFFLINE PERMUTATION of the scale tensor cannot do it at all. A reorder changes which VALUE sits at an
  //     address; a bank conflict is a property of the ADDRESSES the warp issues, which are unchanged -- and composing
  //     the permutation into the read view to keep it correct is exactly the runtime address arithmetic that made
  //     PPU_SCALE_SWIZZLE cost ~7% for zero conflicts removed.
  //
  // BYTES ARE UNCHANGED, in both memories. Stored bytes: the interleave is a pure rearrangement. Shared: scale 4 KiB
  // plus zero 4 KiB is 8 KiB either way -- the fused tile takes 2x the elements and the zero tile goes to zero, so
  // SharedStorage is the same size. Anyone expecting an occupancy gain here will not get one.
  //
  // THE READ SIDE IS DELIBERATELY UNTOUCHED. sS and sZ stay half-typed views over the fused buffer at offsets 0 and
  // 1 with every stride doubled, so all six tensors, the copy atom, the fragments and the four transform arms keep
  // their shapes and their code. The load bank map does not get worse: the current map puts 8 lanes on each of 4
  // banks in pairs; doubling the stride spreads them to 8 banks, still 4-way from the 256-element thread stride, so
  // the SERVICE count per pair of reads is what it was. Halving the read count is a SEPARATE change (one 32-bit read
  // plus a register deinterleave) and is not attempted here -- one variable at a time, and this one is the 73,728.
  static constexpr bool kFusedScaleZero =
#if defined(PPU_PACKED_SCALE_FUSED) && (PPU_PACKED_SCALE_FUSED != 0)
      kPackedScaleOn && (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) && !kScaleSwizzleOk;
#else
      false;
#endif

  // THE ASSUMPTION IS CHECKED, NOT ASSUMED. The fused layouts below are written out compactly rather than derived
  // from SmemLayoutScale, because a stride-doubling transform of an arbitrary (possibly swizzled, possibly padded)
  // layout is not a thing I can state in one line and be sure of. That is only sound while the layout it replaces IS
  // the compact one, so say so and let the build fail otherwise -- the alternative is the failure mode this file has
  // hit twice, where two functions build a tensor of the same name with different layouts and the second one faults
  // as "TSM out of range".
  static constexpr int kSZ_N   = int(shape<0>(ScaleTileShape{}));
  static constexpr int kSZ_G   = int(shape<1>(ScaleTileShape{}));
  static constexpr int kSZ_St  = int(DispatchPolicy::Stages);
  // THE GATE IS A TEMPLATE PARAMETER, NOT `!Fused || ...`. A disjunction does not stop the right-hand side from being
  // INSTANTIATED, and with PPU_SCALE_SWIZZLE on SmemLayoutScale is a ComposedLayout whose stride<> is deleted -- so the
  // first version of this check failed to compile the swizzle build while claiming to be inert there. `if constexpr` on
  // a template parameter is the only form that actually discards, which this file already says in as many words about
  // kPackedScaleOn. The local syntax gate caught it, which is the whole reason that gate exists.
  template <bool Fused, class L>
  static constexpr bool sz_layout_is_compact() {
    if constexpr (!Fused) { return true; }
    else {
      return cute::is_static<L>::value &&
             int(cute::cosize_v<L>) == kSZ_N * kSZ_G * kSZ_St &&
             int(cute::stride<0>(L{})) == 1 &&
             int(cute::stride<1>(L{})) == kSZ_N &&
             int(cute::stride<2>(L{})) == kSZ_N * kSZ_G;
    }
  }
  static_assert(sz_layout_is_compact<kFusedScaleZero, SmemLayoutScale>(),
                "PPU_PACKED_SCALE_FUSED assumes the compact (n, group, stage) scale layout");
  // Same discarding requirement for the flattened read view; see above.
  template <bool Fused, class L>
  static constexpr bool sz_copy_layout_is_compact() {
    if constexpr (!Fused) { return true; }
    else { return int(cute::stride<0>(L{})) == 1 && int(cute::stride<2>(L{})) == kSZ_N; }
  }

  // The WORD view the decoder stores through: one 32-bit slot per (n, group, stage), stride 1 in n so 32 lanes with
  // consecutive n write 32 consecutive words and touch all 32 banks once.
  using SmemLayoutScaleFusedWord = decltype(make_layout(
      make_shape(Int<kSZ_N>{}, Int<kSZ_G>{}, Int<kSZ_St>{}),
      make_stride(_1{}, Int<kSZ_N>{}, Int<kSZ_N * kSZ_G>{})));
  // The HALF views the readers keep using: same shape, every stride doubled. scale is the low half of each word and
  // zero the high half, so they differ only by the base pointer -- see scale_zero_base() below.
  using SmemLayoutScaleFusedHalf = decltype(make_layout(
      make_shape(Int<kSZ_N>{}, Int<kSZ_G>{}, Int<kSZ_St>{}),
      make_stride(_2{}, Int<2 * kSZ_N>{}, Int<2 * kSZ_N * kSZ_G>{})));
  // What every scale/zero TENSOR is built on. One name, so the six construction sites cannot disagree.
  using SmemLayoutScaleSZ = cute::conditional_t<kFusedScaleZero, SmemLayoutScaleFusedHalf, SmemLayoutScale>;

  static constexpr auto
  elements_per_smem_scale() {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return 0;
    }
    else if constexpr (ModeHasScales) {
      // FUSED TAKES BOTH PLANES' ELEMENTS AND THE ZERO TILE GOES TO ZERO, so the total is byte-identical. Written as
      // 2 * cosize of the UNFUSED layout, not cosize of the fused one: the fused layout's strides are doubled, so its
      // cosize is 2*cosize - 1 and the final zero slot would fall outside the allocation.
      return kFusedScaleZero ? 2 * int(cute::cosize_v<SmemLayoutScale>) : int(cute::cosize_v<SmemLayoutScale>);
    }
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled in scale smem allocation");
    }
  }

  static constexpr auto
  elements_per_smem_zero() {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert ||
                  KernelConversionMode == ConversionMode::ConvertAndScale ) {
      return 0;
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      return kFusedScaleZero ? 0 : int(cute::cosize_v<SmemLayoutScale>);   // fused: zero lives in smem_scale's odd halfs
    }
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled in zero smem allocation");
    }
  }

  // THE ONE PLACE THAT KNOWS WHERE THE ZERO PLANE LIVES. Fused, it is the odd half of every 32-bit slot, i.e. the
  // scale buffer's base plus one element; unfused it is its own array. Three sites build a zero tensor and all three
  // go through this, because "six tensors, where I had claimed two" is the recorded way this goes wrong.
  template <class Storage>
  CUTLASS_DEVICE static NonVoidElementZero*
  zero_smem_base(Storage& storage) {
    if constexpr (kFusedScaleZero) {
      return reinterpret_cast<NonVoidElementZero*>(storage.smem_scale.begin()) + 1;
    } else {
      return storage.smem_zero.begin();
    }
  }

public:
  // OBSERVABLE ON PURPOSE. kFusedScaleZero and the layouts it selects sit in the private section because they live
  // beside KernelConversionMode, which they need. But a flag nobody outside can read is a flag that silently does
  // nothing, and that is not hypothetical here: PPU_PACKED_SCALE_FUSED shipped to the box in a state where the only
  // translation unit that used it could not compile, the define was reported as a WARNING nobody's gate checked, the
  // binary built without it, correctness passed, and acu reported the store conflicts unchanged at 81,920 (+0.00%).
  //
  // CORRECTION, and it matters because the wrong version of this sentence closed off the one probe that works.
  // This used to read "every observable the bench and the profiler have -- shared bytes, instruction counts,
  // results -- is identical whether this path is on or off, BY DESIGN". Shared BYTES and RESULTS are identical, and
  // the read side is untouched. The STORE INSTRUCTION COUNT IS NOT: the publication below emits ONE uint32_t
  // assignment where the unfused branch emits two half assignments, so at this launch's
  //     4 warps x 8 groups x 9 publications x 128 CTAs = 36,864
  // shared-store instructions disappear. acu's `Shared Store / Inst` is therefore a valid machine-code probe --
  // ~84,480 -> ~47,616 -- and reading only `Bank Conflicts` is what made the previous round look undecidable.
  // (A backend is still free to lower the 32-bit store into two 16-bit ones. That is exactly what the Inst count
  // detects, and it is why the count is the probe rather than the source.)
  //
  // What the type-level gate is still for: dev/fold_derivation/l100_fused_active.cu answers "is this path selected
  // for THIS configuration" without a device, which no counter can do, and it is what proves a null result is a
  // real null rather than an inactive path.
  static constexpr bool is_fused_scale_zero = kFusedScaleZero;
  static constexpr bool is_packed_scale     = kPackedScaleOn;
  static constexpr int packed_scale_tiles_per_unit = kPackedTilesPerUnit;
  static constexpr bool has_zero_channel    = (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero);
  static constexpr int packed_scale_copy_threads = kPackedOwnerThreads;
  static constexpr int packed_scale_columns_per_thread = kPackedColsPerThread;
  // Type-level witness for launchers and sweep inventory. Zero means this collective uses the ordinary TileM-row
  // AIU A tile; positive values are the maximum real M accepted by the compact plain-copy path.
  using FusedScaleWordLayout = SmemLayoutScaleFusedWord;   // 32-bit slots, stride 1 in n: the conflict-free store
  using FusedScaleHalfLayout = SmemLayoutScaleSZ;          // what every reader sees; stride 2 in n when fused

  struct SharedStorage
  {
    static constexpr int scale_elements = elements_per_smem_scale();
    static constexpr int zero_elements = elements_per_smem_zero();
    // Packed: the cubes overlap, so the allocation is the packed span, not cosize of the logical tile.
    cute::ArrayEngine<RealInternalElementA,
        kPackedA ? kAPackSpan : cute::cosize_v<SmemLayoutAPhysical>> smem_a;
    // If this member is ever resized or removed, note that smem_b follows it directly and array_aligned defaults
    // to 16-B alignment: at the full size (cosize*2 B, a multiple of 32) smem_b happens to land 32-B aligned,
    // which PPU0010's AIU load requires (align_bytes = 32 in gemm_operands.hpp). Shrinking smem_a to one element
    // once put smem_b at offset 2 and produced 'AIU_ld TSM size out of range'. The alignment holds by arithmetic,
    // not by declaration.
    cute::ArrayEngine<BTransportElement, cute::cosize_v<SmemLayoutB>> smem_b;
    // PACKED SCALE CHANNEL (plan #20 option E). When on, the tile holds the gguf's own bytes -- one 16 B unit per
    // (superblock, column) carrying d, dmin and the codes -- so the ZERO TILE GOES TO ZERO ELEMENTS, not shrinks: `mn`
    // lives in the same unit. Chosen by TYPE rather than by #if, because one binary holds units of several shapes:
    // Xplane activates at one full superblock, while the explicit K-pack4 schedule also activates integral sub-runs.
    //
    // smem_zero stays declared, at zero elements, rather than being deleted: it is the LAST member, so a zero-length
    // ArrayEngine cannot move anything (unlike smem_a, whose comment above records what shrinking a leading member did
    // to smem_b's 32 B alignment), and keeping the name lets the ScaleZero paths compile unchanged.
    // F CHANGES NOTHING HERE. The decode happens on the way IN (see packed_decode_stage), so smem still holds the
    // same two fp16 planes and the whole read side -- s2r, fragments, the four transform arms -- is untouched. That is
    // the point: llama.cpp's MMQ keeps the shared read and amortises the decode over its consumers, and the earlier
    // register-decode attempt did the opposite (measured 1.4M extra ALU to save 0.27M shared loads, with LSU only 6%
    // busy and IALU/FALU at 14%).
    cute::ArrayEngine<NonVoidElementScale, scale_elements> smem_scale;
    cute::ArrayEngine<NonVoidElementZero, zero_elements> smem_zero;
    // LAST on purpose: at zero elements it cannot move anything above it, so a default build stays byte-identical.
    cute::ArrayEngine<uint8_t, kPackedScaleOn ? int(cute::cosize(SmemLayoutScaleRawStaged{})) : 0> smem_scale_raw;
  };
  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A = nullptr;
    StrideA dA{};
    ElementB const* ptr_B = nullptr;
    StrideB dB{};
    ElementScale const* ptr_S = nullptr;
    NonVoidStrideScale dS{};
    int group_size = 0;
    ElementZero const* ptr_Z = nullptr;
    int const* group_row_offsets = nullptr;   // ragged grouped: per-expert cumulative A row start; null=uniform
  };

  // Admission belongs beside the collective that owns the physical format.
  // In particular, an interleaved artifact cannot honor an arbitrary dB and a
  // packed metadata unit cannot honor the generic fp16 dS/ptr_Z surface.  Both
  // query and launch reach this exact predicate through the kernel wrappers.
  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape,
                            Arguments const& args) {
    auto mnkl = append<4>(problem_shape, Int<1>{});
    int64_t const N = int64_t(get<1>(mnkl));
    int64_t const K = int64_t(get<2>(mnkl));
    int64_t const L = int64_t(get<3>(mnkl));
    auto const& dB = [&]() -> auto const& {
      if constexpr (SwapAB) return args.dA;
      else return args.dB;
    }();
    detail::MixedArgumentContract c{};
    c.n = N; c.k = K; c.l = L;
    c.group_size = args.group_size;
    c.tile_k = int64_t(size<2>(TileShape{}));
    c.scale_tile_k = int64_t(Scale_TileK);
    c.static_group_size = DispatchPolicy::StaticGroupSize;
    c.low_bits = cutlass::sizeof_bits<RealInternalElementB>::value;
    c.interleave = int(kContinous{});
    c.has_scales = ModeHasScales;
    c.packed_scale = kPackedScaleOn;
    c.packed_tiles_per_unit = kPackedTilesPerUnit;
    c.ptr_Z_nonnull = args.ptr_Z != nullptr;
    c.dB0 = int64_t(get<0>(dB));
    c.dB1 = int64_t(get<1>(dB));
    c.dBL = int64_t(get<2>(dB));
    c.dS0 = int64_t(get<0>(args.dS));
    c.dS1 = int64_t(get<1>(args.dS));
    c.dSL = int64_t(get<2>(args.dS));
    return detail::mixed_arguments_supported(c);
  }

  // Device side kernel params
  struct Params {
    GmemTiledCopyScale gmem_tiled_copy_scale;
    // The raw-byte cp.async beside the fp16 one; stateless, so carrying both costs nothing.
    GmemTiledCopyScalePacked gmem_tiled_copy_scale_packed;
    GmemTiledCopyZero gmem_tiled_copy_zero;

    RealInternalElementA const* ptr_A = nullptr;
    InternalStrideA dA{};
    RealInternalElementB const* ptr_B = nullptr;
    InternalStrideB dB{};

    NonVoidElementScale const* ptr_S = nullptr;
    NonVoidStrideScale dS{};
    NonVoidElementZero const* ptr_Z = nullptr;

    int group_size = 0;
    int64_t scale_k = 0;
    int reload_factor = 0;
    int const* group_row_offsets = nullptr;
  };

  GmemTiledCopyA gmem_tiled_copy_A;
  GmemTiledCopyB gmem_tiled_copy_B;
  int64_t scale_residue_n = 0;
  int64_t scale_residue_k = 0;
  bool scale_valid = true;
  // THE PACKED COPY NEEDS ITS OWN N PREDICATE. scale_valid is derived from the fp16 copy's coordinate, whose
  // thread -> N map is 8*(p % 16) for a (16,8) x (_8,_1) layout, while the packed copy is partitioned ONE COLUMN PER
  // THREAD. Using one for the other is only invisible because every shape measured so far has N a multiple of TileN.
  // With a residue it fails both ways at once: at residue_n = 20 thread 3 owns packed column 3 (valid) but its fp16
  // coordinate is 24, so its cp.async is skipped while the decode loop still reads that column -- bytes nobody
  // copied; and thread 32 has fp16 coordinate 0, so it copies packed column 32 out of a tile that has twenty, an
  // out-of-bounds GLOBAL read. The raw-byte copy therefore always carries its
  // own predicate rather than borrowing the fp16 copy's coordinate.
  bool scale_valid_pk = true;
  // Per-thread stage state for the metadata unit copied by THIS thread. It is consumed before the CTA publication
  // barrier by that same owner, so no shared tag or additional synchronization is needed. For the historical TK256
  // path kPackedTilesPerUnit is one and every use folds to the constant zero.
  int packed_tile_in_unit[DispatchPolicy::Stages] = {};
  //
  // Methods
  //


  template <class ProblemShape>
  static Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    Params p;
    if constexpr (!SwapAB) {
      p.ptr_A = reinterpret_cast<RealInternalElementA const*>(args.ptr_A);
      p.ptr_B = reinterpret_cast<RealInternalElementB const*>(args.ptr_B);
      p.dA = args.dA;
      p.dB = args.dB;
    }
    else {
      p.ptr_A = reinterpret_cast<RealInternalElementA const*>(args.ptr_B);
      p.ptr_B = reinterpret_cast<RealInternalElementB const*>(args.ptr_A);
      p.dA = args.dB;
      p.dB = args.dA;
    }
    p.group_row_offsets = args.group_row_offsets;

    if constexpr (ModeHasScales) {
      p.gmem_tiled_copy_scale = GmemTiledCopyScale{};
      p.ptr_S = reinterpret_cast<NonVoidElementScale const*>(args.ptr_S);
      p.dS = detail::lower_metadata_stride(args.dS);
      // THE SAME TYPE THE MEMBER IS DECLARED AS, not a second construction of it. This line spelled the atom out as
      // uint128 while GmemTiledCopyScalePacked derived it from the unit, so the moment the unit stopped being 16
      // bytes the two disagreed -- and the failure surfaced as "TiledCopy uses too few vals" pointing at a copy that
      // looked correct where it was declared. One relation, one place; the member's own type is that place.
      p.gmem_tiled_copy_scale_packed = GmemTiledCopyScalePacked{};
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        p.gmem_tiled_copy_zero = GmemTiledCopyZero{};
        p.ptr_Z = reinterpret_cast<NonVoidElementZero const*>(args.ptr_Z);
      }
      p.group_size = args.group_size;
      p.scale_k = (get<2>(problem_shape) + args.group_size - 1) / args.group_size;
      p.reload_factor = (args.group_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{});
    }
    return p;
  }

  /// Set up the data needed by this collective for load and mma.
  /// Returns a tuple of tensors. The collective and the kernel layer have the contract.
  /// Returned tuple must contain at least two elements, with the first two elements being gA & gB.
  /// The rest of the tensors can be specified as needed by this collective.
  template <class ProblemShape_MNKL, class BlockCoord_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, BlockCoord_MNKL const& blk_coord_mnkl, Params const& mainloop_params) {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;
    auto [m_coord, n_coord, _, l_coord] = blk_coord_mnkl;

    // A init
    using TilerA = typename GmemTiledCopyA::Tiler_MN;
    gmem_tiled_copy_A.desc_.template init<RealInternalElementA, false, get<0>(TilerA{}), get<1>(TilerA{})>(nullptr, M, K, mainloop_params.dA);
    // RAGGED grouped: expert l_coord starts at group_row_offsets[l_coord] rows using dA's row pitch.
    // Uniform/batched: dA's L pitch owns the expert base. Reconstructing either base with logical K accepted a
    // caller stride and then silently substituted a compact layout (L128).
    auto a_expert_base = detail::mixed_a_expert_base(
        mainloop_params.ptr_A, mainloop_params.dA,
        mainloop_params.group_row_offsets, l_coord);
    Tensor mA_mkl = make_tensor(make_gmem_ptr(a_expert_base),
                                make_shape(M,K,cute::Int<1>{}), mainloop_params.dA);                            // (m,k,1)
    auto gA_logical = [&] {
      if constexpr (kPackedA) {
        // PLAIN, not make_mix_tensor_like: that wrapper carries (ptr, coordinate) for the AIU descriptor and has NO
        // addressable strides (l74), so &gA(...) yields a meaningless address. The packed-row provider writes A with
        // cp.async and therefore needs real strides.
        return local_tile(mA_mkl(_,_,0), TileShape{}, take<0,3>(blk_coord_mnkl), Step<_1, X,_1>{});
      } else {
        auto mA_mk = make_mix_tensor_like(mA_mkl(_,_,0));
        return local_tile(mA_mk, TileShape{}, take<0,3>(blk_coord_mnkl), Step<_1, X,_1>{});
      }
    }();                                                                                                      // (BLK_M,BLK_K,k)
    auto gA = [&] {
      if constexpr (LogicalTileM == PhysicalATileM) {
        return gA_logical;
      } else {
        // Preserve local_tile's base coordinate m_coord*TileM, but widen only its in-tile M extent.  Cutting a
        // physical-16 local_tile directly would advance the second logical tile to row 16 and silently skip rows
        // 8..15.  The descriptor still carries the real problem M, so `.padz` zero-fills the widened tail.
        auto physical_m = make_layout(Int<PhysicalATileM>{}, stride<0>(gA_logical.layout()));
        return make_tensor(gA_logical.data(), replace<0>(gA_logical.layout(), physical_m));
      }
    }();                                                                                                      // (A_PHYS_M,BLK_K,k)

    // B init (include init aiu desc)
    auto mB_nk = load_init_B(mainloop_params, N, K, L, l_coord);                                                // (n,k)
    Tensor gB = [&] {
      if constexpr (kQ4KPack4Transpose) {
        // Physical B is [N,K/4] b16.  Its K-tile count is nevertheless
        // identical to the logical [N,K] Q4 tensor because both the problem
        // and tactic K extents are divided by four here.
        using PhysicalBTile = Shape<decltype(shape<1>(TileShape{})),
                                    Int<PhysicalBTileK>>;
        return local_tile(mB_nk, PhysicalBTile{},
                          make_coord(n_coord, _));
      } else {
        return local_tile(mB_nk, TileShape{}, take<0,3>(blk_coord_mnkl),
                          Step<X, _1, _1>{});
      }
    }();                                                                                                       // (BLK_N,BLK_K{phys},k)

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return cute::make_tuple(gA, gB);
    }
    else if constexpr (ModeHasScales) {
      auto scale_k = mainloop_params.scale_k;
      Tensor gS = detail::make_metadata_tile<ScaleTileShape>(
          mainloop_params.ptr_S, mainloop_params.dS,
          N, scale_k, L, l_coord, n_coord);                                           // (BLK_N, 1, scale_k)

      // init scale_residue_n
      scale_residue_n = detail::mixed_logical_n_residue(
          N, int(size<1>(TileShape{})), n_coord);

      // THE PACKED PLANE: [unit][N][bytes]. A TK256 Q4 tile owns one unit; TK128/TK64 K-pack4 tiles select two/four
      // integral runs from one unit. Built unconditionally and appended because a macro-dependent tuple return type
      // would not compile; kPackedTilesPerUnit is one for every established off/TK256 instantiation.
      int const packed_tiles_ = scale_k / int(Scale_TileK);
      int const packed_units_ = packed_tiles_ / kPackedTilesPerUnit;
      Tensor mSp = make_tensor(make_gmem_ptr(reinterpret_cast<uint8_t const*>(mainloop_params.ptr_S)),
                               make_shape (N, Int<kPackedScaleUnit>{}, packed_units_, L),
                               make_stride(Int<kPackedScaleUnit>{}, _1{},
                                           Int<kPackedScaleUnit>{} * N,
                                           Int<kPackedScaleUnit>{} * N * packed_units_));
      Tensor gSp = local_tile(mSp(_,_,_,l_coord), Shape<Int<Scale_TileN>, Int<kPackedScaleUnit>>{},
                              make_coord(n_coord, 0, _));                           // (BLK_N, bytes, packed_unit)

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(gA, gB, gS, gSp);
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor gZ = detail::make_metadata_tile<ScaleTileShape>(
            mainloop_params.ptr_Z, mainloop_params.dS,
            N, scale_k, L, l_coord, n_coord);
        return cute::make_tuple(gA, gB, gS, gZ, gSp);
      }
      else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                      "Conversion mode not handled in load_init");
      }
    }
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "load_init requires direct conversion or scale metadata");
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  template <
    class... Ts,
    class FrgTensorC,
    class KTileIterator
  >
  CUTLASS_DEVICE void
  operator() (
      Params const& mainloop_params,
      cute::tuple<Ts...> const& load_inputs,
      FrgTensorC &accum,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      char *smem_buf)
  {

    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutA{}) == 3,
      "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");
    static_assert(rank(SmemLayoutB{}) == 3,
      "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");

    int warp_idx = canonical_warp_idx_sync();
    int aiu_warp_group_thread_idx = warp_idx * 32;

    Tensor gA = get<0>(load_inputs);
    Tensor gB = get<1>(load_inputs);
    auto k_iter_shape = cute::shape<2>(gB);

    // Construct shared memory tiles
    if constexpr (kPackedA) {
    // l85's collision check, as a body-level assert. It CANNOT sit in the class body: it calls a member of the
    // same class, which is still incomplete there -- nvcc's EDG front end accepts that and hgcc rejects it with
    // "no type named 'SharedStorage'", which is how the local gate passed and the box build failed.
    static_assert(kAPackRows >= 1 && kAPackRows <= 8,
                  "packed-A provider supports 1 <= R <= 8 across compiled physical cube geometries");
    static_assert(kAPackRows <= LogicalTileM, "packed-A provider cannot publish more rows than logical TileM");
    static_assert(aPackDisjoint(), "packed-A provider runs collide -- fix the derived pitch");
    static_assert(kACubeW == 64, "packed-A run offsets assume AiuContElemSize == 64 halfs");
    static_assert(kAPackPitch % 64 == 0 && kAPackSpan % 64 == 0,
                  "packed-A: every cube base and the complete A span must remain 128-B aligned");
    // The read's pitch and the write's call the same detail::aPackPitchForRows(), so they cannot diverge the way
    // they did when each side carried its own literal.
    static_assert(PhysicalATileM == kACubeH, "packed-A cube authority must be the physical m8/m16 footprint");
    }
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.begin()), SmemLayoutA{}); // (BLK_M_LOGICAL,BLK_K,PIPE)
    Tensor sA_physical = make_tensor(
        make_smem_ptr(storage.smem_a.begin()), SmemLayoutAPhysical{});              // (BLK_M_PHYSICAL,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.begin()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    // CuTe copy slices are logical owners, not a license to replay the same asynchronous shared-memory write from
    // every physical CTA thread.  The fp16 scale/zero tile and the packed raw tile have different copy layouts, so
    // keep both exact owner predicates and both logical slots.  Surplus threads remain consumers only.
    bool const scale_copy_owner =
        kLegacyModuloMetadataPublishers ||
        (ScaleCopyPlan::thread_slots == Scale_NumThreads) ||
        ScaleCopyPlan::owns_physical_thread(thread_idx);
    bool const packed_copy_owner =
        kLegacyModuloMetadataPublishers ||
        (kPackedOwnerThreads == Scale_NumThreads) ||
        PackedMetadataOwnership::owns_physical_thread(thread_idx);
    auto extra_input_partitions = partition_extra_inputs(
        mainloop_params, load_inputs, storage,
        ScaleCopyPlan::logical_slot(thread_idx),
        PackedMetadataOwnership::copy_owner(thread_idx),
        scale_copy_owner);

    CUTE_STATIC_ASSERT_V(size<0>(gA) == size<0>(sA_physical));                 // BLK_M_PHYSICAL
    CUTE_STATIC_ASSERT_V(size<1>(gA) == size<1>(sA_physical));                 // BLK_K
    CUTE_STATIC_ASSERT_V(size<0>(sA) == size<0>(TileShape{}));                 // BLK_M_LOGICAL
    CUTE_STATIC_ASSERT_V(size<0>(gB) == size<0>(sB));                          // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(gB) == size<1>(sB));                          // BLK_K
    if constexpr (kQ4KPack4Transpose) {
      CUTE_STATIC_ASSERT_V(size<1>(sA) == Int<q4_kpack4::kPack>{} * size<1>(sB));
    } else {
      CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));                        // BLK_K
    }
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA_physical)); // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));        // PIPE

    // Partition the copying of A and B tiles across the threads
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)
    auto copy_A_and_B = [&] (auto k_tile, auto k_iter_crd, int pipe) {
      if constexpr (kPackedA) {
        copy_aiu(gmem_tiled_copy_B, tBgB(_,_,_,k_iter_crd), tBsB(_,_,_,pipe), warp_idx);
        copy_A_packed_rows<kAPackRows>(
            gA, storage.smem_a.begin(), k_tile, pipe, thread_idx, gmem_tiled_copy_A.desc_.dim_h);
      } else {
        auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
        Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                         // (ACPY,ACPY_M,ACPY_K,k)
        Tensor tAsA = gmem_thr_copy_A.partition_D(sA_physical);                // (ACPY,ACPY_M_PHYS,ACPY_K,PIPE)
        copy_aiu(
          gmem_tiled_copy_A, tAgA(_,_,_,k_tile), tAsA(_,_,_,pipe),
          gmem_tiled_copy_B, tBgB(_,_,_,k_iter_crd), tBsB(_,_,_,pipe),
          warp_idx
        );
      }
    };

    // Start async loads for all pipes but the last
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages-1; ++k_pipe) {
      auto k_iter_crd = cute::idx2crd(*k_tile_iter, k_iter_shape);
      copy_A_and_B(*k_tile_iter, k_iter_crd, k_pipe);
      copy_async_extra_info(mainloop_params, extra_input_partitions, *k_tile_iter, k_pipe,
                            scale_copy_owner, packed_copy_owner);
      cp_async_fence();
      --k_tile_count;
      if (k_tile_count > 0) { ++k_tile_iter; }
    }

    //
    // MMA Atom partitioning
    //

    // Tile MMA compute thread partitions and allocate accumulators
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                   // (MMA,MMA_M,MMA_K)
    Tensor tCrB_mma = [&] {
      if constexpr (kQ4KPack4Transpose) {
        // Owning rmem fragment only: no load is performed through this logical
        // view.  The physical b16 tile is read below, converted, and retiled
        // into this ordinary N x logical-K MMA destination.
        auto logical_b = make_tensor(
            make_smem_ptr(reinterpret_cast<cutlass::half_t*>(storage.smem_b.begin())),
            make_layout(
                make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
                make_stride(shape<2>(TileShape{}), _1{})));
        return thr_mma.partition_fragment_B(logical_b);
      } else {
        return thr_mma.partition_fragment_B(sB(_,_,0));
      }
    }();                                                                       // (MMA,MMA_N,MMA_K logical)
#if defined(PPU_B_DEQUANT_NOP) && (PPU_B_DEQUANT_NOP != 0)
    // The ablation must not change what the MMA pipe is fed. partition_fragment_B does not initialise, and with the
    // conversion removed nothing else would either, so every atom would consume indeterminate bits -- which as fp16
    // are freely NaN or Inf, and a timing measurement taken over exceptional operands measures the exception
    // handling. One fill, outside the k-loop, so it costs nothing the measurement cares about.
    cute::fill(tCrB_mma, static_cast<typename decltype(tCrB_mma)::value_type>(1.0f));
#endif

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                    // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB_mma) == size<2>(accum));                // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB_mma));                 // MMA_K

    //
    // Copy Atom retiling
    //

    using warpOnM = decltype(get<1>(tiled_mma.get_thr_layout_vmnk().shape()));
    using warpOnN = decltype(get<2>(tiled_mma.get_thr_layout_vmnk().shape()));
    using PermutationM = decltype(tiled_mma.template permutation_mnk<0>());
    using PermutationN = decltype(tiled_mma.template permutation_mnk<1>());

    // B's swzl loader is described by a fixed m16 int8 shadow MMA.  Its M permutation counts the same compute-MMA
    // warps as the main TiledMma, but each shadow warp covers 16 rows.  Reusing PermutationM is correct for the legacy
    // m16 family and silently becomes 8 rows for m8, which is not a legal permutation for the shadow atom.
    static constexpr int MainInstM = size<0>(typename TiledMma::AtomShape_MNK{});
    static_assert(MainInstM == 8 || MainInstM == 16, "the B shadow loader only supports the m8 and m16 families");
    static_assert(PermutationM{} == warpOnM{} * Int<MainInstM>{},
                  "main-M permutation must cover one compute atom per M warp");
    using ShadowPermutationM = Int<warpOnM() * 16>;

    using TiledMma_S8 = TiledMMA<
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
        MMA_Atom<PPU0010_16x16x32_S32S8S8S32_TN>,
#else
        MMA_Atom<PPU0015_16x16x32_S32S8S8S32_TN>,
#endif
        Layout<Shape<warpOnM, warpOnN,_1>>,
        Tile<ShadowPermutationM, PermutationN, _32>>;

    // The K-pack4 reader uses the actual PPU0010 fp16 B-fragment map because
    // each opaque b16 is one transport word.  Its K mode is physical K/4, so
    // one x1 copy step (Kgroup16) feeds four logical K16 MMA atoms.
    using TiledMma_KPack4 = TiledMMA<
        MMA_Atom<PPU0010_16x16x16_F32F16F16F32_TN>,
        Layout<Shape<warpOnM, warpOnN, _1>>,
        Tile<ShadowPermutationM, PermutationN, _16>>;
    using TiledMma_BLoad = cute::conditional_t<
        kQ4KPack4Transpose, TiledMma_KPack4, TiledMma_S8>;

    TiledMma_BLoad tiled_mma_bload;
    auto thr_mma_bload = tiled_mma_bload.get_thread_slice(thread_idx);

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(aiu_warp_group_thread_idx);
    Tensor tCsA = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));        // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                                       // (CPY,CPY_M,CPY_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto sB_load = [&] {
      if constexpr (kQ4KPack4Transpose) return sB;
      else return recast<int8_t>(sB);
    }();
    Tensor tCrB_load = thr_mma_bload.partition_fragment_B(sB_load(_,_,0));

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma_bload);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(aiu_warp_group_thread_idx);
    Tensor tCsB            = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB_load));            // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB_load);                                  // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    // extra inputs partition and retile
    auto partitioned_extra_info = partition_extra_mma_info(tiled_mma, storage, thread_idx);
    auto copy_partitions_extra_info = retile_extra_mma_info(tiled_mma, partitioned_extra_info, thread_idx);
#if defined(PPU_SCALE_PREFETCH) && (PPU_SCALE_PREFETCH != 0)
    // GROUP-AHEAD SCALE PREFETCH. On the FINE path the scale and zero are reloaded from smem at each group's first
    // mma atom and used one or two instructions later -- 8 times per k-tile at gs=32 with TileK=256. With 14.2 warps
    // per CU and no spare (this shape is work-bound), every one of those is a Memory Dependency stall that costs
    // time directly, and Memory Dependency is the top warp state at 0.98.
    //
    // Priced by removing the channel entirely (SK_QUANT on the bench): per-group reload = 7.3% of the kernel, which
    // is this change's ceiling. Dropping the zero as well is another 11.5%, but that is a format question (#20), not
    // a scheduling one.
    //
    // A second register set lets a copy step issue BOTH its groups' loads up front. Built here, not in
    // partition_extra_mma_info, because that helper is shared with the fold and 2plane collectives; retile_D is only
    // a call, so a local pair costs nothing but registers. Passed as ONE named tuple parameter -- appending to
    // transform_B_kblock's cute::tuple<Ts...> does not deduce.
    // Only the WithZero tuple has a get<3>; on the ScaleOnly path that index is out of range, so the pack is built
    // inside an if constexpr and the other modes get an empty tuple.
    auto scale_pf = [&] {
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        auto thr_pf = make_tiled_copy_B(SmemCopyAtomScale{}, tiled_mma).get_thread_slice(thread_idx);
        auto s_pf = cute::make_fragment_like(cute::get<1>(partitioned_extra_info));
        auto z_pf = cute::make_fragment_like(cute::get<3>(partitioned_extra_info));
        return cute::make_tuple(s_pf, z_pf, thr_pf.retile_D(s_pf), thr_pf.retile_D(z_pf));
      } else {
        return cute::tuple<>{};
      }
    }();
#else
    auto scale_pf = cute::tuple<>{};
#endif

    //
    // PIPELINED MAIN LOOP
    //

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrB_copy_view);
    auto K_ATOM_PER_COPY = size<2>(tCrB_mma) / size<2>(tCrB_copy_view);
    auto A_BLOCK_MAX = size<2>(tCrA_copy_view);
    auto MMA_K_ATOMS = size<2>(tCrA);
    using ARegisterSchedule = detail::MixedARegisterSchedule<
        decltype(MMA_K_ATOMS)::value, decltype(A_BLOCK_MAX)::value,
        decltype(K_BLOCK_MAX)::value>;
    static_assert(ARegisterSchedule::BAtomsPerCopy ==
                      decltype(K_ATOM_PER_COPY)::value,
                  "B delivery and MMA atom partitions must agree");

    int initial_read_stage = 0;
    Tensor tCsA_p = tCsA(_,_,_,initial_read_stage);
    Tensor tCsB_p = tCsB(_,_,_,initial_read_stage);
    auto bind_read = [&] (int stage) {
      tCsA_p = tCsA(_,_,_,stage);
      tCsB_p = tCsB(_,_,_,stage);
    };
    auto publish = [&] (int stage) {
      packed_decode_stage<kPackedScaleOn>(
          storage, stage, packed_tile_in_unit[stage], thread_idx, scale_residue_n);
    };
    auto prepare = [&] (auto k_block, int read_stage, auto prime) {
      copy_B_and_extra_info(smem_tiled_copy_B, tCsB, tCrB_copy_view,
          partitioned_extra_info, copy_partitions_extra_info, k_block, read_stage);
      // NO M-PINNING LOOP HERE, and that is a measured decision. CPY_M = size<1>(tCsA) is 1 both with and without
      // PPU_A_CUBE_H (fold_derivation/l77), so M does not live on mode 1 and a loop over it is a no-op. With
      // CUBE_H=1 cute instead moves mode 2 from basis 2 to basis 0 with stride 64 and halves the A register
      // fragment (ArrayEngine 128 -> 64, with a stride-0 component), i.e. it re-derives the geometry itself.
      detail::prepare_mixed_a_for_b<ARegisterSchedule>(
          smem_tiled_copy_A, tCsA_p, tCrA_copy_view, k_block, prime);
      transform_B_kblock<RealInternalElementB>(tCrB_copy_view, tCrB_mma, partitioned_extra_info, k_block,
          K_ATOM_PER_COPY, copy_partitions_extra_info, read_stage, scale_pf);
    };
    auto prefetch = [&] (auto k_tile, int write_stage) {
          auto k_iter_crd = cute::idx2crd(k_tile, k_iter_shape);
          copy_A_and_B(k_tile, k_iter_crd, write_stage);
          copy_async_extra_info(mainloop_params, extra_input_partitions, k_tile, write_stage,
                                scale_copy_owner, packed_copy_owner);
    };
    auto consume = [&] (auto k_block, int read_stage) {
        CUTLASS_PRAGMA_UNROLL
        for (int k_loop = 0; k_loop < K_ATOM_PER_COPY; k_loop++) {
          auto atom_idx = k_block * K_ATOM_PER_COPY + k_loop;
          // Transform before compute
          cute::transform(tCrA(_,_,atom_idx), TransformA{});
          cute::transform(tCrB_mma(_,_,atom_idx), TransformB{});
          // gemm for one tiled_mma atom on K
          cute::gemm(tiled_mma, tCrA(_,_,atom_idx), tCrB_mma(_,_,atom_idx), accum);
        }
        detail::finish_mixed_a_after_consume<ARegisterSchedule>(
            smem_tiled_copy_A, tCsA_p, tCrA_copy_view, k_block);
    };

    detail::run_mixed_pipeline<DispatchPolicy::Stages>(K_BLOCK_MAX, k_tile_iter, k_tile_count,
        bind_read, publish, prepare, prefetch, consume);
  }

private:
  CUTLASS_DEVICE
  auto load_init_B(Params const& mainloop_params, int N, int K, int L, int l_coord) {
    if constexpr (kQ4KPack4Transpose) {
      using TilerB = typename GmemTiledCopyB::Tiler_MN;
      using Transport = cutlass::half_t;
      int const physical_k = K / q4_kpack4::kPack;
      static_assert(sizeof_bits<Transport>::value ==
                        q4_kpack4::kPack * sizeof_bits<RealInternalElementB>::value,
                    "one K-pack4 transport word must cover four logical codes");
      // Keep the outer expert coordinate inside the CuTe tensor until it is
      // selected.  The first grouped K-pack4 port manually advanced a byte
      // pointer by l_coord and then sliced this L mode by l_coord as well;
      // expert e consequently read physical expert 2*e while dense (e=0)
      // remained exact.  One CuTe L slice now owns both the coordinate view
      // and the raw base exported to the opaque AIU descriptor.
      auto physical_stride =
          make_stride(_1{}, int64_t(N), int64_t(N) * physical_k);
      Tensor mB_nkl = make_tensor(
          make_gmem_ptr(reinterpret_cast<Transport const*>(mainloop_params.ptr_B)),
          make_shape(N, physical_k, L), physical_stride);
      auto mB_nk = mB_nkl(_,_,l_coord);
      static_assert(rank(decltype(mB_nk.layout()){}) == 2,
                    "the selected K-pack4 expert view must not retain an L mode");
      auto const* expert_base = reinterpret_cast<uint8_t const*>(
          raw_pointer_cast(mB_nk.data()));
      gmem_tiled_copy_B.desc_.template init<
          Transport, true, get<0>(TilerB{}), get<1>(TilerB{})>(
              const_cast<uint8_t*>(expert_base), N, physical_k,
              take<0,2>(physical_stride));
      return make_mix_tensor_like(mB_nk);
    } else {
    auto kCon = kContinous{};
    using TilerB = typename GmemTiledCopyB::Tiler_MN;
    if constexpr (kCon != 1) {
      auto const b_shape = make_shape(N, make_shape(kCon, K / kCon));
      auto const b_stride = make_stride(kCon, make_stride(cute::Int<1>{}, kCon * N));
      auto layout_counting = make_layout(
        b_shape,
        make_stride(ScaledBasis<_1, 1>{}, make_stride(ScaledBasis<_1, 0>{}, ScaledBasis<int, 1>{N}))
      );
      Tensor mB_nk_counting = make_counting_tensor(layout_counting);
      auto const* expert_base = detail::mixed_packed_byte_expert_base(
          mainloop_params.ptr_B,
          int64_t(N) * int64_t(K) * sizeof_bits<RealInternalElementB>::value / 8,
          l_coord);
      gmem_tiled_copy_B.desc_.template init<RealInternalElementB, false, get<0>(TilerB{}), get<1>(TilerB{})>(
            const_cast<uint8_t*>(expert_base), N * K / kCon, kCon, b_stride);
      return mB_nk_counting;
    } else {
      Tensor mB_nk = make_mix_tensor_like(
          detail::mixed_subbyte_l_slice<RealInternalElementB>(
              mainloop_params.ptr_B, make_shape(N, K, L),
              mainloop_params.dB, l_coord));

      gmem_tiled_copy_B.desc_.template init<RealInternalElementB, false, get<0>(TilerB{}), get<1>(TilerB{})>(
            nullptr, N, K, mainloop_params.dB);
      return mB_nk;
    }
    }
  }

  template <class... Ts>
  CUTLASS_DEVICE
  auto copy_async_extra_info(
        Params const& mainloop_params,
        cute::tuple<Ts...>& extra_input_partitions,
        int k_idx,
        int write_stage,
        bool scale_copy_owner,
        bool packed_copy_owner) {
    if constexpr (ModeHasScales) {
      auto tSgS = get<0>(extra_input_partitions);
      auto tSsS = get<1>(extra_input_partitions);
      auto tScS = get<2>(extra_input_partitions);
      // The packed pair sits AFTER whatever the mode already appended -- 3,4 for ScaleOnly and 5,6 for ScaleZero. Named
      // once here so the two copy sites below cannot disagree about the index, which is the shape of bug that has cost
      // this work the most time.
      // (tSgSp, tSsSp): 3,4 for ScaleOnly and 5,6 for ScaleZero. Named ONCE so the two sites cannot disagree.
      static constexpr int kPkG = (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) ? 5 : 3;
      static constexpr int kPkC = kPkG + 2;
      // per-column path
      if constexpr(DispatchPolicy::StaticGroupSize == -1) {
        // Packed: ONE cp.async of the gguf's own bytes into staging. The decode is at the barrier, in mma().
        if constexpr (kPackedScaleOn) {
          if (packed_copy_owner) {
            packed_tile_in_unit[write_stage] = 0;
            detail::copy_packed_metadata_if<kPackedColsPerThread>(
                mainloop_params.gmem_tiled_copy_scale_packed, scale_valid_pk, scale_residue_n,
                get<kPkG>(extra_input_partitions)(_,_,_,0),
                get<kPkG+1>(extra_input_partitions)(_,_,_,write_stage),
                get<kPkC>(extra_input_partitions));
          }
        } else if (scale_copy_owner) {
          copy(mainloop_params.gmem_tiled_copy_scale, tSgS(_,_,_,0), tSsS(_,_,_,write_stage));
        }
        // NOT under kPackedScaleOn: `mn` rides in the scale unit, and smem_zero is zero elements there, so issuing
        // this copy would write past the end of the allocation. Found by asking what the ScaleZero fixture would do,
        // not by a compiler -- a 0-length ArrayEngine is a valid pointer and cp.async would happily scribble past it.
        if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero && !kPackedScaleOn) {
          auto tZgZ = get<3>(extra_input_partitions);
          auto tZsZ = get<4>(extra_input_partitions);
          if (scale_copy_owner) {
            copy(mainloop_params.gmem_tiled_copy_zero, tZgZ(_,_,_,0), tZsZ(_,_,_,write_stage));
          }
        }
      }
      else {
        int scale_load_k;
        // specific group-wise path
        if constexpr (DispatchPolicy::StaticGroupSize > 0) {
          constexpr int reload_factor = (DispatchPolicy::StaticGroupSize + size<2>(TileShape{}) - 1) / size<2>(TileShape{});
          scale_load_k = k_idx / reload_factor;
        }
        // default path
        else {
          scale_load_k = k_idx / mainloop_params.reload_factor; // This will always be 0 when group_size == K.
        }
        // kPackedScaleOn picks the predicate that belongs to the copy actually being issued.
        if ((kPackedScaleOn || scale_valid) && (scale_load_k * Scale_TileK < scale_residue_k)) {
          if constexpr (kPackedScaleOn) {
            // scale_load_k is a logical metadata TILE index. K-pack4 TK64/TK128 share one source unit across four/two
            // consecutive tiles; the per-stage phase selects the compile-time group run after the unit lands.
            if (packed_copy_owner) {
              packed_tile_in_unit[write_stage] = scale_load_k % kPackedTilesPerUnit;
              detail::copy_packed_metadata_if<kPackedColsPerThread>(
                  mainloop_params.gmem_tiled_copy_scale_packed, scale_valid_pk, scale_residue_n,
                  get<kPkG>(extra_input_partitions)(_,_,_,scale_load_k / kPackedTilesPerUnit),
                  get<kPkG+1>(extra_input_partitions)(_,_,_,write_stage),
                  get<kPkC>(extra_input_partitions));
            }
          } else if (scale_copy_owner) {
            copy(mainloop_params.gmem_tiled_copy_scale,
                 tSgS(_,_,_,scale_load_k), tSsS(_,_,_,write_stage));
          }
          if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero && !kPackedScaleOn) {
            auto tZgZ = get<3>(extra_input_partitions);
            auto tZsZ = get<4>(extra_input_partitions);
            if (scale_copy_owner) {
              copy(mainloop_params.gmem_tiled_copy_zero,
                   tZgZ(_,_,_,scale_load_k), tZsZ(_,_,_,write_stage));
            }
          }
        }
      }
    }
  }

  template <class... Ts>
  CUTLASS_DEVICE
  auto partition_extra_inputs(
        Params const& mainloop_params,
        cute::tuple<Ts...> const& load_inputs,
        SharedStorage& shared_tensors,
        int const scale_thread_idx,
        int const packed_thread_idx,
        bool scale_copy_owner) {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return cute::tuple{};
    }
    else if constexpr (ModeHasScales) {
      Tensor sS = make_tensor(make_smem_ptr(shared_tensors.smem_scale.begin()), SmemLayoutScaleSZ{});
      Tensor gS = get<2>(load_inputs);
      // Construct identity layout for sS
      constexpr static Tensor cS = make_identity_tensor(make_shape(size<0>(sS), size<1>(sS)));

      auto gmem_thr_copy_scale = mainloop_params.gmem_tiled_copy_scale.get_slice(scale_thread_idx);

      Tensor tSgS = gmem_thr_copy_scale.partition_S(gS);
      Tensor tSsS = gmem_thr_copy_scale.partition_D(sS);
      Tensor tScS = gmem_thr_copy_scale.partition_S(cS);

      // THE PACKED g2s, partitioned beside the fp16 one and appended to the tuple for the same return-type reason as
      // gSp. One value tile is one complete metadata column.  The owner-thread mode may be narrower than TileN;
      // CuTe then puts the additional columns in the slice's rest mode (owner t gets t, t+owners, ...).
      static constexpr int kPackedLoadIdx =
          (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) ? 4 : 3;
      Tensor sSraw = make_tensor(make_smem_ptr(shared_tensors.smem_scale_raw.begin()), SmemLayoutScaleRawStaged{});
      // Every physical thread constructs an in-range view, but only PackedMetadataOwnership's exact owner set issues
      // the asynchronous copy.  Keeping slot construction separate from publication prevents both raw OOB slices and
      // the former modulo-replayed same-address writers.
      auto gmem_thr_copy_raw = mainloop_params.gmem_tiled_copy_scale_packed.get_slice(
          packed_thread_idx);
      Tensor tSgSp = gmem_thr_copy_raw.partition_S(cute::get<kPackedLoadIdx>(load_inputs));
      Tensor tSsSp = gmem_thr_copy_raw.partition_D(sSraw);
      Tensor cSp   = make_identity_tensor(make_shape(Int<Scale_TileN>{}, Int<kPackedScaleUnit>{}));
      Tensor tScSp = gmem_thr_copy_raw.partition_S(cSp);

      // Ordinary fp16 metadata still needs predicated-copy destination initialization.  Packed metadata does not:
      // its decode owner is the sole writer of the complete destination column, including explicit zeroes for the N
      // tail below.  Keeping the historical clear only behind the legacy switch preserves the exact race negative.
      if constexpr (!kPackedScaleOn || kLegacyModuloMetadataPublishers) {
        if (scale_copy_owner) clear(tSsS);
      }

      // THE K BOUND. The fp16 path counts GROUPS from this thread's first coordinate. The packed path also keeps the
      // bound in logical groups; its source-unit quotient and tile-within-unit remainder are applied only at the copy
      // site. Thus `scale_load_k * Scale_TileK < scale_residue_k` is valid for TK64/TK128/TK256 and for nonzero
      // Split-K starts. The N bound is unchanged: mode 0 is the column in both layouts, while mode 1 is bytes here and
      // groups there.
      if constexpr (kPackedScaleOn)
        scale_residue_k = int64_t(mainloop_params.scale_k / int(Scale_TileK)) * int64_t(Scale_TileK);
      else
        scale_residue_k = mainloop_params.scale_k - get<1>(tScS(0,0,0));
      scale_valid = get<0>(tScS(0,0,0)) < scale_residue_n;
      // From the PACKED partition's own identity tensor rather than from thread_idx arithmetic: the guard and the
      // partition it guards then come from one object, which is the only form of this that cannot drift.
      if constexpr (kPackedScaleOn) {
        scale_valid_pk = get<0>(tScSp(0,0,0)) < scale_residue_n;
      }

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(tSgS, tSsS, tScS, tSgSp, tSsSp, tScSp);
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor sZ  = make_tensor(make_smem_ptr(zero_smem_base(shared_tensors)), SmemLayoutScaleSZ{});
        Tensor gZ = get<3>(load_inputs);

        auto gmem_thr_copy_zero = mainloop_params.gmem_tiled_copy_zero.get_slice(scale_thread_idx);

        Tensor tZgZ = gmem_thr_copy_zero.partition_S(gZ);
        Tensor tZsZ = gmem_thr_copy_zero.partition_D(sZ);
        // Same reason as the copies: with the packed path on, smem_zero holds zero elements.
        if constexpr (!kPackedScaleOn) {
          if (scale_copy_owner) clear(tZsZ);
        }

        return cute::make_tuple(tSgS, tSsS, tScS, tZgZ, tZsZ, tSgSp, tSsSp, tScSp);
      }
      else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                      "Conversion mode not handled for input partitioning");
      }
    }
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Input partitioning requires direct conversion or scale metadata");
    }
  }

  // ONE entry point for the scale/zero register fragment, shared by all three collectives so the construction cannot
  // drift between them. The body is the plain cute idiom, deliberately.
  //
  // AN EARLIER VERSION OF THIS FUNCTION HAND-ASSEMBLED A STRIDE-0 "BROADCAST" LAYOUT here, on the theory that the
  // fragment materialised a 4x replication of each scale (32 half slots holding 8 distinct values). acu says that is
  // a NO-OP: the slots are register-resident, the loop is fully unrolled and the equalities are provable, so the
  // compiler had already coalesced them -- and it had CSE'd the copy's filtered iterations down to the 8 distinct
  // addresses too. Measured registers were IDENTICAL either way (and 186 against estimates of 176/164, i.e. above
  // both, so the fragment is buried under address arithmetic anyway).
  //
  // It was reverted for a second reason that outlives the measurement: that version replaced a one-line cute idiom
  // with a hand-built stride tuple that hardcoded "mode 2 is MMA_K" by position. Less cute-native, rank-fragile, and
  // it bought nothing. A cute layout describes what the PROGRAM asks for; whether the hardware does it is a codegen
  // question, and for register-resident provably-equal values the compiler wins first. Check that a cute-level
  // redundancy survives to the ISA before trading idiom for it.
  template <class TiledMma, class STensor>
  CUTLASS_DEVICE
  static auto make_scale_fragment(TiledMma const& thr_mma, STensor const& sS) {
    return MetadataPolicy::make_fragment(thr_mma, sS(_,_,Int<0>{}));
  }

  // The same layout, HOST-callable, for the compile-time witness below (make_scale_fragment is CUTLASS_DEVICE and
  // cannot appear even unevaluated in a host constexpr context). make_fragment_like<T>(t) is
  // make_tensor<T>(make_layout_like(t.layout())), so these two stay in step by construction.
  template <class TiledMma, class STensor>
  CUTE_HOST_DEVICE static constexpr auto scale_fragment_layout(TiledMma const& thr_mma, STensor const& sS) {
    return MetadataPolicy::fragment_layout(thr_mma, sS(_,_,Int<0>{}));
  }

  // Kept from the reverted work because it has independent value: it is what let a STALE submodule be distinguished
  // from an inert change. cosize is the scale fragment's real register footprint in halves.
  static constexpr int scale_frag_cosize() {
    if constexpr (ModeHasScales) {
      return cute::cosize_v<decltype(scale_fragment_layout(
          TiledMma{}.get_thread_slice(0),
          make_tensor(make_smem_ptr((NonVoidElementScale*)nullptr), SmemLayoutScaleSZ{})))>;
    } else {
      return 0;
    }
  }

  // ---------------------------------------------------------------------------------------------------------------
  // THE PACKED SCALE CHANNEL'S DEVICE STEPS. Defined unconditionally: they are templates, so an off unit never
  // instantiates them, and the call sites select with `if constexpr (kPackedScaleOn)` per UNIT rather than per binary.
  // THE PACKED SCALE CHANNEL'S TWO DEVICE STEPS. Derived facts, all from fold_derivation/l94 on the collective's own
  // objects (l95 proves the probe's mma IS this one):
  //   * a lane's scale fragment touches exactly kPackedSlots DISTINCT columns -- 2 at TN=128/w16x16, the same for all
  //     256 lanes -- so the lane holds that many 16 B units and nothing more.
  //   * value -> slot is periodic with period 8 and is `(v >> 2) & 1` at this config; expressed here as a division by
  //     the run length rather than a stored table, and CHECKED at runtime against the coordinate tensor below, because
  //     the run length is the one number that changes with the warp shape.
  //   * every value sharing a slot shares its column, hence its scale: per group a lane decodes kPackedSlots values,
  //     not one per fragment element.
  static constexpr int kPackedSlots = 2;
  // Q4_K for now: unsigned codes with no centre, and a min channel. Q3_K would be <32, false> and Q6_K
  // <0, false>; they are template parameters of group_of precisely so no second decode is written.
  static constexpr int  kPackedScaleBias = PackedUnit::kScaleBias;
  static constexpr bool kPackedHasMin    = PackedUnit::kHasMin;
  // 8 cancels the int4 converter's own -8, which this path leaves in place: see group_of's comment for why that is the
  // better of the two ways to reconcile them.
  // ZMul CANCELS THE WEIGHT CONVERTER'S OWN SHIFT, so it follows the weight's element width -- and writing it as a
  // literal 8 was wrong for exactly the reason I gave for keeping it one. The int4 converter emits q-8, so int4
  // needs 8; the uint2 converter emits q in [0,3] with NO bias ("the per-group affine 'zero' term absorbs the
  // offset", fast_numeric_conversion_for_mix_gemm.h at the W2A16 specialisation), so a 2-bit weight needs 0.
  // Q2_K's weights are 2-bit, so the literal would have shifted every one of them by 8*scale.
  static constexpr int  kPackedZMul      =
      (cute::sizeof_bits_v<RealInternalElementB> == 4) ? 8 : 0;
  // ROUNDED UP. Q3_K's unit is 14 bytes and Q6_K's 18, i.e. 3.5 and 4.5 words, and truncating loses the tail --
  // which for Q3_K is groups 12..15 and for Q6_K the last two scales, read as zero. The staging tile is padded to
  // whole words for the same reason, so reading the extra bytes is in-bounds.
  static constexpr int kPackedUnitWords = (kPackedScaleUnit + 3) / 4;

  // THE PACKED (f16x2) PER-GROUP DECODE, which needs both fields to pack against each other and the 6-bit unsigned
  // extraction code_pair_from_words performs. Q4_K is the only format with a min, so this is exactly Q4_K today; the
  // scalar group_of_words stays for everything else and is what the `else` arm below still calls. The bias/mask/OR
  // identity it rests on is NOT restricted to unsigned codes -- l96 (A0) pinned its true bound at [-128, 895] -- so a
  // future signed format only needs its own extraction, not its own arithmetic.
  //
  // PPU_PACKED_PAIR=0 FORCES THE SCALAR DECODE BACK, and it exists to BISECT rather than to tune. rowC of
  // test_q4k_packed_gemm -- the only row where kPackedScaleOn is true -- regressed when the packed pair landed, and
  // the two candidate causes cannot be separated by reading: (i) the f16x2 asm, which has zero local coverage because
  // the local gate compiles under nvcc where the scalar fallback is selected, and (ii) anything else the same commits
  // touched. One build each answers it: PAIR=0 restoring MATCH indicts the packed arithmetic/asm, PAIR=0 still failing
  // exonerates it and points at the rest of the change.
#if defined(PPU_PACKED_PAIR) && (PPU_PACKED_PAIR == 0)
  static constexpr bool kPackedPairFast = false;
#else
  // AND SIX-BIT FIELDS. The pair path folds the field width into kMagic1152x2, so Q2_K (4 bits) and Q6_K
  // (8 bits) must take the scalar arm -- which they would otherwise enter, since Q2_K has a min and a zero
  // bias and satisfies the old condition exactly.
  static constexpr bool kPackedPairFast = kPackedHasMin && (kPackedScaleBias == 0)
                                      && (PackedUnit::kScaleBits == 6) && (PackedUnit::kMinBits == 6);
#endif


  // ON IS A TEMPLATE PARAMETER, and that is the whole point: `if constexpr` only skips INSTANTIATING a discarded branch
  // when its condition is value-dependent. With kPackedScaleOn (a class constant) both branches were instantiated, so
  // every unit paid for the packed tensors even with the path off -- measured as ~5% on the TK=64 control rows, which are
  // supposed to be byte-identical between the two builds. Routing the gate through a template parameter makes the
  // discarding real.

  // THE DECODING LOADER: gmem native unit -> registers -> the SAME fp16 smem planes the fp16 path uses.
  //
  // This is llama.cpp's shape (ggml-cuda/mmq.cuh, load_tiles_q4_K) and the reason it is right is amortisation, not
  // instruction count in isolation: there the decode runs once per (row, group) for the whole CTA and the result goes to
  // SHARED memory, so every lane that needs it reads it rather than re-deriving it. Doing it per lane in registers, which
  // is what the previous attempt did, multiplies the work by the number of lanes sharing a column (four, per l94 (7))
  // times the warps -- measured as ~1.4M extra ALU against 0.27M shared loads saved, and LSU was only 6% busy.
  //
  // Here: each owner thread reads every complete 16 B column unit in its TiledCopy slice (d, dmin and all 8 groups'
  // 6-bit codes), decodes with compile-time bit positions, and stores fp16 scale and zero. 64 columns x 8 groups =
  // 512 decodes per CTA per k-tile, independent of whether 32 or 64 threads own those columns.
  //
  // WHAT IT COSTS: this channel is no longer cp.async, because arithmetic between gmem and smem is exactly what cp.async
  // cannot do. The load is issued where the cp.async used to be and the barrier that already guards the stage
  // (cp_async_wait + __syncthreads at the last k_block) still guards it, so no new sync is added -- but the LDG's latency
  // is now exposed rather than overlapped, and only measurement says whether the B cp.async in flight covers it.
  // DECODE ONE STAGE, smem -> smem, at the barrier the pipeline already has.
  //
  // llama.cpp's shape (ggml-cuda/mmq.cuh, load_tiles_q4_K): the decode runs once per (column, group) for the whole CTA
  // and the result goes to SHARED memory, so every lane reads it instead of re-deriving it. The per-lane register version
  // did the opposite and paid ~1.4M extra ALU to save 0.27M shared loads, with LSU at 6% busy.
  //
  // Called AFTER cp_async_wait (the staged bytes have landed) and BEFORE __syncthreads (the planes are still private, and
  // the sync publishes them), so it adds no barrier of its own. mma() already holds shared_tensors and thread_idx, which
  // is why this needs no plumbing through any tuple.
  template <bool On, class Storage>
  CUTLASS_DEVICE static void
  packed_decode_stage(Storage& storage, int stage, int tile_in_unit,
                      int thread_idx, int64_t residue_n) {
    if constexpr (On) {
      constexpr int kGrp     = int(Scale_TileK);
      Tensor sRaw = make_tensor(make_smem_ptr(storage.smem_scale_raw.begin()), SmemLayoutScaleRawStaged{});
      Tensor sS   = make_tensor(make_smem_ptr(storage.smem_scale.begin()),  SmemLayoutScaleSZ{});
      Tensor sZ   = make_tensor(make_smem_ptr(zero_smem_base(storage)),     SmemLayoutScaleSZ{});
      // THE FUSED STORE'S OWN VIEW: 32-bit slots, stride 1 in n. Built beside sS/sZ rather than instead of them so
      // the unfused arm below is byte-identical and the two cannot drift apart.
      Tensor sSZw = make_tensor(make_smem_ptr(reinterpret_cast<uint32_t*>(storage.smem_scale.begin())),
                                SmemLayoutScaleFusedWord{});
      (void)sSZw;
      // ONE OWNER SLICE, ITS OWN COLUMN SET. cp_async_wait is PER THREAD and the __syncthreads that publishes this
      // stage comes AFTER this function, so a thread may only read bytes its own TiledCopy slice issued. When the CTA
      // is narrower than TileN, that slice owns columns t, t+owners, ...; the shared PackedMetadataOwnership type is
      // the sole copy/decode map.
      //
      // MEASURED, NOT ARGUED: a version of this loop paired two adjacent columns per thread to turn the 2 B store into
      // a 4 B one (worth ~0.6%: acu put the scalar form at +71,680 tsm.st with +73,728 bank conflicts and 1.83
      // transactions per instruction, since 32 lanes x 2 B covers 16 of 32 banks). It read column 2p+1, which thread
      // 2p+1 copied, and test_q4k_packed_gemm's rowC -- the ONLY row where kPackedScaleOn is true -- went to
      // bad=128/4096 with the failures concentrated in odd columns. rowA and rowB kept passing because at TK=128 they
      // are on the fp16 path, which is exactly why the gate has a row per Scale_TileK.
      //
      // To get the wide store back, the COPY must be paired too (thread layout Scale_TileN/2 over a 32 B value layout)
      // so the two maps coincide again; the alternative, a __syncthreads before the decode, costs more than the store
      // saves (8 barriers per CTA against 0.6%).
      //
      constexpr int kOwnTh  = int(cute::size(GmemTiledCopyScalePacked{}));
      // ONE THREAD, ITS OWN COLUMNS. At cpt == 1 this is the original loop unchanged. For cpt > 1 the TiledCopy value
      // mode repeats outside the owner-thread mode, so owner t receives t + sub*owners -- not adjacent columns.
      constexpr int kCPT = kPackedColsPerThread;
      if (thread_idx >= kOwnTh) return;
      int owner = thread_idx;
      if constexpr (kCPT > 1) owner = PackedMetadataOwnership::copy_owner(thread_idx);
      CUTLASS_PRAGMA_UNROLL
      for (int sub = 0; sub < kCPT; ++sub) {
        int const n = PackedMetadataOwnership::column(owner, sub);
        // PACKED METADATA TOTAL-OVERWRITE CONTRACT.  The copy/decode owner also owns tail initialization, so there is
        // one writer map and no clear/decode race.  A full tile takes the decode arm and executes no clear; on the last
        // N tile the same owner writes every skipped destination explicitly without reading the unfilled raw slot.
        if (int64_t(n) >= residue_n) {
          cute::for_each(cute::make_int_sequence<kGrp>{}, [&](auto g_) {
            constexpr int G = decltype(g_)::value;
            if constexpr (kFusedScaleZero) {
              sSZw(n, cute::Int<G>{}, stage) = uint32_t(0);
            } else {
              sS(n, cute::Int<G>{}, stage) = NonVoidElementScale{};
              if constexpr (kPackedHasMin) {
                sZ(n, cute::Int<G>{}, stage) = NonVoidElementZero{};
              }
            }
          });
          continue;
        }
        uint8_t const* unit = reinterpret_cast<uint8_t const*>(&sRaw(n, cute::Int<0>{}, stage));
        uint32_t u[kPackedUnitWords];
        // THE LAST WORD MAY BE PARTIAL. With a unit that is not a multiple of four bytes the final read would run
        // past it, so the tail is assembled byte by byte -- the bytes beyond the unit are never referenced by any
        // field, and reading them would be out of bounds on the last column of the tile.
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < kPackedUnitWords; ++w) {
          if constexpr (kPackedScaleUnit % 4 == 0) {
            u[w] = *reinterpret_cast<uint32_t const*>(unit + 4 * w);
          } else {
            uint32_t acc = 0;
            CUTLASS_PRAGMA_UNROLL
            for (int b = 0; b < 4; ++b) {
              int const idx = 4 * w + b;
              if (idx < kPackedScaleUnit) acc |= uint32_t(unit[idx]) << (8 * b);
            }
            u[w] = acc;
          }
        }
        // half2(d, -dmin) for the whole column: one xor, hoisted out of the group loop. Replaces head_of_words'
        // two bitcasts AND the per-group negate of the min term.
        uint32_t const m2 = cutlass::gguf_packed::mul2_of_words(u);
        auto const h = cutlass::gguf_packed::head_of_words(u);
        (void)m2; (void)h;
        // ONE body, called from both halves with a compile-time G. The group index has to be a template argument --
        // every bit position in the unit is derived from it -- so the two halves cannot share a runtime offset.
        auto decode_group = [&] (auto source_g_, auto destination_g_) {
          constexpr int SourceG = decltype(source_g_)::value;
          constexpr int G = decltype(destination_g_)::value;
          // TIMING-ONLY ABLATION. PPU_PACKED_SCALE_NOP=1 keeps the native transport and the shared STORES but drops
          // the decode ARITHMETIC, so three builds decompose the +12.9% instead of attributing all of it at once:
          //     baseline B   fp16 planes, cp.async writes them, no decode
          //     nop      N   native 16 B transport + the same stores, no arithmetic
          //     full     P   everything
          // giving arithmetic = P - N and transport+stores = N - B. RESULTS ARE DELIBERATELY WRONG under this flag.
          // The store still consumes the unit's first half so the 16 B smem load cannot be dead-code eliminated --
          // an ablation the compiler optimises away measures the compiler, not the kernel.
          //
          // A switch by this name was DOCUMENTED in PLAN_task20_scale.md and HANDOFF_packed_scale.md while its code
          // no longer existed: it did not survive the F/F' rewrites and nothing noticed, because a timing flag has no
          // gate that fails. Same defect shape as the rest of this task, one level up.
#if defined(PPU_PACKED_SCALE_NOP) && (PPU_PACKED_SCALE_NOP != 0)
          // NORMAL VALUES, NOT d AND dmin. Writing the unit's own d and dmin kept the 16 B load alive but made the
          // planes depend on the input, and d IS subnormal for 80.5% of superblocks on the real fixture.
          //
          // THAT IS NOT WHY packnop CAME OUT SLOWER, and I asserted it was. test_moe_splitk_bench fills its scale
          // buffer with a constant 0.0625 and its zero buffer with -0.0625 and hands the same allocation to the
          // packed path, which reinterprets it as 16-byte units -- so the bench's d and dmin are 0.0625, normal, and
          // the fixture's subnormals never reach it. Second time I attached that measured fact to the wrong effect.
          // The packnop anomaly is still unexplained; the leading candidates are timing noise at 2.3% against a
          // documented 13% dispersion, and a different compiler schedule, since the full decoder consumes all four
          // u[] words while this consumes only u[0] and the other shared loads may simply be eliminated.
          //
          // The change is kept because input-independent constants are the right shape for an ablation regardless --
          // it just does not buy what the first version of this comment claimed.
          //
          // 0x3C00 is 1.0 and 0x0000 is +0; ORing one bit of u[0] into the mantissa keeps the load from being dead
          // while staying firmly normal. The planes are still wrong on purpose -- read the time, never the MATCH.
          cutlass::gguf_packed::GroupScale sz;
          sz.scale = cutlass::half_t::bitcast(uint16_t(0x3C00u | (u[0] & 1u)));
          sz.zero  = cutlass::half_t::bitcast(uint16_t((u[0] >> 16) & 1u));
          if (false)
#else
          cutlass::gguf_packed::GroupScale sz;
#endif
          if constexpr (kPackedPairFast) {
            // BOTH FIELDS OF THE GROUP IN ONE 32-BIT LANE PAIR: one integer add carries the bias, the mask and the
            // magic OR for scale AND min together, then one ppu.sub.f16x2 and one ppu.fma.rtte.f16x2. 15 opcodes per
            // group down to ~11, and bit-identical rather than close -- l96 (A) checks that over 32768 real Q4_K
            // groups and (A0) checks each of the four identities it rests on separately. This touches only the
            // thread's OWN column, so it is independent of the constraint above.
            sz = cutlass::gguf_packed::group_pair_of_words<SourceG, kPackedZMul, kPackedScaleBias>(u, m2);
          } else {
            sz = cutlass::gguf_packed::group_of_words<
                SourceG, kPackedScaleBias, kPackedHasMin, kPackedZMul, kPackedFmt>(u, h);
          }
          // (n, group, stage): SmemLayoutScale's own modes. NOT the read side's flattened (n, 1, stage*SK+g) -- two
          // functions build a tensor called sS with DIFFERENT layouts, and using the wrong one faulted as
          // "TSM out of range" once already.
          if constexpr (kFusedScaleZero) {
            // ONE 32-BIT STORE. scale in the low half, zero in the high half -- the order the readers' even/odd views
            // above assume, and the order a little-endian half2 already has.
            //
            // sz.zero IS THE VALUE TO STORE, NOT y2. group_pair_of_words computes half2(d*sc, -dmin*mn) and then adds
            // `kPackedZMul * scale` to the zero AFTER splitting it (gguf_packed_scale.h, the ZMul arm), which cancels
            // the int4 converter's own -8. Packing the pre-correction pair back into 32 bits looks like it saves the
            // split and is simply WRONG -- it drops that cancellation.
            sSZw(n, cute::Int<G>{}, stage) = cutlass::gguf_packed::pack_h2(sz.scale, sz.zero);
          } else {
            sS(n, cute::Int<G>{}, stage) = sz.scale;
            if constexpr (kPackedHasMin) sZ(n, cute::Int<G>{}, stage) = sz.zero;
          }
        };
        if constexpr (kPackedTilesPerUnit == 1) {
          // Preserve the historical one-superblock path without a runtime phase branch.
          cute::for_each(cute::make_int_sequence<kGrp>{}, [&] (auto i_) {
            constexpr int G = decltype(i_)::value;
            decode_group(cute::Int<G>{}, cute::Int<G>{});
          });
        } else {
          // The pipeline phase is runtime state, while GGUF bit positions are template constants. Enumerating at most
          // four phases gives both: exactly one branch runs and each selected source group remains compile-time.
          cute::for_each(cute::make_int_sequence<kPackedTilesPerUnit>{}, [&] (auto tile_) {
            constexpr int Tile = decltype(tile_)::value;
            if (tile_in_unit != Tile) return;
            constexpr int GroupBase =
                (Tile % kPackedTilesPerSb) * int(Scale_TileK);
            cute::for_each(cute::make_int_sequence<kGrp>{}, [&] (auto i_) {
              constexpr int G = decltype(i_)::value;
              decode_group(cute::Int<GroupBase + G>{}, cute::Int<G>{});
            });
          });
        }
      }
    }
  }


  // ON IS A TEMPLATE PARAMETER, and that is the whole point: `if constexpr` only skips INSTANTIATING a discarded branch
  // when its condition is value-dependent. With kPackedScaleOn (a class constant) both branches were instantiated, so
  // every unit paid for the packed tensors even with the path off -- measured as ~5% on the TK=64 control rows, which are
  // supposed to be byte-identical between the two builds. Routing the gate through a template parameter makes the
  // discarding real.

  // THE DECODING LOADER: gmem native unit -> registers -> the SAME fp16 smem planes the fp16 path uses.
  //
  // This is llama.cpp's shape (ggml-cuda/mmq.cuh, load_tiles_q4_K) and the reason it is right is amortisation, not
  // instruction count in isolation: there the decode runs once per (row, group) for the whole CTA and the result goes to
  // SHARED memory, so every lane that needs it reads it rather than re-deriving it. Doing it per lane in registers, which
  // is what the previous attempt did, multiplies the work by the number of lanes sharing a column (four, per l94 (7))
  // times the warps -- measured as ~1.4M extra ALU against 0.27M shared loads saved, and LSU was only 6% busy.
  //
  // Here: one thread per column reads its 16 B unit (d, dmin and all 8 groups' 6-bit codes), decodes with the bit
  // positions as compile-time constants, and stores the fp16 scale and zero. 64 columns x 8 groups = 512 decodes per CTA
  // per k-tile, against 2048 for the per-lane version.
  //
  // WHAT IT COSTS: this channel is no longer cp.async, because arithmetic between gmem and smem is exactly what cp.async
  // cannot do. The load is issued where the cp.async used to be and the barrier that already guards the stage
  // (cp_async_wait + __syncthreads at the last k_block) still guards it, so no new sync is added -- but the LDG's latency
  // is now exposed rather than overlapped, and only measurement says whether the B cp.async in flight covers it.
  /// Utilities for partitioning extra inputs for loading from smem in the mainloop.
  template <class TiledMma>
  CUTLASS_DEVICE
  auto partition_extra_mma_info(
    TiledMma const& tiled_mma,
    SharedStorage& storage,
    int thread_idx) {

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // noting to do
      return cute::tuple{};
    }
    else if constexpr (ModeHasScales) {
      auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
      auto smem_tiled_copy_S   = make_tiled_copy_B(SmemCopyAtomScale{}, tiled_mma);
      auto smem_thr_copy_S     = smem_tiled_copy_S.get_thread_slice(thread_idx);

      static constexpr int smem_scale_k = Scale_TileK * DispatchPolicy::Stages;
      // THE SAME swizzle as SmemLayoutScale, for the reason spelled out there: one buffer, two views, and they are
      // equal only while both carry it.
      using PlainSmemCopyLayoutScale = decltype(tile_to_shape(SmemLayoutAtomScale{},
          make_shape(shape<0>(ScaleTileShape{}), Int<1>{}, Int<smem_scale_k>{})));
      using UnfusedSmemCopyLayoutScale =
          typename detail::MaybeScaleSwizzle<kScaleSwizzleOk, ScaleSwizzleT, PlainSmemCopyLayoutScale>::type;
      // THE READ SIDE'S FLATTENED VIEW, FUSED. Same shape (n, 1, stage*Scale_TileK + g), strides doubled, so a reader
      // written against the unfused layout keeps its code and only walks 4 bytes per element instead of 2. Checked
      // against the unfused layout rather than assumed, for the reason given at SmemLayoutScaleFusedWord: this is the
      // SECOND view of the same buffer, and the two going out of step is the documented failure here.
      static_assert(sz_copy_layout_is_compact<kFusedScaleZero, UnfusedSmemCopyLayoutScale>(),
                    "PPU_PACKED_SCALE_FUSED assumes the compact flattened scale copy layout");
      using FusedSmemCopyLayoutScale = decltype(make_layout(
          make_shape(Int<kSZ_N>{}, Int<1>{}, Int<smem_scale_k>{}),
          make_stride(_2{}, _0{}, Int<2 * kSZ_N>{})));
      using SmemCopyLayoutScale =
          cute::conditional_t<kFusedScaleZero, FusedSmemCopyLayoutScale, UnfusedSmemCopyLayoutScale>;
      Tensor sS   = make_tensor(make_smem_ptr(storage.smem_scale.begin()), SmemCopyLayoutScale{});
      Tensor tCsS = smem_thr_copy_S.partition_S(sS);
      Tensor tCrS = make_scale_fragment(thr_mma, sS);

      // APPENDED, never inserted: every existing site reads this tuple positionally (get<0>..get<3>), so the packed
      // extras go after them and no index moves. Appended UNCONDITIONALLY so the arity never depends on a macro -- with
      // `if constexpr` selecting the fill, a discarded branch still has to name get<4>, and a macro-dependent arity
      // would make that ill-formed on the other configuration.
      // Built only when the path is on; an empty tuple otherwise, so an off unit constructs nothing at all. The
      // reinterpret is needed because with the path off that engine is half_t.
      // Built unconditionally. packed_or_empty (a template-parameter gate around a generic lambda) is the standard way
      // to avoid that, and EDG instantiates the lambda body anyway -- so the empty stand-in reached partition_S. Its
      // cost is measured, not guessed: ~1.5 us on the TK=64 control rows. Second-order next to the decode, and revisited
      // only after the decode's win is on the table; one variable at a time.
      // POINTED AT THE BUFFER ITS LAYOUT DESCRIBES. This tensor is dead -- ScaleOnly puts it at tuple index 2 and that
      // arm reads only 0 and 1; ScaleZero puts it at 4 and nothing reads past 3 -- but it was built on smem_scale, the
      // fp16 plane, with SmemLayoutScalePackedStaged, which describes the 16-byte staging. That is a leftover from
      // when the staging lived inside smem_scale, and it is exactly the kind of thing that is revived and then wrong
      // twice over: wrong buffer, and (since PPU_SCALE_SWIZZLE) an unswizzled view of a buffer every live view now
      // swizzles. Repointed rather than deleted so the tuple indices, which every consumer reads positionally, do not
      // move. Found by enumerating every tensor built on smem_scale/smem_zero -- six of them, where I had claimed two.
      Tensor sSp  = make_tensor(make_smem_ptr(reinterpret_cast<uint8_t*>(
                                    kPackedScaleOn ? (void*)storage.smem_scale_raw.begin()
                                                   : (void*)storage.smem_scale.begin())),
                                SmemLayoutScalePackedStaged{});
      Tensor tCcS = smem_thr_copy_S.partition_S(make_identity_tensor(shape(sS)));
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(tCsS, tCrS, sSp, tCcS);
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor sZ   = make_tensor(make_smem_ptr(zero_smem_base(storage)), SmemCopyLayoutScale{});
        Tensor tCsZ = smem_thr_copy_S.partition_S(sZ);
        Tensor tCrZ = make_scale_fragment(thr_mma, sZ);
        return cute::make_tuple(tCsS, tCrS, tCsZ, tCrZ, sSp, tCcS);
      }
      else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                      "Conversion mode not handled while partitioning scale fragments");
      }
    }
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Scale-fragment partitioning requires direct conversion or scale metadata");
    }
  }

  /// Returns the tiled copy and copy views for the extra inputs.
  template <class TiledMma, class... Ts>
  CUTLASS_DEVICE
  auto retile_extra_mma_info(
    TiledMma const& tiled_mma,
    cute::tuple<Ts...>& partitioned_extra_info,
    int const thread_idx) {

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // noting to do
      return cute::tuple{};
    }
    else if constexpr (ModeHasScales) {
      auto smem_tiled_copy_S = make_tiled_copy_B(SmemCopyAtomScale{}, tiled_mma);
      auto smem_thr_copy_S   = smem_tiled_copy_S.get_thread_slice(thread_idx);
      Tensor tCrS_copy_view  = smem_thr_copy_S.retile_D(cute::get<1>(partitioned_extra_info));        // (CPY,CPY_N,CPY_K)

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(smem_tiled_copy_S, tCrS_copy_view);
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor tCrZ_copy_view  = smem_thr_copy_S.retile_D(cute::get<3>(partitioned_extra_info));      // (CPY,CPY_N,CPY_K)
        return cute::make_tuple(smem_tiled_copy_S, tCrS_copy_view, tCrZ_copy_view);
      }
      else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                      "Conversion mode not handled while retiling scale fragments");
      }
    }
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Scale-fragment retiling requires direct conversion or scale metadata");
    }
  }

  /// Utilities to copy B and extra inputs from smem to RF
  template <class SmemTiledCopy,
            class TensorSmemView,
            class TensorCopyView,
            class... Ts,
            class... Us
            >
  CUTLASS_DEVICE
  void copy_B_and_extra_info(
    SmemTiledCopy const& smem_tiled_copy_B,
    TensorSmemView const& tCsB,
    TensorCopyView& tCrB_copy_view,
    cute::tuple<Ts...> const& partitioned_mma_extra_info,
    cute::tuple<Us...> const& tiled_copy_and_views,
    int k_block,
    int read_stage) {

    copy(smem_tiled_copy_B, tCsB(_,_,k_block,read_stage), tCrB_copy_view(_,_,k_block));

    // COARSE is a relation between the scale tile and the ACTUAL retiled B copy view. It is not a group-size rule:
    // KBM_ also depends on SmemCopyAtomB and tiled_mma_s8. When the scale tile has no more K slots than the copy
    // view, one scale covers one or more whole copy steps and is loaded here once per GroupK steps. Otherwise the
    // FINE path reloads scale per MMA atom in transform_B_kblock.
    constexpr int KBM_ = decltype(cute::size<2>(tCrB_copy_view))::value;
    using ScalePolicy = typename MetadataPolicy::template Coarse<KBM_>;
    if constexpr (ScalePolicy::active) {
     if (ScalePolicy::starts_group(k_block)) {
      // We are starting a new group k-tile so copy the scale
      if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
        // nothing to do
      }
      else if constexpr (ModeHasScales) {
        MetadataPolicy::reload(partitioned_mma_extra_info, tiled_copy_and_views,
                               ScalePolicy::group(k_block), read_stage);
      }
      else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                      "COARSE scale copy requires a scale-bearing conversion mode");
      }
     }
    }
  }
  /// A's first R rows into the PACKED cube layout. Each row has four contiguous 16-half runs at the offsets l86
  /// exported. Logical k advances inside each 8-half transfer; odd cache lines swap the two transfers in a run.
  /// cp.async and not the AIU: the AIU write is .padz and would write each cube's 15 zero rows over its packed
  /// neighbour's real rows. kAWrThreads counts 128-bit transfers; threads stride that logical copy domain when R
  /// needs more transfers than the CTA has threads. Rows between the problem's M and R are zero-filled just as the
  /// ordinary .padz AIU write did -- reading them from GMEM would make an M<R launch walk past A.
  template <int R, class GA, class EA>
  CUTLASS_DEVICE
  void copy_A_packed_rows(
      GA const& gA, EA* smem_a, int k_tile, int pipe, int thread_idx, int valid_rows) {
    static_assert(R == kAPackRows, "packed-A writer and allocation must use the same row count");
    constexpr int kThreads = int(cute::size(TiledMma{}));
    int const per_cube = R * kASlices * 2;
    for (int logical_thread = thread_idx; logical_thread < kAWrThreads; logical_thread += kThreads) {
      int const c   = logical_thread / per_cube;                    // which cube in this stage
      int const row = (logical_thread % per_cube) / (kASlices * 2); // which row slot (real or zfill)
      int const run = (logical_thread % (kASlices * 2)) / 2;        // which of this row's four runs
      int const h   = logical_thread % 2;                            // which logical 8-half half of the run
      int const physical_h = h ^ ((row / 4) & 1);                   // l86: odd lines swap the half-runs
      // The raw op, not a Copy_Atom: the atom's SrcLayout only matches when reached through a TiledCopy, and this
      // thread -> (cube,row,run,half) map is not a tiler. ZFILL preserves the AIU .padz contract for M < R.
      auto const& gsrc = *reinterpret_cast<cute::uint128_t const*>(
          &gA(row, c * kACubeW + run * 16 + h * 8, k_tile));
      auto&       sdst = *reinterpret_cast<cute::uint128_t*>(
          smem_a + kAPackPitch * c + kAPackStagePitch * pipe +
          detail::aPackRunOffsetHalfs(kACubeH, row, run) + physical_h * 8);
      PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>::copy(gsrc, sdst, row < valid_rows);
    }
  }

  // PPU_B_DEQUANT_NOP -- TIMING ONLY, RESULTS ARE DELIBERATELY WRONG. It answers the one question the packed-scale
  // NOP cannot: how much of the 20.11 us baseline is the int4->fp16 dequant pipeline itself? That chain is 1,898,496
  // instructions against 131,072 mma, i.e. 43% of dynamic instruction count -- but an instruction share is not a
  // cycle share, and those instructions can issue while memory operations are outstanding. Only removing them says.
  //
  // WHAT IT KEEPS, because an ablation that changes memory traffic answers a different question: the B s2r loads
  // (cvt_in stays live), the scale/zero smem reads (one element of each fragment is still consumed, so the copies
  // cannot be dead-code eliminated), the mma count, the tile shapes and every barrier. What it drops: the conversion
  // for all but one word, and N-1 of the N elementwise scale/zero applications per atom.
  //
  // The one-element form is deliberate. Skipping the transforms entirely would let the compiler delete the scale
  // copies with them, and the run would then measure a kernel that also stopped reading shared memory -- the same
  // trap PPU_PACKED_SCALE_NOP avoids by keeping its unit load consumed.
  template <class TA, class TB, class TC, class Op>
  CUTLASS_DEVICE static void bdq_transform(TA&& a, TB&& b, TC&& c, Op op) {
#if defined(PPU_B_DEQUANT_NOP) && (PPU_B_DEQUANT_NOP != 0)
    c(0) = op(a(0), b(0));
#else
    cute::transform(a, b, c, op);
#endif
  }

  /// Utilities to transform B.
  // FINE-grained scale (gs < B-copy-step K, i.e. Scale_TileK > K_BLOCK_MAX): a single copy step's K_ATOM_PER_COPY
  // mma atoms straddle MORE than one scale group, so one pre-loaded scale reg can't cover the step (and the coarse
  // GroupK = K_BLOCK_MAX/Scale_TileK is 0). Here each mma atom reloads ITS group's scale straight from smem:
  // atom `a` (0..mma_K_atoms) belongs to group a/(mma_K_atoms/Scale_TileK), at smem slot read_stage*Scale_TileK+g.
  // This needs tiled_copy_and_views (smem_tiled_copy_S + reg-copy dst views) + read_stage, so both are passed in.
  template <typename RealInternalElementB,
            class TCrB_load,
            class TCrB_mma,
            int K_ATOM_PER_COPY,
            class... Ts,
            class CopyViews,
            class KBlockT,
            class PfPack>
  CUTLASS_DEVICE
  void transform_B_kblock(
    TCrB_load const& tCrB_load,
    TCrB_mma& tCrB_mma,
    cute::tuple<Ts...>& partitioned_extra_info,
    // KBlockT, NOT int: the callers already hold a static k_block (for_each gives Int<x>, and k_block_next is
    // (Int<x> + _1) % K_BLOCK_MAX, also static). Taking it as an int erased that, so atom_idx, g = atom_idx/APG_
    // and the `% APG_ == 0` guard all turned into runtime work -- s.cmp 0.54 + s.csel 0.41 + s.cbr 0.56 +
    // s.shra 0.09 + s.mull 0.16 per mma in the profile, ~1.8 of 34. Kept static, the guard folds to if constexpr
    // and vanishes with the division and modulo. Constant folding only; numerics unchanged.
    KBlockT const& k_block,
    cute::Int<K_ATOM_PER_COPY> k_atom,
    CopyViews const& tiled_copy_and_views,
    int const read_stage,
    // The second scale/zero register set, or an empty tuple. A separate template parameter and NOT an extension of
    // the Ts... pack above: appending to cute::tuple<Ts...> fails deduction.
    PfPack const& pf) {

    static constexpr int K_BLOCK_STATIC = int(KBlockT{});
    Tensor cvt_in  = recast<RealInternalElementB>(tCrB_load(_, _, k_block));
    Tensor cvt_out = [&] {
      if constexpr (kQ4KPack4Transpose) {
        // K-pack4's physical transposed-b16 loader and logical m8 MMA fragment
        // have the same 32-value converter cohort, but not always the same N
        // stride.  Reusing cvt_in.layout() here made distinct (N16,K64)
        // cohorts alias whenever the loader's N stride was below the compute
        // fragment's 128-half stride (for example TN32/WN32: cohort bases
        // 0,1,1,2,... instead of 0,4,1,5,...).  L231 composes the real
        // TiledMMA/SmemLayoutB types across all admitted TN/WN geometries and
        // proves that only the cohort base differs; the 32-value converter
        // order itself remains exact.
        //
        // Preserve cvt_in's first mode -- it is the proven int4 converter
        // emission order -- and derive a compact N layout beginning at the
        // compute fragment's own N stride.  Nested physical N modes such as
        // (2,2) then become (128,256), rather than inheriting (32,256).
#if defined(PPU_Q4_KPACK4_LEGACY_LOADER_OUTPUT_LAYOUT) && \
    (PPU_Q4_KPACK4_LEGACY_LOADER_OUTPUT_LAYOUT != 0)
        return make_tensor(
            tCrB_mma(_, _, k_block * K_ATOM_PER_COPY).data(),
            cvt_in.layout());
#else
        auto dst_n_stride = compact_col_major(
            shape<1>(cvt_in.layout()), stride<1>(tCrB_mma.layout()));
        auto dst_layout = make_layout(
            shape(cvt_in.layout()),
            make_stride(stride<0>(cvt_in.layout()), dst_n_stride));
        return make_tensor(
            tCrB_mma(_, _, k_block * K_ATOM_PER_COPY).data(), dst_layout);
#endif
      } else {
        return make_tensor(
            tCrB_mma(_, _, k_block * K_ATOM_PER_COPY).data(),
            cvt_in.layout());
      }
    }();

    using CPY_VEC = Int<4 * 32 / sizeof_bits<RealInternalElementB>::value>;
#if defined(PPU_B_DEQUANT_NOP) && (PPU_B_DEQUANT_NOP != 0)
    // Nothing. The earlier version copied one 32-bit word of cvt_in into cvt_out to keep the B load alive, on the
    // belief that the rest of cvt_out held the previous k-tile's converted halfs -- it does not, the fragment is
    // never initialised, and those raw int4 bits read as fp16 are NaN or Inf about as often as not. The B s2r does
    // not need that crutch: it is a TSM_LD_SWZL implemented as asm volatile, so it survives its results going unused.
    (void)cvt_in; (void)cvt_out;
#else
    convert_tensor(cvt_in, cvt_out, CPY_VEC{});
#endif

    constexpr int MMA_KA_ = decltype(cute::size<2>(tCrB_mma))::value;   // total mma-K atoms in the tile
    constexpr int KBM_    = MMA_KA_ / K_ATOM_PER_COPY;                  // K_BLOCK_MAX (copy steps)
    using FinePolicy = typename MetadataPolicy::template Fine<KBM_, MMA_KA_>;
    constexpr bool FINE = FinePolicy::active;
    constexpr int APG_  = FinePolicy::atoms_per_group;

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // do nothing
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      auto tCrS = cute::get<1>(partitioned_extra_info);
      if constexpr (!FINE) {
        cute::for_each(cute::make_int_sequence<K_ATOM_PER_COPY>{}, [&] (auto i_) {
          constexpr int atom_idx = K_BLOCK_STATIC * K_ATOM_PER_COPY + decltype(i_)::value;
          bdq_transform(tCrB_mma(_, _, Int<atom_idx>{}), tCrS(_, _, 0), tCrB_mma(_, _, Int<atom_idx>{}), cute::multiplies{});
        });
      } else {
        // FINE: write via tCrS_copy_view (a retile VIEW of the ORIGINAL fragment) then read the ORIGINAL back --
        // NOT the local `tCrS` copy above (make_fragment_like is owning, so `auto tCrS` snapshots stale rmem).
    // COMPILE-TIME ATOM INDEX. i used to be a runtime int, so atom_idx, g = atom_idx / APG_ and the
    // `atom_idx % APG_ == 0` guard all became runtime work even though K_ATOM_PER_COPY, APG_ and k_block are
    // constants: the profile showed s.cmp 0.54 + s.csel 0.41 + s.cbr 0.56 + s.shra 0.09 + s.mull 0.16 per mma,
    // about 1.8 of 34. With for_each the index is Int<i>, the guard folds to if constexpr and disappears, and the
    // division and modulo fold away. Same reason the main loop uses for_each -- see its comment about needing
    // k_block to be Int<x>. Numerics are unchanged; this is constant folding only.
        cute::for_each(cute::make_int_sequence<K_ATOM_PER_COPY>{}, [&] (auto i_) {
          constexpr int I = decltype(i_)::value;
          constexpr int atom_idx = K_BLOCK_STATIC * K_ATOM_PER_COPY + I;
          if constexpr (FinePolicy::starts_group(atom_idx))              // reload only at a group's first atom
            MetadataPolicy::reload(partitioned_extra_info, tiled_copy_and_views,
                                   FinePolicy::group(atom_idx), read_stage);
          bdq_transform(tCrB_mma(_, _, Int<atom_idx>{}), cute::get<1>(partitioned_extra_info)(_, _, 0),
                          tCrB_mma(_, _, Int<atom_idx>{}), cute::multiplies{});
        });
      }
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      auto tCrS = cute::get<1>(partitioned_extra_info);
      auto tCrZ = cute::get<3>(partitioned_extra_info);
      if constexpr (!FINE) {
        cute::for_each(cute::make_int_sequence<K_ATOM_PER_COPY>{}, [&] (auto i_) {
          constexpr int atom_idx = K_BLOCK_STATIC * K_ATOM_PER_COPY + decltype(i_)::value;
          bdq_transform(tCrB_mma(_, _, Int<atom_idx>{}), tCrS(_, _, 0), tCrB_mma(_, _, Int<atom_idx>{}), cute::multiplies{});
          bdq_transform(tCrB_mma(_, _, Int<atom_idx>{}), tCrZ(_, _, 0), tCrB_mma(_, _, Int<atom_idx>{}), cute::plus{});
        });
      } else {
        // FINE: see ConvertAndScale note -- write via the copy VIEWs, read the ORIGINAL fragments back.
    // COMPILE-TIME ATOM INDEX. i used to be a runtime int, so atom_idx, g = atom_idx / APG_ and the
    // `atom_idx % APG_ == 0` guard all became runtime work even though K_ATOM_PER_COPY, APG_ and k_block are
    // constants: the profile showed s.cmp 0.54 + s.csel 0.41 + s.cbr 0.56 + s.shra 0.09 + s.mull 0.16 per mma,
    // about 1.8 of 34. With for_each the index is Int<i>, the guard folds to if constexpr and disappears, and the
    // division and modulo fold away. Same reason the main loop uses for_each -- see its comment about needing
    // k_block to be Int<x>. Numerics are unchanged; this is constant folding only.
        // PREFETCH IS AN APPLICABILITY, NOT A REQUIREMENT. Two register sets cover a copy step with at most two
        // scale groups; TileK=64 configs have more and must keep the original per-group reload. Writing that as a
        // static_assert made three units fail to compile instead of falling back -- the same shape of mistake as a
        // boundary check that only guards one end.
        constexpr int GRP = (K_ATOM_PER_COPY % APG_ == 0) ? (K_ATOM_PER_COPY / APG_) : 0;
        // Prefetching a smem read that no longer happens: where the packed path is on, a group costs kPackedSlots
        // decodes from registers and there is no load latency to hide. Per UNIT, not per binary.
        constexpr bool kPfOk = !kPackedScaleOn &&
#if defined(PPU_SCALE_PREFETCH) && (PPU_SCALE_PREFETCH != 0)
            (GRP == 2) && (cute::tuple_size<PfPack>::value == 4);
#else
            false;
#endif
        if constexpr (kPfOk) {
          auto smem_tiled_copy_S = cute::get<0>(tiled_copy_and_views);
          auto tCsS              = cute::get<0>(partitioned_extra_info);
          auto tCsZ              = cute::get<2>(partitioned_extra_info);
          auto tCrS_copy_view    = cute::get<1>(tiled_copy_and_views);
          auto tCrZ_copy_view    = cute::get<2>(tiled_copy_and_views);
          // BOTH groups load before any transform, group gi into register set gi % 2, so the second group's data has
          // a whole group of atoms to arrive in instead of being used one instruction after its load. pf carries the
          // second set: fragments 0,1 and their copy views 2,3.
          constexpr int g0 = FinePolicy::group(K_BLOCK_STATIC * K_ATOM_PER_COPY);
          cute::for_each(cute::make_int_sequence<GRP>{}, [&] (auto gi_) {
            constexpr int GI = decltype(gi_)::value;
            const int sk = read_stage * int(Scale_TileK) + g0 + GI;
            if constexpr (GI % 2 == 0) {
              copy(smem_tiled_copy_S, tCsS(_,_,0,sk), tCrS_copy_view(_,_,0));
              copy(smem_tiled_copy_S, tCsZ(_,_,0,sk), tCrZ_copy_view(_,_,0));
            } else {
              copy(smem_tiled_copy_S, tCsS(_,_,0,sk), cute::get<2>(pf)(_,_,0));
              copy(smem_tiled_copy_S, tCsZ(_,_,0,sk), cute::get<3>(pf)(_,_,0));
            }
          });
          cute::for_each(cute::make_int_sequence<K_ATOM_PER_COPY>{}, [&] (auto i_) {
            constexpr int I        = decltype(i_)::value;
            constexpr int atom_idx = K_BLOCK_STATIC * K_ATOM_PER_COPY + I;
            constexpr int GI       = I / APG_;
            if constexpr (GI % 2 == 0) {
              bdq_transform(tCrB_mma(_,_,Int<atom_idx>{}), cute::get<1>(partitioned_extra_info)(_,_,0),
                              tCrB_mma(_,_,Int<atom_idx>{}), cute::multiplies{});
              bdq_transform(tCrB_mma(_,_,Int<atom_idx>{}), cute::get<3>(partitioned_extra_info)(_,_,0),
                              tCrB_mma(_,_,Int<atom_idx>{}), cute::plus{});
            } else {
              bdq_transform(tCrB_mma(_,_,Int<atom_idx>{}), cute::get<0>(pf)(_,_,0),
                              tCrB_mma(_,_,Int<atom_idx>{}), cute::multiplies{});
              bdq_transform(tCrB_mma(_,_,Int<atom_idx>{}), cute::get<1>(pf)(_,_,0),
                              tCrB_mma(_,_,Int<atom_idx>{}), cute::plus{});
            }
          });
                } else {
          cute::for_each(cute::make_int_sequence<K_ATOM_PER_COPY>{}, [&] (auto i_) {
            constexpr int I = decltype(i_)::value;
            constexpr int atom_idx = K_BLOCK_STATIC * K_ATOM_PER_COPY + I;
            if constexpr (FinePolicy::starts_group(atom_idx)) {          // reload only at a group's first atom
              MetadataPolicy::reload(partitioned_extra_info, tiled_copy_and_views,
                                     FinePolicy::group(atom_idx), read_stage);
            }
            bdq_transform(tCrB_mma(_, _, Int<atom_idx>{}), cute::get<1>(partitioned_extra_info)(_, _, 0),
                            tCrB_mma(_, _, Int<atom_idx>{}), cute::multiplies{});
            bdq_transform(tCrB_mma(_, _, Int<atom_idx>{}), cute::get<3>(partitioned_extra_info)(_, _, 0),
                            tCrB_mma(_, _, Int<atom_idx>{}), cute::plus{});
          });
        }
      }
    }
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled while transforming B");
    }
  }

  /// Utilities for transforming the A operand prior to issuing tensor cell math.
  template <class EngineIn,
            class EngineOut,
            class TensorLayoutIn,
            class TensorLayoutOut,
            int ConversionVectorWidth = cosize_v<TensorLayoutIn>>
  CUTLASS_DEVICE void
  convert_tensor(
    Tensor<EngineIn,TensorLayoutIn> const& in,
    Tensor<EngineOut,TensorLayoutOut>& out,
    cute::Int<ConversionVectorWidth> width = {}) {

    /// The converter consumes one contiguous source cohort and emits one
    /// contiguous destination cohort.  Their static logical sizes must match,
    /// but their rest-mode strides need not: K-pack4 deliberately transports
    /// physical (N,K/4) and writes logical (N,K) MMA storage.
    constexpr int N = size(TensorLayoutIn{});
    constexpr int NOut = size(TensorLayoutOut{});
    // constexpr int N = cosize_v<TensorLayoutIn>;

    /// The inputs must be backed by registers & be statically sized.
    static_assert(is_rmem<EngineIn>::value, "Input tensor for A conversion must come from registers");
    static_assert(is_rmem<EngineOut>::value, "Output tensor for A conversion must come from registers");
    static_assert(is_static_v<TensorLayoutIn>, "Input tensor layout for the conversion must be static");
    static_assert(is_static_v<TensorLayoutOut>, "Output tensor layout for the conversion must be static");
    static_assert(N == NOut, "Input and output conversion tensors must have the same logical size");
    // static_assert(cosize_v<TensorLayoutIn> == size(TensorLayoutIn{}), "Cosize and size of the layout must be equal.");
    static_assert(N % ConversionVectorWidth == 0, "Conversion vector width must divide cosize of the tensor layout.");

    using SrcType = typename EngineIn::value_type;
    using DstType = typename EngineOut::value_type;

    using SrcArray = cutlass::Array<SrcType, ConversionVectorWidth>;
    using DstArray = cutlass::Array<DstType, ConversionVectorWidth>;

    // constexpr cutlass::FloatRoundStyle RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
    // using Converter = cutlass::NumericArrayConverter<DstType, SrcType, ConversionVectorWidth, RoundStyle>;

    // SrcType int8_t consider as uint8_t
    using Converter = cutlass::MixGemmNumericArrayConverter<DstType, SrcType, ConversionVectorWidth>;

    constexpr int NumIterations = N / ConversionVectorWidth;

    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < NumIterations; ++ii) {
      SrcArray const* src_array_ptr = reinterpret_cast<SrcArray const*>(raw_pointer_cast(in(_, ii).data()));
      DstArray* dst_array_ptr = reinterpret_cast<DstArray*>(raw_pointer_cast(out(_, ii).data()));
      *dst_array_ptr = Converter::convert(*src_array_ptr);
    }
  }


public:
  // A PUBLIC FORWARDER FOR THE PUBLICATION STEP, so it can be compiled in isolation and its emitted stores counted.
  // packed_decode_stage is private and is only ever reached from mma(), which drags in the whole pipeline -- so the
  // question "does the fused branch actually emit ONE 32-bit shared store" had no local answer, and the flag shipped
  // to the box in a state where the counter it must move did not move. Byte-neutral changes cannot be seen in any
  // run (same shared bytes, same results, same instruction mix on every other path), so the only observables left
  // are the type, which l100_fused_active.cu asserts, and the generated code, which this makes reachable.
  template <bool On, class Storage>
  CUTLASS_DEVICE static void probe_packed_decode_stage(Storage& storage, int stage, int thread_idx,
                                                       int64_t residue_n) {
    packed_decode_stage<On>(storage, stage, 0, thread_idx, residue_n);
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
