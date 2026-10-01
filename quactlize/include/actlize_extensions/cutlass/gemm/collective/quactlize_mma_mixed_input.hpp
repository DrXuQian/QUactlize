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
#include "actlize_extensions/cutlass/gguf_bfloat_scale.h"
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
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_decode_input.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_pipeline.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_a_pack.hpp"
#include "q4_kpack4_offline.hpp"
#include "kquant_kpack_offline.hpp"

#include "cutlass/detail/collective.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective::detail {
template <class BDelivery, class CopyPolicy, class SrcTensor, class DstTensor>
CUTLASS_DEVICE void issue_mixed_b_g2s(
    CopyPolicy const& copy_policy,
    SrcTensor const& src,
    DstTensor&& dst,
    int& warp_idx) {
  if constexpr (BDelivery::single_issuer) {
    copy_aiu(copy_policy, src, static_cast<DstTensor&&>(dst), warp_idx);
  } else {
    static_assert(BDelivery::thread_partitioned,
                  "B writer must declare single-issuer or thread-partitioned ownership");
    copy(copy_policy, src, static_cast<DstTensor&&>(dst));
  }
}
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
  using MetadataPublication = typename DispatchPolicy::MetadataPublication;
  static_assert(std::is_same_v<MetadataPublication, SeparateHalfPlanes> ||
                    std::is_same_v<MetadataPublication, InterleavedHalf2>,
                "mixed-input metadata publication requires a known schedule tag");
  static constexpr bool kKPackTranspose =
      kpack_schedule_traits<KernelSchedule>::Value;
  static_assert(kKPackTranspose, "this collective accepts canonical K-pack only");
  static constexpr int kKPackLow =
      kpack_schedule_traits<KernelSchedule>::LowPack;
  static constexpr int kKPackHigh =
      kpack_schedule_traits<KernelSchedule>::HighPack;
  static constexpr int kKPackScheduledDeliveryN =
      kpack_schedule_traits<KernelSchedule>::DeliveryN;
  using ScheduledBDeliveryPolicy =
      typename kpack_schedule_traits<KernelSchedule>::BDelivery;
  using BDeliveryPolicy = cute::conditional_t<
      cute::is_void_v<ScheduledBDeliveryPolicy>,
      detail::quactlize_b_delivery::ProductionBDelivery,
      ScheduledBDeliveryPolicy>;
  static_assert(!kKPackTranspose || kKPackHigh == 0,
                "the single-plane collective cannot consume a second K-pack plane");
  static_assert(
      !kKPackTranspose ||
          std::is_same_v<
              BDeliveryPolicy,
              detail::quactlize_b_delivery::ProductionBDelivery>,
      "the shipping K-pack schedule must retain its matched AIU/TSM B delivery");
  // Historical names remain public for the Q4 type/codegen gates.
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
  static constexpr bool kPackedTileDividesSb =
      int(Scale_TileK) > 0 && PackedUnit::kGroups % int(Scale_TileK) == 0;
  static constexpr bool kPackedKpack4Subtile =
      kKPackTranspose && kPackedFormatMatchesElement &&
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
  static constexpr int kKPackResolvedDeliveryN =
      kKPackTranspose ? int(size<0>(InternalSmemLayoutAtomB{})) : 0;
  static_assert(!kKPackTranspose ||
                    (kKPackResolvedDeliveryN == 16 ||
                     kKPackResolvedDeliveryN == 32 ||
                     kKPackResolvedDeliveryN == 64),
                "K-pack collective must retain a named resident delivery N");
  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  // static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  // static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  // using ConvertedElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  // using ConvertedElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;
  // using InternalElementA = cute::conditional_t<!SwapAB, ConvertedElementA, ConvertedElementB>;
  // using InternalElementB = cute::conditional_t<!SwapAB, ConvertedElementB, ConvertedElementA>;

  using RealInternalElementA = cute::conditional_t<!SwapAB, ElementA, ElementB>;
  using DecodeInputTraits = detail::DecodeInputTraits<TransformA_, RealInternalElementA>;
  using GlobalElementA = typename DecodeInputTraits::SourceElement;
  static constexpr bool kDecodeInput = DecodeInputTraits::enabled;
  using ArgumentElementA = cute::conditional_t<kDecodeInput,GlobalElementA,ElementA>;
  static_assert(!kDecodeInput || !SwapAB,"decode input conversion requires ordinary A");
  static constexpr int LogicalTileM = int(size<0>(TileShape{}));
  static constexpr int PhysicalATileM = LogicalTileM < 16 ? 16 : LogicalTileM;
  static constexpr int kACubeH      = PhysicalATileM;                         // physical PPU0010 A cube height
  static constexpr int kACubeW      = 64;                                    // AiuContElemSize for fp16
  static constexpr int kASlices     = kACubeW / 16;                          // 8 words per slice
  static constexpr int kScheduledAPackRows =
      a_provider_schedule_traits<KernelSchedule>::Rows;
  static constexpr int kAPackRows = kScheduledAPackRows;
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
      kKPackTranspose, cutlass::half_t, RealInternalElementB>;
  static constexpr int PhysicalBTileK = kKPackTranspose
      ? int(size<2>(TileShape{})) / kKPackLow
      : int(size<2>(TileShape{}));
  static_assert(!kKPackTranspose ||
                    (kKPackLow * sizeof_bits<RealInternalElementB>::value == 16 &&
                     int(size<2>(TileShape{})) %
                         (kquant_kpack::kReaderPhysicalK * kKPackLow) == 0),
                "K-pack collective needs one full b16 word and a whole physical K16 read");
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

  // Scale and zero-point tiles share the same compact (N, group, stage)
  // layout. Both the publication view and the flattened MMA read view below
  // are derived from this one layout authority.
  using SmemLayoutScale = decltype(tile_to_shape(
    SmemLayoutAtomScale{},
    make_shape(shape<0>(ScaleTileShape{}), shape<1>(ScaleTileShape{}), Int<DispatchPolicy::Stages>{})));

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

  // ONE INTERLEAVED (scale, zero) TILE INSTEAD OF TWO PLANES.
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
  //     address; a bank conflict is a property of the ADDRESSES the warp issues, which are unchanged.
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
      std::is_same_v<MetadataPublication, InterleavedHalf2> &&
      kPackedScaleOn &&
      (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero);

  // The fused views below require the compact layout selected above. Keep that
  // relationship explicit so the publication and flattened read views cannot
  // drift apart.
  static constexpr int kSZ_N   = int(shape<0>(ScaleTileShape{}));
  static constexpr int kSZ_G   = int(shape<1>(ScaleTileShape{}));
  static constexpr int kSZ_St  = int(DispatchPolicy::Stages);
  // Use a template gate so the inactive fused-layout checks are not
  // instantiated for conversion modes that do not carry scale and zero.
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
                "interleaved metadata publication assumes compact (n, group, stage) scale layout");
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
  // Public witnesses let type/codegen gates prove that the requested schedule
  // reached this exact collective. Shared bytes and numerical semantics stay
  // unchanged; only the packed ScaleZero publication address map/store width
  // differs.
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
    cute::ArrayEngine<NonVoidElementScale, scale_elements> smem_scale;
    cute::ArrayEngine<NonVoidElementZero, zero_elements> smem_zero;
    // LAST on purpose: at zero elements it cannot move anything above it, so a default build stays byte-identical.
    cute::ArrayEngine<uint8_t, kPackedScaleOn ? int(cute::cosize(SmemLayoutScaleRawStaged{})) : 0> smem_scale_raw;
  };
  // Host side kernel arguments
  struct Arguments {
    ArgumentElementA const* ptr_A = nullptr;
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

    GlobalElementA const* ptr_A = nullptr;
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
      p.ptr_A = reinterpret_cast<GlobalElementA const*>(args.ptr_A);
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
    auto a_expert_base = detail::mixed_a_expert_base(
        mainloop_params.ptr_A, mainloop_params.dA,
        mainloop_params.group_row_offsets, l_coord);
    Tensor mA_mkl = make_tensor(make_gmem_ptr(a_expert_base),
                                make_shape(M,K,cute::Int<1>{}), mainloop_params.dA);                            // (m,k,1)
    auto gA_logical = [&] {
      if constexpr (kPackedA || kDecodeInput) {
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
      {
        // Physical B is [N,K/Pack] b16.  Its K-tile count is nevertheless
        // identical to the logical [N,K] tensor because both the problem and
        // tactic K extents are divided by the same plane-owned Pack here.
        using PhysicalBTile = Shape<decltype(shape<1>(TileShape{})),
                                    Int<PhysicalBTileK>>;
        return local_tile(mB_nk, PhysicalBTile{},
                          make_coord(n_coord, _));
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
        (ScaleCopyPlan::thread_slots == Scale_NumThreads) ||
        ScaleCopyPlan::owns_physical_thread(thread_idx);
    bool const packed_copy_owner =
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
    {
      CUTE_STATIC_ASSERT_V(size<1>(sA) == Int<kKPackLow>{} * size<1>(sB));
    }
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA_physical)); // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));        // PIPE

    // Partition the copying of A and B tiles across the threads
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)
    auto copy_A_and_B = [&] (auto k_tile, auto k_iter_crd, int pipe) {
      if constexpr (kDecodeInput) {
        detail::issue_mixed_b_g2s<BDeliveryPolicy>(gmem_tiled_copy_B,
            tBgB(_,_,_,k_iter_crd),tBsB(_,_,_,pipe),warp_idx);
        detail::copy_decode_a<PhysicalATileM,int(size<2>(TileShape{})),int(size(TiledMma{})),
            kPackedA,kAPackRows,kAPackPitch,kAPackStagePitch>(gA,storage.smem_a.begin(),
                int(k_tile),pipe,thread_idx,gmem_tiled_copy_A.desc_.dim_h);
      } else if constexpr (kPackedA) {
        detail::issue_mixed_b_g2s<BDeliveryPolicy>(
            gmem_tiled_copy_B, tBgB(_,_,_,k_iter_crd),
            tBsB(_,_,_,pipe), warp_idx);
        copy_A_packed_rows<kAPackRows>(
            gA, storage.smem_a.begin(), k_tile, pipe, thread_idx, gmem_tiled_copy_A.desc_.dim_h);
      } else {
        auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
        Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                         // (ACPY,ACPY_M,ACPY_K,k)
        Tensor tAsA = gmem_thr_copy_A.partition_D(sA_physical);                // (ACPY,ACPY_M_PHYS,ACPY_K,PIPE)
        if constexpr (BDeliveryPolicy::single_issuer) {
          // Preserve the shipping paired AIU call exactly; splitting this
          // branch merely for abstraction would make codegen equality an
          // assumption instead of a measured invariant.
          copy_aiu(
            gmem_tiled_copy_A, tAgA(_,_,_,k_tile), tAsA(_,_,_,pipe),
            gmem_tiled_copy_B, tBgB(_,_,_,k_iter_crd), tBsB(_,_,_,pipe),
            warp_idx
          );
        } else {
          copy_aiu(gmem_tiled_copy_A, tAgA(_,_,_,k_tile),
                   tAsA(_,_,_,pipe), warp_idx);
          detail::issue_mixed_b_g2s<BDeliveryPolicy>(
              gmem_tiled_copy_B, tBgB(_,_,_,k_iter_crd),
              tBsB(_,_,_,pipe), warp_idx);
        }
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
      {
        // Owning rmem fragment only: no load is performed through this logical
        // view.  The physical b16 tile is read below, converted, and retiled
        // into this ordinary N x logical-K MMA destination.
        auto logical_b = make_tensor(
            make_smem_ptr(reinterpret_cast<cutlass::half_t*>(storage.smem_b.begin())),
            make_layout(
                make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
                make_stride(shape<2>(TileShape{}), _1{})));
        return thr_mma.partition_fragment_B(logical_b);
      }
    }();                                                                       // (MMA,MMA_N,MMA_K logical)
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

    // K-pack uses the actual PPU0010 fp16 B-fragment map because each opaque
    // b16 is one transport word.  One physical K16 copy step feeds Pack
    // logical K16 MMA atoms.
    using TiledMma_KPack4 = TiledMMA<
        MMA_Atom<PPU0010_16x16x16_F32F16F16F32_TN>,
        Layout<Shape<warpOnM, warpOnN, _1>>,
        Tile<ShadowPermutationM, PermutationN, _16>>;
    using TiledMma_BLoad = cute::conditional_t<
        kKPackTranspose, TiledMma_KPack4, TiledMma_S8>;

    TiledMma_BLoad tiled_mma_bload;
    auto thr_mma_bload = tiled_mma_bload.get_thread_slice(thread_idx);

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(aiu_warp_group_thread_idx);
    Tensor tCsA = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));        // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                                       // (CPY,CPY_M,CPY_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto sB_load = [&] {
      return sB;
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
      detail::prepare_mixed_a_for_b<ARegisterSchedule>(
          smem_tiled_copy_A, tCsA_p, tCrA_copy_view, k_block, prime);
      transform_B_kblock<RealInternalElementB>(tCrB_copy_view, tCrB_mma, partitioned_extra_info, k_block,
          K_ATOM_PER_COPY, copy_partitions_extra_info, read_stage);
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
    {
      using TilerB = typename GmemTiledCopyB::Tiler_MN;
      using Transport = cutlass::half_t;
      int const physical_k = K / kKPackLow;
      static_assert(sizeof_bits<Transport>::value ==
                        kKPackLow * sizeof_bits<RealInternalElementB>::value,
                    "one K-pack transport word must cover one complete logical code pack");
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
                    "the selected K-pack expert view must not retain an L mode");
      auto const* expert_base = reinterpret_cast<uint8_t const*>(
          raw_pointer_cast(mB_nk.data()));
      gmem_tiled_copy_B.desc_.template init<
          Transport, true, get<0>(TilerB{}), get<1>(TilerB{})>(
              const_cast<uint8_t*>(expert_base), N, physical_k,
              take<0,2>(physical_stride));
      return make_mix_tensor_like(mB_nk);
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

      // Ordinary fp16 metadata still needs predicated-copy destination initialization. Packed metadata does not:
      // its decode owner is the sole writer of the complete destination column, including explicit zeroes for the N
      // tail below.
      if constexpr (!kPackedScaleOn) {
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

  // The packed f16x2 per-group decode requires both fields to use six-bit
  // unsigned codes. Other formats keep the scalar group decoder. The pair
  // path folds the field width into kMagic1152x2, so Q2_K (four-bit fields)
  // and Q6_K (eight-bit fields) must not enter it.
  static constexpr bool kPackedPairFast = kPackedHasMin && (kPackedScaleBias == 0)
                                      && (PackedUnit::kScaleBits == 6) && (PackedUnit::kMinBits == 6);


  // ON IS A TEMPLATE PARAMETER, and that is the whole point: `if constexpr` only skips INSTANTIATING a discarded branch
  // when its condition is value-dependent. With kPackedScaleOn (a class constant) both branches were instantiated, so
  // every unit paid for the packed tensors even with the path off -- measured as ~5% on the TK=64 control rows, which are
  // supposed to be byte-identical between the two builds. Routing the gate through a template parameter makes the
  // discarding real.

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
        auto const bh = cutlass::gguf_packed::bfloat_head_of_words(u);
        (void)m2; (void)h; (void)bh;
        // ONE body, called from both halves with a compile-time G. The group index has to be a template argument --
        // every bit position in the unit is derived from it -- so the two halves cannot share a runtime offset.
        auto decode_group = [&] (auto source_g_, auto destination_g_) {
          constexpr int SourceG = decltype(source_g_)::value;
          constexpr int G = decltype(destination_g_)::value;
          auto const sz = [&] {
          if constexpr (cute::is_same_v<NonVoidElementScale, cutlass::bfloat16_t>) {
            return cutlass::gguf_packed::bfloat_group_of_words<SourceG, kPackedZMul, kPackedFmt>(u, bh);
          } else if constexpr (kPackedPairFast) {
            return cutlass::gguf_packed::group_pair_of_words<SourceG, kPackedZMul, kPackedScaleBias>(u, m2);
          } else {
            return cutlass::gguf_packed::group_of_words<
                SourceG, kPackedScaleBias, kPackedHasMin, kPackedZMul, kPackedFmt>(u, h);
          }
          }();
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
      // Flatten group and stage for the MMA reader while retaining the same
      // compact N-major address map as SmemLayoutScale.
      using UnfusedSmemCopyLayoutScale = decltype(tile_to_shape(SmemLayoutAtomScale{},
          make_shape(shape<0>(ScaleTileShape{}), Int<1>{}, Int<smem_scale_k>{})));
      // THE READ SIDE'S FLATTENED VIEW, FUSED. Same shape (n, 1, stage*Scale_TileK + g), strides doubled, so a reader
      // written against the unfused layout keeps its code and only walks 4 bytes per element instead of 2. Checked
      // against the unfused layout rather than assumed, for the reason given at SmemLayoutScaleFusedWord: this is the
      // SECOND view of the same buffer, and the two going out of step is the documented failure here.
      static_assert(sz_copy_layout_is_compact<kFusedScaleZero, UnfusedSmemCopyLayoutScale>(),
                    "interleaved metadata publication assumes the compact flattened scale copy layout");
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
      // twice over: wrong buffer and a layout for a different allocation.
      // Repointed rather than deleted so the tuple indices, which every
      // consumer reads positionally, do not move.
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

  template <class TA, class TB, class TC, class Op>
  CUTLASS_DEVICE static void transform_fragment(TA&& a, TB&& b, TC&& c, Op op) {
    detail::transform_metadata(a, b, c, op);
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
            class KBlockT>
  CUTLASS_DEVICE
  void transform_B_kblock(
    TCrB_load const& tCrB_load,
    TCrB_mma& tCrB_mma,
    cute::tuple<Ts...>& partitioned_extra_info,
    // KBlockT, NOT int: the callers already hold a static k_block (for_each gives Int<x>, and k_block_next is
    // (Int<x> + _1) % K_BLOCK_MAX, also static). Taking it as an int erased that, so atom_idx and its metadata
    // group all turned into runtime work -- s.cmp 0.54 + s.csel 0.41 + s.cbr 0.56 +
    // s.shra 0.09 + s.mull 0.16 per mma in the profile, ~1.8 of 34. Kept static, the guard folds to if constexpr
    // and vanishes with the division and modulo. Constant folding only; numerics unchanged.
    KBlockT const& k_block,
    cute::Int<K_ATOM_PER_COPY> k_atom,
    CopyViews const& tiled_copy_and_views,
    int const read_stage) {

    static constexpr int K_BLOCK_STATIC = int(KBlockT{});
    Tensor cvt_in  = recast<RealInternalElementB>(tCrB_load(_, _, k_block));
    Tensor cvt_out = [&] {
      {
        auto dst_n_stride = compact_col_major(
            shape<1>(cvt_in.layout()), stride<1>(tCrB_mma.layout()));
        auto dst_layout = make_layout(
            shape(cvt_in.layout()),
            make_stride(stride<0>(cvt_in.layout()), dst_n_stride));
        return make_tensor(
            tCrB_mma(_, _, k_block * K_ATOM_PER_COPY).data(), dst_layout);
      }
    }();

    using CPY_VEC = Int<4 * 32 / sizeof_bits<RealInternalElementB>::value>;
    convert_tensor(cvt_in, cvt_out, CPY_VEC{});

    constexpr int MMA_KA_ = decltype(cute::size<2>(tCrB_mma))::value;   // total mma-K atoms in the tile
    constexpr int KBM_    = MMA_KA_ / K_ATOM_PER_COPY;                  // K_BLOCK_MAX (copy steps)
    using FinePolicy = typename MetadataPolicy::template Fine<KBM_, MMA_KA_>;
    constexpr bool FINE = FinePolicy::active;

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // do nothing
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      auto tCrS = cute::get<1>(partitioned_extra_info);
      if constexpr (!FINE) {
        cute::for_each(cute::make_int_sequence<K_ATOM_PER_COPY>{}, [&] (auto i_) {
          constexpr int atom_idx = K_BLOCK_STATIC * K_ATOM_PER_COPY + decltype(i_)::value;
          transform_fragment(tCrB_mma(_, _, Int<atom_idx>{}), tCrS(_, _, 0), tCrB_mma(_, _, Int<atom_idx>{}), cute::multiplies{});
        });
      } else {
        // FINE: write via tCrS_copy_view (a retile VIEW of the ORIGINAL fragment) then read the ORIGINAL back --
        // NOT the local `tCrS` copy above (make_fragment_like is owning, so `auto tCrS` snapshots stale rmem).
    // COMPILE-TIME ATOM INDEX. i used to be a runtime int, so the atom/group lookup and group-boundary guard
    // all became runtime work even though K_ATOM_PER_COPY, the metadata group size and k_block are
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
          transform_fragment(tCrB_mma(_, _, Int<atom_idx>{}), cute::get<1>(partitioned_extra_info)(_, _, 0),
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
          transform_fragment(tCrB_mma(_, _, Int<atom_idx>{}), tCrS(_, _, 0), tCrB_mma(_, _, Int<atom_idx>{}), cute::multiplies{});
          transform_fragment(tCrB_mma(_, _, Int<atom_idx>{}), tCrZ(_, _, 0), tCrB_mma(_, _, Int<atom_idx>{}), cute::plus{});
        });
      } else {
        // FINE: see ConvertAndScale note -- write via the copy VIEWs, read the ORIGINAL fragments back.
        // Keep the atom index compile-time so group boundaries, division and
        // modulo all fold before code generation.
        cute::for_each(cute::make_int_sequence<K_ATOM_PER_COPY>{}, [&] (auto i_) {
          constexpr int I = decltype(i_)::value;
          constexpr int atom_idx = K_BLOCK_STATIC * K_ATOM_PER_COPY + I;
          if constexpr (FinePolicy::starts_group(atom_idx)) {
            MetadataPolicy::reload(partitioned_extra_info, tiled_copy_and_views,
                                   FinePolicy::group(atom_idx), read_stage);
          }
          transform_fragment(tCrB_mma(_, _, Int<atom_idx>{}), cute::get<1>(partitioned_extra_info)(_, _, 0),
                          tCrB_mma(_, _, Int<atom_idx>{}), cute::multiplies{});
          transform_fragment(tCrB_mma(_, _, Int<atom_idx>{}), cute::get<3>(partitioned_extra_info)(_, _, 0),
                          tCrB_mma(_, _, Int<atom_idx>{}), cute::plus{});
        });
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


};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
