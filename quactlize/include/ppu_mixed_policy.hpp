#pragma once

#include <type_traits>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "fold_traits.hpp"
#include "ppu_group_schedule.hpp"
#include "ppu_tactic_space.hpp"
#include "quactlize_actlize.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"

// The operator adapters differ in problem scheduling and epilogues, not in how one mixed-input mainloop is built.
// Keep every mainloop-affecting launcher choice here so adding a quant tuple, fold wrapper, layout rule, or builder
// argument to one operator cannot silently omit it from the other.
namespace ppu_mixed_policy {

enum class QuantMode { PerColScaleOnly, FinegrainedScaleOnly, FinegrainedScaleZero };
constexpr bool is_finegrained(QuantMode mode) { return mode != QuantMode::PerColScaleOnly; }
constexpr bool has_zero(QuantMode mode) { return mode == QuantMode::FinegrainedScaleZero; }

template <class Element, bool = std::is_void_v<Element>>
struct ElementBits;

template <class Element>
struct ElementBits<Element, false> : std::integral_constant<int, cutlass::sizeof_bits<Element>::value> {};

template <class Element>
struct ElementBits<Element, true> : std::integral_constant<int, 0> {};

template <class Element>
inline constexpr int element_bits_v = ElementBits<Element>::value;

template <class Element, int ArtifactTileK, bool = std::is_void_v<Element>>
struct ElementFold;

template <class Element, int ArtifactTileK>
struct ElementFold<Element, ArtifactTileK, false>
    : std::integral_constant<int, fold::delivery_fold_v<element_bits_v<Element>, ArtifactTileK>> {};

template <class Element, int ArtifactTileK>
struct ElementFold<Element, ArtifactTileK, true> : std::integral_constant<int, 1> {};

template <QuantMode Mode, class ElementB, class PlaneB2 = void,
          class ElementScale = cutlass::half_t, class ElementZero = cutlass::half_t>
struct OperandInfo {
  using Type = std::conditional_t<!std::is_void_v<PlaneB2>,
      std::conditional_t<has_zero(Mode),
          cute::tuple<ElementB, ElementScale, ElementZero, PlaneB2>,
          cute::tuple<ElementB, ElementScale, cutlass::gemm::collective::detail::NoZero, PlaneB2>>,
      std::conditional_t<has_zero(Mode), cute::tuple<ElementB, ElementScale, ElementZero>,
                                         cute::tuple<ElementB, ElementScale>>>;
};

template <int ArtifactLowFold, int ArtifactHighFold, class BaseSchedule>
struct ArtifactFoldedSchedule {
  // The wrapper is also required when only the high plane folds: it is the type-level ABI carrying the resident
  // provider's two independent physical layouts into CollectiveBuilder.
  //
  // IT IS NOW UNCONDITIONAL, and the reason is dispatch rather than folding. quactlize's CollectiveBuilder arm is
  // the COMPLEMENT of actlize's -- it claims KernelAiuFold<...> and Gs32, and leaves PerCol/Gs128/Gs64 to the
  // vendor, because claiming those as well made the two specialisations ambiguous. So an unfolded gs=128 row that
  // passed a bare KernelAiuMultistageMixedInputFinegrainedGs128 would silently select ACTLIZE's collective: it
  // compiles, it runs, and it is not the kernel this project measured. Wrapping at F=1 keeps it on ours.
  //
  // ROUTING-NEUTRAL, not merely harmless: the builder reads ArtifactLowFold back through fold_schedule_traits and
  // floors it at 1, so KernelAiuFold<1, Base, 0> yields HasFold = (1 > 1) = false and BaseSchedule = Base --
  // bit-for-bit the derivation the bare tag produced. Only the specialisation that MATCHES changes.
  using Type = cutlass::gemm::KernelAiuFold<(ArtifactLowFold > 1 ? ArtifactLowFold : 1),
                                            BaseSchedule, ArtifactHighFold>;
};

struct AiuAProvider {};
struct PackedRowAProvider {};
struct OrdinaryBProvider {};
template <int Fold> struct FoldedBProvider {};
template <int LowFold, int HighFold> struct TwoPlaneBProvider {};
template <bool Packed, bool HasZero> struct MetadataProvider {};
template <bool AtomAtATime> struct ConversionProvider {};

template <class Collective, class = void>
struct PackedMetadata : std::false_type {};

template <class Collective>
struct PackedMetadata<Collective, std::void_t<decltype(Collective::is_packed_scale)>>
    : std::bool_constant<Collective::is_packed_scale> {};

template <class Collective, class = void>
struct AtomAtATimeConversion : std::false_type {};

template <class Collective>
struct AtomAtATimeConversion<Collective, std::void_t<decltype(Collective::kBChunk)>>
    : std::bool_constant<Collective::kBChunk> {};

template <class Collective, class BaseSchedule, class KernelSchedule, class ElementBInfo,
          class LayoutA, class LayoutB, class TileShape, class ScaleTileShape, class WarpShape,
          class AProvider, class BProvider, QuantMode Mode, int LowBits, int HighBits,
          int TacticTileK, int ArtifactTileK, int ArtifactLowFold, int ArtifactHighFold,
          int Stages, bool Interleaved>
struct MixedPolicyDescriptor {
  using CollectiveMainloop = Collective;
  using BaseScheduleType = BaseSchedule;
  using KernelScheduleType = KernelSchedule;
  using ElementBInfoType = ElementBInfo;
  using LayoutAType = LayoutA;
  using LayoutBType = LayoutB;
  using TileShapeType = TileShape;
  using ScaleTileShapeType = ScaleTileShape;
  using WarpShapeType = WarpShape;
  using AProviderType = AProvider;
  using BProviderType = BProvider;
  using MetadataPolicyType = typename Collective::MetadataPolicy;
  using PipelineDriverType = typename Collective::PipelineDriver;
  using MetadataProviderType = MetadataProvider<PackedMetadata<Collective>::value, has_zero(Mode)>;
  using ConversionProviderType = ConversionProvider<AtomAtATimeConversion<Collective>::value>;

  static constexpr QuantMode quant_mode = Mode;
  static constexpr int low_bits = LowBits;
  static constexpr int high_bits = HighBits;
  static constexpr int tactic_tile_k = TacticTileK;
  static constexpr int artifact_tile_k = ArtifactTileK;
  static constexpr int artifact_low_fold = ArtifactLowFold;
  static constexpr int artifact_high_fold = ArtifactHighFold;
  static constexpr int low_fold = ArtifactLowFold;    // compatibility with existing descriptor consumers
  static constexpr int high_fold = ArtifactHighFold;
  static constexpr int stages = Stages;
  // ScaleTileShape is only the consumer window (ceil(TacticTileK/group_size)). Scale/zero remain logical
  // (N,K/group_size) planes, so unlike B they have no TileK-dependent physical fold and need no artifact TileK field.
  static constexpr int scale_tile_k = int(cute::size<1>(ScaleTileShape{}));
  static constexpr bool interleaved = Interleaved;
  static constexpr bool packed_metadata = PackedMetadata<Collective>::value;
  static constexpr bool atom_at_a_time = AtomAtATimeConversion<Collective>::value;
};

template <QuantMode Mode, class BaseSchedule, class TileShape, class ScaleTileShape, class WarpShape,
          int Stages, bool AiuInterleaved, class ElementB = cutlass::int4b_t, class PlaneB2 = void,
          int ArtifactTileK_ = 0>
struct MainloopPolicy {
  using ElementA = cutlass::half_t;
  using ElementScale = cutlass::half_t;
  using ElementZero = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = std::conditional_t<AiuInterleaved,
      cutlass::layout::ColumnMajorInterleaved<256>, cutlass::layout::ColumnMajor>;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int LowBits = element_bits_v<ElementB>;
  static constexpr int HighBits = element_bits_v<PlaneB2>;
  static constexpr int TacticTileK = int(cute::size<2>(TileShape{}));
  // Zero is the source-compatible spelling for callers predating the split; it means this tactic also defines the
  // artifact. New multi-TileK callers pass the artifact value explicitly.
  static constexpr int ArtifactTileK = ArtifactTileK_ > 0 ? ArtifactTileK_ : TacticTileK;
  static_assert(ArtifactTileK > 0 && ArtifactTileK <= TacticTileK && TacticTileK % ArtifactTileK == 0,
                "ArtifactTileK must completely tile TacticTileK");
  static constexpr int ArtifactLowFold = fold::delivery_fold_v<LowBits, ArtifactTileK>;
  static constexpr int ArtifactHighFold = ElementFold<PlaneB2, ArtifactTileK>::value;
  static_assert(int(cute::size<1>(TileShape{})) % ArtifactLowFold == 0,
                "ArtifactLowFold must divide the tactic's TileN");
  static_assert(HighBits == 0 || int(cute::size<1>(TileShape{})) % ArtifactHighFold == 0,
                "ArtifactHighFold must divide the tactic's TileN");
  static_assert((ArtifactLowFold * TacticTileK * LowBits) % 256 == 0,
                "ArtifactLowFold must completely cover the tactic in whole 32-byte AIU runs");
  static_assert(HighBits == 0 || (ArtifactHighFold * TacticTileK * HighBits) % 256 == 0,
                "ArtifactHighFold must completely cover the tactic in whole 32-byte AIU runs");
  // Compatibility spellings. These values now unambiguously describe the artifact, never the tactic.
  static constexpr int TileK = TacticTileK;
  static constexpr int LowFold = ArtifactLowFold;
  static constexpr int HighFold = ArtifactHighFold;

  using KernelSchedule = typename ArtifactFoldedSchedule<ArtifactLowFold, ArtifactHighFold,
                                                          BaseSchedule>::Type;
  using ElementBInfo = typename OperandInfo<Mode, ElementB, PlaneB2, ElementScale, ElementZero>::Type;
  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementBInfo, LayoutB, AlignmentB, float,
      cute::tuple<TileShape, ScaleTileShape>, WarpShape, cute::Int<Stages>, KernelSchedule>::CollectiveOp;

#if defined(PPU_A_PACK) && (PPU_A_PACK != 0)
  static constexpr bool PackedRowA = ArtifactLowFold == 1 && HighBits == 0;
#else
  static constexpr bool PackedRowA = false;
#endif
  using AProvider = std::conditional_t<PackedRowA, PackedRowAProvider, AiuAProvider>;
  using BProvider = std::conditional_t<(HighBits > 0), TwoPlaneBProvider<ArtifactLowFold, ArtifactHighFold>,
      std::conditional_t<(ArtifactLowFold > 1), FoldedBProvider<ArtifactLowFold>, OrdinaryBProvider>>;
  using Descriptor = MixedPolicyDescriptor<CollectiveOp, BaseSchedule, KernelSchedule, ElementBInfo,
      LayoutA, LayoutB, TileShape, ScaleTileShape, WarpShape, AProvider, BProvider,
      Mode, LowBits, HighBits, TacticTileK, ArtifactTileK, ArtifactLowFold, ArtifactHighFold,
      Stages, AiuInterleaved>;
};

// One guard for every adapter. TacticSpace is the only operator-specific input: it preserves the declared dense-only
// quarantine while the instantiated mainloop, delivery checks, and scale-copy coverage remain one shared contract.
template <class TacticSpace, class Policy>
struct KernelPolicyGuard {
  using Mainloop = typename Policy::CollectiveOp;
  using TileShape = typename Policy::Descriptor::TileShapeType;
  using WarpShape = typename Policy::Descriptor::WarpShapeType;
  static constexpr ppu_tactics::Candidate tactic{{ppu_tactics::Format::I2, "mixed-policy",
      Policy::LowBits, Policy::HighBits}, int(cute::size<0>(TileShape{})), int(cute::size<1>(TileShape{})),
      Policy::TacticTileK, int(cute::size<0>(WarpShape{})), int(cute::size<1>(WarpShape{})),
      Policy::ArtifactTileK};
  using TiledMma = typename Mainloop::TiledMma;

  static_assert(int(cute::size(TiledMma{})) == 32 * ppu_tactics::cta_warps(tactic),
                "mixed policy: tactic warp count must equal the instantiated TiledMma launch size");
  static_assert(Mainloop::scale_copy_thread_coverage,
                "mixed policy: scale copy must cover every slot with the instantiated CTA threads");
  static_assert(TacticSpace::kernel_exclusion(tactic) == ppu_tactics::Exclusion::None,
                "mixed policy: tactic violates this operator's emitted kernel search-space rules");
  static_assert(fold::CheckDelivery<Policy::LowBits, cute::size<1>(TileShape{}), Policy::TacticTileK,
                                    cute::size<0>(WarpShape{}), cute::size<1>(WarpShape{})>::ok,
                "mixed policy low plane: swzl over-delivers at this warp shape");
  static_assert(fold::CheckDelivery<Policy::HighBits, cute::size<1>(TileShape{}), Policy::TacticTileK,
                                    cute::size<0>(WarpShape{}), cute::size<1>(WarpShape{})>::ok,
                "mixed policy high plane: swzl over-delivers at this warp shape");
  static constexpr bool value = true;
};

template <class TacticSpace, class Policy>
inline constexpr bool kernel_policy_valid_v = KernelPolicyGuard<TacticSpace, Policy>::value;

}  // namespace ppu_mixed_policy
