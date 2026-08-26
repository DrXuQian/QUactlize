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

template <int ArtifactLowFold, int ArtifactHighFold, class BaseSchedule, int ArtifactTileK>
struct ArtifactFoldedSchedule {
  // The wrapper is also required when only the high plane folds: it is the type-level ABI carrying the resident
  // provider's two independent physical layouts AND delivery TileK into CollectiveBuilder. The folds alone lose A
  // at F=1, where A64/A128/A256 all have the same fold but different resident copy quanta.
  //
  // IT IS NOW UNCONDITIONAL, and the reason is dispatch rather than folding. quactlize's CollectiveBuilder arm is
  // the COMPLEMENT of actlize's -- it claims KernelAiuFold<...> and Gs32, and leaves PerCol/Gs128/Gs64 to the
  // vendor, because claiming those as well made the two specialisations ambiguous. So an unfolded gs=128 row that
  // passed a bare KernelAiuMultistageMixedInputFinegrainedGs128 would silently select ACTLIZE's collective: it
  // compiles, it runs, and it is not the kernel this project measured. Wrapping at F=1 keeps it on ours.
  //
  // ROUTING-NEUTRAL, not merely harmless: the builder reads ArtifactLowFold back through fold_schedule_traits and
  // floors it at 1, so KernelAiuFold<1, Base, 0, A> yields HasFold = (1 > 1) = false and BaseSchedule = Base --
  // bit-for-bit the derivation the bare tag produced. Only the specialisation that MATCHES changes.
  using Type = cutlass::gemm::KernelAiuFold<(ArtifactLowFold > 1 ? ArtifactLowFold : 1),
                                            BaseSchedule, ArtifactHighFold, ArtifactTileK>;
};

struct AiuAProvider {};
struct PackedRowAProvider {};
struct OrdinaryBProvider {};
template <int Fold> struct FoldedBProvider {};
template <int ArtifactFold, int ComputeFold> struct VirtualFoldedBProvider {};
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
                                                          BaseSchedule, ArtifactTileK>::Type;
  using ElementBInfo = typename OperandInfo<Mode, ElementB, PlaneB2, ElementScale, ElementZero>::Type;
  using CollectiveBuilderType = cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementBInfo, LayoutB, AlignmentB, float,
      cute::tuple<TileShape, ScaleTileShape>, WarpShape, cute::Int<Stages>, KernelSchedule>;
  static_assert(CollectiveBuilderType::ArtifactTileK == ArtifactTileK,
                "ArtifactTileK must survive the shared policy/schedule boundary into CollectiveBuilder");
  using CollectiveOp = typename CollectiveBuilderType::CollectiveOp;

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

// Q4 tile-free resident bytes with a fold-2 logical MMA fragment.  This policy is deliberately separate from
// MainloopPolicy: default callers keep their exact schedule and CollectiveOp type, while an experimental build can
// opt into the L224-proved same-thread fragment permutation.  The first slice admits only T>=A64.  T32 requires a
// two-consume macrostep so the two halves of one A64 resident delivery are reused without doubling weight traffic;
// that lifetime is not represented here and is rejected at compile time.
template <int ComputeLowFold, QuantMode Mode, class BaseSchedule,
          class TileShape, class ScaleTileShape, class WarpShape,
          int Stages, bool AiuInterleaved, class ElementB = cutlass::int4b_t,
          int ArtifactTileK_ = 64>
struct VirtualFoldMainloopPolicy
    : MainloopPolicy<Mode, BaseSchedule, TileShape, ScaleTileShape, WarpShape,
                     Stages, AiuInterleaved, ElementB, void, ArtifactTileK_> {
private:
  using Ordinary = MainloopPolicy<Mode, BaseSchedule, TileShape, ScaleTileShape, WarpShape,
                                  Stages, AiuInterleaved, ElementB, void, ArtifactTileK_>;
public:
  static constexpr int LowBits = Ordinary::LowBits;
  static constexpr int HighBits = Ordinary::HighBits;
  static constexpr int TacticTileK = Ordinary::TacticTileK;
  static constexpr int ArtifactTileK = Ordinary::ArtifactTileK;
  static constexpr int ArtifactLowFold = Ordinary::ArtifactLowFold;
  static constexpr int ArtifactHighFold = Ordinary::ArtifactHighFold;
  static constexpr int ComputeFold = ComputeLowFold;
  static constexpr int TileK = Ordinary::TileK;
  static constexpr int LowFold = Ordinary::LowFold;
  static constexpr int HighFold = Ordinary::HighFold;
  static_assert(LowBits == 4 && HighBits == 0,
                "the first virtual-fold policy is the proved one-plane Q4 path only");
  static_assert(ArtifactTileK == 64 && ArtifactLowFold == 1 && ComputeLowFold == 2,
                "the first virtual-fold policy is exactly A64/F1 -> compute F2");
  static_assert(TacticTileK >= ArtifactTileK && TacticTileK % ArtifactTileK == 0,
                "sub-artifact TacticTileK requires the separately proved macrostep reader");

  using ElementA = typename Ordinary::ElementA;
  using ElementScale = typename Ordinary::ElementScale;
  using ElementZero = typename Ordinary::ElementZero;
  using LayoutA = typename Ordinary::LayoutA;
  using LayoutB = typename Ordinary::LayoutB;
  static constexpr int AlignmentA = Ordinary::AlignmentA;
  static constexpr int AlignmentB = Ordinary::AlignmentB;
  using KernelSchedule = cutlass::gemm::KernelAiuVirtualFold<ComputeLowFold,
                                                             typename Ordinary::KernelSchedule>;
  using ElementBInfo = typename Ordinary::ElementBInfo;
  using CollectiveBuilderType = cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementBInfo, LayoutB, AlignmentB, float,
      cute::tuple<TileShape, ScaleTileShape>, WarpShape, cute::Int<Stages>, KernelSchedule>;
  static_assert(CollectiveBuilderType::ArtifactTileK == ArtifactTileK,
                "virtual compute fold must preserve the A64 resident-byte contract");
  static_assert(CollectiveBuilderType::ArtifactLowFold == ArtifactLowFold,
                "virtual compute fold must not change physical gmem/smem folding");
  static_assert(CollectiveBuilderType::ComputeLowFold == ComputeLowFold,
                "virtual compute fold was lost before TiledMma construction");
  using CollectiveOp = typename CollectiveBuilderType::CollectiveOp;

  static constexpr bool PackedRowA = false;
  using AProvider = AiuAProvider;
  using BProvider = VirtualFoldedBProvider<ArtifactLowFold, ComputeLowFold>;
  struct Descriptor : MixedPolicyDescriptor<CollectiveOp, BaseSchedule, KernelSchedule, ElementBInfo,
      LayoutA, LayoutB, TileShape, ScaleTileShape, WarpShape, AProvider, BProvider,
      Mode, LowBits, HighBits, TacticTileK, ArtifactTileK, ArtifactLowFold, ArtifactHighFold,
      Stages, AiuInterleaved> {
    static constexpr int compute_low_fold = ComputeLowFold;
    static constexpr bool virtual_compute_fold = true;
  };
};

// Independent dense-M==1 A provider.  This intentionally does not add a parameter to MainloopPolicy: callers that
// omit this type continue to instantiate the exact old KernelAiuFold schedule, CollectiveOp and GemmKernel.  Only
// the ordinary unfolded one-plane collective is admitted here; folded and two-plane paths keep their existing A
// provider until separately proved.
template <int APackRows, QuantMode Mode, class BaseSchedule,
          class TileShape, class ScaleTileShape, class WarpShape,
          int Stages, bool AiuInterleaved, class ElementB = cutlass::int4b_t,
          int ArtifactTileK_ = 0>
struct PackedAMainloopPolicy
    : MainloopPolicy<Mode, BaseSchedule, TileShape, ScaleTileShape, WarpShape,
                     Stages, AiuInterleaved, ElementB, void, ArtifactTileK_> {
private:
  using Ordinary = MainloopPolicy<Mode, BaseSchedule, TileShape, ScaleTileShape, WarpShape,
                                  Stages, AiuInterleaved, ElementB, void, ArtifactTileK_>;
public:
  static_assert(APackRows == 1,
                "the first shipping packed-A provider is deliberately the exact dense M==1 path");
  static constexpr int PackedARows = APackRows;
  static_assert(Ordinary::ArtifactLowFold == 1 && Ordinary::HighBits == 0,
                "packed-A shipping seam is ordinary unfolded one-plane only");
  static_assert(int(cute::size<0>(TileShape{})) == 8 && int(cute::size<0>(WarpShape{})) == 8,
                "packed-A shipping seam is bound to the exact TM8/WM8 m8 instruction family");

  using ElementA = typename Ordinary::ElementA;
  using ElementScale = typename Ordinary::ElementScale;
  using ElementZero = typename Ordinary::ElementZero;
  using LayoutA = typename Ordinary::LayoutA;
  using LayoutB = typename Ordinary::LayoutB;
  static constexpr int AlignmentA = Ordinary::AlignmentA;
  static constexpr int AlignmentB = Ordinary::AlignmentB;
  static constexpr int LowBits = Ordinary::LowBits;
  static constexpr int HighBits = Ordinary::HighBits;
  static constexpr int TacticTileK = Ordinary::TacticTileK;
  static constexpr int ArtifactTileK = Ordinary::ArtifactTileK;
  static constexpr int ArtifactLowFold = Ordinary::ArtifactLowFold;
  static constexpr int ArtifactHighFold = Ordinary::ArtifactHighFold;
  static constexpr int TileK = Ordinary::TileK;
  static constexpr int LowFold = Ordinary::LowFold;
  static constexpr int HighFold = Ordinary::HighFold;

  using KernelSchedule = cutlass::gemm::KernelAiuPackedA<APackRows, typename Ordinary::KernelSchedule>;
  using ElementBInfo = typename Ordinary::ElementBInfo;
  using CollectiveBuilderType = cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementBInfo, LayoutB, AlignmentB, float,
      cute::tuple<TileShape, ScaleTileShape>, WarpShape, cute::Int<Stages>, KernelSchedule>;
  static_assert(CollectiveBuilderType::ArtifactTileK == ArtifactTileK,
                "packed-A wrapper must preserve the resident B artifact contract");
  using CollectiveOp = typename CollectiveBuilderType::CollectiveOp;

  static constexpr bool PackedRowA = true;
  using AProvider = PackedRowAProvider;
  using BProvider = OrdinaryBProvider;
  using Descriptor = MixedPolicyDescriptor<CollectiveOp, BaseSchedule, KernelSchedule, ElementBInfo,
      LayoutA, LayoutB, TileShape, ScaleTileShape, WarpShape, AProvider, BProvider,
      Mode, LowBits, HighBits, TacticTileK, ArtifactTileK, ArtifactLowFold, ArtifactHighFold,
      Stages, AiuInterleaved>;
};

// One guard for every adapter. TacticSpace is a public route name (DenseSpace or GroupedSpace); both are aliases of
// the one legality generator, while the instantiated mainloop and delivery checks remain one shared contract. Scale
// copy coverage is asserted by the mainloop's shared capped plan, where its concrete CTA and copy layout are known.
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
