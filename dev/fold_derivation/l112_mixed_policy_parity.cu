// L112 -- DENSE AND GROUPED MUST BUILD THE SAME MIXED-INPUT MAINLOOP POLICY.
//
// The adapters intentionally differ after this seam: problem shape, scheduler, epilogue and output view are the
// operator. Everything represented by MixedPolicyDescriptor must be identical for matching inputs. These cases
// cover every B-provider form plus the ScaleOnly two-plane tuple that previously drifted into ScaleZero.
#include <cstdio>
#include <type_traits>

#include "fpA_intB_ppu.cuh"
#include "moe_grouped_ppu.cuh"

// The optional collectives this file INSTANTIATES. quactlize_actlize.hpp carries the base only.
#include "quactlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_fold.hpp"
#include "quactlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_mixed_input_2plane.hpp"

using namespace cute;
using Q = ppu_mixed_policy::QuantMode;

template <class Dense, class Grouped>
struct SamePolicy {
  static_assert(std::is_same_v<typename Dense::CollectiveOp, typename Grouped::CollectiveOp>,
                "dense/grouped mixed CollectiveMainloop types diverged");
  static_assert(std::is_same_v<typename Dense::Descriptor, typename Grouped::Descriptor>,
                "dense/grouped mixed policy descriptors diverged");
  static_assert(std::is_same_v<typename Dense::Descriptor::MetadataPolicyType,
                               typename Grouped::Descriptor::MetadataPolicyType>,
                "dense/grouped mixed metadata policies diverged");
  static_assert(std::is_same_v<typename Dense::Descriptor::PipelineDriverType,
                               typename Grouped::Descriptor::PipelineDriverType>,
                "dense/grouped mixed pipeline drivers diverged");
  static_assert(Dense::Descriptor::low_bits == Grouped::Descriptor::low_bits);
  static_assert(Dense::Descriptor::high_bits == Grouped::Descriptor::high_bits);
  static_assert(Dense::Descriptor::low_fold == Grouped::Descriptor::low_fold);
  static_assert(Dense::Descriptor::high_fold == Grouped::Descriptor::high_fold);
  static_assert(Dense::Descriptor::scale_tile_k == Grouped::Descriptor::scale_tile_k);
  static_assert(Dense::Descriptor::packed_metadata == Grouped::Descriptor::packed_metadata);
  static_assert(Dense::Descriptor::atom_at_a_time == Grouped::Descriptor::atom_at_a_time);
  static constexpr bool value = true;
};

template <Q Mode, int GroupSize, class Tile, class Scale, class Warp, int Stages,
          class Low, class High = void, bool Interleaved = true,
          int ArtifactTileK = 0>
struct AdapterPair {
  using Dense = fpa_intb_ppu::MixedMainloopPolicy<Mode, ppu_group_schedule::FinegrainedSchedule<GroupSize>,
      Tile, Scale, Warp, Stages, Interleaved, Low, High, ArtifactTileK>;
#if defined(PPU_PLANT_MIXED_POLICY_DRIFT) && (PPU_PLANT_MIXED_POLICY_DRIFT != 0)
  // Keep every input but the physical B layout the same. The planted compile must fail SamePolicy below; this is
  // how the local tier proves that the parity assertion observes a policy difference instead of comparing nothing.
  using Grouped = moe_grouped_ppu::MixedMainloopPolicy<Mode, ppu_group_schedule::FinegrainedSchedule<GroupSize>,
      Tile, Scale, Warp, Stages, false, Low, High, ArtifactTileK>;
#else
  using Grouped = moe_grouped_ppu::MixedMainloopPolicy<Mode, ppu_group_schedule::FinegrainedSchedule<GroupSize>,
      Tile, Scale, Warp, Stages, Interleaved, Low, High, ArtifactTileK>;
#endif
  static constexpr bool value = SamePolicy<Dense, Grouped>::value;
};

using T64 = Shape<_64, _128, _64>;
using W64x32 = Shape<_64, _32, _64>;
using W32x32 = Shape<_32, _32, _64>;
using W32x64 = Shape<_32, _64, _64>;
using S64Gs16 = Shape<_128, _4>;
using S64Gs32 = Shape<_128, _2>;
using S64Gs128 = Shape<_128, _1>;
using T128 = Shape<_64, _128, _128>;
using W64x64T128 = Shape<_64, _64, _128>;
using W32x64T128 = Shape<_32, _64, _128>;
using S128Gs32 = Shape<_128, _4>;
using T128N64 = Shape<_64, _64, _128>;
using W32x32T128 = Shape<_32, _32, _128>;
using S128N64Gs32 = Shape<_64, _4>;
using T256 = Shape<_64, _128, _256>;
using W64x64T256 = Shape<_64, _64, _256>;
using W32x64T256 = Shape<_32, _64, _256>;
using S256Gs32 = Shape<_128, _8>;

// Ordinary one-plane, both metadata modes.
static_assert(AdapterPair<Q::FinegrainedScaleOnly, 32, T64, S64Gs32, W64x32, 3,
                          cutlass::int4b_t>::value);
static_assert(AdapterPair<Q::FinegrainedScaleZero, 128, T64, S64Gs128, W32x32, 4,
                          cutlass::int4b_t>::value);
// Folded one-plane (int2/TK64 => F=2).
static_assert(AdapterPair<Q::FinegrainedScaleZero, 16, T64, S64Gs16, W32x64, 3,
                          cutlass::uint2b_t>::value);
// Two planes, with and without zero. The latter is the tuple branch that once silently selected ScaleZero.
static_assert(AdapterPair<Q::FinegrainedScaleZero, 16, T64, S64Gs16, W32x32, 3,
                          cutlass::int4b_t, cutlass::uint2b_t>::value);
static_assert(AdapterPair<Q::FinegrainedScaleOnly, 32, T64, S64Gs32, W32x32, 3,
                          cutlass::int4b_t, cutlass::uint2b_t>::value);

// ARTIFACT A MUST CROSS THE SCHEDULE SEAM EVEN WHEN F==1. A fold-only transport cannot distinguish int4 A64 from
// A128/A256, while this tactic deliberately has T=128. Assert each named boundary separately so a future refactor
// cannot leave the descriptor correct while silently reverting the builder to tactic-derived copy geometry.
using CrossT = AdapterPair<Q::FinegrainedScaleOnly, 32, T128, S128Gs32, W64x64T128, 4,
                           cutlass::int4b_t, void, true, 64>;
static_assert(CrossT::value);
using CrossDense = typename CrossT::Dense;
using CrossScheduleTraits = cutlass::gemm::fold_schedule_traits<typename CrossDense::KernelSchedule>;
static_assert(CrossDense::TacticTileK == 128 && CrossDense::ArtifactTileK == 64);
static_assert(CrossDense::ArtifactLowFold == 1,
              "F=1 is the case where ArtifactTileK cannot be reconstructed from the fold");
static_assert(CrossDense::Descriptor::artifact_tile_k == 64);
static_assert(CrossScheduleTraits::ArtifactTileK == 64);
static_assert(CrossDense::CollectiveBuilderType::ArtifactTileK == 64);
static_assert(CrossDense::CollectiveBuilderType::FullBlockK == 128);
static_assert(CrossDense::CollectiveBuilderType::CopyBlockK == 64);
static_assert(size<1>(typename CrossDense::CollectiveBuilderType::SmemLayoutAtomB0{}) == 64);
static_assert(CrossDense::CollectiveBuilderType::DefaultOperandB::InstNum == 2);

// Rebuild through the source-compatible legacy schedule (A=0). Its fallback A=T deliberately retains the old
// tactic-wide 64-B cube, while the explicit A64 contract uses two resident-sized 32-B cubes.
using CrossBaseSchedule = ppu_group_schedule::FinegrainedSchedule<32>;
using LegacyCrossSchedule = cutlass::gemm::KernelAiuFold<1, CrossBaseSchedule, 1>;
using LegacyCrossTraits = cutlass::gemm::fold_schedule_traits<LegacyCrossSchedule>;
static_assert(LegacyCrossTraits::ArtifactTileK == 0,
              "legacy KernelAiuFold spellings must retain their unspecified-artifact sentinel");
using LegacyCrossBuilder = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
    typename CrossDense::ElementA, typename CrossDense::LayoutA, CrossDense::AlignmentA,
    typename CrossDense::ElementBInfo, typename CrossDense::LayoutB, CrossDense::AlignmentB, float,
    cute::tuple<T128, S128Gs32>, W64x64T128, cute::Int<4>, LegacyCrossSchedule>;
static_assert(LegacyCrossBuilder::ArtifactTileK == 0);
static_assert(LegacyCrossBuilder::FullBlockK == 128 && LegacyCrossBuilder::CopyBlockK == 128);
static_assert(size<1>(typename LegacyCrossBuilder::SmemLayoutAtomB0{}) == 128);
static_assert(LegacyCrossBuilder::DefaultOperandB::InstNum == 1);
static_assert(!std::is_same_v<typename CrossDense::CollectiveOp,
                              typename LegacyCrossBuilder::CollectiveOp>,
              "an explicit cross-T artifact must select artifact-sized copy cubes");

// The three folded rows pinned by l115 must instantiate the same Full/Copy split in the real shared builder.
using CrossI2T128 = AdapterPair<Q::FinegrainedScaleOnly, 32, T128N64, S128N64Gs32, W32x32T128, 3,
                               cutlass::uint2b_t, void, true, 64>;
using CrossI1T128 = AdapterPair<Q::FinegrainedScaleOnly, 32, T128, S128Gs32, W32x64T128, 3,
                               cutlass::uint1b_t, void, true, 64>;
using CrossI1T256 = AdapterPair<Q::FinegrainedScaleOnly, 32, T256, S256Gs32, W32x64T256, 2,
                               cutlass::uint1b_t, void, true, 64>;
static_assert(CrossI2T128::value && CrossI1T128::value && CrossI1T256::value);
using CrossI2Builder = typename CrossI2T128::Dense::CollectiveBuilderType;
using CrossI1T128Builder = typename CrossI1T128::Dense::CollectiveBuilderType;
using CrossI1T256Builder = typename CrossI1T256::Dense::CollectiveBuilderType;
static_assert(CrossI2Builder::FullBlockK == 256 && CrossI2Builder::CopyBlockK == 128);
static_assert(size<1>(typename CrossI2Builder::SmemLayoutAtomB0{}) == 128 &&
              CrossI2Builder::DefaultOperandB::InstNum == 2);
static_assert(CrossI1T128Builder::FullBlockK == 512 && CrossI1T128Builder::CopyBlockK == 256);
static_assert(size<1>(typename CrossI1T128Builder::SmemLayoutAtomB0{}) == 256 &&
              CrossI1T128Builder::DefaultOperandB::InstNum == 2);
static_assert(CrossI1T256Builder::FullBlockK == 1024 && CrossI1T256Builder::CopyBlockK == 256);
static_assert(size<1>(typename CrossI1T256Builder::SmemLayoutAtomB0{}) == 256 &&
              CrossI1T256Builder::DefaultOperandB::InstNum == 4);
using CrossI2Mainloop = typename CrossI2T128::Dense::CollectiveOp;
using CrossI1T128Mainloop = typename CrossI1T128::Dense::CollectiveOp;
using CrossI1T256Mainloop = typename CrossI1T256::Dense::CollectiveOp;
static_assert(sizeof(typename CrossI2Mainloop::SharedStorage) > 0 &&
              sizeof(typename CrossI1T128Mainloop::SharedStorage) > 0 &&
              sizeof(typename CrossI1T256Mainloop::SharedStorage) > 0,
              "all three l115 rows must instantiate the folded collective body and storage contract");

// TWO-PLANE ARTIFACT CONTRACT. Both operands keep F*T as their full tactic span but size their copy cubes from F*A.
// The folds and A travel in the dispatch policy because atomK now describes a resident copy cube; atomK/T is no
// longer an integer and cannot be used to reconstruct either fold. Q3 exercises F2/F4 and P2_DIV=1.
using CrossTwo = AdapterPair<Q::FinegrainedScaleOnly, 32, T128, S128Gs32, W64x64T128, 4,
                            cutlass::uint2b_t, cutlass::uint1b_t, true, 64>;
static_assert(CrossTwo::value);
using CrossTwoDense = typename CrossTwo::Dense;
static_assert(CrossTwoDense::ArtifactLowFold == 2 && CrossTwoDense::ArtifactHighFold == 4);
using CrossTwoBuilder = typename CrossTwoDense::CollectiveBuilderType;
using CrossTwoMainloop = typename CrossTwoDense::CollectiveOp;
static_assert(CrossTwoBuilder::FullBlockK == 256 && CrossTwoBuilder::CopyBlockK == 128);
static_assert(CrossTwoBuilder::FullBlockK2 == 512 && CrossTwoBuilder::CopyBlockK2 == 256);
static_assert(CrossTwoBuilder::DefaultOperandB::InstNum == 2 &&
              CrossTwoBuilder::DefaultOperandB2::InstNum == 2);
static_assert(CrossTwoMainloop::P1Fold == 2 && CrossTwoMainloop::P2Fold == 4 &&
              CrossTwoMainloop::ArtifactTileK == 64 && CrossTwoMainloop::ArtifactP2Div == 1);
using LegacyCrossTwoSchedule = cutlass::gemm::KernelAiuFold<2, CrossBaseSchedule, 4>;
using LegacyCrossTwoBuilder = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
    typename CrossTwoDense::ElementA, typename CrossTwoDense::LayoutA, CrossTwoDense::AlignmentA,
    typename CrossTwoDense::ElementBInfo, typename CrossTwoDense::LayoutB, CrossTwoDense::AlignmentB, float,
    cute::tuple<T128, S128Gs32>, W64x64T128, cute::Int<4>, LegacyCrossTwoSchedule>;
static_assert(LegacyCrossTwoBuilder::CopyBlockK == LegacyCrossTwoBuilder::FullBlockK &&
              LegacyCrossTwoBuilder::CopyBlockK2 == LegacyCrossTwoBuilder::FullBlockK2);
static_assert(!std::is_same_v<typename CrossTwoDense::CollectiveOp,
                              typename LegacyCrossTwoBuilder::CollectiveOp>,
              "an explicit cross-T artifact must select artifact-sized cubes in both planes");

// Guard the other two fold pairs. Each has two resident-sized copies per full tactic span, and each artifact-sized
// policy must differ from the A=T legacy fallback.
using CrossQ5 = AdapterPair<Q::FinegrainedScaleOnly, 32, T128, S128Gs32, W64x64T128, 4,
                           cutlass::int4b_t, cutlass::uint1b_t, true, 64>;
using CrossQ5Dense = typename CrossQ5::Dense;
using LegacyQ5Schedule = cutlass::gemm::KernelAiuFold<1, CrossBaseSchedule, 4>;
using LegacyQ5Builder = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
    typename CrossQ5Dense::ElementA, typename CrossQ5Dense::LayoutA, CrossQ5Dense::AlignmentA,
    typename CrossQ5Dense::ElementBInfo, typename CrossQ5Dense::LayoutB, CrossQ5Dense::AlignmentB, float,
    cute::tuple<T128, S128Gs32>, W64x64T128, cute::Int<4>, LegacyQ5Schedule>;
using CrossQ5Builder = typename CrossQ5Dense::CollectiveBuilderType;
using CrossQ5Mainloop = typename CrossQ5Dense::CollectiveOp;
static_assert(CrossQ5::value && CrossQ5Builder::FullBlockK == 128 && CrossQ5Builder::CopyBlockK == 64 &&
              CrossQ5Builder::FullBlockK2 == 512 && CrossQ5Builder::CopyBlockK2 == 256);
static_assert(CrossQ5Mainloop::P1Fold == 1 && CrossQ5Mainloop::P2Fold == 4 &&
              CrossQ5Mainloop::ArtifactP2Div == 1);
static_assert(!std::is_same_v<typename CrossQ5Dense::CollectiveOp,
                              typename LegacyQ5Builder::CollectiveOp>,
              "Q5 cross-T must size both copy cubes from the artifact");

using CrossQ6 = AdapterPair<Q::FinegrainedScaleOnly, 32, T128, S128Gs32, W64x64T128, 4,
                           cutlass::int4b_t, cutlass::uint2b_t, true, 64>;
using CrossQ6Dense = typename CrossQ6::Dense;
using LegacyQ6Schedule = cutlass::gemm::KernelAiuFold<1, CrossBaseSchedule, 2>;
using LegacyQ6Builder = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
    typename CrossQ6Dense::ElementA, typename CrossQ6Dense::LayoutA, CrossQ6Dense::AlignmentA,
    typename CrossQ6Dense::ElementBInfo, typename CrossQ6Dense::LayoutB, CrossQ6Dense::AlignmentB, float,
    cute::tuple<T128, S128Gs32>, W64x64T128, cute::Int<4>, LegacyQ6Schedule>;
using CrossQ6Builder = typename CrossQ6Dense::CollectiveBuilderType;
using CrossQ6Mainloop = typename CrossQ6Dense::CollectiveOp;
static_assert(CrossQ6::value && CrossQ6Builder::FullBlockK == 128 && CrossQ6Builder::CopyBlockK == 64 &&
              CrossQ6Builder::FullBlockK2 == 256 && CrossQ6Builder::CopyBlockK2 == 128);
static_assert(CrossQ6Mainloop::P1Fold == 1 && CrossQ6Mainloop::P2Fold == 2 &&
              CrossQ6Mainloop::ArtifactP2Div == 1);
static_assert(!std::is_same_v<typename CrossQ6Dense::CollectiveOp,
                              typename LegacyQ6Builder::CollectiveOp>,
              "Q6 cross-T must size both copy cubes from the artifact");

// Exact shipping row that l115 uses to discharge ConsumerMap: Q6 A128/F1/F1 -> T256. P2_DIV remains two while
// each plane gains two artifact copy steps, so this also pins the chunk/scale re-derivation boundary.
using ShipQ6 = AdapterPair<Q::FinegrainedScaleOnly, 32, T256, S256Gs32, W64x64T256, 4,
                          cutlass::int4b_t, cutlass::uint2b_t, true, 128>;
static_assert(ShipQ6::value);
using ShipQ6Builder = typename ShipQ6::Dense::CollectiveBuilderType;
using ShipQ6Mainloop = typename ShipQ6::Dense::CollectiveOp;
static_assert(ShipQ6Builder::FullBlockK == 256 && ShipQ6Builder::CopyBlockK == 128 &&
              ShipQ6Builder::FullBlockK2 == 256 && ShipQ6Builder::CopyBlockK2 == 128);
static_assert(ShipQ6Builder::DefaultOperandB::InstNum == 2 &&
              ShipQ6Builder::DefaultOperandB2::InstNum == 2);
static_assert(ShipQ6Mainloop::P1Fold == 1 && ShipQ6Mainloop::P2Fold == 1 &&
              ShipQ6Mainloop::ArtifactTileK == 128 && ShipQ6Mainloop::ArtifactP2Div == 2);

int main() {
  std::printf("[l112] dense/grouped match; A sizes both planes; folds/P2_DIV cross the dispatch policy\n");
  return 0;
}
