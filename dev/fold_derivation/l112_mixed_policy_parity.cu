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
  static_assert(Dense::Descriptor::compact_a_rows == Grouped::Descriptor::compact_a_rows);
  static_assert(Dense::Descriptor::packed_metadata == Grouped::Descriptor::packed_metadata);
  static_assert(Dense::Descriptor::atom_at_a_time == Grouped::Descriptor::atom_at_a_time);
  static constexpr bool value = true;
};

template <Q Mode, int GroupSize, class Tile, class Scale, class Warp, int Stages,
          class Low, class High = void, bool Interleaved = true,
          int ArtifactTileK = 0, int ACompactRows = cutlass::gemm::kDefaultACompactRows>
struct AdapterPair {
  using Dense = fpa_intb_ppu::MixedMainloopPolicy<Mode, ppu_group_schedule::FinegrainedSchedule<GroupSize>,
      Tile, Scale, Warp, Stages, Interleaved, Low, High, ArtifactTileK, ACompactRows>;
#if defined(PPU_PLANT_MIXED_POLICY_DRIFT) && (PPU_PLANT_MIXED_POLICY_DRIFT != 0)
  // Keep every input but the physical B layout the same. The planted compile must fail SamePolicy below; this is
  // how the local tier proves that the parity assertion observes a policy difference instead of comparing nothing.
  using Grouped = moe_grouped_ppu::MixedMainloopPolicy<Mode, ppu_group_schedule::FinegrainedSchedule<GroupSize>,
      Tile, Scale, Warp, Stages, false, Low, High, ArtifactTileK, ACompactRows>;
#else
  using Grouped = moe_grouped_ppu::MixedMainloopPolicy<Mode, ppu_group_schedule::FinegrainedSchedule<GroupSize>,
      Tile, Scale, Warp, Stages, Interleaved, Low, High, ArtifactTileK, ACompactRows>;
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

// Compact A is a per-policy axis. Explicit zero must remain ordinary even when PPU_A_CPASYNC supplies a nonzero
// compatibility default, while an explicit positive capacity reaches only the collective that implements it.
using ExplicitOrdinary = AdapterPair<Q::FinegrainedScaleOnly, 32, T64, S64Gs32, W64x32, 3,
                                     cutlass::int4b_t, void, true, 64, 0>;
using ExplicitCompact8 = AdapterPair<Q::FinegrainedScaleOnly, 32, T64, S64Gs32, W64x32, 3,
                                     cutlass::int4b_t, void, true, 64, 8>;
using UnsupportedFoldCompact8 = AdapterPair<Q::FinegrainedScaleZero, 16, T64, S64Gs16, W32x64, 3,
                                            cutlass::uint2b_t, void, true, 64, 8>;
static_assert(ExplicitOrdinary::Dense::ACompactRows == 0);
static_assert(ExplicitOrdinary::Dense::Descriptor::compact_a_rows == 0);
static_assert(ExplicitCompact8::Dense::ACompactRows == 8);
static_assert(ExplicitCompact8::Dense::Descriptor::compact_a_rows == 8);
static_assert(UnsupportedFoldCompact8::Dense::ACompactRows == 8);
static_assert(UnsupportedFoldCompact8::Dense::Descriptor::compact_a_rows == 0,
              "a requested compact capacity must not claim support in the folded collective");

int main() {
  std::printf("[l112] dense/grouped mixed policies match: ordinary, compact-8, folded and two-plane\n");
  return 0;
}
