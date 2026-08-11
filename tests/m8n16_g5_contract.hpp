#pragma once

// One type-level spelling of the #112/G5 launch.  The production harness and
// the L125 host algebra oracle both consume this contract; neither is allowed
// to reconstruct the template tuple independently.

#include <type_traits>

#include "cutlass/numeric_types.h"
#include "m8n16_g5_layout_spec.hpp"
#include "ppu_group_schedule.hpp"
#include "ppu_mixed_policy.hpp"

namespace m8n16_g5_contract {

inline constexpr int kBits = m8n16_g5_layout_spec::kBits;
inline constexpr int kN = m8n16_g5_layout_spec::kN;
inline constexpr int kTacticK = m8n16_g5_layout_spec::kTacticK;
inline constexpr int kStoredRowK = m8n16_g5_layout_spec::kStoredRowK;
inline constexpr int kK = m8n16_g5_layout_spec::kK;
inline constexpr int kGroupSize = m8n16_g5_layout_spec::kGroupSize;
inline constexpr int kScaleK = m8n16_g5_layout_spec::kScaleK;
inline constexpr int kStages = m8n16_g5_layout_spec::kStages;
inline constexpr int kExperts = m8n16_g5_layout_spec::kExperts;
inline constexpr bool kAiuInterleaved = m8n16_g5_layout_spec::kAiuInterleaved;
inline constexpr ppu_mixed_policy::QuantMode kQuantMode =
    ppu_mixed_policy::QuantMode::FinegrainedScaleZero;

using WeightElement = cutlass::int4b_t;
using Schedule = ppu_group_schedule::FinegrainedSchedule<kGroupSize>;

template <int TM, int WM>
struct Launch {
  static constexpr int kTileM = TM;
  static constexpr int kWarpM = WM;
  static constexpr auto QuantMode = kQuantMode;
  static constexpr int Stages = kStages;
  static constexpr bool AiuInterleaved = kAiuInterleaved;
  using BaseSchedule = Schedule;
  using ElementB = WeightElement;
  using Tile = cute::Shape<cute::Int<TM>, cute::Int<kN>, cute::Int<kTacticK>>;
  using Scale = cute::Shape<cute::Int<kN>, cute::Int<kTacticK / kGroupSize>>;
  using Warp = cute::Shape<cute::Int<WM>, cute::Int<kN>, cute::Int<kTacticK>>;
  using Policy = ppu_mixed_policy::MainloopPolicy<
      QuantMode, BaseSchedule, Tile, Scale, Warp, Stages,
      AiuInterleaved, ElementB>;
  using Mainloop = typename Policy::CollectiveOp;
};

using M8 = Launch<8, 8>;
using M16 = Launch<16, 16>;

static_assert(kScaleK == 8 && cute::size<1>(typename M8::Scale{}) == 2);
static_assert(int(cute::size(typename M8::Mainloop::TiledMma{})) ==
              m8n16_g5_layout_spec::kCtaThreads);
static_assert(M8::Policy::ArtifactTileK == kTacticK);
static_assert(M8::Policy::ArtifactLowFold == 1);
static_assert(std::is_same_v<typename M8::Mainloop::ElementB, WeightElement>);
static_assert(M8::Mainloop::MetadataPolicy::has_zero);
static_assert(!M8::Mainloop::is_packed_scale,
              "G5's fp16 zero-plane oracle does not describe packed metadata");

}  // namespace m8n16_g5_contract
