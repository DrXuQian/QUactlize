// L122 -- STREAM-K FIXUP MUST STRIPE OVER THE EXACT CTA THREAD COHORT.
//
// fixup() stores a full FP32 accumulator tile in block-striped order.  Its
// named-barrier arrival cohort and its workspace stripe width are therefore
// one invariant: both must equal the exact CTA thread count.  A 64-thread CTA
// paired with the legacy 128-thread cohort leaves half of every tile unwritten
// (and writes the other half past the tile); a 128-thread CTA paired with 64
// aliases its two half-CTAs onto the same slots.
//
// This is a device-free oracle bound to the real grouped Operations.  It proves
// the type seam and address coverage only; barrier progress and numerical
// output remain device gates.
#include <array>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "cutlass/block_striped.h"
#include "moe_grouped_streamk_ppu.cuh"

#ifndef L122_SELECTED_COHORT
#define L122_SELECTED_COHORT 64
#endif

namespace {

using QM = moe_grouped_streamk_ppu::QuantMode;
using int4_t = cutlass::int4b_t;
using Cluster = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;

// S068 decode champion: TM16/TN32/TK256, WM16/WN16, s3, Min2.
using Operation64 = moe_grouped_streamk_ppu::Operation<
    QM::FinegrainedScaleZero,
    ppu_group_schedule::FinegrainedSchedule<32>,
    cute::Shape<cute::Int<16>, cute::Int<32>, cute::Int<256>>,
    cute::Shape<cute::Int<32>, cute::Int<8>>,
    cute::Shape<cute::Int<16>, cute::Int<16>, cute::Int<256>>,
    3, true, int4_t, void, 64, 2>;

// Existing 128-thread phase-2 control: TM16/TN256/TK64, WM16/WN64, s3, Min2.
using Operation128 = moe_grouped_streamk_ppu::Operation<
    QM::FinegrainedScaleZero,
    ppu_group_schedule::FinegrainedSchedule<32>,
    cute::Shape<cute::Int<16>, cute::Int<256>, cute::Int<64>>,
    cute::Shape<cute::Int<256>, cute::Int<2>>,
    cute::Shape<cute::Int<16>, cute::Int<64>, cute::Int<64>>,
    3, true, int4_t, void, 64, 2>;

template <class Operation>
struct Geometry {
  using Kernel = typename Operation::Kernel;
  using Scheduler = typename Kernel::TileScheduler;
  using TileShape = typename Kernel::TileShape;
  using TiledMma = typename Kernel::TiledMma;
  using Fragment = decltype(cute::make_fragment_like<float>(
      cute::partition_fragment_C(
          TiledMma{}, cute::take<0, 2>(TileShape{}))));

  static constexpr int Threads = int(Kernel::MaxThreadsPerBlock);
  static constexpr int TileElements =
      int(cute::size<0>(TileShape{})) * int(cute::size<1>(TileShape{}));
  static constexpr int FragmentElements = int(cute::size(Fragment{}));
  using AccumulatorArray = cutlass::Array<float, FragmentElements>;
  using Reducer = cutlass::BlockStripedReduce<Threads, AccumulatorArray>;
  static constexpr int Stripes = Reducer::kStripes;

  static_assert(std::is_same_v<typename Fragment::value_type, float>);
  static_assert(Threads == int(cute::size(TiledMma{})));
  static_assert(Scheduler::FixupThreadCount == uint32_t(Threads));
  static_assert(Stripes == FragmentElements,
                "FP32 fixup must stripe at scalar accumulator granularity");
  static_assert(Threads * Stripes == TileElements,
                "one CTA must cover exactly one full accumulator tile");
  static_assert(sizeof(AccumulatorArray) * Threads ==
                    sizeof(float) * TileElements,
                "fragment geometry and FP32 workspace tile disagree");
};

using G64 = Geometry<Operation64>;
using G128 = Geometry<Operation128>;
static_assert(G64::Threads == 64 && G64::TileElements == 512 &&
              G64::FragmentElements == 8 && G64::Stripes == 8);
static_assert(G128::Threads == 128 && G128::TileElements == 4096 &&
              G128::FragmentElements == 32 && G128::Stripes == 32);

// Omitting the new cohort argument must remain exactly the legacy 128-thread
// scheduler type.  This is source/default compatibility, not a DSO ABI claim:
// the scheduler is header-only.
using DefaultLegacy =
    cutlass::gemm::kernel::detail::PersistentTileSchedulerPPUStreamK<
        typename G128::TileShape, Cluster>;
using ExplicitLegacy =
    cutlass::gemm::kernel::detail::PersistentTileSchedulerPPUStreamK<
        typename G128::TileShape, Cluster, 8, 128>;
using DefaultMin2 =
    cutlass::gemm::kernel::detail::PersistentTileSchedulerPPUStreamK<
        typename G128::TileShape, Cluster, 2>;
using ExplicitMin2Cohort128 =
    cutlass::gemm::kernel::detail::PersistentTileSchedulerPPUStreamK<
        typename G128::TileShape, Cluster, 2, 128>;
using ExplicitMin2Cohort64 =
    cutlass::gemm::kernel::detail::PersistentTileSchedulerPPUStreamK<
        typename G64::TileShape, Cluster, 2, 64>;
using SelectedCohort =
    cutlass::gemm::kernel::detail::PersistentTileSchedulerPPUStreamK<
        typename G64::TileShape, Cluster, 2, L122_SELECTED_COHORT>;

static_assert(std::is_same_v<DefaultLegacy, ExplicitLegacy>);
static_assert(std::is_same_v<DefaultMin2, ExplicitMin2Cohort128>);
static_assert(std::is_same_v<typename G128::Scheduler,
                             ExplicitMin2Cohort128>);
static_assert(std::is_same_v<typename G64::Scheduler,
                             ExplicitMin2Cohort64>);
static_assert(DefaultLegacy::FixupThreadCount == 128);
static_assert(DefaultMin2::FixupThreadCount == 128);

struct Coverage {
  uint64_t visits = 0;
  uint64_t holes = 0;
  uint64_t duplicate_visits = 0;
  uint64_t out_of_tile = 0;

  bool exact(uint64_t expected) const {
    return visits == expected && holes == 0 && duplicate_visits == 0 &&
           out_of_tile == 0;
  }
};

// Replay the address arithmetic used by BlockStripedReduce.  The stripe axis
// matters: checking only (q, thread) would miss both historical failure modes.
template <int ActualThreads, int CohortThreads, int Stripes,
          int TileElements, int Q = 3>
Coverage cover() {
  std::array<uint16_t, Q * TileElements> counts{};
  Coverage result;
  for (int q = 0; q < Q; ++q) {
    for (int thread = 0; thread < ActualThreads; ++thread) {
      int const cohort_thread = thread % CohortThreads;
      for (int stripe = 0; stripe < Stripes; ++stripe) {
        ++result.visits;
        int const local = stripe * CohortThreads + cohort_thread;
        if (local < 0 || local >= TileElements) {
          ++result.out_of_tile;
          continue;
        }
        ++counts[std::size_t(q) * TileElements + local];
      }
    }
  }
  for (uint16_t count : counts) {
    if (count == 0) {
      ++result.holes;
    } else if (count > 1) {
      result.duplicate_visits += uint64_t(count - 1);
    }
  }
  return result;
}

void print(char const* name, Coverage const& c) {
  std::printf("%s visits=%llu holes=%llu duplicate_visits=%llu "
              "out_of_tile=%llu\n",
              name, static_cast<unsigned long long>(c.visits),
              static_cast<unsigned long long>(c.holes),
              static_cast<unsigned long long>(c.duplicate_visits),
              static_cast<unsigned long long>(c.out_of_tile));
}

} // namespace

int main() {
  static_assert(sizeof(SelectedCohort) > 0);
  constexpr int Q = 3;
  Coverage const green64 =
      cover<G64::Threads, G64::Threads, G64::Stripes,
            G64::TileElements, Q>();
  Coverage const green128 =
      cover<G128::Threads, G128::Threads, G128::Stripes,
            G128::TileElements, Q>();

  // Red controls replay the two exact silent failures, not synthetic damage:
  // 64 actual / 128 cohort leaves half the tile untouched and crosses its end;
  // 128 actual / 64 cohort aliases two threads on every owned slot.
  Coverage const red64_as_128 =
      cover<G64::Threads, 128, G64::Stripes, G64::TileElements, Q>();
  Coverage const red128_as_64 =
      cover<G128::Threads, 64, G128::Stripes, G128::TileElements, Q>();

  bool ok = true;
  ok &= green64.exact(uint64_t(Q) * G64::TileElements);
  ok &= green128.exact(uint64_t(Q) * G128::TileElements);
  ok &= red64_as_128.visits == uint64_t(Q) * G64::TileElements;
  ok &= red64_as_128.holes == 768 &&
        red64_as_128.duplicate_visits == 0 &&
        red64_as_128.out_of_tile == 768;
  ok &= red128_as_64.visits == uint64_t(Q) * G128::TileElements;
  ok &= red128_as_64.holes == 6144 &&
        red128_as_64.duplicate_visits == 6144 &&
        red128_as_64.out_of_tile == 0;

  print("L122 green64", green64);
  print("L122 green128", green128);
  print("L122 red64-as-128", red64_as_128);
  print("L122 red128-as-64", red128_as_64);
  std::printf("L122 exact-fixup-cohort=%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
