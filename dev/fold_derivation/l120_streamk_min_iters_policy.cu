// L120 -- ONE STREAM-K QUANTUM TYPE MUST GOVERN HOST DECOMPOSITION AND DEVICE OWNERSHIP.
//
// The vendor scheduler historically fixed the minimum K stripe at 8.  Grouped decode needs a reviewed shorter
// stripe, but deriving a Params class cannot change the base methods' statically-bound constant, and copying only
// the host or device half silently makes decomposition disagree with seam ownership.  The vendor seam therefore
// carries MinIters in both the Params and scheduler types, with the legacy spelling remaining exactly ParamsT<8>.
//
// This is a device-free host oracle.  S068's target geometry has Q=128 output tiles and Kt=8.  At Min=8 the
// heuristic must decline Stream-K, while forced Stream-K can create at most one unit per output tile.  Min=2 is
// deliberately only a type/mechanism witness here; grouped phase 1 still ships Min=8.
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "cutlass/gemm/kernel/ppu_tile_scheduler_stream_k.hpp"

#ifndef L120_SELECTED_MIN
#define L120_SELECTED_MIN 2
#endif

namespace {

using namespace cutlass::gemm::kernel::detail;

using LegacyParams = PersistentTileSchedulerPPUStreamKParams;
using Params8 = PersistentTileSchedulerPPUStreamKParamsT<8>;
using Params2 = PersistentTileSchedulerPPUStreamKParamsT<2>;
using SelectedParams = PersistentTileSchedulerPPUStreamKParamsT<L120_SELECTED_MIN>;

using Tile = cute::Shape<cute::Int<16>, cute::Int<32>, cute::Int<256>>;
using Cluster = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;
using DefaultScheduler = PersistentTileSchedulerPPUStreamK<Tile, Cluster>;
using Scheduler8 = PersistentTileSchedulerPPUStreamK<Tile, Cluster, 8>;
using Scheduler2 = PersistentTileSchedulerPPUStreamK<Tile, Cluster, 2>;

static_assert(std::is_same_v<LegacyParams, Params8>);
static_assert(!std::is_same_v<Params8, Params2>);
static_assert(std::is_same_v<DefaultScheduler, Scheduler8>);
static_assert(std::is_same_v<typename DefaultScheduler::Params, Params8>);
static_assert(std::is_same_v<typename Scheduler2::Params, Params2>);
static_assert(DefaultScheduler::MinIters == 8 && Scheduler2::MinIters == 2);

// A policy specialization must not change the payload ABI.  Params travel from host lowering into device code;
// a size/alignment change here would turn an apparently source-compatible default into a binary contract change.
static_assert(sizeof(Params8) == sizeof(Params2));
static_assert(alignof(Params8) == alignof(Params2));
static_assert(std::is_trivially_copyable_v<Params8> == std::is_trivially_copyable_v<Params2>);
static_assert(std::is_trivially_copyable_v<Params8>);

template <class Params>
struct Result {
  uint32_t heuristic_tiles;
  uint32_t forced_tiles;
  uint64_t forced_units;
  uint32_t at_min_heuristic_tiles;
  uint32_t at_min_forced_tiles;
  uint64_t at_min_forced_units;
  uint32_t above_min_heuristic_tiles;
  uint32_t above_min_forced_tiles;
  uint64_t above_min_forced_units;
};

template <class Params>
Result<Params> measure() {
  constexpr uint64_t Q = 128;
  constexpr uint64_t W = 432;
  constexpr uint32_t Kt = 8;
  cutlass::gemm::GemmCoord cluster(1, 1, 1);
  auto h = Params::get_num_sk_tiles(Q, W, 1, Kt, Params::DecompositionMode::Heuristic);
  auto f = Params::get_num_sk_tiles(Q, W, 1, Kt, Params::DecompositionMode::StreamK);
  auto u = Params::get_num_sk_units(cluster, W, f, Kt);
  constexpr uint32_t Min = Params::min_iters_per_sk_unit_;
  auto at_h = Params::get_num_sk_tiles(Q, W, 1, Min, Params::DecompositionMode::Heuristic);
  auto at_f = Params::get_num_sk_tiles(Q, W, 1, Min, Params::DecompositionMode::StreamK);
  auto at_u = Params::get_num_sk_units(cluster, W, at_f, Min);
  auto above_h = Params::get_num_sk_tiles(Q, W, 1, Min + 1, Params::DecompositionMode::Heuristic);
  auto above_f = Params::get_num_sk_tiles(Q, W, 1, Min + 1, Params::DecompositionMode::StreamK);
  auto above_u = Params::get_num_sk_units(cluster, W, above_f, Min + 1);
  return {h, f, u, at_h, at_f, at_u, above_h, above_f, above_u};
}

} // namespace

int main() {
  // Force SelectedParams to instantiate so -DL120_SELECTED_MIN=0 is a compile-fail control, not an unused alias.
  static_assert(sizeof(SelectedParams) > 0);
  auto a = measure<Params8>();
  auto b = measure<Params2>();
  bool ok = true;
  ok &= a.heuristic_tiles == 0;
  ok &= a.forced_tiles == 128 && a.forced_units == 128;
  ok &= a.at_min_heuristic_tiles == 0 && a.at_min_forced_tiles == 128 && a.at_min_forced_units == 128;
  ok &= a.above_min_heuristic_tiles == 128 && a.above_min_forced_tiles == 128 &&
        a.above_min_forced_units == 144;
  ok &= b.heuristic_tiles == 128;
  ok &= b.forced_tiles == 128 && b.forced_units == 432;
  ok &= b.at_min_heuristic_tiles == 0 && b.at_min_forced_tiles == 128 && b.at_min_forced_units == 128;
  ok &= b.above_min_heuristic_tiles == 128 && b.above_min_forced_tiles == 128 &&
        b.above_min_forced_units == 192;
  std::printf("L120 legacy=min8 size=%zu align=%zu trivial=%d; "
              "S068 min8 heuristic_tiles=%u forced_tiles=%u forced_units=%llu; "
              "min2 heuristic_tiles=%u forced_tiles=%u forced_units=%llu; boundary=%s\n",
              sizeof(Params8), alignof(Params8), int(std::is_trivially_copyable_v<Params8>),
              a.heuristic_tiles, a.forced_tiles, static_cast<unsigned long long>(a.forced_units),
              b.heuristic_tiles, b.forced_tiles, static_cast<unsigned long long>(b.forced_units),
              ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
