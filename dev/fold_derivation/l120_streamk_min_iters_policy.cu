// L120 -- ONE STREAM-K QUANTUM TYPE MUST GOVERN HOST DECOMPOSITION AND DEVICE OWNERSHIP.
//
// The vendor scheduler historically fixed the minimum K stripe at 8.  Grouped decode needs a reviewed shorter
// stripe, but deriving a Params class cannot change the base methods' statically-bound constant, and copying only
// the host or device half silently makes decomposition disagree with seam ownership.  The vendor seam therefore
// carries MinIters in both the Params and scheduler types, with the legacy spelling remaining exactly ParamsT<8>.
//
// This is a device-free host oracle.  S068's target geometry has Q=128 output tiles and Kt=8.  At Min=8 the
// heuristic must decline Stream-K, while forced Stream-K can create at most one unit per output tile.  Grouped
// phase 2 explicitly selects Min=2 while the vendor-compatible default and legacy payload remain exactly Min=8.
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "cutlass/gemm/kernel/ppu_tile_scheduler_stream_k.hpp"
#include "lowbit_dense_configs.inc"

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

struct TailFallbackCase {
  uint32_t sk_tiles;
  uint64_t sk_units;
  uint32_t k_tiles;
  uint32_t expected_groups;
};

struct DenseConfig {
  uint32_t tm, tn, tk, wm, wn, stages, bchunk;
};

#define L120_DENSE_ROW(TM, TN, TK, WM, WN, ST, BC, BODY) \
  DenseConfig{TM, TN, TK, WM, WN, ST, BC},
constexpr DenseConfig DenseConfigs[] = {
  LOWBIT_DENSE_CFG_LIST(L120_DENSE_ROW, unused)
};
#undef L120_DENSE_ROW

// Every distinct production input signature among L201's 404 rows whose
// legacy G=8 boundary repair collapses at least one work unit.  L201 is the
// independent boundary/cell oracle; this table binds that result back to the
// actual Params helper rather than letting a correct parallel model coexist
// with different production code.
constexpr TailFallbackCase TailFallbackCases[] = {
    {80, 144, 16, 36}, {224, 288, 16, 36}, {152, 360, 32, 40},
    {160, 216, 16, 54}, {160, 288, 16, 72}, {304, 360, 16, 45},
    {448, 576, 16, 72}, {248, 360, 16, 90}, {320, 432, 16, 108},
    {320, 576, 16, 144}, {136, 360, 32, 40}, {56, 72, 16, 9},
    {272, 360, 16, 45}, {128, 504, 32, 42}, {40, 72, 16, 18},
    {112, 144, 16, 18}, {184, 216, 16, 27}, {184, 360, 16, 90},
    {256, 504, 16, 126},
};

static_assert(std::is_same_v<LegacyParams, Params8>);
static_assert(!std::is_same_v<Params8, Params2>);
static_assert(std::is_same_v<DefaultScheduler, Scheduler8>);
static_assert(std::is_same_v<typename DefaultScheduler::Params, Params8>);
static_assert(std::is_same_v<typename Scheduler2::Params, Params2>);
static_assert(std::is_same_v<SelectedParams, Params2>);
static_assert(DefaultScheduler::Params::min_iters_per_sk_unit_ == 8);
static_assert(Scheduler2::Params::min_iters_per_sk_unit_ == 2);
static_assert(int(Params8::DecompositionMode::Heuristic) == 0);
static_assert(int(Params8::DecompositionMode::DataParallel) == 1);
static_assert(int(Params8::DecompositionMode::SplitK) == 2);
static_assert(int(Params8::DecompositionMode::StreamK) == 3);
static_assert(int(Params8::DecompositionMode::StreamKTail) == 4,
              "tail-only must append without renumbering shipping modes");
static_assert(int(Params8::DecompositionMode::StreamKTailMinPeers) == 5,
              "tail-min-peers must append without renumbering existing modes");

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
  constexpr uint64_t TailQ = 2048;
  constexpr uint64_t TailW = 576;
  constexpr uint32_t TailKt = 64;
  auto legacy_tiles = Params8::get_num_sk_tiles(
      TailQ, TailW, 1, TailKt, Params8::DecompositionMode::StreamK);
  auto tail_tiles = Params8::get_num_sk_tiles(
      TailQ, TailW, 1, TailKt, Params8::DecompositionMode::StreamKTail);
  auto min_peer_tiles = Params8::get_num_sk_tiles(
      TailQ, TailW, 1, TailKt,
      Params8::DecompositionMode::StreamKTailMinPeers);
  auto tail_units = Params8::get_num_sk_units(
      cutlass::gemm::GemmCoord(1, 1, 1), TailW, tail_tiles, TailKt);
  auto tail_groups = Params8::get_stream_k_tail_safe_groups(
      8, tail_tiles, tail_units, TailKt);
  auto min_peer_groups = Params8::get_stream_k_tail_min_peer_groups(
      min_peer_tiles, tail_units, TailKt);
  uint32_t fallback_bindings = 0;
  for (auto const& signature : TailFallbackCases) {
    uint32_t const selected = Params8::get_stream_k_tail_safe_groups(
        8, signature.sk_tiles, signature.sk_units, signature.k_tiles);
    fallback_bindings += selected == signature.expected_groups;
  }
  uint32_t eligible_rows = 0;
  uint32_t production_attempts = 0;
  uint32_t production_preferred = 0;
  uint32_t production_fallback = 0;
  uint64_t production_dp_tiles = 0;
  uint64_t production_sk_tiles = 0;
  uint64_t production_sk_units = 0;
  uint32_t production_groups[145] = {};
  uint32_t production_min_peer_groups[289] = {};
  bool production_domain_ok = true;
  for (auto const& config : DenseConfigs) {
    uint32_t const threads = 32 * (config.tm / config.wm) * (config.tn / config.wn);
    if (config.tm < 16 || (threads != 64 && threads != 128) || config.stages - 1 > 8) {
      continue;
    }
    ++eligible_rows;
    uint64_t const q = uint64_t((2048 + config.tm - 1) / config.tm) *
                       uint64_t((4096 + config.tn - 1) / config.tn);
    uint32_t const kt = (4096 + config.tk - 1) / config.tk;
    for (uint32_t bpc = 1; bpc <= 8; ++bpc) {
      ++production_attempts;
      uint64_t const workers = 72u * bpc;
      uint32_t const sk = Params8::get_num_sk_tiles(
          q, workers, 1, kt, Params8::DecompositionMode::StreamKTail);
      uint64_t const units = Params8::get_num_sk_units(
          cutlass::gemm::GemmCoord(1, 1, 1), workers, sk, kt);
      bool const preferred_exact = Params8::stream_k_tail_intervals_cover_exactly(
          8, sk, units, kt);
      uint32_t const groups = Params8::get_stream_k_tail_safe_groups(8, sk, units, kt);
      uint32_t const min_groups = Params8::get_stream_k_tail_min_peer_groups(
          sk, units, kt);
      production_preferred += preferred_exact;
      production_fallback += !preferred_exact;
      production_dp_tiles += q - sk;
      production_sk_tiles += sk;
      production_sk_units += units;
      if (groups >= sizeof(production_groups) / sizeof(production_groups[0])) {
        production_domain_ok = false;
      } else {
        ++production_groups[groups];
      }
      if (min_groups >= sizeof(production_min_peer_groups) /
                            sizeof(production_min_peer_groups[0])) {
        production_domain_ok = false;
      } else {
        ++production_min_peer_groups[min_groups];
      }
      production_domain_ok &= sk == q % workers && groups != 0 &&
          min_groups != 0 &&
          Params8::stream_k_tail_intervals_cover_exactly(
              min_groups, sk, units, kt, true);
    }
  }
  constexpr struct { uint32_t groups, count; } ExpectedGroupCensus[] = {
      {8, 4212}, {9, 33}, {18, 20}, {27, 10}, {36, 6}, {40, 67},
      {42, 35}, {45, 51}, {54, 18}, {72, 36}, {90, 46}, {108, 36},
      {126, 10}, {144, 36},
  };
  uint32_t expected_group_total = 0;
  for (auto const& expected : ExpectedGroupCensus) {
    production_domain_ok &= production_groups[expected.groups] == expected.count;
    expected_group_total += expected.count;
    production_groups[expected.groups] = 0;
  }
  for (uint32_t count : production_groups) {
    production_domain_ok &= count == 0;
  }
  constexpr struct { uint32_t groups, count; } ExpectedMinPeerGroupCensus[] = {
      {8, 52}, {12, 39}, {16, 234}, {18, 123}, {24, 64}, {32, 449},
      {36, 419}, {48, 60}, {54, 146}, {63, 60}, {64, 434}, {72, 243},
      {80, 15}, {90, 141}, {96, 35}, {104, 36}, {108, 583},
      {126, 70}, {128, 134}, {136, 50}, {144, 167}, {152, 3},
      {160, 18}, {180, 383}, {208, 50}, {216, 292}, {252, 30},
      {256, 10}, {288, 276},
  };
  uint32_t expected_min_peer_total = 0;
  for (auto const& expected : ExpectedMinPeerGroupCensus) {
    production_domain_ok &=
        production_min_peer_groups[expected.groups] == expected.count;
    expected_min_peer_total += expected.count;
    production_min_peer_groups[expected.groups] = 0;
  }
  for (uint32_t count : production_min_peer_groups) {
    production_domain_ok &= count == 0;
  }
  production_domain_ok &= sizeof(DenseConfigs) / sizeof(DenseConfigs[0]) == 1772;
  production_domain_ok &= eligible_rows == 577 && production_attempts == 4616;
  production_domain_ok &= production_preferred == 4212 && production_fallback == 404;
  production_domain_ok &= expected_group_total == production_attempts;
  production_domain_ok &= expected_min_peer_total == production_attempts;
  production_domain_ok &= production_dp_tiles == 18973008 &&
                          production_sk_tiles == 671408 &&
                          production_sk_units == 1240056;
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
  ok &= legacy_tiles == 896 && tail_tiles == 320 &&
        min_peer_tiles == tail_tiles && tail_units == 576 &&
        tail_groups == 8 && min_peer_groups == 288;
  ok &= fallback_bindings == sizeof(TailFallbackCases) / sizeof(TailFallbackCases[0]);
  ok &= production_domain_ok;
  std::printf("L120 legacy=min8 size=%zu align=%zu trivial=%d; "
              "S068 min8 heuristic_tiles=%u forced_tiles=%u forced_units=%llu; "
              "min2 heuristic_tiles=%u forced_tiles=%u forced_units=%llu; "
              "Q2048/W576/Kt64 legacy/tail=%u/%u tail_units=%llu groups=%u min-peer-groups=%u; "
              "fallback-bindings=%u/%zu; "
              "production-domain=%u/%u preferred/fallback=%u/%u; "
              "boundary=%s\n",
              sizeof(Params8), alignof(Params8), int(std::is_trivially_copyable_v<Params8>),
              a.heuristic_tiles, a.forced_tiles, static_cast<unsigned long long>(a.forced_units),
              b.heuristic_tiles, b.forced_tiles, static_cast<unsigned long long>(b.forced_units),
              legacy_tiles, tail_tiles,
              static_cast<unsigned long long>(tail_units), tail_groups,
              min_peer_groups,
              fallback_bindings,
              sizeof(TailFallbackCases) / sizeof(TailFallbackCases[0]),
              eligible_rows, production_attempts,
              production_preferred, production_fallback,
              ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
