// L114 -- THE CAPPED SCALE COPY, THROUGH THE REAL PPU ATOM AND ALL THREE COLLECTIVES.
//
// Provider 0 enumerates a real N-contiguous gmem scale tile with the shared plan. Providers 1..3 independently
// instantiate ordinary, folded and two-plane mainloops and prove that their public copy alias is exactly that same
// real-atom construction. L114_PLANT_UNCAPPED_SCALE_COPY is the paired negative: the old layout must fail the shared
// witness before any runtime copy can silently truncate it.
#ifndef L114_PROVIDER
#define L114_PROVIDER 0
#endif

#if L114_PROVIDER == 0

#include <array>
#include <cstdio>

#include "cute/tensor.hpp"
#include "cute/arch/copy_ppu.hpp"
#include "cute/atom/copy_traits_ppu.hpp"
#include "cutlass/numeric_types.h"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_metadata_policy.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_packed_metadata_ownership.hpp"

using namespace cute;
namespace md = cutlass::gemm::collective::detail;

#if defined(L114_PLANT_UNCAPPED_SCALE_COPY) && (L114_PLANT_UNCAPPED_SCALE_COPY != 0)
// The old (H16,W8)xV8 construction: 128 logical copy slots against CTA64.
using MustReject = md::ScaleCopyCoverage<128, 8, 64, 16, 8>;
static_assert(MustReject::value, "negative control unexpectedly survived");
#endif

using RealAtom = Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cutlass::half_t>;
using Plan8 = md::ScaleCopyPlan<128, 8, 64>;
using Plan16 = md::ScaleCopyPlan<128, 16, 64>;
using Plan256 = md::ScaleCopyPlan<128, 8, 256>;
using Q4A32Plan = md::ScaleCopyPlan<64, 4, 256>;
// Exact pass/fail contrast from the Q4_K M=1 incident.  TM8/TN64/WN64 has one
// 32-thread warp, so the old wrapped protocol happens to be exact.  WN16 has
// four warps (128 CTA threads) but only 64 logical metadata slots, so modulo
// replay publishes every destination twice.
using Q4A64PassPlan = md::ScaleCopyPlan<64, 8, 32>;
using Q4A64FailPlan = md::ScaleCopyPlan<64, 8, 128>;
static_assert(Plan8::thread_layout_h_uncapped == 16 && Plan8::thread_layout_h == 8 &&
              Plan8::thread_layout_w == 8 && Plan8::values_per_thread == 16 &&
              Plan8::thread_slots == 64 && Plan8::Coverage::value);
static_assert(Plan16::thread_layout_h_uncapped == 16 && Plan16::thread_layout_h == 4 &&
              Plan16::thread_layout_w == 16 && Plan16::values_per_thread == 32 &&
              Plan16::thread_slots == 64 && Plan16::Coverage::value);
static_assert(Plan256::thread_layout_h_uncapped == 16 && Plan256::thread_layout_h == 16 &&
              Plan256::thread_layout_w == 8 && Plan256::values_per_thread == 8 &&
              Plan256::thread_slots == 128 && Plan256::Coverage::value);
static_assert(Q4A32Plan::thread_layout_h == 8 &&
              Q4A32Plan::thread_layout_w == 4 &&
              Q4A32Plan::values_per_thread == 8 &&
              Q4A32Plan::thread_slots == 32 && Q4A32Plan::Coverage::value);
static_assert(Q4A64PassPlan::thread_layout_h == 4 &&
              Q4A64PassPlan::thread_layout_w == 8 &&
              Q4A64PassPlan::values_per_thread == 16 &&
              Q4A64PassPlan::thread_slots == 32 && Q4A64PassPlan::Coverage::value);
static_assert(Q4A64FailPlan::thread_layout_h == 8 &&
              Q4A64FailPlan::thread_layout_w == 8 &&
              Q4A64FailPlan::values_per_thread == 8 &&
              Q4A64FailPlan::thread_slots == 64 && Q4A64FailPlan::Coverage::value);

struct CopyStats {
  int unique = 0;
  int visits = 0;
  int duplicates = 0;
  int oob = 0;
  int min_hits = 0;
  int max_hits = 0;
};

struct WarpPublicationStats {
  int clear_warps = 0;
  int decode_warps = 0;
  int surplus_clear_warps = 0;
  int same_warp_pairs = 0;
  int cross_warp_pairs = 0;
  int owner_cross_warp_pairs = 0;
  int surplus_cross_warp_pairs = 0;
  int cross_warp_values = 0;
  int min_cross_warp_values = 0;
  int max_cross_warp_values = 0;
};

enum class Protocol { RawAll, WrappedAll, OwnersOnly };

template <class Plan, Protocol P>
CopyStats copy_stats() {
  using ScaleCopy = decltype(make_tiled_copy(
      RealAtom{},
      Layout<Shape<Int<Plan::thread_layout_h>, Int<Plan::thread_layout_w>>>{},
      Layout<Shape<Int<Plan::values_per_thread>, _1>>{}));
  static_assert(Plan::Coverage::within_cta && Plan::Coverage::atom_aligned &&
                Plan::Coverage::full_tile);
  static_assert(int(size(ScaleCopy{})) == Plan::thread_slots);
  // Production constructs an N-contiguous (N,scale_k,L) tensor and local_tiles (TileN,ScaleTileK). Model two such
  // K tiles so partition_S must retain the same rest mode the real copy() iterates, then inspect interior tile zero.
  auto scale = make_counting_tensor(make_layout(
      make_shape(Int<Plan::tile_n>{}, Int<2 * Plan::tile_k>{}),
      make_stride(_1{}, Int<Plan::tile_n>{})));
  auto tiled_scale = local_tile(
      scale, make_shape(Int<Plan::tile_n>{}, Int<Plan::tile_k>{}),
      make_coord(0, _));
  std::array<unsigned char, Plan::tile_n * Plan::tile_k> seen{};
  CopyStats stats;
  for (int thread = 0; thread < Plan::cta_threads; ++thread) {
    if constexpr (P == Protocol::OwnersOnly) {
      if (!Plan::owns_physical_thread(thread)) continue;
    }
    int const logical_thread = P == Protocol::RawAll ?
        thread : Plan::logical_slot(thread);
    auto part = ScaleCopy{}.get_slice(logical_thread).partition_S(tiled_scale);
    auto tile0 = part(_, _, _, 0);
    for (int i = 0; i < int(size(tile0)); ++i) {
      int const offset = int(tile0(i));
      if (offset < 0 || offset >= int(seen.size())) {
        ++stats.oob;
      } else {
        ++stats.visits;
        if (seen[offset]++) ++stats.duplicates;
      }
    }
  }
  stats.min_hits = seen[0];
  for (auto hit : seen) {
    stats.unique += hit != 0;
    stats.min_hits = hit < stats.min_hits ? hit : stats.min_hits;
    stats.max_hits = hit > stats.max_hits ? hit : stats.max_hits;
  }
  return stats;
}

// Reconstruct the exact HISTORICAL source race in the Q4/A64 packed path.
// clear(tSsS) used ScaleCopyPlan's (N,group) TiledCopy, while
// packed_decode_stage used one packed-column owner over every group.  The old
// collective wrapped every physical CTA thread onto the smaller scale-copy
// layout, performed the clear before any mainloop work, then decoded after a
// per-thread async wait and before the first CTA barrier.  A scale-copy warp
// could therefore clear after a different packed-column warp decoded.  The
// production packed path no longer executes this clear at all: L217 proves its
// one decode-owner total overwrite, including the N tail.  This census remains
// the exact must-red legacy signature and the explanation for why merely
// filtering surplus clear publishers was not a complete repair.
template <class Plan>
WarpPublicationStats warp_publication_stats() {
  static_assert(Plan::cta_threads % 32 == 0);
  using ScaleCopy = decltype(make_tiled_copy(
      RealAtom{},
      Layout<Shape<Int<Plan::thread_layout_h>, Int<Plan::thread_layout_w>>>{},
      Layout<Shape<Int<Plan::values_per_thread>, _1>>{}));
  using Packed = md::PackedMetadataColumnOwnership<Plan::tile_n, Plan::cta_threads>;
  constexpr int Warps = Plan::cta_threads / 32;
  constexpr int Values = Plan::tile_n * Plan::tile_k;
  std::array<std::array<unsigned char, Values>, Warps> clears{};
  std::array<std::array<unsigned char, Values>, Warps> decodes{};

  auto metadata = make_counting_tensor(make_layout(
      make_shape(Int<Plan::tile_n>{}, Int<Plan::tile_k>{}),
      make_stride(_1{}, Int<Plan::tile_n>{})));
  for (int physical = 0; physical < Plan::cta_threads; ++physical) {
    auto part = ScaleCopy{}.get_slice(Plan::logical_slot(physical)).partition_D(metadata);
    for (int i = 0; i < int(size(part)); ++i) {
      int const address = int(part(i));
      if (address >= 0 && address < Values)
        clears[std::size_t(physical / 32)][std::size_t(address)] = 1;
    }
  }
  for (int physical = 0; physical < Packed::owner_threads; ++physical) {
    for (int sub = 0; sub < Packed::columns_per_thread; ++sub) {
      int const n = Packed::column(physical, sub);
      for (int group = 0; group < Plan::tile_k; ++group)
        decodes[std::size_t(physical / 32)]
               [std::size_t(n + Plan::tile_n * group)] = 1;
    }
  }

  WarpPublicationStats stats;
  for (int warp = 0; warp < Warps; ++warp) {
    bool clear_nonempty = false;
    bool decode_nonempty = false;
    for (int i = 0; i < Values; ++i) {
      clear_nonempty |= clears[std::size_t(warp)][std::size_t(i)] != 0;
      decode_nonempty |= decodes[std::size_t(warp)][std::size_t(i)] != 0;
    }
    stats.clear_warps += clear_nonempty;
    stats.decode_warps += decode_nonempty;
    if (clear_nonempty && !decode_nonempty) ++stats.surplus_clear_warps;
  }

  // A cross-warp intersection is a legal late-clear race: clear() happens
  // before the pipeline, packed decode happens after only a per-thread async
  // wait, and the first CTA barrier is after decode.  Same-warp intersections
  // retain program order; different warps have no such edge.  Surplus wrapped
  // publishers add races, but the scale-copy and packed-column maps are
  // orthogonal enough that active owner warps can also cross.
  for (int clear_warp = 0; clear_warp < Warps; ++clear_warp) {
    bool const surplus = clear_warp * 32 >= Plan::thread_slots;
    for (int decode_warp = 0; decode_warp < Warps; ++decode_warp) {
      int overlap = 0;
      int clear_count = 0;
      int decode_count = 0;
      for (int i = 0; i < Values; ++i) {
        bool const c = clears[std::size_t(clear_warp)][std::size_t(i)] != 0;
        bool const d = decodes[std::size_t(decode_warp)][std::size_t(i)] != 0;
        clear_count += c;
        decode_count += d;
        overlap += c && d;
      }
      if (overlap != 0) {
        std::printf("[l114] Q4/A64 CTA%d clear-warp=%d decode-warp=%d "
                    "overlap=%d clear-values=%d decode-values=%d surplus=%d\n",
                    Plan::cta_threads, clear_warp, decode_warp, overlap,
                    clear_count, decode_count, surplus ? 1 : 0);
      }
      if (overlap != 0) {
        if (clear_warp == decode_warp) {
          ++stats.same_warp_pairs;
        } else {
          ++stats.cross_warp_pairs;
          stats.cross_warp_values += overlap;
          if (surplus) ++stats.surplus_cross_warp_pairs;
          else ++stats.owner_cross_warp_pairs;
          if (stats.min_cross_warp_values == 0 || overlap < stats.min_cross_warp_values)
            stats.min_cross_warp_values = overlap;
          if (overlap > stats.max_cross_warp_values)
            stats.max_cross_warp_values = overlap;
        }
      }
    }
  }
  return stats;
}

void print_stats(char const* label, CopyStats const& stats) {
  std::printf("[l114] %s: unique=%d visits=%d dup=%d oob=%d hits=%d..%d\n",
              label, stats.unique, stats.visits, stats.duplicates, stats.oob,
              stats.min_hits, stats.max_hits);
}

int main() {
  auto const sk8 = copy_stats<Plan8, Protocol::OwnersOnly>();
  auto const sk16 = copy_stats<Plan16, Protocol::OwnersOnly>();
  auto const cta256_raw = copy_stats<Plan256, Protocol::RawAll>();
  auto const cta256_owner = copy_stats<Plan256, Protocol::OwnersOnly>();
  auto const cta256_wrap = copy_stats<Plan256, Protocol::WrappedAll>();
  auto const q4a32_owner = copy_stats<Q4A32Plan, Protocol::OwnersOnly>();
  auto const q4a32_legacy = copy_stats<Q4A32Plan, Protocol::WrappedAll>();
  auto const q4a64_pass_owner = copy_stats<Q4A64PassPlan, Protocol::OwnersOnly>();
  auto const q4a64_pass_legacy = copy_stats<Q4A64PassPlan, Protocol::WrappedAll>();
  auto const q4a64_fail_owner = copy_stats<Q4A64FailPlan, Protocol::OwnersOnly>();
  auto const q4a64_fail_legacy = copy_stats<Q4A64FailPlan, Protocol::WrappedAll>();
  auto const q4a64_pass_warps = warp_publication_stats<Q4A64PassPlan>();
  auto const q4a64_fail_warps = warp_publication_stats<Q4A64FailPlan>();
  print_stats("real-atom SK8 CTA64 owners", sk8);
  print_stats("real-atom SK16 CTA64 owners", sk16);
  print_stats("real-atom SK8 CTA256 raw", cta256_raw);
  print_stats("real-atom SK8 CTA256 owners", cta256_owner);
  print_stats("real-atom SK8 CTA256 legacy-wrapped", cta256_wrap);
  print_stats("Q4/A32 CTA256 owners", q4a32_owner);
  print_stats("Q4/A32 CTA256 legacy-wrapped", q4a32_legacy);
  print_stats("Q4/A64 CTA32 passing owners", q4a64_pass_owner);
  print_stats("Q4/A64 CTA32 passing legacy-wrapped", q4a64_pass_legacy);
  print_stats("Q4/A64 CTA128 failing owners", q4a64_fail_owner);
  print_stats("Q4/A64 CTA128 failing legacy-wrapped", q4a64_fail_legacy);
  std::printf("[l114] Q4/A64 CTA32 warp-publication clear=%d decode=%d "
              "surplus=%d same-warp=%d cross-warp=%d owner-cross=%d "
              "surplus-cross=%d overlap=%d..%d\n",
              q4a64_pass_warps.clear_warps, q4a64_pass_warps.decode_warps,
              q4a64_pass_warps.surplus_clear_warps,
              q4a64_pass_warps.same_warp_pairs,
              q4a64_pass_warps.cross_warp_pairs,
              q4a64_pass_warps.owner_cross_warp_pairs,
              q4a64_pass_warps.surplus_cross_warp_pairs,
              q4a64_pass_warps.min_cross_warp_values,
              q4a64_pass_warps.max_cross_warp_values);
  std::printf("[l114] Q4/A64 CTA128 warp-publication clear=%d decode=%d "
              "surplus=%d same-warp=%d cross-warp=%d owner-cross=%d "
              "surplus-cross=%d overlap=%d..%d\n",
              q4a64_fail_warps.clear_warps, q4a64_fail_warps.decode_warps,
              q4a64_fail_warps.surplus_clear_warps,
              q4a64_fail_warps.same_warp_pairs,
              q4a64_fail_warps.cross_warp_pairs,
              q4a64_fail_warps.owner_cross_warp_pairs,
              q4a64_fail_warps.surplus_cross_warp_pairs,
              q4a64_fail_warps.min_cross_warp_values,
              q4a64_fail_warps.max_cross_warp_values);

  bool const sk8_full = sk8.unique == 1024 && sk8.visits == 1024 && sk8.duplicates == 0 &&
                        sk8.oob == 0 && sk8.min_hits == 1 && sk8.max_hits == 1;
  bool const sk16_full = sk16.unique == 2048 && sk16.visits == 2048 && sk16.duplicates == 0 &&
                         sk16.oob == 0 && sk16.min_hits == 1 && sk16.max_hits == 1;
  // CuTe accepts surplus raw thread slices instead of wrapping them: t=128 starts at k=8, just outside TileK=8.
  bool const raw_walks_out = cta256_raw.unique == 1024 && cta256_raw.visits == 1024 &&
                             cta256_raw.duplicates == 0 && cta256_raw.oob == 1024 &&
                             cta256_raw.min_hits == 1 && cta256_raw.max_hits == 1;
  bool const owner_exact = cta256_owner.unique == 1024 && cta256_owner.visits == 1024 &&
                           cta256_owner.duplicates == 0 && cta256_owner.oob == 0 &&
                           cta256_owner.min_hits == 1 && cta256_owner.max_hits == 1;
  // The old modulo protocol reaches the right addresses but violates the
  // publication contract: the same async destination has two physical
  // publishers here and eight at the exact Q4/A32 row.
  bool const wrap_negative = cta256_wrap.unique == 1024 && cta256_wrap.visits == 2048 &&
                             cta256_wrap.duplicates == 1024 && cta256_wrap.oob == 0 &&
                             cta256_wrap.min_hits == 2 && cta256_wrap.max_hits == 2;
  bool const q4a32_exact = q4a32_owner.unique == 256 && q4a32_owner.visits == 256 &&
                           q4a32_owner.duplicates == 0 && q4a32_owner.oob == 0 &&
                           q4a32_owner.min_hits == 1 && q4a32_owner.max_hits == 1;
  bool const q4a32_legacy_red = q4a32_legacy.unique == 256 &&
                                q4a32_legacy.visits == 2048 &&
                                q4a32_legacy.duplicates == 1792 &&
                                q4a32_legacy.oob == 0 &&
                                q4a32_legacy.min_hits == 8 &&
                                q4a32_legacy.max_hits == 8;
  auto const exact_q4a64 = [] (CopyStats const& s) {
    return s.unique == 512 && s.visits == 512 && s.duplicates == 0 &&
           s.oob == 0 && s.min_hits == 1 && s.max_hits == 1;
  };
  bool const q4a64_pass_exact = exact_q4a64(q4a64_pass_owner) &&
                                exact_q4a64(q4a64_pass_legacy);
  bool const q4a64_fail_owner_exact = exact_q4a64(q4a64_fail_owner);
  bool const q4a64_fail_legacy_red =
      q4a64_fail_legacy.unique == 512 && q4a64_fail_legacy.visits == 1024 &&
      q4a64_fail_legacy.duplicates == 512 && q4a64_fail_legacy.oob == 0 &&
      q4a64_fail_legacy.min_hits == 2 && q4a64_fail_legacy.max_hits == 2;
  bool const q4a64_pass_no_late_clear =
      q4a64_pass_warps.clear_warps == 1 && q4a64_pass_warps.decode_warps == 1 &&
      q4a64_pass_warps.surplus_clear_warps == 0 &&
      q4a64_pass_warps.same_warp_pairs == 1 &&
      q4a64_pass_warps.cross_warp_pairs == 0;
  bool const q4a64_fail_exact_late_clear =
      q4a64_fail_warps.clear_warps == 4 && q4a64_fail_warps.decode_warps == 2 &&
      q4a64_fail_warps.surplus_clear_warps == 2 &&
      q4a64_fail_warps.same_warp_pairs == 2 &&
      q4a64_fail_warps.cross_warp_pairs == 6 &&
      q4a64_fail_warps.owner_cross_warp_pairs == 2 &&
      q4a64_fail_warps.surplus_cross_warp_pairs == 4 &&
      q4a64_fail_warps.cross_warp_values == 768 &&
      q4a64_fail_warps.min_cross_warp_values == 128 &&
      q4a64_fail_warps.max_cross_warp_values == 128;
  std::printf("[l114] coverage: SK8=%s SK16=%s CTA256-raw-oob=%s "
              "CTA256-owner=%s legacy-wrap=%s Q4A32-owner=%s Q4A32-legacy=%s "
              "Q4A64-CTA32=%s Q4A64-CTA128-owner=%s Q4A64-CTA128-legacy=%s "
              "Q4A64-CTA32-race=%s Q4A64-CTA128-race=%s\n",
              sk8_full ? "FULL" : "FAIL", sk16_full ? "FULL" : "FAIL",
              raw_walks_out ? "LOCKED" : "FAIL", owner_exact ? "EXACT" : "FAIL",
              wrap_negative ? "RED" : "FALSE-GREEN",
              q4a32_exact ? "EXACT" : "FAIL",
              q4a32_legacy_red ? "RED" : "FALSE-GREEN",
              q4a64_pass_exact ? "EXACT" : "FAIL",
              q4a64_fail_owner_exact ? "EXACT" : "FAIL",
              q4a64_fail_legacy_red ? "RED" : "FALSE-GREEN",
              q4a64_pass_no_late_clear ? "ABSENT" : "FAIL",
              q4a64_fail_exact_late_clear ? "EXACT-32-COLUMN" : "FAIL");
  return sk8_full && sk16_full && raw_walks_out && owner_exact &&
         wrap_negative && q4a32_exact && q4a32_legacy_red &&
         q4a64_pass_exact && q4a64_fail_owner_exact && q4a64_fail_legacy_red &&
         q4a64_pass_no_late_clear && q4a64_fail_exact_late_clear ? 0 : 1;
}

#else

#include <cstdio>
#include <type_traits>

#include "ppu_mixed_policy.hpp"
#include "actlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_fold.hpp"
#include "actlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_mixed_input_2plane.hpp"

using namespace cute;
using Q = ppu_mixed_policy::QuantMode;
using Schedule = ppu_group_schedule::FinegrainedSchedule<16>;
using TileShape = Shape<_64, _128, _256>;
using ScaleTileShape = Shape<_128, _16>;
using WarpShape = Shape<_64, _64, _256>;

#if L114_PROVIDER == 1
using Policy = ppu_mixed_policy::MainloopPolicy<
    Q::FinegrainedScaleOnly, Schedule, TileShape, ScaleTileShape, WarpShape,
    2, true, cutlass::int4b_t, void, 256>;
using ExpectedProvider = ppu_mixed_policy::OrdinaryBProvider;
static constexpr char const* kProviderName = "ordinary";
#elif L114_PROVIDER == 2
using Policy = ppu_mixed_policy::MainloopPolicy<
    Q::FinegrainedScaleOnly, Schedule, TileShape, ScaleTileShape, WarpShape,
    2, true, cutlass::uint1b_t, void, 128>;
using ExpectedProvider = ppu_mixed_policy::FoldedBProvider<2>;
static constexpr char const* kProviderName = "fold";
#elif L114_PROVIDER == 3
using Policy = ppu_mixed_policy::MainloopPolicy<
    Q::FinegrainedScaleOnly, Schedule, TileShape, ScaleTileShape, WarpShape,
    2, true, cutlass::uint2b_t, cutlass::uint1b_t, 256>;
using ExpectedProvider = ppu_mixed_policy::TwoPlaneBProvider<1, 1>;
static constexpr char const* kProviderName = "two-plane";
#else
#error L114_PROVIDER must be 0, 1, 2, or 3
#endif

using Mainloop = typename Policy::CollectiveOp;
using Plan = typename Mainloop::ScaleCopyPlan;
using ExpectedScaleCopy = decltype(make_tiled_copy(
    Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, typename Mainloop::NonVoidElementScale>{},
    Layout<Shape<Int<Plan::thread_layout_h>, Int<Plan::thread_layout_w>>>{},
    Layout<Shape<Int<Plan::values_per_thread>, _1>>{}));
static_assert(std::is_same_v<typename Policy::Descriptor::BProviderType, ExpectedProvider>);
static_assert(ppu_mixed_policy::kernel_policy_valid_v<ppu_tactics::DenseSpace, Policy>);
static_assert(std::is_same_v<typename Mainloop::GmemTiledCopyScale, ExpectedScaleCopy>,
              "collective is not using the capped real-PPU-atom scale copy");
static_assert(int(size(typename Mainloop::TiledMma{})) == 64);
static_assert(Mainloop::Scale_TileN == 128 && Mainloop::Scale_TileK == 16);
static_assert(int(size(typename Mainloop::GmemTiledCopyScale{})) == Plan::thread_slots &&
              Plan::thread_slots == 64);
static_assert(Plan::thread_layout_h == 4 && Plan::values_per_thread == 32,
              "the newly admitted TK256/w64x64 row must actively exercise four real atom iterations");
static_assert(Plan::Coverage::within_cta && Plan::Coverage::atom_aligned &&
              Plan::Coverage::full_tile);

int main() {
  std::printf("[l114] %s collective binds the capped real-atom scale copy\n", kProviderName);
  return 0;
}

#endif
