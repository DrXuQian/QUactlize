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
#include "quactlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_metadata_policy.hpp"

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
static_assert(Plan8::thread_layout_h_uncapped == 16 && Plan8::thread_layout_h == 8 &&
              Plan8::thread_layout_w == 8 && Plan8::values_per_thread == 16 &&
              Plan8::thread_slots == 64 && Plan8::Coverage::value);
static_assert(Plan16::thread_layout_h_uncapped == 16 && Plan16::thread_layout_h == 4 &&
              Plan16::thread_layout_w == 16 && Plan16::values_per_thread == 32 &&
              Plan16::thread_slots == 64 && Plan16::Coverage::value);
static_assert(Plan256::thread_layout_h_uncapped == 16 && Plan256::thread_layout_h == 16 &&
              Plan256::thread_layout_w == 8 && Plan256::values_per_thread == 8 &&
              Plan256::thread_slots == 128 && Plan256::Coverage::value);

struct CopyStats {
  int unique = 0;
  int visits = 0;
  int duplicates = 0;
  int oob = 0;
  int min_hits = 0;
  int max_hits = 0;
};

template <class Plan, int ScaleGroups, int PhysicalThreads, bool Wrap>
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
      make_shape(Int<128>{}, Int<2 * ScaleGroups>{}), make_stride(_1{}, Int<128>{})));
  auto tiled_scale = local_tile(scale, make_shape(Int<128>{}, Int<ScaleGroups>{}), make_coord(0, _));
  std::array<unsigned char, 128 * ScaleGroups> seen{};
  CopyStats stats;
  for (int thread = 0; thread < PhysicalThreads; ++thread) {
    int const logical_thread = Wrap ? thread % Plan::thread_slots : thread;
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

void print_stats(char const* label, CopyStats const& stats) {
  std::printf("[l114] %s: unique=%d visits=%d dup=%d oob=%d hits=%d..%d\n",
              label, stats.unique, stats.visits, stats.duplicates, stats.oob,
              stats.min_hits, stats.max_hits);
}

int main() {
  auto const sk8 = copy_stats<Plan8, 8, 64, true>();
  auto const sk16 = copy_stats<Plan16, 16, 64, true>();
  auto const cta256_raw = copy_stats<Plan256, 8, 256, false>();
  auto const cta256_wrap = copy_stats<Plan256, 8, 256, true>();
  print_stats("real-atom SK8 CTA64 wrapped", sk8);
  print_stats("real-atom SK16 CTA64 wrapped", sk16);
  print_stats("real-atom SK8 CTA256 raw", cta256_raw);
  print_stats("real-atom SK8 CTA256 wrapped", cta256_wrap);

  bool const sk8_full = sk8.unique == 1024 && sk8.visits == 1024 && sk8.duplicates == 0 &&
                        sk8.oob == 0 && sk8.min_hits == 1 && sk8.max_hits == 1;
  bool const sk16_full = sk16.unique == 2048 && sk16.visits == 2048 && sk16.duplicates == 0 &&
                         sk16.oob == 0 && sk16.min_hits == 1 && sk16.max_hits == 1;
  // CuTe accepts surplus raw thread slices instead of wrapping them: t=128 starts at k=8, just outside TileK=8.
  bool const raw_walks_out = cta256_raw.unique == 1024 && cta256_raw.visits == 1024 &&
                             cta256_raw.duplicates == 0 && cta256_raw.oob == 1024 &&
                             cta256_raw.min_hits == 1 && cta256_raw.max_hits == 1;
  // The production modulo makes the surplus 128 threads repeat the complete tile, with no new addresses.
  bool const wrap_is_idempotent = cta256_wrap.unique == 1024 && cta256_wrap.visits == 2048 &&
                                  cta256_wrap.duplicates == 1024 && cta256_wrap.oob == 0 &&
                                  cta256_wrap.min_hits == 2 && cta256_wrap.max_hits == 2;
  std::printf("[l114] coverage: SK8=%s SK16=%s CTA256-raw-oob=%s CTA256-wrap-duplicate=%s\n",
              sk8_full ? "FULL" : "FAIL", sk16_full ? "FULL" : "FAIL",
              raw_walks_out ? "LOCKED" : "FAIL", wrap_is_idempotent ? "LOCKED" : "FAIL");
  return sk8_full && sk16_full && raw_walks_out && wrap_is_idempotent ? 0 : 1;
}

#else

#include <cstdio>
#include <type_traits>

#include "ppu_mixed_policy.hpp"
#include "quactlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_fold.hpp"
#include "quactlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_mixed_input_2plane.hpp"

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
