// L139 -- host oracle for the classic-Marlin 2N x 4K CTA-local reduction.
//
// This is deliberately an integer/layout + exact-FP32 oracle.  It names the
// real PPU0010 builder result for the intended 16x128x128 / w16x64x32 tactic,
// projects each compute thread's VMNK coordinate onto the corresponding
// 2N x 1K output cohort, and then replays a shared-FP32 reduction.  No device
// timing, barrier ordering, or cross-CTA fixup is modeled here.
//
// The three planted failures are load-bearing:
//   1. use the wrong compact-thread projection for nonzero K cohorts;
//   2. omit one of the four K cohorts;
//   3. let the three non-survivor cohorts write output.
// Each must make the exact raw-output contract red.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#if defined(__HGGCCC__)
#include "quactlize_extensions/cutlass/quactlize_mix_gemm_convert.h"
#include "quactlize_extensions/cutlass/gemm/collective/builders/quactlize_mma_builder.inl"
#endif

namespace {
using namespace cute;

constexpr int kTM = 16;
constexpr int kTN = 128;
constexpr int kTK = 128;
constexpr int kWM = 16;
constexpr int kWN = 64;
constexpr int kWarpK = 32;
constexpr int kWOM = kTM / kWM;
constexpr int kWON = kTN / kWN;
constexpr int kWK = kTK / kWarpK;
constexpr int kAtomK = 16;
constexpr int kComputePermutationK = 64;

using TileShape = Shape<Int<kTM>, Int<kTN>, Int<kTK>>;
using WarpShape = Shape<Int<kWM>, Int<kWN>, Int<kWarpK>>;
using Atom = PPU0010_16x16x16_F32F16F16F32_TN;
using ComputeMma = TiledMMA<
    MMA_Atom<Atom>, Layout<Shape<Int<kWOM>, Int<kWON>, Int<kWK>>>,
    Tile<_16, _32, Int<kComputePermutationK>>>;

#if defined(L139_TYPE_ONLY)
using Built = cutlass::gemm::collective::quactlize_detail::get_tiled_mma<
    cutlass::arch::PPU0010, cutlass::half_t, cutlass::half_t, float,
    TileShape, WarpShape, Int<kComputePermutationK>>;
static_assert(std::is_same_v<ComputeMma, typename Built::TiledMma>,
    "the host oracle must name the builder's exact target TiledMMA type");
#endif

// The output cohort uses the same instruction, M/N warp ownership, and
// compute permutation; it only projects the K cohort to zero.
using OutputMma = TiledMMA<
    MMA_Atom<Atom>, Layout<Shape<Int<kWOM>, Int<kWON>, _1>>,
    Tile<_16, _32, Int<kComputePermutationK>>>;

static_assert(std::is_same_v<Atom,
    PPU0010_16x16x16_F32F16F16F32_TN>,
    "L139 must track the real PPU0010 FP16/F32 MMA atom");
static_assert(size(ComputeMma{}) == 256,
    "classic-aligned target must be an eight-warp CTA");
static_assert(size(OutputMma{}) == 64,
    "only the 2N (K=0) cohort may enter fixup/epilogue");
static_assert(size<3>(typename ComputeMma::ThrLayoutVMNK{}) == kWK,
    "the builder must expose four real K cohorts");
static_assert(decltype(ComputeMma{}.template permutation_mnk<2>()){} ==
                  Int<kComputePermutationK>{},
    "L139 must use the target compute permutation");

using Fragment = decltype(make_fragment_like<float>(partition_fragment_C(
    ComputeMma{}, make_shape(Int<kTM>{}, Int<kTN>{}))));
constexpr int kComputeThreads = size(ComputeMma{});
constexpr int kOutputThreads = size(OutputMma{});
constexpr int kFragmentSlots = size(Fragment{});
constexpr int kTileSlots = kTM * kTN;
constexpr int kScratchCohorts = kWK - 1;
constexpr int kScratchBytes =
    kScratchCohorts * kTileSlots * int(sizeof(float));

static_assert(kFragmentSlots == 32,
    "the intended target owns 32 FP32 C registers per thread");
static_assert(kOutputThreads * kFragmentSlots == kTileSlots);
static_assert(kScratchBytes == 24576,
    "three non-survivor FP32 tiles are the minimal all-live scratch");

#if defined(L139_TYPE_ONLY)

} // namespace

int main() {
  std::puts("L139 type PASS: host oracle TiledMMA is exactly the shipping builder target");
  return 0;
}

#else

uint32_t bits(float x) {
  uint32_t u = 0;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

struct ThreadInfo {
  int v = -1;
  int m = -1;
  int n = -1;
  int wk = -1;
  int compact = -1;
  std::array<int, kFragmentSlots> logical{};
};

int output_thread_from_vmnk(int v, int m, int n) {
  auto output_layout = OutputMma{}.get_thr_layout_vmnk();
  return int(output_layout(make_coord(v, m, n, 0)));
}

std::array<ThreadInfo, kComputeThreads> make_thread_table() {
  std::array<ThreadInfo, kComputeThreads> out{};
  auto identity = make_identity_tensor(make_shape(Int<kTM>{}, Int<kTN>{}));
  auto compute_layout = ComputeMma{}.get_thr_layout_vmnk();

  for (int t = 0; t < kComputeThreads; ++t) {
    auto coord = compute_layout.get_flat_coord(t);
    auto& x = out[t];
    x.v = int(get<0>(coord));
    x.m = int(get<1>(coord));
    x.n = int(get<2>(coord));
    x.wk = int(get<3>(coord));
    x.compact = output_thread_from_vmnk(x.v, x.m, x.n);

    auto part = ComputeMma{}.get_thread_slice(t).partition_C(identity);
    Fragment fragment;
    auto physical_to_fragment = right_inverse(fragment.layout());
    for (int i = 0; i < kFragmentSlots; ++i) {
      auto c = part(physical_to_fragment(i));
      int row = int(get<0>(c));
      int col = int(get<1>(c));
      x.logical[i] = row * kTN + col;
    }
  }
  return out;
}

struct GeometryResult {
  bool ok = true;
  int bad_compact_formula = 0;
  int bad_owner = 0;
  int bad_isomorphism = 0;
  int coverage_holes = 0;
  int coverage_duplicates = 0;
};

GeometryResult verify_geometry(
    std::array<ThreadInfo, kComputeThreads> const& table) {
  GeometryResult r;
  std::array<std::array<int, kWK>, kOutputThreads> owner{};
  for (auto& row : owner) row.fill(-1);
  std::array<std::array<int, kTileSlots>, kWK> coverage{};
  for (auto& row : coverage) row.fill(0);

  for (int t = 0; t < kComputeThreads; ++t) {
    auto const& x = table[t];
    // For Layout<Shape<1,2,4>>, V is the lane-fast mode and N follows it.
    // This equality is diagnostic only; the implementation contract above
    // derives compact from OutputMma::ThrLayoutVMNK, not from this formula.
    r.bad_compact_formula += x.compact != x.v + 32 * x.n;
    bool valid = 0 <= x.compact && x.compact < kOutputThreads &&
                 0 <= x.wk && x.wk < kWK;
    if (!valid) {
      ++r.bad_owner;
      continue;
    }
    if (owner[x.compact][x.wk] >= 0) ++r.bad_owner;
    owner[x.compact][x.wk] = t;
    for (int logical : x.logical) {
      if (0 <= logical && logical < kTileSlots) ++coverage[x.wk][logical];
      else ++r.bad_owner;
    }
  }

  for (int compact = 0; compact < kOutputThreads; ++compact) {
    int reference = owner[compact][0];
    if (reference < 0) {
      ++r.bad_owner;
      continue;
    }
    for (int wk = 0; wk < kWK; ++wk) {
      int t = owner[compact][wk];
      if (t < 0) {
        ++r.bad_owner;
        continue;
      }
      for (int i = 0; i < kFragmentSlots; ++i)
        r.bad_isomorphism += table[t].logical[i] !=
                             table[reference].logical[i];
    }
  }
  for (int wk = 0; wk < kWK; ++wk)
    for (int x : coverage[wk]) {
      r.coverage_holes += x == 0;
      r.coverage_duplicates += x > 1 ? x - 1 : 0;
    }
  r.ok = r.bad_compact_formula == 0 && r.bad_owner == 0 &&
         r.bad_isomorphism == 0 && r.coverage_holes == 0 &&
         r.coverage_duplicates == 0;
  return r;
}

float contribution(int logical, int wk) {
  // Integral values with a tiny bound make every partial and final FP32 sum
  // exactly representable.  Raw equality is therefore fixed before running.
  int magnitude = 1 + ((13 * logical + 7 * wk) % 31);
  int sign = ((logical + 3 * wk) & 1) ? -1 : 1;
  return float(sign * magnitude);
}

struct ReductionResult {
  bool ok = true;
  int raw_bitdiff = 0;
  int output_holes = 0;
  int output_duplicates = 0;
  int compact_coordinate_mismatches = 0;
};

ReductionResult simulate(
    std::array<ThreadInfo, kComputeThreads> const& table, int fault) {
  std::vector<float> direct(kTileSlots, 0.0f);
  std::vector<float> survivor(kTileSlots, 0.0f);
  std::vector<float> scratch(kScratchCohorts * kTileSlots, 0.0f);
  std::vector<int> output_hits(kTileSlots, 0);
  ReductionResult r;

  std::array<int, kOutputThreads> survivor_thread{};
  survivor_thread.fill(-1);
  for (int t = 0; t < kComputeThreads; ++t) {
    auto const& x = table[t];
    if (x.wk == 0 && 0 <= x.compact && x.compact < kOutputThreads)
      survivor_thread[x.compact] = t;
  }

  for (int t = 0; t < kComputeThreads; ++t) {
    auto const& x = table[t];
    for (int i = 0; i < kFragmentSlots; ++i) {
      int logical = x.logical[i];
      direct[logical] += contribution(logical, x.wk);

      int compact = x.compact;
      if (fault == 1 && x.wk != 0)
        compact = output_thread_from_vmnk(x.v, x.m, (x.n + 1) % kWON);
      int stripe = i * kOutputThreads + compact;
      r.compact_coordinate_mismatches += stripe < 0 || stripe >= kTileSlots ||
          (stripe >= 0 && stripe < kTileSlots &&
           table[t].logical[i] != table[compact].logical[i]);
      if (stripe < 0 || stripe >= kTileSlots) continue;

      if (x.wk == 0) survivor[stripe] = contribution(logical, x.wk);
      else scratch[(x.wk - 1) * kTileSlots + stripe] =
               contribution(logical, x.wk);
    }
  }

  for (int compact = 0; compact < kOutputThreads; ++compact) {
    int survivor_t = survivor_thread[compact];
    if (survivor_t < 0) {
      r.output_holes += kFragmentSlots;
      continue;
    }
    auto const& x = table[survivor_t];
    for (int i = 0; i < kFragmentSlots; ++i) {
      int stripe = i * kOutputThreads + compact;
      int last_wk = fault == 2 ? kWK - 1 : kWK;
      for (int wk = 1; wk < last_wk; ++wk)
        survivor[stripe] += scratch[(wk - 1) * kTileSlots + stripe];
      int logical = x.logical[i];
      ++output_hits[logical];
      r.raw_bitdiff += bits(survivor[stripe]) != bits(direct[logical]);
    }
  }

  if (fault == 3) {
    // Plant the forbidden behavior: every nonzero K cohort writes its own
    // partial in addition to the reduced wk=0 survivor.
    for (int t = 0; t < kComputeThreads; ++t) {
      auto const& x = table[t];
      if (x.wk == 0) continue;
      for (int logical : x.logical) ++output_hits[logical];
    }
  }

  for (int hits : output_hits) {
    r.output_holes += hits == 0;
    r.output_duplicates += hits > 1 ? hits - 1 : 0;
  }
  r.ok = r.raw_bitdiff == 0 && r.output_holes == 0 &&
         r.output_duplicates == 0 &&
         r.compact_coordinate_mismatches == 0;
  return r;
}

} // namespace

#ifndef L139_FAULT
#define L139_FAULT 0
#endif

int main() {
  auto table = make_thread_table();
  auto geometry = verify_geometry(table);
  auto reduction = simulate(table, L139_FAULT);
  bool ok = geometry.ok && reduction.ok;

  std::printf(
      "L139 target=16x128x128 warp=16x64x32 topology=2Nx4K "
      "threads=%d output_threads=%d frag=%d scratch_min=%dB\n",
      kComputeThreads, kOutputThreads, kFragmentSlots, kScratchBytes);
  std::printf(
      "L139 compact=OutputThrLayoutVMNK(v,m,n,0)=v+32*n "
      "formula_bad=%d owner_bad=%d isomorphism_bad=%d "
      "coverage_holes=%d coverage_duplicates=%d\n",
      geometry.bad_compact_formula, geometry.bad_owner,
      geometry.bad_isomorphism, geometry.coverage_holes,
      geometry.coverage_duplicates);
  std::printf(
      "L139 reduction fault=%d raw_bitdiff=%d output_holes=%d "
      "output_duplicates=%d compact_coordinate_mismatches=%d\n",
      L139_FAULT, reduction.raw_bitdiff, reduction.output_holes,
      reduction.output_duplicates, reduction.compact_coordinate_mismatches);

  if (L139_FAULT == 0) {
    std::puts(ok ? "L139 PASS: real 2Nx4K C fragments are isomorphic and "
                   "wk0 FP32 reduction is raw-bit exact"
                 : "L139 FAIL: positive contract is red");
    return ok ? 0 : 1;
  }
  if (ok) {
    std::printf("L139 UNEXPECTED-GREEN fault=%d\n", L139_FAULT);
    return 0;
  }
  std::printf("L139 EXPECTED-RED fault=%d\n", L139_FAULT);
  return 2;
}

#endif
