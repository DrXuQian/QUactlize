// L231 -- compose the production Q4 K-pack4 load fragment with the production
// m8 compute fragment for every decode (TileN,WarpN) geometry.
//
// The historical collective converted one four-register physical-b16
// delivery into 32 fp16 values and wrote them contiguously into the raw
// storage of tCrB_mma.  That is valid only when the raw storage order of the
// transposed m16 loader and the logical m8 compute fragment compose to the
// same 32-value cohort.  L228 proved that statement only for one N16
// microgeometry.  This oracle evaluates the exact production TiledMMA types,
// physical SmemLayoutB, partition_fragment_B layouts and converter emission
// map for all 12 geometries admitted by the K-pack4 pilot.

#include <array>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "actlize_extensions/cutlass/quactlize_mix_gemm_convert.h"
#include "q4_kpack4_offline.hpp"

namespace {
using namespace cute;

constexpr int kTK = 256;
constexpr int kPhysicalK = kTK / q4_kpack4::kPack;
constexpr int kWidth = 32;

#ifndef L231_ROTATE_DESTINATION
#define L231_ROTATE_DESTINATION 0
#endif
#ifndef L231_LEGACY_CANDIDATE
#define L231_LEGACY_CANDIDATE 0
#endif
#ifndef L231_KPACK4_DELIVERY_N
#define L231_KPACK4_DELIVERY_N 0
#endif
static_assert(L231_KPACK4_DELIVERY_N == 0 ||
              L231_KPACK4_DELIVERY_N == 16 ||
              L231_KPACK4_DELIVERY_N == 32 ||
              L231_KPACK4_DELIVERY_N == 64);

struct Metrics {
  int exact = 0;
  int candidate_exact = 0;
  int total = 0;
  int source_holes = 0;
  int destination_holes = 0;
  int desired_duplicates = 0;
  int current_duplicates = 0;
  int candidate_duplicates = 0;
  int cohort_set_mismatch = 0;
  int noncontiguous_desired = 0;
  int wrong_n = 0;
  int wrong_k = 0;
  int cohort_iterations = 0;
  int cohort_map_conflicts = 0;
  std::array<int, 32> cohort_map{};
  std::array<int, 32> current_cohort_map{};
  std::array<int, 32> candidate_cohort_map{};
  std::array<int, 9> delta_hist{};  // desired-current, clamped to [-4,4]
};

template <int TN, int WN>
Metrics prove_geometry() {
  static_assert(TN % WN == 0);
  static_assert(WN == 16 || WN == 32 || WN == 64);
  using WarpOnN = Int<TN / WN>;
  using PermutationN = Int<(TN / WN) * 16>;
  using Compute = TiledMMA<
      MMA_Atom<PPU0010_8x16x16_F32F16F16F32_TN>,
      Layout<Shape<_1, WarpOnN, _1>>,
      Tile<_8, PermutationN, Int<kTK>>>;
  using Loader = TiledMMA<
      MMA_Atom<PPU0010_16x16x16_F32F16F16F32_TN>,
      Layout<Shape<_1, WarpOnN, _1>>,
      Tile<_16, PermutationN, _16>>;
  // DefaultGemm_AIU_Operand<PPU0010,half,true,TN,K/4,true> derives
  // CUBE_W from a 128-byte maximum N-contiguous run.  Spell only that public
  // type algebra here: including gemm_operands.hpp drags device collectives
  // into an nvcc host oracle and crosses the documented compiler boundary.
  static constexpr int kCubeW = L231_KPACK4_DELIVERY_N > 0
      ? (TN < L231_KPACK4_DELIVERY_N ? TN : L231_KPACK4_DELIVERY_N)
      : (TN < 64 ? TN : 64);
  static_assert(kCubeW <= TN && TN % kCubeW == 0,
                "K-pack4 delivery N must exactly tile tactic N");
  using SmemLayoutAtomB =
      Layout<Shape<Int<kCubeW>, Int<kPhysicalK>>,
             Stride<_1, Int<kCubeW>>>;
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(Int<TN>{}, Int<kPhysicalK>{}, _1{})));

  auto physical = make_tensor(
      make_smem_ptr((cutlass::half_t*)nullptr), SmemLayoutB{});
  auto logical = make_tensor(
      make_smem_ptr((cutlass::half_t*)nullptr),
      make_layout(Shape<Int<TN>, Int<kTK>>{},
                  Stride<Int<kTK>, _1>{}));
  auto physical_identity = make_identity_tensor(
      Shape<Int<TN>, Int<kPhysicalK>>{});
  auto logical_identity = make_identity_tensor(Shape<Int<TN>, Int<kTK>>{});

  Metrics m{};
  m.cohort_map.fill(-1);
  m.current_cohort_map.fill(-1);
  m.candidate_cohort_map.fill(-1);
  std::array<int, TN * kTK> desired_hits{};
  std::array<int, TN * kTK> current_hits{};
  std::array<int, TN * kTK> candidate_hits{};

  for (int thread = 0; thread < int(size(Compute{})); ++thread) {
    auto load_thr = Loader{}.get_thread_slice(thread);
    auto load_frag = load_thr.partition_fragment_B(physical(_, _, 0));
    auto load_part = load_thr.partition_B(physical_identity);
    auto load_pi = right_inverse(load_frag.layout());

    auto compute_thr = Compute{}.get_thread_slice(thread);
    auto compute_frag = compute_thr.partition_fragment_B(logical);
    auto compute_part = compute_thr.partition_B(logical_identity);
    auto compute_pi = right_inverse(compute_frag.layout());

    constexpr int load_halfs = int(size(decltype(load_frag.layout()){}));
    constexpr int compute_halfs = int(size(decltype(compute_frag.layout()){}));
    static_assert(compute_halfs == load_halfs * q4_kpack4::kPack);
    static_assert((load_halfs * q4_kpack4::kPack) % kWidth == 0);
    constexpr int n_iterations = int(size<1>(decltype(load_frag.layout()){}));
    constexpr int k_blocks = int(size<2>(decltype(load_frag.layout()){}));
    m.cohort_iterations = n_iterations * k_blocks;

    std::array<int, TN * kTK> logical_to_raw{};
    logical_to_raw.fill(-1);
    for (int raw = 0; raw < compute_halfs; ++raw) {
      auto const c = compute_part(compute_pi(raw));
      int const n = int(get<0>(c));
      int const k = int(get<1>(c));
      int const flat = n * kTK + k;
      if (flat < 0 || flat >= TN * kTK || logical_to_raw[flat] >= 0)
        ++m.destination_holes;
      else
        logical_to_raw[flat] = raw;
    }

    for (int k_block = 0; k_block < k_blocks; ++k_block) {
      auto load_step = load_frag(_, _, k_block);
      auto cvt_in = recast<cutlass::int4b_t>(load_step);
      auto candidate_n_layout = [&] {
#if L231_LEGACY_CANDIDATE
        return make_layout(shape<1>(cvt_in.layout()),
                           stride<1>(cvt_in.layout()));
#else
        return make_layout(
            shape<1>(cvt_in.layout()),
            compact_col_major(shape<1>(cvt_in.layout()),
                              stride<1>(compute_frag.layout())));
#endif
      }();
      int const load_step_base =
          int(load_frag.layout()(0, 0, k_block));
      int const compute_step_base =
          int(compute_frag.layout()(0, 0, k_block * 4));
      for (int ii = 0; ii < n_iterations; ++ii) {
        int const cvt_n_base = int(cvt_in.layout()(0, ii));
        int const source_half_base = load_step_base + cvt_n_base / 4;
        int const current_base = compute_step_base + cvt_n_base;
        int const candidate_base =
            compute_step_base + int(candidate_n_layout(ii));
        int const cohort_slot = k_block * n_iterations + ii;
        std::array<int, kWidth> desired_in_cohort{};
        for (int vreg = 0; vreg < 4; ++vreg) {
          for (int code = 0; code < 8; ++code) {
            int const source_half = source_half_base + 2 * vreg + code / 4;
            auto const p = load_part(load_pi(source_half));
            int const n = int(get<0>(p));
            int const kg = int(get<1>(p));
            int const k = q4_kpack4::logical_k(kg, code % 4);
            int const logical_flat = n * kTK + k;
            if (logical_flat < 0 || logical_flat >= TN * kTK) {
              ++m.source_holes;
              continue;
            }
            int const desired = logical_to_raw[logical_flat];
            if (desired < 0) {
              ++m.destination_holes;
              continue;
            }
            int const emitted =
                (cutlass::MixGemmEmit<4>::index(code, vreg) +
                 L231_ROTATE_DESTINATION) % kWidth;
            int const current = current_base + emitted;
            int const candidate = candidate_base + emitted;
            auto const got_c = compute_part(compute_pi(current));
            auto const candidate_c = compute_part(compute_pi(candidate));
            int const got_n = int(get<0>(got_c));
            int const got_k = int(get<1>(got_c));
            m.exact += got_n == n && got_k == k;
            m.candidate_exact += int(get<0>(candidate_c)) == n &&
                                 int(get<1>(candidate_c)) == k;
            m.wrong_n += got_n != n;
            m.wrong_k += got_k != k;
            ++m.total;
            ++desired_hits[logical_flat];
            int const current_flat = got_n * kTK + got_k;
            if (current_flat >= 0 && current_flat < TN * kTK)
              ++current_hits[current_flat];
            else
              ++m.destination_holes;
            int const candidate_flat =
                int(get<0>(candidate_c)) * kTK + int(get<1>(candidate_c));
            if (candidate_flat >= 0 && candidate_flat < TN * kTK)
              ++candidate_hits[candidate_flat];
            else
              ++m.destination_holes;
            desired_in_cohort[emitted] = desired;
            int delta = desired - current;
            if (delta < -4) delta = -4;
            if (delta > 4) delta = 4;
            ++m.delta_hist[std::size_t(delta + 4)];
          }
        }
        std::array<int, kWidth> sorted = desired_in_cohort;
        for (int i = 0; i < kWidth; ++i)
          for (int j = i + 1; j < kWidth; ++j)
            if (sorted[j] < sorted[i]) {
              int const tmp = sorted[i]; sorted[i] = sorted[j]; sorted[j] = tmp;
            }
        bool contiguous = true;
        for (int i = 1; i < kWidth; ++i)
          contiguous &= sorted[i] == sorted[0] + i;
        m.noncontiguous_desired += !contiguous;
        bool same_set = contiguous && sorted[0] == current_base;
        m.cohort_set_mismatch += !same_set;
        int const desired_cohort = contiguous ? sorted[0] / kWidth : -1;
        if (m.cohort_map[std::size_t(cohort_slot)] < 0)
          m.cohort_map[std::size_t(cohort_slot)] = desired_cohort;
        else
          m.cohort_map_conflicts +=
              m.cohort_map[std::size_t(cohort_slot)] != desired_cohort;
        int const current_cohort = current_base / kWidth;
        if (m.current_cohort_map[std::size_t(cohort_slot)] < 0)
          m.current_cohort_map[std::size_t(cohort_slot)] = current_cohort;
        else
          m.cohort_map_conflicts +=
              m.current_cohort_map[std::size_t(cohort_slot)] != current_cohort;
        int const candidate_cohort = candidate_base / kWidth;
        if (m.candidate_cohort_map[std::size_t(cohort_slot)] < 0)
          m.candidate_cohort_map[std::size_t(cohort_slot)] = candidate_cohort;
        else
          m.cohort_map_conflicts +=
              m.candidate_cohort_map[std::size_t(cohort_slot)] !=
              candidate_cohort;
      }
    }
  }

  for (int i = 0; i < TN * kTK; ++i) {
    m.desired_duplicates += desired_hits[std::size_t(i)] != 1;
    m.current_duplicates += current_hits[std::size_t(i)] != 1;
    m.candidate_duplicates += candidate_hits[std::size_t(i)] != 1;
  }
  return m;
}

template <int TN, int WN>
bool report(bool expected_current_exact) {
  Metrics const m = prove_geometry<TN, WN>();
  bool const current_exact = m.exact == m.total;
  bool const candidate_exact = m.candidate_exact == m.total;
  bool const structurally_exact =
      m.source_holes == 0 && m.destination_holes == 0 &&
      m.desired_duplicates == 0 && m.candidate_duplicates == 0 &&
      m.cohort_map_conflicts == 0;
  bool const current_expectation = L231_KPACK4_DELIVERY_N == 0
      ? current_exact == expected_current_exact : true;
  bool const ok = current_expectation &&
                  candidate_exact && structurally_exact;
  std::printf(
      "L231 GEOMETRY TN=%d WN=%d warp_on_n=%d current=%s exact=%d/%d "
      "candidate=%s wrong_n=%d wrong_k=%d cohort_set_mismatch=%d "
      "noncontiguous_desired=%d source_holes=%d destination_holes=%d "
      "desired_denominator_bad=%d current_denominator_bad=%d "
      "candidate_denominator_bad=%d result=%s\n",
      TN, WN, TN / WN, current_exact ? "IDENTITY" : "NONIDENTITY",
      m.exact, m.total, candidate_exact ? "IDENTITY" : "NONIDENTITY",
      m.wrong_n, m.wrong_k, m.cohort_set_mismatch,
      m.noncontiguous_desired, m.source_holes, m.destination_holes,
      m.desired_duplicates, m.current_duplicates, m.candidate_duplicates,
      ok ? "PASS" : "FAIL");
  std::printf("L231 COHORT_MAP TN=%d WN=%d map=", TN, WN);
  for (int i = 0; i < m.cohort_iterations; ++i)
    std::printf("%s%d:%d->%d/%d", i == 0 ? "" : ",", i,
                m.current_cohort_map[std::size_t(i)],
                m.candidate_cohort_map[std::size_t(i)],
                m.cohort_map[std::size_t(i)]);
  std::printf(" conflicts=%d\n", m.cohort_map_conflicts);
  return ok;
}

template <int TN, int WN>
void print_layouts() {
  using WarpOnN = Int<TN / WN>;
  using PermutationN = Int<(TN / WN) * 16>;
  using Compute = TiledMMA<
      MMA_Atom<PPU0010_8x16x16_F32F16F16F32_TN>,
      Layout<Shape<_1, WarpOnN, _1>>,
      Tile<_8, PermutationN, Int<kTK>>>;
  using Loader = TiledMMA<
      MMA_Atom<PPU0010_16x16x16_F32F16F16F32_TN>,
      Layout<Shape<_1, WarpOnN, _1>>,
      Tile<_16, PermutationN, _16>>;
  static constexpr int kCubeW = L231_KPACK4_DELIVERY_N > 0
      ? (TN < L231_KPACK4_DELIVERY_N ? TN : L231_KPACK4_DELIVERY_N)
      : (TN < 64 ? TN : 64);
  static_assert(kCubeW <= TN && TN % kCubeW == 0,
                "K-pack4 delivery N must exactly tile tactic N");
  using Atom = Layout<Shape<Int<kCubeW>, Int<kPhysicalK>>,
                      Stride<_1, Int<kCubeW>>>;
  using Stage = decltype(tile_to_shape(
      Atom{}, make_shape(Int<TN>{}, Int<kPhysicalK>{}, _1{})));
  auto physical = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr), Stage{});
  auto logical = make_tensor(
      make_smem_ptr((cutlass::half_t*)nullptr),
      make_layout(Shape<Int<TN>, Int<kTK>>{}, Stride<Int<kTK>, _1>{}));
  auto lf = Loader{}.get_thread_slice(0).partition_fragment_B(physical(_, _, 0));
  auto cvt = recast<cutlass::int4b_t>(lf);
  auto cf = Compute{}.get_thread_slice(0).partition_fragment_B(logical);
  std::printf("L231 LAYOUT TN=%d WN=%d load=", TN, WN); print(lf.layout());
  std::printf(" cvt="); print(cvt.layout());
  std::printf(" compute="); print(cf.layout());
  std::putchar('\n');
}

}  // namespace

int main() {
  bool ok = true;
  print_layouts<32, 32>();
  print_layouts<64, 32>();
  print_layouts<64, 64>();
  print_layouts<128, 32>();
  print_layouts<128, 64>();
  print_layouts<256, 64>();
  ok &= report<16, 16>(true);
  ok &= report<32, 16>(true);
  ok &= report<32, 32>(false);
  ok &= report<64, 16>(true);
  ok &= report<64, 32>(false);
  ok &= report<64, 64>(false);
  ok &= report<128, 16>(true);
  ok &= report<128, 32>(true);
  ok &= report<128, 64>(false);
  ok &= report<256, 16>(true);
  ok &= report<256, 32>(true);
  ok &= report<256, 64>(true);
  std::printf("L231 KPACK4_PRODUCTION_FRAGMENT %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
