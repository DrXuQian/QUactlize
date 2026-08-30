// L236 -- compose the production per-plane b16 K-pack loader with the
// production converter destination for Q2/Q3/Q5/Q6.
//
// This is not an offline round-trip.  It walks the actual PPU0010 m16
// transposed-load fragment, the actual compute-B fragment, MixGemmEmit and
// (for two-plane formats) HiPlaneSrc + MixGemm2Plane.  PPU0010's m8 and m16
// fp16 atoms publish the same BLayout (proved below), so one exhaustive B
// composition covers both decode and dense/grouped tile-M families.  Every
// logical (N,K) code must reach exactly one compute slot, and every high-plane
// bit selected by the converter must name the same logical code as its low
// plane partner.

#include <array>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "actlize_extensions/cutlass/quactlize_mix_gemm_convert.h"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_2plane_source_layout.hpp"
#include "kquant_kpack_offline.hpp"

namespace {
using namespace cute;

static_assert(std::is_same_v<
    typename MMA_Traits<PPU0010_8x16x16_F32F16F16F32_TN>::BLayout,
    typename MMA_Traits<PPU0010_16x16x16_F32F16F16F32_TN>::BLayout>,
    "m8/m16 production atoms must retain one exact B-fragment ABI");

constexpr int kTK = 256;

#ifndef L236_ROTATE_DESTINATION
#define L236_ROTATE_DESTINATION 0
#endif
#ifndef L236_LEGACY_LOADER_STRIDE
#define L236_LEGACY_LOADER_STRIDE 0
#endif
#ifndef L236_SHIFT_HIGH_SOURCE
#define L236_SHIFT_HIGH_SOURCE 0
#endif
#ifndef L236_PRINT_MAP
#define L236_PRINT_MAP 0
#endif
#ifndef L236_PAIR_DIAG
#define L236_PAIR_DIAG 0
#endif

template <int Bits> struct ElementForBits;
template <> struct ElementForBits<1> { using type = cutlass::uint1b_t; };
template <> struct ElementForBits<2> { using type = cutlass::uint2b_t; };
template <> struct ElementForBits<4> { using type = cutlass::int4b_t; };

template <int TN, int WN>
struct MmaTypes {
  static_assert(TN % WN == 0);
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
};

struct PlaneMetrics {
  int exact = 0;
  int total = 0;
  int source_holes = 0;
  int destination_holes = 0;
  int source_denominator_bad = 0;
  int destination_denominator_bad = 0;
};

template <int Bits, int Group, int TN, int WN>
PlaneMetrics prove_plane() {
  constexpr int Pack = 16 / Bits;
  constexpr int PhysicalK = kTK / Pack;
  constexpr int CubeW = TN < 64 ? TN : 64;
  constexpr int CodesPerVreg = 32 / Bits;
  constexpr int Width = 4 * CodesPerVreg;
  using Elem = typename ElementForBits<Bits>::type;
  using Compute = typename MmaTypes<TN, WN>::Compute;
  using Loader = typename MmaTypes<TN, WN>::Loader;
  using Atom = Layout<Shape<Int<CubeW>, Int<PhysicalK>>,
                      Stride<_1, Int<CubeW>>>;
  using Stage = decltype(tile_to_shape(
      Atom{}, make_shape(Int<TN>{}, Int<PhysicalK>{}, _1{})));

  auto physical = make_tensor(
      make_smem_ptr((cutlass::half_t*)nullptr), Stage{});
  auto logical = make_tensor(
      make_smem_ptr((cutlass::half_t*)nullptr),
      make_layout(Shape<Int<TN>, Int<kTK>>{},
                  Stride<Int<kTK>, _1>{}));
  auto physical_identity =
      make_identity_tensor(Shape<Int<TN>, Int<PhysicalK>>{});
  auto logical_identity = make_identity_tensor(Shape<Int<TN>, Int<kTK>>{});

  PlaneMetrics m{};
  std::array<int, TN * kTK> source_hits{};
  std::array<int, TN * kTK> destination_hits{};
  for (int thread = 0; thread < int(size(Compute{})); ++thread) {
    auto load_thr = Loader{}.get_thread_slice(thread);
    auto load_frag = load_thr.partition_fragment_B(physical(_, _, 0));
    auto load_part = load_thr.partition_B(physical_identity);
    auto load_pi = right_inverse(load_frag.layout());

    auto compute_thr = Compute{}.get_thread_slice(thread);
    auto compute_frag = compute_thr.partition_fragment_B(logical);
    auto compute_part = compute_thr.partition_B(logical_identity);
    auto compute_pi = right_inverse(compute_frag.layout());

    constexpr int LoadHalfs = int(size(decltype(load_frag.layout()){}));
    constexpr int ComputeHalfs = int(size(decltype(compute_frag.layout()){}));
    static_assert(ComputeHalfs == LoadHalfs * Pack);
    constexpr int KBlocks = int(size<2>(decltype(load_frag.layout()){}));
    constexpr int NumIter = int(size<1>(decltype(load_frag.layout()){}));
    static_assert(Width * NumIter * KBlocks == ComputeHalfs);

    for (int k_block = 0; k_block < KBlocks; ++k_block) {
      auto load_step = load_frag(_, _, k_block);
      auto cvt_in = recast<Elem>(load_step);
      auto destination_n_layout = [&] {
#if L236_LEGACY_LOADER_STRIDE
        return make_layout(shape<1>(cvt_in.layout()),
                           stride<1>(cvt_in.layout()));
#else
        return make_layout(
            shape<1>(cvt_in.layout()),
            compact_col_major(shape<1>(cvt_in.layout()),
                              stride<1>(compute_frag.layout())));
#endif
      }();
      int const load_step_base = int(load_frag.layout()(0, 0, k_block));
      int const compute_step_base =
          int(compute_frag.layout()(0, 0, k_block * Pack));
      for (int ii = 0; ii < NumIter; ++ii) {
        int const cvt_n_base = int(cvt_in.layout()(0, ii));
        int const source_half_base = load_step_base + cvt_n_base / Pack;
        int const destination_base =
            compute_step_base + int(destination_n_layout(ii));
        for (int vreg = 0; vreg < 4; ++vreg) {
          for (int code = 0; code < CodesPerVreg; ++code) {
            int const source_half =
                source_half_base + 2 * vreg + code / Pack;
            auto const source = load_part(load_pi(source_half));
            int const n = int(get<0>(source));
            int const physical_kg = int(get<1>(source));
            int const k = kquant_kpack::PlaneMap<Bits, Group>::logical_k(
                physical_kg, code % Pack);
            int const source_flat = n * kTK + k;
            if (source_flat < 0 || source_flat >= TN * kTK) {
              ++m.source_holes;
              continue;
            }
            int const emitted =
                (cutlass::MixGemmEmit<Bits>::index(code, vreg) +
                 L236_ROTATE_DESTINATION) % Width;
            int const destination_raw = destination_base + emitted;
            if (destination_raw < 0 || destination_raw >= ComputeHalfs) {
              ++m.destination_holes;
              continue;
            }
            auto const destination = compute_part(compute_pi(destination_raw));
            int const got_n = int(get<0>(destination));
            int const got_k = int(get<1>(destination));
#if L236_PRINT_MAP
            if constexpr (TN == 32 && WN == 16) {
              if (n == 0)
                std::printf("L236 MAP bits=%d group=%d kg=%d slot=%d -> n=%d k=%d\n",
                            Bits, Group, physical_kg, code % Pack,
                            got_n, got_k);
            }
#endif
            m.exact += got_n == n && got_k == k;
            ++m.total;
            ++source_hits[std::size_t(source_flat)];
            int const destination_flat = got_n * kTK + got_k;
            if (destination_flat >= 0 && destination_flat < TN * kTK)
              ++destination_hits[std::size_t(destination_flat)];
            else
              ++m.destination_holes;
          }
        }
      }
    }
  }
  for (int i = 0; i < TN * kTK; ++i) {
    m.source_denominator_bad += source_hits[std::size_t(i)] != 1;
    m.destination_denominator_bad +=
        destination_hits[std::size_t(i)] != 1;
  }
  return m;
}

struct PairMetrics {
  int exact = 0;
  int total = 0;
  int holes = 0;
  int low_denominator_bad = 0;
  int high_denominator_bad = 0;
};

template <int LowBits, int HighBits, int Group, int TN, int WN>
PairMetrics prove_pairing() {
  constexpr int LowPack = 16 / LowBits;
  constexpr int HighPack = 16 / HighBits;
  constexpr int LowPhysicalK = kTK / LowPack;
  constexpr int HighPhysicalK = kTK / HighPack;
  constexpr int CubeW = TN < 64 ? TN : 64;
  constexpr int P2Div = LowBits / HighBits;
  using Low = typename ElementForBits<LowBits>::type;
  using High = typename ElementForBits<HighBits>::type;
  using Cvt = cutlass::MixGemm2Plane<LowBits, HighBits>;
  using Loader = typename MmaTypes<TN, WN>::Loader;
  using Compute = typename MmaTypes<TN, WN>::Compute;
  using LowAtom = Layout<Shape<Int<CubeW>, Int<LowPhysicalK>>,
                         Stride<_1, Int<CubeW>>>;
  using HighAtom = Layout<Shape<Int<CubeW>, Int<HighPhysicalK>>,
                          Stride<_1, Int<CubeW>>>;
  using LowStage = decltype(tile_to_shape(
      LowAtom{}, make_shape(Int<TN>{}, Int<LowPhysicalK>{}, _1{})));
  using HighStage = decltype(tile_to_shape(
      HighAtom{}, make_shape(Int<TN>{}, Int<HighPhysicalK>{}, _1{})));
  auto low_tensor = make_tensor(
      make_smem_ptr((cutlass::half_t*)nullptr), LowStage{});
  auto high_tensor = make_tensor(
      make_smem_ptr((cutlass::half_t*)nullptr), HighStage{});
  auto low_identity =
      make_identity_tensor(Shape<Int<TN>, Int<LowPhysicalK>>{});
  auto high_identity =
      make_identity_tensor(Shape<Int<TN>, Int<HighPhysicalK>>{});

  PairMetrics m{};
  std::array<int, TN * kTK> low_hits{};
  std::array<int, TN * kTK> high_hits{};
  for (int thread = 0; thread < int(size(Compute{})); ++thread) {
    auto loader = Loader{}.get_thread_slice(thread);
    auto low_frag = loader.partition_fragment_B(low_tensor(_, _, 0));
    auto high_frag = loader.partition_fragment_B(high_tensor(_, _, 0));
    auto low_part = loader.partition_B(low_identity);
    auto high_part = loader.partition_B(high_identity);
    auto low_pi = right_inverse(low_frag.layout());
    auto high_pi = right_inverse(high_frag.layout());
    constexpr int LowKBlocks = int(size<2>(decltype(low_frag.layout()){}));
    constexpr int HighKBlocks = int(size<2>(decltype(high_frag.layout()){}));
    static_assert(LowKBlocks / HighKBlocks == P2Div);

    for (int k_block = 0; k_block < LowKBlocks; ++k_block) {
      int const high_k_block = k_block / P2Div;
      auto low_cvt = recast<Low>(low_frag(_, _, k_block));
      auto high_cvt = recast<High>(high_frag(_, _, high_k_block));
      constexpr int NumIter = int(size<1>(decltype(low_cvt.layout()){}));
      constexpr int HighIter = int(size<1>(decltype(high_cvt.layout()){}));
      using HiSrc = HiPlaneSrc<HighIter, NumIter, P2Div>;
#if L236_PRINT_MAP
      if constexpr (LowBits == 4 && HighBits == 1 && TN == 32 && WN == 16) {
        if (thread == 0 && k_block == 0) {
          std::printf("L236 PAIRLAY low="); print(low_cvt.layout());
          std::printf(" high="); print(high_cvt.layout());
          std::printf(" low_frag="); print(low_frag.layout());
          std::printf(" high_frag="); print(high_frag.layout());
          std::printf(" NumIter=%d HighIter=%d P2Div=%d LowKB=%d HighKB=%d\n",
                      NumIter, HighIter, P2Div, LowKBlocks, HighKBlocks);
        }
      }
#endif
      int const low_step_base = int(low_frag.layout()(0, 0, k_block));
      int const high_step_base =
          int(high_frag.layout()(0, 0, high_k_block));
      for (int ii = 0; ii < NumIter; ++ii) {
        int const low_n_base = int(low_cvt.layout()(0, ii));
        int const high_slot = HiSrc::slot(ii);
        int const high_n_base = int(high_cvt.layout()(0, high_slot));
        int const high_base =
            HiSrc::base(ii, k_block) + L236_SHIFT_HIGH_SOURCE;
        for (int vreg = 0; vreg < 4; ++vreg) {
          for (int t = 0; t < Cvt::kPairs; ++t) {
            for (int half = 0; half < 2; ++half) {
              int const low_code = Cvt::lo_code(t, half);
              int const low_half = low_step_base + low_n_base / LowPack +
                                   2 * vreg + low_code / LowPack;
              auto const low_coord = low_part(low_pi(low_half));
              int const low_n = int(get<0>(low_coord));
              int const low_k =
                  kquant_kpack::PlaneMap<LowBits, Group>::logical_k(
                      int(get<1>(low_coord)), low_code % LowPack);

              int const high_code = Cvt::hi_code(t, vreg, half);
              int const high_vreg = high_base + Cvt::hi_vreg(vreg);
              int const high_half = high_step_base +
                                    high_n_base / HighPack +
                                    2 * high_vreg + high_code / HighPack;
              constexpr int HighHalfs =
                  int(size(decltype(high_frag.layout()){}));
              if (high_half < 0 || high_half >= HighHalfs) {
                ++m.holes;
                continue;
              }
              auto const high_coord = high_part(high_pi(high_half));
              using HighMap = kquant_kpack::HighPlaneMap<
                  LowBits, HighBits, Group>;
              int const high_n = HighMap::logical_n(
                  int(get<0>(high_coord)), int(get<1>(high_coord)),
                  high_code % HighPack);
              int const high_k = [&] {
                if constexpr (LowBits == 4 && HighBits == 1 && Group == 32)
                  return HighMap::logical_k(
                      int(get<0>(high_coord)), int(get<1>(high_coord)),
                      high_code % HighPack);
                else
                  return HighMap::logical_k(
                      int(get<1>(high_coord)), high_code % HighPack);
              }();
#if L236_PRINT_MAP
              if constexpr (TN == 32 && WN == 16) {
                if (low_n == 0)
                  std::printf(
                      "L236 PAIRMAP low=%d high=%d kb=%d ii=%d v=%d t=%d h=%d "
                      "base=%d hv=%d lc=%d hc=%d -> lown=%d lowk=%d "
                      "physn=%d physkg=%d physslot=%d highn=%d highk=%d\n",
                      LowBits, HighBits, k_block, ii, vreg, t, half,
                      high_base, high_vreg, low_code, high_code,
                      low_n, low_k, int(get<0>(high_coord)),
                      int(get<1>(high_coord)), high_code % HighPack,
                      high_n, high_k);
              }
#endif
              m.exact += low_n == high_n && low_k == high_k;
              ++m.total;
              int const low_flat = low_n * kTK + low_k;
              int const high_flat = high_n * kTK + high_k;
              if (low_flat >= 0 && low_flat < TN * kTK)
                ++low_hits[std::size_t(low_flat)];
              else
                ++m.holes;
              if (high_flat >= 0 && high_flat < TN * kTK)
                ++high_hits[std::size_t(high_flat)];
              else
                ++m.holes;
            }
          }
        }
      }
    }
  }
  for (int i = 0; i < TN * kTK; ++i) {
    m.low_denominator_bad += low_hits[std::size_t(i)] != 1;
    m.high_denominator_bad += high_hits[std::size_t(i)] != 1;
  }
  return m;
}

template <int Bits, int Group, int TN, int WN>
bool report_plane(char const* name) {
  auto const m = prove_plane<Bits, Group, TN, WN>();
  bool const ok = m.exact == m.total && m.total == TN * kTK &&
                  m.source_holes == 0 && m.destination_holes == 0 &&
                  m.source_denominator_bad == 0 &&
                  m.destination_denominator_bad == 0;
  std::printf(
      "L236 PLANE format=%s bits=%d TN=%d WN=%d exact=%d/%d "
      "source_holes=%d destination_holes=%d source_denominator_bad=%d "
      "destination_denominator_bad=%d result=%s\n",
      name, Bits, TN, WN, m.exact, m.total, m.source_holes,
      m.destination_holes, m.source_denominator_bad,
      m.destination_denominator_bad, ok ? "PASS" : "FAIL");
  return ok;
}

template <int LowBits, int HighBits, int Group, int TN, int WN>
bool report_pair(char const* name) {
  auto const m = prove_pairing<LowBits, HighBits, Group, TN, WN>();
  bool const ok = m.exact == m.total && m.total == TN * kTK &&
                  m.holes == 0 && m.low_denominator_bad == 0 &&
                  m.high_denominator_bad == 0;
  std::printf(
      "L236 PAIR format=%s low=%d high=%d TN=%d WN=%d exact=%d/%d "
      "holes=%d low_denominator_bad=%d high_denominator_bad=%d result=%s\n",
      name, LowBits, HighBits, TN, WN, m.exact, m.total, m.holes,
      m.low_denominator_bad, m.high_denominator_bad,
      ok ? "PASS" : "FAIL");
  return ok;
}

template <int Bits, int Group>
bool plane_geometries(char const* name) {
  bool ok = true;
  ok &= report_plane<Bits, Group, 32, 16>(name);
  ok &= report_plane<Bits, Group, 32, 32>(name);
  ok &= report_plane<Bits, Group, 64, 16>(name);
  ok &= report_plane<Bits, Group, 64, 32>(name);
  ok &= report_plane<Bits, Group, 64, 64>(name);
  ok &= report_plane<Bits, Group, 128, 16>(name);
  ok &= report_plane<Bits, Group, 128, 32>(name);
  ok &= report_plane<Bits, Group, 128, 64>(name);
  ok &= report_plane<Bits, Group, 256, 16>(name);
  ok &= report_plane<Bits, Group, 256, 32>(name);
  ok &= report_plane<Bits, Group, 256, 64>(name);
  return ok;
}

template <int LowBits, int HighBits, int Group>
bool pair_geometries(char const* name) {
  bool ok = true;
  ok &= report_pair<LowBits, HighBits, Group, 32, 16>(name);
  ok &= report_pair<LowBits, HighBits, Group, 32, 32>(name);
  ok &= report_pair<LowBits, HighBits, Group, 64, 16>(name);
  ok &= report_pair<LowBits, HighBits, Group, 64, 32>(name);
  ok &= report_pair<LowBits, HighBits, Group, 64, 64>(name);
  ok &= report_pair<LowBits, HighBits, Group, 128, 16>(name);
  ok &= report_pair<LowBits, HighBits, Group, 128, 32>(name);
  ok &= report_pair<LowBits, HighBits, Group, 128, 64>(name);
  ok &= report_pair<LowBits, HighBits, Group, 256, 16>(name);
  ok &= report_pair<LowBits, HighBits, Group, 256, 32>(name);
  ok &= report_pair<LowBits, HighBits, Group, 256, 64>(name);
  return ok;
}

}  // namespace

int main() {
#if L236_PAIR_DIAG
  bool const ok = report_pair<4, 1, 32, 32, 16>("Q5");
  return ok ? 0 : 1;
#else
  bool ok = true;
  ok &= plane_geometries<2, 16>("Q2/Q3-low");
  ok &= plane_geometries<1, 16>("Q3-high");
  ok &= plane_geometries<4, 32>("Q5-low");
  ok &= plane_geometries<4, 16>("Q6-low");
  ok &= plane_geometries<2, 16>("Q6-high");
  ok &= pair_geometries<2, 1, 16>("Q3");
  ok &= pair_geometries<4, 1, 32>("Q5");
  ok &= pair_geometries<4, 2, 16>("Q6");
  std::printf("L236 KQUANT_KPACK_PRODUCTION_FRAGMENT %s planes=55 pairs=33\n",
              ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
#endif
}
