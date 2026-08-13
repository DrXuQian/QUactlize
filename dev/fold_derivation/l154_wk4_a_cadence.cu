// L154 -- causal oracle for the classic-aligned WK4 A/B copy cadence.
//
// The production B shadow copy feeds two MMA K atoms from one register-copy
// block.  A has one copy block per atom.  Driving both operands with B's block
// index therefore leaves A's upper 64 K values unfilled.  This oracle derives
// the three extents from the exact CuTe types, then independently reconstructs
// the exact fixture's first eight outputs.  The old cadence must reproduce the
// eight values observed on ppu001; the fixed cadence must reproduce golden.

#include <array>
#include <cstdio>

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cutlass/numeric_types.h"

namespace {
using namespace cute;

using Compute = TiledMMA<
    MMA_Atom<PPU0010_16x16x16_F32F16F16F32_TN>,
    Layout<Shape<_1, _2, _4>>, Tile<_16, _32, _64>>;
using Shadow = TiledMMA<
    MMA_Atom<PPU0010_16x16x32_S32S8S8S32_TN>,
    Layout<Shape<_1, _2, _2>>, Tile<_16, _32, _64>>;

using SmemA = decltype(tile_to_shape(
    Layout<Shape<_8, _64>, Stride<_64, _1>>{},
    make_shape(_16{}, _128{}, _4{})));
using SmemB = decltype(tile_to_shape(
    Layout<Shape<_8, _64>, Stride<_64, _1>>{},
    make_shape(_128{}, _128{}, _4{})));
using AOp = PPU0010_TSM_LD_SWZL<cutlass::half_t, 16, 64, true, false, 2>;
using BOp = PPU0010_TSM_LD_SWZL<int8_t, 128, 32, true, false, 2>;

auto a_view() {
  auto s = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr), SmemA{})(_,_,0);
  auto frag = Compute{}.get_thread_slice(0).partition_fragment_A(s);
  return make_tiled_copy_A(Copy_Atom<AOp, cutlass::half_t>{}, Compute{})
      .get_thread_slice(0).retile_D(frag);
}

auto b_view() {
  auto s4 = make_tensor(make_smem_ptr((cutlass::int4b_t*)nullptr), SmemB{})(_,_,0);
  auto s8 = recast<int8_t>(s4);
  auto frag = Shadow{}.get_thread_slice(0).partition_fragment_B(s8);
  return make_tiled_copy_B(Copy_Atom<BOp, int8_t>{}, Shadow{})
      .get_thread_slice(0).retile_D(frag);
}

using AView = decltype(a_view());
using BView = decltype(b_view());
using BFrag = decltype(Compute{}.get_thread_slice(0).partition_fragment_B(
    make_tensor(make_smem_ptr((cutlass::half_t*)nullptr), SmemB{})(_,_,0)));

constexpr int kABlocks = int(size<2>(typename AView::layout_type{}));
constexpr int kBBlocks = int(size<2>(typename BView::layout_type{}));
constexpr int kBAtoms = int(size<2>(typename BFrag::layout_type{}));
constexpr int kAtomsPerBCopy = kBAtoms / kBBlocks;
static_assert(kABlocks == 2 && kBBlocks == 1 && kAtomsPerBCopy == 2);
static_assert(kABlocks == kBBlocks * kAtomsPerBCopy,
              "every B copy block must load one A block per consumed atom");

constexpr std::array<int, 8> kDeviceGot =
    {167, 122, 141, 144, 155, 166, 137, 148};
constexpr std::array<int, 8> kGolden =
    {277, 328, 283, 286, 321, 292, 303, 306};

struct Sums {
  std::array<int, 8> lower64{};
  std::array<int, 8> all{};
};

constexpr Sums exact_fixture_sums() {
  Sums out{};
  for (int kg = 0; kg < 32; ++kg) {
    int const within = (29 * kg) % 128;
    int const k = kg * 128 + within;
    int const a = ((k >> 3) & 1) ? -1 : 1;
    for (int n = 0; n < 8; ++n) {
      int const code = (5 * k + 3 * n) & 7;
      int const q = ((k >> 3) & 1) ? (-8 + code) : code;
      int const scale = std::array<int, 3>{1, 2, 4}[(5 * kg + 3 * n) % 3];
      int const contribution = a * q * scale;
      out.all[n] += contribution;
      if (within < 64) out.lower64[n] += contribution;
    }
  }
  return out;
}

constexpr auto kSums = exact_fixture_sums();
constexpr bool same(std::array<int, 8> const& a,
                    std::array<int, 8> const& b) {
  for (int i = 0; i < 8; ++i)
    if (a[i] != b[i]) return false;
  return true;
}
static_assert(same(kSums.lower64, kDeviceGot),
              "old A cadence must exactly explain the ppu001 wrong values");
static_assert(same(kSums.all, kGolden),
              "both A subblocks must reproduce the exact-fixture golden");

} // namespace

int main() {
  std::printf("L154 cadence A_K_BLOCKS=%d B_K_BLOCKS=%d K_ATOM_PER_COPY=%d\n",
              kABlocks, kBBlocks, kAtomsPerBCopy);
  std::printf("L154 old-lower64=");
  for (int x : kSums.lower64) std::printf("%d,", x);
  std::printf(" device=EXACT\nL154 fixed-all=");
  for (int x : kSums.all) std::printf("%d,", x);
  std::printf(" golden=EXACT\n");
#if defined(L154_OLD_A_CADENCE)
  std::puts("L154 EXPECTED-RED: one B block loaded only A block zero");
  return 1;
#else
  std::puts("L154 PASS: each B copy loads both A atom subblocks");
  return 0;
#endif
}
