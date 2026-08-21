// L218 -- exact Q4_K/A32 scale+zero global-to-shared copy map.
//
// L211/L216 start with correctly populated shared memory.  This witness
// covers the preceding shipping seam: make_metadata_tile -> the capped
// 16-byte cp.async tiled copy -> SmemLayoutScale(stage).  Values encode the
// global (group,n) coordinate, so a group rotation, stage alias, truncated
// thread layout, or source/destination transpose is constructive red.

#include <cstdio>
#include <vector>

#include "cute/tensor.hpp"
#include "cute/arch/copy_ppu.hpp"
#include "cute/atom/copy_traits_ppu.hpp"
#include "cutlass/numeric_types.h"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_metadata_policy.hpp"

using namespace cute;

namespace {

constexpr int N = 1024;
constexpr int K = 5120;
constexpr int GS = 32;
constexpr int TN = 64;
constexpr int TK = 128;
constexpr int Groups = TK / GS;
constexpr int Stages = 8;
constexpr int CtaThreads = 256;
constexpr int ScaleK = K / GS;
constexpr int KTiles = K / TK;

using ScaleTile = Shape<Int<TN>, Int<Groups>>;
using ScaleAtom = Layout<Shape<_8, _1>>;
using Storage = decltype(tile_to_shape(
    ScaleAtom{}, make_shape(Int<TN>{}, Int<Groups>{}, Int<Stages>{})));
using Plan = cutlass::gemm::collective::detail::ScaleCopyPlan<
    TN, Groups, CtaThreads>;
using GmemCopy = decltype(make_tiled_copy(
    Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cutlass::half_t>{},
    Layout<Shape<Int<Plan::thread_layout_h>,
                 Int<Plan::thread_layout_w>>>{},
    Layout<Shape<Int<Plan::values_per_thread>, _1>>{}));

constexpr int encoded(int global_group, int n) {
  return global_group * 10000 + n;
}

bool one_tile(int k_tile, bool transpose_negative, int physical_threads,
              int expected_hits, int& shown) {
  // Production metadata ABI is (N,scale_k) with strides (1,N).  Identity
  // values retain those logical coordinates while local_tile applies the
  // exact ScaleTile and k-tile rest decomposition.
  auto metadata = make_identity_tensor(make_shape(Int<N>{}, Int<ScaleK>{}));
  auto gS = local_tile(metadata, ScaleTile{}, make_coord(0, _));
  auto sS = make_counting_tensor(Storage{});

  int const stage = k_tile % Stages;
  std::vector<int> resident(cosize(Storage{}), -1);
  std::vector<int> hits(cosize(Storage{}), 0);
  bool ok = true;
  GmemCopy copy;
  for (int physical = 0; physical < physical_threads; ++physical) {
    int const slot = Plan::logical_slot(physical);
    auto thr = copy.get_slice(slot);
    auto src = thr.partition_S(gS);
    auto dst = thr.partition_D(sS);
    auto src_tile = src(_, _, _, k_tile);
    auto dst_stage = dst(_, _, _, stage);
    if (size(src_tile) != size(dst_stage)) return false;
    for (int i = 0; i < int(size(src_tile)); ++i) {
      auto coord = src_tile(i);
      int n = int(get<0>(coord));
      int group = int(get<1>(coord));
      int address = int(dst_stage(i));
      int value = transpose_negative ? encoded(n, group) : encoded(group, n);
      if (address < 0 || address >= int(resident.size())) return false;
      if (resident[address] >= 0 && resident[address] != value) ok = false;
      resident[address] = value;
      ++hits[address];
    }
  }

  for (int group = 0; group < Groups; ++group) {
    for (int n = 0; n < TN; ++n) {
      int const address = int(Storage{}(n, group, stage));
      int const want = encoded(k_tile * Groups + group, n);
      if (hits[address] != expected_hits || resident[address] != want) {
        ok = false;
        if (shown++ < 12) {
          std::printf(
              "L218 %s kt=%d stage=%d group=%d n=%d address=%d "
              "hits=%d got=%d want=%d\n",
              transpose_negative ? "negative-witness" : "bad",
              k_tile, stage, group, n, address, hits[address],
              resident[address], want);
        }
      }
    }
  }
  return ok;
}

bool run() {
  static_assert(Plan::thread_layout_h == 8);
  static_assert(Plan::thread_layout_w == 4);
  static_assert(Plan::thread_slots == 32);
  static_assert(Plan::values_per_thread == 8);
  static_assert(Plan::Coverage::value);
  static_assert(cosize(Storage{}) == TN * Groups * Stages);

  int shown = 0;
  bool positive = true;
  for (int kt = 0; kt < KTiles; ++kt)
    positive &= one_tile(kt, false, Plan::thread_slots, 1, shown);

  int legacy_shown = 12;
  bool const legacy_wrap_false_green =
      one_tile(1, false, CtaThreads, 1, legacy_shown);
  bool const legacy_wrap_red = !legacy_wrap_false_green;
  int negative_shown = 0;
  bool const transposed_false_green =
      one_tile(1, true, Plan::thread_slots, 1, negative_shown);
  bool const negative_red = !transposed_false_green;
  bool const ok = positive && legacy_wrap_red && negative_red;
  std::printf(
      "L218 metadata-gmem-smem tiles=%d values=%d owners=%d/%d "
      "positive=%s legacy-wrap-negative=%s transpose-negative=%s result=%s\n",
      KTiles, KTiles * TN * Groups, Plan::thread_slots, CtaThreads,
      positive ? "EXACT" : "FAIL",
      legacy_wrap_red ? "RED" : "FALSE-GREEN",
      negative_red ? "RED" : "FALSE-GREEN", ok ? "PASS" : "FAIL");
  return ok;
}

}  // namespace

int main() { return run() ? 0 : 1; }
