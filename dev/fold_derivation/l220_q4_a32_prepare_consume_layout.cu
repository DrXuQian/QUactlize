// L220 -- compose the exact Q4_K/A32 prepare/consume register coordinates.
//
// The device factorial proved that moving consume ahead of prepare closes the
// error.  That is a bisection, not permission to discard software pipelining:
// this oracle asks which CuTe destination written by prepare(next d0) overlaps
// a coordinate read by consume(current d3).

#define main l123_embedded_main
#include "l123_warp_nk_topology.cu"
#undef main

#include <array>
#include <cstdio>

#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_a_schedule.hpp"

namespace {

using Case = Q4A32Case;
using P = FoldPair<Case, 32, 1>;
using Mma = typename P::Mma;
using Schedule = cutlass::gemm::collective::detail::MixedARegisterSchedule<8, 1, 4>;
using MatchingSchedule = cutlass::gemm::collective::detail::MixedARegisterSchedule<8, 4, 4>;

// Exact builder derivation for the ordinary fp16 A operand at TM64/TK128:
// 128 K elements = two 64-half (128-byte) swizzle deliveries.
using ALoadOp = PPU0010_TSM_LD_SWZL<
    cutlass::half_t, 64, 64, true, false, 2>;
using ALoadAtom = Copy_Atom<ALoadOp, cutlass::half_t>;

template <int K, class Tensor>
void mark_mode2(Tensor const& tensor, std::array<int, 4096>& out) {
  for (int i = 0; i < int(size<0>(tensor)); ++i)
    for (int j = 0; j < int(size<1>(tensor)); ++j) {
      int const offset = int(tensor.layout()(make_coord(i, j, Int<K>{})));
      if (offset >= 0 && offset < int(out.size())) ++out[offset];
    }
}

int overlap(std::array<int, 4096> const& lhs,
            std::array<int, 4096> const& rhs) {
  int out = 0;
  for (int i = 0; i < int(lhs.size()); ++i) out += lhs[i] && rhs[i];
  return out;
}

template <class Tensor>
void show(char const* name, Tensor const& tensor) {
  std::printf("L220 %s size=%d cosize=%d layout=", name,
              int(size(tensor)), int(cosize(tensor.layout())));
  print(tensor.layout());
  std::printf("\n");
}

}  // namespace

int main() {
  Mma mma;
  auto thr = mma.get_thread_slice(0);

  auto sA = make_tensor(
      make_smem_ptr((cutlass::half_t*)nullptr),
      make_layout(Shape<Int<Case::tm>, Int<Case::tk>>{},
                  Stride<Int<Case::tk>, _1>{}));
  auto tCrA = thr.partition_fragment_A(sA);
  auto a_copy = make_tiled_copy_A(ALoadAtom{}, mma);
  auto a_view = a_copy.get_thread_slice(0).retile_D(tCrA);

  constexpr int Deliveries = 4;
  constexpr int AtomsPerDelivery = 2;
  static_assert(decltype(size<2>(tCrA))::value ==
                Deliveries * AtomsPerDelivery);

  std::array<int, 4096> a_prepare_d0{};
  std::array<int, 4096> a_consume_d3{};
  // This is the production call shape: prepare indexes the A copy view with
  // the B delivery coordinate directly.
  mark_mode2<0>(a_view, a_prepare_d0);
  mark_mode2<6>(tCrA, a_consume_d3);
  mark_mode2<7>(tCrA, a_consume_d3);

  auto sB = make_tensor(
      make_smem_ptr((cutlass::half_t*)nullptr),
      make_layout(Shape<Int<Case::tn>, Int<Case::tk>>{},
                  Stride<Int<Case::tk>, _1>{}));
  auto tCrB = thr.partition_fragment_B(sB);
  using Scatter = cutlass::MixGemmArtifactScatter<4, 2, Deliveries>;
  std::array<int, 4096> b_prepare_d0{};
  std::array<int, 4096> b_consume_d3{};
  for (int e = 0; e < Scatter::VEC; ++e)
    ++b_prepare_d0[Scatter::flat(e, 0)];
  mark_mode2<6>(tCrB, b_consume_d3);
  mark_mode2<7>(tCrB, b_consume_d3);

  show("A-fragment", tCrA);
  show("A-copy-view", a_view);
  show("B-fragment", tCrB);
  int const a_overlap = overlap(a_prepare_d0, a_consume_d3);
  int const b_overlap = overlap(b_prepare_d0, b_consume_d3);
  constexpr bool schedule_exact =
      Schedule::ABlocks == 1 && Schedule::BBlocks == 4 &&
      Schedule::AAtomsPerCopy == 8 && Schedule::BAtomsPerCopy == 2 &&
      Schedule::template prepare_loads<0, 0>() &&
      Schedule::template delay_prepare<0, false>() &&
      !Schedule::template delay_prepare<0, true>() &&
      Schedule::template load_after_consume<3>() &&
      Schedule::loads_per_k_tile() == 1;
  constexpr bool matching_unchanged =
      MatchingSchedule::template prepare_loads<0, 0>() &&
      MatchingSchedule::template prepare_loads<1, 1>() &&
      MatchingSchedule::template prepare_loads<2, 2>() &&
      MatchingSchedule::template prepare_loads<3, 3>() &&
      !MatchingSchedule::template delay_prepare<0, false>() &&
      !MatchingSchedule::template load_after_consume<3>() &&
      MatchingSchedule::loads_per_k_tile() == 4;
  std::printf(
      "L220 wrap d3->d0 A_copy_K=%d A_prepare=%d A_consume=%d "
      "A_overlap=%d B_prepare=%d B_consume=%d B_overlap=%d\n",
      int(size<2>(a_view)), int(size(a_view(_, _, Int<0>{}))),
      int(size(tCrA(_, _, Int<6>{})) + size(tCrA(_, _, Int<7>{}))),
      a_overlap, Scatter::VEC,
      int(size(tCrB(_, _, Int<6>{})) + size(tCrB(_, _, Int<7>{}))),
      b_overlap);
  std::printf(
      "L220 schedule A_blocks=%d B_blocks=%d A_atoms=%d B_atoms=%d "
      "loads/tile=%d delay-wrap=%d post-consume=%d matching-unchanged=%d\n",
      Schedule::ABlocks, Schedule::BBlocks, Schedule::AAtomsPerCopy,
      Schedule::BAtomsPerCopy, Schedule::loads_per_k_tile(),
      int(Schedule::template delay_prepare<0, false>()),
      int(Schedule::template load_after_consume<3>()),
      int(matching_unchanged));

  // A nonzero overlap is a diagnosis, so the oracle itself succeeds when it
  // finds and uniquely separates that source-level collision from B.
  bool const diagnosed = a_overlap == 16 && b_overlap == 0 &&
                         schedule_exact && matching_unchanged;
  std::printf("L220 verdict=%s\n",
              diagnosed ? "A_PREPARE_OVERWRITES_LIVE_D3" : "UNRESOLVED");
  return diagnosed ? 0 : 1;
}
