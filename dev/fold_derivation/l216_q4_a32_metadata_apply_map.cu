// L216 -- compose the exact Q4/A32 B-fragment slot order with the metadata
// fragment slot order.  L211 proves that metadata reaches its own fragment;
// this oracle asks the missing question: does element-wise apply_metadata()
// pair every B value with scale/zero for the same logical N?

#include <array>
#include <cstdio>
#include <vector>

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

using namespace cute;

struct L216F16Atom {};
namespace cute {
template <> struct MMA_Traits<L216F16Atom> {
  using ValTypeD = float;
  using ValTypeA = cutlass::half_t;
  using ValTypeB = cutlass::half_t;
  using ValTypeC = float;
  using Shape_MNK = Shape<_16, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _2, _2>>,
                         Stride<Stride<_32, _1>, Stride<_16, _128, _8>>>;
  using BLayout = ALayout;
  using CLayout = Layout<Shape<Shape<_4, _8>, Shape<_4, _2>>,
                         Stride<Stride<_16, _1>, Stride<_64, _8>>>;
};
}  // namespace cute

namespace {

constexpr int TM = 64, TN = 64, TK = 128, WM = 16, WN = 32;
constexpr int WOM = TM / WM, WON = TN / WN;
constexpr int Groups = 4, Stages = 8;
using Mma = TiledMMA<MMA_Atom<L216F16Atom>,
                     Layout<Shape<Int<WOM>, Int<WON>, _1>>,
                     Tile<Int<WOM * 16>, Int<WON * 16>, Int<TK>>>;
using BLayout = Layout<Shape<Int<TN>, Int<TK>>, Stride<Int<TK>, _1>>;
using ScaleAtom = Layout<Shape<_8, _1>>;
using ScaleStorage = decltype(tile_to_shape(
    ScaleAtom{}, make_shape(Int<TN>{}, Int<Groups>{}, Int<Stages>{})));
using ScaleFlat = decltype(tile_to_shape(
    ScaleAtom{}, make_shape(Int<TN>{}, _1{}, Int<Groups * Stages>{})));

bool run() {
  int bad = 0;
  int total = 0;
  int shown = 0;
  std::vector<int> metadata_n(TN * TK, -1);
  std::vector<int> hits(TN * TK, 0);
  int owner_conflicts = 0;
  auto b_identity = make_identity_tensor(make_shape(Int<TN>{}, Int<TK>{}));
  std::vector<int> scale_smem(cosize(ScaleStorage{}), -1);
  for (int stage = 0; stage < Stages; ++stage)
    for (int group = 0; group < Groups; ++group)
      for (int n = 0; n < TN; ++n)
        scale_smem[ScaleStorage{}(n, group, stage)] = n;
  auto scale_flat = make_tensor(scale_smem.data(), ScaleFlat{});
  auto scale_copy = make_tiled_copy_B(Copy_Atom<DefaultCopy, int>{}, Mma{});

  for (int thread = 0; thread < int(size(Mma{})); ++thread) {
    auto thr = Mma{}.get_thread_slice(thread);
    auto b_frag = thr.partition_fragment_B(
        make_tensor((cutlass::half_t*)nullptr, BLayout{}));
    auto s_ref = thr.partition_fragment_B(
        make_tensor((int*)nullptr, ScaleStorage{})(_, _, 0));
    auto s_frag = make_tensor<int>(make_layout_like(s_ref.layout()));
    auto b_logical = thr.partition_B(b_identity);
    auto scale_thr = scale_copy.get_thread_slice(thread);
    auto scale_src = scale_thr.partition_S(scale_flat);
    auto scale_dst = scale_thr.retile_D(s_frag);
    clear(s_frag);
    copy(scale_copy, scale_src(_, _, 0, 0), scale_dst(_, _, 0));

    // apply_metadata sees one MMA-K atom at a time and zips that rank-2
    // slice linearly with the whole rank-3 metadata fragment at K slot zero.
    for (int atom = 0; atom < int(size<2>(b_frag)); ++atom) {
      auto b_slice = b_frag(_, _, atom);
      auto b_coords = b_logical(_, _, atom);
      auto s_slice = s_frag(_, _, 0);
      auto b_inv = right_inverse(b_slice.layout());
      if (thread == 0 && atom == 0) {
        std::printf("L216 b_slice_layout="); print(b_slice.layout());
        std::printf(" s_slice_layout="); print(s_slice.layout());
        std::printf("\nL216 b_inv="); print(b_inv);
        std::printf("\n");
      }
      if (size(b_slice) != size(s_slice)) {
        std::printf("L216 size mismatch thread=%d atom=%d B=%d S=%d\n",
                    thread, atom, int(size(b_slice)), int(size(s_slice)));
        return false;
      }
      for (int i = 0; i < int(size(b_slice)); ++i) {
        auto b_nk = b_coords(b_inv(i));
        int const bn = int(get<0>(b_nk));
        int const sn = int(s_slice(i));
        int const bk = int(get<1>(b_nk));
        int const key = bn * TK + bk;
        if (metadata_n[key] >= 0 && metadata_n[key] != sn) {
          ++owner_conflicts;
        }
        metadata_n[key] = sn;
        ++hits[key];
        ++total;
        if (bn != sn) {
          ++bad;
          if (shown++ < 16) {
            std::printf(
                "L216 bad thread=%d atom=%d slot=%d b_n=%d s_n=%d b_k=%d\n",
                thread, atom, i, bn, sn, bk);
          }
        }
      }
    }
  }
  int uncovered = 0;
  for (int i = 0; i < TN * TK; ++i) uncovered += hits[i] == 0;
  std::printf(
      "L216 metadata-apply same-N=%d/%d bad=%d map_uncovered=%d "
      "owner_conflicts=%d\n",
      total - bad, total, bad, uncovered, owner_conflicts);

  // Re-evaluate the exact device fixture through the composed map.  Codes
  // stay at their intended (n,k); only scale/zero N is replaced by the slot
  // actually zipped to that code.  Group selection remains k/32.  A correct
  // composition must reproduce the golden output exactly.
  constexpr int M = 64, N = 1024, K = 5120, GS = 32;
  std::array<int, 8> active{};
  for (int s = 0; s < 8; ++s) {
    int const begin = s * K / 8;
    int const span = K / 8;
    active[s] = begin + ((37 * s + 11) % span);
  }
  auto decoded = [] (int n, int k) {
    return ((13 * n + 7 * k + 3) % 15) - 7;
  };
  auto scale = [] (int n, int g) {
    return 1 << ((17 * g + 29 * n + 1) % 3);
  };
  auto zero = [] (int n, int g) {
    return ((11 * g + 7 * n) % 3 - 1) * 3;
  };
  std::uint64_t signature_bad = 0, planted_bad = 0;
  float first_want = 0.f, first_got = 0.f;
  bool have_first = false;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float want = 0.f, got = 0.f, planted = 0.f;
      int const n_base = n - n % TN;
      for (int s = 0; s < 8; ++s) {
        int const k = active[s];
        int const g = k / GS;
        float const a = ((m + s) & 1) ? -0.5f : 0.5f;
        int const code = decoded(n, k);
        want += a * float(scale(n, g) * code + zero(n, g));
        int const local_source_n = metadata_n[(n % TN) * TK + (k % TK)];
        int const source_n = n_base + local_source_n;
        got += a * float(scale(source_n, g) * code + zero(source_n, g));
        // Constructive negative: a one-column metadata rotation must be
        // detected by the same exact fixture and denominator.
        int const planted_n = n_base + (local_source_n + 1) % TN;
        planted += a * float(scale(planted_n, g) * code + zero(planted_n, g));
      }
      if (want != got) {
        ++signature_bad;
        if (!have_first) {
          first_want = want;
          first_got = got;
          have_first = true;
        }
      }
      planted_bad += planted != want;
    }
  }
  std::printf(
      "L216 exact-fixture predicted_bad=%llu first=%g->%g "
      "rotate-N-negative=%llu/%d\n",
      static_cast<unsigned long long>(signature_bad),
      double(first_want), double(first_got),
      static_cast<unsigned long long>(planted_bad), M * N);

  bool const ok = bad == 0 && uncovered == 0 && owner_conflicts == 0 &&
                  signature_bad == 0 && planted_bad != 0;
  std::printf("L216 result=%s\n", ok ? "PASS" : "FAIL");
  return ok;
}

}  // namespace

int main() { return run() ? 0 : 1; }
