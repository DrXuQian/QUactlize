// L224 -- exact A-register prepare/consume lifetime for the failing
// Q4_K/A64 TM8/TN64/TK256/WM8/WN16/Stages2 packed-A specialization.
//
// Keep this a host CuTe oracle. The full shipping collective forces PPU
// device bodies through nvcc and cannot be iterated on the host. L186 binds
// the shipping specialization to the exact Mma/SmemCopyAtom aliases repeated
// below; this file composes those aliases and exhaustively compares the
// physical register offsets written by prepare(next) with those consumed by
// the current MMA delivery.

#include <array>
#include <cstdio>

#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cute/tensor.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_a_schedule.hpp"

namespace {
using namespace cute;

using Atom = PPU0010_8x16x16_F32F16F16F32_TN;
using Mma = TiledMMA<MMA_Atom<Atom>,
    Layout<Shape<_1, _4, _1>>, Tile<_8, _64, _16>>;
using SmemAtomA = Layout<Shape<_8, _64>, Stride<_64, _1>>;
using SmemStageA = decltype(tile_to_shape(
    SmemAtomA{}, make_shape(_8{}, _256{})));
using SmemLayoutA = decltype(append(
    SmemStageA{}, make_layout(_2{}, Int<4096>{})));
using SmemCopyOpA = PPU0010_TSM_LD_SWZL_M8<
    cutlass::half_t, 16, 64, true, false, 4, 64, 1216>;
using SmemCopyAtomA = Copy_Atom<SmemCopyOpA, cutlass::half_t>;

// The production int4 shadow delivery has four CPY_K blocks at TK256 (see
// the collective's K_BLOCK_MAX and the existing L57/L62 derivations). A's
// exact view below has 16 blocks and its MMA fragment has 16 K atoms.
constexpr int kBBlocks = 4;

struct OutputOwner {
  int thread = -1;
  int fragment = -1;
};

template <class Tensor>
void mark_mode2(Tensor const& tensor, int k, std::array<int, 8192>& out) {
  for (int i = 0; i < int(size<0>(tensor)); ++i) {
    for (int j = 0; j < int(size<1>(tensor)); ++j) {
      int const offset = int(tensor.layout()(make_coord(i, j, k)));
      if (0 <= offset && offset < int(out.size())) {
        ++out[std::size_t(offset)];
      }
    }
  }
}

int overlap(std::array<int, 8192> const& a,
            std::array<int, 8192> const& b) {
  int result = 0;
  for (int i = 0; i < int(a.size()); ++i) {
    result += a[std::size_t(i)] != 0 && b[std::size_t(i)] != 0;
  }
  return result;
}

constexpr int q4_code(int n, int k) {
  return (((13 * n + 7 * k + 3) & 7) - 3) & 15;
}

constexpr int active_offset(int superblock) {
  return (37 * superblock + 11) & 255;
}

constexpr int active_sign(int superblock) {
  return (superblock & 1) ? -1 : 1;
}

template <class Tensor>
void show(char const* name, Tensor const& tensor) {
  std::printf("L224 %s size=%d cosize=%d layout=", name,
              int(size(tensor)), int(cosize(tensor.layout())));
  print(tensor.layout());
  std::printf("\n");
}
}  // namespace

int main() {
  Mma mma;
  auto thr = mma.get_thread_slice(0);
  auto s_a = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                         SmemLayoutA{});
  auto t_cr_a = thr.partition_fragment_A(s_a(_, _, Int<0>{}));
  auto a_copy = make_tiled_copy_A(SmemCopyAtomA{}, mma);
  auto a_view = a_copy.get_thread_slice(0).retile_D(t_cr_a);

  constexpr int MmaAtoms = decltype(size<2>(t_cr_a))::value;
  constexpr int ABlocks = decltype(size<2>(a_view))::value;
  using Schedule = cutlass::gemm::collective::detail::MixedARegisterSchedule<
      MmaAtoms, ABlocks, kBBlocks>;
  static_assert(MmaAtoms == 16 && ABlocks == 16 &&
                Schedule::AAtomsPerCopy == 1 &&
                Schedule::BAtomsPerCopy == 4,
                "L224 must remain the exact M8/TK256 register schedule");

  show("A-fragment", t_cr_a);
  show("A-copy-view", a_view);

  // Bind the observed aligned 32-output footprint to the production
  // partition_C map. At M=1 each warp owns 16 live columns, so one bad
  // 32-column band is exactly two adjacent N warps; it is not evidence for a
  // single thread, register, or producer warp by itself.
  auto c_identity = make_identity_tensor(Shape<_8, _64>{});
  std::array<OutputOwner, 64> output_owners{};
  std::array<int, 4> live_per_warp{};
  int output_duplicates = 0;
  for (int thread = 0; thread < int(size(Mma{})); ++thread) {
    auto coordinates = mma.get_thread_slice(thread).partition_C(c_identity);
    for (int fragment = 0; fragment < int(size(coordinates)); ++fragment) {
      auto mn = coordinates(fragment);
      int const m = int(get<0>(mn));
      int const n = int(get<1>(mn));
      if (m != 0) continue;
      if (n < 0 || n >= int(output_owners.size()) ||
          output_owners[std::size_t(n)].thread >= 0) {
        ++output_duplicates;
        continue;
      }
      output_owners[std::size_t(n)] = OutputOwner{thread, fragment};
      ++live_per_warp[std::size_t(thread / 32)];
    }
  }
  int output_holes = 0;
  int output_band_bad = 0;
  for (int n = 0; n < int(output_owners.size()); ++n) {
    output_holes += output_owners[std::size_t(n)].thread < 0;
    output_band_bad +=
        output_owners[std::size_t(n)].thread / 32 != n / 16;
  }
  bool const output_ownership_exact = output_holes == 0 &&
      output_duplicates == 0 && output_band_bad == 0 &&
      live_per_warp == std::array<int, 4>{{16, 16, 16, 16}};
  std::printf(
      "L224 output-ownership live_per_warp=%d,%d,%d,%d "
      "bands=N0-15:W0,N16-31:W1,N32-47:W2,N48-63:W3 "
      "aligned32=two-adjacent-N-warps holes=%d duplicates=%d band_bad=%d "
      "verdict=%s\n",
      live_per_warp[0], live_per_warp[1], live_per_warp[2],
      live_per_warp[3], output_holes, output_duplicates, output_band_bad,
      output_ownership_exact ? "EXACT" : "BAD");

  int identity_bad = 0;
  for (int a = 0; a < ABlocks; ++a) {
    std::array<int, 8192> prepared{};
    mark_mode2(a_view, a, prepared);
    int match = -1;
    for (int atom = 0; atom < MmaAtoms; ++atom) {
      std::array<int, 8192> consumed{};
      mark_mode2(t_cr_a, atom, consumed);
      if (overlap(prepared, consumed) == int(size(a_view(_, _, Int<0>{})))) {
        match = atom;
      }
    }
    identity_bad += match != a;
    std::printf("L224 block-map A_copy=%d MMA_atom=%d\n", a, match);
  }

  int max_overlap = 0;
  int total_overlap = 0;
  for (int consumed_b = 0; consumed_b < kBBlocks; ++consumed_b) {
    int const prepare_b = (consumed_b + 1) % kBBlocks;
    std::array<int, 8192> prepared{};
    std::array<int, 8192> consumed{};
    for (int a = 0; a < ABlocks; ++a) {
      int const first_b =
          (a * Schedule::AAtomsPerCopy) / Schedule::BAtomsPerCopy;
      if (first_b == prepare_b) mark_mode2(a_view, a, prepared);
    }
    for (int atom = consumed_b * Schedule::BAtomsPerCopy;
         atom < (consumed_b + 1) * Schedule::BAtomsPerCopy; ++atom) {
      mark_mode2(t_cr_a, atom, consumed);
    }
    int const current = overlap(prepared, consumed);
    max_overlap = current > max_overlap ? current : max_overlap;
    total_overlap += current;
    std::printf("L224 transition consume_b=%d prepare_b=%d overlap=%d\n",
                consumed_b, prepare_b, current);
  }
  std::printf(
      "L224 schedule mma_atoms=%d A_blocks=%d B_blocks=%d "
      "A_atoms_per_copy=%d B_atoms_per_copy=%d max_overlap=%d total=%d\n",
      MmaAtoms, ABlocks, kBBlocks, Schedule::AAtomsPerCopy,
      Schedule::BAtomsPerCopy, max_overlap, total_overlap);

  // Frozen ca01dc6 signature: S4 plane 2 spans superblocks [10,15) and one
  // captured 32-output band contained 6.0 instead of 1.0. Replacing sb13's
  // complete A tile with its predecessor is one exact explanation for that
  // value pair. It is not a classifier for the whole incident family: later
  // captures also contained deltas -4 and -6. Enumerate every complete A-tile
  // substitution below so this oracle cannot silently promote one compatible
  // signature into the root cause.
  constexpr int n = 32;
  int expected = 0;
  int stale13 = 0;
  int stale_matches = 0;
  for (int sb = 10; sb < 15; ++sb) {
    int const current = active_sign(sb) *
        q4_code(n, sb * 256 + active_offset(sb));
    int const previous = active_sign(sb - 1) *
        q4_code(n, sb * 256 + active_offset(sb - 1));
    expected += current;
    std::printf(
        "L224 plane2 n=%d sb=%d current=%d previous_tile=%d\n",
        n, sb, current, previous);
  }
  for (int replaced = 10; replaced < 15; ++replaced) {
    int candidate = expected;
    candidate -= active_sign(replaced) *
        q4_code(n, replaced * 256 + active_offset(replaced));
    candidate += active_sign(replaced - 1) *
        q4_code(n, replaced * 256 + active_offset(replaced - 1));
    stale_matches += candidate == 6;
    if (replaced == 13) stale13 = candidate;
    std::printf("L224 stale-replace sb=%d plane2=%d\n", replaced,
                candidate);
  }
  std::printf(
      "L224 incident expected=%d observed=6 stale_sb13=%d "
      "stale_matches=%d verdict=%s\n",
      expected, stale13, stale_matches,
      expected == 1 && stale13 == 6 && stale_matches == 1
          ? "ONE_SIGNATURE_ADMITS_PREVIOUS_A_TILE"
          : "INCIDENT_NOT_CLASSIFIED");

  int adjacent_minus5 = 0;
  int adjacent_plus5 = 0;
  int any_minus4 = 0;
  int any_minus6 = 0;
  for (int col = 0; col < 8; ++col) {
    for (int dst = 1; dst < 20; ++dst) {
      int const current = active_sign(dst) *
          q4_code(col, dst * 256 + active_offset(dst));
      int const previous = active_sign(dst - 1) *
          q4_code(col, dst * 256 + active_offset(dst - 1));
      adjacent_minus5 += previous - current == -5;
      adjacent_plus5 += previous - current == 5;
      for (int src = 0; src < 20; ++src) {
        if (src == dst) continue;
        int const replacement = active_sign(src) *
            q4_code(col, dst * 256 + active_offset(src));
        any_minus4 += replacement - current == -4;
        any_minus6 += replacement - current == -6;
      }
    }
  }
  bool const family_not_stale_a =
      adjacent_minus5 > 0 && adjacent_plus5 > 0 && any_minus4 > 0 &&
      any_minus6 == 0;
  std::printf(
      "L224 complete-A-substitution adjacent_delta_minus5=%d "
      "adjacent_delta_plus5=%d any_source_delta_minus4=%d "
      "any_source_delta_minus6=%d verdict=%s\n",
      adjacent_minus5, adjacent_plus5, any_minus4, any_minus6,
      family_not_stale_a ? "STALE_A_NOT_COMPLETE_FAMILY_ROOT" :
                           "FIXTURE_CLASSIFIER_CHANGED");
  std::printf("L224 identity_bad=%d verdict=%s\n", identity_bad,
              max_overlap == 0 && identity_bad == 0 && expected == 1 &&
                      stale13 == 6 && stale_matches == 1 &&
                      family_not_stale_a && output_ownership_exact
                  ? "REGISTER_LIFETIME_EXACT"
                  : "PREPARE_OR_BLOCK_MAP_BAD");
  return max_overlap == 0 && identity_bad == 0 && expected == 1 &&
                 stale13 == 6 && stale_matches == 1 && family_not_stale_a &&
                 output_ownership_exact
      ? 0
      : 1;
}
