// AT A FIXED ARTIFACT FOLD, DOES THE B-FRAGMENT (n,k) MAP AT A LARGE TacticTileK EXTEND THE SMALL ONE?
//
// THE DECISION. The proposal is to make the SMALLEST TileK's folded layout canonical and let every larger TileK
// read the same bytes -- one weight file, which is what the M-routes need since GEMV / split-affine / dense all
// read the same tensor. The algebra is unconditional (frag_big.layout() o pi_small is a composition of cute
// layouts, and cute is closed under composition). What is NOT unconditional is whether the big map EXTENDS the
// small one along K, or reorders it.
//
// WHAT THE FIRST VERSION OF THIS FILE GOT WRONG, because it is the whole reason the answer looked easy. It built
//     TiledMMA<MMA_Atom<...>, Layout<Shape<WarpOnM, WarpOnN, _1>>>
// which is get_tiled_mma's PermutionK_ = void branch. The real builder takes the OTHER branch
// (ppu_mma_builder.inl:647) and supplies Tile<WOM*16, WON*16, MmaPermK>, and the offline generator does the same
// (xplane_offline.hpp:92). So it compared two maps that were both missing the permutation, got YES everywhere,
// and proved only that the ATOM repeats -- not that the folded consumer mapping does. codex caught it.
//
// AND THE PERMUTATION IS FOLD-DEPENDENT, which is exactly why omitting it hid the question:
//     MixGemmMmaPermK<Bits, BlockK, FoldF>::value = (FoldF > 1) ? BlockK : (32 * 8 / Bits)
// Unfolded it is a constant independent of TileK. FOLDED IT IS TileK ITSELF -- so at a fixed artifact fold the
// permutation at TileK=64 and TileK=256 are different numbers, and whether the maps still line up is a real
// question rather than a formality.
//
// SO THE COMPARISON HERE FIXES (Bits, F) AT THE ARTIFACT'S VALUES AND VARIES ONLY TacticTileK. That is the
// actual proposal: one artifact folded for the smallest TileK, consumed at several.
//
//   nvcc -std=c++17 -arch=sm_80 --expt-relaxed-constexpr -w -I<stub_inc> -I<actlize/include> \
//        -x cu dev/fold_derivation/tilek_fragment_map.cu -o /tmp/tkmap && /tmp/tkmap
//
// TWO TRAPS (memory/ppu-w2a16-aiu-cube-width-bug): do NOT define __HGGCCC__ (CUTLASS_DEVICE becomes
// __host__ __device__ and the PPU asm bodies host-compile), do NOT include collective_builder.hpp.
#include <cstdio>
#include <utility>
#include <vector>

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"

using namespace cute;

// EXACTLY xplane_offline.hpp:86-100's construction. Reproduced rather than included because that header pulls
// the offline generators with it, and MmaPermK comes from the SHARED rule -- its own comment records that
// restating it "silently broke a folded plane: at Block_K=64 with F=2 the composition covered 4160 of 8192
// elements".
template <int Bits, int F, int TM, int TN, int TK, int WM, int WN>
struct OfflineMma {
  static constexpr int InstM = 16, InstN = 16;
  static constexpr int warpM = (WM > InstM) ? WM : InstM, warpN = (WN > InstN) ? WN : InstN;
  static constexpr int WOM = TM / warpM, WON = TN / warpN;
  static constexpr int MmaPermK = cutlass::MixGemmMmaPermK<Bits, TK, F>::value;
  using type = TiledMMA<MMA_Atom<PPU0010_16x16x16_F32F16F16F32_TN>,
                        Layout<Shape<Int<WOM>, Int<WON>, _1>>,
                        Tile<Int<WOM * 16>, Int<WON * 16>, Int<MmaPermK>>>;
};

// ONE gmem TENSOR, PARTITIONED TWO WAYS. This is the whole point and the previous draft got it wrong: it built a
// separate (TN, TK) tensor per TileK, so the two consumers were reading DIFFERENT tensors -- different shape,
// different N-stride -- and any agreement between them said nothing about sharing one artifact. The proposal is
// one weight file; the probe must be one tensor.
//
// So: a single (TN, KFULL) identity tensor stands for the artifact. The TileK=64 consumer takes the k-slice
// [koff, koff+64) of it, the TileK=256 consumer takes the whole thing, and because both report coordinates in
// the SAME frame there is no shift arithmetic to get wrong -- the k values are directly comparable.
template <int Bits, int F, int TM, int TN, int TK, int WM, int WN, int KFULL>
std::vector<std::pair<int,int>> frag_map(int tid, int koff) {
  typename OfflineMma<Bits, F, TM, TN, TK, WM, WN>::type mma;
  auto artifact = make_identity_tensor(make_shape(Int<TN>{}, Int<KFULL>{}));
  // The k-tile this consumer sees, cut out of the one artifact. An identity tensor's VALUES are its absolute
  // coordinates and local_tile preserves them, so the slice already reports the artifact's frame -- an earlier
  // draft added domain_offset on top and double-counted the offset, which showed up immediately as a k beyond
  // the artifact's own extent. Reading the number rather than the verdict is what caught it.
  auto tile = local_tile(artifact, make_shape(Int<TN>{}, Int<TK>{}), make_coord(0, koff / TK));
  auto part = mma.get_thread_slice(tid).partition_B(tile);
  std::vector<std::pair<int,int>> out;
  for (int i = 0; i < int(size(part)); ++i) {
    auto c = part(i);
    out.emplace_back(int(get<0>(c)), int(get<1>(c)));
  }
  return out;
}

// SAME COORDINATES, SAME ORDER. Both sides now report (n,k) in the ARTIFACT's frame, so this is a direct
// equality -- no shift arithmetic, which removes the last place a wrong assumption could hide. `small` is the
// concatenation of what the consecutive small-TileK tiles read; `big` is what one large-TileK tile reads.
static bool same(std::vector<std::pair<int,int>> const& small, std::vector<std::pair<int,int>> const& big,
                 char const* tag) {
  if (small.size() != big.size()) {
    std::printf("    %-26s NO  %zu slots vs %zu\n", tag, small.size(), big.size());
    return false;
  }
  for (size_t i = 0; i < small.size(); ++i)
    if (small[i] != big[i]) {
      std::printf("    %-26s NO  slot %zu: small reads (n%d,k%d), large reads (n%d,k%d)\n",
                  tag, i, small[i].first, small[i].second, big[i].first, big[i].second);
      return false;
    }
  std::printf("    %-26s YES all %zu slots identical\n", tag, small.size());
  return true;
}

// One artifact: (Bits, F) fixed at what the SMALLEST TileK forces. Read at 64 (r tiles) and at 64*r (one tile).
template <int Bits, int F, int TM, int TN, int WM, int WN, int KFULL>
std::vector<std::pair<int,int>> small_tiles(int tid) {
  std::vector<std::pair<int,int>> out;
  for (int koff = 0; koff < KFULL; koff += 64) {
    auto one = frag_map<Bits, F, TM, TN, 64, WM, WN, KFULL>(tid, koff);
    out.insert(out.end(), one.begin(), one.end());
  }
  return out;
}

template <int Bits, int F, int TM, int TN, int WM, int WN>
void artifact(char const* tag) {
  std::printf("\n  %s   PermK: TK64=%d TK128=%d TK256=%d\n", tag,
              cutlass::MixGemmMmaPermK<Bits, 64, F>::value,
              cutlass::MixGemmMmaPermK<Bits, 128, F>::value,
              cutlass::MixGemmMmaPermK<Bits, 256, F>::value);
  same(small_tiles<Bits, F, TM, TN, WM, WN, 128>(0),
       frag_map<Bits, F, TM, TN, 128, WM, WN, 128>(0, 0), "2x TK64  vs  TK128");
  same(small_tiles<Bits, F, TM, TN, WM, WN, 256>(0),
       frag_map<Bits, F, TM, TN, 256, WM, WN, 256>(0, 0), "4x TK64  vs  TK256");
  same(small_tiles<Bits, F, TM, TN, WM, WN, 256>(33),
       frag_map<Bits, F, TM, TN, 256, WM, WN, 256>(33, 0), "4x TK64  vs  TK256 (t33)");
}

// MUST FAIL. The previous version of this file said YES everywhere and was wrong; a check that has only ever
// passed has not been shown to be a check.
static int controls() {
  std::printf("negative controls -- both must say NO:\n");
  int bad = 0;
  auto s = small_tiles<4, 1, 64, 64, 64, 32, 256>(0);
  // (1) A DIFFERENT THREAD's large-tile map. Same shapes, different slots, so a comparison that only checked
  //     sizes would pass this.
  if (same(s, frag_map<4, 1, 64, 64, 256, 64, 32, 256>(1, 0), "vs another thread")) ++bad;
  // (2) A DIFFERENT WARP SHAPE partitions N differently.
  if (same(s, frag_map<4, 1, 64, 64, 256, 64, 16, 256>(0, 0), "vs different warp shape")) ++bad;
  std::printf(bad ? "  *** A CONTROL PASSED: the comparison is blind, every YES below is void ***\n"
                  : "  both fired: a YES below carries information\n");
  return bad;
}

int main() {
  std::printf("Fixed ARTIFACT (Bits, F); only TacticTileK varies. YES on every row means one weight file serves\n"
              "every TileK, and the remaining blocker is that fold is DERIVED from (bits, TileK) rather than\n"
              "carried (ppu_dense_layout.cu:192).\n\n");
  int const blind = controls();

  std::printf("\nint4, artifact folded at TileK=64 -> F=1 (run is already 32B, nothing to fold)\n");
  artifact<4, 1, 64, 64, 64, 32>("(64,64) w64x32  [A0/A1 winner]");
  artifact<4, 1, 16, 32, 16, 16>("(16,32) w16x16  [D4 decode band]");

  std::printf("\nint2, artifact folded at TileK=64 -> F=2   <- the case the atom-only probe could not see\n");
  artifact<2, 2, 64, 64, 64, 32>("(64,64) w64x32");
  artifact<2, 2, 64, 128, 64, 64>("(64,128) w64x64");

  std::printf("\nint1, artifact folded at TileK=64 -> F=4\n");
  artifact<1, 4, 64, 128, 64, 64>("(64,128) w64x64 [int1 winner]");
  return blind ? 2 : 0;
}
