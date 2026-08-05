// DOES THE B-FRAGMENT (value)->(n,k) MAP AT A LARGE TileK EXTEND THE SMALL-TileK MAP, OR REORDER IT?
//
// THE DECISION THIS SETTLES. The offline artifact's byte order is pi = frag.layout()^-1, and the fold F exists
// only because the AIU needs a >=32-byte contiguous-K run: F = 32/run, so F*run = 32 exactly and at TileK=64
// every width's folded row is exactly 32 bytes (int4 1x32, int2 2x16, int1 4x8). The proposal on the table is to
// make the SMALL-TileK folded layout canonical and let larger TileK read the same bytes through a different cute
// layout -- one weight file for every TileK, which is what the M-routes need since GEMV / split-affine / dense
// all read the same tensor.
//
// The algebra is unconditional: frag_256.layout() o pi_64 is a composition of cute layouts, and cute is closed
// under composition and inversion, so that layout ALWAYS exists. What is not unconditional is whether the map at
// TileK=256 is the map at TileK=64 EXTENDED along MMA_K with a uniform stride. If it is, a large-TileK read is a
// stride and the AIU takes it. If the K extent reorders, it is a different permutation and needs the converter
// to emit differently -- a genuinely separate artifact.
//
// THAT DISTINCTION IS EXACTLY THE KIND THIS PROJECT GETS WRONG BY REASONING. "The k-atom map covers 16 K
// elements and repeats, so raising TileK only extends MMA_K" is a written-down relation. This prints the one
// read off the object.
//
//   nvcc -std=c++17 -arch=sm_80 --expt-relaxed-constexpr -I<stub_inc> -I<actlize/include> \
//        -x cu dev/fold_derivation/tilek_fragment_map.cu -o /tmp/tkmap && /tmp/tkmap
//
// TWO TRAPS, both paid for once already (memory/ppu-w2a16-aiu-cube-width-bug):
//   * do NOT define __HGGCCC__ -- CUTLASS_DEVICE then becomes __host__ __device__ and the PPU asm bodies get
//     compiled for the host.
//   * do NOT include collective_builder.hpp -- it drags in the mainloop's operator() and __syncthreads reaches
//     host code.
#include <cstdio>
#include <map>
#include <vector>

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"

using namespace cute;

// The TiledMma exactly as cutlass/gemm/collective/builders/ppu_mma_builder.inl:279 get_tiled_mma builds it:
// MMA_Atom tiled by Layout<Shape<TileM/WarpM, TileN/WarpN, _1>>. Reproduced here rather than included because
// the builder header pulls the collective in with it, and the thing under test is the atom's own map.
template <int TileM, int TileN, int WarpM, int WarpN>
using BuilderTiledMma = TiledMMA<MMA_Atom<PPU0010_16x16x16_F32F16F16F32_TN>,
                                 Layout<Shape<Int<TileM / WarpM>, Int<TileN / WarpN>, _1>>>;

// -> for thread `tid`, the ordered list of (n, k) its B-fragment slots address, over a (TileN, TileK) tile.
// make_identity_tensor + partition_B is the same route the earlier fragment derivation used: the tensor's
// VALUES are its coordinates, so partitioning it reports where each slot came from instead of what it holds.
template <int TileM, int TileN, int TileK, int WarpM, int WarpN>
std::vector<std::pair<int, int>> frag_map(int tid) {
  using TM = BuilderTiledMma<TileM, TileN, WarpM, WarpN>;
  TM tiled_mma;
  auto thr = tiled_mma.get_slice(tid);
  auto ident = make_identity_tensor(make_shape(Int<TileN>{}, Int<TileK>{}));
  auto part = thr.partition_B(ident);
  std::vector<std::pair<int, int>> out;
  for (int i = 0; i < int(size(part)); ++i) {
    auto c = part(i);
    out.emplace_back(int(get<0>(c)), int(get<1>(c)));
  }
  return out;
}

// THE PROPERTY. Let A be the map at TileK=Ka and B the map at TileK=Kb (Kb = r*Ka). B EXTENDS A iff B, read in
// slot order, is A's (n,k) list repeated r times with k shifted by a constant multiple of Ka each repetition.
// Anything else -- a different slot order, a different n for the same slot, a non-uniform k step -- is a
// reorder, and then the artifact cannot be shared.
static bool extends(std::vector<std::pair<int, int>> const& a,
                    std::vector<std::pair<int, int>> const& b, int Ka, int r, char const* tag) {
  if (int(b.size()) != int(a.size()) * r) {
    std::printf("  %-22s NO  -- slot count %zu is not %d x %zu\n", tag, b.size(), r, a.size());
    return false;
  }
  for (int rep = 0; rep < r; ++rep) {
    for (size_t i = 0; i < a.size(); ++i) {
      auto const& want = a[i];
      auto const& got = b[rep * a.size() + i];
      if (got.first != want.first || got.second != want.second + rep * Ka) {
        std::printf("  %-22s NO  -- rep %d slot %zu: expected (n%d,k%d), got (n%d,k%d)\n",
                    tag, rep, i, want.first, want.second + rep * Ka, got.first, got.second);
        return false;
      }
    }
  }
  std::printf("  %-22s YES -- %zu slots repeat %d times, k advancing by exactly %d\n",
              tag, a.size(), r, Ka);
  return true;
}

template <int TileM, int TileN, int WarpM, int WarpN>
void report(char const* geom) {
  std::printf("\n%s   (TiledMma launches %d threads = %d warp(s))\n",
              geom, int(size(BuilderTiledMma<TileM, TileN, WarpM, WarpN>{})),
              int(size(BuilderTiledMma<TileM, TileN, WarpM, WarpN>{})) / 32);
  auto a64  = frag_map<TileM, TileN, 64, WarpM, WarpN>(0);
  auto a128 = frag_map<TileM, TileN, 128, WarpM, WarpN>(0);
  auto a256 = frag_map<TileM, TileN, 256, WarpM, WarpN>(0);
  std::printf("  thread 0, TileK=64 slots: ");
  for (size_t i = 0; i < a64.size() && i < 12; ++i) std::printf("(%d,%d) ", a64[i].first, a64[i].second);
  std::printf("%s\n", a64.size() > 12 ? "..." : "");
  extends(a64, a128, 64, 2, "TK64 -> TK128");
  extends(a64, a256, 64, 4, "TK64 -> TK256");
  // A SECOND THREAD, because a property that holds only for thread 0 is a property of thread 0. Thread 33 is in
  // the second warp when there is one, so it also exercises the WarpOnN tiling rather than just the atom.
  auto b64  = frag_map<TileM, TileN, 64, WarpM, WarpN>(33);
  auto b256 = frag_map<TileM, TileN, 256, WarpM, WarpN>(33);
  extends(b64, b256, 64, 4, "TK64 -> TK256 (t33)");
}

// NEGATIVE CONTROLS. Everything below reported YES on its first run, which is the shape this project has been
// bitten by all week: a check that has only ever passed has not been shown to be a check. These two MUST print
// NO, and if either stops doing so the YES rows above it mean nothing.
static int controls() {
  std::printf("negative controls -- both must say NO:\n");
  int bad = 0;
  auto a64  = frag_map<64, 64, 64, 64, 32>(0);
  auto a256 = frag_map<64, 64, 256, 64, 32>(0);
  // (1) WRONG STRIDE. Same maps, but claim k advances by 32 per repetition instead of 64. If `extends` ignored
  //     the k arithmetic and only counted slots, this would pass.
  if (extends(a64, a256, 32, 4, "wrong k stride (32)")) { std::printf("  ^^ CONTROL PASSED\n"); ++bad; }
  // (2) DIFFERENT WARP SHAPE. w64x16 partitions N differently, so its slots address different n. If `extends`
  //     compared only shapes and not coordinates, this would pass.
  auto other = frag_map<64, 64, 256, 64, 16>(0);
  if (extends(a64, other, 64, 4, "different warp shape")) { std::printf("  ^^ CONTROL PASSED\n"); ++bad; }
  std::printf("%s\n", bad ? "  *** A CONTROL PASSED -- the comparison is blind, ignore every YES below ***"
                           : "  both controls fired: a YES below carries information\n");
  return bad;
}

int main() {
  std::printf("B-fragment (n,k) map vs TileK. YES on every row means one artifact can serve every TileK and\n"
              "the only thing standing in the way is that fold is DERIVED from (bits, TileK) rather than\n"
              "carried with the artifact (ppu_dense_layout.cu:192).\n\n");
  int const blind = controls();
  report<64, 64, 64, 32>("(TileM 64, TileN 64) w64x32   <- BACKTEST A0/A1's winner, 2 warps");
  report<64, 64, 32, 32>("(TileM 64, TileN 64) w32x32   <- 4 warps");
  report<16, 32, 16, 16>("(TileM 16, TileN 32) w16x16   <- BACKTEST D4's geometry, decode band");
  report<64, 128, 64, 64>("(TileM 64, TileN 128) w64x64  <- the recorded int1 winner");
  return blind ? 2 : 0;
}
