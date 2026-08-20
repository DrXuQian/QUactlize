// L212 -- Q4/A32 folded global-to-shared address witness.
//
// The A32/F2 reader was proved from shared memory onward in L123/L211.  This
// file instantiates the exact production AIU operand and the exact counting
// tensor used by MainloopPPUAiuFold::load_init_B, then exposes every source
// coordinate selected for one TN64/TK128 tactic tile.  The final comparison is
// deliberately against an arithmetic description of the interleaved-256
// artifact, not xplane::place_derived, so a producer/consumer copy of the same
// wrong layout cannot make the gate green.

#include <array>
#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

#include "cute/tensor.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cutlass/numeric_types.h"

using namespace cute;

namespace {

constexpr int N = 1024;
constexpr int K = 5120;
constexpr int Fold = 2;
constexpr int ArtifactTileK = 32;
constexpr int TacticTileN = 64;
constexpr int TacticTileK = 128;
constexpr int PhysicalN = TacticTileN / Fold;
constexpr int PhysicalK = Fold * TacticTileK;
constexpr int Interleave = 256;
constexpr int PhysicalRows = N / Fold;
constexpr int Slabs = K * Fold / Interleave;

constexpr int AiuContElemSize = 64;
constexpr int InstNum = PhysicalK / AiuContElemSize;
constexpr int BitsPerAiu = PhysicalN * (AiuContElemSize * 4 / 8) * 8;
using CopyInst = PPU0010_AIU_LOAD<Int<BitsPerAiu>, cutlass::int4b_t, false>;
using GmemCopy = decltype(make_tiled_copy(
    Copy_Atom<CopyInst, cutlass::int4b_t>{},
    Layout<Shape<_1, _1>, Stride<_1, _1>>{},
    Layout<Shape<Int<PhysicalN>, Int<AiuContElemSize>>>{}));

// Independent byte/code address for the folded artifact's interleaved-256
// storage.  `physical_n` is a row in N/F, `physical_k` spans F*K.  One 256-code
// slab stores every physical row before the next slab.
constexpr int expected_code(int physical_n, int physical_k) {
  return (physical_k / Interleave) * PhysicalRows * Interleave
       + physical_n * Interleave + physical_k % Interleave;
}

bool run() {
  static_assert(AiuContElemSize == 64,
                "A32/F2 int4 is one 32-byte AIU copy quantum");
  static_assert(InstNum == 4,
                "TK128/F2 contains four A32 copy quanta");

  auto const b_shape = make_shape(Int<PhysicalRows>{},
      make_shape(Int<Interleave>{}, Int<Slabs>{}));
  auto const layout_counting = make_layout(
      b_shape,
      make_stride(ScaledBasis<_1, 1>{},
                  make_stride(ScaledBasis<_1, 0>{},
                              ScaledBasis<int, 1>{PhysicalRows})));
  auto mB = make_counting_tensor(layout_counting);

  // Same physical tile and Step<X,N,K> projection as the collective.  Select
  // N tile 0 while retaining all tactic-K tiles.
  using FoldTilerB = Shape<Int<TacticTileN>, Int<PhysicalN>, Int<PhysicalK>>;
  auto gB = local_tile(mB, FoldTilerB{}, make_coord(0, 0, _), Step<X, _1, _1>{});

  GmemCopy copy;
  auto slice = copy.get_slice(0);
  auto part = slice.partition_S(gB);

  std::printf("L212 gB="); print(gB.layout()); std::printf("\n");
  std::printf("L212 part="); print(part.layout()); std::printf("\n");
  std::printf("L212 operand aiu_elems=%d inst=%d bits=%d\n",
              AiuContElemSize, InstNum, BitsPerAiu);

  // Each partition rest coordinate must select a distinct 32-byte quantum.
  // The PPU mix iterator exposes (coord_w, coord_h): coord_w is a code offset
  // inside the 256-code interleave and coord_h is the flattened
  // (physical_n, slab) row.  The descriptor consumes coord_w in packed-int4
  // units and coord_h with a 256-code row pitch.  Decode that ABI directly;
  // do not reinterpret the two coordinates as logical (n,k).
  std::set<int> observed;
  bool ok = true;
  int shown = 0;
  constexpr int Rest = InstNum;
  constexpr int KTiles = K / TacticTileK;
  for (int kt = 0; kt < KTiles; ++kt) {
    auto kc = idx2crd(kt, shape<2>(gB));
    for (int r = 0; r < Rest; ++r) {
      auto atom = part(_, _, r, kc);
      auto coord = atom.data().coord_;
      int const coord_w = int(get<0>(coord));
      int const coord_h = int(get<1>(coord));
      int const got = coord_h * Interleave + coord_w;
      int const want = expected_code(0, kt * PhysicalK + r * AiuContElemSize);
      bool const one = got == want;
      ok &= one;
      observed.insert(got);
      if (shown++ < 16)
        std::printf("L212 kt=%d rest=%d coord=(%d,%d) code=%d want=%d %s\n",
                    kt, r, coord_w, coord_h, got, want,
                    one ? "PASS" : "FAIL");
    }
  }
  ok &= int(observed.size()) == KTiles * Rest;
  // Constructive negative: swapping the descriptor's W/H roles must fail.
  // This catches the exact coordinate-interpretation mistake that the
  // positive witness is intended to rule out.
  auto first = part(_, _, 1, idx2crd(0, shape<2>(gB))).data().coord_;
  int const swapped = int(get<0>(first)) * Interleave + int(get<1>(first));
  int const first_want = expected_code(0, AiuContElemSize);
  bool const negative_red = swapped != first_want;
  ok &= negative_red;
  std::printf("L212 distinct=%zu/%d swapped-coordinate-negative=%s result=%s\n",
              observed.size(),
              KTiles * Rest,
              negative_red ? "RED" : "FALSE-GREEN",
              ok ? "PASS" : "FAIL");
  return ok;
}

}  // namespace

int main() { return run() ? 0 : 1; }
