// Host-only emitter for the complete finite 025 arrangement/tactic domain.  No PPU SDK or device is needed:
//   c++ -std=c++17 -Iquactlize/include dev/fold_derivation/emit_tactic_space.cpp -o /tmp/emit_tactics
//   /tmp/emit_tactics
//
// Every Cartesian-product cell is printed. F is derived, over-folds are not candidates, and an exclusion has exactly
// one clause (the first violated invariant). Dense and grouped call separate rule wrappers so ci can compare them.
#include <cstdio>

#include "ppu_tactic_space.hpp"

namespace {

template <class Space>
void emit(char const* op) {
  using namespace ppu_tactics;
  for (auto const& spec : kFormats)
    for (int tk : kTileK)
      for (int tm : kTileM)
        for (int tn : kTileN)
          for (int wm : kWarpM)
            for (int wn : kWarpN) {
              Candidate const c{spec, tm, tn, tk, wm, wn};
              Exclusion const why = Space::sweep_exclusion(c);
              int const flo = fold_for(spec.low_bits, tk);
              int const fhi = spec.high_bits ? fold_for(spec.high_bits, tk) : 1;
              std::printf("%-7s format=%-4s bits=%d", op, spec.name, spec.low_bits);
              if (spec.high_bits) std::printf("+%d", spec.high_bits);
              std::printf(" tk=%-3d f=%d", tk, flo);
              if (spec.high_bits) std::printf("/%d", fhi);
              std::printf(" tile=%dx%dx%d warp=%dx%d ", tm, tn, tk, wm, wn);
              if (why == Exclusion::None) std::printf("reachable\n");
              else std::printf("excluded: %s\n", exclusion_clause(why));
            }
}

}  // namespace

int main() {
  emit<ppu_tactics::DenseSpace>("dense");
  emit<ppu_tactics::GroupedSpace>("grouped");
  return 0;
}
