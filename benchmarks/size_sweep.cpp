// HOW BIG IS THE SWEEP ONCE STAGES ARE AN AXIS? -- asked before anyone spends box time on it.
//
// WHY THIS IS A SEPARATE PROGRAM AND NOT A NUMBER IN A DOCUMENT. The user's decision (2026-08-04) is that
// split-K stays out and stages go in. Stages were not enumerated: emit_tactic_space.cpp calls sweep_exclusion,
// which pins stages=2 as an EXISTENCE test ("if the shallowest pipeline cannot fit, no deeper one can"). That is
// the right filter for "is this cell buildable at all" and the wrong one for "what will the sweep cost", because
// the smem predicate is per_stage*stages, so deep stages fall out for most tiles. Guessing which way that lands
// is how a sweep gets launched and then abandoned halfway.
//
// It reads ppu_tactic_space.hpp -- the same rules both launchers static_assert against -- rather than restating
// the smem formula here. A sizing tool with its own copy of the predicate would size a space nobody runs.
//
//   c++ -std=c++17 -Iquactlize/include benchmarks/size_sweep.cpp -o /tmp/size_sweep && /tmp/size_sweep
//
// Prints, per operator: cells legal at each stage depth, and the instantiation count under two policies -- the
// full reachable set, and the set left once the RESIDENT ARRANGEMENT pins TK. The second is the one that matters:
// F is derived from (bits, TK), so a weight packed at fold F can only be read at the TK that yields F. Pinning
// the arrangement therefore pins TK for every folded width, and leaves it free only where F=1 absorbs the tile.
#include <cstdio>
#include <map>
#include <vector>

#include "ppu_tactic_space.hpp"

namespace {

// The stage depths build.sh actually offers (MOE_STAGES="4;6;8;12"), plus 2 because that is what the existence
// test uses and the difference between "legal at 2" and "legal at 4" is the whole point of printing this.
constexpr int kStages[] = {2, 4, 6, 8, 12};

// TK pinned by the user's decided arrangement. A width whose F is 1 at every TK<=256 keeps them all; a folded
// width keeps only the TK that produces its F. Derived below rather than listed, so it cannot disagree with
// fold_for.
bool tk_survives_pinning(ppu_tactics::FormatSpec const& s, int tk) {
  int const f_lo = ppu_tactics::fold_for(s.low_bits, tk);
  int const f_hi = s.high_bits ? ppu_tactics::fold_for(s.high_bits, tk) : 1;
  // The pinned arrangement is the one the user fixed: fold as deep as the 32-byte floor demands at the SMALLEST
  // tile that is legal for the width, and no deeper. Concretely: keep tk if no smaller legal tk gives the same
  // pair (f_lo, f_hi) -- i.e. this tk is the widest tile that still reads that artifact.
  for (int smaller : ppu_tactics::kTileK) {
    if (smaller >= tk) continue;
    int const g_lo = ppu_tactics::fold_for(s.low_bits, smaller);
    int const g_hi = s.high_bits ? ppu_tactics::fold_for(s.high_bits, smaller) : 1;
    if (g_lo == f_lo && g_hi == f_hi) return false;   // a smaller tile reads the same bytes; this one is a dup
  }
  return true;
}

template <class Space>
void size_one(char const* op) {
  using namespace ppu_tactics;
  std::map<int, int> legal_all, legal_pinned;
  int reachable = 0, reachable_pinned = 0;

  for (auto const& spec : kFormats)
    for (int tk : kTileK)
      for (int tm : kTileM)
        for (int tn : kTileN)
          for (int wm : kWarpM)
            for (int wn : kWarpN) {
              Candidate const c{spec, tm, tn, tk, wm, wn};
              if (Space::sweep_exclusion(c) != Exclusion::None) continue;
              bool const pinned = tk_survives_pinning(spec, tk);
              ++reachable;
              if (pinned) ++reachable_pinned;
              for (int st : kStages) {
                if (Space::topology_exclusion(c, st) != Exclusion::None) continue;
                ++legal_all[st];
                if (pinned) ++legal_pinned[st];
              }
            }

  std::printf("\n== %s ==\n  reachable cells: %d   (after the resident arrangement pins TK: %d)\n",
              op, reachable, reachable_pinned);
  int tot_all = 0, tot_pin = 0;
  std::printf("  %-8s %10s %10s\n", "stages", "all", "pinned");
  for (int st : kStages) {
    std::printf("  %-8d %10d %10d\n", st, legal_all[st], legal_pinned[st]);
    tot_all += legal_all[st];
    tot_pin += legal_pinned[st];
  }
  // THE NUMBER THAT DECIDES WHETHER THIS IS A SWEEP OR A PROJECT. Each (cell, stage) pair is a distinct kernel
  // instantiation, i.e. a compile -- M is a runtime loop and costs nothing extra to build.
  std::printf("  -> %d instantiations to build (%d under pinning)\n", tot_all, tot_pin);
}

}  // namespace

int main() {
  std::printf("Sweep size with STAGES as an axis and split-K excluded (user, 2026-08-04).\n"
              "Legality is ppu_tactic_space.hpp's own topology_exclusion(c, stages); nothing is restated here.\n");
  size_one<ppu_tactics::DenseSpace>("dense");
  size_one<ppu_tactics::GroupedSpace>("grouped");
  std::printf("\nM is a RUNTIME loop, so it multiplies timings, not compiles. Read the instantiation count as\n"
              "the cost of the sweep; anything in the thousands is a project that gets abandoned halfway, and\n"
              "the reduction to reach for first is the pinned column -- it is free, because a tile that reads\n"
              "the same artifact bytes as a smaller one is measuring the same layout twice.\n");
  return 0;
}
