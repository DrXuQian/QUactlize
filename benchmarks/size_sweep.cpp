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
// Prints, per operator, the cells legal at each stage depth and the resulting instantiation count.
//
// DO NOT DEDUPLICATE TileK merely because two tactics read the same resident bytes.  That collapses arrangement
// equivalence into kernel equivalence: an unfolded F=1 artifact can be read at several TileK values, but those
// kernels have different A-smem, K-loop counts and transaction widths.  The repository has measured TileK moving
// the winner.  A folded artifact pins TileK; an unfolded artifact makes several TileK tactics compatible -- it
// does not make their instantiations duplicates.
#include <cstdio>
#include <map>
#include <vector>

#include "ppu_tactic_space.hpp"

namespace {

// User-set scope (INBOX 032b): stages above 4 are out, and the sweep axis is exactly {2,4}.  Stage 3 has historical
// measured wins, so this is a declared scope decision -- not a claim that s3 is performance-dominated.
constexpr int kStages[] = {2, 4};

template <class Space>
void size_one(char const* op) {
  using namespace ppu_tactics;
  std::map<int, int> legal;
  int reachable = 0;

  for (auto const& spec : kFormats)
    for (int tk : kTileK)
      for (int tm : kTileM)
        for (int tn : kTileN)
          for (int wm : kWarpM)
            for (int wn : kWarpN) {
              Candidate const c{spec, tm, tn, tk, wm, wn};
              if (Space::sweep_exclusion(c) != Exclusion::None) continue;
              ++reachable;
              for (int st : kStages) {
                if (Space::topology_exclusion(c, st) != Exclusion::None) continue;
                ++legal[st];
              }
            }

  std::printf("\n== %s ==\n  reachable cells: %d\n", op, reachable);
  int total = 0;
  std::printf("  %-8s %10s\n", "stages", "legal");
  for (int st : kStages) {
    std::printf("  %-8d %10d\n", st, legal[st]);
    total += legal[st];
  }
  // THE NUMBER THAT DECIDES WHETHER THIS IS A SWEEP OR A PROJECT. Each (cell, stage) pair is a distinct kernel
  // instantiation, i.e. a compile -- M is a runtime loop and costs nothing extra to build.
  std::printf("  -> %d instantiations to build\n", total);
}

}  // namespace

int main() {
  std::printf("Sweep size with STAGES as an axis and split-K excluded (user, 2026-08-04).\n"
              "Legality is ppu_tactic_space.hpp's own topology_exclusion(c, stages); nothing is restated here.\n");
  size_one<ppu_tactics::DenseSpace>("dense");
  size_one<ppu_tactics::GroupedSpace>("grouped");
  std::printf("\nM is a RUNTIME loop, so it multiplies timings, not compiles.  Arrangement-equivalent tactics\n"
              "remain distinct kernel instantiations: same stored bytes does not imply the same TileK schedule.\n");
  return 0;
}
