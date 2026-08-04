// EMIT test_lowbit_dense_bench's compiled config table from the shared tactic rules.
//
// TWO PROBLEMS THIS REPLACES, and the second is worse than the first.
//
//  1. The table was 17 hand-written rows. The pruned set for one (schema, TileK) binary is 27-93 rows under
//     codex's H1/H2 primary geometry plus its guards, so the sweep was searching a fifth of its own space and
//     nothing said so.
//  2. THE LIST AND THE DISPATCH WERE TWO HAND-MAINTAINED COPIES. supported_configs() returned rows; the
//     LOWBIT_DENSE_DISPATCH if-chain instantiated them; nothing checked that the two agreed. A row present in the list
//     and absent from the chain reaches `config %s not compiled in` and exit(1) at run time -- after the build,
//     on the box, in the middle of a sweep. Both now expand from ONE X-macro list, so the failure is not
//     expressible.
//
// The rules come from ppu_tactic_space.hpp -- the same header both launchers static_assert against -- so this
// program has no copy of the legality predicate, only of the pruning policy it is asked to apply.
//
//   c++ -std=c++17 -Iquactlize/include benchmarks/emit_dense_configs.cpp -o /tmp/emit_dense
//   /tmp/emit_dense <bits> <tile_k> > benchmarks/lowbit_dense_configs.inc
//
// bits is the LOW plane width (1, 2 or 4); tile_k must match the binary's BENCH_TSK. Both are build-time
// constants of test_lowbit_dense_bench, which is why the table is generated per binary rather than filtered at run
// time: instantiating a config for the wrong TileK costs compile time and can never be selected.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <set>
#include <tuple>
#include <vector>

#include "ppu_tactic_space.hpp"

using namespace ppu_tactics;

namespace {

// The user's stage scope: above 4 is out. 3 stays -- s2, s3 and s4 have each been a measured winner for some
// format/shape, so dropping s3 would start from a truncated space (that error was made once already, in a
// relayed transcription of "stage 大于4就没必要了", which excludes >4 and says nothing about 3).
// STAGES ARE AN ARGUMENT, not a constant. They were {2,3,4} here from the user's "stage 大于4就没必要了" --
// and the first real sweep row on the MoE side came back at s6 and won, on a bench whose own ladder is
// {2,3,4,6,8,12}. That scope reached this file and never reached that bench, which is the only reason the
// winner was ever built. A scope decision that can only be changed by editing the generator is a scope
// decision nobody re-examines; passing it in makes "what did this table cover" answerable from the command.
std::vector<int> g_stages{2, 3, 4};

using Row = std::tuple<int, int, int, int, int>;    // tm, tn, wm, wn, stages

// Do not spell "largest legal" as min(TM,64). It happens to agree in today's WN<=64 producer domain, but WN=128
// can make WM64 fail the accumulator ceiling while WM32 remains legal. Derive both H1 rungs from the filtered set,
// which is the claim the policy actually makes and remains correct if producer reachability expands.
int largest_wm(std::vector<Candidate> const& legal, Candidate const& c, int below = 1 << 30) {
  int best = -1;
  for (auto const& q : legal)
    if (q.tm == c.tm && q.tn == c.tn && q.wn == c.wn && q.wm < below && q.wm > best) best = q.wm;
  return best;
}

bool primary(std::vector<Candidate> const& legal, Candidate const& c) {
  return c.wm == largest_wm(legal, c) && c.tn == 2 * c.wn;
}

// H1: for EVERY TileM, guard the next-smaller legal WM at its lightest and heaviest legal ratio-two N geometry.
// The previous transcription restricted this to extreme TileM and kept every smaller WM at every N shape: it
// omitted the interior-TileM guard where a recorded prefill winner lives, while compiling rows H1 never requested.
bool h1_guard(std::vector<Candidate> const& legal, Candidate const& c) {
  if (c.tn != 2 * c.wn) return false;
  int const wm_max = largest_wm(legal, c);
  if (c.wm != largest_wm(legal, c, wm_max)) return false;
  int n_lo = std::numeric_limits<int>::max(), n_hi = 0;
  for (auto const& q : legal)
    if (q.tm == c.tm && primary(legal, q)) {
      n_lo = std::min(n_lo, q.tn);
      n_hi = std::max(n_hi, q.tn);
    }
  return c.tn == n_lo || c.tn == n_hi;
}

// N-geometry guard: retain EVERY legal TileN/WarpN ratio at EVERY TileM. An enumerated {1,4} guard had no mechanism
// for excluding ratio 8, and restricting the expanded guard to extreme TileM still omitted the motivating winner:
// grouped i4 64x128:64 w64x16 s6 is ratio 8 at interior TileM=64. That grouped row does not select dense's winner,
// but it falsifies the grouped evidence used to prune N geometry. Until dense measurements support a replacement,
// there is no justified N-ratio exclusion. H1 still prunes WarpM: each N geometry keeps its largest legal WarpM,
// while the next-smaller-WarpM guards above test H1 at the lightest and heaviest ratio-two shapes for every TileM.
bool n_geometry_guard(std::vector<Candidate> const& legal, Candidate const& c) {
  return c.wm == largest_wm(legal, c);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: emit_dense_configs <bits:1|2|4> <tile_k> [stage ...]  (default 2 3 4)\n"); return 2; }
  const int bits = std::atoi(argv[1]);
  const int tk   = std::atoi(argv[2]);
  if (argc > 3) {
    g_stages.clear();
    for (int i = 3; i < argc; ++i) g_stages.push_back(std::atoi(argv[i]));
  }

  FormatSpec const* spec = nullptr;
  for (auto const& s : kFormats)
    if (s.low_bits == bits && s.high_bits == 0) spec = &s;
  if (!spec) { std::fprintf(stderr, "no single-plane format with low_bits=%d in kFormats\n", bits); return 2; }

  // Legality FIRST and from the shared header, so the pruning policy below only ever removes rows that could
  // have been built. A policy applied to an unfiltered grid would emit configurations that fail to compile.
  std::vector<Candidate> ok;
  for (int tm : kTileM)
    for (int tn : kTileN)
      for (int wm : kWarpM)
        for (int wn : kWarpN) {
          Candidate const c{*spec, tm, tn, tk, wm, wn};
          if (DenseSpace::sweep_exclusion(c) != Exclusion::None) continue;
          // FEWER THAN FOUR WARPS PER CTA ABORTS ON THE DEVICE. Measured on ppu001 2026-08-04: every config
          // with (TM/WM)*(TN/WN) < 4 hit an unconditional assert and core-dumped, and every config with 4 or
          // more ran. 126 of the 293 rows in the generated table were in that set, so a sweep lost all 293
          // measurements to the first one -- a device assert poisons the context and cannot be caught, so the
          // only defence is not launching it.
          //
          // THIS BELONGS IN ppu_tactic_space.hpp, NOT HERE. It is a LEGALITY constraint -- the collective
          // cannot execute these -- and the predicate is what both operators consult, so the MoE sweep will
          // walk into the same wall. It is here because it unblocks the dense sweep today; INBOX 047 asks for
          // it to move, together with the actual mechanism, which is not yet known. The empirical boundary is
          // between 2 and 4 warps; no 3-warp configuration exists, so nothing distinguishes ">= 4" from
          // ">= 3" in the evidence.
          if ((tm / wm) * (tn / wn) < 4) continue;
          ok.push_back(c);
        }
  if (ok.empty()) { std::fprintf(stderr, "no legal tactic at bits=%d tile_k=%d\n", bits, tk); return 1; }

  std::set<Row> rows;
  int n_prim = 0, n_guard = 0;
  for (int st : g_stages) {
    std::vector<Candidate> legal;
    for (auto const& c : ok)
      if (DenseSpace::topology_exclusion(c, st) == Exclusion::None) legal.push_back(c);

    for (auto const& c : legal) {
      bool const p = primary(legal, c);
      bool const g = h1_guard(legal, c) || n_geometry_guard(legal, c);
      if (!p && !g) continue;
      if (rows.insert(Row{c.tm, c.tn, c.wm, c.wn, st}).second) { if (p) ++n_prim; else ++n_guard; }
    }
  }

  std::printf("// GENERATED by benchmarks/emit_dense_configs.cpp -- do not edit.\n");
  std::printf("//   bits=%d tile_k=%d   %zu configs (%d primary, %d guard) over %zu stage(s)\n",
              bits, tk, rows.size(), n_prim, n_guard, g_stages.size());
  std::printf("//   stages:");
  for (int st : g_stages) std::printf(" %d", st);
  std::printf("   <- what this table COVERS; a winner outside it cannot be found by a sweep using it\n");
  std::printf("//\n");
  std::printf("// Regenerate after changing ppu_tactic_space.hpp or the pruning policy:\n");
  std::printf("//   c++ -std=c++17 -Iquactlize/include benchmarks/emit_dense_configs.cpp -o /tmp/emit_dense &&\\\n");
  std::printf("//   /tmp/emit_dense %d %d > benchmarks/lowbit_dense_configs.inc\n", bits, tk);
  std::printf("//\n");
  std::printf("// The second X argument carries the dispatch BODY through the list; supported_configs() passes\n");
  std::printf("// nothing for it. That is what lets ONE list feed both the runtime table and the compile-time\n");
  std::printf("// if-chain, so a row cannot exist in one and not the other.\n");
  // THE GUARD THAT MAKES A STALE .inc A COMPILE ERROR. bits and TileK are build-time constants of the binary
  // (QUANT= and BENCH_TSK=), and a table generated for another pair is not merely suboptimal -- every row in it
  // is a tactic this binary cannot select, so the sweep would search an empty set and report whatever the
  // fallback does. Emitting the pair here lets the consumer static_assert it.
  std::printf("#define LOWBIT_DENSE_CFG_BITS  %d\n", bits);
  std::printf("#define LOWBIT_DENSE_CFG_TILEK %d\n\n", tk);
  std::printf("#define LOWBIT_DENSE_CFG_LIST(X, B) \\\n");
  size_t i = 0;
  for (auto const& r : rows) {
    std::printf("  X(%d,%d,%d,%d,%d,B)%s\n", std::get<0>(r), std::get<1>(r), std::get<2>(r),
                std::get<3>(r), std::get<4>(r), ++i == rows.size() ? "" : " \\");
  }
  return 0;
}
