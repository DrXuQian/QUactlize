// EMIT A BENCH'S COMPILED CONFIG TABLE FROM THE SHARED TACTIC RULES -- for EITHER operator.
//
// WAS emit_dense_configs.cpp, which named the one operator it could serve. It now takes --space, because the
// dense and grouped sweeps searching different sets is how "dense and grouped should be comparable" fails
// silently: two tables, two pruning policies, and a difference that shows up as a performance claim rather than
// as a diff. The MoE side still defines its set as a Cartesian product of MOE_*_LIST environment axes, which
// cannot express an arbitrary set at all; this program can, and emits the same shape of table for both.
//
// --space=compare IS THE POINT, not a convenience. ppu_tactic_space.hpp keeps DenseSpace and GroupedSpace as
// separate wrappers over one implementation, and says why at its line 185: "The emitter asks each launcher for
// its own answer and a comparator checks them; sharing only the current implementation makes equality explicit
// today without making future drift invisible." The comparator it refers to did not exist. Without it, the two
// being identical was a fact about the source that nobody re-established, so the first divergence would have
// been absorbed rather than reported -- and this program would have kept emitting one operator's answer for
// both. `--space=compare` walks the full grid and prints every candidate where the two disagree.
//
// IT NO LONGER DEMANDS THAT THEY BE IDENTICAL, and the change is the whole point of having built the check.
// INBOX 064 (codex, commit 44052c7) split the four-warp rule after the two facts collided: the sub-four-warp
// device abort measured on ppu001 is a DENSE-ROUTE quarantine, while the grouped kernel has POSITIVELY MEASURED
// two-warp rows through the same mainloop -- (64,128,64) w64x64 is the recorded int1 61.2% winner. Requiring
// identity would have deleted a measured optimum.
//
// So the comparator now classifies rather than counts: a disagreement is DECLARED when dense's own verdict is
// the quarantine, DRIFT otherwise, and only DRIFT exits non-zero. Today that is 891 declared and 0 drift. The
// value of the check is unchanged -- it is still the only thing that would report an UNintended divergence --
// but its verdict is "no drift", not "no difference".
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
//   c++ -std=c++17 -Iquactlize/include benchmarks/emit_tactic_configs.cpp -o /tmp/emit_tactic
//   /tmp/emit_tactic <bits> <tile_k> [--space=dense|grouped|compare] [stage ...]
//                                                        > benchmarks/lowbit_<space>_configs.inc
//
// bits is the LOW plane width (1, 2 or 4); tile_k must match the binary's BENCH_TSK. Both are build-time
// constants of the bench, which is why the table is generated per binary rather than filtered at run time:
// instantiating a config for the wrong TileK costs compile time and can never be selected.
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

// The four-warp minimum USED TO BE DUPLICATED HERE. codex's 047 moved it into ppu_tactic_space.hpp's
// common_kernel_exclusion, which sweep_exclusion reaches, so the copy became dead. It was not deleted on the
// strength of reading that call chain: the copy was stubbed out and both binaries emitted over nine (bits,
// tile_k) pairs -- 1/2/4 x 32/64/128 -- giving identical row counts every time, with the stub verified present
// so that "same" could not mean "the probe never applied". The emitted set was then checked to CONTAIN the
// property rather than merely be unchanged by its removal: min (TM/WM)*(TN/WN) over the 227 emitted rows is 4,
// and no row is below it. Both halves are needed -- unchanged output would also be the symptom of a constraint
// that neither copy was applying.
template <class Space>
std::vector<Candidate> legal_grid(FormatSpec const& spec, int tk) {
  // Legality FIRST and from the shared header, so the pruning policy only ever removes rows that could have
  // been built. A policy applied to an unfiltered grid would emit configurations that fail to compile.
  std::vector<Candidate> ok;
  for (int tm : kTileM)
    for (int tn : kTileN)
      for (int wm : kWarpM)
        for (int wn : kWarpN) {
          Candidate const c{spec, tm, tn, tk, wm, wn};
          if (Space::sweep_exclusion(c) != Exclusion::None) continue;
          ok.push_back(c);
        }
  return ok;
}

// THE COMPARATOR ppu_tactic_space.hpp:185 ASKS FOR. Every predicate the emitter consults, over the whole grid
// and every stage in scope, asked of both spaces. Reports disagreements rather than asserting: a divergence is
// news about the two operators, not necessarily a defect, and the emitter cannot know which. Returns the count
// so a gate can require zero while a human reads what differs.
// THE ONE DECLARED ASYMMETRY. codex's 064 resolution (ppu_tactic_space.hpp at 44052c7): the sub-four-warp device
// abort observed on ppu001 is a DENSE-ROUTE QUARANTINE, not PPU collective legality. The grouped kernel has
// POSITIVELY MEASURED two-warp rows through the same mainloop -- (64,64,64) w64x32 and (64,128,64) w64x64, the
// latter being the recorded int1 61.2% winner -- and GemmUniversalAdapter's max(4,...) is only legacy WarpCount
// metadata while kThreadCount and both kernels' block shapes remain size(TiledMma). So the dense evidence cannot
// exclude grouped rows, and requiring the two spaces to agree would delete a measured optimum.
//
// A DISAGREEMENT IS ACCEPTED ONLY WHEN DENSE'S OWN VERDICT IS THAT ONE. Dense short-circuits at the warp check, so
// grouped may report anything downstream -- None, or a later exclusion dense never reached. What must not happen
// is the two disagreeing for any OTHER reason, and that is what still exits non-zero. Adding a second entry here
// is a deliberate act and should carry the same kind of citation.
constexpr Exclusion kDeclaredDenseOnly[] = { Exclusion::DenseSubFourWarpDeviceAbort };

bool is_declared(int dense_verdict) {
  for (Exclusion e : kDeclaredDenseOnly)
    if (int(e) == dense_verdict) return true;
  return false;
}

int compare_spaces(FormatSpec const& spec, int tk) {
  int diffs = 0, declared_n = 0;
  auto say = [&](Candidate const& c, char const* what, int a, int b, int st) {
    bool const ok = is_declared(a);
    std::printf("  %-9s %-22s tm=%-4d tn=%-4d wm=%-3d wn=%-4d st=%-3d  dense=%-38s grouped=%s\n",
                ok ? "DECLARED" : "DRIFT", what, c.tm, c.tn, c.wm, c.wn, st,
                exclusion_clause(Exclusion(a)), exclusion_clause(Exclusion(b)));
    if (ok) ++declared_n; else ++diffs;
  };
  for (int tm : kTileM)
    for (int tn : kTileN)
      for (int wm : kWarpM)
        for (int wn : kWarpN) {
          Candidate const c{spec, tm, tn, tk, wm, wn};
          int const kd = int(DenseSpace::kernel_exclusion(c)), kg = int(GroupedSpace::kernel_exclusion(c));
          if (kd != kg) say(c, "kernel_exclusion", kd, kg, 0);
          int const sd = int(DenseSpace::sweep_exclusion(c)), sg = int(GroupedSpace::sweep_exclusion(c));
          if (sd != sg) say(c, "sweep_exclusion", sd, sg, 0);
          int const ad = int(DenseSpace::static_sweep_exclusion(c)),
                    ag = int(GroupedSpace::static_sweep_exclusion(c));
          if (ad != ag) say(c, "static_sweep_exclusion", ad, ag, 0);
          if (DenseSpace::compact_a_supported(c) != GroupedSpace::compact_a_supported(c))
            say(c, "compact_a_supported", DenseSpace::compact_a_supported(c),
                GroupedSpace::compact_a_supported(c), 0);
          for (int st : g_stages) {
            int const td = int(DenseSpace::topology_exclusion(c, st)),
                      tg = int(GroupedSpace::topology_exclusion(c, st));
            if (td != tg) say(c, "topology_exclusion", td, tg, st);
            // Compact-A capacities are the ladder the launchers actually instantiate.
            for (int rows : {1, 2, 4}) {
              int const cd = int(DenseSpace::compact_a_topology_exclusion(c, st, rows)),
                        cg = int(GroupedSpace::compact_a_topology_exclusion(c, st, rows));
              if (cd != cg) say(c, "compact_a_topology", cd, cg, st);
            }
          }
        }
  std::printf("%d declared difference(s), %d unexpected disagreement(s)\n", declared_n, diffs);
  return diffs;
}

}  // namespace

template <class Space>
static int emit(FormatSpec const& spec, int bits, int tk, char const* space_name, char const* macro_prefix) {
  std::vector<Candidate> const ok = legal_grid<Space>(spec, tk);
  if (ok.empty()) { std::fprintf(stderr, "no legal tactic at bits=%d tile_k=%d\n", bits, tk); return 1; }

  std::set<Row> rows;
  int n_prim = 0, n_guard = 0;
  for (int st : g_stages) {
    std::vector<Candidate> legal;
    for (auto const& c : ok)
      if (Space::topology_exclusion(c, st) == Exclusion::None) legal.push_back(c);

    for (auto const& c : legal) {
      bool const p = primary(legal, c);
      bool const g = h1_guard(legal, c) || n_geometry_guard(legal, c);
      if (!p && !g) continue;
      if (rows.insert(Row{c.tm, c.tn, c.wm, c.wn, st}).second) { if (p) ++n_prim; else ++n_guard; }
    }
  }

  std::printf("// GENERATED by benchmarks/emit_tactic_configs.cpp -- do not edit.\n");
  std::printf("//   space=%s bits=%d tile_k=%d   %zu configs (%d primary, %d guard) over %zu stage(s)\n",
              space_name, bits, tk, rows.size(), n_prim, n_guard, g_stages.size());
  std::printf("//   stages:");
  for (int st : g_stages) std::printf(" %d", st);
  std::printf("   <- what this table COVERS; a winner outside it cannot be found by a sweep using it\n");
  std::printf("//\n");
  std::printf("// Regenerate after changing ppu_tactic_space.hpp or the pruning policy:\n");
  std::printf("//   c++ -std=c++17 -Iquactlize/include benchmarks/emit_tactic_configs.cpp -o /tmp/emit_tactic &&\\\n");
  std::printf("//   /tmp/emit_tactic %d %d --space=%s", bits, tk, space_name);
  for (int st : g_stages) std::printf(" %d", st);
  std::printf(" > benchmarks/lowbit_%s_configs.inc\n", space_name);
  std::printf("//\n");
  std::printf("// The second X argument carries the dispatch BODY through the list; supported_configs() passes\n");
  std::printf("// nothing for it. That is what lets ONE list feed both the runtime table and the compile-time\n");
  std::printf("// if-chain, so a row cannot exist in one and not the other.\n");
  // THE GUARD THAT MAKES A STALE .inc A COMPILE ERROR. bits and TileK are build-time constants of the binary
  // (QUANT= and BENCH_TSK=), and a table generated for another pair is not merely suboptimal -- every row in it
  // is a tactic this binary cannot select, so the sweep would search an empty set and report whatever the
  // fallback does. Emitting the pair here lets the consumer static_assert it.
  //
  // THE PREFIX IS PER SPACE for the same reason: a grouped table included where a dense one is expected would
  // satisfy every static_assert about bits and TileK and still be the wrong operator's set. Distinct macro
  // names make that a redefinition or an undefined macro rather than a silent substitution.
  std::printf("#define %s_CFG_BITS  %d\n", macro_prefix, bits);
  std::printf("#define %s_CFG_TILEK %d\n\n", macro_prefix, tk);
  std::printf("#define %s_CFG_LIST(X, B) \\\n", macro_prefix);
  size_t i = 0;
  for (auto const& r : rows) {
    std::printf("  X(%d,%d,%d,%d,%d,B)%s\n", std::get<0>(r), std::get<1>(r), std::get<2>(r),
                std::get<3>(r), std::get<4>(r), ++i == rows.size() ? "" : " \\");
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: emit_tactic_configs <bits:1|2|4> <tile_k> [--space=dense|grouped|compare] [stage ...]\n"
                 "  --space defaults to dense.  stages default to 2 3 4.\n"
                 "  --space=compare emits no table: it walks the grid asking BOTH spaces every predicate the\n"
                 "  emitter uses and prints the disagreements, exiting non-zero if there are any.\n");
    return 2;
  }
  const int bits = std::atoi(argv[1]);
  const int tk   = std::atoi(argv[2]);

  char const* space = "dense";
  std::vector<int> stages;
  for (int i = 3; i < argc; ++i) {
    if (std::strncmp(argv[i], "--space=", 8) == 0) { space = argv[i] + 8; continue; }
    stages.push_back(std::atoi(argv[i]));
  }
  if (!stages.empty()) g_stages = stages;

  FormatSpec const* spec = nullptr;
  for (auto const& s : kFormats)
    if (s.low_bits == bits && s.high_bits == 0) spec = &s;
  if (!spec) { std::fprintf(stderr, "no single-plane format with low_bits=%d in kFormats\n", bits); return 2; }

  if (std::strcmp(space, "dense") == 0)   return emit<DenseSpace>(*spec, bits, tk, "dense", "LOWBIT_DENSE");
  if (std::strcmp(space, "grouped") == 0) return emit<GroupedSpace>(*spec, bits, tk, "grouped", "LOWBIT_GROUPED");
  if (std::strcmp(space, "compare") == 0) {
    std::printf("comparing DenseSpace vs GroupedSpace: bits=%d tile_k=%d stages", bits, tk);
    for (int st : g_stages) std::printf(" %d", st);
    std::printf("\n");
    int const d = compare_spaces(*spec, tk);
    // A disagreement is NEWS, not a failure of this program -- but it must not be exit 0, or a gate that runs
    // this to establish "the two spaces still agree" would pass while they diverge, which is the exact shape of
    // check this repo has been bitten by before.
    return d == 0 ? 0 : 1;
  }
  std::fprintf(stderr, "unknown --space=%s (want dense, grouped or compare)\n", space);
  return 2;
}
