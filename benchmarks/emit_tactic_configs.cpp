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
// THE TWO SPACES AGREE AGAIN, AND THE COMPARATOR SHOULD REPORT ZERO. For a day they did not: a four-warp
// minimum sat on the dense side recording a device abort observed once on ppu001, while the grouped kernel had
// POSITIVELY MEASURED two-warp rows through the same mainloop -- (64,64,64) w64x32 is the recorded int4 65.0%
// winner and (64,128,64) w64x64 the int1 61.2% one. The comparator was taught to classify that difference as
// DECLARED and exit zero on 891 of them.
//
// That was the wrong repair. The difference was not between the two OPERATORS; it was a remembered observation
// living in the file that defines what is legal, and the right fix was to delete it rather than to teach the
// checker to expect it (2026-08-05, see ppu_tactic_space.hpp's dense_kernel_exclusion). The classification
// machinery stays because a real asymmetry could appear later and should have a visible home -- but its list is
// now empty, so "declared" and "drift" both being zero is the expected reading.
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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

// The CONSUMER TileKs to emit rows for. Empty means "just the artifact's", which reproduces every table this
// program produced before TacticTileK became a row field -- so adding the axis does not silently change what an
// existing regenerate command emits.
std::vector<int> g_tactic_tks;
// The compact-A capacities this table offers. See Row's comment for why only {0} today.
std::vector<int> g_compact_rows;
// Off by default: emit every LEGAL row. --prune=primary-guard narrows to the heuristic's choice.
bool g_prune = false;
// Selected by --format=<name>; null means the positional single-plane lookup.
char const* g_format = nullptr;

// THERE WAS A --space=quarantined HERE, AND ITS REMOVAL IS THE POINT. It emitted the COMPLEMENT of the dense
// space -- only the rows dense refused -- so that the sub-four-warp geometry holding the measured int4 optimum
// could be built as its own binary without touching the exclusion that banned it.
//
// That is a mechanism invented to work around another mechanism, and the outer one should not have existed:
// ppu_tactic_space.hpp was recording a device abort somebody saw once, in the file that defines what is legal.
// With that exclusion deleted (2026-08-05) the rows are simply in the dense table, where they belong, and this
// mode has nothing left to emit. It also cost more than it gave: it stamped `space=dense` into its own header,
// so a quarantined table on the box was indistinguishable from the shipping one, and its by-design compile
// failure was read as a dense regression across INBOX 069/070/071/072/073/077.
//
// If a sweep ever needs to bound the blast radius of a config that aborts, that is a property of the HARNESS --
// flush results as they are produced instead of at the end -- not a reason to carve the search space.

// PROVENANCE IS PART OF THE GENERATED INTERFACE. A stale table used to carry the same header prose and the same
// regeneration command as the current one, so neither a build log nor a failed template instantiation identified
// which tactic space had produced it. Hash the two source inputs whose semantics determine the emitted rows. The
// exact-regeneration gate separately rebuilds this emitter before comparing output; these hashes are identifiers,
// not a claim that a previously compiled /tmp/emit_tactic is current.
constexpr char kSpaceSource[] = "quactlize/include/ppu_tactic_space.hpp";
constexpr char kEmitterSource[] = "benchmarks/emit_tactic_configs.cpp";

std::uint64_t fnv1a64_file(char const* path, bool& ok) {
  std::ifstream in(path, std::ios::binary);
  if (!in) { ok = false; return 0; }
  std::uint64_t hash = UINT64_C(14695981039346656037);
  char bytes[4096];
  while (in) {
    in.read(bytes, sizeof bytes);
    for (std::streamsize i = 0; i < in.gcount(); ++i) {
      hash ^= static_cast<unsigned char>(bytes[i]);
      hash *= UINT64_C(1099511628211);
    }
  }
  ok = in.eof();
  return hash;
}

// tm, tn, TACTIC tile_k, wm, wn, stages. TacticTileK joined the row on 2026-08-05: it is a consumer choice --
// how the kernel reads the bytes -- and no longer implies a layout, because the artifact carries its own fold.
// ArtifactTileK is NOT here and must never be: it identifies the resident bytes, one per weight file, and a
// tactic that could change it would invalidate every artifact on disk. That distinction is the whole point of
// the split, and putting both in a row under one name is how it would be lost.
// tm, tn, TacticTileK, wm, wn, stages, A_COMPACT_ROWS.
//
// THE SEVENTH FIELD IS THE COMPACT-A CAPACITY, and it is here rather than a build switch for the reason TileK
// is: capacity changes SmemLayoutA and therefore SharedStorageSize and therefore the instantiated collective.
// As PPU_A_CPASYNC it was one number for the whole binary, so a compact build REFUSED every launch above its
// capacity instead of falling through to a wider row.
//
// ONLY ZERO IS EMITTED TODAY, deliberately. Zero is ordinary unrestricted A, so this table is byte-identical to
// the previous one apart from the extra field and every config NAME is unchanged -- which matters because those
// names are what the box's --config strings say. Widening the emitted set is a separate change, and it needs one
// more thing this does not: two rows differing only in capacity would share a name, so the name has to grow a
// segment at the same moment the set does.
using Row = std::tuple<int, int, int, int, int, int, int>;

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
std::vector<Candidate> legal_grid(FormatSpec const& spec, int tactic_tk, int artifact_tk) {
  // Legality FIRST and from the shared header, so the pruning policy only ever removes rows that could have
  // been built. A policy applied to an unfiltered grid would emit configurations that fail to compile.
  std::vector<Candidate> ok;
  for (int tm : kTileM)
    for (int tn : kTileN)
      for (int wm : kWarpM)
        for (int wn : kWarpN) {
          Candidate const c{spec, tm, tn, tactic_tk, wm, wn, artifact_tk};
          if (Space::sweep_exclusion(c) != Exclusion::None) continue;
          ok.push_back(c);
        }
  return ok;
}

// THE COMPARATOR ppu_tactic_space.hpp:185 ASKS FOR. Every predicate the emitter consults, over the whole grid
// and every stage in scope, asked of both spaces. Reports disagreements rather than asserting: a divergence is
// news about the two operators, not necessarily a defect, and the emitter cannot know which. Returns the count
// so a gate can require zero while a human reads what differs.
// THERE ARE NO DECLARED ASYMMETRIES LEFT, and the list stays here so that adding one is a deliberate act with a
// visible home rather than an `if` buried in a predicate.
//
// It had exactly one entry, DenseSubFourWarpDeviceAbort, until 2026-08-05. That entry did not describe a
// difference between the two operators; it described a device abort somebody once saw on the dense route, and
// keeping it meant the dense space could never search the geometry that holds the measured int4 optimum. It was
// deleted at the source -- see ppu_tactic_space.hpp's dense_kernel_exclusion -- so the two spaces now agree on
// every kernel-level predicate and this comparator should report ZERO differences. If it reports any, that is
// news either way: a genuine divergence, or somebody reintroducing a remembered observation as a rule.
//
// A future entry must be a real asymmetry between the dense and grouped ROUTES -- something one kernel can do
// that the other structurally cannot -- and must cite where in the source that difference lives. "We measured a
// failure once" is not such a citation.
constexpr Exclusion kDeclaredDenseOnly[] = {};

bool is_declared(int dense_verdict) {
  for (Exclusion e : kDeclaredDenseOnly)
    if (int(e) == dense_verdict) return true;
  return false;
}

int compare_spaces(FormatSpec const& spec, int tk) {
  // artifact == tactic here on purpose: this asks whether the two SPACES disagree at one geometry,
  // not whether a shared artifact is readable. Feeding a split pair would change the question.
  int const tactic_tk = tk, artifact_tk = tk;
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
          Candidate const c{spec, tm, tn, tactic_tk, wm, wn, artifact_tk};
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
// artifact_tk identifies the resident bytes; tactic_tks are the consumer choices emitted as rows. Passing one
// tactic_tk equal to artifact_tk reproduces the pre-2026-08-05 table exactly, plus its new TileK column.
static int emit(FormatSpec const& spec, int bits, int artifact_tk, std::vector<int> const& tactic_tks,
                char const* space_name, char const* macro_prefix) {
  bool space_ok = true, emitter_ok = true;
  std::uint64_t const space_hash = fnv1a64_file(kSpaceSource, space_ok);
  std::uint64_t const emitter_hash = fnv1a64_file(kEmitterSource, emitter_ok);
  if (!space_ok || !emitter_ok) {
    std::fprintf(stderr,
                 "cannot hash generated-table source '%s'; run emit_tactic_configs from the repository root\n",
                 !space_ok ? kSpaceSource : kEmitterSource);
    return 1;
  }

  std::set<Row> rows;
  int n_prim = 0, n_guard = 0, n_other = 0;
  // PRIMARY/GUARD ARE DECIDED WITHIN ONE TacticTileK, not across them. primary() and the guards compare a
  // candidate against the OTHER legal candidates of the same grid; pooling two TileKs would let a row at one
  // TileK suppress a row at another, which is a pruning decision nobody made.
  for (int tactic_tk : tactic_tks) {
    std::vector<Candidate> const ok = legal_grid<Space>(spec, tactic_tk, artifact_tk);
    if (ok.empty()) {
      std::fprintf(stderr, "no legal tactic at bits=%d artifact_tk=%d tactic_tk=%d\n",
                   bits, artifact_tk, tactic_tk);
      return 1;
    }
  for (int st : g_stages) {
  // LEGALITY IS PER CAPACITY, because capacity is an input to it. Until 2026-08-07 this loop asked
  // topology_exclusion(c, st) -- which passes a_rows = TileM -- ONCE, and then cross-multiplied the survivors by
  // g_compact_rows. ppu_tactic_space.hpp has compact_a_topology_exclusion(c, stages, compact_rows) with its own
  // static_asserts, and the emitter called it only from the --space=compare diagnostic. Emission never did.
  //
  // BOTH DIRECTIONS WERE WRONG, and the one that matters is the first:
  //   * A row that is illegal at a_rows=TileM can be LEGAL at a_rows=1, because per_stage_smem is
  //     a_rows*TileK*2 + ... and A is the dominant term (8192 of 11264 B at 16x16x256). Those rows were never
  //     emitted -- and they are exactly the extra reach compact A exists to buy at small M.
  //   * CompactAUnavailable was never applied. The compact-A reader lives only in the ordinary unfolded
  //     one-plane collective, so a grouped/two-plane table asked for capacities would have carried rows whose
  //     own collective rejects them.
  //
  // primary/guard still rank WITHIN one (stage, capacity) grid, for the reason stated above about TileK: a row
  // at one capacity must not suppress a row at another, which would be a pruning decision nobody made.
  for (int acr : g_compact_rows) {
    std::vector<Candidate> legal;
    for (auto const& c : ok)
      if ((acr == 0 ? Space::topology_exclusion(c, st)
                    : Space::compact_a_topology_exclusion(c, st, acr)) == Exclusion::None)
        legal.push_back(c);
    if (legal.empty()) continue;   // a capacity this format cannot host at all; say so below, do not emit rows

    for (auto const& c : legal) {
      bool const p = primary(legal, c);
      bool const g = h1_guard(legal, c) || n_geometry_guard(legal, c);
      // PRUNING IS A HEURISTIC, NOT LEGALITY, and it is off by default because this repository has twice paid
      // for confusing the two. primary/guard says "we expect this row not to win"; the exclusions in
      // ppu_tactic_space.hpp say "this row cannot be built". A row dropped here COMPILES AND RUNS -- the sweep
      // simply never sees it. The sub-four-warp quarantine hid the 65.7% dense optimum that way, and the 1/2/4
      // compact-A whitelist hid capacity 8. Measured today: pruning removes 532 of 1164 rows, 46%.
      if (g_prune && !p && !g) continue;
      if (rows.insert(Row{c.tm, c.tn, c.tactic_tile_k, c.wm, c.wn, st, acr}).second) { if (p) ++n_prim; else if (g) ++n_guard; else ++n_other; }
    }
  }
  }
  }

  std::printf("// GENERATED by benchmarks/emit_tactic_configs.cpp -- do not edit.\n");
  std::printf("//   space=%s bits=%d artifact_tile_k=%d   %zu configs (%d primary, %d guard, %d unranked) over %zu stage(s)\n",
              space_name, bits, artifact_tk, rows.size(), n_prim, n_guard, n_other, g_stages.size());
  std::printf("//   tactic_tile_k:");
  for (int k : tactic_tks) std::printf(" %d", k);
  std::printf("   <- the CONSUMER choices in this table. artifact_tile_k=%d is the resident layout and is NOT a\n"
              "//                     row field: one weight file, many readers.\n", artifact_tk);
  std::printf("//   stages:");
  for (int st : g_stages) std::printf(" %d", st);
  std::printf("   <- what this table COVERS; a winner outside it cannot be found by a sweep using it\n");
  std::printf("//   provenance: rows=%zu space_fnv1a64=%016llx emitter_fnv1a64=%016llx\n",
              rows.size(), static_cast<unsigned long long>(space_hash),
              static_cast<unsigned long long>(emitter_hash));
  std::printf("//\n");
  std::printf("// Regenerate after changing ppu_tactic_space.hpp or the pruning policy:\n");
  std::printf("//   c++ -std=c++17 -Iquactlize/include benchmarks/emit_tactic_configs.cpp -o /tmp/emit_tactic &&\\\n");
  std::printf("//   /tmp/emit_tactic %d %d --space=%s --tactic-tk=", bits, artifact_tk, space_name);
  for (size_t j = 0; j < tactic_tks.size(); ++j) std::printf("%s%d", j ? "," : "", tactic_tks[j]);
  std::printf(" --compact-rows=");
  for (size_t j = 0; j < g_compact_rows.size(); ++j) std::printf("%s%d", j ? "," : "", g_compact_rows[j]);
  std::printf(" --prune=%s", g_prune ? "primary-guard" : "none");
  if (g_format) std::printf(" --format=%s", g_format);
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
  // RENAMED FROM _CFG_TILEK. It was the only TileK there was; now it names the RESIDENT layout, and a row
  // carries the consumer's own. Keeping the old spelling would leave two different quantities under one name --
  // the exact shape that made a quarantined table read as a dense one earlier today.
  std::printf("#define %s_CFG_ARTIFACT_TILEK %d\n", macro_prefix, artifact_tk);
  std::printf("#define %s_CFG_ROWS  %zu\n", macro_prefix, rows.size());
  std::printf("#define %s_CFG_SPACE_FNV1A64   \"%016llx\"\n", macro_prefix,
              static_cast<unsigned long long>(space_hash));
  std::printf("#define %s_CFG_EMITTER_FNV1A64 \"%016llx\"\n\n", macro_prefix,
              static_cast<unsigned long long>(emitter_hash));
  std::printf("#define %s_CFG_LIST(X, B) \\\n", macro_prefix);
  size_t i = 0;
  for (auto const& r : rows) {
    // tm, tn, TacticTileK, wm, wn, stages -- in Row's order. An earlier edit added the %d and repeated
    // get<3> instead of shifting the tail, which emitted stages=wm and dropped stages entirely. It was visible
    // only because the printed row carried a stage value (16) that is not in the stage list.
    std::printf("  X(%d,%d,%d,%d,%d,%d,%d,B)%s\n", std::get<0>(r), std::get<1>(r), std::get<2>(r),
                std::get<3>(r), std::get<4>(r), std::get<5>(r), std::get<6>(r),
                ++i == rows.size() ? "" : " \\");
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
  int bits = std::atoi(argv[1]);
  const int tk   = std::atoi(argv[2]);

  char const* space = "dense";
  std::vector<int> stages;
  for (int i = 3; i < argc; ++i) {
    if (std::strncmp(argv[i], "--space=", 8) == 0) { space = argv[i] + 8; continue; }
    if (std::strncmp(argv[i], "--format=", 9) == 0) { g_format = argv[i] + 9; continue; }
    if (std::strcmp(argv[i], "--prune=primary-guard") == 0) { g_prune = true; continue; }
    if (std::strcmp(argv[i], "--prune=none") == 0) { g_prune = false; continue; }
    if (std::strncmp(argv[i], "--compact-rows=", 15) == 0) {
      g_compact_rows.clear();
      for (char const* s = argv[i] + 15; *s;) {
        g_compact_rows.push_back(std::atoi(s));
        while (*s && *s != ',') ++s;
        if (*s == ',') ++s;
      }
      continue;
    }
    if (std::strncmp(argv[i], "--tactic-tk=", 12) == 0) {
      g_tactic_tks.clear();
      for (char const* s = argv[i] + 12; *s;) {
        g_tactic_tks.push_back(std::atoi(s));
        while (*s && *s != ',') ++s;
        if (*s == ',') ++s;
      }
      continue;
    }
    stages.push_back(std::atoi(argv[i]));
  }
  if (!stages.empty()) g_stages = stages;

  // FORMAT SELECTION IS BY NAME, because a bit width is not a format's identity. kFormats holds six entries and
  // low_bits=2 is BOTH i2 and Q3_K (2+1); the old lookup disambiguated with `&& high_bits == 0`, which picked i2
  // and, as a side effect, made every TWO-PLANE format unreachable from this tool. That is why q3/q5/q6 have
  // never had a generated table on either operator, and why ArtifactHighRun, HighFoldDoesNotDivideTileN and
  // HighDelivery -- three exclusions written for the high plane -- have never been exercised by an emission.
  //
  // The old positional form still works and still means "the single-plane format with these bits". What is gone
  // is the silent part: asking for bits=2 while meaning Q3_K used to produce i2's table with a plausible row
  // count and no diagnostic.
  FormatSpec const* spec = nullptr;
  if (g_format) {
    for (auto const& s : kFormats)
      if (std::strcmp(s.name, g_format) == 0) spec = &s;
    if (!spec) {
      std::fprintf(stderr, "no format named '%s' in kFormats. Known:", g_format);
      for (auto const& s : kFormats) std::fprintf(stderr, " %s", s.name);
      std::fprintf(stderr, "\n");
      return 2;
    }
    bits = spec->low_bits;
  } else {
    for (auto const& s : kFormats)
      if (s.low_bits == bits && s.high_bits == 0) spec = &s;
    if (!spec) {
      std::fprintf(stderr, "no single-plane format with low_bits=%d in kFormats; for a two-plane format pass "
                           "--format=<name>\n", bits);
      return 2;
    }
  }

  if (g_tactic_tks.empty()) g_tactic_tks.push_back(tk);   // no --tactic-tk: the artifact's own, as before
  // No --compact-rows: capacity 0, ordinary unrestricted A. The DEFAULT IS THE OLD BEHAVIOUR ON PURPOSE -- the
  // axis exists in the table before anything searches it, so this table stays byte-identical to the previous one
  // apart from the field, and every config name the box already uses keeps working.
  if (g_compact_rows.empty()) g_compact_rows.push_back(0);

  if (std::strcmp(space, "dense") == 0)   return emit<DenseSpace>(*spec, bits, tk, g_tactic_tks, "dense", "LOWBIT_DENSE");
  if (std::strcmp(space, "grouped") == 0) {
    // ONE MACRO PREFIX PER FORMAT. Five grouped tables in one build all defining LOWBIT_GROUPED_CFG_LIST would
    // silently take whichever was included last -- and every one of them looks well-formed on its own, so the
    // collision has no diagnostic. The prefix carries the format when a format was named.
    static char pfx[64];
    if (g_format) {
      std::snprintf(pfx, sizeof pfx, "LOWBIT_GROUPED_%s", g_format);
      for (char* q = pfx; *q; ++q) *q = (*q >= 'a' && *q <= 'z') ? char(*q - 'a' + 'A') : *q;
    } else {
      std::snprintf(pfx, sizeof pfx, "LOWBIT_GROUPED");
    }
    return emit<GroupedSpace>(*spec, bits, tk, g_tactic_tks, "grouped", pfx);
  }
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
