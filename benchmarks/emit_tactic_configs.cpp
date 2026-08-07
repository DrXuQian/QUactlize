// EMIT A BENCH'S COMPILED CONFIG TABLE FROM THE SHARED TACTIC RULES -- for EITHER operator.
//
// WAS emit_dense_configs.cpp, then two template entries over duplicated DenseSpace/GroupedSpace wrappers. The
// legality generator is now one TacticSpace; --space selects only the durable table label and macro prefix. A route
// cannot grow a private pruning rule here: it would first have to break the alias invariant in ppu_tactic_space.hpp.
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
//   /tmp/emit_tactic <bits> <tile_k> [--space=dense|grouped] [stage ...]
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
// Off by default: emit every LEGAL row. --prune=primary-guard narrows to the heuristic's choice.
bool g_prune = false;
// --m-max=N declares "this table is for problems with M <= N", which lets one PRUNE be stated: at M <= N every
// TileM covers the whole problem in ONE m-tile (grid = ceil(M/TileM) * ceil(N/TileN), and ceil(M/TileM) is 1 for
// every TileM >= N), so a TileM larger than the smallest one that covers N buys no CTAs and only puts more warps
// on padding rows. kTileM starts at 16 because every MMA atom has M = 16.
//
// A PRUNE, NOT AN EXCLUSION, and the distinction is the one this file already argues for below: these rows COMPILE
// AND RUN. codex rejected an earlier version of exactly this idea on the grounds that TileM=128 "may benefit most
// from compacting" -- true, ordinary A at TileM=128 is where the wasted smem is largest, but the saving raises
// blocks/CU and blocks/CU is not what binds at M=1: D6's grid is 512 CTAs against the 792 slots capacity 0 already
// provides. Freed slots with no CTAs to put in them buy nothing.
//
// OFF BY DEFAULT AND IT MUST STAY OFF UNTIL THE SWEEP ANSWERS IT. benchmarks/sweep_real_shapes.py covers m=1 and
// m=4 across every row; that run is what says whether a TileM > 16 row ever wins at small M. Pruning first would
// destroy the evidence that would justify pruning. The dropped count is printed, because a sweep that silently
// covers less is the failure this repository keeps paying for.
int g_m_max = 0;
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
// tm, tn, TacticTileK, wm, wn, stages, PPU_B_CHUNK request.  The request is a row field because it changes the
// instantiated collective.  Whether it is effective remains a property of that collective's actual TiledMma and
// is deliberately not re-derived by this host-only emitter.
using Row = std::tuple<int, int, int, int, int, int, int>;

// Do not spell "largest legal" as min(TM,64). It happens to agree in today's WN<=64 producer domain, but WN=128
// can make WM64 fail the accumulator ceiling while WM32 remains legal. Derive both H1 rungs from the filtered set,
// which is the claim the policy actually makes and remains correct if producer reachability expands.
int largest_wm(std::vector<Candidate> const& legal, Candidate const& c, int below = 1 << 30) {
  int best = -1;
  for (auto const& q : legal)
    if (q.tm == c.tm && q.tn == c.tn && q.wn == c.wn && q.b_chunk == c.b_chunk &&
        q.wm < below && q.wm > best) best = q.wm;
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
std::vector<Candidate> legal_grid(FormatSpec const& spec, int tactic_tk, int artifact_tk) {
  // Legality FIRST and from the shared header, so the pruning policy only ever removes rows that could have
  // been built. A policy applied to an unfiltered grid would emit configurations that fail to compile.
  std::vector<Candidate> ok;
  for (int tm : kTileM)
    for (int tn : kTileN)
      for (int wm : kWarpM)
        for (int wn : kWarpN)
          for (int b_chunk : kBChunkModes) {
            Candidate const c{spec, tm, tn, tactic_tk, wm, wn, artifact_tk, b_chunk};
            if (TacticSpace::sweep_exclusion(c) != Exclusion::None) continue;
            ok.push_back(c);
          }
  return ok;
}

}  // namespace

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
  int n_prim = 0, n_guard = 0, n_other = 0, n_mdrop = 0;
  // PRIMARY/GUARD ARE DECIDED WITHIN ONE TacticTileK, not across them. primary() and the guards compare a
  // candidate against the OTHER legal candidates of the same grid; pooling two TileKs would let a row at one
  // TileK suppress a row at another, which is a pruning decision nobody made.
  std::vector<int> n_tkdrop;
  for (int tactic_tk : tactic_tks) {
    std::vector<Candidate> const ok = legal_grid(spec, tactic_tk, artifact_tk);
    // A LIST EMITS ITS LEGAL SUBSET. This used to `return 1` on the first member with no legal grid, which killed
    // the whole run and produced ZERO rows -- so `--tactic-tk=32,64,128,256` on an artifact_tk=64 format emitted
    // nothing at all rather than the 64/128/256 rows it obviously should. That is why no grouped table was ever
    // generated with a list: the one command anyone would try silently returns an empty file. Dense escaped it only
    // because someone happened to hand it a pre-filtered list (64,128,256).
    //
    // ArtifactTileK must tile TacticTileK, so which members are legal is a FUNCTION of the artifact, not something
    // the caller should have to know. Skipping is right; skipping SILENTLY is not -- a table that searched less has
    // to say so, same rule as --m-max.
    if (ok.empty()) {
      std::fprintf(stderr, "note: tactic_tk=%d has no legal grid at bits=%d artifact_tk=%d -- skipped\n",
                   tactic_tk, bits, artifact_tk);
      n_tkdrop.push_back(tactic_tk);
      continue;
    }
  for (int st : g_stages) {
  {
    std::vector<Candidate> legal;
    for (auto const& c : ok)
      if (TacticSpace::topology_exclusion(c, st) == Exclusion::None)
        legal.push_back(c);

    for (auto const& c : legal) {
      bool const p = primary(legal, c);
      bool const g = h1_guard(legal, c) || n_geometry_guard(legal, c);
      // PRUNING IS A HEURISTIC, NOT LEGALITY, and it is off by default because this repository has twice paid
      // for confusing the two. primary/guard says "we expect this row not to win"; the exclusions in
      // ppu_tactic_space.hpp say "this row cannot be built". A row dropped here COMPILES AND RUNS -- the sweep
      // simply never sees it. The sub-four-warp quarantine hid the 65.7% dense optimum that way, and the 1/2/4
      // compact-A whitelist hid capacity 8. Measured today: pruning removes 532 of 1164 rows, 46%.
      if (g_prune && !p && !g) continue;
      int const m_max = g_m_max;
      // The smallest TileM that still covers m_max in one tile; every larger one is the same grid with more
      // padding warps. Derived from kTileM rather than written as a number, so a new TileM is covered on arrival.
      if (m_max > 0) {
        int keep = kTileM.back();
        for (int tm : kTileM) if (tm >= m_max) { keep = tm; break; }
        if (c.tm > keep) { ++n_mdrop; continue; }
      }
      if (rows.insert(Row{c.tm, c.tn, c.tactic_tile_k, c.wm, c.wn, st, c.b_chunk}).second) {
        if (p) ++n_prim; else if (g) ++n_guard; else ++n_other;
      }
    }
  }
  }
  }

  std::printf("// GENERATED by benchmarks/emit_tactic_configs.cpp -- do not edit.\n");
  std::printf("//   space=%s bits=%d artifact_tile_k=%d   %zu configs (%d primary, %d guard, %d unranked) over %zu stage(s)\n",
              space_name, bits, artifact_tk, rows.size(), n_prim, n_guard, n_other, g_stages.size());
  // COVERAGE LOSS IS NEVER SILENT. --m-max drops rows that compile and run; a table that searched less has to say
  // so on its own face, because the reader of a sweep result has no other way to know the winner was excluded.
  if (n_mdrop > 0)
    std::printf("//   TileM prune dropped %d row(s) whose TileM exceeds the smallest that covers this table's M\n"
                "//                     bound (a capacity row's bound IS its capacity; --m-max sets it otherwise).\n"
                "//                     These are LEGAL rows a sweep using this table cannot find.\n", n_mdrop);
  // EVERY member illegal is still an error: an empty table is not a smaller search, it is no search.
  if (!n_tkdrop.empty() && n_tkdrop.size() == tactic_tks.size()) {
    std::fprintf(stderr, "no legal tactic_tk at all for bits=%d artifact_tk=%d -- refusing to emit an empty table\n",
                 bits, artifact_tk);
    return 1;
  }
  if (!n_tkdrop.empty()) {
    // An empty member used to mean only ArtifactTileK did not tile TacticTileK. Consumer-map and producer-map gates
    // can now empty a whole member too, so naming the old reason here would turn an honest coverage declaration into
    // a false diagnosis. The exact exclusions remain printable by emit_tactic_space; this table only needs to retain
    // the requested members so its provenance replay can reproduce the same legal subset.
    std::printf("//   tactic_tk SKIPPED (no legal grid under TacticSpace):");
    for (int k : n_tkdrop) std::printf(" %d", k);
    std::printf("\n");
  }
  std::printf("//   tactic_tile_k:");
  for (int k : tactic_tks) if (std::find(n_tkdrop.begin(), n_tkdrop.end(), k) == n_tkdrop.end()) std::printf(" %d", k);
  std::printf("   <- the CONSUMER choices in this table. artifact_tile_k=%d is the resident layout and is NOT a\n"
              "//                     row field: one weight file, many readers.\n", artifact_tk);
  std::printf("//   stages:");
  for (int st : g_stages) std::printf(" %d", st);
  std::printf("   <- what this table COVERS; a winner outside it cannot be found by a sweep using it\n");
  std::printf("//   ppu_b_chunk: 0 1   <- requested mode; each bench tag reports requested->actually effective\n");
  std::printf("//   provenance: rows=%zu space_fnv1a64=%016llx emitter_fnv1a64=%016llx\n",
              rows.size(), static_cast<unsigned long long>(space_hash),
              static_cast<unsigned long long>(emitter_hash));
  std::printf("//\n");
  std::printf("// Regenerate after changing ppu_tactic_space.hpp or the pruning policy:\n");
  std::printf("//   c++ -std=c++17 -Iquactlize/include benchmarks/emit_tactic_configs.cpp -o /tmp/emit_tactic &&\\\n");
  std::printf("//   /tmp/emit_tactic %d %d --space=%s --tactic-tk=", bits, artifact_tk, space_name);
  for (size_t j = 0; j < tactic_tks.size(); ++j) std::printf("%s%d", j ? "," : "", tactic_tks[j]);
  std::printf(" --prune=%s", g_prune ? "primary-guard" : "none");
  if (g_m_max > 0) std::printf(" --m-max=%d", g_m_max);
  if (g_format) std::printf(" --format=%s", g_format);
  for (int st : g_stages) std::printf(" %d", st);
  // A bounded-M table is the decode table, not the full table.  This suffix is part of the generated
  // maintenance contract: before it existed, every decode header printed a command that would overwrite the
  // corresponding full table even though the emitted bytes were the decode payload.
  char const* band_suffix = g_m_max > 0 ? "_decode" : "";
  if (g_format) std::printf(" > benchmarks/lowbit_%s_%s%s_configs.inc\n", space_name, g_format, band_suffix);
  else          std::printf(" > benchmarks/lowbit_%s%s_configs.inc\n", space_name, band_suffix);
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
    // tm, tn, TacticTileK, wm, wn, stages, PPU_B_CHUNK -- in Row's order. An earlier edit added the %d and repeated
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
                 "usage: emit_tactic_configs <bits:1|2|4> <tile_k> [--space=dense|grouped] [stage ...]\n"
                 "  --space defaults to dense and selects only the table label/prefix. stages default to 2 3 4.\n");
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
    if (std::strncmp(argv[i], "--m-max=", 8) == 0) { g_m_max = std::atoi(argv[i] + 8); continue; }
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

  // ONE MACRO PREFIX PER (SPACE, FORMAT). Several tables can be consumed by one build system, and every one is
  // well-formed in isolation, so sharing LOWBIT_DENSE_CFG_LIST or LOWBIT_GROUPED_CFG_LIST would silently let the
  // last include win. Keep the legacy generic dense prefix only for its sole i4 table; named formats carry a suffix.
  static char pfx[64];
  auto prefix = [&](char const* base) {
    if (g_format) std::snprintf(pfx, sizeof pfx, "%s_%s", base, g_format);
    else          std::snprintf(pfx, sizeof pfx, "%s", base);
    for (char* q = pfx; *q; ++q) *q = (*q >= 'a' && *q <= 'z') ? char(*q - 'a' + 'A') : *q;
    return pfx;
  };
  if (std::strcmp(space, "dense") == 0)
    return emit(*spec, bits, tk, g_tactic_tks, "dense", prefix("LOWBIT_DENSE"));
  if (std::strcmp(space, "grouped") == 0) {
    return emit(*spec, bits, tk, g_tactic_tks, "grouped", prefix("LOWBIT_GROUPED"));
  }
  std::fprintf(stderr, "unknown --space=%s (want dense or grouped)\n", space);
  return 2;
}
