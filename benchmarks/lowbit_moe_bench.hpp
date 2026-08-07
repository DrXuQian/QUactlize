// Shared harness for the MoE shape-band sweep. The sweep itself lives in one moe_bench_*.cu per format, NOT here.
//
// WHY SPLIT AT ALL. Every row of the sweep is a full kernel instantiation, and one .cu is ONE hgcc invocation, so a
// 200-row sweep in a single translation unit compiles strictly serially. cutlass_build_dev_kernels emits one
// add_custom_command per .cu (cmake/PPUToolchain.cmake), so N sources become N independent commands and `make -j`
// runs them concurrently. Splitting is the only way to instantiate MORE and wait LESS -- the alternative,
// instantiating less, is not on the table.
//
// The split is by (format, WarpN) rather than by format alone so the largest unit stays near the size of the old
// single-file sweep: q3/q5 have one legal WarpN so they are one file each; q6/i2/i4 have two, so they are two.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "xplane_offline.hpp"
#include "fold_traits.hpp"   // warps_per_cu_chunked: the occupancy model WITH the register term
#include "moe_grouped_ppu.cuh"
#include "moe_only_filter.hpp"
#include "moe_router_fixture.hpp"

using half_t  = cutlass::half_t;
// THE QUANT MODE, and it used to be hardcoded ScaleZero at BOTH launch sites -- so every low-bit number this harness
// has ever produced, decode band included, was an affine run. That was invisible because the 2-plane driver ignored
// QuantOp anyway (fixed: moe_grouped_ppu's NoZero slot), so asking for ScaleOnly changed nothing and nobody noticed.
// Default is unchanged. TO SELECT ScaleOnly, and this line exists because the switch had no recorded
// invocation anywhere and was therefore reported as a coverage gap it never was:
//   PPU_DEFS=LOWBIT_QMODE=1 TARGET=test_lowbit_moe_bench ./build.sh
// build.sh forwards PPU_DEFS to the DEVICE compile (CMakeLists.txt.in:11); target_compile_definitions alone
// would reach only the host half, which is the trap that comment further up in build.sh is about.
// LOWBIT_QMODE=1 selects ScaleOnly. The mode goes in the header line, because a row that cannot
// be told apart from the other mode is how "(ScaleOnly)" ended up labelling affine runs for months.
#if defined(LOWBIT_QMODE) && (LOWBIT_QMODE == 1)
#  define LOWBIT_QMODE_SEL QM::FinegrainedScaleOnly
#  define LOWBIT_QMODE_STR "ScaleOnly"
#else
#  define LOWBIT_QMODE_SEL QM::FinegrainedScaleZero
#  define LOWBIT_QMODE_STR "ScaleZero"
#endif

using int4_t  = cutlass::int4b_t;
using uint2_t = cutlass::uint2b_t;
using uint1_t = cutlass::uint1b_t;
using GS      = moe_grouped_ppu::GroupShape;
using DStride = moe_grouped_ppu::DStride;
using QM      = moe_grouped_ppu::QuantMode;

constexpr int moe_scale_groups(int k, int gs) { return (k + gs - 1) / gs; }
constexpr int moe_metadata_planes(QM mode) { return moe_grouped_ppu::has_zero(mode) ? 2 : 1; }
static_assert(moe_scale_groups(65, 32) == 3, "scale metadata covers a partial final group");
static_assert(moe_metadata_planes(QM::FinegrainedScaleOnly) == 1 &&
              moe_metadata_planes(QM::FinegrainedScaleZero) == 2,
              "ScaleOnly has no zero metadata plane");

static constexpr double PEAK    = 500.0e12;
// HBM peak for ppu001 (see the ppu-ptx skill). Used to turn the traffic model into a percentage, which is the only form
// in which "is this bandwidth-bound?" is answerable.
static constexpr double HBM_GBS = 2766.0;

// WHICH SOURCE DID THIS BINARY COMPILE AGAINST. Bump this whenever a macro in this header changes shape. The banner prints
// it, so a run can no longer be ambiguous about whether it used the fixed header -- a failed build showed a macro expansion
// containing `fold::FoldTraits` while the checked-out tree had `moe_fold`, and there was no way to tell from here whether
// the box had built an older commit or the overlay had a stale copy. That ambiguity has cost rounds twice in this work (the
// per-row PPU_B_CHUNK request/effective tag exists for the same reason), so it gets an invariant instead of a guess.
#define LOWBIT_MOE_BENCH_REV 17

// The table band is generated next to each target's dispatcher and included before this header. Other users of this
// shared harness (the split-K probe) remain explicitly unbanded rather than accidentally inheriting lowbit-MoE metadata.
#ifndef MOE_TABLE_BANDED
#define MOE_TABLE_BANDED 0
#define MOE_TABLE_DECODE 0
#define MOE_TABLE_M_MAX 0
#define MOE_TABLE_BAND_STR "unbanded"
#define MOE_TABLE_BENCH_STR "lowbit_moe"
#endif
static_assert((MOE_TABLE_DECODE != 0) == (MOE_TABLE_M_MAX > 0),
              "a decode table needs a positive Mmax and a full table must not carry one");

// TileK is a BUILD knob, not a row: it changes the per-plane fold factor, so sweeping it at runtime would mean packing
// every row twice. One extra build (PPU_DEFS=MOE_TK=128) covers it.
#ifndef MOE_TK
#define MOE_TK 64
#endif

// PPU_B_CHUNK is now a row axis: every generated unit fixes one requested value and every row tag reports both that
// request and the instantiated policy's effective atom-at-a-time conversion.  The tally verifies that both halves of
// the requested axis were linked; it is no longer a global-build consistency vote.
// DECODE A BROADCAST, as an A/B in ONE binary. A's m-stride goes to 0 so the TileM-1 padding rows read the
// expert's real row instead of 15 other rows -- legal only because the epilogue's residue mask discards them,
// and only when every expert has ONE row, which launch() enforces. Off by default: above Mmax == 1 it would be
// refused, and a sweep that silently refused half its rows is worse than one that never tried.
inline bool moe_abcast() { static int c = -1; if (c < 0) c = std::getenv("MOE_ABCAST") ? 1 : 0; return c != 0; }

struct MoeChunkTally { int requested_on = 0, requested_off = 0; };
inline MoeChunkTally& moe_chunk_tally() { static MoeChunkTally t; return t; }
inline void moe_chunk_vote(int request) {
  if (request != 0) ++moe_chunk_tally().requested_on; else ++moe_chunk_tally().requested_off;
}

// ROW SELECTION, for acu and for cheap re-runs. acu needs EXACTLY ONE kernel launch in the process, and the sweep issues
// 336 x 21 of them; there was no way to profile a single row without editing and rebuilding, which is how a cross-config
// mismatch gets introduced (the same trap FOLD_ONCE exists for in test_fold_int2).
//
//   MOE_ONLY=<substring>   run only rows whose tag contains it, e.g. MOE_ONLY="i2 64x128:64 w64x32 s3" or MOE_ONLY=i4
//   MOE_ACU=1              with MOE_ONLY, issue ONE launch instead of 1 warmup + 20 timed
//   MOE_VERBOSE=1          expand the default identity/candidate/verdict output with model and run explanations
//
// The shape-level test matches in BOTH directions on purpose: a full row tag ("i2 64x128:64 w64x32 s3") contains the shape
// string, and a loose filter ("i4") is contained BY it. Getting this backwards would silently pack nothing and print an
// empty sweep, which reads exactly like a broken build.
inline const char* moe_only() { static const char* v = std::getenv("MOE_ONLY"); return v; }
inline bool moe_acu() { static const bool v = std::getenv("MOE_ACU") != nullptr; return v; }
inline bool moe_verbose() {
  static const char* v = std::getenv("MOE_VERBOSE");
  return v != nullptr && std::strcmp(v, "1") == 0;
}
inline bool moe_row_selected(const char* tag) {
  return moe_only_filter::row_selected(tag, moe_only());
}
inline bool moe_shape_selected(const char* shape) {
  return moe_only_filter::shape_selected(shape, moe_only());
}

// Everything a sweep needs about the band. Passed by const reference so the per-format translation units share ONE
// set of device buffers instead of each allocating its own.
struct Band {
  int L, N, K, gs, Rows, mode, topk;
  int total, Mmax, scale_k, active;
  std::vector<int>   me, offs;
  std::vector<GS>    rsh;                 // host-side; filter_and_run takes a host pointer for this one
  half_t*   dA;   half_t*  dSc;  half_t* dZr;
  half_t**  pd;   DStride* sd;   GS*     rdev;  int* gm;  int* offdev;
  char*     ws;   size_t   wsb;
};

#include "bench_select.hpp"
#include "bench_samples.hpp"
#include "bench_floor.cuh"


// ---- sample emission (docs/BENCH_DESIGN.md) --------------------------------------------------------------
// The bench emits what it ran and what it measured; benchmarks/analyse.py decides. This is additive: with
// BENCH_JSONL unset nothing is written and every existing behaviour is unchanged.
//
// THE PASS INDEX IS A GLOBAL because the repetition happens in main's loop around moe_run_all() -- the whole
// candidate list is repeated rather than each candidate in place, so that drift lands on every candidate
// instead of on whichever ran during it. The macros that time a row are several frames below that loop and
// have no other way to know which pass they are in.
inline int& moe_pass() { static int p = 0; return p; }

// The distribution NAMED and VERSIONED. "ragged" is not reproducible and does not say how ragged; a name that
// changes when the generator changes is what stops an old log from being reinterpreted under new rules.
inline char const* moe_dist_name(int mode) {
  switch (mode) {
    case 0:  return "uniform-v1";
    case 1:  return "ragged-coarse-v1";      // multiples of TileM: never enters the masked path
    case 3:  return "decode-topk-1row-v1";
    case 4:  return moe_router_fixture::kName;
    default: return "skew-h8-v1";            // ~12% empty, 1-in-8 heavy tail, rest scattered off TileM
  }
}

inline bench_samples::Sample moe_identity(const Band& bd, char const* schema,
                                         int tm, int tn, int tk, int wm, int wn, int st,
                                         int bc, int bc_eff) {
  bench_samples::Sample s{};
  // The fixture is identified by its SHAPE and distribution rather than a hand-typed label: two runs of the
  // same shape must land in the same group, and a typo in a label would silently split them into two verdicts.
  static char name[128];
#if MOE_TABLE_BANDED
  std::snprintf(name, sizeof name, "moe-%s-m%d-n%d-k%d-gs%d-L%d-r%d-topk%d",
                MOE_TABLE_BAND_STR, MOE_TABLE_M_MAX, bd.N, bd.K, bd.gs, bd.L, bd.Rows, bd.topk);
#else
  std::snprintf(name, sizeof name, "moe-n%d-k%d-gs%d-L%d-r%d-topk%d",
                bd.N, bd.K, bd.gs, bd.L, bd.Rows, bd.topk);
#endif
  s.fixture = name;  s.dist = moe_dist_name(bd.mode);  s.schema = schema;
  s.n = bd.N; s.k = bd.K; s.gs = bd.gs;
  s.experts = bd.L; s.rows = bd.Rows; s.mmax = bd.Mmax;
  s.tm = tm; s.tn = tn; s.tk = tk; s.wm = wm; s.wn = wn; s.st = st;
  s.bc = bc; s.bc_eff = bc_eff;
  s.pass = moe_pass();
  return s;
}

// BEFORE THE LAUNCH, and it is not symmetric with moe_sample by accident: a device assert takes the whole
// process, and this bench reports a candidate only through report() AFTER the timing returns. Without this, the
// row that killed a sweep is unnamed and analyse.py's unfinished() has nothing to find. `us` is deliberately
// absent -- there is nothing measured yet.
inline void moe_attempt(const Band& bd, char const* schema,
                        int tm, int tn, int tk, int wm, int wn, int st, int bc, int bc_eff) {
  if (!bench_samples::enabled()) return;
  bench_samples::attempt(moe_identity(bd, schema, tm, tn, tk, wm, wn, st, bc, bc_eff));
}

inline void moe_sample(const Band& bd, char const* schema,
                       int tm, int tn, int tk, int wm, int wn, int st,
                       int bc, int bc_eff, double us) {
  if (!bench_samples::enabled()) return;
  bench_samples::Sample s = moe_identity(bd, schema, tm, tn, tk, wm, wn, st, bc, bc_eff);
  s.us = us;
  bench_samples::emit(s);
}

// iters == 0 means EXACTLY ONE launch and no warmup: acu attributes counters to the whole process, so a warmup would
// double-count and 20 iterations would make the report meaningless.
template <class F> inline double time_it(F&& f, int iters) {
  if (iters == 0) {
    auto a = std::chrono::high_resolution_clock::now();
    f(); CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    auto b = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(b - a).count();
  }
  f(); CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) f();
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

// F WITHOUT INSTANTIATING FoldTraits, and that distinction is the whole point. FoldTraits carries static_asserts --
// `delivery <= slots` among them -- so computing F from it fires those asserts for configurations moe_ok would have
// rejected. That is what broke the TileK=32 build: moe_ok correctly rejects int2 at WarpN=32 (slots = 32*32/32 = 32 against
// delivery = 128/2 = 64), but `constexpr int _F = FoldTraits<...>::F` sat ABOVE the `if constexpr (moe_ok<...>)` gate, and
// a template argument is instantiated whether or not a later gate would have discarded the statement. Same closed form
// here, no asserts; FoldTraits still fires for the configurations that actually launch, from inside the collective.
template <int Bits> constexpr int moe_fold(int TK) { const int c = TK * Bits / 8; return c >= 32 ? 1 : 32 / c; }

// THE TRAFFIC MODEL LIVES HERE, not in prose. I argued "not bandwidth-bound" from traffic computed by hand in a
// message; that number changes with every shape and tile, so the tool has to produce it or the argument is
// unfalsifiable.
//
// AND IT MUST BE A LOWER BOUND. The first version printed only the DMA-issued upper bound and reported 116-181% of
// HBM, which answers nothing: a bound looser than the hardware peak is not a measurement. Its largest looseness is A:
// `ceil(N/TN) * total * K * 2` assumes every n-tile column re-reads every REAL A row from DRAM, even though those CTAs
// read identical A and may hit L2. Padded rows are absent from both bounds because their global loads never issue.
//
//   floor  every byte crosses the bus AT LEAST once:  A = total_rows*K*2, B+S once per ACTIVE expert, D = rows*N*2
//   ceil   A once per n-tile, B+S once per m-tile, D once
//   noreuse = ceil/floor -- how much reuse the L2 must be supplying. A ratio, deliberately not a percentage.
//
// WHICH END IS CONCLUSIVE, and I had this BACKWARDS in the first version of this comment and in three messages built on it.
// Achieved traffic lies in [floor, ceil], so achieved bandwidth lies in [floor/t, ceil/t]:
//   * ceil/t  < peak  =>  DEFINITELY NOT saturated. CONCLUSIVE, and it is the ceiling that gives it.
//   * floor/t ~ peak  =>  DEFINITELY saturated. CONCLUSIVE the other way.
//   * a LOW floor/t says NOTHING -- the truth may sit at the ceiling. That is the error: "low floor means not
//     bandwidth-bound" is false, because the lower bound being small does not bound the actual from above.
// Both ends are printed now. And the strongest argument needs neither: if two configs of the same format differ by 1.4-2.8x
// in ceiling traffic and land within 0.4-6% in TIME, traffic is not the limiter -- that is what the decode run showed.
// OCCUPANCY IS PRINTED, not inferred. The decode measurement showed 256 CTAs and 128 CTAs of the same format landing within
// 0.4% of each other -- both fit in one wave, so the time is ONE CTA's K-loop latency and adding CTAs cannot touch it. What
// can is the number of tile loads in flight, which per CTA is exactly `Stages`, and per CU is `blocks/CU * Stages`. That
// quantity was never on screen, so the hypothesis could only be argued.
//
// `blk` COMES FROM fold::warps_per_cu_chunked, NOT from a formula written here. The first version of this used
// `min(262144/smem, 64/warps)` -- which fold_traits.hpp and the README both record as KNOWN-WRONG: acu said outright
// "theoretical occupancy (25.0%) is limited by the number of required registers", shared memory allowing 10 blocks where
// registers allowed 4, and that missing register term is why an earlier `blk` "kept failing to order the data". The correct
// model bills registers ROUNDED UP TO A POWER OF TWO (129 costs exactly what 256 costs) and is pinned by static_assert to
// acu's measured points. Re-deriving it here would have been the same mistake a second time.
// `wav` uses CU = 72, which acu confirmed independently through its own wave arithmetic (1024 - 3*288 = 160).
//
// AND `ifl` IS INVARIANT IN Stages WHENEVER smem BINDS, which it does at every decode shape here (blk_smem 19 < blk_warp 32
// for i4 (32,64,32) w32x32). Since blk = 256KB / (smem_per_stage * Stages), the product blk*Stages collapses to
// 256KB / smem_per_stage: i4 (32,64,32) gives ifl 76 at s4 and 72 at s12 -- deepening the pipeline does not add in-flight
// loads, it CONCENTRATES the same total into fewer CTAs and costs warps. So "more stages" and "more occupancy" pull opposite
// ways here, and the lever that actually raises the total is SHRINKING smem per stage: TileN=32 halves both the B and the
// scale term (3328 -> 2688 B, ifl <= 97). Dropping the zero tile (#20 Phase 1) is only 128 B of 3328, i.e. 3.8%.
// Which of depth-vs-width wins at a fixed total is exactly what the sweep is for.
// THE PADDED ROW COUNT, factored out because the VERDICT needs it and only report() had it. A summary that prints
// MFU without it is unreadable: MFU counts REAL rows (bd.total) while the kernel grinds TM*mt, so a fast kernel on a
// ragged band prints a low number and looks broken. That is exactly how a 226.63 us / 30.3% MFU row -- FASTER than
// the 246 us dense record that reads 55.8% -- came to look like a regression.
inline long long padded_mtiles(const Band& bd, int TM, long long* mt_max_out = nullptr) {
  long long mt = 0, mt_max = 0;
  for (int e = 0; e < bd.L; ++e) {
    const long long t = (bd.me[e] + TM - 1) / TM; mt += t; mt_max = std::max(mt_max, t);
  }
  if (mt_max_out) *mt_max_out = mt_max;
  return mt;
}
inline double masked_fraction(const Band& bd, int TM) {
  const long long mt = padded_mtiles(bd, TM);
  return mt ? 1.0 - double(bd.total) / (double(TM) * double(mt)) : 0.0;
}

inline void report(const Band& bd, const char* tag, double us, int TM, int TN, int TK, int WM, int WN, int Stages,
                   int bits_total, int wcu) {
  long long mt_max = 0;
  const long long mt = padded_mtiles(bd, TM, &mt_max);
  const double masked = masked_fraction(bd, TM);
  // skew = the heaviest expert's m-tile count against the mean. With a non-persistent scheduler this is what the
  // last-wave tail is made of, so it belongs next to the time rather than in a separate analysis.
  const double skew  = mt ? double(mt_max) / (double(mt) / double(bd.L)) : 0.0;
  const double ntile = std::ceil(double(bd.N) / double(TN));
  const double wb    = double(bd.N) * double(bd.K) * double(bits_total) / 8.0;         // one expert's weights
  const double scale_groups = double(moe_scale_groups(bd.K, bd.gs));
  constexpr int metadata_planes = moe_metadata_planes(LOWBIT_QMODE_SEL);
  const double sb = double(bd.N) * scale_groups * 2.0 * double(metadata_planes);       // fp16 scale [+ zero]
  // dfl IS THE DISTINCT-BYTE LOWER BOUND. Its A term is the real footprint `total*K*2`, not the padded compute area:
  // padding loads do not issue. The AIU path uses a 2-D `.padz` copy whose dim_h is the expert's real M; classic CUTLASS
  // predicates the out-of-bounds global load. Those rows are not alternate cache lines and are not bus traffic.
  //
  // At decode `mt == active` makes the B and S terms exact -- each active expert's weights and scale metadata are read
  // once. It does NOT lock total traffic: A may be fetched again by separate n-tile CTAs, so dfl remains a lower bound.
  // The ceiling may therefore count `a_dram` once per n-tile, but it must not reintroduce padded rows: those loads do
  // not issue regardless of how often the real A footprint misses cache.
  const double a_dram = double(bd.total) * double(bd.K) * 2.0;
  // HBM% IS DISTINCT BYTES OVER TIME. Nothing else. The previous version put `a_pad` -- mt*TM*K, the PADDED row
  // count -- in the numerator, so a config with a bigger TileM "moved more data" and printed a HIGHER bandwidth
  // while reading the same rows. That is how a 1-row-per-expert decode band reported more HBM traffic at TileM=128
  // than at TileM=16 off identical input. Padding rows are out of bounds and their loads are suppressed, so the bus
  // does not carry them.
  //
  //   A = total*K*2       every REAL row, once
  //   B = active*wb       every ACTIVE expert's weights, once
  //   S = active*sb       its scales, plus zeros only in ScaleZero mode, once
  //   C = total*N*2       the output rows that exist
  const double dfl    = a_dram + double(bd.active) * (wb + sb) + double(bd.total) * double(bd.N) * 2.0;
  const double dce    = ntile * a_dram + double(mt) * (wb + sb) + double(bd.total) * double(bd.N) * 2.0;
  const double gbs = dfl / (us * 1e-6) / 1e9;
  const double tf  = 2.0 * double(bd.total) * double(bd.N) * double(bd.K) / (us * 1e-6) / 1e12;
  // S's SHARE OF THE FLOOR is what #20 can possibly buy, and without it the floor cannot answer that question. At decode
  // (1 row per active expert) the weight term is read exactly once per active expert, so floor and ceiling coincide on B
  // and S and this share is a tight number rather than a bound.
  const double s_share = dfl > 0 ? double(bd.active) * sb / dfl : 0.0;
  const double gbs_c = dce / (us * 1e-6) / 1e9;
  // THE TWO QUANTITIES TileK CONTROLS, both candidates for why a memory-bound kernel misses its own roofline.
  //   run  = F*TK*bits/8, the AIU CONTIGUOUS run in bytes. The fold targets the AIU's 32 B minimum, which is right on
  //          prefill (it shrinks A-smem) and may be exactly wrong at decode: 24.8% of peak is suspiciously close to
  //          32/128, i.e. one 32 B run per 128 B line.
  //   kit  = K/TK, the k-iteration count. 64 dependent iterations at TK=32 against 8 at TK=256.
  // Both improve monotonically with TileK, so they are confounded across a TileK scan -- report them so the confound is
  // visible rather than discovered later.
  const int  fold_l = (TK * bits_total / 8) >= 32 ? 1 : 32 / (TK * bits_total / 8);
  const int  run_b  = fold_l * TK * bits_total / 8;
  const long kit    = bd.K / TK;
  const int  warps  = (TM / WM) * (TN / WN);
  const int  blk    = warps > 0 ? wcu / warps : 0;          // wcu comes from fold::warps_per_cu_chunked at the call site
  const long ctas   = mt * (long)ntile;
  const double waves = (blk > 0) ? double(ctas) / (72.0 * double(blk)) : 0.0;   // CU = 72, confirmed by acu wave arithmetic
  // ACHIEVED occupancy is set by the GRID, not by the per-CU limits, and the identity says which knobs can move it:
  //     grid warps = ctas * warps/blk = [mt * N/TN] * [(TM/WM)(TN/WN)] = mt * N * TM / (WM * WN)
  // TileN and TileK CANCEL. At the decode winner that is 8*2048*16/(16*32) = 512 warps over 72 CUs = 7.1 warps/CU, and acu
  // measured achieved 6.97 -- the identity closes. So the only levers are mt (fixed by the router), TM/WM, and WN; smem caps
  // THEORETICAL occupancy (acu: 21.88%, Block Limit Shared Mem 7) while ACHIEVED (10.90%) is grid-limited, which is why
  // shrinking the scale or zero tile raises a ceiling that is not being reached.
  const double gwarps = double(mt) * double(bd.N) * double(TM) / (double(WM) * double(WN));
  const double wcu_grid = gwarps / 72.0;
  std::printf("    %-30s %8.2f us | %6.1f TF/s (%4.1f%% MFU) | mt=%-5lld msk=%4.1f%% skw=%.1fx |"
              " %6.0f GB/s (HBM %4.1f%%) S=%4.1f%% | blk %-2d wrp/CU %-3d grid_wrp/CU %5.1f cta=%-5ld wav=%4.2f run=%-3dB kit=%-3ld%s\n",
              tag, us, tf, 100.0 * tf * 1e12 / PEAK, mt, 100.0 * masked, skew,
              gbs, 100.0 * gbs / HBM_GBS, 100.0 * s_share,
              blk, blk * warps, wcu_grid, ctas, waves, run_b, kit,
              gbs_c < 0.9 * HBM_GBS ? "  NOT-BW" : "");
}

// DID THIS ROW ACTUALLY RUN? Only a launch refusal excludes it: moeg_fail_count() grows when can_implement, workspace or
// initialize refuses the launch, and that signal is independent of cache state. The old second net divided one expert's
// weights by the per-iteration time and excluded a row above the HBM peak. time_it() warms the SAME buffers before timing,
// so those bytes can be cache-hot; the derived rate is useful evidence of a suspicious timing, but not evidence that the
// kernel did not run. Keep the warning and, critically, keep the row in report(), samples and the winner verdict.
inline bool moe_row_ran(const Band& bd, const char* tag, double us, int fail0, int bits_total) {
  const bool refused = moe_grouped_ppu::moeg_fail_count() > fail0;
  if (refused) {
    std::printf("    %-30s %8.2f us | DID NOT RUN (launch refused) -- excluded from the verdict\n", tag, us);
    return false;
  }
  const double wb  = double(bd.N) * double(bd.K) * double(bits_total) / 8.0;
  const double gbs = (double(bd.active) * wb) / (us * 1e-6) / 1e9;
  if (gbs > HBM_GBS) {
    std::printf("    %-30s %8.2f us | WARNING cache-hot weight rate %.0f GB/s > HBM peak -- row retained\n",
                tag, us, gbs);
  }
  return true;
}

// ---------------------------------------------------------------------------------------------------------------
// LEGALITY, as one constexpr predicate rather than as a hand-maintained row list.
//
// The delivery bound is per PLANE and the binding plane is the SPARSEST one: one swzl delivery hands a thread a fixed
// 16 B = 128/bits codes, the fp16 B fragment has slots = WN*TK/32, and over-delivery silently drops the surplus. So
// every plane needs WN*TK*bits >= 4096, i.e. the minimum over the planes decides. That is why Q3/Q5 (an int1 plane)
// are pinned to WN=64 at TK=64 while Q6 (int4+int2) may run WN=32 -- a fact this file's own section headers assert and
// the box has measured, so the predicate is cross-checked against something other than my own derivation.
//
// smem: A is TM*TK*2 per stage; each plane's B is Ng_p * (F_p*TK*bits_p/8) and Ng_p = TN/F_p, so the fold CANCELS and
// B is simply TN*TK*bits_total/8 whatever the folding does. 256 KB per CU is a hard cap (not configurable higher).
template <int TM, int TN, int TK, int WM, int WN, int Stages, int BitsLo, int BitsHi = 0>
constexpr bool moe_ok() {
  // The old arithmetic lived here and the host-side tactic emitter copied it. That would let the two disagree while
  // both still compiled. The launcher, this gate and the emitter now ask GroupedSpace for the same answer.
  constexpr ppu_tactics::Candidate c{{ppu_tactics::Format::I2, "synthetic", BitsLo, BitsHi},
                                      TM, TN, TK, WM, WN};
  return ppu_tactics::GroupedSpace::topology_exclusion(c, Stages) == ppu_tactics::Exclusion::None;
}

// The request macro is not the verdict. Ordinary collectives have no kBChunk member, folded collectives apply a
// TiledMma-dependent gate, and the two-plane collective owns a third implementation. MainloopPolicy's descriptor
// folds those cases into one witness without making this bench duplicate any of their predicates.
template <int TM, int TN, int TK, int WM, int WN, int Stages, class ElementB, class PlaneB2 = void>
constexpr bool moe_b_chunk_effective() {
  using TileShape = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;
  using ScaleShape = cute::Shape<cute::Int<TN>,
      cute::Int<ppu_group_schedule::scale_groups_v<TK, 32>>>;
  using WarpShape = cute::Shape<cute::Int<WM>, cute::Int<WN>, cute::Int<TK>>;
  using Policy = moe_grouped_ppu::MixedMainloopPolicy<
      LOWBIT_QMODE_SEL, ppu_group_schedule::FinegrainedSchedule<32>, TileShape, ScaleShape, WarpShape,
      Stages, true, ElementB, PlaneB2, TK>;
  return Policy::Descriptor::atom_at_a_time;
}

// (The hand-written (TileM, WarpM) grid that used to sit here is gone. One unit fixes one shape, so TileM and WarpM are
// generated axes now, and moe_ok's WM <= TM rejects the illegal corner -- the L-shape is a CONSEQUENCE of the predicate
// rather than a list someone maintains. The five points it used to enumerate were missing (128,32) and (256,32).)

// STAGES IS AN AXIS, and it was not one. It was baked into each hand-written row, which left int4's single-plane
// winner at s3 and int2's IDENTICAL shape at s2 -- so `i4 382.76 vs i2 420.83` was not a format comparison, it was s3
// vs s2. On dense the same tile moved 1.46x between s2 and s3. 4 is included because it is the toolchain's default
// and therefore the value someone gets by accident.
// DEEPER THAN 4, because `Stages` IS the per-CTA count of tile loads in flight and the decode band is latency-bound with a
// tiny tile: i4 (32,64,32) is ~3.25 KB/stage, so s16 costs 52 KB of the 256 KB budget. The collective caps nothing above
// `Stages >= 2`; smem does, and moe_ok's smem predicate filters the rest, so these cost nothing on the prefill shapes where
// A-smem is 8-32 KB/stage and they simply do not fit.
// MOE_STAGES narrows this axis from the build, for the same reason the tile lists can be narrowed: the cost is entirely
// front-end template instantiation at ~3.0 s per kernel, so halving the stage list halves the per-unit time.
//   PPU_DEFS="PPU_B_CHUNK=1 MOE_STAGES_4 MOE_STAGES_8"   -- define the ones you want; none defined = all six
#if defined(MOE_STAGES_2) || defined(MOE_STAGES_3) || defined(MOE_STAGES_4) || \
    defined(MOE_STAGES_6) || defined(MOE_STAGES_8) || defined(MOE_STAGES_12)
   // at least one was requested: leave the others undefined so only the requested stages are emitted
#else
#  define MOE_STAGES_2
#  define MOE_STAGES_3
#  define MOE_STAGES_4
#  define MOE_STAGES_6
#  define MOE_STAGES_8
#  define MOE_STAGES_12
#endif
#define MOE_STAGE_LIST(F, ...)                                                                 \
  MOE_STG_2(F, __VA_ARGS__)  MOE_STG_3(F, __VA_ARGS__)  MOE_STG_4(F, __VA_ARGS__)              \
  MOE_STG_6(F, __VA_ARGS__)  MOE_STG_8(F, __VA_ARGS__)  MOE_STG_12(F, __VA_ARGS__)
#ifdef MOE_STAGES_2
#  define MOE_STG_2(F, ...)  F(__VA_ARGS__, 2)
#else
#  define MOE_STG_2(F, ...)
#endif
#ifdef MOE_STAGES_3
#  define MOE_STG_3(F, ...)  F(__VA_ARGS__, 3)
#else
#  define MOE_STG_3(F, ...)
#endif
#ifdef MOE_STAGES_4
#  define MOE_STG_4(F, ...)  F(__VA_ARGS__, 4)
#else
#  define MOE_STG_4(F, ...)
#endif
#ifdef MOE_STAGES_6
#  define MOE_STG_6(F, ...)  F(__VA_ARGS__, 6)
#else
#  define MOE_STG_6(F, ...)
#endif
#ifdef MOE_STAGES_8
#  define MOE_STG_8(F, ...)  F(__VA_ARGS__, 8)
#else
#  define MOE_STG_8(F, ...)
#endif
#ifdef MOE_STAGES_12
#  define MOE_STG_12(F, ...)  F(__VA_ARGS__, 12)
#else
#  define MOE_STG_12(F, ...)
#endif

// ---- two-plane: pack ONCE per shape, then time every stage count against the same device buffer.
//
// The pack used to run per EXPERT, 64 times, on byte-identical input -- the weight pattern does not depend on e. At
// N=K=2048 that is 268 M positions per row for one row's worth of information. Pack expert 0 and memcpy the rest; the
// sweep only became affordable at this size because of it.
#define MOE2_TIME(BD,BEST,NAME,LOELEM,HIELEM,LOB,HIB,TMv,TNv,TKv,WMv,WNv,Sv)                                       \
  if constexpr (moe_ok<TMv,TNv,TKv,WMv,WNv,Sv,LOB,HIB>()) {                                                        \
    constexpr bool _bc = moe_b_chunk_effective<TMv,TNv,TKv,WMv,WNv,Sv,LOELEM,HIELEM>();                            \
    char _t[80]; moe_only_filter::format_tag(_t, sizeof _t, NAME, TMv, TNv, TKv, WMv, WNv, Sv,                      \
                                              int(UNIT_B_CHUNK), int(_bc), moe_abcast());                            \
    if (moe_row_selected(_t)) {                                                                                    \
      (BEST).any_selected = true;                                                                                  \
      auto _go = [&]{                                                                                              \
        moe_grouped_ppu::filter_and_run<LOWBIT_QMODE_SEL,TMv,TNv,TKv,WMv,WNv,Sv,LOELEM,HIELEM>(            \
            (BD).dA, _b1.get(), (BD).dSc, (BD).dZr, (BD).pd, (BD).sd, (BD).gm,                                     \
            (BD).Mmax, (BD).N, (BD).K, (BD).L, (BD).gs, (BD).rdev, (BD).rsh.data(),                                \
            (BD).mode ? (BD).offdev : nullptr, (BD).ws, (BD).wsb, nullptr, _b2.get(),                                \
            /*k_full=*/-1, /*prefix_ready=*/false, /*splitk=*/1, moe_abcast()); };                                  \
      double u; const int _f0 = moe_grouped_ppu::moeg_fail_count();                                                \
      moe_attempt(BD, NAME, TMv, TNv, TKv, WMv, WNv, Sv, int(UNIT_B_CHUNK), int(_bc));                             \
      std::printf("  -> %s\n", _t); std::fflush(stdout);                                                           \
      if (moe_acu()) { u = time_it(_go, 0); if (moe_verbose()) std::printf("  [acu] ONE COLD launch (not a timing): %s\n", _t); }       \
      else             u = time_it(_go, 20);                                                                       \
      constexpr int _wcu = _bc                                                                                     \
          ? fold::warps_per_cu_chunked<TMv,TNv,TKv,WMv,WNv,Sv,(LOB)+(HIB),32,true>                                 \
          : fold::warps_per_cu<TMv,TNv,TKv,WMv,WNv,Sv,(LOB)+(HIB),32,true>;                                        \
      if (moe_row_ran(BD, _t, u, _f0, (LOB)+(HIB))) { report(BD,_t,u,TMv,TNv,TKv,WMv,WNv,Sv,(LOB)+(HIB),_wcu); upd(BEST, _t, u); moe_sample(BD, NAME, TMv, TNv, TKv, WMv, WNv, Sv, int(UNIT_B_CHUNK), int(_bc), u); } \
    }                                                                                                              \
  }

#define MOE2(BD,BEST,NAME,LOELEM,HIELEM,LOB,HIB,TMv,TNv,TKv,WMv,WNv) do {                                          \
  char _sh[80]; moe_only_filter::format_shape(_sh, sizeof _sh, NAME, TMv, TNv, TKv, WMv, WNv);                     \
  if constexpr (moe_ok<TMv,TNv,TKv,WMv,WNv,2,LOB,HIB>()) if (moe_shape_selected(_sh)) {                            \
    constexpr int _F1 = moe_fold<LOB>(TKv);                                                                        \
    constexpr int _F2 = moe_fold<HIB>(TKv);                                                                        \
    const size_t _lo = (size_t)(BD).K*(BD).N*(LOB)/8, _hi = (size_t)(BD).K*(BD).N*(HIB)/8;                          \
    std::vector<int8_t> _blo((size_t)(BD).L*_lo), _bhi((size_t)(BD).L*_hi);                                        \
    { std::vector<uint8_t> _l((size_t)(BD).K*(BD).N), _h((size_t)(BD).K*(BD).N);                                    \
      for (size_t i = 0; i < _l.size(); ++i) {                                                                     \
        const int q = int((i * 2654435761u >> 5) % (unsigned)((1u<<(LOB))<<(HIB)));                                 \
        _l[i] = uint8_t(q & ((1u<<(LOB))-1u)); _h[i] = uint8_t(q >> (LOB)); }                                       \
      xplane::place_derived<LOB,TMv,TNv,TKv,WMv,WNv,_F1>(_blo.data(), _l, (BD).N, (BD).K);                          \
      xplane::place_hi<LOB,HIB,TMv,TNv,TKv,WMv,WNv,_F2,_F1>(_bhi.data(), _h, (BD).N, (BD).K);                       \
      for (int e = 1; e < (BD).L; ++e) {                                                                           \
        std::memcpy(_blo.data() + (size_t)e*_lo, _blo.data(), _lo);                                                 \
        std::memcpy(_bhi.data() + (size_t)e*_hi, _bhi.data(), _hi); } }                                            \
    cutlass::DeviceAllocation<LOELEM> _b1((size_t)(BD).L*_lo);                                                     \
    _b1.copy_from_host(reinterpret_cast<LOELEM const*>(_blo.data()));                                              \
    cutlass::DeviceAllocation<HIELEM> _b2((size_t)(BD).L*_hi);                                                     \
    _b2.copy_from_host(reinterpret_cast<HIELEM const*>(_bhi.data()));                                              \
    MOE_STAGE_LIST(MOE2_TIME, BD,BEST,NAME,LOELEM,HIELEM,LOB,HIB,TMv,TNv,TKv,WMv,WNv)                              \
  } } while (0)

// ---- single plane, same structure.
#define MOE1_TIME(BD,BEST,NAME,ELEM,BITS,TMv,TNv,TKv,WMv,WNv,Sv)                                                   \
  if constexpr (moe_ok<TMv,TNv,TKv,WMv,WNv,Sv,BITS>()) {                                                           \
    constexpr bool _bc = moe_b_chunk_effective<TMv,TNv,TKv,WMv,WNv,Sv,ELEM>();                                     \
    char _t[80]; moe_only_filter::format_tag(_t, sizeof _t, NAME, TMv, TNv, TKv, WMv, WNv, Sv,                      \
                                              int(UNIT_B_CHUNK), int(_bc), moe_abcast());                            \
    if (moe_row_selected(_t)) {                                                                                    \
      (BEST).any_selected = true;                                                                                  \
      auto _go = [&]{                                                                                              \
        /* PlaneB2 is NAMED (void) on purpose: passing nullptr for B2 while letting PlaneB2 deduce makes the    \
           deduction fail on std::nullptr_t, and a failed deduction is NOT rescued by the default template       \
           argument -- the call simply stops matching. */                                                        \
        moe_grouped_ppu::filter_and_run<LOWBIT_QMODE_SEL,TMv,TNv,TKv,WMv,WNv,Sv,ELEM,void>(                \
            (BD).dA, _db.get(), (BD).dSc, (BD).dZr, (BD).pd, (BD).sd, (BD).gm,                                     \
            (BD).Mmax, (BD).N, (BD).K, (BD).L, (BD).gs, (BD).rdev, (BD).rsh.data(),                                \
            (BD).mode ? (BD).offdev : nullptr, (BD).ws, (BD).wsb, nullptr,                                          \
            /*B2=*/nullptr, /*k_full=*/-1, /*prefix_ready=*/false, /*splitk=*/1, moe_abcast()); };                   \
      double u; const int _f0 = moe_grouped_ppu::moeg_fail_count();                                                \
      moe_attempt(BD, NAME, TMv, TNv, TKv, WMv, WNv, Sv, int(UNIT_B_CHUNK), int(_bc));                             \
      std::printf("  -> %s\n", _t); std::fflush(stdout);                                                           \
      if (moe_acu()) { u = time_it(_go, 0); if (moe_verbose()) std::printf("  [acu] ONE COLD launch (not a timing): %s\n", _t); }       \
      else             u = time_it(_go, 20);                                                                       \
      constexpr int _wcu = _bc                                                                                     \
          ? fold::warps_per_cu_chunked<TMv,TNv,TKv,WMv,WNv,Sv,(BITS),32,true>                                      \
          : fold::warps_per_cu<TMv,TNv,TKv,WMv,WNv,Sv,(BITS),32,true>;                                             \
      if (moe_row_ran(BD, _t, u, _f0, (BITS))) { report(BD,_t,u,TMv,TNv,TKv,WMv,WNv,Sv,(BITS),_wcu); upd(BEST, _t, u); moe_sample(BD, NAME, TMv, TNv, TKv, WMv, WNv, Sv, int(UNIT_B_CHUNK), int(_bc), u); } \
    }                                                                                                              \
  }

#define MOE1(BD,BEST,NAME,ELEM,BITS,TMv,TNv,TKv,WMv,WNv) do {                                                      \
  char _sh[80]; moe_only_filter::format_shape(_sh, sizeof _sh, NAME, TMv, TNv, TKv, WMv, WNv);                     \
  if constexpr (moe_ok<TMv,TNv,TKv,WMv,WNv,2,BITS>()) if (moe_shape_selected(_sh)) {                               \
    constexpr int _F = moe_fold<BITS>(TKv);                                                                        \
    const size_t _per = (size_t)(BD).K*(BD).N*(BITS)/8;                                                            \
    std::vector<int8_t> _bb((size_t)(BD).L*_per);                                                                  \
    { std::vector<uint8_t> _q((size_t)(BD).K*(BD).N);                                                              \
      for (size_t i = 0; i < _q.size(); ++i) _q[i] = uint8_t((i * 2654435761u >> 5) & ((1u<<(BITS))-1u));           \
      xplane::place_derived<BITS,TMv,TNv,TKv,WMv,WNv,_F>(_bb.data(), _q, (BD).N, (BD).K);                           \
      for (int e = 1; e < (BD).L; ++e) std::memcpy(_bb.data() + (size_t)e*_per, _bb.data(), _per); }                \
    cutlass::DeviceAllocation<ELEM> _db((size_t)(BD).L*_per);                                                      \
    _db.copy_from_host(reinterpret_cast<ELEM const*>(_bb.data()));                                                 \
    MOE_STAGE_LIST(MOE1_TIME, BD,BEST,NAME,ELEM,BITS,TMv,TNv,TKv,WMv,WNv)                                          \
  } } while (0)

// The unit entry points are DECLARED IN GENERATED CODE (moe_bench_units.inc, written by CMakeLists), one per shape.
// Nothing is declared here on purpose: 128 hand-written declarations would be a second copy of the enumeration, and the
// whole reason the sweep is generated is that the enumeration must live in exactly one place.
