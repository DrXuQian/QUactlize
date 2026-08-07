#pragma once
// SHARED BENCH MEASUREMENT, plus the temporary in-process C++ selection.
//
// Only the SELECTION section is scheduled for deletion (docs/BENCH_DESIGN.md step 3). It is kept until
// benchmarks/analyse.py is shown to reproduce it on the same samples; deleting first would leave a window in
// which nothing produces a verdict. benchmarks/xcheck_select.cpp runs that code and the analyser over one
// planted file and compares, so "reproduces it" is a check rather than a belief.
//
// The measurement section is the permanent common contract for dense and MoE: one machine peak, one useful-
// FLOP/MFU expression, one distinct-vs-tile traffic model, one config tag, and one repetitions reader. Keeping
// those quantities here prevents two harnesses from printing the same labels for different arithmetic.
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace bench_measure {

inline constexpr double kPeakFlopsPerSecond = 500.0e12;
inline constexpr double kHbmGBPerSecond = 2766.0;
inline constexpr std::size_t kTagBytes = 96;

// The row identity is structured first and rendered second. Report/model code must never recover TM/TN/etc.
// by parsing a display string: the MoE verdict did that with "%dx" even though its tag starts with "i4 ", so
// its masked fraction silently became zero for every winning row.
struct Tactic {
  char const* schema = nullptr;
  int tm = 0, tn = 0, tk = 0, wm = 0, wn = 0, stages = 0;
  int bc_requested = 0, bc_effective = 0;
  bool abcast = false;
};

// Canonical public syntax is the existing MOE_ONLY syntax. `schema == nullptr` is the dense spelling; the
// geometry and every row axis are otherwise byte-for-byte the same. Dense's old compact selector remains an
// accepted input alias in its caller, but all new reports and cache writes use this form.
inline int format_tag(char* out, std::size_t cap, Tactic const& c) {
  if (c.schema && *c.schema)
    return std::snprintf(out, cap, "%s %dx%d:%d w%dx%d s%d bc%d->%d%s",
                         c.schema, c.tm, c.tn, c.tk, c.wm, c.wn, c.stages,
                         c.bc_requested, c.bc_effective, c.abcast ? " B" : "");
  return std::snprintf(out, cap, "%dx%d:%d w%dx%d s%d bc%d->%d%s",
                       c.tm, c.tn, c.tk, c.wm, c.wn, c.stages,
                       c.bc_requested, c.bc_effective, c.abcast ? " B" : "");
}

struct ByteTerms {
  double activations = 0.0;
  double weights = 0.0;
  double metadata = 0.0;
  double output = 0.0;
  constexpr double total() const { return activations + weights + metadata + output; }
};

// A and D are problem-wide footprints. Weight/metadata are one resident copy (one dense matrix or one expert),
// with explicit copy counts for the distinct lower bound and for the no-cache tile request. This represents both
// existing models without erasing dense's useful tile/reuse question:
//
//   dense: distinct resident copies=1, tile resident copies=ceil(M/TM)
//   MoE:   distinct resident copies=active experts, tile resident copies=sum ceil(M_e/TM)
struct TiledGemmTrafficInput {
  double activation_bytes = 0.0;
  double weight_bytes_per_copy = 0.0;
  double metadata_bytes_per_copy = 0.0;
  double output_bytes = 0.0;
  double distinct_resident_copies = 0.0;
  double tile_activation_copies = 0.0;
  double tile_resident_copies = 0.0;
};

struct Traffic {
  ByteTerms distinct;  // every unique byte at least once: the only numerator that may be called %HBM
  ByteTerms tile;      // tile-level requests with no cache reuse: a ceiling, reported as GB/s + reuse
};

inline Traffic make_traffic(TiledGemmTrafficInput const& i) {
  return {
    {i.activation_bytes,
     i.distinct_resident_copies * i.weight_bytes_per_copy,
     i.distinct_resident_copies * i.metadata_bytes_per_copy,
     i.output_bytes},
    {i.tile_activation_copies * i.activation_bytes,
     i.tile_resident_copies * i.weight_bytes_per_copy,
     i.tile_resident_copies * i.metadata_bytes_per_copy,
     i.output_bytes}
  };
}

inline double gbs(double bytes, double us) {
  return us > 0.0 ? bytes / (us * 1.0e-6) / 1.0e9 : 0.0;
}

inline double hbm_pct(double achieved_gbs) {
  return 100.0 * achieved_gbs / kHbmGBPerSecond;
}

struct Compute {
  double useful_flops = 0.0;
  double tflops = 0.0;
  double mfu_pct = 0.0;
};

inline double mfu_pct(double tflops) {
  return 100.0 * tflops * 1.0e12 / kPeakFlopsPerSecond;
}

inline Compute compute(double useful_flops, double us) {
  const double tflops = us > 0.0 ? useful_flops / (us * 1.0e-6) / 1.0e12 : 0.0;
  return {useful_flops, tflops, mfu_pct(tflops)};
}

struct HbmModel {
  Traffic traffic;
  double distinct_bytes = 0.0;
  double tile_bytes = 0.0;
  double distinct_gbs = 0.0;
  double distinct_hbm_pct = 0.0;
  double tile_gbs = 0.0;
  double tile_reuse = 0.0;
  double distinct_metadata_share = 0.0;
  bool tile_l2_served = false;
};

inline HbmModel hbm(Traffic const& traffic, double us) {
  const double distinct_bytes = traffic.distinct.total();
  const double tile_bytes = traffic.tile.total();
  const double distinct_gbs = gbs(distinct_bytes, us);
  const double tile_gbs = gbs(tile_bytes, us);
  return {traffic, distinct_bytes, tile_bytes, distinct_gbs, hbm_pct(distinct_gbs), tile_gbs,
          distinct_bytes > 0.0 ? tile_bytes / distinct_bytes : 0.0,
          distinct_bytes > 0.0 ? traffic.distinct.metadata / distinct_bytes : 0.0,
          tile_gbs > kHbmGBPerSecond};
}

struct Metrics {
  double us = 0.0;
  Compute compute;
  HbmModel hbm;
};

inline Metrics measure(double us, double useful_flops, Traffic const& traffic) {
  return {us, compute(useful_flops, us), hbm(traffic, us)};
}

inline double ridge_flops_per_byte() {
  return kPeakFlopsPerSecond / (kHbmGBPerSecond * 1.0e9);
}

// Common numeric fragment. Tile traffic is intentionally NOT divided by HBM: it includes cache-served re-reads.
// A tile rate above the DRAM peak therefore says "L2-served", not ">100% HBM".
inline int format_metrics(char* out, std::size_t cap, Metrics const& m) {
  return std::snprintf(out, cap,
      "%6.1f TF/s (%4.1f%% MFU) | distinct %6.0f GB/s (%4.1f%% HBM) | "
      "tile %6.0f GB/s (%.1fx distinct%s)",
      m.compute.tflops, m.compute.mfu_pct,
      m.hbm.distinct_gbs, m.hbm.distinct_hbm_pct,
      m.hbm.tile_gbs, m.hbm.tile_reuse, m.hbm.tile_l2_served ? ", L2-served" : "");
}

// One reader, with an optional bench-local override. Dense calls read_reps(); MoE calls read_reps("MOE_REPS"),
// preserving MOE_REPS > BENCH_REPS without making an exported MOE_REPS accidentally alter a dense run.
inline int read_reps(char const* local_override_env = nullptr) {
  char const* e = (local_override_env && *local_override_env) ? std::getenv(local_override_env) : nullptr;
  if (e == nullptr || *e == '\0') e = std::getenv("BENCH_REPS");
  const int r = (e && *e) ? std::atoi(e) : 1;
  return r < 1 ? 1 : r;
}

}  // namespace bench_measure

// ================= SELECTION ==================================================================================
// A WINNER PICKED FROM ONE TIMING IS PICKED FROM NOISE. The recorded cross-run spread for one configuration on
// this machine is 13%, and the old rule here was `if (u < b.us)` over a single measurement per candidate -- so any
// two configurations within 13% of each other were being ordered by whichever happened to run in a good moment.
// The sweep exists to separate configurations that differ by less than that.
//
// THREE CHOICES, EACH LOAD-BEARING:
//
//  * REPEAT THE WHOLE CANDIDATE LIST, do not repeat each candidate in place. Clock drift, thermal state and
//    another process arriving are all time-correlated: if candidate A is measured five times in a row and then B
//    is, a drift between them lands entirely on one of the two. Interleaving spreads it over both. This is why
//    the repetition belongs in the caller's loop around moe_run_all() and not inside time_it().
//  * MEDIAN, not mean or min. One stall inflates a mean; a min is the best moment rather than the expected one,
//    and taking a min over more repeats makes every candidate look better without making the comparison better.
//  * REPORT A BAND AND REFUSE TO SEPARATE OVERLAPPING ONES. With a handful of repeats a quantile confidence
//    interval is arithmetic theatre, so the band is [min, max] over the repeats -- conservative, and it cannot
//    claim a separation the samples do not show. Candidates whose band overlaps the leader's are reported AS a
//    tie, which is also what makes codex's guard rule (H1/H2) operable: a guard inside the leader's band means
//    expand the stratum, and that test needs a band to exist.
struct Sample {
  char tag[bench_measure::kTagBytes] = {};
  bench_measure::Tactic tactic{};
  bool has_tactic = false;
  std::vector<double> us;
};

struct Best {
  char tag[bench_measure::kTagBytes];
  double us;                       // median of the leader
  std::vector<Sample> seen;        // every candidate, every repeat
  bench_measure::Tactic tactic{};  // structured identity of the leader; never parse tag to recover it
  bool has_tactic = false;
  int reps_seen = 0;
  bool any_selected = false;       // passed MOE_ONLY and reached a launch attempt, including rejected rows
};

inline double median_of(std::vector<double> v) {
  if (v.empty()) return 1e18;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}

// Called by every generated unit for every kernel it runs. Accumulates rather than compares: the comparison
// cannot be made until all repeats are in, and doing it here is what made the old version single-shot.
inline void upd_impl(Best& b, const char* t, double u, bench_measure::Tactic const* tactic) {
  for (auto& s : b.seen)
    if (std::strncmp(s.tag, t, bench_measure::kTagBytes) == 0) {
      s.us.push_back(u);
      if (tactic && !s.has_tactic) { s.tactic = *tactic; s.has_tactic = true; }
      return;
    }
  b.seen.push_back(Sample{});
  std::snprintf(b.seen.back().tag, bench_measure::kTagBytes, "%s", t);
  if (tactic) { b.seen.back().tactic = *tactic; b.seen.back().has_tactic = true; }
  b.seen.back().us.push_back(u);
}

inline void upd(Best& b, const char* t, double u) { upd_impl(b, t, u, nullptr); }

inline void upd(Best& b, bench_measure::Tactic const& tactic, double u) {
  char tag[bench_measure::kTagBytes];
  bench_measure::format_tag(tag, sizeof tag, tactic);
  upd_impl(b, tag, u, &tactic);
}

// Resolve after every repeat has run. Returns the number of candidates that TIE with the leader -- 0 means the
// leader is separated, and anything above 0 is the honest statement that this sweep did not resolve them.
inline int settle(Best& b) {
  b.us = 1e18; b.tag[0] = '\0'; b.has_tactic = false;
  for (auto& s : b.seen) {
    const double m = median_of(s.us);
    if (m < b.us) {
      b.us = m;
      std::snprintf(b.tag, bench_measure::kTagBytes, "%s", s.tag);
      b.tactic = s.tactic;
      b.has_tactic = s.has_tactic;
    }
  }
  int ties = 0;
  double lo = 1e18, hi = -1e18;
  for (auto& s : b.seen)
    if (std::strncmp(s.tag, b.tag, bench_measure::kTagBytes) == 0) {
      lo = *std::min_element(s.us.begin(), s.us.end());
      hi = *std::max_element(s.us.begin(), s.us.end());
    }
  for (auto& s : b.seen) {
    if (std::strncmp(s.tag, b.tag, bench_measure::kTagBytes) == 0) continue;
    const double slo = *std::min_element(s.us.begin(), s.us.end());
    if (slo <= hi) ++ties;                       // its band reaches into the leader's: not separated
  }
  (void)lo;
  return ties;
}
