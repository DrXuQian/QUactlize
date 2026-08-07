#pragma once
// THE C++ SELECTION -- isolated because it is scheduled for deletion (docs/BENCH_DESIGN.md step 3).
//
// It is kept only until benchmarks/analyse.py is shown to reproduce it on the same samples; deleting first
// would leave a window in which nothing produces a verdict. benchmarks/xcheck_select.cpp runs THIS code and
// the analyser over one planted file and compares, so "reproduces it" is a check rather than a belief.
//
// Isolated in its own header so that deletion is the removal of one #include rather than surgery on a bench.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
struct Sample { char tag[64]; std::vector<double> us; };

struct Best {
  char tag[64];
  double us;                       // median of the leader
  std::vector<Sample> seen;        // every candidate, every repeat
  int reps_seen = 0;
};

inline double median_of(std::vector<double> v) {
  if (v.empty()) return 1e18;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}

// Called by every generated unit for every kernel it runs. Accumulates rather than compares: the comparison
// cannot be made until all repeats are in, and doing it here is what made the old version single-shot.
inline void upd(Best& b, const char* t, double u) {
  for (auto& s : b.seen)
    if (std::strncmp(s.tag, t, 64) == 0) { s.us.push_back(u); return; }
  b.seen.push_back(Sample{});
  std::snprintf(b.seen.back().tag, 64, "%s", t);
  b.seen.back().us.push_back(u);
}

// Resolve after every repeat has run. Returns the number of candidates that TIE with the leader -- 0 means the
// leader is separated, and anything above 0 is the honest statement that this sweep did not resolve them.
inline int settle(Best& b) {
  b.us = 1e18; b.tag[0] = '\0';
  for (auto& s : b.seen) {
    const double m = median_of(s.us);
    if (m < b.us) { b.us = m; std::snprintf(b.tag, 64, "%s", s.tag); }
  }
  int ties = 0;
  double lo = 1e18, hi = -1e18;
  for (auto& s : b.seen)
    if (std::strncmp(s.tag, b.tag, 64) == 0) {
      lo = *std::min_element(s.us.begin(), s.us.end());
      hi = *std::max_element(s.us.begin(), s.us.end());
    }
  for (auto& s : b.seen) {
    if (std::strncmp(s.tag, b.tag, 64) == 0) continue;
    const double slo = *std::min_element(s.us.begin(), s.us.end());
    if (slo <= hi) ++ties;                       // its band reaches into the leader's: not separated
  }
  (void)lo;
  return ties;
}

// How many times the whole list is run. One repeat is legal and is NOT a ranking -- the banner says so, because
// a run that cannot rank must not print something that reads like a ranking.
// TWO NAMES FOR ONE QUANTITY, and they never agreed. benchmarks/sweep_real_shapes.py sets BENCH_REPS (its
// --reps), this read only MOE_REPS, and nothing compared them -- so a sweep asking for 3 passes silently got 5,
// and the driver's own record of how many samples a candidate has disagreed with what the bench wrote. Read
// BOTH, MOE_REPS winning when both are set so an explicit per-run override still beats the driver.
//
// DEFAULT IS 1, not 5. The user asked for one pass; five was the default and stayed. One repeat is legal and is
// NOT a ranking -- the banner says so, because a run that cannot rank must not print something that reads like
// one.
inline int moe_reps() {
  const char* e = std::getenv("MOE_REPS");
  if (e == nullptr || *e == '\0') e = std::getenv("BENCH_REPS");
  const int r = (e && *e) ? std::atoi(e) : 1;
  return r < 1 ? 1 : r;
}

