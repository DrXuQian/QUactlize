// EMIT SAMPLES; DECIDE NOTHING. One line per (fixture, config, pass), and `us` is the only measurement on it.
//
// WHY THIS EXISTS (docs/BENCH_DESIGN.md). The selection logic -- median over interleaved repeats, a [min,max]
// band, tie reporting -- was written into the MoE bench in C++, where it has no unit test and where the dense
// bench would need a second copy of it. That is the two-copy defect this project keeps producing, one level up
// from the config table where it was just removed. So the benches emit samples and benchmarks/analyse.py
// decides, because a decision procedure that can be fed planted data is one that can be shown to be wrong.
//
// A SECOND REASON, which turned out to matter more. The recorded cross-run spread is 13%. When selection happens
// inside the bench, that spread is consumed and thrown away: the log carries a verdict and not the evidence for
// it, so "was that separation real?" needs a new run. With samples on disk it is a question about a file.
//
//   BENCH_JSONL=/path/run.jsonl ./the_bench ...
//
// Unset -> nothing is written and the bench behaves exactly as before. This is deliberately additive: the C++
// selection stays until the analyser reproduces it on the same data (BENCH_DESIGN step 3), because deleting
// first would leave a window in which nothing produces a verdict at all.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace bench_samples {

// WHAT WAS RUN, as opposed to what was measured. Everything here is chosen before the kernel launches; `us` is
// the only field the machine produces. Keeping that split visible in the struct is what stops a derived
// quantity (a median, an MFU, a "best") from being written into a sample file and later read as a measurement.
struct Sample {
  char const* fixture;      // named, versioned: "qwen35-a3b-expert-gate"
  char const* dist;         // the row distribution, named and versioned: "skew-h8-v1"
  char const* schema;       // "i4", "q3", ...
  int n, k, gs;
  int experts, rows, mmax;  // MoE; mmax decides which compact-A capacities are reachable, so it is not optional
  int tm, tn, tk, wm, wn, st;
  int pass;                 // which repetition of the WHOLE candidate list this came from
  double us;                // the measurement
};

inline std::FILE* stream() {
  // Opened once, appended to. Append and not truncate: a sweep is often several invocations over different
  // shapes, and truncating would silently keep only the last -- the same shape of loss as a log that describes
  // a different run than the one being read.
  static std::FILE* f = [] () -> std::FILE* {
    char const* p = std::getenv("BENCH_JSONL");
    if (!p || !*p) return nullptr;
    std::FILE* h = std::fopen(p, "a");
    if (!h) std::fprintf(stderr, "[bench_samples] cannot open BENCH_JSONL=%s -- samples are NOT being written\n", p);
    return h;
  }();
  return f;
}

inline bool enabled() { return stream() != nullptr; }

// The run header. Emitted once per process, BEFORE any sample. An analyser must be able to refuse to merge two
// incompatible runs rather than averaging them, and that needs the build's identity in the file: which format
// the library was compiled for, which TileK, whether the packed path was even on.
inline void run_header(char const* bench, char const* build_defines, int reps) {
  std::FILE* f = stream();
  if (!f) return;
  std::fprintf(f, "{\"rec\":\"run\",\"bench\":\"%s\",\"build\":\"%s\",\"reps\":%d}\n",
               bench, build_defines, reps);
  std::fflush(f);
}

inline void emit(Sample const& s) {
  std::FILE* f = stream();
  if (!f) return;
  // No JSON library and no escaping beyond this: every string field is an identifier chosen in this repo, so a
  // quote in one is a bug in the caller rather than data to be escaped. Assert it instead of silently mangling.
  auto plain = [](char const* p) { return p && !std::strchr(p, '"') && !std::strchr(p, '\\'); };
  if (!plain(s.fixture) || !plain(s.dist) || !plain(s.schema)) {
    std::fprintf(stderr, "[bench_samples] refusing to write a sample with a quote in a name field\n");
    return;
  }
  std::fprintf(f,
      "{\"rec\":\"s\",\"fixture\":\"%s\",\"dist\":\"%s\",\"schema\":\"%s\","
      "\"n\":%d,\"k\":%d,\"gs\":%d,\"experts\":%d,\"rows\":%d,\"mmax\":%d,"
      "\"tm\":%d,\"tn\":%d,\"tk\":%d,\"wm\":%d,\"wn\":%d,\"st\":%d,\"pass\":%d,\"us\":%.6f}\n",
      s.fixture, s.dist, s.schema, s.n, s.k, s.gs, s.experts, s.rows, s.mmax,
      s.tm, s.tn, s.tk, s.wm, s.wn, s.st, s.pass, s.us);
}

inline void flush() { if (std::FILE* f = stream()) std::fflush(f); }

}  // namespace bench_samples
