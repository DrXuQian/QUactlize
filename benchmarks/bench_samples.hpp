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

// WHAT WAS RUN, as opposed to what was measured. Everything through `pass` is chosen before the kernel launches;
// `us` is the primary machine measurement, with optional same-batch audit measurements below it. Keeping that split
// visible in the struct is what stops a derived quantity (a median, an MFU, a "best") from being written into a
// sample file and later read as a measurement.
struct Sample {
  char const* fixture;      // named, versioned: "qwen35-a3b-expert-gate"
  char const* dist;         // the row distribution, named and versioned: "skew-h8-v1"
  char const* schema;       // "i4", "q3", ...
  int n, k, gs;
  int experts, rows, mmax;  // MoE; mmax decides which compact-A capacities are reachable, so it is not optional
  int tm, tn, tk, wm, wn, st;
  // The CTA's K extent per warp cohort. Zero is the legacy "there is no warp-K axis" spelling. Keeping zero as
  // the default lets every existing emitter preserve its byte-for-byte record shape, while standalone Marlin
  // rows opt in with a positive value and cannot collapse distinct WarpK tactics in the durable sample file.
  int warp_k = 0;
  // PPU_B_CHUNK, AS TWO FIELDS, because it is two different facts. `bc` is what the ROW asked for -- it became a
  // tactic row field on 2026-08-07 and is no longer a property of the build, so it cannot live in run_header's
  // `build` string any more. `bc_eff` is what the collective's capability predicate actually granted: chunking
  // needs bits in {1,2} AND a fragment that is exactly one delivery, and a row that asks and is refused compiles
  // to the SAME kernel as bc=0. Recording only the request would put two identical kernels in the file under
  // different identities and let an analyser "discover" a difference between them.
  int bc, bc_eff;
  int pass;                 // which repetition of the WHOLE candidate list this came from
  double us;                // the measurement
  // Optional audit trail for a benchmark whose primary `us` is a device-event span. Older/dense emitters leave
  // timing null and preserve their byte-for-byte JSON shape; the MoE emitter records the same batch's host wall
  // plus the per-launch range so a future reader cannot mistake the new primary for the superseded wall protocol.
  char const* timing = nullptr;
  double wall_us = 0.0;
  double launch_min_us = 0.0, launch_max_us = 0.0, launch_spread_pct = 0.0;
  int launches = 0;
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

// No JSON library and no escaping beyond this: every string field is an identifier chosen in this repo, so a
// quote in one is a bug in the caller rather than data to be escaped. Refuse instead of silently mangling.
inline bool plain_names(Sample const& s) {
  auto plain = [](char const* p) { return p && !std::strchr(p, '"') && !std::strchr(p, '\\'); };
  if (plain(s.fixture) && plain(s.dist) && plain(s.schema) && (!s.timing || plain(s.timing))) return true;
  std::fprintf(stderr, "[bench_samples] refusing to write a record with a quote in a name field\n");
  return false;
}

// WHICH CANDIDATE IS ABOUT TO RUN, written and FLUSHED BEFORE THE LAUNCH.
//
// A DEVICE ASSERT TAKES THE WHOLE PROCESS, so a sweep can end at any candidate with no further output. Flushing
// completed samples preserves the successful prefix, which is necessary and not sufficient: it says which
// candidates SUCCEEDED and leaves the reader to infer the next one from a config list they must reconstruct.
// Worse in this bench specifically, the candidate name reaches stdout only after run_config() RETURNS, so a
// launch that dies is not named anywhere at all -- and "reproduce the failing row by name" is the whole recovery
// procedure. (Raised by codex reviewing the 293-row sweep plan; the first version of this change flushed samples
// and stopped there, which would have left exactly that hole.)
//
// So the file gets one `a` record per launch. The reader's rule is one line: AN ATTEMPT WITH NO MATCHING SAMPLE
// IS WHERE IT STOPPED. `us` is not written here -- there is nothing to write yet, and a placeholder would be a
// measurement-shaped value in a measurement file.
inline void attempt(Sample const& s) {
  std::FILE* f = stream();
  if (!f || !plain_names(s)) return;
  std::fprintf(f,
      "{\"rec\":\"a\",\"fixture\":\"%s\",\"dist\":\"%s\",\"schema\":\"%s\","
      "\"n\":%d,\"k\":%d,\"gs\":%d,\"experts\":%d,\"rows\":%d,\"mmax\":%d,"
      "\"tm\":%d,\"tn\":%d,\"tk\":%d,\"wm\":%d,\"wn\":%d,\"st\":%d",
      s.fixture, s.dist, s.schema, s.n, s.k, s.gs, s.experts, s.rows, s.mmax,
      s.tm, s.tn, s.tk, s.wm, s.wn, s.st);
  if (s.warp_k > 0) std::fprintf(f, ",\"warp_k\":%d", s.warp_k);
  std::fprintf(f, ",\"bc\":%d,\"bc_eff\":%d,\"pass\":%d}\n", s.bc, s.bc_eff, s.pass);
  std::fflush(f);
}

inline void emit(Sample const& s) {
  std::FILE* f = stream();
  if (!f || !plain_names(s)) return;
  std::fprintf(f,
      "{\"rec\":\"s\",\"fixture\":\"%s\",\"dist\":\"%s\",\"schema\":\"%s\","
      "\"n\":%d,\"k\":%d,\"gs\":%d,\"experts\":%d,\"rows\":%d,\"mmax\":%d,"
      "\"tm\":%d,\"tn\":%d,\"tk\":%d,\"wm\":%d,\"wn\":%d,\"st\":%d",
      s.fixture, s.dist, s.schema, s.n, s.k, s.gs, s.experts, s.rows, s.mmax,
      s.tm, s.tn, s.tk, s.wm, s.wn, s.st);
  if (s.warp_k > 0) std::fprintf(f, ",\"warp_k\":%d", s.warp_k);
  std::fprintf(f, ",\"bc\":%d,\"bc_eff\":%d,\"pass\":%d,\"us\":%.6f",
               s.bc, s.bc_eff, s.pass, s.us);
  if (s.timing) {
    std::fprintf(f,
        ",\"timing\":\"%s\",\"wall_us\":%.6f,\"launches\":%d,"
        "\"launch_min_us\":%.6f,\"launch_max_us\":%.6f,\"launch_spread_pct\":%.6f",
        s.timing, s.wall_us, s.launches, s.launch_min_us, s.launch_max_us, s.launch_spread_pct);
  }
  std::fprintf(f, "}\n");
  // FLUSHED PER SAMPLE, not once at the end. run_header() already flushed; emit() did not, and the benches call
  // flush() only after every pass and candidate -- so an abort discarded every buffered sample, which is the
  // entire content of the file. Per-sample flushing cannot make the SUCCESSORS run (a poisoned context needs the
  // process restarted) but it does make the prefix durable, and that is what a resume would consume.
  std::fflush(f);
}

// TRIED AND EXCLUDED, with a reason. Not the same as a crash and not the same as never having been tried, and
// a log that cannot tell those three apart cannot support pruning.
//
// WITHOUT THIS the three collapse into one: a config the bench rejects (unsupported for the shape, failed its
// correctness check, over a resource bound) emits an `a` and then nothing, which is byte-identical to a device
// assert -- so unfinished() reports a false alarm on a healthy run, and the pruning question "which options are
// never viable anywhere" has no evidence at all, only absence. Absence is what a config that was never in the
// table also looks like.
//
// `us` is absent for the same reason it is absent from an attempt: nothing was measured.
inline void excluded(Sample const& s, char const* why) {
  std::FILE* f = stream();
  if (!f || !plain_names(s)) return;
  auto plain = [](char const* p) { return p && !std::strchr(p, '"') && !std::strchr(p, '\\'); };
  char const* reason = plain(why) ? why : "unprintable";
  std::fprintf(f,
      "{\"rec\":\"x\",\"fixture\":\"%s\",\"dist\":\"%s\",\"schema\":\"%s\","
      "\"n\":%d,\"k\":%d,\"gs\":%d,\"experts\":%d,\"rows\":%d,\"mmax\":%d,"
      "\"tm\":%d,\"tn\":%d,\"tk\":%d,\"wm\":%d,\"wn\":%d,\"st\":%d",
      s.fixture, s.dist, s.schema, s.n, s.k, s.gs, s.experts, s.rows, s.mmax,
      s.tm, s.tn, s.tk, s.wm, s.wn, s.st);
  if (s.warp_k > 0) std::fprintf(f, ",\"warp_k\":%d", s.warp_k);
  std::fprintf(f, ",\"bc\":%d,\"bc_eff\":%d,\"pass\":%d,\"why\":\"%s\"}\n",
               s.bc, s.bc_eff, s.pass, reason);
  std::fflush(f);
}

inline void flush() { if (std::FILE* f = stream()) std::fflush(f); }

}  // namespace bench_samples
