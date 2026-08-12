// Finite, machine-readable low-bit GEMV tactic sweep.  Every compiled unit is
// one static-legal (format, layout, StepK, threads, CtaN, chunk) cell; the unit
// instantiates every exact CtaM for the selected dense/grouped route.
//
// Build: GEMV_GROUPS=i4-native TARGET=test_gemv_perf ./build.sh
// Plan:  test_gemv_perf --manifest-json /tmp/gemv-plan.json
// Run:   benchmarks/sweep_gemv_perf.py run /tmp/gemv-plan.json ...
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "gemv_perf_common.hpp"
#include "gemv_perf_plan.hpp"
#include "gemv_perf_units.inc"     // GENERATED: compiled-group metadata + unit calls

namespace {

using namespace ppu_gemv::tactic_space;

char const* env_required(char const* name) {
  char const* value = std::getenv(name);
  if (!value || !*value) {
    std::fprintf(stderr, "GEMV sweep requires non-empty %s\n", name);
    return nullptr;
  }
  return value;
}

bool write_manifest(char const* path) {
  std::string const text = gemv_perf_plan::manifest_json(
      gemv_compiled_groups, GEMV_GROUP_COUNT);
  if (std::strcmp(path, "-") == 0) {
    return std::fwrite(text.data(), 1, text.size(), stdout) == text.size();
  }
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(text.data(), std::streamsize(text.size()));
  return bool(stream);
}

bool parse_nonnegative(char const* text, int& value) {
  if (!text || !*text) return false;
  errno = 0;
  char* end = nullptr;
  long parsed = std::strtol(text, &end, 10);
  if (errno || *end || parsed < 0 || parsed > INT32_MAX) return false;
  value = int(parsed);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  int only_case = -1;
  char const* manifest_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--shape-case=", 13) == 0) {
      if (!parse_nonnegative(argv[i] + 13, only_case)) {
        std::fprintf(stderr, "invalid --shape-case: %s\n", argv[i]);
        return 2;
      }
    } else if (std::strcmp(argv[i], "--manifest-json") == 0 && i + 1 < argc) {
      manifest_path = argv[++i];
    } else if (argc == 2 && parse_nonnegative(argv[i], only_case)) {
      // Historical one-integer spelling retained for hand-driven captures.
    } else {
      std::fprintf(stderr, "usage: %s [--manifest-json PATH|-] [--shape-case=N]\n", argv[0]);
      return 2;
    }
  }
  if (manifest_path) return write_manifest(manifest_path) ? 0 : 2;

  auto const cases = gemv_perf_manifest::shape_cases();
  if (only_case >= int(cases.size())) {
    std::fprintf(stderr, "shape case %d outside [0,%zu)\n", only_case, cases.size());
    return 2;
  }

  gemv_perf_samples::JsonlWriter writer;
  SweepRuntime sweep;
  sweep.writer = writer.requested() ? &writer : nullptr;
  sweep.sweep = writer.requested();
  sweep.measured_launches = 20;
  if (char const* samples = std::getenv("GEMV_SWEEP_SAMPLES")) {
    int parsed = 0;
    if (!parse_nonnegative(samples, parsed) || parsed != 20) {
      std::fprintf(stderr,
                   "GEMV_SWEEP_SAMPLES must be exactly 20 for gemv-sweep-raw-v1\n");
      return 2;
    }
  }
  if (sweep.sweep) {
    if (only_case < 0 || only_filter() || std::getenv("GEMV_FMT") ||
        std::getenv("GEMV_CFG") || acu_mode()) {
      std::fprintf(stderr, "machine sweep requires exactly one --shape-case and forbids row/acu filters\n");
      return 2;
    }
    char const* run_id = env_required("GEMV_SWEEP_RUN_ID");
    char const* job_id = env_required("GEMV_SWEEP_JOB_ID");
    char const* attempt = env_required("GEMV_SWEEP_ATTEMPT");
    char const* build = env_required("GEMV_SWEEP_BUILD");
    if (!run_id || !job_id || !attempt || !build || !writer.enabled()) return 2;
    auto const expected_job = gemv_perf_manifest::shape_id(cases[std::size_t(only_case)]);
    if (expected_job != job_id) {
      std::fprintf(stderr, "GEMV_SWEEP_JOB_ID=%s, selected case is %s\n",
                   job_id, expected_job.c_str());
      return 2;
    }
    sweep.run_id = run_id;
    sweep.attempt_id = attempt;
    if (!writer.write_run({run_id, build,
                           gemv_perf_plan::space_id(gemv_compiled_groups, GEMV_GROUP_COUNT),
                           !gemv_perf_plan::is_full_space(gemv_compiled_groups, GEMV_GROUP_COUNT)}))
      return 2;
  }

  std::printf("== finite low-bit GEMV tactic sweep ==\n");
  std::printf("   rev %d, %d generated units, %d format/layout groups, space=%s%s\n",
              GEMV_PERF_REV, GEMV_UNIT_COUNT, GEMV_GROUP_COUNT,
              gemv_perf_plan::space_id(gemv_compiled_groups, GEMV_GROUP_COUNT).c_str(),
              gemv_perf_plan::is_full_space(gemv_compiled_groups, GEMV_GROUP_COUNT)
                  ? "" : " (PARTIAL)");
  if (acu_mode())
    std::printf("   *** GEMV_ACU: ONE COLD LAUNCH PER ROW. Captures, not timings. ***\n");

  for (std::size_t i = 0; i < cases.size(); ++i) {
    if (only_case >= 0 && int(i) != only_case) continue;
    bool available = false;
    for (auto const& group : gemv_compiled_groups)
      available = available || group.format == cases[i].semantics.format;
    if (!available) continue;

    Shape const sh = make_shape(cases[i]);
    auto const routed = sh.experts > 0
        ? gemv_perf_fixture::make_route(sh.experts, sh.rows, sh.topk)
        : gemv_perf_fixture::Route{};
    int const active = sh.experts > 0 ? int(routed.active_ids.size()) : 1;
    if (sh.experts > 0 && active != sh.active) {
      std::fprintf(stderr, "shape %s expected active=%d, router produced %d\n",
                   sh.name.c_str(), sh.active, active);
      return 2;
    }
    int const total_rows = sh.experts > 0 ? routed.total_rows : sh.rows;
    int const max_rows = sh.experts > 0 ? routed.max_rows : sh.rows;
    int const sk = sh.K / sh.gs;
    auto const* format = traits_of(sh.format);
    double const floor_b = double(active) *
        (double(sh.N) * sh.K * (format->low_bits + format->high_bits) / 8.0 +
         double(sk) * sh.N * 2.0 * (has_zero(sh.quant) ? 2 : 1)) +
        double(total_rows) * sh.K * 2.0 + double(total_rows) * sh.N * 2.0;
    std::printf("\n-- [%zu] %s  format=%s gs=%d %s E=%d active=%d real_rows=%d Mmax=%d\n",
                i, sh.name.c_str(), name_of(sh.format), sh.gs, name_of(sh.quant),
                sh.experts, active, total_rows, max_rows);
    std::printf("     distinct memory roof: %.2f us (%.2f MB at %.0f GB/s)\n",
                floor_b / (HBM_GBS * 1e9) * 1e6, floor_b / 1e6, HBM_GBS);

    std::vector<Best> bests(GEMV_GROUP_COUNT);
    gemv_run_all(sh, bests.data(), sweep);
    if (sweep.failed || (sweep.writer && !sweep.writer->ok())) return 2;
    if (!acu_mode() && !sweep.sweep) {
      Best overall;
      for (int g = 0; g < GEMV_GROUP_COUNT; ++g) {
        if (gemv_compiled_groups[g].format != sh.format || bests[g].us >= 1e29) continue;
        std::printf("        %-12s %-40s %8.3f us  %5.1f%% nameplate\n",
                    gemv_group_names[g], bests[g].tag, bests[g].us, bests[g].pct);
        if (bests[g].us < overall.us) overall = bests[g];
      }
      if (overall.us < 1e29)
        std::printf("     LOWEST IN COMPILED %s SPACE: %s %.3f us\n",
                    gemv_perf_plan::is_full_space(gemv_compiled_groups, GEMV_GROUP_COUNT)
                        ? "FULL" : "PARTIAL", overall.tag, overall.us);
    }
  }

  if (!sweep.sweep)
    std::printf("\n  launches refused: %d\n", gemv_fail_count());
  return 0;
}
