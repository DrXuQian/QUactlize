#pragma once

// Pure-host manifest construction for the finite GEMV sweep.  The benchmark
// and its local oracle both call this implementation; the oracle is not a
// second enumerator that could agree with itself while the device binary emits
// another candidate set.

#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemv_perf_manifest.hpp"

namespace gemv_perf_plan {

using gemv_perf_manifest::CompiledGroup;
using namespace ppu_gemv::tactic_space;

inline bool is_full_space(CompiledGroup const* groups, std::size_t count) {
  static constexpr char const* kAll[] = {
      "i4-native", "i4-tileK", "i2-native", "i2-tileK", "i1-native",
      "i1-tileK", "q3-native", "q3-tileK", "q6-native", "q6-tileK"};
  if (count != sizeof(kAll) / sizeof(kAll[0])) return false;
  for (char const* required : kAll) {
    bool found = false;
    for (std::size_t i = 0; i < count; ++i)
      found = found || std::string(groups[i].name) == required;
    if (!found) return false;
  }
  return true;
}

inline std::string space_id(CompiledGroup const* groups, std::size_t count) {
  std::string out = "gemv-v1/groups:";
  for (std::size_t i = 0; i < count; ++i) {
    if (i) out.push_back(',');
    out += groups[i].name;
  }
  return out;
}

struct Plan {
  std::uint64_t total = 0;
  std::uint64_t legal = 0;
  std::uint64_t pruned = 0;
  std::map<std::string, std::uint64_t> reasons;
  std::vector<std::string> jobs;
};

inline std::vector<Candidate> candidates_for(
    gemv_perf_manifest::ShapeCase const& shape,
    CompiledGroup const* groups, std::size_t group_count,
    Plan& plan) {
  std::vector<Candidate> legal;
  auto const route = shape.geometry.route;
  Problem const problem{route, gemv_perf_manifest::real_rows(shape.geometry),
                        shape.geometry.n, shape.geometry.k, shape.semantics.group_size};
  for (std::size_t group_index = 0; group_index < group_count; ++group_index) {
    auto const& group = groups[group_index];
    if (group.format != shape.semantics.format) continue;
    auto enumerate_ctam = [&](int cta_m) {
      for (int step_k : kStepKs)
        for (int threads : kThreads)
          for (int cta_n : kCtaNs)
            for (int chunk : kChunks) {
              Candidate const c{group.format, group.layout, group.tile_size_k,
                                step_k, threads, route, cta_m, cta_n, chunk};
              ++plan.total;
              auto const static_why = static_exclusion(c);
              if (static_why != Exclusion::None) {
                ++plan.pruned;
                ++plan.reasons[std::string("STATIC/") + name_of(static_why)];
                continue;
              }
              auto const shape_why = shape_exclusion(c, problem);
              if (shape_why != ShapeExclusion::None) {
                ++plan.pruned;
                ++plan.reasons[std::string("SHAPE/") + name_of(shape_why)];
                continue;
              }
              ++plan.legal;
              legal.push_back(c);
            }
    };
    if (route == Route::Dense)
      for (int cta_m : kDenseCtaMs) enumerate_ctam(cta_m);
    else
      for (int cta_m : kGroupedCtaMs) enumerate_ctam(cta_m);
  }
  return legal;
}

inline std::string manifest_json(CompiledGroup const* groups, std::size_t group_count) {
  if (!groups || group_count == 0) throw std::runtime_error("compiled GEMV group set is empty");
  Plan plan;
  auto const cases = gemv_perf_manifest::shape_cases();
  for (std::size_t i = 0; i < cases.size(); ++i) {
    bool has_format = false;
    for (std::size_t g = 0; g < group_count; ++g)
      has_format = has_format || groups[g].format == cases[i].semantics.format;
    if (!has_format) continue;
    auto legal = candidates_for(cases[i], groups, group_count, plan);
    if (legal.empty())
      throw std::runtime_error(
          "compiled format has no legal candidates for " + gemv_perf_manifest::shape_id(cases[i]));
    std::string argv = "--shape-case=" + std::to_string(i);
    plan.jobs.push_back(gemv_perf_manifest::job_json(cases[i], legal, argv.c_str()));
  }
  if (plan.total != plan.legal + plan.pruned || plan.jobs.empty())
    throw std::runtime_error("GEMV manifest census is internally inconsistent");

  std::string out = "{\"counts\":{\"legal\":" + std::to_string(plan.legal) +
      ",\"prune_reasons\":{";
  bool first = true;
  for (auto const& [why, count] : plan.reasons) {
    if (!first) out.push_back(',');
    first = false;
    gemv_perf_manifest::detail::append_json_string(out, why.c_str());
    out += ":" + std::to_string(count);
  }
  out += "},\"pruned\":" + std::to_string(plan.pruned) +
      ",\"total\":" + std::to_string(plan.total) + "},\"jobs\":[";
  first = true;
  for (auto const& job : plan.jobs) {
    if (!first) out.push_back(',');
    first = false;
    out += job;
  }
  out += "],\"partial_space\":";
  out += is_full_space(groups, group_count) ? "false" : "true";
  out += ",\"schema\":\"gemv-sweep-manifest-v1\",\"space_id\":";
  auto sid = space_id(groups, group_count);
  gemv_perf_manifest::detail::append_json_string(out, sid.c_str());
  out += "}\n";
  return out;
}

}  // namespace gemv_perf_plan
