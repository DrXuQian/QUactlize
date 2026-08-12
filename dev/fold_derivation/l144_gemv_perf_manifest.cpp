#include <cstdio>
#include <set>
#include <string>

#include "benchmarks/gemv_perf_manifest.hpp"

int main() {
  using namespace gemv_perf_manifest;
  int errors = 0;
  std::set<std::string> geometry_ids;
  std::set<std::string> shape_ids;
  std::set<std::string> shape_jsons;

  for (auto const& geometry : kGeometries) {
    errors += !geometry_ids.insert(geometry.id).second;
    std::printf("{\"rec\":\"geometry\",\"id\":\"%s\",\"route\":\"%s\","
                "\"experts\":%d,\"m\":%d,\"n\":%d,\"k\":%d,"
                "\"topk\":%d,\"active\":%d}\n",
                geometry.id, ppu_gemv::tactic_space::name_of(geometry.route),
                geometry.experts, geometry.rows, geometry.n, geometry.k,
                geometry.topk, geometry.active);
  }

  auto cases = shape_cases();
  for (auto const& shape : cases) {
    auto id = shape_id(shape);
    auto json = shape_json(shape);
    errors += !shape_ids.insert(id).second;
    errors += !shape_jsons.insert(json).second;
    std::printf("{\"rec\":\"case\",\"format\":\"%s\",\"shape_id\":\"%s\","
                "\"shape\":%s}\n",
                ppu_gemv::tactic_space::name_of(shape.semantics.format),
                id.c_str(), json.c_str());
  }

  ppu_gemv::tactic_space::Candidate sample{
      Format::Int4, Layout::TileK, 256, 16, 128, Route::Grouped, 4, 8, 4};
  auto const cfg = config_json(sample);
  auto const cid = config_id(sample);
  auto const job = job_json(cases.front(), {sample}, "0");
  std::printf("{\"rec\":\"summary\",\"geometries\":%zu,\"cases\":%zu,"
              "\"config_id\":\"%s\",\"config\":%s,\"job\":%s,\"errors\":%d}\n",
              kGeometries.size(), cases.size(), cid.c_str(), cfg.c_str(), job.c_str(), errors);
  return errors ? 1 : 0;
}
