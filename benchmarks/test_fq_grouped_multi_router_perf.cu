#define main fq_kquant_layout_perf_embedded_main
#include "test_fq_kquant_layout_perf.cu"
#undef main

#include <set>

namespace {

struct RouterCase {
  std::string profile;
  int n = 0;
  int k = 0;
  std::vector<int> rows;
  int total = 0;
  int maximum = 0;
  int active = 0;
  int zero = 0;
  int work16 = 0;
  int work32 = 0;
  int work128 = 0;
  uint64_t rows_hash = 0;
};

struct RouterCli {
  int iterations = 11;
  int warmups = 3;
  int round = 1;
  std::vector<RouterCase> cases;
};

bool parse_rows(char const* text, std::vector<int>& rows) {
  std::string input(text ? text : "");
  std::size_t begin = 0;
  while (begin <= input.size()) {
    std::size_t const end = input.find(',', begin);
    std::string const token = input.substr(
        begin, end == std::string::npos ? end : end - begin);
    char* parsed_end = nullptr;
    long const value = std::strtol(token.c_str(), &parsed_end, 10);
    if (token.empty() || !parsed_end || *parsed_end || value < 0 ||
        value > INT32_MAX) return false;
    rows.push_back(int(value));
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return rows.size() == 256;
}

bool nonnegative(char const* text, int& value) {
  char* end = nullptr;
  long const parsed = std::strtol(text, &end, 10);
  if (!text[0] || !end || *end || parsed < 0 || parsed > INT32_MAX)
    return false;
  value = int(parsed);
  return true;
}

bool parse_case(char const* text, RouterCase& row) {
  std::vector<std::string> fields;
  std::string input(text ? text : "");
  std::size_t begin = 0;
  for (int i = 0; i < 11; ++i) {
    std::size_t const end = input.find(':', begin);
    if (end == std::string::npos) return false;
    fields.push_back(input.substr(begin, end - begin));
    begin = end + 1;
  }
  fields.push_back(input.substr(begin));
  if (fields.size() != 12 || fields[0].empty()) return false;
  row.profile = fields[0];
  int* outputs[] = {&row.n, &row.k, &row.total, &row.maximum, &row.active,
                    &row.zero, &row.work16, &row.work32, &row.work128};
  for (int i = 0; i < 9; ++i) {
    bool const parsed = i == 5
        ? nonnegative(fields[std::size_t(i + 1)].c_str(), *outputs[i])
        : positive(fields[std::size_t(i + 1)].c_str(), *outputs[i]);
    if (!parsed) return false;
  }
  char* hash_end = nullptr;
  unsigned long long const expected_hash =
      std::strtoull(fields[10].c_str(), &hash_end, 0);
  if (!fields[10].size() || !hash_end || *hash_end) return false;
  if (!parse_rows(fields[11].c_str(), row.rows)) return false;
  int total = 0, maximum = 0, active = 0;
  int work16 = 0, work32 = 0, work128 = 0;
  uint64_t rows_hash = UINT64_C(14695981039346656037);
  for (int value : row.rows) {
    total += value;
    maximum = std::max(maximum, value);
    active += value > 0;
    if (value) {
      work16 += (value + 15) / 16;
      work32 += (value + 31) / 32;
      work128 += (value + 127) / 128;
    }
    uint32_t const word = uint32_t(value);
    for (int byte = 0; byte < 4; ++byte) {
      rows_hash ^= uint8_t(word >> (8 * byte));
      rows_hash *= UINT64_C(1099511628211);
    }
  }
  row.rows_hash = rows_hash;
  return row.n % 256 == 0 && row.k % 256 == 0 && total == row.total &&
      maximum == row.maximum && active == row.active &&
      256 - active == row.zero && work16 == row.work16 &&
      work32 == row.work32 && work128 == row.work128 &&
      rows_hash == uint64_t(expected_hash);
}

bool parse_router_cli(int argc, char** argv, RouterCli& cli) {
  for (int i = 1; i < argc; ++i) {
    auto value = [&](char const* prefix) -> char const* {
      std::size_t const n = std::strlen(prefix);
      return std::strncmp(argv[i], prefix, n) == 0 ? argv[i] + n : nullptr;
    };
    if (char const* v = value("--iterations=")) {
      if (!positive(v, cli.iterations)) return false;
    } else if (char const* v = value("--warmups=")) {
      if (!positive(v, cli.warmups)) return false;
    } else if (char const* v = value("--round=")) {
      if (!positive(v, cli.round)) return false;
    } else if (char const* v = value("--case=")) {
      RouterCase row;
      if (!parse_case(v, row)) return false;
      cli.cases.push_back(std::move(row));
    } else {
      return false;
    }
  }
  if (cli.cases.size() != 6) return false;
  std::set<std::string> names;
  for (auto const& row : cli.cases) names.insert(row.profile);
  return names == std::set<std::string>{
      "balanced", "hot-skewed", "sparse-empty", "tilem-boundary",
      "permutation-a", "permutation-b"};
}

bool run_router_case(RouterCase const& route, RouterCli const& cli,
                     DeviceWeights& weights) {
  auto fail = [&](char const* phase, int code, char const* config) {
    std::printf(
        "FQ_GROUPED_ROUTER_FAILURE q=%d round=%d profile=%s phase=%s "
        "config=%s code=%d\n", kQtype, cli.round, route.profile.c_str(),
        phase, config, code);
    return false;
  };
  std::vector<int> offsets(257, 0);
  for (int expert = 0; expert < 256; ++expert)
    offsets[std::size_t(expert + 1)] = offsets[std::size_t(expert)] +
                                      route.rows[std::size_t(expert)];
  ProblemData host = make_problem(route.total, route.n, route.k);
  if (host.a.empty() || host.golden.empty()) return fail("FIXTURE", -1, "NONE");
  cutlass::DeviceAllocation<half_t> a(host.a.size()), out(host.golden.size()),
      golden(host.golden.size());
  cutlass::DeviceAllocation<int> d_offsets(offsets.size());
  a.copy_from_host(host.a.data()); golden.copy_from_host(host.golden.data());
  d_offsets.copy_from_host(offsets.data());
  int64_t const ws_bytes =
      quactlize_ppu_grouped_fully_quantized_workspace_bytes_for_arrangement_v2(
          route.total, route.maximum, route.n, route.k, 256, kQtype,
          &weights.descriptor);
  if (ws_bytes <= 0) return fail("WORKSPACE_QUERY", int(ws_bytes), "NONE");
  cutlass::DeviceAllocation<uint8_t> workspace{std::size_t(ws_bytes)};
  cutlass::DeviceAllocation<unsigned int> counter(1);
  auto configs = grouped_configs(route.total, route.n, route.k, 256,
                                 route.maximum, weights.descriptor, true);
  if (configs.empty()) return fail("CONFIG_QUERY", 0, "NONE");
  for (auto const& config : configs) {
    int launch_rc = 0;
    auto launch = [&] {
      launch_rc = quactlize_ppu_grouped_fully_quantized_dev_for_arrangement_v2(
          reinterpret_cast<uint16_t const*>(a.get()), weights.low.get(),
          F::HighBits ? weights.high.get() : nullptr, weights.units.get(),
          d_offsets.get(), reinterpret_cast<uint16_t*>(out.get()),
          route.total, route.n, route.k, 256, route.maximum, kQtype,
          workspace.get(), ws_bytes, nullptr, config.wire,
          &weights.descriptor);
      return launch_rc;
    };
    if (hggcMemset(out.get(), 0x7b, host.golden.size() * sizeof(half_t)) !=
        hggcSuccess) return fail("OUTPUT_POISON", -1, config.label.c_str());
    if (launch() != 0 || hggcDeviceSynchronize() != hggcSuccess)
      return fail("LAUNCH", launch_rc, config.label.c_str());
    unsigned int bad = 0;
    if (!raw_bad(reinterpret_cast<uint16_t const*>(out.get()),
                 reinterpret_cast<uint16_t const*>(golden.get()),
                 host.golden.size(), counter, bad) || bad)
      return fail("RAW_MISMATCH", int(bad), config.label.c_str());
    Timing timing;
    if (!measure(launch, cli.warmups, cli.iterations, timing))
      return fail("TIMING", launch_rc, config.label.c_str());
    std::printf(
        "FQ_GROUPED_ROUTER_CELL q=%d round=%d profile=%s layout=kpack "
        "mapping_id=0x%016llx n=%d k=%d experts=256 total_rows=%d "
        "max_rows=%d active=%d zero=%d work_tm16=%d work_tm32=%d "
        "work_tm128=%d rows_hash=0x%016llx config=%s provider=standard-aiu iterations=%d "
        "raw_bad=0 median_us=%.9f min_us=%.9f max_us=%.9f samples=",
        kQtype, cli.round, route.profile.c_str(),
        static_cast<unsigned long long>(weights.descriptor.mapping_id),
        route.n, route.k, route.total, route.maximum, route.active,
        route.zero, route.work16, route.work32, route.work128,
        static_cast<unsigned long long>(route.rows_hash),
        config.label.c_str(), cli.iterations, timing.median, timing.minimum,
        timing.maximum);
    print_samples(timing.samples); std::printf("\n");
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  RouterCli cli;
  if (!parse_router_cli(argc, argv, cli)) return 2;
  bool ok = true;
  std::map<std::pair<int, int>, std::vector<RouterCase const*>> families;
  for (auto const& row : cli.cases) families[{row.n, row.k}].push_back(&row);
  for (auto const& family : families) {
    HostWeights host = make_weights(family.first.first, family.first.second, 256, true);
    if (!host.exact) { ok = false; break; }
    DeviceWeights weights(host);
    for (RouterCase const* row : family.second)
      if (!run_router_case(*row, cli, weights)) { ok = false; break; }
    if (!ok) break;
  }
  std::printf(
      "FQ_GROUPED_ROUTER_RUN schema=grouped-kpack-multi-router-v1 q=%d "
      "round=%d layout=kpack iterations=%d warmups=%d cells=%zu status=%s\n",
      kQtype, cli.round, cli.iterations, cli.warmups, cli.cases.size(),
      ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
