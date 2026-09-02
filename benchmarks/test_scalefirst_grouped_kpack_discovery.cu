// One generated shard owns an exact parent range for one canonical K-pack
// format.  The fixture is either a real token/top-k histogram or an exact
// authority-provided row histogram; correctness is raw fp16 before timing.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "cutlass/util/device_memory.h"
#include "kpack_grouped_fixture_rows.hpp"
#include "kquant_kpack_offline.hpp"
#include "moe_router_fixture.hpp"
#include "q4_kpack4_offline.hpp"
#include "scalefirst_grouped_kpack_discovery.hpp"
#include "scalefirst_grouped_registry.inc"

#ifndef SCALEFIRST_GROUPED_KPACK_QTYPE
#error "SCALEFIRST_GROUPED_KPACK_QTYPE must select Q2/Q3/Q4/Q5/Q6"
#endif
#ifndef SCALEFIRST_GROUPED_KPACK_WEIGHT_LAYOUT
#error "SCALEFIRST_GROUPED_KPACK_WEIGHT_LAYOUT must bind layout1/2"
#endif
static_assert(SCALEFIRST_GROUPED_KPACK_QTYPE ==
                  SCALEFIRST_GROUPED_GENERATED_QTYPE);
static_assert(SCALEFIRST_GROUPED_KPACK_WEIGHT_LAYOUT ==
                  SCALEFIRST_GROUPED_GENERATED_WEIGHT_LAYOUT);

namespace scalefirst_grouped_kpack_generated {
#define SCALEFIRST_GROUPED_DECLARE(FN,Q,L,TM,TN,TK,WM,WN,ST,DN)       \
  bool FN(scalefirst_grouped_kpack::Inputs const&,                    \
          scalefirst_grouped_kpack::Options const&,                   \
          scalefirst_grouped_kpack::Result&);
SCALEFIRST_GROUPED_REGISTRY_ROWS(SCALEFIRST_GROUPED_DECLARE)
#undef SCALEFIRST_GROUPED_DECLARE
}  // namespace scalefirst_grouped_kpack_generated

namespace {

using namespace scalefirst_grouped_kpack;
using GS = moe_grouped_ppu::GroupShape;
using DStride = moe_grouped_ppu::DStride;

struct Cli {
  int tokens = 4, topk = 2, experts = 16;
  int n = 256, k = 512;
  std::uint64_t schedule_seed = UINT64_C(0x243f6a8885a308d3);
  Options options;
  std::string symbol, symbol_file, rows_file;
  std::string workload_key = "adhoc", router_profile = "token-topk";
};

bool one_token(std::string const& value) {
  return !value.empty() && value.find_first_of(" \t\r\n") == std::string::npos;
}

bool positive(char const* text, int& value) {
  char* end = nullptr;
  long const parsed = std::strtol(text, &end, 10);
  if (!text[0] || !end || *end || parsed <= 0 ||
      parsed > std::numeric_limits<int>::max())
    return false;
  value = int(parsed);
  return true;
}

bool parse_cli(int argc, char** argv, Cli& cli) {
  for (int index = 1; index < argc; ++index) {
    auto value = [&](char const* prefix) -> char const* {
      std::size_t const n = std::strlen(prefix);
      return std::strncmp(argv[index], prefix, n) == 0
          ? argv[index] + n : nullptr;
    };
    if (char const* v = value("--tokens=")) {
      if (!positive(v, cli.tokens)) return false;
    } else if (char const* v = value("--topk=")) {
      if (!positive(v, cli.topk)) return false;
    } else if (char const* v = value("--experts=")) {
      if (!positive(v, cli.experts)) return false;
    } else if (char const* v = value("--n=")) {
      if (!positive(v, cli.n)) return false;
    } else if (char const* v = value("--k=")) {
      if (!positive(v, cli.k)) return false;
    } else if (char const* v = value("--iterations=")) {
      if (!positive(v, cli.options.iterations)) return false;
    } else if (char const* v = value("--warmups=")) {
      if (!positive(v, cli.options.warmups)) return false;
    } else if (char const* v = value("--correctness-repeats=")) {
      if (!positive(v, cli.options.correctness_repeats)) return false;
    } else if (char const* v = value("--schedule-seed=")) {
      char* end = nullptr;
      cli.schedule_seed = std::strtoull(v, &end, 0);
      if (!v[0] || !end || *end) return false;
    } else if (char const* v = value("--symbol=")) {
      cli.symbol = v;
      if (cli.symbol.empty()) return false;
    } else if (char const* v = value("--symbol-file=")) {
      cli.symbol_file = v;
      if (cli.symbol_file.empty()) return false;
    } else if (char const* v = value("--rows-file=")) {
      cli.rows_file = v;
      if (cli.rows_file.empty()) return false;
    } else if (char const* v = value("--workload-key=")) {
      cli.workload_key = v;
      if (!one_token(cli.workload_key)) return false;
    } else if (char const* v = value("--router-profile=")) {
      cli.router_profile = v;
      if (!one_token(cli.router_profile)) return false;
    } else return false;
  }
  return cli.topk <= cli.experts && cli.n % 16 == 0 && cli.k % 256 == 0 &&
      (cli.symbol.empty() || cli.symbol_file.empty());
}

std::vector<RegistryRow> registry() {
  return {
#define SCALEFIRST_GROUPED_REGISTER(FN,Q,L,TM,TN,TK,WM,WN,ST,DN)     \
    {#FN,Q,L,TM,TN,TK,WM,WN,ST,DN,                                  \
     &scalefirst_grouped_kpack_generated::FN},
    SCALEFIRST_GROUPED_REGISTRY_ROWS(SCALEFIRST_GROUPED_REGISTER)
#undef SCALEFIRST_GROUPED_REGISTER
  };
}

bool select_registry(Cli const& cli, std::vector<RegistryRow>& rows,
                     std::string& error) {
  rows = registry();
  if (!cli.symbol.empty()) {
    rows.erase(std::remove_if(
        rows.begin(), rows.end(), [&](RegistryRow const& row) {
          return cli.symbol != row.symbol;
        }), rows.end());
    if (rows.size() != 1) {
      error = "--symbol names no unique generated row";
      return false;
    }
    return true;
  }
  if (cli.symbol_file.empty()) return true;
  std::ifstream stream(cli.symbol_file);
  if (!stream) {
    error = "cannot open --symbol-file";
    return false;
  }
  std::set<std::string> requested;
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line.find_first_of(" \t") != std::string::npos ||
        !requested.insert(line).second) {
      error = "symbol file has an empty, duplicate or whitespace-bearing row";
      return false;
    }
  }
  if (!stream.eof() || requested.empty()) {
    error = "symbol file is empty or unreadable";
    return false;
  }
  std::vector<RegistryRow> selected;
  for (RegistryRow const& row : rows) {
    auto found = requested.find(row.symbol);
    if (found != requested.end()) {
      selected.push_back(row);
      requested.erase(found);
    }
  }
  if (!requested.empty()) {
    error = "symbol file names an unknown generated row";
    return false;
  }
  rows.swap(selected);
  return true;
}

constexpr int low_bits() {
  return SCALEFIRST_GROUPED_KPACK_QTYPE == 10 ||
                 SCALEFIRST_GROUPED_KPACK_QTYPE == 11 ? 2 : 4;
}
constexpr int high_bits() {
  return SCALEFIRST_GROUPED_KPACK_QTYPE == 11 ||
                 SCALEFIRST_GROUPED_KPACK_QTYPE == 13 ? 1 :
         SCALEFIRST_GROUPED_KPACK_QTYPE == 14 ? 2 : 0;
}
constexpr int group_size() {
  return SCALEFIRST_GROUPED_KPACK_QTYPE == 12 ||
                 SCALEFIRST_GROUPED_KPACK_QTYPE == 13 ? 32 : 16;
}
constexpr int center_correction() {
  return SCALEFIRST_GROUPED_KPACK_QTYPE == 11 ? -4 :
         SCALEFIRST_GROUPED_KPACK_QTYPE == 14 ? -24 : 0;
}

int code_value(int expert, int n, int k) {
  int const seed = 13 * expert + 7 * n + 5 * k + 3;
  if constexpr (SCALEFIRST_GROUPED_KPACK_QTYPE == 10) return seed & 3;
  if constexpr (SCALEFIRST_GROUPED_KPACK_QTYPE == 11) return seed & 7;
  if constexpr (SCALEFIRST_GROUPED_KPACK_QTYPE == 12) return seed & 15;
  if constexpr (SCALEFIRST_GROUPED_KPACK_QTYPE == 13) return seed & 31;
  return seed & 63;
}

int decoded_value(int code) {
  if constexpr (SCALEFIRST_GROUPED_KPACK_QTYPE == 10) return code;
  if constexpr (SCALEFIRST_GROUPED_KPACK_QTYPE == 11) return code - 4;
  if constexpr (SCALEFIRST_GROUPED_KPACK_QTYPE == 12 ||
                SCALEFIRST_GROUPED_KPACK_QTYPE == 13) return code - 8;
  return code - 32;
}

void put_native(std::vector<std::uint8_t>& plane, int bits, int n, int k,
                int K, int value) {
  std::uint64_t const bit = (std::uint64_t(n) * K + k) * bits;
  plane[bit >> 3] |= std::uint8_t(value << (bit & 7));
}

template <bool Recover>
int transform_generic(std::uint8_t const* low_in,
                      std::uint8_t const* high_in,
                      std::uint8_t* low_out, std::uint8_t* high_out,
                      int n, int k) {
  if constexpr (SCALEFIRST_GROUPED_KPACK_QTYPE == 10)
    return kquant_kpack::transform<2, 0, 16, Recover>(
        low_in, high_in, low_out, high_out, n, k);
  if constexpr (SCALEFIRST_GROUPED_KPACK_QTYPE == 11)
    return kquant_kpack::transform<2, 1, 16, Recover>(
        low_in, high_in, low_out, high_out, n, k);
  if constexpr (SCALEFIRST_GROUPED_KPACK_QTYPE == 13)
    return kquant_kpack::transform<4, 1, 32, Recover>(
        low_in, high_in, low_out, high_out, n, k);
  if constexpr (SCALEFIRST_GROUPED_KPACK_QTYPE == 14)
    return kquant_kpack::transform<4, 2, 16, Recover>(
        low_in, high_in, low_out, high_out, n, k);
  return 25;
}

struct HostFixture {
  std::vector<int> rows, offsets;
  std::vector<GS> shapes;
  std::vector<half_t> a, scale, zero, golden;
  std::vector<std::uint8_t> low, high;
  int total = 0, maximum = 0, active = 0, empty = 0;
  bool roundtrip = false;
};

HostFixture make_fixture(Cli const& cli) {
  HostFixture fixture;
  char why[128]{};
  if (cli.rows_file.empty()) {
    moe_router_fixture::Rows route;
    if (!moe_router_fixture::route(
            cli.tokens, cli.topk, cli.experts, route, why, sizeof why))
      return fixture;
    fixture.rows = route.per_expert;
    fixture.total = route.total;
    fixture.maximum = route.max;
    fixture.active = route.active;
    fixture.empty = route.zero;
  } else {
    kpack_grouped_fixture_rows::Rows route;
    if (!kpack_grouped_fixture_rows::load(
            cli.rows_file.c_str(), cli.experts, route, why, sizeof why))
      return fixture;
    fixture.rows = std::move(route.per_expert);
    fixture.total = route.total;
    fixture.maximum = route.max;
    fixture.active = route.active;
    fixture.empty = route.zero;
  }
  fixture.offsets.assign(std::size_t(cli.experts + 1), 0);
  for (int expert = 0; expert < cli.experts; ++expert)
    fixture.offsets[std::size_t(expert + 1)] =
        fixture.offsets[std::size_t(expert)] +
        fixture.rows[std::size_t(expert)];
  if (fixture.offsets.back() != fixture.total || fixture.empty < 0 ||
      fixture.active <= 0 || fixture.active > cli.experts)
    return HostFixture{};

  fixture.shapes.reserve(cli.experts);
  for (int expert = 0; expert < cli.experts; ++expert)
    fixture.shapes.push_back(cute::make_shape(
        fixture.rows[std::size_t(expert)], cli.n, cli.k));
  fixture.a.assign(std::size_t(fixture.total) * cli.k, half_t(0.f));
  fixture.golden.resize(std::size_t(fixture.total) * cli.n);
  int const groups = cli.k / group_size();
  fixture.scale.resize(std::size_t(cli.experts) * groups * cli.n);
  fixture.zero.resize(fixture.scale.size());

  for (int expert = 0; expert < cli.experts; ++expert)
    for (int group = 0; group < groups; ++group)
      for (int n = 0; n < cli.n; ++n) {
        float const scale = float(1 << ((expert + group + n) & 1));
        float const affine = float((3 * expert + group + n) % 3 - 1);
        std::size_t const index =
            (std::size_t(expert) * groups + group) * cli.n + n;
        fixture.scale[index] = half_t(scale);
        fixture.zero[index] = half_t(
            affine + center_correction() * scale);
      }

  std::size_t const low_bytes = std::size_t(cli.n) * cli.k * low_bits() / 8;
  std::size_t const high_bytes =
      std::size_t(cli.n) * cli.k * high_bits() / 8;
  fixture.low.resize(std::size_t(cli.experts) * low_bytes);
  fixture.high.resize(std::size_t(cli.experts) * high_bytes);
  bool exact = true;
  for (int expert = 0; expert < cli.experts; ++expert) {
    std::vector<std::uint8_t> native_low(low_bytes, 0), back_low(low_bytes, 0);
    std::vector<std::uint8_t> native_high(high_bytes, 0), back_high(high_bytes, 0);
    for (int n = 0; n < cli.n; ++n)
      for (int k = 0; k < cli.k; ++k) {
        int const code = code_value(expert, n, k);
        put_native(native_low, low_bits(), n, k, cli.k,
                   code & ((1 << low_bits()) - 1));
        if constexpr (high_bits() != 0)
          put_native(native_high, high_bits(), n, k, cli.k,
                     code >> low_bits());
      }
    std::uint8_t* placed_low = fixture.low.data() + std::size_t(expert) * low_bytes;
    std::uint8_t* placed_high = high_bits()
        ? fixture.high.data() + std::size_t(expert) * high_bytes : nullptr;
    int prepare_rc = 0, recover_rc = 0;
    if constexpr (SCALEFIRST_GROUPED_KPACK_QTYPE == 12) {
      prepare_rc = q4_kpack4::prepare(
          native_low.data(), placed_low, cli.n, cli.k);
      recover_rc = prepare_rc ? prepare_rc : q4_kpack4::recover(
          placed_low, back_low.data(), cli.n, cli.k);
    } else {
      prepare_rc = transform_generic<false>(
          native_low.data(), high_bits() ? native_high.data() : nullptr,
          placed_low, placed_high, cli.n, cli.k);
      recover_rc = prepare_rc ? prepare_rc : transform_generic<true>(
          placed_low, placed_high, back_low.data(),
          high_bits() ? back_high.data() : nullptr, cli.n, cli.k);
    }
    exact = exact && prepare_rc == 0 && recover_rc == 0 &&
            native_low == back_low && native_high == back_high;
  }

  for (int expert = 0; expert < cli.experts; ++expert)
    for (int local = 0; local < fixture.rows[std::size_t(expert)]; ++local) {
      int const row = fixture.offsets[std::size_t(expert)] + local;
      int const active_k = (37 * row + 11 * expert + 5) % cli.k;
      fixture.a[std::size_t(row) * cli.k + active_k] = half_t(1.f);
      int const group = active_k / group_size();
      for (int n = 0; n < cli.n; ++n) {
        std::size_t const metadata =
            (std::size_t(expert) * groups + group) * cli.n + n;
        float const scale = float(fixture.scale[metadata]);
        float const affine = float(fixture.zero[metadata]) -
            center_correction() * scale;
        fixture.golden[std::size_t(row) * cli.n + n] = half_t(
            scale * decoded_value(code_value(expert, n, active_k)) + affine);
      }
    }
  fixture.roundtrip = exact;
  return fixture;
}

void print_samples(std::vector<double> const& samples) {
  std::printf("[");
  for (std::size_t index = 0; index < samples.size(); ++index)
    std::printf("%s%.9f", index ? "," : "", samples[index]);
  std::printf("]");
}

}  // namespace

int main(int argc, char** argv) {
  Cli cli;
  if (!parse_cli(argc, argv, cli)) return 2;
  HostFixture host = make_fixture(cli);
  if (!host.roundtrip || host.total <= 0 || host.empty < 0) {
    std::fprintf(stderr, "SF_GROUPED_FIXTURE_FAIL\n");
    return 2;
  }
  int device = 0;
  if (hggcGetDevice(&device) != hggcSuccess) return 2;
  int const cu = cutlass::KernelHardwareInfo::
      query_device_multiprocessor_count(device);
  if (cu <= 0) return 2;
  std::vector<RegistryRow> rows;
  std::string selection_error;
  if (!select_registry(cli, rows, selection_error)) {
    std::fprintf(stderr, "SF_GROUPED_SELECTION_FAIL reason=%s\n",
                 selection_error.c_str());
    return 2;
  }
  std::uint64_t const rows_hash =
      kpack_grouped_fixture_rows::rows_fnv64(host.rows);
  std::mt19937_64 rng(cli.schedule_seed ^ rows_hash ^
                      (std::uint64_t(cli.n) << 17) ^
                      (std::uint64_t(cli.k) << 33));
  std::shuffle(rows.begin(), rows.end(), rng);

  cutlass::DeviceAllocation<half_t> d_a(host.a.size()),
      d_scale(host.scale.size()), d_zero(host.zero.size()),
      d_output(host.golden.size());
  cutlass::DeviceAllocation<std::uint8_t> d_low(host.low.size()),
      d_high(std::max<std::size_t>(host.high.size(), 1));
  cutlass::DeviceAllocation<int> d_rows(host.rows.size()),
      d_offsets(host.offsets.size());
  cutlass::DeviceAllocation<GS> d_shapes(host.shapes.size());
  cutlass::DeviceAllocation<half_t*> d_output_ptrs(host.rows.size());
  cutlass::DeviceAllocation<DStride> d_output_strides(host.rows.size());
  constexpr std::size_t workspace_bytes = std::size_t(64) << 20;
  cutlass::DeviceAllocation<char> workspace(workspace_bytes);
  d_a.copy_from_host(host.a.data());
  d_scale.copy_from_host(host.scale.data());
  d_zero.copy_from_host(host.zero.data());
  d_low.copy_from_host(host.low.data());
  if (!host.high.empty()) d_high.copy_from_host(host.high.data());
  d_rows.copy_from_host(host.rows.data());
  d_offsets.copy_from_host(host.offsets.data());
  d_shapes.copy_from_host(host.shapes.data());
  std::vector<half_t*> output_ptrs;
  std::vector<DStride> output_strides;
  for (int expert = 0; expert < cli.experts; ++expert) {
    output_ptrs.push_back(
        d_output.get() + std::size_t(host.offsets[std::size_t(expert)]) * cli.n);
    output_strides.push_back(cutlass::make_cute_packed_stride(
        DStride{}, cute::make_shape(
            host.rows[std::size_t(expert)], cli.n, 1)));
  }
  d_output_ptrs.copy_from_host(output_ptrs.data());
  d_output_strides.copy_from_host(output_strides.data());

  Inputs inputs{
      d_a.get(), d_low.get(), host.high.empty() ? nullptr : d_high.get(),
      d_scale.get(), d_zero.get(), d_output.get(), host.golden.data(),
      d_output_ptrs.get(), d_output_strides.get(), d_rows.get(),
      d_shapes.get(), host.shapes.data(), d_offsets.get(), workspace.get(),
      workspace_bytes, host.total, host.maximum, cli.n, cli.k, cli.experts,
      group_size(), host.active, host.empty, cu, nullptr};

  std::printf(
      "SF_GROUPED_SHARD q=%d layout=%d mapping_id=0x%016llx typed_rows=%d "
      "selected_rows=%zu router=%s tokens=%d topk=%d experts=%d "
      "total_rows=%d max_rows=%d active=%d empty=%d "
      "workload=%s router_profile=%s rows_hash=0x%016llx "
      "iterations=%d warmups=%d correctness_repeats=%d "
      "schedule_seed=0x%016llx roundtrip=PASS\n",
      SCALEFIRST_GROUPED_KPACK_QTYPE,
      SCALEFIRST_GROUPED_KPACK_WEIGHT_LAYOUT,
      static_cast<unsigned long long>(
          SCALEFIRST_GROUPED_KPACK_WEIGHT_LAYOUT == 1
              ? q4_kpack4::kMappingId : kquant_kpack::kMappingId),
      SCALEFIRST_GROUPED_GENERATED_TYPED_ROWS, rows.size(),
      cli.rows_file.empty() ? moe_router_fixture::kName : "exact-rows-v1",
      cli.tokens, cli.topk, cli.experts,
      host.total, host.maximum, host.active, host.empty,
      cli.workload_key.c_str(), cli.router_profile.c_str(),
      static_cast<unsigned long long>(rows_hash),
      cli.options.iterations, cli.options.warmups,
      cli.options.correctness_repeats,
      static_cast<unsigned long long>(cli.schedule_seed));
  for (int split : {2, 4, 8})
    std::printf("SF_GROUPED_SPLITK q=%d S=%d status=STRUCTURAL_UNAVAILABLE "
                "reason=NO_GROUPED_SPLITK_KERNEL_OR_REDUCER\n",
                SCALEFIRST_GROUPED_KPACK_QTYPE, split);

  std::size_t measured_cells = 0, structural_cells = 0, records = 0;
  for (std::size_t ordinal = 0; ordinal < rows.size(); ++ordinal) {
    RegistryRow const& row = rows[ordinal];
    Result result;
    if (!row.run(inputs, cli.options, result)) return 2;
    if (result.cells.empty()) return 2;
    for (CellResult const& cell : result.cells) {
      bool const structural = cell.state == State::SharedStorage ||
                              cell.state == State::Occupancy;
      bool const measured = cell.state == State::Measured;
      measured_cells += measured;
      structural_cells += structural;
      std::printf(
          "SF_GROUPED_CELL q=%d layout=%d symbol=%s config=%dx%dx%d_w%dx%d_s%d "
          "scope=FULL_OUTPUT algorithm=%s policy=%s grid=%d occupancy=%d "
          "capacity_b_mask=0x%llx balanced_b_mask=0x%llx state=%s "
          "raw_bad=%llu first_bad=%zu want=0x%04x got=0x%04x "
          "failure_repeat=%d median_us=%.9f min_us=%.9f max_us=%.9f "
          "execution_ordinal=%zu samples=",
          row.qtype, row.weight_layout, row.symbol,
          row.tm, row.tn, row.tk, row.wm, row.wn, row.stages,
          cell.algorithm, cell.policy, cell.grid, cell.occupancy,
          static_cast<unsigned long long>(cell.capacity_b_mask),
          static_cast<unsigned long long>(cell.balanced_b_mask),
          state_name(cell.state),
          static_cast<unsigned long long>(cell.raw_bad), cell.first_bad,
          unsigned(cell.first_want), unsigned(cell.first_got),
          cell.failure_repeat, cell.median_us, cell.min_us, cell.max_us,
          ordinal);
      print_samples(cell.samples_us);
      std::printf("\n");
      ++records;
      if (!measured && !structural) return 1;
    }
  }
  std::printf("SF_GROUPED_COMPLETE q=%d status=PASS rows=%zu cells=%zu "
              "measured=%zu structural=%zu "
              "correctness=RAW_FP16 timing=AFTER_CORRECTNESS "
              "splitk=STRUCTURAL_UNAVAILABLE\n",
              SCALEFIRST_GROUPED_KPACK_QTYPE, rows.size(), records,
              measured_cells, structural_cells);
  return 0;
}
