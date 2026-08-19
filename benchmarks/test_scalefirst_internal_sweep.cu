// One generated binary owns one (qtype, ArtifactTileK, BChunk) shard and any
// number of runtime dense shapes.  Static rejects remain in manifest.json;
// this executable emits every runtime NP/persistent/Split-K coordinate for
// every compiled row and fails closed on missing or extra coordinates.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "cutlass/util/device_memory.h"
#include "scalefirst_internal_sweep_bench.hpp"
#include "xplane_offline.hpp"
#include "scalefirst_registry.inc"

#ifndef SCALEFIRST_SWEEP_QTYPE
#error "SCALEFIRST_SWEEP_QTYPE must match the generated registry"
#endif
#ifndef SCALEFIRST_SWEEP_ARTIFACT_TK
#error "SCALEFIRST_SWEEP_ARTIFACT_TK must match the generated registry"
#endif
#ifndef SCALEFIRST_SWEEP_BCHUNK
#error "SCALEFIRST_SWEEP_BCHUNK must match the generated registry"
#endif
static_assert(SCALEFIRST_SWEEP_QTYPE == SCALEFIRST_GENERATED_QTYPE);
static_assert(SCALEFIRST_SWEEP_ARTIFACT_TK == SCALEFIRST_GENERATED_ARTIFACT_TK);
static_assert(SCALEFIRST_SWEEP_BCHUNK == SCALEFIRST_GENERATED_BCHUNK);

extern "C" int quactlize_ppu_prepare_dense_for_tile(
    std::uint8_t const*, std::uint8_t const*, std::uint8_t*, std::uint8_t*,
    int, int, int, int);
extern "C" int quactlize_ppu_recover_dense_for_tile(
    std::uint8_t const*, std::uint8_t const*, std::uint8_t*, std::uint8_t*,
    int, int, int, int);

namespace scalefirst_internal_sweep_generated {
#define SCALEFIRST_DECLARE(FN,Q,A,TM,TN,TK,WM,WN,ST,BC)               \
  bool FN(scalefirst_internal_sweep::DeviceInputs const&,             \
          scalefirst_internal_sweep::Options const&,                  \
          scalefirst_internal_sweep::RowResult&);
SCALEFIRST_REGISTRY_ROWS(SCALEFIRST_DECLARE)
#undef SCALEFIRST_DECLARE
}

namespace {

using namespace scalefirst_internal_sweep;

std::vector<RegistryRow> registry() {
  return {
#define SCALEFIRST_REGISTER(FN,Q,A,TM,TN,TK,WM,WN,ST,BC)              \
    {#FN,Q,A,TM,TN,TK,WM,WN,ST,BC,                                   \
     &scalefirst_internal_sweep_generated::FN},
    SCALEFIRST_REGISTRY_ROWS(SCALEFIRST_REGISTER)
#undef SCALEFIRST_REGISTER
  };
}

struct Shape { int m = 1, n = 4096, k = 4096; };
struct Cli {
  int iterations = 5, repeats = 2;
  std::uint64_t schedule_seed = UINT64_C(0x6a09e667f3bcc909);
  unsigned algorithm_mask = Options::kAllAlgorithms;
  std::string symbol_file;
  std::vector<Shape> shapes;
};

bool parse_shape(char const* text, Shape& shape) {
  char tail = 0;
  return std::sscanf(text, "%dx%dx%d%c", &shape.m, &shape.n, &shape.k,
                     &tail) == 3 && shape.m > 0 && shape.n > 0 && shape.k > 0;
}

bool parse_cli(int argc, char** argv, Cli& cli) {
  for (int i = 1; i < argc; ++i) {
    if (!std::strncmp(argv[i], "--shape=", 8)) {
      Shape shape;
      if (!parse_shape(argv[i] + 8, shape)) return false;
      cli.shapes.push_back(shape);
    } else if (!std::strncmp(argv[i], "--iterations=", 13)) {
      cli.iterations = std::atoi(argv[i] + 13);
    } else if (!std::strncmp(argv[i], "--correctness-repeats=", 22)) {
      cli.repeats = std::atoi(argv[i] + 22);
    } else if (!std::strncmp(argv[i], "--schedule-seed=", 16)) {
      char* end = nullptr;
      cli.schedule_seed = std::strtoull(argv[i] + 16, &end, 0);
      if (!end || *end) return false;
    } else if (!std::strncmp(argv[i], "--algorithm=", 12)) {
      char const* value = argv[i] + 12;
      if (!std::strcmp(value, "all"))
        cli.algorithm_mask = Options::kAllAlgorithms;
      else if (!std::strcmp(value, "nonpersistent"))
        cli.algorithm_mask = Options::kNonPersistent;
      else if (!std::strcmp(value, "persistent"))
        cli.algorithm_mask = Options::kPersistent;
      else if (!std::strcmp(value, "split"))
        cli.algorithm_mask = Options::kSplitK;
      else if (!std::strcmp(value, "full-output"))
        cli.algorithm_mask = Options::kNonPersistent | Options::kPersistent;
      else return false;
    } else if (!std::strncmp(argv[i], "--symbol-file=", 14)) {
      cli.symbol_file = argv[i] + 14;
      if (cli.symbol_file.empty()) return false;
    } else return false;
  }
  if (cli.shapes.empty()) cli.shapes.push_back({1, 4096, 4096});
  return cli.iterations > 0 && cli.repeats > 0 && cli.algorithm_mask != 0;
}

bool selected_registry(Cli const& cli, std::vector<RegistryRow>& selected,
                       std::string& error) {
  auto const all = registry();
  if (cli.symbol_file.empty()) {
    selected = all;
    return true;
  }
  std::ifstream stream(cli.symbol_file);
  if (!stream) {
    error = "cannot open symbol file: " + cli.symbol_file;
    return false;
  }
  std::unordered_set<std::string> requested;
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line.find_first_of(" \t") != std::string::npos) {
      error = "symbol file contains an empty/whitespace-bearing record";
      return false;
    }
    if (!requested.insert(line).second) {
      error = "symbol file contains a duplicate: " + line;
      return false;
    }
  }
  if (!stream.eof() || requested.empty()) {
    error = requested.empty() ? "symbol file is empty" :
                                "failed while reading symbol file";
    return false;
  }
  for (auto const& row : all) {
    auto found = requested.find(row.symbol);
    if (found != requested.end()) {
      selected.push_back(row);
      requested.erase(found);
    }
  }
  if (!requested.empty()) {
    error = "symbol file names an unknown generated symbol: " +
            *requested.begin();
    return false;
  }
  return true;
}

constexpr int low_bits() {
  return SCALEFIRST_SWEEP_QTYPE == 8 ? 8 :
         SCALEFIRST_SWEEP_QTYPE == 10 || SCALEFIRST_SWEEP_QTYPE == 11 ? 2 : 4;
}
constexpr int high_bits() {
  return SCALEFIRST_SWEEP_QTYPE == 11 || SCALEFIRST_SWEEP_QTYPE == 13 ? 1 :
         SCALEFIRST_SWEEP_QTYPE == 14 ? 2 : 0;
}
constexpr int group_size() {
  return SCALEFIRST_SWEEP_QTYPE == 8 || SCALEFIRST_SWEEP_QTYPE == 12 ||
         SCALEFIRST_SWEEP_QTYPE == 13 ? 32 : 16;
}
constexpr bool has_zero() {
  return SCALEFIRST_SWEEP_QTYPE == 10 || SCALEFIRST_SWEEP_QTYPE == 12 ||
         SCALEFIRST_SWEEP_QTYPE == 13;
}

int code_value(int n, int k) {
  int const logical = ((13 * n + 7 * k + 3) % 15) - 7;
  switch (SCALEFIRST_SWEEP_QTYPE) {
    case 8: return logical + 128;
    case 10: return (logical + 8) & 3;
    case 11: return logical < -4 ? 0 : logical > 3 ? 7 : logical + 4;
    case 12: return logical + 8;
    // Exercise the Q5 high plane on every fixture.  The ScaleZero converter
    // still decodes code-8; choosing 17..31 changes only the test data, not
    // that bias convention.  The previous 1..15 range left the entire high
    // plane zero and let a disconnected high-plane reader pass.
    case 13: return logical + 24;
    case 14: return logical + 32;
  }
  return 0;
}

int decoded_value(int code) {
  switch (SCALEFIRST_SWEEP_QTYPE) {
    case 8: return code - 128;
    case 10: return code;
    case 11: return code - 4;
    case 12: return code - 8;
    case 13: return code - 8;
    case 14: return code - 32;
  }
  return 0;
}

void put_native(std::vector<std::uint8_t>& plane, int bits, int n, int k,
                int K, int value) {
  std::uint64_t const bit = (std::uint64_t(n) * K + k) * bits;
  plane[bit >> 3] |= std::uint8_t(value << (bit & 7));
}

struct Fixture {
  std::vector<half_t> a, scales, zeros, golden;
  std::vector<std::uint8_t> low_native, high_native, low, high;
  bool exact = false, roundtrip = false, high_plane_covered = false;
};

Fixture make_fixture(Shape shape) {
  Fixture f;
  int constexpr LB = low_bits(), HB = high_bits(), GS = group_size();
  std::size_t const codes = std::size_t(shape.n) * shape.k;
  f.a.assign(std::size_t(shape.m) * shape.k, half_t(0.f));
  f.scales.resize(std::size_t(shape.k / GS) * shape.n);
  if constexpr (has_zero()) f.zeros.resize(f.scales.size());
  f.golden.resize(std::size_t(shape.m) * shape.n);
  f.low_native.assign(codes * LB / 8, 0);
  f.high_native.assign(HB ? codes * HB / 8 : 0, 0);
  f.low.resize(f.low_native.size());
  f.high.resize(f.high_native.size());

  // One exact nonzero in each of the eight K eighths.  A=+/-0.5,
  // scale in {1,2,4}, zero in {-3,0,3}; every partial and final value is an
  // exact half-integer bounded far below fp16's exact range.
  std::array<int, 8> active{};
  for (int s = 0; s < 8; ++s) {
    int const begin = s * shape.k / 8;
    int const span = shape.k / 8;
    active[s] = begin + ((37 * s + 11) % span);
    for (int m = 0; m < shape.m; ++m)
      f.a[std::size_t(m) * shape.k + active[s]] =
          half_t(((m + s) & 1) ? -0.5f : 0.5f);
  }
  for (int g = 0; g < shape.k / GS; ++g)
    for (int n = 0; n < shape.n; ++n) {
      f.scales[std::size_t(g) * shape.n + n] =
          half_t(float(1 << ((17 * g + 29 * n + 1) % 3)));
      if constexpr (has_zero())
        f.zeros[std::size_t(g) * shape.n + n] =
            half_t(float(((11 * g + 7 * n) % 3 - 1) * 3));
    }

  std::vector<std::uint8_t> kmajor(codes);
  for (int n = 0; n < shape.n; ++n)
    for (int k = 0; k < shape.k; ++k) {
      int const code = code_value(n, k);
      kmajor[std::size_t(k) * shape.n + n] = std::uint8_t(code);
      put_native(f.low_native, LB, n, k, shape.k,
                 code & ((1 << LB) - 1));
      if constexpr (HB != 0)
        put_native(f.high_native, HB, n, k, shape.k, code >> LB);
    }
  if constexpr (SCALEFIRST_SWEEP_QTYPE == 8) {
    static_assert(SCALEFIRST_SWEEP_ARTIFACT_TK == 32,
                  "Q8 has one canonical A32 artifact");
    xplane::place_derived<8,64,64,32,32,32,1,32>(
        reinterpret_cast<std::int8_t*>(f.low.data()), kmajor,
        shape.n, shape.k);
    std::vector<std::uint8_t> back;
    xplane::recover_derived<8,64,64,32,32,32,1,32>(
        reinterpret_cast<std::int8_t const*>(f.low.data()), back,
        shape.n, shape.k);
    f.roundtrip = back == kmajor;
  } else {
    if (quactlize_ppu_prepare_dense_for_tile(
            f.low_native.data(), HB ? f.high_native.data() : nullptr,
            f.low.data(), HB ? f.high.data() : nullptr,
            shape.n, shape.k, SCALEFIRST_SWEEP_QTYPE,
            SCALEFIRST_SWEEP_ARTIFACT_TK) != 0) return f;
    std::vector<std::uint8_t> low_back(f.low_native.size());
    std::vector<std::uint8_t> high_back(f.high_native.size());
    f.roundtrip = quactlize_ppu_recover_dense_for_tile(
        f.low.data(), HB ? f.high.data() : nullptr,
        low_back.data(), HB ? high_back.data() : nullptr,
        shape.n, shape.k, SCALEFIRST_SWEEP_QTYPE,
        SCALEFIRST_SWEEP_ARTIFACT_TK) == 0 &&
        low_back == f.low_native && high_back == f.high_native;
  }
  if constexpr (HB == 0) {
    f.high_plane_covered = true;
  } else {
    f.high_plane_covered = std::any_of(
        f.high_native.begin(), f.high_native.end(),
        [](std::uint8_t value) { return value != 0; });
  }

  bool exact = true;
  for (int m = 0; m < shape.m; ++m)
    for (int n = 0; n < shape.n; ++n) {
      float sum = 0;
      for (int k : active) {
        int const g = k / GS;
        float const scale = float(f.scales[std::size_t(g) * shape.n + n]);
        float const zero = has_zero() ?
            float(f.zeros[std::size_t(g) * shape.n + n]) : 0.f;
        sum += float(f.a[std::size_t(m) * shape.k + k]) *
            (scale * decoded_value(code_value(n, k)) + zero);
      }
      half_t rounded(sum);
      exact &= float(rounded) == sum;
      f.golden[std::size_t(m) * shape.n + n] = rounded;
    }
  f.exact = exact;
  return f;
}

template <class T>
void copy_to(cutlass::DeviceAllocation<T>& allocation,
             std::vector<T> const& source) {
  if (!source.empty()) allocation.copy_from_host(source.data());
}

void print_samples(std::vector<double> const& samples) {
  std::printf("[");
  for (std::size_t i = 0; i < samples.size(); ++i)
    std::printf("%s%.9f", i ? "," : "", samples[i]);
  std::printf("]");
}

int run_shape(Shape shape, Cli const& cli, int device, int cu,
              std::vector<RegistryRow> const& rows) {
  if (shape.n % 256 || shape.k % 256 || shape.k % 8) {
    std::fprintf(stderr, "shape %dx%dx%d violates resident/split alignment\n",
                 shape.m, shape.n, shape.k);
    return 2;
  }
  Fixture fixture = make_fixture(shape);
  if (!fixture.roundtrip || !fixture.exact || !fixture.high_plane_covered) {
    std::fprintf(stderr,
                 "fixture failed roundtrip=%d exact=%d high_plane_covered=%d\n",
                 int(fixture.roundtrip), int(fixture.exact),
                 int(fixture.high_plane_covered));
    return 2;
  }
  cutlass::DeviceAllocation<half_t> dA(fixture.a.size());
  cutlass::DeviceAllocation<std::uint8_t> dLow(fixture.low.size());
  cutlass::DeviceAllocation<std::uint8_t> dHigh(fixture.high.size());
  cutlass::DeviceAllocation<half_t> dScale(fixture.scales.size());
  cutlass::DeviceAllocation<half_t> dZero(fixture.zeros.size());
  cutlass::DeviceAllocation<half_t> dOutput(std::size_t(shape.m) * shape.n);
  std::size_t const workspace_bytes =
      std::size_t(shape.m) * shape.n * 8 * sizeof(float) + 4096;
  cutlass::DeviceAllocation<char> dWorkspace(workspace_bytes);
  copy_to(dA, fixture.a); copy_to(dLow, fixture.low); copy_to(dHigh, fixture.high);
  copy_to(dScale, fixture.scales); copy_to(dZero, fixture.zeros);
  DeviceInputs inputs{
      dA.get(), dLow.get(), fixture.high.empty() ? nullptr : dHigh.get(),
      dScale.get(), fixture.zeros.empty() ? nullptr : dZero.get(),
      dOutput.get(), dWorkspace.get(), workspace_bytes, fixture.golden.data(),
      shape.m, shape.n, shape.k, device, cu};
  Options options{cli.iterations, cli.repeats, true, cli.algorithm_mask};
  std::vector<std::size_t> order(rows.size());
  std::iota(order.begin(), order.end(), 0);
  std::mt19937_64 rng(cli.schedule_seed ^ std::uint64_t(shape.m) ^
                      (std::uint64_t(shape.n) << 17) ^
                      (std::uint64_t(shape.k) << 33));
  std::shuffle(order.begin(), order.end(), rng);
  std::size_t runtime_cells = 0, measured_cells = 0, records = 0;
  for (std::size_t ordinal = 0; ordinal < order.size(); ++ordinal) {
    auto const& registry_row = rows[order[ordinal]];
    RowResult result;
    std::printf("SF_ATTEMPT shape=%dx%dx%d ordinal=%zu/%zu symbol=%s\n",
                shape.m, shape.n, shape.k, ordinal + 1, order.size(),
                registry_row.symbol);
    std::fflush(stdout);
    if (!registry_row.run(inputs, options, result)) {
      if (result.cells.empty()) {
        std::fprintf(stderr,
                     "SF_FATAL symbol=%s shape=%dx%dx%d algorithm=NONE "
                     "state=INPUT_OR_SETUP step=BEFORE_CELL\n",
                     registry_row.symbol, shape.m, shape.n, shape.k);
      } else {
        auto const& failed = result.cells.back();
        std::fprintf(
            stderr,
            "SF_FATAL symbol=%s shape=%dx%dx%d algorithm=%s state=%s "
            "step=%s repeat=%d raw_bad=%llu first_bad=%zu "
            "want=0x%04x got=0x%04x\n",
            registry_row.symbol, shape.m, shape.n, shape.k,
            failed.algorithm, state_name(failed.state), failed.failure_step,
            failed.failure_repeat,
            static_cast<unsigned long long>(failed.raw_bad),
            failed.first_bad_index, unsigned(failed.first_bad_want),
            unsigned(failed.first_bad_got));
      }
      return 1;
    }
    runtime_cells += result.cells.size();
    for (auto const& cell : result.cells) {
      bool const measured = cell.state == State::Measured;
      measured_cells += measured;
      int const sample_count = measured ? int(cell.samples_us.size()) : 1;
      if (measured && sample_count != cli.iterations) {
        std::fprintf(stderr, "sample denominator drift for %s/%s\n",
                     registry_row.symbol, cell.algorithm);
        return 1;
      }
      for (int sample = 0; sample < sample_count; ++sample) {
        double const us = measured ? cell.samples_us[std::size_t(sample)] : 0.;
        double const tflops = us > 0 ?
            (2. * shape.m * shape.n * shape.k) / (us * 1.e6) : 0.;
        double const mfu = tflops / 500. * 100.;
        double const distinct_bytes =
            double(shape.m) * shape.k * 2. +
            double(shape.n) * shape.k * (low_bits() + high_bits()) / 8. +
            double(shape.n) * (shape.k / group_size()) *
                (has_zero() ? 4. : 2.) +
            double(shape.m) * shape.n * 2.;
        double const mbu = us > 0 ? distinct_bytes / (us * 1.e3) / 2766. * 100. : 0.;
        std::printf(
            "SF_CELL {\"shape\":\"%dx%dx%d\",\"qtype\":%d,"
            "\"artifact_tile_k\":%d,\"bchunk\":%d,\"symbol\":\"%s\","
            "\"config\":\"%dx%dx%d_w%dx%d_s%d_bc%d\","
            "\"algorithm\":\"%s\",\"metric_scope\":\"%s\","
            "\"policy\":\"%s\",\"split\":%d,\"grid\":%d,"
            "\"occupancy\":%d,\"capacity_b_mask\":\"0x%llx\","
            "\"balanced_b_mask\":\"0x%llx\",\"status\":\"%s\","
            "\"reason\":\"%s\",\"sample\":%d,\"sample_us\":%.9f,"
            "\"MFU_pct\":%.9f,\"distinct_MBU_model_pct\":%.9f,"
            "\"raw_bad\":%llu,\"fingerprint\":\"0x%llx\","
            "\"reducer_correctness_untimed\":%d,\"partial_bytes\":%zu,"
            "\"shipping_smem\":%zu,\"persistent_smem\":%zu,"
            "\"split_smem\":%zu,\"execution_ordinal\":%zu}\n",
            shape.m, shape.n, shape.k, SCALEFIRST_SWEEP_QTYPE,
            SCALEFIRST_SWEEP_ARTIFACT_TK, SCALEFIRST_SWEEP_BCHUNK,
            registry_row.symbol, registry_row.tm, registry_row.tn,
            registry_row.tk, registry_row.wm, registry_row.wn,
            registry_row.stages, registry_row.bchunk, cell.algorithm,
            cell.metric_scope, cell.policy, cell.split, cell.grid,
            cell.occupancy,
            static_cast<unsigned long long>(cell.capacity_b_mask),
            static_cast<unsigned long long>(cell.balanced_b_mask),
            measured ? "MEASURED" : "INADMISSIBLE", state_name(cell.state),
            sample, us, mfu, mbu,
            static_cast<unsigned long long>(cell.raw_bad),
            static_cast<unsigned long long>(cell.fingerprint),
            int(cell.reducer_correctness_untimed), cell.partial_bytes,
            cell.shipping_smem, cell.persistent_smem, cell.split_smem, ordinal);
        ++records;
      }
    }
  }
  std::printf(
      "SF_COMPLETE status=COMPLETE shape=%dx%dx%d typed_rows=%zu "
      "runtime_cells=%zu measured_cells=%zu records=%zu iterations=%d "
      "fixture=ORDER-INDEPENDENT+FP16-EXACT roundtrip=PASS "
      "high_plane_coverage=PASS\n",
      shape.m, shape.n, shape.k, rows.size(), runtime_cells, measured_cells,
      records, cli.iterations);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Cli cli;
  if (!parse_cli(argc, argv, cli)) {
    std::fprintf(stderr,
        "usage: %s [--shape=MxNxK] [--iterations=N] "
        "[--correctness-repeats=N] [--schedule-seed=N] "
        "[--algorithm=all|nonpersistent|persistent|split|full-output] "
        "[--symbol-file=PATH]\n", argv[0]);
    return 2;
  }
  std::vector<RegistryRow> rows;
  std::string selection_error;
  if (!selected_registry(cli, rows, selection_error)) {
    std::fprintf(stderr, "SF_SELECTION_FAIL reason=%s\n",
                 selection_error.c_str());
    return 2;
  }
  int device = 0;
  if (hggcGetDevice(&device) != hggcSuccess) return 2;
  int const cu = cutlass::KernelHardwareInfo::
      query_device_multiprocessor_count(device);
  if (cu <= 0) return 2;
  std::printf(
      "SF_SHARD qtype=%d artifact_tile_k=%d bchunk=%d typed_rows=%d "
      "selected_rows=%zu algorithm_mask=0x%x device=%d cu=%d "
      "iterations=%d correctness_repeats=%d "
      "schedule_seed=0x%llx\n",
      SCALEFIRST_SWEEP_QTYPE, SCALEFIRST_SWEEP_ARTIFACT_TK,
      SCALEFIRST_SWEEP_BCHUNK, SCALEFIRST_GENERATED_TYPED_ROWS,
      rows.size(), cli.algorithm_mask, device, cu, cli.iterations, cli.repeats,
      static_cast<unsigned long long>(cli.schedule_seed));
  for (auto const& shape : cli.shapes) {
    int const rc = run_shape(shape, cli, device, cu, rows);
    if (rc) return rc;
  }
  return 0;
}
