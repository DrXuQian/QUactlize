/***************************************************************************************************
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Exact timing census for the shipping dense FullyQuantized Split-K reducer.
 *
 * The generated case list contains one row per unique (M,N,S) in the route
 * workload denominator.  This file instantiates the same checked reduction
 * handle used by dense_splitk_parallel_ppu::KernelTypes; it does not contain a
 * second reduction algorithm.  Its fixture and validator are diagnostic-only:
 * they produce deterministic FP32 partial planes, poison D, and compare every
 * output bit with an independently expressed fixed-order golden before timing.
 **************************************************************************************************/

#include <hggc_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "cutlass/cutlass.h"
#include "actlize_extensions/cutlass/gemm/device/ppu_mixed_input_splitk_parallel.hpp"
#include "fq_splitk_reducer_lookup_cases.inc"

namespace splitk = cutlass::gemm::device::splitk_parallel;

namespace {

using half_t = cutlass::half_t;
using Reduction = splitk::PpuMixedInputSplitKParallelM1FastReduction<2>;

constexpr int kCanonicalWarmups = 3;
constexpr int kCanonicalSamples = 11;
constexpr int kThreads = 256;
constexpr int kMaximumFixtureBlocks = 4096;
constexpr unsigned char kOutputPoisonByte = 0x7b;
constexpr uint32_t kNoBadIndex = 0xffffffffu;

struct Case {
  int ordinal;
  int64_t m;
  int64_t n;
  int split;
  uint64_t workspace_bytes;
  uint64_t output_bytes;
  char const* case_id;
  char const* expected_implementation;
};

#define FQ_REDUCER_CASE(ORDINAL, M, N, S, WORKSPACE, OUTPUT, ID, IMPL) \
  {ORDINAL, M, N, S, UINT64_C(WORKSPACE), UINT64_C(OUTPUT), ID, IMPL},
constexpr Case kCases[] = {FQ_SPLITK_REDUCER_CASES(FQ_REDUCER_CASE)};
#undef FQ_REDUCER_CASE

static_assert(sizeof(kCases) / sizeof(kCases[0]) ==
                  FQ_SPLITK_REDUCER_CASE_COUNT,
              "compiled reducer case denominator differs from generated plan");
static_assert(FQ_SPLITK_REDUCER_CASE_COUNT == 1035,
              "canonical reducer lookup denominator drifted");
static_assert(std::is_same_v<
                  Reduction::Fallback,
                  splitk::PpuMixedInputSplitKParallelReduction<8>>,
              "shipping generic fallback implementation changed");

struct Cli {
  int warmups = kCanonicalWarmups;
  int samples = kCanonicalSamples;
  int round = 1;
  int case_begin = 0;
  int case_end = FQ_SPLITK_REDUCER_CASE_COUNT;
  uint64_t schedule_seed = 0;
  bool schedule_seed_set = false;
  bool plant_output_fault = false;
};

bool parse_positive(char const* text, int& value) {
  char* end = nullptr;
  long const parsed = std::strtol(text, &end, 10);
  if (!text || !text[0] || !end || *end || parsed <= 0 ||
      parsed > std::numeric_limits<int>::max()) return false;
  value = int(parsed);
  return true;
}

bool parse_nonnegative(char const* text, int& value) {
  char* end = nullptr;
  long const parsed = std::strtol(text, &end, 10);
  if (!text || !text[0] || !end || *end || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) return false;
  value = int(parsed);
  return true;
}

bool parse_cli(int argc, char** argv, Cli& cli) {
  for (int index = 1; index < argc; ++index) {
    auto value = [&](char const* prefix) -> char const* {
      std::size_t const size = std::strlen(prefix);
      return std::strncmp(argv[index], prefix, size) == 0
          ? argv[index] + size : nullptr;
    };
    if (char const* v = value("--warmups=")) {
      if (!parse_positive(v, cli.warmups)) return false;
    } else if (char const* v = value("--samples=")) {
      if (!parse_positive(v, cli.samples)) return false;
    } else if (char const* v = value("--round=")) {
      if (!parse_positive(v, cli.round)) return false;
    } else if (char const* v = value("--case-begin=")) {
      if (!parse_nonnegative(v, cli.case_begin)) return false;
    } else if (char const* v = value("--case-end=")) {
      if (!parse_nonnegative(v, cli.case_end)) return false;
    } else if (char const* v = value("--schedule-seed=")) {
      char* end = nullptr;
      cli.schedule_seed = std::strtoull(v, &end, 0);
      if (!v[0] || !end || *end) return false;
      cli.schedule_seed_set = true;
    } else if (char const* v = value("--plant-output-fault=")) {
      if (!std::strcmp(v, "0")) cli.plant_output_fault = false;
      else if (!std::strcmp(v, "1")) cli.plant_output_fault = true;
      else return false;
    } else {
      return false;
    }
  }
  return cli.warmups == kCanonicalWarmups &&
      cli.samples == kCanonicalSamples && cli.schedule_seed_set &&
      cli.case_begin >= 0 && cli.case_begin < cli.case_end &&
      cli.case_end <= FQ_SPLITK_REDUCER_CASE_COUNT;
}

bool runtime_ok(hggcError_t status, char const* operation,
                Case const* row = nullptr) {
  if (status == hggcSuccess) return true;
  std::fprintf(
      stderr,
      "FQ_REDUCER_LOOKUP_ERROR case_id=%s operation=%s code=%d text=%s\n",
      row ? row->case_id : "NONE", operation, int(status),
      hggcGetErrorString(status));
  return false;
}

CUTLASS_HOST_DEVICE
float fixture_value(int split, uint64_t element) {
  // Period 257 carries a conversion sentinel; the ordinary period-31 body
  // varies independently with split and absolute [M,N] element offset.
  if (element % UINT64_C(257) == 0) {
    constexpr float sentinel[8] = {
        2048.0f, 1.0f, -2048.0f, 0.0f,
        1024.0f, 0.5f, -1024.0f, 0.0f};
    return sentinel[split];
  }
  int const value = int((element * UINT64_C(17) +
                         uint64_t(split) * UINT64_C(29) + UINT64_C(11)) %
                        UINT64_C(31)) - 15;
  return float(value) * 0.25f;
}

CUTLASS_HOST_DEVICE
uint16_t golden_bits(int partitions, uint64_t element) {
  float accumulator = 0.0f;
  for (int split = 0; split < partitions; ++split) {
    accumulator += fixture_value(split, element);
  }
  return half_t(accumulator).raw();
}

__global__ void initialize_partials_kernel(
    float* workspace, uint64_t elements_per_partition,
    uint64_t total_elements) {
  uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  uint64_t const stride = uint64_t(gridDim.x) * blockDim.x;
  for (; index < total_elements; index += stride) {
    int const split = int(index / elements_per_partition);
    uint64_t const element = index -
        uint64_t(split) * elements_per_partition;
    workspace[index] = fixture_value(split, element);
  }
}

__global__ void validate_output_kernel(
    half_t const* output, uint64_t elements, int partitions,
    uint32_t* bad_count, uint32_t* first_bad) {
  uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  uint64_t const stride = uint64_t(gridDim.x) * blockDim.x;
  for (; index < elements; index += stride) {
    if (output[index].raw() != golden_bits(partitions, index)) {
      atomicAdd(bad_count, 1u);
      atomicMin(first_bad, uint32_t(index));
    }
  }
}

__global__ void plant_output_fault_kernel(half_t* output) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    output[0] = half_t::bitcast(uint16_t(output[0].raw() ^ 1u));
  }
}

int fixture_blocks(uint64_t elements) {
  uint64_t const needed = (elements + kThreads - 1) / kThreads;
  return int(std::min<uint64_t>(needed, kMaximumFixtureBlocks));
}

uint64_t splitmix64(uint64_t& state) {
  uint64_t value = (state += UINT64_C(0x9e3779b97f4a7c15));
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

uint64_t case_order_hash(std::vector<int> const& order) {
  uint64_t hash = UINT64_C(1469598103934665603);
  for (int ordinal : order) {
    uint32_t value = uint32_t(ordinal);
    for (int byte = 0; byte < 4; ++byte) {
      hash ^= uint8_t(value >> (8 * byte));
      hash *= UINT64_C(1099511628211);
    }
  }
  return hash;
}

bool initialize_fixture(Case const& row, float* workspace, half_t* output) {
  uint64_t const elements = uint64_t(row.m) * uint64_t(row.n);
  uint64_t const partials = elements * uint64_t(row.split);
  initialize_partials_kernel<<<fixture_blocks(partials), kThreads>>>(
      workspace, elements, partials);
  return runtime_ok(hggcGetLastError(), "initialize-partials-launch", &row) &&
      runtime_ok(hggcMemset(output, kOutputPoisonByte,
                            std::size_t(row.output_bytes)),
                 "poison-output", &row) &&
      runtime_ok(hggcDeviceSynchronize(), "initialize-fixture-sync", &row);
}

struct CheckResult {
  uint32_t bad = 0;
  uint32_t first = kNoBadIndex;
  uint16_t want = 0;
  uint16_t got = 0;
};

bool validate_output(Case const& row, half_t const* output,
                     uint32_t* device_check, CheckResult& result) {
  uint64_t const elements = uint64_t(row.m) * uint64_t(row.n);
  if (!runtime_ok(hggcMemset(device_check, 0, sizeof(uint32_t)),
                  "clear-bad-count", &row) ||
      !runtime_ok(hggcMemset(device_check + 1, 0xff, sizeof(uint32_t)),
                  "clear-first-bad", &row)) return false;
  validate_output_kernel<<<fixture_blocks(elements), kThreads>>>(
      output, elements, row.split, device_check, device_check + 1);
  if (!runtime_ok(hggcGetLastError(), "validate-output-launch", &row) ||
      !runtime_ok(hggcDeviceSynchronize(), "validate-output-sync", &row) ||
      !runtime_ok(hggcMemcpy(&result.bad, device_check, sizeof(uint32_t),
                            hggcMemcpyDeviceToHost),
                  "copy-bad-count", &row) ||
      !runtime_ok(hggcMemcpy(&result.first, device_check + 1,
                            sizeof(uint32_t), hggcMemcpyDeviceToHost),
                  "copy-first-bad", &row)) return false;
  if (result.bad && uint64_t(result.first) < elements) {
    half_t observed;
    if (!runtime_ok(hggcMemcpy(
            &observed, output + result.first, sizeof(observed),
            hggcMemcpyDeviceToHost), "copy-first-value", &row)) return false;
    result.want = golden_bits(row.split, result.first);
    result.got = observed.raw();
  }
  return true;
}

struct EventPair {
  hggcEvent_t start{};
  hggcEvent_t stop{};

  bool initialize(Case const& row) {
    return runtime_ok(hggcEventCreate(&start), "create-start-event", &row) &&
        runtime_ok(hggcEventCreate(&stop), "create-stop-event", &row);
  }
  ~EventPair() {
    if (stop) hggcEventDestroy(stop);
    if (start) hggcEventDestroy(start);
  }
};

char const* actual_implementation(Reduction const& reducer) {
  return reducer.fast_path_selected_for_diagnostics()
      ? "M1_FAST_E2" : "GENERIC_E8";
}

std::pair<uint64_t, uint64_t> launch_shape(
    Case const& row, bool fast) {
  if (fast) {
    dim3 grid{};
    if (row.split == 2)
      grid = Reduction::KernelS2::grid_shape(row.n);
    else if (row.split == 4)
      grid = Reduction::KernelS4::grid_shape(row.n);
    else
      grid = Reduction::KernelS8::grid_shape(row.n);
    dim3 const block = Reduction::KernelS2::block_shape();
    return {uint64_t(grid.x) * grid.y * grid.z,
            uint64_t(block.x) * block.y * block.z};
  }
  using Kernel = Reduction::Fallback::Kernel;
  dim3 const grid = Kernel::grid_shape(row.m, row.n);
  dim3 const block = Kernel::block_shape();
  return {uint64_t(grid.x) * grid.y * grid.z,
          uint64_t(block.x) * block.y * block.z};
}

bool run_case(Case const& row, int execution_ordinal, Cli const& cli) {
  void* workspace_raw = nullptr;
  void* output_raw = nullptr;
  void* check_raw = nullptr;
  auto release = [&] {
    if (check_raw) hggcFree(check_raw);
    if (output_raw) hggcFree(output_raw);
    if (workspace_raw) hggcFree(workspace_raw);
  };
  if (!runtime_ok(hggcMalloc(&workspace_raw, std::size_t(row.workspace_bytes)),
                  "allocate-workspace", &row) ||
      !runtime_ok(hggcMalloc(&output_raw, std::size_t(row.output_bytes)),
                  "allocate-output", &row) ||
      !runtime_ok(hggcMalloc(&check_raw, 2 * sizeof(uint32_t)),
                  "allocate-check", &row)) {
    release();
    return false;
  }
  auto* workspace = static_cast<float*>(workspace_raw);
  auto* output = static_cast<half_t*>(output_raw);
  auto* device_check = static_cast<uint32_t*>(check_raw);
  if (!initialize_fixture(row, workspace, output)) {
    release();
    return false;
  }

  typename Reduction::Arguments arguments{
      row.m, row.n, row.split, workspace,
      std::size_t(row.workspace_bytes), output, row.n};
  if (Reduction::can_implement(arguments) != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
                 "FQ_REDUCER_LOOKUP_ERROR case_id=%s operation=can-implement\n",
                 row.case_id);
    release();
    return false;
  }
  Reduction reducer;
  if (reducer.initialize(arguments) != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
                 "FQ_REDUCER_LOOKUP_ERROR case_id=%s operation=initialize\n",
                 row.case_id);
    release();
    return false;
  }
  char const* implementation = actual_implementation(reducer);
  if (std::strcmp(implementation, row.expected_implementation)) {
    std::fprintf(
        stderr,
        "FQ_REDUCER_LOOKUP_ERROR case_id=%s operation=implementation "
        "want=%s got=%s\n",
        row.case_id, row.expected_implementation, implementation);
    release();
    return false;
  }
  auto const [grid_ctas, block_threads] = launch_shape(
      row, reducer.fast_path_selected_for_diagnostics());

  if (reducer.run(nullptr) != cutlass::Status::kSuccess ||
      !runtime_ok(hggcDeviceSynchronize(), "correctness-reducer-sync", &row)) {
    std::fprintf(stderr,
                 "FQ_REDUCER_LOOKUP_ERROR case_id=%s operation=correctness-run\n",
                 row.case_id);
    release();
    return false;
  }
  if (cli.plant_output_fault) {
    plant_output_fault_kernel<<<1, 1>>>(output);
    if (!runtime_ok(hggcGetLastError(), "plant-output-fault", &row)) {
      release();
      return false;
    }
  }
  CheckResult correctness;
  if (!validate_output(row, output, device_check, correctness) ||
      correctness.bad != 0) {
    std::printf(
        "FQ_REDUCER_LOOKUP_CASE ordinal=%d execution_ordinal=%d "
        "case_id=%s M=%" PRId64 " N=%" PRId64 " S=%d "
        "partial_dtype=fp32 output_dtype=fp16 implementation=%s "
        "workspace_bytes=%" PRIu64 " output_bytes=%" PRIu64 " "
        "grid_ctas=%" PRIu64 " block_threads=%" PRIu64 " raw_bad=%u "
        "first_bad=%u first_want=0x%04x first_got=0x%04x status=FAIL\n",
        row.ordinal, execution_ordinal, row.case_id, row.m, row.n, row.split,
        implementation, row.workspace_bytes, row.output_bytes, grid_ctas,
        block_threads, correctness.bad, correctness.first,
        unsigned(correctness.want), unsigned(correctness.got));
    release();
    return false;
  }

  for (int warmup = 0; warmup < cli.warmups; ++warmup) {
    if (reducer.run(nullptr) != cutlass::Status::kSuccess) {
      std::fprintf(stderr,
                   "FQ_REDUCER_LOOKUP_ERROR case_id=%s operation=warmup index=%d\n",
                   row.case_id, warmup);
      release();
      return false;
    }
  }
  if (!runtime_ok(hggcDeviceSynchronize(), "warmup-sync", &row)) {
    release();
    return false;
  }

  EventPair events;
  if (!events.initialize(row)) {
    release();
    return false;
  }
  std::vector<double> samples;
  samples.reserve(std::size_t(cli.samples));
  for (int sample = 0; sample < cli.samples; ++sample) {
    if (!runtime_ok(hggcEventRecord(events.start, nullptr),
                    "record-start", &row) ||
        reducer.run(nullptr) != cutlass::Status::kSuccess ||
        !runtime_ok(hggcEventRecord(events.stop, nullptr),
                    "record-stop", &row) ||
        !runtime_ok(hggcEventSynchronize(events.stop),
                    "synchronize-stop", &row)) {
      std::fprintf(stderr,
                   "FQ_REDUCER_LOOKUP_ERROR case_id=%s operation=timed-run index=%d\n",
                   row.case_id, sample);
      release();
      return false;
    }
    float milliseconds = 0.0f;
    if (!runtime_ok(hggcEventElapsedTime(
                        &milliseconds, events.start, events.stop),
                    "elapsed-time", &row) ||
        !(milliseconds > 0.0f) || !std::isfinite(milliseconds)) {
      std::fprintf(stderr,
                   "FQ_REDUCER_LOOKUP_ERROR case_id=%s operation=invalid-elapsed index=%d\n",
                   row.case_id, sample);
      release();
      return false;
    }
    double const microseconds = double(milliseconds) * 1000.0;
    samples.push_back(microseconds);
    std::printf(
        "FQ_REDUCER_LOOKUP_SAMPLE ordinal=%d execution_ordinal=%d "
        "case_id=%s round=%d sample=%d M=%" PRId64 " N=%" PRId64
        " S=%d implementation=%s us=%.9f\n",
        row.ordinal, execution_ordinal, row.case_id, cli.round, sample,
        row.m, row.n, row.split, implementation, microseconds);
  }

  CheckResult close;
  if (!validate_output(row, output, device_check, close) || close.bad != 0) {
    std::fprintf(stderr,
                 "FQ_REDUCER_LOOKUP_ERROR case_id=%s operation=ordered-close "
                 "raw_bad=%u first_bad=%u\n",
                 row.case_id, close.bad, close.first);
    release();
    return false;
  }
  std::vector<double> ordered = samples;
  std::sort(ordered.begin(), ordered.end());
  double const median = ordered[ordered.size() / 2];
  std::printf(
      "FQ_REDUCER_LOOKUP_CASE ordinal=%d execution_ordinal=%d case_id=%s "
      "M=%" PRId64 " N=%" PRId64 " S=%d partial_dtype=fp32 "
      "output_dtype=fp16 implementation=%s workspace_bytes=%" PRIu64 " "
      "output_bytes=%" PRIu64 " grid_ctas=%" PRIu64 " "
      "block_threads=%" PRIu64 " round=%d warmups=%d samples=%d raw_bad=0 "
      "first_bad=%u median_us=%.9f min_us=%.9f max_us=%.9f status=PASS\n",
      row.ordinal, execution_ordinal, row.case_id, row.m, row.n, row.split,
      implementation, row.workspace_bytes, row.output_bytes, grid_ctas,
      block_threads, cli.round, cli.warmups, cli.samples, kNoBadIndex, median,
      ordered.front(), ordered.back());
  release();
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  Cli cli;
  if (!parse_cli(argc, argv, cli)) {
    std::fprintf(
        stderr,
        "usage: %s --schedule-seed=UINT64 [--round=N] [--warmups=3] [--samples=11] "
        "[--case-begin=0] [--case-end=%d] [--plant-output-fault=0|1]\n",
        argv[0], FQ_SPLITK_REDUCER_CASE_COUNT);
    return 2;
  }

  std::vector<int> order(std::size_t(cli.case_end - cli.case_begin));
  std::iota(order.begin(), order.end(), cli.case_begin);
  uint64_t random = cli.schedule_seed;
  for (std::size_t remaining = order.size(); remaining > 1; --remaining) {
    std::size_t const selected = std::size_t(splitmix64(random) % remaining);
    std::swap(order[remaining - 1], order[selected]);
  }
  uint64_t const order_hash = case_order_hash(order);
  std::printf(
      "FQ_REDUCER_LOOKUP_RUN schema=%s plan_sha256=%s total_cases=%d "
      "case_begin=%d case_end=%d selected_cases=%zu round=%d warmups=%d samples=%d "
      "schedule_seed=0x%016" PRIx64 " order_hash=0x%016" PRIx64 " "
      "partial_dtype=fp32 output_dtype=fp16 reducer=M1FastReductionE2 "
      "fixture=period31-plus-rounding257-v1 plant_output_fault=%d status=BEGIN\n",
      FQ_SPLITK_REDUCER_PLAN_SCHEMA, FQ_SPLITK_REDUCER_PLAN_SHA256,
      FQ_SPLITK_REDUCER_CASE_COUNT, cli.case_begin, cli.case_end,
      order.size(), cli.round, cli.warmups, cli.samples, cli.schedule_seed, order_hash,
      cli.plant_output_fault ? 1 : 0);

  int failures = 0;
  std::size_t measured = 0;
  for (std::size_t execution = 0; execution < order.size(); ++execution) {
    Case const& row = kCases[order[execution]];
    if (!run_case(row, int(execution), cli)) {
      failures = 1;
      break;
    }
    ++measured;
  }
  std::printf(
      "FQ_REDUCER_LOOKUP_DONE plan_sha256=%s selected_cases=%zu "
      "measured=%zu failures=%d round=%d warmups=%d samples=%d "
      "schedule_seed=0x%016" PRIx64 " order_hash=0x%016" PRIx64 " "
      "status=%s\n",
      FQ_SPLITK_REDUCER_PLAN_SHA256, order.size(),
      measured, failures, cli.round,
      cli.warmups, cli.samples, cli.schedule_seed, order_hash,
      failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
