/***************************************************************************************************
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Final PPU search for dense M1 fixed Split-K parallel.
 *
 * Search domain = every committed int4/ArtifactTK64 tactic that can instantiate
 * DensePackedAKernelTypes<1> (TM=WM=8), crossed at runtime with S={1,2,4,8}.
 * No historical tactic is privileged.  The 17 us number is checked only after
 * the global S=1 winner has been selected.
 **************************************************************************************************/

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "cutlass/device_kernel.h"
#include "cutlass/util/device_memory.h"
#include "dense_splitk_parallel_bench.hpp"
#include "helper.h"
#include "unfused_weight_dequantize.hpp"
#include "xplane_offline.hpp"

#include "dense_splitk_sweep_configs.inc"

using namespace dense_splitk_sweep;

namespace dense_splitk_sweep_generated {
#define DENSE_SPLITK_DECLARE(FN,TM,TN,TK,WM,WN,ST,BC)                   \
  bool FN(DeviceInputs const&, Options const&, RowResult&);
DENSE_SPLITK_SWEEP_CONFIGS(DENSE_SPLITK_DECLARE)
#undef DENSE_SPLITK_DECLARE
}  // namespace dense_splitk_sweep_generated

namespace {

std::vector<RegistryRow> const& registry() {
  static std::vector<RegistryRow> const rows{
#define DENSE_SPLITK_REGISTRY(FN,TM,TN,TK,WM,WN,ST,BC)                 \
    {#FN, TM, TN, TK, WM, WN, ST, BC,                                 \
     &dense_splitk_sweep_generated::FN},
    DENSE_SPLITK_SWEEP_CONFIGS(DENSE_SPLITK_REGISTRY)
#undef DENSE_SPLITK_REGISTRY
  };
  return rows;
}

struct Cli {
  int iterations = 5;
  int warmup_rotations = 1;
  int correctness_repeats = 1;
  int cold_budget_mib = 512;
  int64_t l2_bytes = 0;
  int cu = 0;
  double ce_ghz = 1.70;
  double hbm_gbs = 2766.0;
  bool span_curve = false;
};

bool parse_positive_int(char const* text, int& value) {
  char* end = nullptr;
  long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || parsed <= 0 ||
      parsed > (std::numeric_limits<int>::max)()) return false;
  value = int(parsed);
  return true;
}

bool parse_cli(int argc, char** argv, Cli& cli) {
  for (int i = 1; i < argc; ++i) {
    auto value = [&](char const* prefix) -> char const* {
      std::size_t const n = std::strlen(prefix);
      return std::strncmp(argv[i], prefix, n) == 0 ? argv[i] + n : nullptr;
    };
    if (char const* v = value("--iterations=")) {
      if (!parse_positive_int(v, cli.iterations)) return false;
    } else if (char const* v = value("--warmup-rotations=")) {
      if (!parse_positive_int(v, cli.warmup_rotations)) return false;
    } else if (char const* v = value("--correctness-repeats=")) {
      if (!parse_positive_int(v, cli.correctness_repeats)) return false;
    } else if (char const* v = value("--cold-budget-mib=")) {
      if (!parse_positive_int(v, cli.cold_budget_mib)) return false;
    } else if (char const* v = value("--l2-bytes=")) {
      char* end = nullptr;
      long long parsed = std::strtoll(v, &end, 10);
      if (end == v || *end != '\0' || parsed <= 0) return false;
      cli.l2_bytes = parsed;
    } else if (char const* v = value("--cu=")) {
      if (!parse_positive_int(v, cli.cu)) return false;
    } else if (char const* v = value("--ce-ghz=")) {
      char* end = nullptr;
      cli.ce_ghz = std::strtod(v, &end);
      if (end == v || *end != '\0' || !(cli.ce_ghz > 0)) return false;
    } else if (char const* v = value("--hbm-gbs=")) {
      char* end = nullptr;
      cli.hbm_gbs = std::strtod(v, &end);
      if (end == v || *end != '\0' || !(cli.hbm_gbs > 0)) return false;
    } else if (!std::strcmp(argv[i], "--span-curve")) {
      cli.span_curve = true;
    } else {
      return false;
    }
  }
  return true;
}

template <class T, class = void>
struct HasL2 : std::false_type {};
template <class T>
struct HasL2<T, std::void_t<decltype(std::declval<T>().l2CacheSize)>>
    : std::true_type {};
template <class T, class = void>
struct HasCu : std::false_type {};
template <class T>
struct HasCu<T, std::void_t<decltype(std::declval<T>().multiProcessorCount)>>
    : std::true_type {};

template <class Prop>
int64_t measured_l2(Prop const& prop) {
  if constexpr (HasL2<Prop>::value) return int64_t(prop.l2CacheSize);
  return 0;
}
template <class Prop>
int measured_cu(Prop const& prop) {
  if constexpr (HasCu<Prop>::value) return int(prop.multiProcessorCount);
  return 0;
}

int signed_q_at(int k, int n) {
  return ((5 * k + 3 * n + 1) & 15) - 8;
}

struct Fixture {
  std::vector<half_t> a;
  std::vector<int8_t> resident_b;
  std::vector<half_t> scales;
  std::vector<half_t> golden;
  bool artifact_ok = false;
  bool exact = false;
};

Fixture make_fixture() {
  Fixture f;
  constexpr int scale_k = kK / kGroupSize;
  f.a.assign(std::size_t(kM) * kK, half_t(0.f));
  f.scales.resize(std::size_t(scale_k) * kN);
  f.golden.resize(std::size_t(kM) * kN);
  std::vector<int> active_k(scale_k);
  for (int kg = 0; kg < scale_k; ++kg) {
    active_k[kg] = kg * kGroupSize + ((29 * kg + 17) % kGroupSize);
    f.a[std::size_t(active_k[kg])] = half_t(1.f);
    for (int n = 0; n < kN; ++n) {
      f.scales[std::size_t(kg) * kN + n] =
          half_t(float(1 << ((5 * kg + 3 * n) % 3)));
    }
  }

  std::vector<uint8_t> canonical(std::size_t(kK) * kN);
  std::vector<int8_t> packed(std::size_t(kK) * kN / 2, int8_t(0));
  for (int n = 0; n < kN; ++n) {
    for (int k = 0; k < kK; ++k) {
      int const q = signed_q_at(k, n);
      canonical[std::size_t(k) * kN + n] = uint8_t((q + 8) & 15);
      std::size_t const linear = std::size_t(n) * kK + k;
      auto* byte = reinterpret_cast<uint8_t*>(packed.data()) + linear / 2;
      *byte |= uint8_t(q & 15) << (4 * (linear & 1));
    }
  }
  f.resident_b.resize(packed.size());
  preprocess_weights_for_mixed_gemm<false, 256>(
      f.resident_b.data(), packed.data(),
      {std::size_t(kK), std::size_t(kN)},
      QuantTypeClass::PACKED_INT4_WEIGHT_ONLY);

  // Independent layout anchor: one explicit TK64 xplane writer must produce
  // the exact shipping-preprocessor bytes, and its inverse must recover every
  // logical code.  The sweep changes only the reader tactic.
  std::vector<int8_t> anchored(f.resident_b.size());
  xplane::place_derived<4, 8, 128, 64, 8, 32, 1, 64>(
      anchored.data(), canonical, kN, kK);
  std::vector<uint8_t> recovered(canonical.size(), 0xff);
  xplane::recover_derived<4, 8, 128, 64, 8, 32, 1, 64>(
      anchored.data(), recovered, kN, kK);
  std::size_t byte_diff = 0, roundtrip_bad = 0;
  for (std::size_t i = 0; i < anchored.size(); ++i)
    byte_diff += anchored[i] != f.resident_b[i];
  for (std::size_t i = 0; i < recovered.size(); ++i)
    roundtrip_bad += recovered[i] != canonical[i];
  f.artifact_ok = byte_diff == 0 && roundtrip_bad == 0;

  int max_output = 0;
  for (int n = 0; n < kN; ++n) {
    int sum = 0;
    for (int kg = 0; kg < scale_k; ++kg) {
      sum += signed_q_at(active_k[kg], n) *
          int(float(f.scales[std::size_t(kg) * kN + n]));
    }
    max_output = std::max(max_output, std::abs(sum));
    f.golden[std::size_t(n)] = half_t(float(sum));
  }
  int const conservative_bound = scale_k * 8 * 4;
  f.exact = conservative_bound < (1 << 11) && max_output < (1 << 11);
  std::printf(
      "[splitk fixture] mode=ScaleOnly gs=128 artifact=TK64-xplane "
      "shipping_byte_diff=%zu/%zu roundtrip_bad=%zu/%zu "
      "integer_bound=%d max_output=%d -> %s\n",
      byte_diff, anchored.size(), roundtrip_bad, recovered.size(),
      conservative_bound, max_output,
      f.artifact_ok && f.exact ? "ORDER-INDEPENDENT+FP16-EXACT" : "FAIL");
  return f;
}

struct EmptyKernel {
  struct Params {};
  struct SharedStorage {};
  CUTLASS_DEVICE void operator()(Params const&, SharedStorage&) {}
};

double measure_empty_launch_us(int launches = 256) {
  EventPair event;
  EmptyKernel::Params params;
  CUTLASS_PPU_CHECK(hggcEventRecord(event.start, nullptr));
  for (int i = 0; i < launches; ++i) {
    cutlass::Kernel<EmptyKernel><<<dim3(1, 1, 1), dim3(1, 1, 1), 0, nullptr>>>(params);
  }
  CUTLASS_PPU_CHECK(hggcEventRecord(event.stop, nullptr));
  CUTLASS_PPU_CHECK(hggcEventSynchronize(event.stop));
  double total = 0;
  if (!elapsed_us(event.start, event.stop, total)) return -1;
  return total / launches;
}

struct Observation {
  RegistryRow const* row = nullptr;
  CellResult cell{};
};

bool better(Observation const& lhs, Observation const& rhs) {
  return lhs.cell.e2e_median_us < rhs.cell.e2e_median_us;
}

void print_cell(RegistryRow const& row, CellResult const& cell,
                int cu, double ce_ghz, double hbm_gbs,
                std::size_t representation_bytes) {
  if (cell.state != CellState::Measured) {
    std::printf(
        "CELL provider=packedA cfg=%dx%dx%d_w%dx%d_s%d_bc%d S=%d "
        "state=%s execution_ordinal=%llu output_tiles=%llu "
        "work_units=%llu k_steps=%d\n",
        row.tm, row.tn, row.tk, row.wm, row.wn, row.stages, row.b_chunk,
        cell.split, state_name(cell.state),
        static_cast<unsigned long long>(cell.execution_ordinal),
        static_cast<unsigned long long>(cell.output_tiles),
        static_cast<unsigned long long>(cell.work_units), cell.k_steps);
    return;
  }
  double const seconds = cell.e2e_median_us * 1e-6;
  double const gbs = representation_bytes / seconds / 1e9;
  double const hbm = 100.0 * gbs / hbm_gbs;
  double const weights_per_core_cycle =
      double(kN) * kK / (seconds * cu * ce_ghz * 1e9);
  std::printf(
      "CELL provider=packedA cfg=%dx%dx%d_w%dx%d_s%d_bc%d S=%d "
      "state=MEASURED e2e_median=%.6f_us mean=%.6f min=%.6f max=%.6f "
      "execution_ordinal=%llu "
      "output_tiles=%llu work_units=%llu tiles_per_cu=%.6f "
      "Kt=%d k_steps=%d partial_bytes=%zu partial_logical_rw_bytes=%zu "
      "distinct_gbs_model=%.3f hbm_nameplate_model=%.3f%% "
      "weights_per_core_cycle=%.6f raw_bad=%llu fingerprint=%016llx\n",
      row.tm, row.tn, row.tk, row.wm, row.wn, row.stages, row.b_chunk,
      cell.split, cell.e2e_median_us, cell.e2e_mean_us,
      cell.e2e_min_us, cell.e2e_max_us,
      static_cast<unsigned long long>(cell.execution_ordinal),
      static_cast<unsigned long long>(cell.output_tiles),
      static_cast<unsigned long long>(cell.work_units),
      double(cell.work_units) / cu, cell.k_tiles, cell.k_steps,
      cell.partial_bytes, cell.partial_logical_rw_bytes, gbs, hbm,
      weights_per_core_cycle,
      static_cast<unsigned long long>(cell.raw_bad),
      static_cast<unsigned long long>(cell.fingerprint));
  for (std::size_t sample = 0; sample < cell.e2e_samples_us.size(); ++sample) {
    std::printf(
        "CELL_SAMPLE cfg=%dx%dx%d_w%dx%d_s%d_bc%d S=%d sample=%zu "
        "e2e=%.9f_us\n",
        row.tm, row.tn, row.tk, row.wm, row.wn, row.stages, row.b_chunk,
        cell.split, sample, cell.e2e_samples_us[sample]);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Cli cli;
  if (!parse_cli(argc, argv, cli)) {
    std::fprintf(stderr,
        "usage: %s [--iterations=N] [--warmup-rotations=N] "
        "[--correctness-repeats=N] [--cold-budget-mib=N] "
        "[--l2-bytes=N] [--cu=N] [--ce-ghz=F] [--hbm-gbs=F] "
        "[--span-curve]\n",
        argv[0]);
    return 2;
  }
  if (registry().size() != DENSE_SPLITK_SWEEP_ROWS ||
      DENSE_SPLITK_SWEEP_ROWS != 201 ||
      DENSE_SPLITK_SWEEP_SOURCE_ROWS != 1772) {
    std::fprintf(stderr,
        "[splitk sweep] registry authority mismatch: registry=%zu filtered=%d source=%d\n",
        registry().size(), DENSE_SPLITK_SWEEP_ROWS,
        DENSE_SPLITK_SWEEP_SOURCE_ROWS);
    return 1;
  }

  int device = 0;
  hggcDeviceProp prop{};
  CUTLASS_PPU_CHECK(hggcGetDevice(&device));
  CUTLASS_PPU_CHECK(hggcGetDeviceProperties(&prop, device));
  int64_t const detected_l2 = measured_l2(prop);
  int const detected_cu = measured_cu(prop);
  if (cli.l2_bytes == 0) cli.l2_bytes = detected_l2;
  if (cli.cu == 0) cli.cu = detected_cu;
  if (cli.l2_bytes <= 0 || cli.cu <= 0) {
    std::fprintf(stderr,
        "[splitk sweep] device API did not expose L2/CU; pass --l2-bytes and --cu explicitly\n");
    return 1;
  }
  std::printf(
      "[splitk device] ordinal=%d measured_cu=%d measured_l2=%lld "
      "effective_cu=%d effective_l2=%lld\n",
      device, detected_cu, static_cast<long long>(detected_l2),
      cli.cu, static_cast<long long>(cli.l2_bytes));

  Fixture const fixture = make_fixture();
  if (!fixture.artifact_ok || !fixture.exact) return 1;
  std::size_t const representation_bytes =
      fixture.resident_b.size() + fixture.scales.size() * sizeof(half_t);
  std::size_t const cold_need = std::max<std::size_t>(
      std::size_t(std::ceil(2.16 * double(cli.l2_bytes))),
      std::size_t(128) << 20);
  int cold_copies = int((cold_need + representation_bytes - 1) /
                        representation_bytes);
  std::size_t const cold_budget = std::size_t(cli.cold_budget_mib) << 20;
  if (cold_copies < 3 ||
      std::size_t(cold_copies) * representation_bytes > cold_budget ||
      std::size_t(cold_copies) * representation_bytes <=
          2 * std::size_t(cli.l2_bytes)) {
    std::fprintf(stderr,
        "[splitk sweep] cold rotation not established: copies=%d rep=%zu "
        "rotation=%zu l2=%lld budget=%zu\n",
        cold_copies, representation_bytes,
        std::size_t(cold_copies) * representation_bytes,
        static_cast<long long>(cli.l2_bytes), cold_budget);
    return 1;
  }

  cutlass::DeviceAllocation<half_t> d_a(fixture.a.size());
  cutlass::DeviceAllocation<uint8_t> d_b(
      std::size_t(cold_copies) * fixture.resident_b.size());
  cutlass::DeviceAllocation<half_t> d_scales(
      std::size_t(cold_copies) * fixture.scales.size());
  cutlass::DeviceAllocation<half_t> d_output(
      2 * kOutputGuardElements + std::size_t(kM) * kN);
  dense_splitk_parallel_ppu::WorkspacePlan max_plan;
  if (!dense_splitk_parallel_ppu::query_workspace_plan(
          kM, kN, 8, max_plan)) return 1;
  cutlass::DeviceAllocation<char> d_workspace(
      2 * kWorkspaceGuardBytes + max_plan.partial_bytes);
  d_a.copy_from_host(fixture.a.data());
  for (int copy = 0; copy < cold_copies; ++copy) {
    CUTLASS_PPU_CHECK(hggcMemcpy(
        d_b.get() + std::size_t(copy) * fixture.resident_b.size(),
        fixture.resident_b.data(), fixture.resident_b.size(),
        hggcMemcpyHostToDevice));
    CUTLASS_PPU_CHECK(hggcMemcpy(
        d_scales.get() + std::size_t(copy) * fixture.scales.size(),
        fixture.scales.data(), fixture.scales.size() * sizeof(half_t),
        hggcMemcpyHostToDevice));
  }

  DeviceInputs inputs{
      d_a.get(), d_b.get(), d_scales.get(), nullptr,
      fixture.resident_b.size(), fixture.scales.size(), cold_copies,
      d_output.get(), d_output.size(), d_workspace.get(), d_workspace.size(),
      fixture.golden.data()};

  if (cli.span_curve) {
    struct RequestedSpan {
      int tm, tn, tk, wm, wn, stages, b_chunk, split;
      char const* role;
    };
    constexpr RequestedSpan requested[] = {
        {8, 16, 256, 8, 16, 4, 0, 1, "best-e2e-S1"},
        {8, 32, 256, 8, 16, 3, 0, 2, "best-e2e-S2"},
        {8, 128, 256, 8, 16, 3, 0, 4, "best-e2e-S4"},
        {8, 128, 256, 8, 32, 2, 0, 8, "best-e2e-S8"},
        {8, 128, 128, 8, 32, 3, 0, 8, "reshape-matched-S8"},
    };
    std::printf(
        "[splitk span curve] mode=producer-reducer-median samples=7 "
        "correctness_repeats=8 selection=bound-to-72cu-full-sweep\n");
    bool all_ok = true;
    for (RequestedSpan const& request : requested) {
      auto const found = std::find_if(
          registry().begin(), registry().end(), [&](RegistryRow const& row) {
            return row.tm == request.tm && row.tn == request.tn &&
                row.tk == request.tk && row.wm == request.wm &&
                row.wn == request.wn && row.stages == request.stages &&
                row.b_chunk == request.b_chunk;
          });
      if (found == registry().end()) {
        std::fprintf(stderr,
            "[splitk span curve] missing requested row role=%s\n",
            request.role);
        all_ok = false;
        continue;
      }
      Options diagnostic;
      diagnostic.only_split = request.split;
      diagnostic.measure = false;
      diagnostic.correctness_repeats = 8;
      diagnostic.diagnose_spans = true;
      diagnostic.span_repeats = 7;
      RowResult result;
      bool const call_ok = found->run(inputs, diagnostic, result);
      std::size_t const split_index = std::size_t(
          std::find(kSplits.begin(), kSplits.end(), request.split) -
          kSplits.begin());
      CellResult const& cell = result.cells[split_index];
      bool const ok = call_ok && cell.state == CellState::Measured &&
          cell.raw_bad == 0 && cell.span_recorded && cell.span_samples == 7;
      std::printf(
          "SPAN_CURVE role=%s cfg=%dx%dx%d_w%dx%d_s%d_bc%d S=%d "
          "producer_median=%.6f_us producer_range=[%.6f,%.6f]_us "
          "reducer=%s%.6f_us reducer_range=%s[%.6f,%.6f]_us "
          "span_samples=%d correctness=%s\n",
          request.role, found->tm, found->tn, found->tk, found->wm,
          found->wn, found->stages, found->b_chunk, request.split,
          cell.producer_us, cell.producer_min_us, cell.producer_max_us,
          request.split == 1 ? "NA/" : "", cell.reducer_us,
          request.split == 1 ? "NA/" : "", cell.reducer_min_us,
          cell.reducer_max_us, cell.span_samples, ok ? "RAW-BIT/PASS" : "FAIL");
      all_ok = all_ok && ok;
    }
    return all_ok ? 0 : 1;
  }

  Options options;
  options.iterations = cli.iterations;
  options.warmup_rotations = cli.warmup_rotations;
  options.correctness_repeats = cli.correctness_repeats;

  std::printf(
      "[splitk sweep] source=lowbit_dense_configs.inc source_rows=%d "
      "packedA_rows=%d cells=%d bits=4 mode=ScaleOnly gs=128 "
      "artifact_tk=64 space_fnv=%s emitter_fnv=%s shape=1x4096x4096 "
      "S=1,2,4,8 cold_copies=%d representation_bytes=%zu rotation_bytes=%zu "
      "l2_bytes=%lld cold_multiple=%.6f iterations=%d launches_per_sample=%d "
      "cu=%d ce_ghz=%.6f hbm_gbs=%.3f execution_order=shuffled "
      "schedule_seed=0x6a09e667f3bcc909\n",
      DENSE_SPLITK_SWEEP_SOURCE_ROWS, DENSE_SPLITK_SWEEP_ROWS,
      DENSE_SPLITK_SWEEP_ROWS * int(kSplits.size()),
      DENSE_SPLITK_SWEEP_SPACE_FNV1A64,
      DENSE_SPLITK_SWEEP_EMITTER_FNV1A64, cold_copies,
      representation_bytes, std::size_t(cold_copies) * representation_bytes,
      static_cast<long long>(cli.l2_bytes),
      double(std::size_t(cold_copies) * representation_bytes) / cli.l2_bytes,
      cli.iterations, cold_copies, cli.cu, cli.ce_ghz, cli.hbm_gbs);

  struct CellTask { std::size_t row; std::size_t split; };
  constexpr uint64_t kScheduleSeed = 0x6a09e667f3bcc909ull;
  std::vector<CellTask> tasks;
  tasks.reserve(registry().size() * kSplits.size());
  for (std::size_t row = 0; row < registry().size(); ++row) {
    for (std::size_t split = 0; split < kSplits.size(); ++split) {
      tasks.push_back(CellTask{row, split});
    }
  }
  std::mt19937_64 rng(kScheduleSeed);
  std::shuffle(tasks.begin(), tasks.end(), rng);
  std::vector<RowResult> sweep_results(registry().size());
  for (std::size_t ordinal = 0; ordinal < tasks.size(); ++ordinal) {
    CellTask const task = tasks[ordinal];
    Options cell_options = options;
    cell_options.only_split = kSplits[task.split];
    RowResult isolated;
    bool const call_ok = registry()[task.row].run(
        inputs, cell_options, isolated);
    CellResult& result = isolated.cells[task.split];
    result.execution_ordinal = ordinal + 1;
    if (!call_ok && result.state == CellState::Measured) {
      result.state = CellState::LaunchFailed;
    }
    sweep_results[task.row].cells[task.split] = std::move(result);
  }

  std::vector<Observation> measured;
  int inadmissible = 0;
  int failed = 0;
  for (std::size_t row_index = 0; row_index < registry().size(); ++row_index) {
    RegistryRow const& row = registry()[row_index];
    for (CellResult const& cell : sweep_results[row_index].cells) {
      print_cell(row, cell, cli.cu, cli.ce_ghz, cli.hbm_gbs,
                 representation_bytes);
      if (cell.state == CellState::Measured) {
        measured.push_back(Observation{&row, cell});
      } else if (cell.state == CellState::InadmissiblePipelineDepth) {
        ++inadmissible;
      } else {
        ++failed;
      }
    }
  }

  int expected_measured = 0, expected_inadmissible = 0;
  for (RegistryRow const& row : registry()) {
    int const kt = kK / row.tk;
    for (int splits : kSplits) {
      if (kK % row.tk == 0 && kt % splits == 0 &&
          (splits == 1 || kt / splits >= row.stages - 1))
        ++expected_measured;
      else
        ++expected_inadmissible;
    }
  }
  if (int(measured.size()) != expected_measured ||
      inadmissible != expected_inadmissible || failed != 0 ||
      expected_measured != 684 || expected_inadmissible != 120 ||
      expected_measured + expected_inadmissible !=
          DENSE_SPLITK_SWEEP_ROWS * int(kSplits.size())) {
    std::fprintf(stderr,
        "[splitk sweep] denominator FAIL: measured=%zu/%d inadmissible=%d/%d failures=%d\n",
        measured.size(), expected_measured, inadmissible,
        expected_inadmissible, failed);
    return 1;
  }

  constexpr std::size_t kConfirmationCandidates = 8;
  auto top_for = [&](bool split) {
    std::vector<Observation> candidates;
    for (Observation const& observation : measured) {
      if ((observation.cell.split > 1) == split) {
        candidates.push_back(observation);
      }
    }
    std::sort(candidates.begin(), candidates.end(), better);
    if (candidates.size() > kConfirmationCandidates) {
      candidates.resize(kConfirmationCandidates);
    }
    return candidates;
  };
  struct ConfirmationCandidate {
    Observation observation;
    char const* cohort = nullptr;
    std::size_t initial_rank = 0;
  };
  auto confirm = [&](std::vector<ConfirmationCandidate> const& candidates,
                     uint64_t seed) {
    std::vector<std::size_t> order(candidates.size());
    std::iota(order.begin(), order.end(), std::size_t(0));
    std::mt19937_64 confirm_rng(seed);
    std::shuffle(order.begin(), order.end(), confirm_rng);
    std::vector<Observation> confirmed;
    confirmed.reserve(candidates.size());
    for (std::size_t ordinal = 0; ordinal < order.size(); ++ordinal) {
      std::size_t const initial_rank = order[ordinal];
      ConfirmationCandidate const& ranked = candidates[initial_rank];
      Observation const& candidate = ranked.observation;
      Options confirmation = options;
      confirmation.only_split = candidate.cell.split;
      confirmation.iterations = std::max(options.iterations, 7);
      confirmation.correctness_repeats = std::max(
          options.correctness_repeats, 2);
      RowResult result;
      bool const call_ok = candidate.row->run(inputs, confirmation, result);
      std::size_t const split_index = std::size_t(
          std::find(kSplits.begin(), kSplits.end(), candidate.cell.split) -
          kSplits.begin());
      CellResult cell = std::move(result.cells[split_index]);
      cell.execution_ordinal = 100000 + ordinal + 1;
      std::printf(
          "CONFIRM_CANDIDATE cohort=%s initial_rank=%zu confirmation_order=%zu "
          "cfg=%dx%dx%d_w%dx%d_s%d_bc%d S=%d\n",
          ranked.cohort, ranked.initial_rank, ordinal + 1,
          candidate.row->tm, candidate.row->tn, candidate.row->tk,
          candidate.row->wm, candidate.row->wn, candidate.row->stages,
          candidate.row->b_chunk, candidate.cell.split);
      print_cell(*candidate.row, cell, cli.cu, cli.ce_ghz, cli.hbm_gbs,
                 representation_bytes);
      if (!call_ok || cell.state != CellState::Measured || cell.raw_bad != 0) {
        confirmed.clear();
        return confirmed;
      }
      confirmed.push_back(Observation{candidate.row, std::move(cell)});
    }
    return confirmed;
  };
  std::vector<Observation> initial_s1 = top_for(false);
  std::vector<Observation> initial_split = top_for(true);
  std::vector<ConfirmationCandidate> confirmation_candidates;
  confirmation_candidates.reserve(initial_s1.size() + initial_split.size());
  for (std::size_t rank = 0; rank < initial_s1.size(); ++rank) {
    confirmation_candidates.push_back(
        ConfirmationCandidate{initial_s1[rank], "S1", rank + 1});
  }
  for (std::size_t rank = 0; rank < initial_split.size(); ++rank) {
    confirmation_candidates.push_back(
        ConfirmationCandidate{initial_split[rank], "SPLIT", rank + 1});
  }
  // One independently shuffled confirmation schedule interleaves both arms;
  // two sequential cohort runs would let clock drift masquerade as Split-K.
  std::vector<Observation> confirmed = confirm(
      confirmation_candidates, kScheduleSeed ^ 0xc09f1a4d7b2e6835ull);
  std::vector<Observation> confirmed_s1, confirmed_split;
  for (Observation& observation : confirmed) {
    (observation.cell.split == 1 ? confirmed_s1 : confirmed_split)
        .push_back(std::move(observation));
  }
  if (confirmed_s1.size() != initial_s1.size() ||
      confirmed_split.size() != initial_split.size() ||
      confirmed_s1.empty() || confirmed_split.empty()) {
    std::fprintf(stderr,
        "[splitk sweep] independent top-%zu confirmation failed\n",
        kConfirmationCandidates);
    return 1;
  }
  auto confirmed_best = [](std::vector<Observation> const& values) {
    return &*std::min_element(values.begin(), values.end(), better);
  };
  Observation const* best_s1 = confirmed_best(confirmed_s1);
  Observation const* best_split = confirmed_best(confirmed_split);

  auto diagnose = [&](Observation const& winner, char const* label) -> bool {
    Options diagnostic;
    diagnostic.only_split = winner.cell.split;
    diagnostic.measure = false;
    diagnostic.correctness_repeats = 8;
    diagnostic.diagnose_spans = true;
    RowResult result;
    if (!winner.row->run(inputs, diagnostic, result)) return false;
    CellResult const& cell = result.cells[
        std::size_t(std::find(kSplits.begin(), kSplits.end(), winner.cell.split) -
                    kSplits.begin())];
    std::printf(
        "WINNER_SPAN kind=%s cfg=%dx%dx%d_w%dx%d_s%d_bc%d S=%d "
        "producer=%.6f_us reducer=%s%.6f_us repeated_correctness=8 "
        "midpoint_events=DIAGNOSTIC_NOT_RANKING\n",
        label, winner.row->tm, winner.row->tn, winner.row->tk,
        winner.row->wm, winner.row->wn, winner.row->stages,
        winner.row->b_chunk, winner.cell.split, cell.producer_us,
        winner.cell.split == 1 ? "NA/" : "", cell.reducer_us);
    return cell.state == CellState::Measured && cell.span_recorded &&
        cell.raw_bad == 0;
  };
  if (!diagnose(*best_s1, "S1") || !diagnose(*best_split, "SPLIT")) {
    std::fprintf(stderr, "[splitk sweep] winner attribution/repeat gate failed\n");
    return 1;
  }

  double const empty_launch_us = measure_empty_launch_us();
  if (!(empty_launch_us > 0)) {
    std::fprintf(stderr,
        "[splitk sweep] empty-launch attribution failed: %.9f us\n",
        empty_launch_us);
    return 1;
  }
  double const speedup = best_s1->cell.e2e_median_us /
                         best_split->cell.e2e_median_us;
  // A median ordering is not a performance result when the independently
  // repeated sample envelopes overlap.  Require the slowest observed split
  // launch to beat the fastest observed S=1 launch.  This deliberately
  // conservative rule is fixed before the PPU run and makes sub-resolution
  // differences UNRESOLVED rather than manufacturing a winner.
  double const confirmation_gap_us = best_s1->cell.e2e_min_us -
                                     best_split->cell.e2e_max_us;
  bool const performance_win = confirmation_gap_us > 0;
  bool const historical_admitted =
      best_s1->cell.e2e_median_us >= 16.49 &&
      best_s1->cell.e2e_median_us <= 17.51;
  std::printf(
      "GLOBAL_S1 cfg=%dx%dx%d_w%dx%d_s%d_bc%d S=1 e2e=%.6f_us "
      "historical_17us_admission=%s range=[16.49,17.51]\n",
      best_s1->row->tm, best_s1->row->tn, best_s1->row->tk,
      best_s1->row->wm, best_s1->row->wn, best_s1->row->stages,
      best_s1->row->b_chunk, best_s1->cell.e2e_median_us,
      historical_admitted ? "PASS" : "FAIL");
  std::printf(
      "GLOBAL_SPLIT cfg=%dx%dx%d_w%dx%d_s%d_bc%d S=%d e2e=%.6f_us "
      "speedup_vs_global_S1=%.6fx empty_launch=%.6f_us "
      "confirmation_envelopes=%s gap=%.6f_us "
      "S1=[%.6f,%.6f]_us SPLIT=[%.6f,%.6f]_us\n",
      best_split->row->tm, best_split->row->tn, best_split->row->tk,
      best_split->row->wm, best_split->row->wn, best_split->row->stages,
      best_split->row->b_chunk, best_split->cell.split,
      best_split->cell.e2e_median_us, speedup, empty_launch_us,
      performance_win ? "DISJOINT/SPLIT-WINS" : "OVERLAP/UNRESOLVED",
      confirmation_gap_us, best_s1->cell.e2e_min_us,
      best_s1->cell.e2e_max_us, best_split->cell.e2e_min_us,
      best_split->cell.e2e_max_us);
  std::printf(
      "[splitk sweep] %s: denominator=%d measured=%d inadmissible=%d "
      "correctness=RAW-BIT winner_binding=SWEEPED+TOP8-CONFIRMED performance=%s "
      "historical_environment=%s\n",
      performance_win && historical_admitted ? "PASS" : "UNADJUDICATED",
      DENSE_SPLITK_SWEEP_ROWS * int(kSplits.size()), expected_measured,
      expected_inadmissible, performance_win ? "WIN" : "UNRESOLVED",
      historical_admitted ? "ADMITTED" : "DRIFTED");
  return performance_win && historical_admitted ? 0 : 3;
}
