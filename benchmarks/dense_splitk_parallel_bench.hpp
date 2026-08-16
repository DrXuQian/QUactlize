/***************************************************************************************************
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Shared ABI and per-row implementation for the generated dense fixed Split-K sweep.
 *
 * One generated wrapper owns one committed dense tactic.  S is deliberately runtime data, so the
 * 201 legal M1 packed-A tactics produce 201 compiled types rather than 804 look-alike types.  All
 * ranking spans call PreparedOnePlaneLauncher::run(), which contains no midpoint event; the
 * attribution seam is invoked only after a winner has already been selected.
 **************************************************************************************************/
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

#include "cutlass/cutlass.h"
#include "dense_splitk_parallel_ppu.cuh"
#include "helper.h"
#include "ppu_group_schedule.hpp"

namespace dense_splitk_sweep {

using half_t = cutlass::half_t;
using int4_t = cutlass::int4b_t;
using QuantMode = fpa_intb_ppu::QuantMode;

inline constexpr int kM = 1;
inline constexpr int kN = 4096;
inline constexpr int kK = 4096;
inline constexpr int kGroupSize = 128;
inline constexpr int kArtifactTileK = 64;
inline constexpr std::array<int, 4> kSplits{{1, 2, 4, 8}};
inline constexpr std::size_t kOutputGuardElements = 64;
inline constexpr std::size_t kWorkspaceGuardBytes = 256;
inline constexpr uint16_t kOutputLeftCanary = 0x3555;
inline constexpr uint16_t kOutputRightCanary = 0x3aaa;
inline constexpr unsigned char kWorkspaceCanary = 0xa5;

enum class CellState : int {
  Measured = 0,
  InadmissiblePipelineDepth = 1,
  InitializeFailed = 2,
  LaunchFailed = 3,
  CorrectnessFailed = 4,
  TimingFailed = 5,
};

inline char const* state_name(CellState state) {
  switch (state) {
    case CellState::Measured: return "MEASURED";
    case CellState::InadmissiblePipelineDepth:
      return "INADMISSIBLE_PIPELINE_DEPTH";
    case CellState::InitializeFailed: return "INITIALIZE_FAILED";
    case CellState::LaunchFailed: return "LAUNCH_FAILED";
    case CellState::CorrectnessFailed: return "CORRECTNESS_FAILED";
    case CellState::TimingFailed: return "TIMING_FAILED";
  }
  return "UNKNOWN";
}

struct Options {
  int iterations = 5;              // each sample is one full cold rotation
  int warmup_rotations = 1;
  int correctness_repeats = 1;
  int only_split = 0;              // zero means all four
  bool measure = true;
  bool diagnose_spans = false;
  int span_repeats = 1;
};

struct DeviceInputs {
  half_t const* a = nullptr;
  uint8_t const* resident_b = nullptr;
  half_t const* scales = nullptr;
  half_t const* zeros = nullptr;
  std::size_t resident_b_bytes = 0;
  std::size_t scale_elements = 0;
  int cold_copies = 0;
  half_t* output_storage = nullptr;
  std::size_t output_storage_elements = 0;
  char* workspace_storage = nullptr;
  std::size_t workspace_storage_bytes = 0;
  half_t const* golden = nullptr;   // host pointer, kM*kN values
  int rows = kM;
  int columns = kN;
  int inner = kK;
  int group_size = kGroupSize;
};

struct ExactWarmAbResult {
  double historical_reshape_us = 0;
  double shipping_ordinary_reshape_us = 0;
  double packed_reshape_us = 0;
  double packed_internal_producer_us = 0;
  double packed_internal_reducer_us = 0;
  double packed_internal_full_us = 0;
  double packed_internal_fused_us = 0;
  double packed_internal_publish_only_us = 0;
  uint64_t historical_reshape_bad = 0;
  uint64_t shipping_ordinary_reshape_bad = 0;
  uint64_t packed_reshape_bad = 0;
  uint64_t packed_internal_bad = 0;
  uint64_t packed_internal_fused_bad = 0;
  uint64_t packed_internal_publish_only_partial_byte_diff = 0;
  bool packed_internal_fast_reducer = false;
  bool packed_internal_fused_selected = false;
  bool packed_internal_publish_only_selected = false;
  bool packed_internal_fused_counters_zero = false;
  bool packed_internal_publish_only_counters_zero = false;
  bool packed_internal_publish_only_d_untouched = false;
  int packed_internal_fused_slices_passes = 0;
  int packed_internal_fused_reuse_passes = 0;
  bool post_timing_correct = false;
};

struct CellResult {
  int split = 0;
  CellState state = CellState::InitializeFailed;
  uint64_t raw_bad = 0;
  uint64_t fingerprint = 0;
  uint64_t launches = 0;
  uint64_t execution_ordinal = 0;
  uint64_t output_tiles = 0;
  uint64_t work_units = 0;
  int k_tiles = 0;
  int k_steps = 0;
  std::size_t partial_bytes = 0;
  std::size_t partial_logical_rw_bytes = 0;
  double e2e_median_us = 0;
  double e2e_mean_us = 0;
  double e2e_min_us = 0;
  double e2e_max_us = 0;
  double producer_us = 0;
  double reducer_us = 0;
  double producer_min_us = 0;
  double producer_max_us = 0;
  double reducer_min_us = 0;
  double reducer_max_us = 0;
  int span_samples = 0;
  bool span_recorded = false;
  bool output_redzone = false;
  bool workspace_redzone = false;
  std::vector<double> e2e_samples_us{};
};

struct RowResult {
  std::array<CellResult, 4> cells{};
};

using RunRow = bool (*)(DeviceInputs const&, Options const&, RowResult&);

struct RegistryRow {
  char const* symbol;
  int tm, tn, tk, wm, wn, stages, b_chunk;
  RunRow run;
};

class EventPair {
 public:
  hggcEvent_t start{};
  hggcEvent_t stop{};
  EventPair() {
    CUTLASS_PPU_CHECK(hggcEventCreate(&start));
    CUTLASS_PPU_CHECK(hggcEventCreate(&stop));
  }
  ~EventPair() {
    if (start) hggcEventDestroy(start);
    if (stop) hggcEventDestroy(stop);
  }
  EventPair(EventPair const&) = delete;
  EventPair& operator=(EventPair const&) = delete;
};

class EventTriple {
 public:
  dense_splitk_parallel_ppu::SplitKParallelSpanEvents events{};
  EventTriple() {
    CUTLASS_PPU_CHECK(hggcEventCreate(&events.producer_start));
    CUTLASS_PPU_CHECK(hggcEventCreate(&events.producer_stop));
    CUTLASS_PPU_CHECK(hggcEventCreate(&events.reducer_stop));
  }
  ~EventTriple() {
    if (events.producer_start) hggcEventDestroy(events.producer_start);
    if (events.producer_stop) hggcEventDestroy(events.producer_stop);
    if (events.reducer_stop) hggcEventDestroy(events.reducer_stop);
  }
  EventTriple(EventTriple const&) = delete;
  EventTriple& operator=(EventTriple const&) = delete;
};

inline bool elapsed_us(
    hggcEvent_t start, hggcEvent_t stop, double& value) {
  float milliseconds = 0;
  if (hggcEventElapsedTime(&milliseconds, start, stop) != hggcSuccess ||
      !std::isfinite(milliseconds) || milliseconds <= 0) {
    return false;
  }
  value = double(milliseconds) * 1000.0;
  return true;
}

inline uint64_t fnv1a_half_bits(half_t const* values, std::size_t count) {
  uint64_t hash = 1469598103934665603ull;
  for (std::size_t i = 0; i < count; ++i) {
    uint16_t const bits = values[i].raw();
    hash ^= uint8_t(bits);
    hash *= 1099511628211ull;
    hash ^= uint8_t(bits >> 8);
    hash *= 1099511628211ull;
  }
  return hash;
}

inline bool inspect_output(
    DeviceInputs const& inputs, CellResult& result,
    std::vector<half_t>& host_storage) {
  host_storage.resize(inputs.output_storage_elements);
  if (hggcMemcpy(host_storage.data(), inputs.output_storage,
                 host_storage.size() * sizeof(half_t),
                 hggcMemcpyDeviceToHost) != hggcSuccess) {
    return false;
  }
  std::size_t const output_elements =
      std::size_t(inputs.rows) * std::size_t(inputs.columns);
  if (inputs.rows <= 0 || inputs.columns <= 0 ||
      host_storage.size() != 2 * kOutputGuardElements + output_elements) {
    return false;
  }
  result.output_redzone = true;
  for (std::size_t i = 0; i < kOutputGuardElements; ++i) {
    result.output_redzone = result.output_redzone &&
        host_storage[i].raw() == kOutputLeftCanary &&
        host_storage[host_storage.size() - kOutputGuardElements + i].raw() ==
            kOutputRightCanary;
  }
  half_t const* output = host_storage.data() + kOutputGuardElements;
  result.raw_bad = 0;
  for (std::size_t i = 0; i < output_elements; ++i) {
    result.raw_bad += output[i].raw() != inputs.golden[i].raw();
  }
  result.fingerprint = fnv1a_half_bits(output, output_elements);
  return true;
}

inline bool inspect_workspace_redzone(
    DeviceInputs const& inputs, std::size_t plan_bytes, CellResult& result,
    std::vector<char>& host_workspace) {
  host_workspace.resize(inputs.workspace_storage_bytes);
  if (hggcMemcpy(host_workspace.data(), inputs.workspace_storage,
                 host_workspace.size(), hggcMemcpyDeviceToHost) != hggcSuccess ||
      host_workspace.size() < 2 * kWorkspaceGuardBytes + plan_bytes) {
    return false;
  }
  auto canary = [](char value) {
    return static_cast<unsigned char>(value) == kWorkspaceCanary;
  };
  result.workspace_redzone =
      std::all_of(host_workspace.begin(),
                  host_workspace.begin() + std::ptrdiff_t(kWorkspaceGuardBytes),
                  canary) &&
      std::all_of(host_workspace.begin() +
                      std::ptrdiff_t(kWorkspaceGuardBytes + plan_bytes),
                  host_workspace.end(), canary);
  return true;
}

inline bool reset_output_canaries(DeviceInputs const& inputs) {
  std::vector<half_t> guards(inputs.output_storage_elements,
                             half_t::bitcast(uint16_t(0x7e00)));
  std::fill(guards.begin(),
            guards.begin() + std::ptrdiff_t(kOutputGuardElements),
            half_t::bitcast(kOutputLeftCanary));
  std::fill(guards.end() - std::ptrdiff_t(kOutputGuardElements),
            guards.end(), half_t::bitcast(kOutputRightCanary));
  return hggcMemcpy(inputs.output_storage, guards.data(),
                    guards.size() * sizeof(half_t),
                    hggcMemcpyHostToDevice) == hggcSuccess;
}

inline bool reset_canaries(DeviceInputs const& inputs) {
  return reset_output_canaries(inputs) &&
      hggcMemset(inputs.workspace_storage, kWorkspaceCanary,
                 inputs.workspace_storage_bytes) == hggcSuccess;
}

template <int TM, int TN, int TK, int WM, int WN, int Stages, int BChunk>
bool run_row(DeviceInputs const& inputs, Options const& options,
             RowResult& row_result) {
  using namespace cute;
  using Schedule = ppu_group_schedule::FinegrainedSchedule<kGroupSize>;
  using TileShape = Shape<Int<TM>, Int<TN>, Int<TK>>;
  using ScaleTile = Shape<Int<TN>,
      Int<ppu_group_schedule::scale_groups_v<TK, kGroupSize>>>;
  using WarpShape = Shape<Int<WM>, Int<WN>, Int<TK>>;
  using Shipping = fpa_intb_ppu::DensePackedAKernelTypes<
      1, QuantMode::FinegrainedScaleOnly, Schedule, TileShape, ScaleTile,
      WarpShape,
      Stages, true, int4_t, kArtifactTileK>;
  using Split = dense_splitk_parallel_ppu::KernelTypes<
      Shipping, TileShape, WarpShape>;
  using Prepared = dense_splitk_parallel_ppu::PreparedOnePlaneLauncher<
      Shipping, TileShape, WarpShape>;

  static_assert(TM == 8 && WM == 8,
                "generated M1 packed-A rows must use the m8 instruction family");
  static_assert(BChunk == 0,
                "the committed packed-A legal domain currently contains only bc0");
#if defined(PPU_B_CHUNK)
  static_assert(PPU_B_CHUNK == BChunk,
                "generated unit and committed row disagree on B-chunk mode");
#endif
  static_assert(Shipping::MainloopPolicy::ArtifactTileK == kArtifactTileK &&
                    Shipping::MainloopPolicy::ArtifactLowFold == 1,
                "the sweep must consume the committed unfolded TK64 artifact");
  static_assert(Shipping::MainloopPolicy::PackedARows == 1,
                "the sweep must measure the production M1 packed-A provider");
  static_assert(std::is_same_v<typename Split::CollectiveMainloop,
                               typename Shipping::CollectiveMainloop>,
                "S>1 must reuse the exact generated shipping mainloop");
  static_assert(size<0>(typename Shipping::CollectiveMainloop::TiledMma::AtomShape_MNK{}) == 8,
                "every sweep row must compile the m8n16k16 PPU atom");

  if (inputs.rows != kM || inputs.columns != kN || inputs.inner != kK ||
      inputs.group_size != kGroupSize || inputs.cold_copies <= 0 ||
      inputs.resident_b == nullptr ||
      inputs.scales == nullptr || inputs.output_storage == nullptr ||
      inputs.workspace_storage == nullptr || inputs.golden == nullptr) {
    return false;
  }
  half_t* output = inputs.output_storage + kOutputGuardElements;
  char* workspace = inputs.workspace_storage + kWorkspaceGuardBytes;
  std::size_t const workspace_bytes =
      inputs.workspace_storage_bytes - 2 * kWorkspaceGuardBytes;
  uint64_t const output_tiles =
      uint64_t((kM + TM - 1) / TM) * uint64_t((kN + TN - 1) / TN);
  int const k_tiles = (kK + TK - 1) / TK;
  bool row_ok = true;

  for (std::size_t split_index = 0; split_index < kSplits.size(); ++split_index) {
    int const splits = kSplits[split_index];
    CellResult& result = row_result.cells[split_index];
    result = CellResult{};
    result.split = splits;
    result.output_tiles = output_tiles;
    result.work_units = output_tiles * uint64_t(splits);
    result.k_tiles = k_tiles;
    result.k_steps = k_tiles / splits;
    dense_splitk_parallel_ppu::WorkspacePlan plan;
    if (!dense_splitk_parallel_ppu::query_workspace_plan(kM, kN, splits, plan)) {
      result.state = CellState::InitializeFailed;
      row_ok = false;
      continue;
    }
    result.partial_bytes = plan.partial_bytes;
    result.partial_logical_rw_bytes = splits > 1
        ? 2 * plan.partial_bytes + std::size_t(kM) * kN * sizeof(half_t)
        : 0;

    if (options.only_split != 0 && options.only_split != splits) {
      result.state = CellState::InadmissiblePipelineDepth;
      continue;
    }
    if (kK % TK != 0 || k_tiles % splits != 0 ||
        (splits > 1 && k_tiles / splits < Stages - 1)) {
      result.state = CellState::InadmissiblePipelineDepth;
      continue;
    }

    std::vector<std::unique_ptr<Prepared>> prepared;
    prepared.reserve(std::size_t(inputs.cold_copies));
    bool initialized = true;
    for (int copy = 0; copy < inputs.cold_copies; ++copy) {
      auto handle = std::make_unique<Prepared>();
      uint8_t const* b = inputs.resident_b +
          std::size_t(copy) * inputs.resident_b_bytes;
      half_t const* scales = inputs.scales +
          std::size_t(copy) * inputs.scale_elements;
      initialized = initialized && handle->initialize(
          inputs.a, reinterpret_cast<int4_t const*>(b), scales, inputs.zeros,
          output, kM, kN, kK, kGroupSize, splits,
          workspace, workspace_bytes, nullptr);
      prepared.push_back(std::move(handle));
    }
    if (!initialized) {
      result.state = CellState::InitializeFailed;
      row_ok = false;
      continue;
    }

    std::vector<half_t> host_output;
    std::vector<char> host_workspace;
    uint64_t stable_fingerprint = 0;
    bool correct = true;
    for (int repeat = 0; repeat < options.correctness_repeats; ++repeat) {
      if (!reset_canaries(inputs) ||
          prepared[std::size_t(repeat % inputs.cold_copies)]->run(nullptr) !=
              cutlass::Status::kSuccess ||
          hggcDeviceSynchronize() != hggcSuccess ||
          !inspect_output(inputs, result, host_output) ||
          !inspect_workspace_redzone(inputs, plan.partial_bytes, result,
                                     host_workspace)) {
        result.state = CellState::LaunchFailed;
        correct = false;
        break;
      }
      if (repeat == 0) stable_fingerprint = result.fingerprint;
      correct = correct && result.raw_bad == 0 && result.output_redzone &&
          result.workspace_redzone && result.fingerprint == stable_fingerprint;
    }
    if (!correct) {
      if (result.state != CellState::LaunchFailed) {
        result.state = CellState::CorrectnessFailed;
      }
      row_ok = false;
      continue;
    }

    if (options.measure) {
      bool cell_ok = true;
      for (int warmup = 0; warmup < options.warmup_rotations; ++warmup) {
        for (auto& handle : prepared) {
          if (handle->run(nullptr) != cutlass::Status::kSuccess) {
            result.state = CellState::LaunchFailed;
            cell_ok = false;
            break;
          }
          ++result.launches;
        }
        if (!cell_ok) break;
      }
      if (!cell_ok || hggcDeviceSynchronize() != hggcSuccess) {
        result.state = CellState::LaunchFailed;
        row_ok = false;
        continue;
      }

      std::vector<double> samples;
      samples.reserve(std::size_t(options.iterations));
      for (int iteration = 0; iteration < options.iterations; ++iteration) {
        EventPair events;
        // Poison outside the event span.  Each ranked sample must recreate a
        // valid final D and preserve both redzones; otherwise a skipped launch
        // could inherit the preceding correctness result and still look fast.
        if (!reset_canaries(inputs) ||
            hggcEventRecord(events.start, nullptr) != hggcSuccess) {
          result.state = CellState::TimingFailed;
          row_ok = false;
          break;
        }
        for (auto& handle : prepared) {
          if (handle->run(nullptr) != cutlass::Status::kSuccess) {
            result.state = CellState::LaunchFailed;
            cell_ok = false;
            break;
          }
          ++result.launches;
        }
        if (!cell_ok) {
          row_ok = false;
          break;
        }
        if (hggcEventRecord(events.stop, nullptr) != hggcSuccess ||
            hggcEventSynchronize(events.stop) != hggcSuccess) {
          result.state = CellState::TimingFailed;
          cell_ok = false;
          break;
        }
        double rotation_us = 0;
        if (!elapsed_us(events.start, events.stop, rotation_us) ||
            rotation_us <= 0) {
          result.state = CellState::TimingFailed;
          cell_ok = false;
          break;
        }
        if (!inspect_output(inputs, result, host_output) ||
            !inspect_workspace_redzone(inputs, plan.partial_bytes, result,
                                       host_workspace) ||
            result.raw_bad != 0 || !result.output_redzone ||
            !result.workspace_redzone ||
            result.fingerprint != stable_fingerprint) {
          result.state = CellState::CorrectnessFailed;
          cell_ok = false;
          break;
        }
        samples.push_back(rotation_us / inputs.cold_copies);
      }
      if (!cell_ok || samples.empty()) {
        row_ok = false;
        continue;
      }
      result.e2e_samples_us = samples;
      std::vector<double> ordered = samples;
      std::sort(ordered.begin(), ordered.end());
      result.e2e_min_us = ordered.front();
      result.e2e_max_us = ordered.back();
      result.e2e_median_us = ordered.size() & 1
          ? ordered[ordered.size() / 2]
          : 0.5 * (ordered[ordered.size() / 2 - 1] +
                   ordered[ordered.size() / 2]);
      result.e2e_mean_us =
          std::accumulate(samples.begin(), samples.end(), 0.0) /
          samples.size();
    }

    if (options.diagnose_spans) {
      int const span_repeats = std::max(options.span_repeats, 1);
      std::vector<double> producer_samples;
      std::vector<double> reducer_samples;
      producer_samples.reserve(std::size_t(span_repeats));
      reducer_samples.reserve(std::size_t(span_repeats));
      bool spans_ok = true;
      for (int repeat = 0; repeat < span_repeats; ++repeat) {
        EventTriple events;
        // Start after the copies consumed by the correctness gate.  On the
        // normal >=2.16x-L2 rotation this gives each span a distinct B/scale
        // artifact instead of repeatedly timing prepared.front().
        std::size_t const copy = options.span_repeats == 1
            ? std::size_t(0)
            : std::size_t(
                  (options.correctness_repeats + repeat) % inputs.cold_copies);
        cutlass::Status const status = prepared[copy]->run_with_events(
            events.events, nullptr);
        double producer = 0, reducer = 0;
        if (status != cutlass::Status::kSuccess ||
            !events.events.recorded ||
            hggcEventSynchronize(events.events.reducer_stop) != hggcSuccess ||
            !elapsed_us(events.events.producer_start,
                        events.events.producer_stop, producer) ||
            (splits > 1 &&
             !elapsed_us(events.events.producer_stop,
                         events.events.reducer_stop, reducer))) {
          spans_ok = false;
          break;
        }
        producer_samples.push_back(producer);
        if (splits > 1) reducer_samples.push_back(reducer);
      }
      if (!spans_ok || producer_samples.empty()) {
          result.state = CellState::TimingFailed;
          row_ok = false;
          continue;
      }
      auto summarize = [](std::vector<double> samples, double& median,
                          double& minimum, double& maximum) {
        std::sort(samples.begin(), samples.end());
        minimum = samples.front();
        maximum = samples.back();
        median = samples.size() & 1
            ? samples[samples.size() / 2]
            : 0.5 * (samples[samples.size() / 2 - 1] +
                     samples[samples.size() / 2]);
      };
      summarize(producer_samples, result.producer_us,
                result.producer_min_us, result.producer_max_us);
      if (splits > 1) {
        summarize(reducer_samples, result.reducer_us,
                  result.reducer_min_us, result.reducer_max_us);
      }
      result.span_samples = span_repeats;
      result.span_recorded = true;
    }

    result.state = CellState::Measured;
  }
  return row_ok;
}

template <class Launch>
bool measure_warm_aggregate_for_diagnostics(
    Launch&& launch, int iterations, double& average_us) {
  average_us = 0;
  if (iterations <= 0 || launch() != cutlass::Status::kSuccess ||
      hggcDeviceSynchronize() != hggcSuccess) {
    return false;
  }
  PpuTimer timer;
  timer.start();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (launch() != cutlass::Status::kSuccess) return false;
  }
  timer.stop();
  float const elapsed_ms = timer.elapsed_millis();
  if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0) return false;
  average_us = double(elapsed_ms) * 1000.0 / double(iterations);
  return true;
}

// Exact warm-resident A/B for the reshape claim.  This deliberately lives in
// the generated row TU: both ordinary and packed-A types are built from the
// same committed tactic, so a host-only reconstruction cannot silently compare
// a different mainloop.  The production sweep remains cold and continues to
// call run(); this seam is diagnostic only.
template <int TM, int TN, int TK, int WM, int WN, int Stages, int BChunk>
bool run_exact_warm_ab(
    DeviceInputs const& reshape, DeviceInputs const& internal,
    int iterations, ExactWarmAbResult& result) {
  using namespace cute;
  using Schedule = ppu_group_schedule::FinegrainedSchedule<kGroupSize>;
  using TileShape = Shape<Int<TM>, Int<TN>, Int<TK>>;
  using ScaleTile = Shape<Int<TN>,
      Int<ppu_group_schedule::scale_groups_v<TK, kGroupSize>>>;
  using WarpShape = Shape<Int<WM>, Int<WN>, Int<TK>>;
  using ShippingOrdinary = fpa_intb_ppu::DenseKernelTypes<
      QuantMode::FinegrainedScaleOnly, Schedule, TileShape, ScaleTile,
      WarpShape, Stages, true, int4_t, void, kArtifactTileK>;
  using Packed = fpa_intb_ppu::DensePackedAKernelTypes<
      1, QuantMode::FinegrainedScaleOnly, Schedule, TileShape, ScaleTile,
      WarpShape, Stages, true, int4_t, kArtifactTileK>;
  using Prepared = dense_splitk_parallel_ppu::PreparedOnePlaneLauncher<
      Packed, TileShape, WarpShape>;
  using ShippingOrdinaryGemm = typename ShippingOrdinary::Gemm;
  using ShippingOrdinaryKernel = typename ShippingOrdinary::GemmKernel;
  using OrdinaryMainloop = typename ShippingOrdinary::CollectiveMainloop;

  // The historical 7.854/7.696-us rows came from test_lowbit_dense_bench's
  // Cfg<...>, which used the default (void) scheduler and
  // EpilogueSimtVectorized.  The current shipping ordinary authority uses
  // SplitKSerialScheduler and ...WithoutEvt.  Rebuild the historical pair
  // while reusing the identical production mainloop; otherwise a timing miss
  // could be falsely charged to packed-A or the parallel Split-K producer.
  using HistoricalEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
      TileShape, WarpShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, cutlass::layout::RowMajor,
      128 / cutlass::sizeof_bits<cutlass::half_t>::value,
      cutlass::half_t, cutlass::layout::RowMajor,
      128 / cutlass::sizeof_bits<cutlass::half_t>::value,
      cutlass::epilogue::EpilogueSimtVectorized>::CollectiveOp;
  using HistoricalKernelDefault = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, OrdinaryMainloop, HistoricalEpilogue>;
  using HistoricalKernelExplicitVoid = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, OrdinaryMainloop, HistoricalEpilogue,
      void>;
  static_assert(std::is_same_v<HistoricalKernelDefault, HistoricalKernelExplicitVoid>,
                "historical Cfg default scheduler must remain void");
  using HistoricalGemm =
      cutlass::gemm::device::GemmUniversalAdapter<HistoricalKernelDefault>;

  static_assert(TM == 8 && WM == 8 && BChunk == 0,
                "reshape A/B is bounded to the committed M1 bc0 domain");
  static_assert(std::is_same_v<typename Packed::CollectiveMainloop,
                               typename Prepared::SplitTypes::CollectiveMainloop>,
                "internal producer must retain the exact packed-A mainloop");
  if (iterations <= 0 || reshape.rows != 1 || reshape.columns != 32768 ||
      reshape.inner != 512 || reshape.group_size != kGroupSize ||
      internal.rows != kM || internal.columns != kN || internal.inner != kK ||
      internal.group_size != kGroupSize || reshape.cold_copies != 1 ||
      internal.cold_copies != 1) {
    return false;
  }

  auto output_ptr = [](DeviceInputs const& inputs) {
    return inputs.output_storage + kOutputGuardElements;
  };
  auto workspace_ptr = [](DeviceInputs const& inputs) {
    return inputs.workspace_storage + kWorkspaceGuardBytes;
  };
  auto workspace_bytes = [](DeviceInputs const& inputs) {
    return inputs.workspace_storage_bytes - 2 * kWorkspaceGuardBytes;
  };

  using StrideA = typename ShippingOrdinaryKernel::StrideA;
  using StrideB = typename ShippingOrdinaryKernel::StrideB;
  using StrideC = typename ShippingOrdinaryKernel::StrideC;
  using StrideD = typename ShippingOrdinaryKernel::StrideD;
  using StrideScale = typename OrdinaryMainloop::StrideScale;
  constexpr int LowFold = ShippingOrdinary::MainloopPolicy::ArtifactLowFold;
  StrideA sA = cutlass::make_cute_packed_stride(
      StrideA{}, make_shape(reshape.rows, reshape.inner, 1));
  StrideB sB = cutlass::make_cute_packed_stride(
      StrideB{}, make_shape(reshape.columns / LowFold,
                            reshape.inner * LowFold, 1));
  StrideC sC = cutlass::make_cute_packed_stride(
      StrideC{}, make_shape(reshape.rows, reshape.columns, 1));
  StrideD sD = cutlass::make_cute_packed_stride(
      StrideD{}, make_shape(reshape.rows, reshape.columns, 1));
  StrideScale sS = cutlass::make_cute_packed_stride(
      StrideScale{}, make_shape(
          reshape.columns, reshape.inner / reshape.group_size, 1));
  typename HistoricalGemm::Arguments historical_args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {reshape.rows, reshape.columns, reshape.inner, 1},
      {reshape.a, sA,
       reinterpret_cast<int4_t const*>(reshape.resident_b), sB,
       reshape.scales, sS, reshape.group_size, reshape.zeros},
      {{typename ShippingOrdinary::ElementAccumulator(1.f),
        typename ShippingOrdinary::ElementAccumulator(0.f)},
       static_cast<typename ShippingOrdinary::ElementC*>(nullptr), sC,
       output_ptr(reshape), sD}};
  typename ShippingOrdinaryGemm::Arguments shipping_ordinary_args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {reshape.rows, reshape.columns, reshape.inner, 1},
      {reshape.a, sA,
       reinterpret_cast<int4_t const*>(reshape.resident_b), sB,
       reshape.scales, sS, reshape.group_size, reshape.zeros},
      {{typename ShippingOrdinary::ElementAccumulator(1.f),
        typename ShippingOrdinary::ElementAccumulator(0.f)},
       static_cast<typename ShippingOrdinary::ElementC*>(nullptr), sC,
       output_ptr(reshape), sD},
      1};
  HistoricalGemm historical;
  ShippingOrdinaryGemm shipping_ordinary;
  Prepared packed_reshape;
  Prepared packed_internal_s2;
  Prepared packed_internal_s4;
  Prepared packed_internal;
  if (HistoricalGemm::can_implement(historical_args) != cutlass::Status::kSuccess ||
      HistoricalGemm::get_workspace_size(historical_args) != 0 ||
      historical.initialize(historical_args, nullptr, nullptr) !=
          cutlass::Status::kSuccess ||
      ShippingOrdinaryGemm::can_implement(shipping_ordinary_args) !=
          cutlass::Status::kSuccess ||
      ShippingOrdinaryGemm::get_workspace_size(shipping_ordinary_args) != 0 ||
      shipping_ordinary.initialize(shipping_ordinary_args, nullptr, nullptr) !=
          cutlass::Status::kSuccess ||
      !packed_reshape.initialize(
          reshape.a, reinterpret_cast<int4_t const*>(reshape.resident_b),
          reshape.scales, reshape.zeros, output_ptr(reshape), reshape.rows,
          reshape.columns, reshape.inner, reshape.group_size, 1,
          workspace_ptr(reshape), workspace_bytes(reshape), nullptr) ||
      !packed_internal.initialize(
          internal.a, reinterpret_cast<int4_t const*>(internal.resident_b),
          internal.scales, internal.zeros, output_ptr(internal), internal.rows,
          internal.columns, internal.inner, internal.group_size, 8,
          workspace_ptr(internal), workspace_bytes(internal), nullptr) ||
      !packed_internal_s2.initialize(
          internal.a, reinterpret_cast<int4_t const*>(internal.resident_b),
          internal.scales, internal.zeros, output_ptr(internal), internal.rows,
          internal.columns, internal.inner, internal.group_size, 2,
          workspace_ptr(internal), workspace_bytes(internal), nullptr) ||
      !packed_internal_s4.initialize(
          internal.a, reinterpret_cast<int4_t const*>(internal.resident_b),
          internal.scales, internal.zeros, output_ptr(internal), internal.rows,
          internal.columns, internal.inner, internal.group_size, 4,
          workspace_ptr(internal), workspace_bytes(internal), nullptr)) {
    return false;
  }

  dense_splitk_parallel_ppu::WorkspacePlan reshape_plan, internal_s2_plan,
      internal_s4_plan, internal_plan;
  if (!dense_splitk_parallel_ppu::query_workspace_plan(
          reshape.rows, reshape.columns, 1, reshape_plan) ||
      !dense_splitk_parallel_ppu::query_fused_workspace_plan(
          internal.rows, internal.columns, 2, TM, TN, internal_s2_plan) ||
      !dense_splitk_parallel_ppu::query_fused_workspace_plan(
          internal.rows, internal.columns, 4, TM, TN, internal_s4_plan) ||
      !dense_splitk_parallel_ppu::query_fused_workspace_plan(
          internal.rows, internal.columns, 8, TM, TN, internal_plan)) {
    return false;
  }
  auto validate = [](DeviceInputs const& inputs, std::size_t plan_bytes,
                     std::size_t counter_offset, std::size_t counter_bytes,
                     auto&& launch, uint64_t& raw_bad,
                     bool* counters_zero = nullptr) {
    CellResult observed;
    std::vector<half_t> host_output;
    std::vector<char> host_workspace;
    bool const ok = reset_canaries(inputs) &&
        launch() == cutlass::Status::kSuccess &&
        hggcDeviceSynchronize() == hggcSuccess &&
        inspect_output(inputs, observed, host_output) &&
        inspect_workspace_redzone(inputs, plan_bytes, observed, host_workspace);
    if (counters_zero != nullptr) {
      *counters_zero = ok && counter_bytes != 0 &&
          counter_offset + counter_bytes <= plan_bytes;
      if (*counters_zero) {
        auto first = host_workspace.begin() + std::ptrdiff_t(
            kWorkspaceGuardBytes + counter_offset);
        *counters_zero = std::all_of(
            first, first + std::ptrdiff_t(counter_bytes),
            [](char value) { return value == 0; });
      }
    }
    raw_bad = observed.raw_bad;
    return ok && observed.raw_bad == 0 && observed.output_redzone &&
        observed.workspace_redzone;
  };
  auto historical_launch = [&] { return historical.run(nullptr); };
  auto shipping_ordinary_launch = [&] {
    return shipping_ordinary.run(nullptr);
  };
  auto packed_reshape_launch = [&] { return packed_reshape.run(nullptr); };
  auto packed_internal_full = [&] { return packed_internal.run(nullptr); };
  auto packed_internal_producer = [&] {
    return packed_internal.run_producer_only_for_diagnostics(nullptr);
  };
  auto packed_internal_reducer = [&] {
    return packed_internal.run_reducer_only_for_diagnostics(nullptr);
  };
  auto packed_internal_fused = [&] {
    return packed_internal.run_fused_last_arriver(nullptr);
  };
  auto packed_internal_publish_only = [&] {
    return packed_internal.run_publish_protocol_only_for_diagnostics(nullptr);
  };
  auto packed_internal_fused_validation = [&] {
    cutlass::Status const reset =
        packed_internal.reset_fused_counters_for_diagnostics(nullptr);
    return reset == cutlass::Status::kSuccess
        ? packed_internal.run_fused_last_arriver(nullptr)
        : reset;
  };
  auto packed_internal_s2_fused_validation = [&] {
    cutlass::Status const reset =
        packed_internal_s2.reset_fused_counters_for_diagnostics(nullptr);
    return reset == cutlass::Status::kSuccess
        ? packed_internal_s2.run_fused_last_arriver(nullptr)
        : reset;
  };
  auto packed_internal_s4_fused_validation = [&] {
    cutlass::Status const reset =
        packed_internal_s4.reset_fused_counters_for_diagnostics(nullptr);
    return reset == cutlass::Status::kSuccess
        ? packed_internal_s4.run_fused_last_arriver(nullptr)
        : reset;
  };
  result.packed_internal_fast_reducer =
      packed_internal.reduction_fast_path_selected_for_diagnostics();
  result.packed_internal_fused_selected =
      packed_internal_s2.fused_last_arriver_selected_for_diagnostics() &&
      packed_internal_s4.fused_last_arriver_selected_for_diagnostics() &&
      packed_internal.fused_last_arriver_selected_for_diagnostics();
  result.packed_internal_publish_only_selected =
      packed_internal.publish_protocol_only_selected_for_diagnostics();

  // This timing-only arm must prove that it is the same producer plus the
  // exact publication lifecycle, not a second approximate model.  Compare its
  // FP32 partial bytes with producer-only, require the terminal counter reset,
  // and require every D element to retain the poison value because the same
  // compiled kernel was told to skip final reduction/output.
  auto validate_publish_only = [&] {
    CellResult producer_observed, publish_observed;
    std::vector<half_t> producer_output, publish_output;
    std::vector<char> producer_workspace, publish_workspace;
    if (!reset_canaries(internal) ||
        packed_internal_producer() != cutlass::Status::kSuccess ||
        hggcDeviceSynchronize() != hggcSuccess ||
        !inspect_output(internal, producer_observed, producer_output) ||
        !inspect_workspace_redzone(
            internal, internal_plan.total_bytes, producer_observed,
            producer_workspace) ||
        !producer_observed.output_redzone ||
        !producer_observed.workspace_redzone ||
        !reset_canaries(internal) ||
        packed_internal.reset_fused_counters_for_diagnostics(nullptr) !=
            cutlass::Status::kSuccess ||
        packed_internal_publish_only() != cutlass::Status::kSuccess ||
        hggcDeviceSynchronize() != hggcSuccess ||
        !inspect_output(internal, publish_observed, publish_output) ||
        !inspect_workspace_redzone(
            internal, internal_plan.total_bytes, publish_observed,
            publish_workspace) ||
        !publish_observed.output_redzone ||
        !publish_observed.workspace_redzone) {
      return false;
    }
    auto const producer_partial = producer_workspace.begin() +
        std::ptrdiff_t(kWorkspaceGuardBytes);
    auto const publish_partial = publish_workspace.begin() +
        std::ptrdiff_t(kWorkspaceGuardBytes);
    result.packed_internal_publish_only_partial_byte_diff =
        uint64_t(std::inner_product(
            producer_partial,
            producer_partial + std::ptrdiff_t(internal_plan.partial_bytes),
            publish_partial, uint64_t(0), std::plus<uint64_t>{},
            [](char lhs, char rhs) { return uint64_t(lhs != rhs); }));
    half_t const* published_d =
        publish_output.data() + kOutputGuardElements;
    std::size_t const output_elements =
        std::size_t(internal.rows) * std::size_t(internal.columns);
    result.packed_internal_publish_only_d_untouched = std::all_of(
        published_d, published_d + output_elements,
        [](half_t value) { return value.raw() == uint16_t(0x7e00); });
    auto const counter_first = publish_workspace.begin() + std::ptrdiff_t(
        kWorkspaceGuardBytes + internal_plan.counter_offset);
    result.packed_internal_publish_only_counters_zero = std::all_of(
        counter_first,
        counter_first + std::ptrdiff_t(internal_plan.counter_bytes),
        [](char value) { return value == 0; });
    return result.packed_internal_publish_only_partial_byte_diff == 0 &&
        result.packed_internal_publish_only_d_untouched &&
        result.packed_internal_publish_only_counters_zero;
  };

  // Reuse the same counter allocation without an operator-side memset.  Each
  // launch must both reproduce the exact output and retire every q counter to
  // zero before the next launch.  This is distinct from validate(), whose
  // explicit reset establishes a known first-launch precondition.
  auto validate_fused_reuse = [&] {
    if (!reset_canaries(internal) ||
        packed_internal.reset_fused_counters_for_diagnostics(nullptr) !=
            cutlass::Status::kSuccess) {
      return false;
    }
    result.packed_internal_fused_reuse_passes = 0;
    for (int repetition = 0; repetition < 8; ++repetition) {
      CellResult observed;
      std::vector<half_t> host_output;
      std::vector<char> host_workspace;
      // Poison this launch's observable D and every partial plane while
      // deliberately preserving the completion counters.  A missing D store
      // or missing peer partial must not inherit the previous correct launch.
      if (!reset_output_canaries(internal) ||
          hggcMemset(
              internal.workspace_storage + kWorkspaceGuardBytes,
              kWorkspaceCanary, internal_plan.partial_bytes) != hggcSuccess ||
          packed_internal.run_fused_last_arriver(nullptr) !=
              cutlass::Status::kSuccess ||
          hggcDeviceSynchronize() != hggcSuccess ||
          !inspect_output(internal, observed, host_output) ||
          !inspect_workspace_redzone(
              internal, internal_plan.total_bytes, observed, host_workspace) ||
          observed.raw_bad != 0 || !observed.output_redzone ||
          !observed.workspace_redzone) {
        return false;
      }
      auto first = host_workspace.begin() + std::ptrdiff_t(
          kWorkspaceGuardBytes + internal_plan.counter_offset);
      if (!std::all_of(
              first,
              first + std::ptrdiff_t(internal_plan.counter_bytes),
              [](char value) { return value == 0; })) {
        return false;
      }
      ++result.packed_internal_fused_reuse_passes;
    }
    return true;
  };

  if (!validate(reshape, reshape_plan.partial_bytes, 0, 0, historical_launch,
                result.historical_reshape_bad) ||
      !validate(reshape, reshape_plan.partial_bytes, 0, 0,
                shipping_ordinary_launch,
                result.shipping_ordinary_reshape_bad) ||
      !validate(reshape, reshape_plan.partial_bytes, 0, 0,
                packed_reshape_launch,
                result.packed_reshape_bad) ||
      !validate(internal, internal_plan.partial_bytes, 0, 0,
                packed_internal_full,
                result.packed_internal_bad) ||
      !result.packed_internal_fused_selected) {
    return false;
  }

  uint64_t fused_s2_bad = 0, fused_s4_bad = 0, fused_s8_bad = 0;
  bool fused_s2_counters_zero = false, fused_s4_counters_zero = false,
       fused_s8_counters_zero = false;
  result.packed_internal_fused_slices_passes = 0;
  if (!validate(internal, internal_s2_plan.total_bytes,
                internal_s2_plan.counter_offset,
                internal_s2_plan.counter_bytes,
                packed_internal_s2_fused_validation, fused_s2_bad,
                &fused_s2_counters_zero)) {
    return false;
  }
  ++result.packed_internal_fused_slices_passes;
  if (!validate(internal, internal_s4_plan.total_bytes,
                internal_s4_plan.counter_offset,
                internal_s4_plan.counter_bytes,
                packed_internal_s4_fused_validation, fused_s4_bad,
                &fused_s4_counters_zero)) {
    return false;
  }
  ++result.packed_internal_fused_slices_passes;
  if (!validate(internal, internal_plan.total_bytes,
                internal_plan.counter_offset, internal_plan.counter_bytes,
                packed_internal_fused_validation, fused_s8_bad,
                &fused_s8_counters_zero)) {
    return false;
  }
  ++result.packed_internal_fused_slices_passes;
  result.packed_internal_fused_bad =
      fused_s2_bad + fused_s4_bad + fused_s8_bad;
  result.packed_internal_fused_counters_zero =
      fused_s2_counters_zero && fused_s4_counters_zero &&
      fused_s8_counters_zero;

  if (!validate_fused_reuse() || !validate_publish_only() ||
      !measure_warm_aggregate_for_diagnostics(
          historical_launch, iterations, result.historical_reshape_us) ||
      !measure_warm_aggregate_for_diagnostics(
          shipping_ordinary_launch, iterations,
          result.shipping_ordinary_reshape_us) ||
      !measure_warm_aggregate_for_diagnostics(
          packed_reshape_launch, iterations, result.packed_reshape_us) ||
      !measure_warm_aggregate_for_diagnostics(
          packed_internal_producer, iterations,
          result.packed_internal_producer_us) ||
      !measure_warm_aggregate_for_diagnostics(
          packed_internal_reducer, iterations,
          result.packed_internal_reducer_us) ||
      !measure_warm_aggregate_for_diagnostics(
          packed_internal_full, iterations,
          result.packed_internal_full_us) ||
      !measure_warm_aggregate_for_diagnostics(
          packed_internal_fused, iterations,
          result.packed_internal_fused_us) ||
      packed_internal.reset_fused_counters_for_diagnostics(nullptr) !=
          cutlass::Status::kSuccess ||
      !measure_warm_aggregate_for_diagnostics(
          packed_internal_publish_only, iterations,
          result.packed_internal_publish_only_us)) {
    return false;
  }

  uint64_t historical_post_bad = 0, shipping_ordinary_post_bad = 0,
           packed_reshape_post_bad = 0, packed_internal_post_bad = 0,
           packed_internal_fused_post_bad = 0;
  bool fused_post_counters_zero = false;
  result.post_timing_correct =
      validate(reshape, reshape_plan.partial_bytes, 0, 0, historical_launch,
               historical_post_bad) &&
      validate(reshape, reshape_plan.partial_bytes, 0, 0,
               shipping_ordinary_launch,
               shipping_ordinary_post_bad) &&
      validate(reshape, reshape_plan.partial_bytes, 0, 0,
               packed_reshape_launch,
               packed_reshape_post_bad) &&
      validate(internal, internal_plan.partial_bytes, 0, 0,
               packed_internal_full,
               packed_internal_post_bad) &&
      validate_publish_only() &&
      validate(internal, internal_plan.total_bytes,
               internal_plan.counter_offset, internal_plan.counter_bytes,
               packed_internal_fused_validation,
               packed_internal_fused_post_bad, &fused_post_counters_zero) &&
      historical_post_bad == 0 && shipping_ordinary_post_bad == 0 &&
      packed_reshape_post_bad == 0 && packed_internal_post_bad == 0 &&
      packed_internal_fused_post_bad == 0 && fused_post_counters_zero;
  return result.post_timing_correct;
}

}  // namespace dense_splitk_sweep
