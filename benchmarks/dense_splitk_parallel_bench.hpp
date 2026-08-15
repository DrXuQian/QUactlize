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
  if (host_storage.size() != 2 * kOutputGuardElements + std::size_t(kM) * kN) {
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
  for (std::size_t i = 0; i < std::size_t(kM) * kN; ++i) {
    result.raw_bad += output[i].raw() != inputs.golden[i].raw();
  }
  result.fingerprint = fnv1a_half_bits(output, std::size_t(kM) * kN);
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

inline bool reset_canaries(DeviceInputs const& inputs) {
  std::vector<half_t> guards(inputs.output_storage_elements,
                             half_t::bitcast(uint16_t(0x7e00)));
  std::fill(guards.begin(),
            guards.begin() + std::ptrdiff_t(kOutputGuardElements),
            half_t::bitcast(kOutputLeftCanary));
  std::fill(guards.end() - std::ptrdiff_t(kOutputGuardElements),
            guards.end(), half_t::bitcast(kOutputRightCanary));
  return hggcMemcpy(inputs.output_storage, guards.data(),
                    guards.size() * sizeof(half_t),
                    hggcMemcpyHostToDevice) == hggcSuccess &&
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

  if (inputs.cold_copies <= 0 || inputs.resident_b == nullptr ||
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
      EventTriple events;
      cutlass::Status const status = prepared.front()->run_with_events(
          events.events, nullptr);
      if (status != cutlass::Status::kSuccess ||
          !events.events.recorded ||
          hggcEventSynchronize(events.events.reducer_stop) != hggcSuccess ||
          !elapsed_us(events.events.producer_start,
                      events.events.producer_stop, result.producer_us)) {
        result.state = CellState::TimingFailed;
        row_ok = false;
        continue;
      }
      if (splits > 1) {
        if (!elapsed_us(events.events.producer_stop,
                        events.events.reducer_stop, result.reducer_us)) {
          result.state = CellState::TimingFailed;
          row_ok = false;
          continue;
        }
      }
      result.span_recorded = true;
    }

    result.state = CellState::Measured;
  }
  return row_ok;
}

}  // namespace dense_splitk_sweep
