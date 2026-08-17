#pragma once
// Runtime/type authority for generated all-format ScaleFirst sweep shards.
//
// Full-output boards:
//   - ordinary non-persistent DP;
//   - persistent DP at every deduplicated capacity/balanced grid admitted by
//     the exact compiled kernel occupancy.
// Diagnostic board:
//   - fixed Split-K S2/S4/S8 producer only.  The deterministic reducer is run
//     outside timing and must close raw FP16 correctness before and after the
//     producer samples.

// A producer-only sample is never a product result and never competes with a
// full-output row.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/util/device_memory.h"
#include "dense_splitk_multiformat_ppu.cuh"
#include "helper.h"
#include "ppu_group_schedule.hpp"
#include "scalefirst_persistent_policy.hpp"
#include "quactlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_persistent.hpp"

namespace scalefirst_internal_sweep {

using half_t = cutlass::half_t;
using QuantMode = fpa_intb_ppu::QuantMode;

enum class State : int {
  Measured,
  ShippingSharedStorage,
  PersistentSharedStorage,
  SplitSharedStorage,
  SplitPartition,
  PipelineDepth,
  Occupancy,
  CanImplement,
  Initialize,
  Launch,
  Correctness,
  Timing,
};

inline char const* state_name(State state) {
  switch (state) {
    case State::Measured: return "MEASURED";
    case State::ShippingSharedStorage: return "INADMISSIBLE_SHIPPING_SMEM";
    case State::PersistentSharedStorage: return "INADMISSIBLE_PERSISTENT_SMEM";
    case State::SplitSharedStorage: return "INADMISSIBLE_SPLIT_SMEM";
    case State::SplitPartition: return "INADMISSIBLE_SPLIT_PARTITION";
    case State::PipelineDepth: return "INADMISSIBLE_PIPELINE_DEPTH";
    case State::Occupancy: return "INADMISSIBLE_OCCUPANCY";
    case State::CanImplement: return "INADMISSIBLE_CAN_IMPLEMENT";
    case State::Initialize: return "INITIALIZE_FAIL";
    case State::Launch: return "LAUNCH_FAIL";
    case State::Correctness: return "RAW_FP16_MISMATCH";
    case State::Timing: return "TIMING_FAIL";
  }
  return "UNKNOWN";
}

struct Options {
  int iterations = 5;
  int correctness_repeats = 2;
  bool measure = true;
};

struct DeviceInputs {
  half_t const* a = nullptr;
  std::uint8_t const* low = nullptr;
  std::uint8_t const* high = nullptr;
  half_t const* scales = nullptr;
  half_t const* zeros = nullptr;
  half_t* output = nullptr;
  char* workspace = nullptr;
  std::size_t workspace_bytes = 0;
  half_t const* golden = nullptr;  // host pointer
  int m = 0, n = 0, k = 0;
  int device = 0, cu = 0;
};

struct CellResult {
  char const* algorithm = nullptr;
  char const* metric_scope = nullptr;
  char const* policy = nullptr;
  int split = 1;
  int grid = 0;
  int occupancy = 0;
  std::uint64_t capacity_b_mask = 0;
  std::uint64_t balanced_b_mask = 0;
  State state = State::CanImplement;
  bool reducer_correctness_untimed = false;
  std::uint64_t raw_bad = 0;
  std::uint64_t fingerprint = 0;
  std::size_t shipping_smem = 0;
  std::size_t persistent_smem = 0;
  std::size_t split_smem = 0;
  std::size_t partial_bytes = 0;
  double median_us = 0, min_us = 0, max_us = 0;
  std::vector<double> samples_us;
};

struct RowResult { std::vector<CellResult> cells; };
using RunRow = bool (*)(DeviceInputs const&, Options const&, RowResult&);

struct RegistryRow {
  char const* symbol;
  int qtype, artifact_tile_k;
  int tm, tn, tk, wm, wn, stages, bchunk;
  RunRow run;
};

template <int QType> struct Format;
template <> struct Format<8> {
  using Low = std::int8_t; using High = void;
  static constexpr int GroupSize = 32;
  static constexpr QuantMode Mode = QuantMode::FinegrainedScaleOnly;
  static constexpr ppu_tactics::Format TacticFormat = ppu_tactics::Format::I8;
};
template <> struct Format<10> {
  using Low = cutlass::uint2b_t; using High = void;
  static constexpr int GroupSize = 16;
  static constexpr QuantMode Mode = QuantMode::FinegrainedScaleZero;
  static constexpr ppu_tactics::Format TacticFormat = ppu_tactics::Format::I2;
};
template <> struct Format<11> {
  using Low = cutlass::uint2b_t; using High = cutlass::uint1b_t;
  static constexpr int GroupSize = 16;
  static constexpr QuantMode Mode = QuantMode::FinegrainedScaleOnly;
  static constexpr ppu_tactics::Format TacticFormat = ppu_tactics::Format::Q3_K;
};
template <> struct Format<12> {
  using Low = cutlass::int4b_t; using High = void;
  static constexpr int GroupSize = 32;
  static constexpr QuantMode Mode = QuantMode::FinegrainedScaleZero;
  static constexpr ppu_tactics::Format TacticFormat = ppu_tactics::Format::I4;
};
template <> struct Format<13> {
  using Low = cutlass::int4b_t; using High = cutlass::uint1b_t;
  static constexpr int GroupSize = 32;
  static constexpr QuantMode Mode = QuantMode::FinegrainedScaleZero;
  static constexpr ppu_tactics::Format TacticFormat = ppu_tactics::Format::Q5_K;
};
template <> struct Format<14> {
  using Low = cutlass::int4b_t; using High = cutlass::uint2b_t;
  static constexpr int GroupSize = 16;
  static constexpr QuantMode Mode = QuantMode::FinegrainedScaleOnly;
  static constexpr ppu_tactics::Format TacticFormat = ppu_tactics::Format::Q6_K;
};

template <class T> struct ElementBits {
  static constexpr int value = cutlass::sizeof_bits<T>::value;
};
template <> struct ElementBits<void> { static constexpr int value = 0; };

template <int QType, int ArtifactTileK, int TM, int TN, int TK,
          int WM, int WN, int Stages, int BChunk>
struct RowTypes {
  using F = Format<QType>;
  using Low = typename F::Low;
  using High = typename F::High;
  using Schedule = ppu_group_schedule::FinegrainedSchedule<F::GroupSize>;
  using Tile = cute::Shape<cute::C<TM>, cute::C<TN>, cute::C<TK>>;
  using ScaleTile = cute::Shape<cute::C<TN>, cute::C<
      ppu_group_schedule::scale_groups_v<TK, F::GroupSize>>>;
  using Warp = cute::Shape<cute::C<WM>, cute::C<WN>, cute::C<TK>>;
  using Shipping = fpa_intb_ppu::DenseKernelTypes<
      F::Mode, Schedule, Tile, ScaleTile, Warp, Stages, true,
      Low, High, ArtifactTileK>;
  using Split = dense_splitk_parallel_ppu::KernelTypes<Shipping, Tile, Warp>;
  using PersistentKernel = cutlass::gemm::kernel::PersistentMixedInputKernel<
      cute::Shape<int, int, int, int>,
      typename Shipping::CollectiveMainloop,
      typename Shipping::CollectiveEpilogue>;
  using PersistentGemm = cutlass::gemm::device::GemmUniversalAdapter<
      PersistentKernel>;
  using ShippingGemm = typename Shipping::Gemm;
  using ShippingKernel = typename Shipping::GemmKernel;
  using SplitGemm = typename Split::Gemm;
  using SplitKernel = typename Split::GemmKernel;
  using Reduction = typename Split::Reduction;
  using Prepared = dense_splitk_parallel_ppu::PreparedMultiformatLauncher<
      Shipping, Tile, Warp, High, false>;
  static constexpr ppu_tactics::FormatSpec spec{
      F::TacticFormat, "scalefirst", ElementBits<Low>::value,
      ElementBits<High>::value, F::GroupSize,
      ppu_mixed_policy::has_zero(F::Mode) ? 2 : 1};
  static constexpr ppu_tactics::Candidate candidate{
      spec, TM, TN, TK, WM, WN, ArtifactTileK, BChunk};

  static_assert(PPU_B_CHUNK == BChunk,
                "generated unit must bind the requested BChunk policy");
  static_assert(!dense_splitk_parallel_ppu::MainloopUsesPackedMetadata<
                    typename Shipping::CollectiveMainloop>::value,
                "ScaleFirst must consume fp16 scale/zero planes, not GGUF units");
  static_assert(Shipping::MainloopPolicy::ArtifactLowFold ==
                    ppu_tactics::artifact_low_fold(candidate));
  static_assert(Shipping::MainloopPolicy::ArtifactHighFold ==
                    ppu_tactics::artifact_high_fold(candidate));
  static_assert(std::is_same_v<typename SplitKernel::CollectiveMainloop,
                               typename Shipping::CollectiveMainloop>,
                "fixed Split-K must reuse the exact ScaleFirst collective");
  static_assert(std::is_same_v<typename PersistentKernel::CollectiveMainloop,
                               typename Shipping::CollectiveMainloop>,
                "persistent DP must reuse the exact ScaleFirst collective");
};

template <int QType, int ArtifactTileK, int TM, int TN, int TK,
          int WM, int WN, int Stages, int BChunk>
constexpr bool admit_row_type() {
  using T = RowTypes<QType, ArtifactTileK, TM, TN, TK,
                     WM, WN, Stages, BChunk>;
  return T::Shipping::SharedStorageSize > 0 &&
         T::PersistentKernel::SharedStorageSize > 0 &&
         T::SplitKernel::SharedStorageSize > 0;
}

class EventPair {
 public:
  hggcEvent_t start{}, stop{};
  EventPair() {
    CUTLASS_PPU_CHECK(hggcEventCreate(&start));
    CUTLASS_PPU_CHECK(hggcEventCreate(&stop));
  }
  ~EventPair() {
    if (start) hggcEventDestroy(start);
    if (stop) hggcEventDestroy(stop);
  }
};

inline bool inspect(DeviceInputs const& in, CellResult& result) {
  std::vector<half_t> host(std::size_t(in.m) * in.n);
  if (hggcMemcpy(host.data(), in.output, host.size() * sizeof(half_t),
                 hggcMemcpyDeviceToHost) != hggcSuccess) return false;
  std::uint64_t hash = UINT64_C(1469598103934665603), bad = 0;
  for (std::size_t i = 0; i < host.size(); ++i) {
    auto bits = host[i].raw();
    bad += bits != in.golden[i].raw();
    hash ^= std::uint8_t(bits); hash *= UINT64_C(1099511628211);
    hash ^= std::uint8_t(bits >> 8); hash *= UINT64_C(1099511628211);
  }
  result.raw_bad = bad;
  result.fingerprint = hash;
  return true;
}

template <class Launch>
bool measure(Launch&& launch, int iterations, CellResult& result) {
  result.samples_us.clear();
  for (int i = 0; i < iterations; ++i) {
    EventPair events;
    if (hggcEventRecord(events.start, nullptr) != hggcSuccess ||
        launch() != cutlass::Status::kSuccess ||
        hggcEventRecord(events.stop, nullptr) != hggcSuccess ||
        hggcEventSynchronize(events.stop) != hggcSuccess) return false;
    float ms = 0;
    if (hggcEventElapsedTime(&ms, events.start, events.stop) != hggcSuccess ||
        !(ms > 0) || !std::isfinite(ms)) return false;
    result.samples_us.push_back(double(ms) * 1000.0);
  }
  std::sort(result.samples_us.begin(), result.samples_us.end());
  result.min_us = result.samples_us.front();
  result.max_us = result.samples_us.back();
  auto const count = result.samples_us.size();
  result.median_us = count & 1 ? result.samples_us[count / 2] :
      0.5 * (result.samples_us[count/2 - 1] + result.samples_us[count/2]);
  return true;
}

template <class Launch>
bool validate_and_measure(DeviceInputs const& in, Options const& options,
                          Launch&& launch, CellResult& result) {
  std::uint64_t first = 0;
  for (int repeat = 0; repeat < options.correctness_repeats; ++repeat) {
    if (launch() != cutlass::Status::kSuccess ||
        hggcDeviceSynchronize() != hggcSuccess || !inspect(in, result) ||
        result.raw_bad != 0 || (repeat && result.fingerprint != first)) {
      result.state = State::Correctness;
      return false;
    }
    first = result.fingerprint;
  }
  if (options.measure && !measure(launch, options.iterations, result)) {
    result.state = State::Timing;
    return false;
  }
  result.state = State::Measured;
  return true;
}

template <int QType, int ArtifactTileK, int TM, int TN, int TK,
          int WM, int WN, int Stages, int BChunk>
bool run_row(DeviceInputs const& in, Options const& options, RowResult& row) {
  using T = RowTypes<QType, ArtifactTileK, TM, TN, TK,
                     WM, WN, Stages, BChunk>;
  using F = typename T::F;
  using High = typename T::High;
  using ShippingGemm = typename T::ShippingGemm;
  using ShippingKernel = typename T::ShippingKernel;
  using PersistentGemm = typename T::PersistentGemm;
  using PersistentKernel = typename T::PersistentKernel;
  using SplitGemm = typename T::SplitGemm;
  using SplitKernel = typename T::SplitKernel;
  using Reduction = typename T::Reduction;

  row.cells.clear();
  if (!in.a || !in.low || !in.scales || !in.output || !in.workspace ||
      !in.golden || in.m <= 0 || in.n <= 0 || in.k <= 0 || in.k % TK ||
      in.cu <= 0 || (!std::is_void_v<High> && !in.high) ||
      (ppu_mixed_policy::has_zero(F::Mode) ? !in.zeros : bool(in.zeros))) {
    return false;
  }

  auto mainloop = T::Prepared::make_mainloop_arguments(
      in.a, reinterpret_cast<typename T::Low const*>(in.low), in.scales,
      in.zeros, in.m, in.n, in.k, F::GroupSize,
      reinterpret_cast<High const*>(in.high));
  using StrideC = typename ShippingKernel::StrideC;
  using StrideD = typename ShippingKernel::StrideD;
  StrideC sC = cutlass::make_cute_packed_stride(
      StrideC{}, cute::make_shape(in.m, in.n, 1));
  StrideD sD = cutlass::make_cute_packed_stride(
      StrideD{}, cute::make_shape(in.m, in.n, 1));
  auto make_cell = [&](char const* algorithm, char const* scope) -> CellResult& {
    row.cells.emplace_back();
    auto& cell = row.cells.back();
    cell.algorithm = algorithm; cell.metric_scope = scope;
    cell.shipping_smem = T::Shipping::SharedStorageSize;
    cell.persistent_smem = PersistentKernel::SharedStorageSize;
    cell.split_smem = SplitKernel::SharedStorageSize;
    return cell;
  };

  // Ordinary full-output S1.
  {
    auto& cell = make_cell("NONPERSISTENT", "FULL_OUTPUT");
    cell.policy = "ordinary";
    cell.grid = ((in.m + TM - 1) / TM) * ((in.n + TN - 1) / TN);
    if constexpr (T::Shipping::SharedStorageSize > ppu_tactics::kBlockSmemBytes) {
      cell.state = State::ShippingSharedStorage;
    } else {
      typename ShippingGemm::Arguments args{
          cutlass::gemm::GemmUniversalMode::kGemm,
          {in.m, in.n, in.k, 1}, mainloop,
          {{1.f, 0.f}, static_cast<half_t*>(nullptr), sC, in.output, sD}, 1};
      ShippingGemm gemm;
      if (ShippingGemm::can_implement(args) != cutlass::Status::kSuccess) {
        cell.state = State::CanImplement;
      } else if (gemm.initialize(args, nullptr, nullptr) != cutlass::Status::kSuccess) {
        cell.state = State::Initialize;
        return false;
      } else if (!validate_and_measure(in, options,
                                       [&] { return gemm.run(nullptr); }, cell)) {
        return false;
      }
    }
  }

  // Persistent full-output S1.  The exact final kernel supplies occupancy;
  // capacity/balanced grids are deduplicated by the shared policy.
  {
    int const occupancy = PersistentGemm::maximum_active_blocks();
    std::uint64_t const q = std::uint64_t((in.m + TM - 1) / TM) *
                            std::uint64_t((in.n + TN - 1) / TN);
    auto grids = quactlize::scalefirst_policy::grid_space(q, in.cu, occupancy);
    if (occupancy <= 0 || grids.empty()) {
      auto& cell = make_cell("PERSISTENT", "FULL_OUTPUT");
      cell.policy = "capacity+balanced";
      cell.occupancy = occupancy;
      cell.state = State::Occupancy;
    } else for (auto const& grid : grids) {
      auto& cell = make_cell("PERSISTENT", "FULL_OUTPUT");
      cell.policy = grid.capacity_b_mask && grid.balanced_b_mask ?
          "capacity+balanced" : grid.capacity_b_mask ? "capacity" : "balanced";
      cell.grid = grid.grid; cell.occupancy = occupancy;
      cell.capacity_b_mask = grid.capacity_b_mask;
      cell.balanced_b_mask = grid.balanced_b_mask;
      if constexpr (PersistentKernel::SharedStorageSize > ppu_tactics::kBlockSmemBytes) {
        cell.state = State::PersistentSharedStorage;
      } else {
        using PStrideC = typename PersistentKernel::StrideC;
        using PStrideD = typename PersistentKernel::StrideD;
        PStrideC pC = cutlass::make_cute_packed_stride(
            PStrideC{}, cute::make_shape(in.m, in.n, 1));
        PStrideD pD = cutlass::make_cute_packed_stride(
            PStrideD{}, cute::make_shape(in.m, in.n, 1));
        typename PersistentGemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {in.m, in.n, in.k, 1}, mainloop,
            {{1.f, 0.f}, static_cast<half_t*>(nullptr), pC, in.output, pD},
            cutlass::KernelHardwareInfo{in.device, in.cu}, {}, occupancy,
            std::uint32_t(grid.grid)};
        PersistentGemm gemm;
        if (PersistentGemm::can_implement(args) != cutlass::Status::kSuccess) {
          cell.state = State::CanImplement;
        } else if (gemm.initialize(args, nullptr, nullptr) != cutlass::Status::kSuccess) {
          cell.state = State::Initialize;
          return false;
        } else if (!validate_and_measure(
                       in, options, [&] { return gemm.run(nullptr); }, cell)) {
          return false;
        }
      }
    }
  }

  // Fixed Split-K producer-only board.
  for (int splits : std::array<int, 3>{{2, 4, 8}}) {
    auto& cell = make_cell(splits == 2 ? "SPLITK_S2_PRODUCER" :
                           splits == 4 ? "SPLITK_S4_PRODUCER" :
                                         "SPLITK_S8_PRODUCER",
                           "PRODUCER_ONLY_NOT_PRODUCT_E2E");
    cell.policy = "fixed-split-k"; cell.split = splits;
    int const k_tiles = in.k / TK;
    auto const partition = cutlass::gemm::kernel::fixed_splitk::make_params(
        ((in.m + TM - 1) / TM) * ((in.n + TN - 1) / TN), k_tiles, splits);
    cell.grid = partition.is_valid() ? int(partition.work_units) : 0;
    if constexpr (SplitKernel::SharedStorageSize > ppu_tactics::kBlockSmemBytes) {
      cell.state = State::SplitSharedStorage;
      continue;
    }
    if (!partition.is_valid()) { cell.state = State::SplitPartition; continue; }
    if (int(partition.k_tiles_per_split) < Stages - 1) {
      cell.state = State::PipelineDepth; continue;
    }
    dense_splitk_parallel_ppu::WorkspacePlan plan;
    if (!dense_splitk_parallel_ppu::query_workspace_plan(
            in.m, in.n, splits, plan) ||
        plan.partial_bytes > in.workspace_bytes) {
      cell.state = State::SplitPartition; continue;
    }
    cell.partial_bytes = plan.partial_bytes;
    float* partials = reinterpret_cast<float*>(in.workspace);
    using PartialStride = typename SplitKernel::StrideD;
    PartialStride sP = cutlass::make_cute_packed_stride(
        PartialStride{}, cute::make_shape(in.m, in.n, splits));
    typename SplitGemm::Arguments producer_args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {in.m, in.n, in.k, 1}, mainloop, {partials, sP, partials, sP}, splits};
    typename Reduction::Arguments reducer_args{
        in.m, in.n, splits, partials, in.workspace_bytes, in.output, in.n};
    SplitGemm producer;
    Reduction reducer;
    if (SplitGemm::can_implement(producer_args) != cutlass::Status::kSuccess ||
        Reduction::can_implement(reducer_args) != cutlass::Status::kSuccess) {
      cell.state = State::CanImplement;
      continue;
    }
    if (producer.initialize(producer_args, nullptr, nullptr) != cutlass::Status::kSuccess ||
        reducer.initialize(reducer_args) != cutlass::Status::kSuccess) {
      cell.state = State::Initialize;
      return false;
    }
    auto full = [&] {
      auto status = producer.run(nullptr);
      return status == cutlass::Status::kSuccess ? reducer.run(nullptr) : status;
    };
    std::uint64_t first = 0;
    for (int repeat = 0; repeat < options.correctness_repeats; ++repeat) {
      if (full() != cutlass::Status::kSuccess ||
          hggcDeviceSynchronize() != hggcSuccess || !inspect(in, cell) ||
          cell.raw_bad || (repeat && cell.fingerprint != first)) {
        cell.state = State::Correctness;
        return false;
      }
      first = cell.fingerprint;
    }
    cell.reducer_correctness_untimed = true;
    if (options.measure &&
        !measure([&] { return producer.run(nullptr); }, options.iterations, cell)) {
      cell.state = State::Timing;
      return false;
    }
    if (reducer.run(nullptr) != cutlass::Status::kSuccess ||
        hggcDeviceSynchronize() != hggcSuccess || !inspect(in, cell) ||
        cell.raw_bad || cell.fingerprint != first) {
      cell.state = State::Correctness;
      return false;
    }
    cell.state = State::Measured;
  }
  return true;
}

}  // namespace scalefirst_internal_sweep
