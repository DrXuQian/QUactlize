#pragma once
// Discovery-only grouped ScaleFirst harness for the canonical K-pack maps.
// It owns no public C ABI and does not participate in product dispatch.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "cutlass/cutlass.h"
#include "moe_grouped_ppu.cuh"
#include "actlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_mixed_input_2plane.hpp"
#include "ppu_group_schedule.hpp"
#include "ppu_mixed_policy.hpp"
#include "scalefirst_persistent_policy.hpp"

namespace scalefirst_grouped_kpack {

using half_t = cutlass::half_t;
using QuantMode = ppu_mixed_policy::QuantMode;

enum class State : int {
  Measured,
  InvalidInputs,
  SharedStorage,
  Occupancy,
  Launch,
  Synchronize,
  RawMismatch,
  Timing,
};

inline char const* state_name(State state) {
  switch (state) {
    case State::Measured: return "MEASURED";
    case State::InvalidInputs: return "INVALID_INPUTS";
    case State::SharedStorage: return "INADMISSIBLE_SHARED_STORAGE";
    case State::Occupancy: return "INADMISSIBLE_OCCUPANCY";
    case State::Launch: return "LAUNCH_FAIL";
    case State::Synchronize: return "SYNCHRONIZE_FAIL";
    case State::RawMismatch: return "RAW_FP16_MISMATCH";
    case State::Timing: return "TIMING_FAIL";
  }
  return "UNKNOWN";
}

struct Options {
  int correctness_repeats = 2;
  int warmups = 2;
  int iterations = 5;
  bool measure = true;
};

struct Inputs {
  half_t const* a = nullptr;
  std::uint8_t const* low = nullptr;
  std::uint8_t const* high = nullptr;
  half_t const* scale = nullptr;
  half_t const* zero = nullptr;
  half_t* output = nullptr;
  half_t const* golden = nullptr;  // host [total_rows,N]
  half_t** output_ptrs = nullptr;  // device [experts]
  moe_grouped_ppu::DStride* output_strides = nullptr;  // device [experts]
  int const* group_rows = nullptr;      // device [experts]
  moe_grouped_ppu::GroupShape* shapes = nullptr;  // device [experts]
  moe_grouped_ppu::GroupShape const* host_shapes = nullptr;
  int const* row_offsets = nullptr;     // device [experts+1]
  char* workspace = nullptr;
  std::size_t workspace_bytes = 0;
  int total_rows = 0, max_rows = 0, n = 0, k = 0, experts = 0;
  int group_size = 0, active = 0, empty = 0;
  int cu = 0;
  hggcStream_t stream = nullptr;
};

struct CellResult {
  char const* algorithm = nullptr;
  char const* policy = nullptr;
  int grid = 0, occupancy = 0;
  std::uint64_t capacity_b_mask = 0, balanced_b_mask = 0;
  State state = State::InvalidInputs;
  std::uint64_t raw_bad = 0;
  std::size_t first_bad = std::size_t(-1);
  std::uint16_t first_want = 0, first_got = 0;
  int failure_repeat = -1;
  double median_us = 0, min_us = 0, max_us = 0;
  std::vector<double> samples_us;
};

struct Result { std::vector<CellResult> cells; };

using RunRow = bool (*)(Inputs const&, Options const&, Result&);
struct RegistryRow {
  char const* symbol;
  int qtype, weight_layout, tm, tn, tk, wm, wn, stages;
  int resolved_delivery_n;
  RunRow run;
};

template <int QType> struct Format;
template <> struct Format<10> {
  using Low = cutlass::uint2b_t; using High = void;
  static constexpr int GroupSize = 16, Layout = 2;
};
template <> struct Format<11> {
  using Low = cutlass::uint2b_t; using High = cutlass::uint1b_t;
  static constexpr int GroupSize = 16, Layout = 2;
};
template <> struct Format<12> {
  using Low = cutlass::int4b_t; using High = void;
  static constexpr int GroupSize = 32, Layout = 1;
};
template <> struct Format<13> {
  using Low = cutlass::int4b_t; using High = cutlass::uint1b_t;
  static constexpr int GroupSize = 32, Layout = 2;
};
template <> struct Format<14> {
  using Low = cutlass::int4b_t; using High = cutlass::uint2b_t;
  static constexpr int GroupSize = 16, Layout = 2;
};

template <class T> struct Bits {
  static constexpr int value = cutlass::sizeof_bits<T>::value;
};
template <> struct Bits<void> { static constexpr int value = 0; };

template <int Layout, class Schedule, class Tile, class ScaleTile,
          class Warp, int Stages, class Low, class High, int DeliveryN>
struct PolicySelector;

template <class Schedule, class Tile, class ScaleTile, class Warp,
          int Stages, class Low, class High, int DeliveryN>
struct PolicySelector<1, Schedule, Tile, ScaleTile, Warp, Stages, Low, High, DeliveryN> {
  static_assert(std::is_same_v<Low, cutlass::int4b_t> &&
                    std::is_void_v<High>,
                "layout1 is the Q4 one-plane K-pack reader");
  using Type = ppu_mixed_policy::Q4KPack4MainloopPolicy<
      QuantMode::FinegrainedScaleZero, Schedule, Tile, ScaleTile, Warp,
      Stages, true, 0, DeliveryN>;
};

template <class Schedule, class Tile, class ScaleTile, class Warp,
          int Stages, class Low, class High, int DeliveryN>
struct PolicySelector<2, Schedule, Tile, ScaleTile, Warp, Stages, Low, High, DeliveryN> {
  using Type = ppu_mixed_policy::KPackMainloopPolicy<
      QuantMode::FinegrainedScaleZero, Schedule, Tile, ScaleTile, Warp,
      Stages, true, Low, High, 0, DeliveryN>;
};

template <int QType, int WeightLayout, int TM, int TN, int TK,
          int WM, int WN, int Stages, int DeliveryN>
struct RowTypes {
  static constexpr int kStages = Stages;
  using F = Format<QType>;
  using Low = typename F::Low;
  using High = typename F::High;
  using Schedule = ppu_group_schedule::FinegrainedSchedule<F::GroupSize>;
  using Tile = cute::Shape<cute::C<TM>, cute::C<TN>, cute::C<TK>>;
  using ScaleTile = cute::Shape<
      cute::C<TN>,
      cute::C<ppu_group_schedule::scale_groups_v<TK, F::GroupSize>>>;
  using Warp = cute::Shape<cute::C<WM>, cute::C<WN>, cute::C<TK>>;
  static_assert(WeightLayout == F::Layout,
                "generated qtype/layout pair is not canonical");
  static_assert((DeliveryN == 16 || DeliveryN == 32 || DeliveryN == 64) &&
                DeliveryN <= TN && TN % DeliveryN == 0);
  using Policy = typename PolicySelector<
      WeightLayout, Schedule, Tile, ScaleTile, Warp, Stages, Low, High, DeliveryN>::Type;
  using Mainloop = typename Policy::CollectiveOp;
  using Descriptor = typename Policy::Descriptor;
  using GroupShape = moe_grouped_ppu::GroupShape;
  using GroupProblemShape = moe_grouped_ppu::GroupProblemShape;
  using ElementAccumulator = float;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
      Tile, Warp, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      half_t, cutlass::layout::RowMajor*, 8,
      half_t, cutlass::layout::RowMajor*, 8,
      cutlass::epilogue::EpiloguePtrArraySimtVectorized,
      cutlass::epilogue::fusion::LinearCombination<
          half_t, ElementAccumulator>>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      GroupProblemShape, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using PersistentKernel = cutlass::gemm::kernel::GroupPersistentMixedInputKernel<
      GroupProblemShape, Mainloop, Epilogue>;
  using PersistentGemm =
      cutlass::gemm::device::GemmUniversalAdapter<PersistentKernel>;
  static_assert(Descriptor::quant_mode == QuantMode::FinegrainedScaleZero &&
                    Descriptor::kpack_transpose &&
                    Descriptor::artifact_tile_k == 0 &&
                    Descriptor::artifact_low_fold == 1 &&
                    Descriptor::artifact_high_fold == 1 &&
                    !Descriptor::packed_metadata,
                "grouped K-pack discovery must consume resident scale+zero planes");
  static_assert(Descriptor::transport_tile_k ==
                    (WeightLayout == 1
                         ? q4_kpack4::kTransportK
                         : kquant_kpack::transport_k(
                               Bits<Low>::value, Bits<High>::value)),
                "grouped K-pack transport must match its offline byte map");
  static_assert(WeightLayout == 1
                    ? (Descriptor::kpack4_scheduled_delivery_n == DeliveryN &&
                       Descriptor::kpack4_resolved_delivery_n == DeliveryN)
                    : (Descriptor::kpack_scheduled_delivery_n == DeliveryN &&
                       Descriptor::kpack_resolved_delivery_n == DeliveryN),
                "scheduled and resolved delivery must equal candidate identity");
  static_assert(cute::size<0>(typename Epilogue::SmemLayout{}) ==
                    cute::size<0>(typename Mainloop::TiledMma::AtomShape_MNK{}) *
                        cute::size<1>(typename Mainloop::TiledMma::ThrLayoutVMNK{}),
                "grouped K-pack epilogue ownership must match its mainloop");
  static_assert(ppu_mixed_policy::kernel_policy_valid_v<
                    ppu_tactics::GroupedSpace, Policy>);
  static_assert(PersistentKernel::SharedStorageSize == Kernel::SharedStorageSize,
                "grouped NP/P must differ only in their scheduler");
};

class EventPair {
 public:
  hggcEvent_t start{}, stop{};
  EventPair() {
    if (hggcEventCreate(&start) != hggcSuccess) start = nullptr;
    if (hggcEventCreate(&stop) != hggcSuccess) stop = nullptr;
  }
  ~EventPair() {
    if (start) hggcEventDestroy(start);
    if (stop) hggcEventDestroy(stop);
  }
  bool valid() const { return start != nullptr && stop != nullptr; }
};

inline bool inspect(Inputs const& in, CellResult& result) {
  std::vector<half_t> got(std::size_t(in.total_rows) * in.n);
  if (hggcMemcpy(got.data(), in.output, got.size() * sizeof(half_t),
                 hggcMemcpyDeviceToHost) != hggcSuccess)
    return false;
  result.raw_bad = 0;
  result.first_bad = std::size_t(-1);
  for (std::size_t index = 0; index < got.size(); ++index) {
    if (got[index].raw() != in.golden[index].raw()) {
      if (result.raw_bad == 0) {
        result.first_bad = index;
        result.first_want = in.golden[index].raw();
        result.first_got = got[index].raw();
      }
      ++result.raw_bad;
    }
  }
  return true;
}

template <bool Persistent, class T>
void execute_cell(Inputs const& in, Options const& options,
                  quactlize::scalefirst_policy::GridChoice grid,
                  int occupancy, CellResult& result) {
  using Low = typename T::Low;
  using High = typename T::High;
  result.algorithm = Persistent ? "GROUPED_PERSISTENT" :
                                  "GROUPED_NONPERSISTENT";
  result.policy = Persistent ?
      (grid.capacity_b_mask && grid.balanced_b_mask ? "capacity+balanced" :
       grid.capacity_b_mask ? "capacity" : "balanced") : "ordinary";
  result.grid = Persistent ? grid.grid : 0;
  result.occupancy = Persistent ? occupancy : 0;
  result.capacity_b_mask = Persistent ? grid.capacity_b_mask : 0;
  result.balanced_b_mask = Persistent ? grid.balanced_b_mask : 0;

  auto launch = [&](moe_grouped_ppu::KernelSpanEvents* events) {
    int const fail_before = moe_grouped_ppu::moeg_fail_count();
    bool const ok = moe_grouped_ppu::launch<
        QuantMode::FinegrainedScaleZero, typename T::Schedule,
        typename T::Tile, typename T::ScaleTile, typename T::Warp,
        T::kStages, true, Low, High,
        false, false, false, 0, Persistent, typename T::Policy, 0>(
            in.a, reinterpret_cast<Low const*>(in.low), in.scale, in.zero,
            in.output_ptrs, in.output_strides, in.group_rows,
            in.max_rows, in.n, in.k, in.experts, in.group_size,
            in.shapes, in.host_shapes, in.row_offsets,
            in.workspace, in.workspace_bytes, in.stream,
            reinterpret_cast<High const*>(in.high),
            -1, false, 1, false, events,
            Persistent ? std::uint32_t(grid.grid) : 0u);
    return ok && moe_grouped_ppu::moeg_fail_count() == fail_before;
  };

  for (int repeat = 0; repeat < options.correctness_repeats; ++repeat) {
    result.failure_repeat = repeat;
    if (hggcMemset(in.output, 0x7b,
                   std::size_t(in.total_rows) * in.n * sizeof(half_t)) !=
            hggcSuccess || !launch(nullptr)) {
      result.state = State::Launch;
      return;
    }
    if (hggcDeviceSynchronize() != hggcSuccess) {
      result.state = State::Synchronize;
      return;
    }
    if (!inspect(in, result)) {
      result.state = State::Synchronize;
      return;
    }
    if (result.raw_bad) {
      result.state = State::RawMismatch;
      return;
    }
  }

  if (options.measure) {
    for (int warmup = 0; warmup < options.warmups; ++warmup)
      if (!launch(nullptr)) { result.state = State::Launch; return; }
    if (hggcDeviceSynchronize() != hggcSuccess) {
      result.state = State::Synchronize;
      return;
    }
    result.samples_us.clear();
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      EventPair pair;
      if (!pair.valid()) { result.state = State::Timing; return; }
      moe_grouped_ppu::KernelSpanEvents events{pair.start, pair.stop, false};
      if (!launch(&events) || !events.recorded ||
          hggcEventSynchronize(pair.stop) != hggcSuccess) {
        result.state = State::Timing;
        return;
      }
      float ms = 0;
      if (hggcEventElapsedTime(&ms, pair.start, pair.stop) != hggcSuccess ||
          !(ms > 0)) {
        result.state = State::Timing;
        return;
      }
      result.samples_us.push_back(double(ms) * 1000.0);
    }
    std::sort(result.samples_us.begin(), result.samples_us.end());
    result.min_us = result.samples_us.front();
    result.max_us = result.samples_us.back();
    std::size_t const count = result.samples_us.size();
    result.median_us = count & 1 ? result.samples_us[count / 2] :
        0.5 * (result.samples_us[count / 2 - 1] + result.samples_us[count / 2]);
  }
  result.failure_repeat = -1;
  result.state = State::Measured;
}

template <int QType, int WeightLayout, int TM, int TN, int TK,
          int WM, int WN, int Stages, int DeliveryN>
bool run_row(Inputs const& in, Options const& options, Result& result) {
  using T = RowTypes<QType, WeightLayout, TM, TN, TK, WM, WN, Stages, DeliveryN>;
  using High = typename T::High;
  result = Result{};
  if (!in.a || !in.low || !in.scale || !in.zero || !in.output ||
      !in.golden || !in.output_ptrs || !in.output_strides || !in.group_rows ||
      !in.shapes || !in.host_shapes || !in.row_offsets || !in.workspace ||
      in.workspace_bytes == 0 || in.total_rows <= 0 || in.max_rows <= 0 ||
      in.n <= 0 || in.k <= 0 || in.experts <= 1 || in.group_size != T::F::GroupSize ||
      in.active <= 0 || in.active > in.experts || in.empty < 0 ||
      in.empty != in.experts - in.active || in.cu <= 0 ||
      (!std::is_void_v<High> && !in.high)) {
    CellResult invalid;
    invalid.algorithm = "GROUPED_NONPERSISTENT";
    invalid.policy = "ordinary";
    invalid.state = State::InvalidInputs;
    result.cells.push_back(std::move(invalid));
    return true;
  }
  if constexpr (T::Kernel::SharedStorageSize > ppu_tactics::kBlockSmemBytes) {
    auto append_unavailable = [&](char const* algorithm, char const* policy) {
      CellResult cell;
      cell.algorithm = algorithm;
      cell.policy = policy;
      cell.state = State::SharedStorage;
      result.cells.push_back(std::move(cell));
    };
    append_unavailable("GROUPED_NONPERSISTENT", "ordinary");
    append_unavailable("GROUPED_PERSISTENT", "capacity+balanced");
    return true;
  }

  result.cells.emplace_back();
  execute_cell<false, T>(in, options, {}, 0, result.cells.back());
  if (result.cells.back().state != State::Measured) return true;

  int const occupancy = T::PersistentGemm::maximum_active_blocks();
  if (occupancy < 0) {
    CellResult cell;
    cell.algorithm = "GROUPED_PERSISTENT";
    cell.policy = "capacity+balanced";
    cell.occupancy = occupancy;
    cell.state = State::Launch;
    result.cells.push_back(std::move(cell));
    return true;
  }
  std::uint64_t logical_work = 0;
  for (int expert = 0; expert < in.experts; ++expert) {
    logical_work += std::uint64_t(cute::ceil_div(
        int(cute::get<0>(in.host_shapes[expert])), TM));
  }
  logical_work *= std::uint64_t(cute::ceil_div(in.n, TN));
  auto grids = quactlize::scalefirst_policy::grid_space(
      logical_work, in.cu, occupancy);
  if (grids.empty()) {
    CellResult cell;
    cell.algorithm = "GROUPED_PERSISTENT";
    cell.policy = "capacity+balanced";
    cell.occupancy = occupancy;
    cell.state = State::Occupancy;
    result.cells.push_back(std::move(cell));
    return true;
  }
  for (auto const& grid : grids) {
    result.cells.emplace_back();
    execute_cell<true, T>(in, options, grid, occupancy, result.cells.back());
    if (result.cells.back().state != State::Measured) return true;
  }
  return true;
}

}  // namespace scalefirst_grouped_kpack
