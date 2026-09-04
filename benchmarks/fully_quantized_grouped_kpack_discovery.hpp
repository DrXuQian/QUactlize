#pragma once
// Discovery-only grouped FullyQuantized harness for canonical K-pack.
// It owns no public ABI and does not participate in product dispatch.

#include <algorithm>
#include <array>
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

namespace fully_quantized_grouped_kpack {

using half_t = cutlass::half_t;
using QuantMode = ppu_mixed_policy::QuantMode;

enum class State : int {
  Measured, InvalidInputs, SharedStorage, Occupancy, Launch, Synchronize,
  RawMismatch, Timing,
};

inline char const* state_name(State state) {
  switch (state) {
    case State::Measured: return "MEASURED";
    case State::InvalidInputs: return "INVALID_INPUTS";
    case State::SharedStorage: return "INADMISSIBLE_SHARED_STORAGE";
    case State::Occupancy: return "INADMISSIBLE_OCCUPANCY";
    case State::Launch: return "LAUNCH_OR_CAN_IMPLEMENT_FAIL";
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
  std::uint8_t const* units = nullptr;
  half_t* output = nullptr;
  half_t const* golden = nullptr;  // host [total_rows,N]
  half_t** output_ptrs = nullptr;  // device [experts]
  moe_grouped_ppu::DStride* output_strides = nullptr;
  int const* group_rows = nullptr;
  moe_grouped_ppu::GroupShape* shapes = nullptr;
  moe_grouped_ppu::GroupShape const* host_shapes = nullptr;
  int const* row_offsets = nullptr;
  char* workspace = nullptr;
  std::size_t workspace_bytes = 0;
  int total_rows = 0, max_rows = 0, n = 0, k = 0, experts = 0;
  int group_size = 0, active = 0, empty = 0;
  int device = 0, cu = 0;
  hggcStream_t stream = nullptr;
};

struct CellResult {
  struct BadCoordinate {
    int expert = -1;
    int local_m = -1;
    int n = -1;
    std::uint16_t want = 0;
    std::uint16_t got = 0;
  };

  char const* algorithm = nullptr;
  char const* policy = nullptr;
  int grid = 0, occupancy = 0;
  std::uint64_t capacity_b_mask = 0, balanced_b_mask = 0;
  State state = State::InvalidInputs;
  std::uint64_t raw_bad = 0;
  std::size_t first_bad = std::size_t(-1);
  std::uint16_t first_want = 0, first_got = 0;
  int first_bad_expert = -1, first_bad_local_m = -1, first_bad_n = -1;
  std::uint64_t bad_first_m_tile = 0, bad_later_m_tiles = 0;
  std::uint64_t bad_got_zero = 0, bad_got_poison = 0;
  std::array<std::uint64_t, 16> bad_by_local_m_mod16{};
  std::array<std::uint64_t, 4> bad_by_n_mod64_n16{};
  std::array<BadCoordinate, 8> bad_coordinates{};
  int bad_coordinate_count = 0;
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
  bool persistent;
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
                    std::is_void_v<High>);
  using Type = ppu_mixed_policy::Q4KPack4MainloopPolicy<
      QuantMode::FinegrainedScaleZero, Schedule, Tile, ScaleTile, Warp,
      Stages, true, 0, DeliveryN, cutlass::gemm::SeparateHalfPlanes>;
};

template <class Schedule, class Tile, class ScaleTile, class Warp,
          int Stages, class Low, class High, int DeliveryN>
struct PolicySelector<2, Schedule, Tile, ScaleTile, Warp, Stages, Low, High, DeliveryN> {
  using Type = ppu_mixed_policy::KPackMainloopPolicy<
      QuantMode::FinegrainedScaleZero, Schedule, Tile, ScaleTile, Warp,
      Stages, true, Low, High, 0, DeliveryN>;
};

template <int QType, int WeightLayout, int TM, int TN, int TK,
          int WM, int WN, int Stages, int DeliveryN, bool Persistent>
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
  using NonPersistentKernel = cutlass::gemm::kernel::GemmUniversal<
      moe_grouped_ppu::GroupProblemShape, Mainloop, Epilogue>;
  using PersistentKernel = cutlass::gemm::kernel::GroupPersistentMixedInputKernel<
      moe_grouped_ppu::GroupProblemShape, Mainloop, Epilogue>;
  using Kernel = std::conditional_t<Persistent, PersistentKernel,
                                    NonPersistentKernel>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  static_assert(Descriptor::quant_mode == QuantMode::FinegrainedScaleZero &&
                    Descriptor::kpack_transpose &&
                    Descriptor::artifact_tile_k == 0 &&
                    Descriptor::artifact_low_fold == 1 &&
                    Descriptor::artifact_high_fold == 1 &&
                    Descriptor::packed_metadata &&
                    !Descriptor::interleaved_metadata,
                "grouped FQ K-pack requires packed scale+zero units");
  static_assert(Descriptor::transport_tile_k ==
                    (WeightLayout == 1
                         ? q4_kpack4::kTransportK
                         : kquant_kpack::transport_k(
                               Bits<Low>::value, Bits<High>::value)));
  static_assert(WeightLayout == 1
                    ? (Descriptor::kpack4_scheduled_delivery_n == DeliveryN &&
                       Descriptor::kpack4_resolved_delivery_n == DeliveryN)
                    : (Descriptor::kpack_scheduled_delivery_n == DeliveryN &&
                       Descriptor::kpack_resolved_delivery_n == DeliveryN),
                "scheduled and resolved delivery must equal candidate identity");
  static_assert(std::is_same_v<typename Kernel::CollectiveMainloop, Mainloop>);
  static_assert(ppu_mixed_policy::kernel_policy_valid_v<
                    ppu_tactics::GroupedSpace, Policy>);
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

inline bool inspect(Inputs const& in, int tile_m, CellResult& result) {
  std::vector<half_t> got(std::size_t(in.total_rows) * in.n);
  if (hggcMemcpy(got.data(), in.output, got.size() * sizeof(half_t),
                 hggcMemcpyDeviceToHost) != hggcSuccess)
    return false;
  result.raw_bad = 0;
  result.first_bad = std::size_t(-1);
  int global_row = 0;
  for (int expert = 0; expert < in.experts; ++expert) {
    int const rows = int(cute::get<0>(in.host_shapes[expert]));
    for (int local_m = 0; local_m < rows; ++local_m, ++global_row) {
      for (int n = 0; n < in.n; ++n) {
        std::size_t const index = std::size_t(global_row) * in.n + n;
        std::uint16_t const want = in.golden[index].raw();
        std::uint16_t const actual = got[index].raw();
        if (actual == want) continue;
        if (result.raw_bad == 0) {
          result.first_bad = index;
          result.first_want = want;
          result.first_got = actual;
          result.first_bad_expert = expert;
          result.first_bad_local_m = local_m;
          result.first_bad_n = n;
        }
        if (local_m < tile_m) ++result.bad_first_m_tile;
        else ++result.bad_later_m_tiles;
        result.bad_got_zero += actual == 0;
        result.bad_got_poison += actual == UINT16_C(0x7b7b);
        ++result.bad_by_local_m_mod16[std::size_t(local_m & 15)];
        ++result.bad_by_n_mod64_n16[std::size_t((n & 63) >> 4)];
        if (result.bad_coordinate_count < int(result.bad_coordinates.size())) {
          result.bad_coordinates[std::size_t(result.bad_coordinate_count++)] =
              {expert, local_m, n, want, actual};
        }
        ++result.raw_bad;
      }
    }
  }
  return global_row == in.total_rows;
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
        true, false, false, 0, Persistent, typename T::Policy, 0>(
            in.a, reinterpret_cast<Low const*>(in.low),
            reinterpret_cast<half_t const*>(in.units), nullptr,
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
    if (hggcDeviceSynchronize() != hggcSuccess ||
        !inspect(in, int(cute::size<0>(typename T::Tile{})), result)) {
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
        0.5 * (result.samples_us[count / 2 - 1] +
               result.samples_us[count / 2]);
  }
  result.failure_repeat = -1;
  result.state = State::Measured;
}

template <int QType, int WeightLayout, int TM, int TN, int TK,
          int WM, int WN, int Stages, int DeliveryN, bool Persistent>
bool run_row(Inputs const& in, Options const& options, Result& result) {
  using T = RowTypes<QType, WeightLayout, TM, TN, TK, WM, WN, Stages,
                     DeliveryN, Persistent>;
  using High = typename T::High;
  result = Result{};
  if (!in.a || !in.low || !in.units || !in.output || !in.golden ||
      !in.output_ptrs || !in.output_strides || !in.group_rows || !in.shapes ||
      !in.host_shapes || !in.row_offsets || !in.workspace ||
      in.workspace_bytes == 0 || in.total_rows <= 0 || in.max_rows <= 0 ||
      in.n <= 0 || in.k <= 0 || in.experts <= 1 ||
      in.group_size != T::F::GroupSize || in.active <= 0 ||
      in.active > in.experts || in.empty < 0 ||
      in.empty != in.experts - in.active ||
      (Persistent && in.cu <= 0) ||
      (!std::is_void_v<High> && !in.high)) {
    CellResult invalid;
    invalid.algorithm = Persistent ? "GROUPED_PERSISTENT" :
                                     "GROUPED_NONPERSISTENT";
    invalid.policy = Persistent ? "capacity+balanced" : "ordinary";
    invalid.state = State::InvalidInputs;
    result.cells.push_back(std::move(invalid));
    return true;
  }
  if constexpr (!ppu_tactics::fits_block_smem(
                    T::Kernel::SharedStorageSize)) {
    CellResult cell;
    cell.algorithm = Persistent ? "GROUPED_PERSISTENT" :
                                  "GROUPED_NONPERSISTENT";
    cell.policy = Persistent ? "capacity+balanced" : "ordinary";
    cell.state = State::SharedStorage;
    result.cells.push_back(std::move(cell));
    return true;
  }
  if constexpr (!Persistent) {
    result.cells.emplace_back();
    execute_cell<false, T>(in, options, {}, 0, result.cells.back());
    return true;
  } else {
    int const occupancy = T::Gemm::maximum_active_blocks();
    std::uint64_t logical_work = 0;
    for (int expert = 0; expert < in.experts; ++expert)
      logical_work += std::uint64_t(cute::ceil_div(
          int(cute::get<0>(in.host_shapes[expert])), TM));
    logical_work *= std::uint64_t(cute::ceil_div(in.n, TN));
    auto const grids = quactlize::scalefirst_policy::grid_space(
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
  }
  return true;
}

}  // namespace fully_quantized_grouped_kpack
