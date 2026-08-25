#pragma once
// Exact runtime ABI for generated FullyQuantized tensor-core sweep shards.
//
// S==1 measures the complete shipping epilogue.  S>1 measures only the
// producer kernel.  Every timed producer is followed by the existing
// deterministic reducer, submitted outside the producer event span, so its
// partial workspace is consumed before reuse.  Consequently a producer-only
// row can never be ranked as a product result against S1 or BC GEMV.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <type_traits>
#include <vector>

#include "cutlass/cutlass.h"
#include "dense_splitk_multiformat_ppu.cuh"
#include "helper.h"
#include "ppu_dense_shipping_policy.hpp"
#include "ppu_group_schedule.hpp"
#include "splitk_producer_timing.hpp"

namespace fq_internal_sweep {

using half_t = cutlass::half_t;
using QuantMode = fpa_intb_ppu::QuantMode;

inline constexpr std::array<int, 4> kSplits{{1, 2, 4, 8}};

enum class State : int {
  Measured,
  ShippingSharedStorage,
  SplitSharedStorage,
  SplitPartition,
  PipelineDepth,
  M8DecodeOnly,
  PackedADecodeOnly,
  CanImplement,
  Initialize,
  Launch,
  Correctness,
  PartialCorrectness,
  Timing,
};

inline char const* state_name(State state) {
  switch (state) {
    case State::Measured: return "MEASURED";
    case State::ShippingSharedStorage: return "SHIPPING_SHARED_STORAGE";
    case State::SplitSharedStorage: return "SPLIT_SHARED_STORAGE";
    case State::SplitPartition: return "SPLIT_PARTITION";
    case State::PipelineDepth: return "INADMISSIBLE_PIPELINE_DEPTH";
    case State::M8DecodeOnly: return "M8_DECODE_ONLY_M_GE_8";
    case State::PackedADecodeOnly: return "PACKED_A_DECODE_ONLY_M_NOT_1";
    case State::CanImplement: return "REAL_CAN_IMPLEMENT";
    case State::Initialize: return "INITIALIZE";
    case State::Launch: return "LAUNCH";
    case State::Correctness: return "RAW_FP16_MISMATCH";
    case State::PartialCorrectness: return "PARTIAL_FP32_MISMATCH";
    case State::Timing: return "TIMING";
  }
  return "UNKNOWN";
}

// Host-only controls for the repeated correctness harness. They do not alter
// a generated kernel type or its arguments. TargetPoison overwrites the real
// FP32 partial workspace before each producer launch. ControlPoison performs
// the same memset/synchronize cadence on an equally sized disjoint tail while
// leaving the producer workspace untouched. Their A/B comparison separates
// buffer carry-over from a cadence-sensitive device failure.
enum class RepeatState : int {
  Reuse,
  TargetPoison,
  ControlPoison,
};

inline char const* repeat_state_name(RepeatState state) {
  switch (state) {
    case RepeatState::Reuse: return "reuse";
    case RepeatState::TargetPoison: return "target-poison";
    case RepeatState::ControlPoison: return "control-poison";
  }
  return "unknown";
}

struct Options {
  int iterations = 7;
  int correctness_repeats = 2;
  int only_split = 0;
  bool measure = true;
  int tm8_max_m = ppu_dense_shipping::kDecodeDefaultExclusiveM - 1;
  // Diagnostic-only host routing control.  Production and sweep callers leave
  // this false, so S=1 continues to use ShippingGemm.  When true, runtime S=1
  // is submitted through the exact same SplitGemm type as S=2/4/8, with one
  // FP32 partial plane followed by the deterministic fallback reducer.  This
  // isolates runtime partitioning from kernel/epilogue type differences.
  bool force_custom_splitk_s1 = false;
  RepeatState repeat_state = RepeatState::Reuse;
  // Inspect every completed producer plane, including rounds whose reduced
  // FP16 output is exact. Without this, wrong planes that cancel in reduction
  // can silently seed the next reuse round.
  bool check_partials_each_repeat = false;
};

struct DeviceInputs {
  half_t const* a = nullptr;
  uint8_t const* low = nullptr;
  uint8_t const* high = nullptr;
  uint8_t const* units = nullptr;
  half_t* output = nullptr;
  char* workspace = nullptr;
  std::size_t workspace_bytes = 0;
  // Host-only exact FP32 partial planes used after a diagnostic failure.  The
  // producer never receives these addresses and the normal sweep leaves them
  // null, so this cannot perturb the pre-failure device cadence.
  std::array<float const*, kSplits.size()> partial_golden{};
  half_t const* golden = nullptr;  // host pointer, m*n values
  int m = 0, n = 0, k = 0;
};

struct CellResult {
  int split = 0;
  State state = State::CanImplement;
  bool full_output = false;
  bool reducer_correctness_untimed = false;
  std::uint64_t raw_bad = 0;
  std::uint64_t fingerprint = 0;
  std::size_t first_bad_index = std::size_t(-1);
  std::uint16_t first_bad_want = 0;
  std::uint16_t first_bad_got = 0;
  char const* failure_step = "NONE";
  int failure_repeat = -1;
  std::size_t shipping_smem = 0;
  std::size_t split_smem = 0;
  std::size_t partial_bytes = 0;
  bool partial_probe_attempted = false;
  bool partial_probe_complete = false;
  std::uint64_t partial_value_raw_bad = 0;
  std::uint32_t partial_bad_plane_mask = 0;
  int partial_first_bad_plane = -1;
  std::size_t partial_first_bad_index = std::size_t(-1);
  std::uint32_t partial_first_bad_want = 0;
  std::uint32_t partial_first_bad_got = 0;
  std::uint64_t reducer_replay_raw_bad = 0;
  std::size_t reducer_replay_first_bad = std::size_t(-1);
  std::uint16_t reducer_replay_first_want = 0;
  std::uint16_t reducer_replay_first_got = 0;
  int a_provider_capacity_rows = 0;
  double median_us = 0;
  double min_us = 0;
  double max_us = 0;
  std::vector<double> samples_us;
};

struct RowResult {
  std::array<CellResult, 4> cells{};
};

using RunRow = bool (*)(DeviceInputs const&, Options const&, RowResult&);

struct RegistryRow {
  char const* symbol;
  int qtype, artifact_tile_k;
  int tm, tn, tk, wm, wn, stages, bchunk, a_provider;
  RunRow run;
};

template <int QType> struct Format;
template <> struct Format<10> {
  using Low = cutlass::uint2b_t; using High = void;
  static constexpr int GroupSize = 16, TacticTileK = 256;
};
template <> struct Format<11> {
  using Low = cutlass::uint2b_t; using High = cutlass::uint1b_t;
  static constexpr int GroupSize = 16, TacticTileK = 256;
};
template <> struct Format<12> {
  using Low = cutlass::int4b_t; using High = void;
  static constexpr int GroupSize = 32, TacticTileK = 256;
};
template <> struct Format<13> {
  using Low = cutlass::int4b_t; using High = cutlass::uint1b_t;
  static constexpr int GroupSize = 32, TacticTileK = 256;
};
template <> struct Format<14> {
  using Low = cutlass::int4b_t; using High = cutlass::uint2b_t;
  static constexpr int GroupSize = 16, TacticTileK = 128;
};

template <class T> struct ElementBits {
  static constexpr int value = cutlass::sizeof_bits<T>::value;
};
template <> struct ElementBits<void> { static constexpr int value = 0; };

template <class Policy, class = void>
struct PackedAProviderCapacity : std::integral_constant<int, 0> {};
template <class Policy>
struct PackedAProviderCapacity<Policy, std::void_t<decltype(Policy::PackedARows)>>
    : std::integral_constant<int, Policy::PackedARows> {};

// One source of truth for generated type admission.  The normal benchmark
// wrapper and the local type-only compiler gate both instantiate this exact
// bundle; the latter deliberately stops before CUDA tries to lower PPU-only
// device expressions.
template <int QType, int ArtifactTileK, int TM, int TN, int TK,
          int WM, int WN, int Stages, int BChunk, int AProvider>
struct TcRowTypes {
  using F = Format<QType>;
  using Low = typename F::Low;
  using High = typename F::High;
  using Schedule = ppu_group_schedule::FinegrainedSchedule<F::GroupSize>;
  using Tile = cute::Shape<cute::C<TM>, cute::C<TN>, cute::C<TK>>;
  using ScaleTile = cute::Shape<cute::C<TN>, cute::C<
      ppu_group_schedule::scale_groups_v<TK, F::GroupSize>>>;
  using Warp = cute::Shape<cute::C<WM>, cute::C<WN>, cute::C<TK>>;
  using Ordinary = fpa_intb_ppu::DenseKernelTypes<
      QuantMode::FinegrainedScaleZero, Schedule, Tile, ScaleTile, Warp,
      Stages, true, Low, High, ArtifactTileK>;
  static constexpr ppu_tactics::FormatSpec spec{
      QType == 10 ? ppu_tactics::Format::I2 :
      QType == 11 ? ppu_tactics::Format::Q3_K :
      QType == 12 ? ppu_tactics::Format::I4 :
      QType == 13 ? ppu_tactics::Format::Q5_K :
                    ppu_tactics::Format::Q6_K,
      "fq", ElementBits<Low>::value, ElementBits<High>::value};
  static constexpr ppu_tactics::Candidate candidate{
      spec, TM, TN, TK, WM, WN, ArtifactTileK, BChunk};
  static_assert(AProvider == 0 || AProvider == 1,
                "generated A provider is ordinary(0) or packed-row(1)");
  static constexpr bool use_packed_a = AProvider == 1;
  static_assert(!use_packed_a || (TM == 8 && WM == 8 &&
      std::is_void_v<High> && ppu_tactics::artifact_low_fold(candidate) == 1),
      "packed-row A is the exact TM8/WM8 unfolded one-plane provider");
  using PackedA = fpa_intb_ppu::DensePackedAKernelTypes<
      1, QuantMode::FinegrainedScaleZero, Schedule, Tile, ScaleTile, Warp,
      Stages, true, Low, ArtifactTileK>;
  using Shipping = std::conditional_t<use_packed_a, PackedA, Ordinary>;
  using Split = dense_splitk_parallel_ppu::KernelTypes<Shipping, Tile, Warp>;
  using ShippingGemm = typename Shipping::Gemm;
  using ShippingKernel = typename Shipping::GemmKernel;
  using SplitGemm = typename Split::Gemm;
  using SplitKernel = typename Split::GemmKernel;
  using Reduction = typename Split::Reduction;
  using Mainloop = typename Shipping::CollectiveMainloop;
  static constexpr int a_provider_capacity_rows =
      PackedAProviderCapacity<typename Shipping::MainloopPolicy>::value;

  static_assert(PPU_B_CHUNK == BChunk,
                "generated unit must bind the requested BChunk policy");
  static_assert(dense_splitk_parallel_ppu::MainloopUsesPackedMetadata<
                    typename Shipping::CollectiveMainloop>::value,
                "FQ benchmark must instantiate the packed-unit collective");
  static_assert(Mainloop::packed_scale_copy_threads *
                    Mainloop::packed_scale_columns_per_thread == TN,
                "packed metadata owner slices must cover every TileN column");
  static_assert(!(TM == 8 && WM == 8 && TN == 64 && WN == 64) ||
                    (Mainloop::packed_scale_copy_threads == 32 &&
                     Mainloop::packed_scale_columns_per_thread == 2),
                "the exact TM8/WM8/WN64 family must use 32 two-column metadata owners");
  static_assert(std::is_same_v<typename SplitKernel::CollectiveMainloop,
                               typename Shipping::CollectiveMainloop>,
                "fixed Split-K must reuse the exact S1 collective");
  static_assert((use_packed_a && a_provider_capacity_rows == 1) ||
                (!use_packed_a && a_provider_capacity_rows == 0),
                "A-provider capacity must come from the selected shipping policy");
};

template <int QType, int ArtifactTileK, int TM, int TN, int TK,
          int WM, int WN, int Stages, int BChunk, int AProvider>
constexpr bool admit_tc_row_type() {
  using Types = TcRowTypes<QType, ArtifactTileK, TM, TN, TK,
                           WM, WN, Stages, BChunk, AProvider>;
  return Types::Shipping::SharedStorageSize > 0 &&
         Types::SplitKernel::SharedStorageSize > 0;
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
  EventPair(EventPair const&) = delete;
  EventPair& operator=(EventPair const&) = delete;
};

inline bool elapsed_us(hggcEvent_t start, hggcEvent_t stop, double& us) {
  float ms = 0;
  if (hggcEventElapsedTime(&ms, start, stop) != hggcSuccess ||
      !(ms > 0) || !std::isfinite(ms)) return false;
  us = double(ms) * 1000.0;
  return true;
}

inline std::uint64_t hash_half(half_t const* values, std::size_t count) {
  std::uint64_t hash = UINT64_C(1469598103934665603);
  for (std::size_t i = 0; i < count; ++i) {
    auto bits = values[i].raw();
    hash ^= std::uint8_t(bits); hash *= UINT64_C(1099511628211);
    hash ^= std::uint8_t(bits >> 8); hash *= UINT64_C(1099511628211);
  }
  return hash;
}

inline bool inspect(DeviceInputs const& in, CellResult& out) {
  std::size_t const count = std::size_t(in.m) * in.n;
  std::vector<half_t> host(count);
  if (hggcMemcpy(host.data(), in.output, count * sizeof(half_t),
                 hggcMemcpyDeviceToHost) != hggcSuccess) return false;
  out.raw_bad = 0;
  out.first_bad_index = std::size_t(-1);
  out.first_bad_want = out.first_bad_got = 0;
  for (std::size_t i = 0; i < count; ++i) {
    auto const got = host[i].raw();
    auto const want = in.golden[i].raw();
    if (got != want) {
      if (out.raw_bad == 0) {
        out.first_bad_index = i;
        out.first_bad_want = want;
        out.first_bad_got = got;
      }
      ++out.raw_bad;
    }
  }
  out.fingerprint = hash_half(host.data(), count);
  return true;
}

inline bool inspect_failed_partials(
    DeviceInputs const& in, int splits, float const* partials,
    std::size_t partial_bytes, CellResult& out) {
  auto const split_it = std::find(kSplits.begin(), kSplits.end(), splits);
  if (split_it == kSplits.end()) return false;
  std::size_t const slot = std::size_t(split_it - kSplits.begin());
  float const* expected = in.partial_golden[slot];
  std::size_t const elements = std::size_t(splits) * in.m * in.n;
  if (expected == nullptr || partials == nullptr ||
      partial_bytes != elements * sizeof(float)) return false;

  std::vector<float> host(elements);
  if (hggcMemcpy(host.data(), partials, partial_bytes,
                 hggcMemcpyDeviceToHost) != hggcSuccess) return false;
  out.partial_value_raw_bad = 0;
  out.partial_bad_plane_mask = 0;
  for (int plane = 0; plane < splits; ++plane) {
    for (std::size_t index = 0; index < std::size_t(in.m) * in.n; ++index) {
      std::size_t const offset = std::size_t(plane) * in.m * in.n + index;
      std::uint32_t want = 0, got = 0;
      std::memcpy(&want, expected + offset, sizeof(want));
      std::memcpy(&got, host.data() + offset, sizeof(got));
      // Partial planes are an internal semantic oracle.  Signed zero is
      // equivalent here; retain raw bits only to explain a real value error.
      if (expected[offset] == host[offset]) continue;
      if (out.partial_value_raw_bad == 0) {
        out.partial_first_bad_plane = plane;
        out.partial_first_bad_index = index;
        out.partial_first_bad_want = want;
        out.partial_first_bad_got = got;
      }
      out.partial_bad_plane_mask |= std::uint32_t(1) << plane;
      ++out.partial_value_raw_bad;
    }
  }
  return true;
}

inline bool prepare_repeat_state(
    DeviceInputs const& in, Options const& options, char* partials,
    std::size_t partial_bytes) {
  if (options.repeat_state == RepeatState::Reuse) return true;
  if (partials == nullptr || partial_bytes == 0 ||
      partial_bytes > in.workspace_bytes) return false;

  char* poison = partials;
  if (options.repeat_state == RepeatState::ControlPoison) {
    // The control must be byte-disjoint from the live partial plane while
    // retaining the same transfer size and host/device synchronization.
    if (partial_bytes > in.workspace_bytes / 2) return false;
    poison = in.workspace + (in.workspace_bytes - partial_bytes);
    if (poison < partials + partial_bytes) return false;
  }
  return hggcMemset(poison, 0x7f, partial_bytes) == hggcSuccess &&
         hggcDeviceSynchronize() == hggcSuccess;
}

template <class Launch>
bool measure(Launch&& launch, int iterations, CellResult& result) {
  std::vector<double> samples;
  samples.reserve(std::size_t(iterations));
  for (int i = 0; i < iterations; ++i) {
    EventPair events;
    if (hggcEventRecord(events.start, nullptr) != hggcSuccess ||
        launch() != cutlass::Status::kSuccess ||
        hggcEventRecord(events.stop, nullptr) != hggcSuccess ||
        hggcEventSynchronize(events.stop) != hggcSuccess) return false;
    double us = 0;
    if (!elapsed_us(events.start, events.stop, us)) return false;
    samples.push_back(us);
  }
  std::sort(samples.begin(), samples.end());
  result.min_us = samples.front();
  result.max_us = samples.back();
  result.median_us = samples.size() & 1
      ? samples[samples.size() / 2]
      : 0.5 * (samples[samples.size()/2 - 1] + samples[samples.size()/2]);
  result.samples_us = std::move(samples);
  return true;
}

template <int QType, int ArtifactTileK, int TM, int TN, int TK,
          int WM, int WN, int Stages, int BChunk, int AProvider>
bool run_tc_row(DeviceInputs const& in, Options const& options,
                RowResult& row) {
  using Types = TcRowTypes<QType, ArtifactTileK, TM, TN, TK,
                           WM, WN, Stages, BChunk, AProvider>;
  using F = typename Types::F;
  using Low = typename Types::Low;
  using High = typename Types::High;
  using Tile = typename Types::Tile;
  using Warp = typename Types::Warp;
  using Shipping = typename Types::Shipping;
  using Split = typename Types::Split;
  using ShippingGemm = typename Types::ShippingGemm;
  using ShippingKernel = typename Types::ShippingKernel;
  using SplitGemm = typename Types::SplitGemm;
  using SplitKernel = typename Types::SplitKernel;
  using Reduction = typename Types::Reduction;
  constexpr bool use_packed_a = Types::use_packed_a;

  if (!in.a || !in.low || !in.units || !in.output || !in.workspace ||
      !in.golden || in.m <= 0 || in.n <= 0 || in.k <= 0 ||
      in.k % TK || (!std::is_void_v<High> && !in.high)) return false;

  constexpr bool shipping_fits =
      Shipping::SharedStorageSize <= ppu_tactics::kBlockSmemBytes;
  constexpr bool split_fits =
      SplitKernel::SharedStorageSize <= ppu_tactics::kBlockSmemBytes;
  bool row_ok = true;
  for (std::size_t index = 0; index < kSplits.size(); ++index) {
    CellResult& result = row.cells[index];
    result = CellResult{};
    int const splits = kSplits[index];
    result.split = splits;
    result.full_output = splits == 1 && !options.force_custom_splitk_s1;
    result.shipping_smem = Shipping::SharedStorageSize;
    result.split_smem = SplitKernel::SharedStorageSize;
    result.a_provider_capacity_rows = Types::a_provider_capacity_rows;
    if (options.only_split && options.only_split != splits) continue;
    if constexpr (TM == 8) {
      if (in.m > options.tm8_max_m) {
        result.state = State::M8DecodeOnly;
        continue;
      }
    }
    if constexpr (use_packed_a) {
      if (in.m != 1) {
        result.state = State::PackedADecodeOnly;
        continue;
      }
    }
    if constexpr (!shipping_fits) {
      result.state = State::ShippingSharedStorage;
      continue;
    } else {
      using Prepared = dense_splitk_parallel_ppu::PreparedMultiformatLauncher<
          Shipping, Tile, Warp, High, true>;
      auto mainloop = Prepared::make_mainloop_arguments(
          in.a, reinterpret_cast<Low const*>(in.low), in.units, nullptr,
          in.m, in.n, in.k, F::GroupSize,
          reinterpret_cast<High const*>(in.high));

      if (splits == 1 && !options.force_custom_splitk_s1) {
        using StrideC = typename ShippingKernel::StrideC;
        using StrideD = typename ShippingKernel::StrideD;
        StrideC sC = cutlass::make_cute_packed_stride(
            StrideC{}, cute::make_shape(in.m, in.n, 1));
        StrideD sD = cutlass::make_cute_packed_stride(
            StrideD{}, cute::make_shape(in.m, in.n, 1));
        typename ShippingGemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {in.m, in.n, in.k, 1}, mainloop,
            {{1.f, 0.f}, static_cast<half_t*>(nullptr), sC, in.output, sD}, 1};
        ShippingGemm gemm;
        if (ShippingGemm::can_implement(args) != cutlass::Status::kSuccess) {
          result.state = State::CanImplement; continue;
        }
        if (gemm.initialize(args, nullptr, nullptr) != cutlass::Status::kSuccess) {
          result.state = State::Initialize; row_ok = false; continue;
        }
        auto launch = [&] { return gemm.run(nullptr); };
        bool correct = true;
        std::uint64_t fingerprint = 0;
        for (int repeat = 0; repeat < options.correctness_repeats; ++repeat) {
          result.failure_repeat = repeat;
          if (launch() != cutlass::Status::kSuccess) {
            result.failure_step = "CORRECTNESS_LAUNCH"; correct = false; break;
          }
          if (hggcDeviceSynchronize() != hggcSuccess) {
            result.failure_step = "CORRECTNESS_SYNCHRONIZE"; correct = false; break;
          }
          if (!inspect(in, result)) {
            result.failure_step = "CORRECTNESS_OUTPUT_COPY"; correct = false; break;
          }
          if (result.raw_bad != 0 ||
              (repeat && result.fingerprint != fingerprint)) {
            result.failure_step = result.raw_bad ? "RAW_FP16_MISMATCH" :
                                                   "FINGERPRINT_MISMATCH";
            correct = false; break;
          }
          fingerprint = result.fingerprint;
        }
        if (!correct) { result.state = State::Correctness; row_ok = false; continue; }
        if (options.measure && !measure(launch, options.iterations, result)) {
          result.failure_step = "TIMING";
          result.state = State::Timing; row_ok = false; continue;
        }
        result.failure_step = "NONE"; result.failure_repeat = -1;
        result.state = State::Measured;
        continue;
      }

      if constexpr (!split_fits) {
        result.state = State::SplitSharedStorage;
        continue;
      } else {
        int const k_tiles = in.k / TK;
        auto const partition = cutlass::gemm::kernel::fixed_splitk::make_params(
            ((in.m + TM - 1) / TM) * ((in.n + TN - 1) / TN),
            k_tiles, splits);
        if (!partition.is_valid()) {
          result.state = State::SplitPartition; continue;
        }
        if (int(partition.k_tiles_per_split) < Stages - 1) {
          result.state = State::PipelineDepth; continue;
        }
        dense_splitk_parallel_ppu::WorkspacePlan plan;
        bool workspace_ok = dense_splitk_parallel_ppu::query_workspace_plan(
            in.m, in.n, splits, plan);
        if (options.force_custom_splitk_s1 && splits == 1) {
          // The public launcher intentionally reports zero workspace for its
          // shipping S=1 route.  This diagnostic bypasses that route, so ask
          // the producer/reducer ABI directly for its one FP32 plane.
          workspace_ok = cutlass::gemm::device::splitk_parallel::
              fp32_workspace_size(in.m, in.n, splits, plan.partial_bytes);
          plan.total_bytes = plan.partial_bytes;
        }
        if (!workspace_ok || plan.partial_bytes > in.workspace_bytes) {
          result.state = State::SplitPartition; continue;
        }
        result.partial_bytes = plan.partial_bytes;
        float* partials = reinterpret_cast<float*>(in.workspace);
        using PartialStride = typename SplitKernel::StrideD;
        PartialStride sP = cutlass::gemm::kernel::detail::
            make_compact_fp32_partial_stride<PartialStride>(in.m, in.n);
        typename SplitGemm::Arguments producer_args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {in.m, in.n, in.k, 1}, mainloop,
            {partials, sP, partials, sP}, splits};
        typename Reduction::Arguments reducer_args{
            in.m, in.n, splits, partials, in.workspace_bytes, in.output, in.n};
        SplitGemm producer;
        Reduction reducer;
        if (SplitGemm::can_implement(producer_args) !=
                cutlass::Status::kSuccess ||
            Reduction::can_implement(reducer_args) != cutlass::Status::kSuccess) {
          result.state = State::CanImplement; continue;
        }
        if (producer.initialize(producer_args, nullptr, nullptr) !=
                cutlass::Status::kSuccess ||
            reducer.initialize(reducer_args) != cutlass::Status::kSuccess) {
          result.state = State::Initialize; row_ok = false; continue;
        }
        auto producer_launch = [&] { return producer.run(nullptr); };
        auto full_launch = [&] {
          auto status = producer.run(nullptr);
          return status == cutlass::Status::kSuccess ? reducer.run(nullptr) : status;
        };
        bool correct = true;
        std::uint64_t fingerprint = 0;
        for (int repeat = 0; repeat < options.correctness_repeats; ++repeat) {
          result.failure_repeat = repeat;
          if (!prepare_repeat_state(
                  in, options, reinterpret_cast<char*>(partials),
                  plan.partial_bytes)) {
            result.failure_step = "REPEAT_STATE_PREPARE";
            result.state = State::Initialize;
            correct = false;
            break;
          }
          if (full_launch() != cutlass::Status::kSuccess) {
            result.failure_step = "CORRECTNESS_FULL_LAUNCH"; correct = false; break;
          }
          if (hggcDeviceSynchronize() != hggcSuccess) {
            result.failure_step = "CORRECTNESS_SYNCHRONIZE"; correct = false; break;
          }
          if (!inspect(in, result)) {
            result.failure_step = "CORRECTNESS_OUTPUT_COPY"; correct = false; break;
          }
          bool partial_checked = false;
          if (options.check_partials_each_repeat) {
            result.partial_probe_attempted = true;
            partial_checked = inspect_failed_partials(
                in, splits, partials, plan.partial_bytes, result);
            result.partial_probe_complete = partial_checked;
            if (!partial_checked) {
              result.failure_step = "CORRECTNESS_PARTIAL_COPY";
              result.state = State::Initialize;
              correct = false;
              break;
            }
          }
          bool const partial_bad =
              partial_checked && result.partial_value_raw_bad != 0;
          if (result.raw_bad != 0 || partial_bad ||
              (repeat && result.fingerprint != fingerprint)) {
            if (result.raw_bad != 0 && options.force_custom_splitk_s1) {
              // Observe only after the original producer+reducer failure and
              // device completion.  This preserves the cadence that exposed
              // the bug, then distinguishes bad producer bytes from a
              // same-stream publication/reducer failure.
              bool const partial_ok = partial_checked ||
                  inspect_failed_partials(
                      in, splits, partials, plan.partial_bytes, result);
              result.partial_probe_attempted = true;
              CellResult replay;
              bool const replay_ok =
                  reducer.run(nullptr) == cutlass::Status::kSuccess &&
                  hggcDeviceSynchronize() == hggcSuccess &&
                  inspect(in, replay);
              result.partial_probe_complete = partial_ok && replay_ok;
              if (replay_ok) {
                result.reducer_replay_raw_bad = replay.raw_bad;
                result.reducer_replay_first_bad = replay.first_bad_index;
                result.reducer_replay_first_want = replay.first_bad_want;
                result.reducer_replay_first_got = replay.first_bad_got;
              }
            }
            result.failure_step = result.raw_bad
                ? "RAW_FP16_MISMATCH"
                : partial_bad ? "PARTIAL_FP32_MISMATCH_WITH_CLEAN_OUTPUT"
                              : "FINGERPRINT_MISMATCH";
            if (partial_bad && result.raw_bad == 0)
              result.state = State::PartialCorrectness;
            correct = false; break;
          }
          fingerprint = result.fingerprint;
        }
        if (!correct) {
          if (result.state != State::Initialize &&
              result.state != State::PartialCorrectness)
            result.state = State::Correctness;
          row_ok = false;
          continue;
        }
        result.reducer_correctness_untimed = true;
        if (options.measure) {
          auto timing = splitk_producer_timing::measure(
              producer_launch, [&] { return reducer.run(nullptr); },
              options.iterations);
          result.failure_repeat = timing.failure_repeat;
          result.failure_step =
              splitk_producer_timing::failure_name(timing.failure);
          if (timing.failure != splitk_producer_timing::Failure::None) {
            result.state = splitk_producer_timing::is_launch_failure(
                               timing.failure)
                ? State::Launch : State::Timing;
            row_ok = false;
            continue;
          }
          result.samples_us = std::move(timing.samples_us);
          result.min_us = timing.min_us;
          result.max_us = timing.max_us;
          result.median_us = timing.median_us;
          if (!inspect(in, result) || result.raw_bad != 0 ||
              result.fingerprint != fingerprint) {
            result.failure_step = result.raw_bad
                ? "ORDERED_CLOSE_RAW_FP16_MISMATCH"
                : "ORDERED_CLOSE_REDUCER_OR_COPY";
            result.state = State::Correctness;
            row_ok = false;
            continue;
          }
        }
        result.failure_step = "NONE"; result.failure_repeat = -1;
        result.state = State::Measured;
      }
    }
  }
  return row_ok;
}

}  // namespace fq_internal_sweep
