/***************************************************************************************************
 * L189: live dense Split-K reduction/device seam.
 *
 * This is intentionally a real two-kernel test.  The producer writes FP32 [S][M][N] partials on a
 * non-default stream, the reduction is enqueued immediately on that same stream, and the host waits
 * only after both launches.  N=10 exercises the non-vector tail of an 8-element reduction access.
 **************************************************************************************************/
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "quactlize_extensions/cutlass/gemm/device/ppu_mixed_input_splitk_parallel.hpp"

namespace splitk = cutlass::gemm::device::splitk_parallel;

namespace {

constexpr int kRows = 3;
constexpr int kColumns = 10;
constexpr int kSlices = 4;

using Reduction =
    splitk::PpuMixedInputSplitKParallelM1FastReduction<2>;

constexpr int kFastColumns = 4096;
constexpr int kTailColumns = 66;

__host__ __device__ float partial_value(int split_k, int element) {
  // This cell distinguishes fixed 0,1,2,3 order from a reassociated sum:
  // ((2^24 + 1) - 2^24) + 0 == 0 in FP32, while (2^24 - 2^24) + 1 == 1.
  if (element == 0) {
    constexpr float order_sensitive[kSlices] = {
        16777216.0f, 1.0f, -16777216.0f, 0.0f};
    return order_sensitive[split_k];
  }
  return float(100 * split_k + element) + 0.25f;
}

__global__ void produce_partials(float* workspace) {
  int const elements_per_slice = kRows * kColumns;
  int const index = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  if (index >= kSlices * elements_per_slice) {
    return;
  }
  int const split_k = index / elements_per_slice;
  int const element = index % elements_per_slice;
  workspace[index] = partial_value(split_k, element);
}

__host__ __device__ float fast_partial_value(int split_k, int column) {
  if (column == 0) {
    constexpr float order_sensitive[8] = {
        65504.0f, 0.0009765625f, -65504.0f, 1.0f,
        32768.0f, 0.0009765625f, -32768.0f, 1.0f};
    return order_sensitive[split_k];
  }
  return split_k == 0
      ? float(cutlass::half_t::bitcast(uint16_t(0x2000 + column)))
      : 0.0f;
}

__global__ void produce_fast_partials(
    float* workspace, int partitions, int columns) {
  int const index = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  int const count = partitions * columns;
  if (index >= count) return;
  int const split_k = index / columns;
  int const column = index % columns;
  workspace[index] = fast_partial_value(split_k, column);
}

bool cuda_ok(cudaError_t status, char const* operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "[l189] CUDA FAIL: %s: %s\n", operation,
               cudaGetErrorString(status));
  return false;
}

bool expect_status(
    char const* name, cutlass::Status got, cutlass::Status expected) {
  if (got == expected) {
    return true;
  }
  std::fprintf(stderr, "[l189] admission FAIL: %s got=%s expected=%s\n",
               name, cutlass::cutlassGetStatusString(got),
               cutlass::cutlassGetStatusString(expected));
  return false;
}

}  // namespace

int main() {
  int failures = 0;

  size_t anchor_bytes = 0;
  if (!splitk::fp32_workspace_size(1, 4096, 8, anchor_bytes) ||
      anchor_bytes != 131072) {
    std::fprintf(stderr,
                 "[l189] workspace FAIL: M=1 N=4096 S=8 got=%zu expected=131072\n",
                 anchor_bytes);
    ++failures;
  }

  size_t live_bytes = 0;
  if (!splitk::fp32_workspace_size(
          kRows, kColumns, kSlices, live_bytes)) {
    std::fprintf(stderr, "[l189] live workspace formula rejected valid shape\n");
    return 2;
  }

  float* workspace = nullptr;
  cutlass::half_t* destination = nullptr;
  cudaStream_t stream = nullptr;
  if (!cuda_ok(cudaMalloc(&workspace, live_bytes), "cudaMalloc(workspace)") ||
      !cuda_ok(cudaMalloc(
                   &destination,
                   size_t(kRows) * kColumns * sizeof(cutlass::half_t)),
               "cudaMalloc(destination)") ||
      !cuda_ok(cudaStreamCreate(&stream), "cudaStreamCreate")) {
    return 3;
  }

  Reduction::Arguments args{
      kRows,
      kColumns,
      kSlices,
      workspace,
      live_bytes,
      destination,
      kColumns};

  // Admission RED controls.  They share every field with the live case and
  // vary exactly one property at a time.
  {
    auto plant = args;
    plant.workspace_bytes = live_bytes - 1;
    failures += !expect_status(
        "workspace-short-by-one", Reduction::can_implement(plant),
        cutlass::Status::kErrorInvalidProblem);
  }
  {
    auto plant = args;
    plant.workspace = nullptr;
    failures += !expect_status(
        "null-workspace", Reduction::can_implement(plant),
        cutlass::Status::kErrorWorkspaceNull);
  }
  {
    auto plant = args;
    plant.destination = nullptr;
    failures += !expect_status(
        "null-destination", Reduction::can_implement(plant),
        cutlass::Status::kErrorInvalidProblem);
  }
  {
    auto plant = args;
    plant.split_k_slices = 0;
    failures += !expect_status(
        "zero-slices", Reduction::can_implement(plant),
        cutlass::Status::kErrorInvalidProblem);
  }
  {
    auto plant = args;
    plant.split_k_slices = 3;
    failures += !expect_status(
        "unsupported-slices", Reduction::can_implement(plant),
        cutlass::Status::kErrorInvalidProblem);
  }
  {
    auto plant = args;
    plant.workspace = reinterpret_cast<float*>(
        reinterpret_cast<char*>(workspace) + sizeof(float));
    failures += !expect_status(
        "workspace-not-128b-aligned", Reduction::can_implement(plant),
        cutlass::Status::kErrorMisalignedOperand);
  }
  {
    auto plant = args;
    plant.destination = reinterpret_cast<cutlass::half_t*>(workspace);
    failures += !expect_status(
        "destination-aliases-partials", Reduction::can_implement(plant),
        cutlass::Status::kErrorInvalidProblem);
  }

  Reduction reduction;
  cutlass::Status const install = reduction.initialize(args);
  if (install != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "[l189] live admission FAIL: %s\n",
                 cutlass::cutlassGetStatusString(install));
    ++failures;
  } else {
    if (reduction.fast_path_selected_for_diagnostics()) {
      std::fprintf(stderr,
                   "[l189] fallback FAIL: M3/N10 tail selected M1 fast path\n");
      ++failures;
    }
    int const count = kRows * kColumns * kSlices;
    auto launch_producer = [&](hggcStream_t ordered_stream) {
      produce_partials<<<(count + 63) / 64, 64, 0,
                         reinterpret_cast<cudaStream_t>(ordered_stream)>>>(
          workspace);
      return cudaGetLastError() == cudaSuccess
          ? cutlass::Status::kSuccess
          : cutlass::Status::kErrorInternal;
    };

    cutlass::Status const launch =
        splitk::launch_main_then_reduce_same_stream(
            launch_producer, reduction,
            reinterpret_cast<hggcStream_t>(stream));
    if (launch != cutlass::Status::kSuccess) {
      std::fprintf(stderr, "[l189] two-launch seam FAIL: %s\n",
                   cutlass::cutlassGetStatusString(launch));
      ++failures;
    }
    if (!cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) {
      ++failures;
    }
  }

  std::vector<cutlass::half_t> result(size_t(kRows) * kColumns);
  if (!cuda_ok(cudaMemcpy(
                   result.data(), destination,
                   result.size() * sizeof(cutlass::half_t),
                   cudaMemcpyDeviceToHost),
               "cudaMemcpy(result)")) {
    ++failures;
  }

  int bad = 0;
  for (int element = 0; element < kRows * kColumns; ++element) {
    // Volatile prevents the host oracle from reassociating this intentionally
    // order-sensitive sequence.
    volatile float expected = 0.0f;
    for (int split_k = 0; split_k < kSlices; ++split_k) {
      expected = expected + partial_value(split_k, element);
    }
    cutlass::half_t const expected_half{float(expected)};
    if (result[size_t(element)].raw() != expected_half.raw()) {
      if (bad < 8) {
        std::fprintf(stderr,
                     "[l189] value FAIL: element=%d got=0x%04x expected=0x%04x\n",
                     element, unsigned(result[size_t(element)].raw()),
                     unsigned(expected_half.raw()));
      }
      ++bad;
    }
  }
  if (bad != 0) {
    failures += bad;
  }
  if (result[0].raw() != cutlass::half_t(0.0f).raw()) {
    std::fprintf(stderr,
                 "[l189] fixed-order witness FAIL: first output is not zero\n");
    ++failures;
  }

  // Production M=1 fast path: all three supported S values must select the
  // one-warp kernel and remain bit-identical to an increasing-S host replay.
  float* fast_workspace = nullptr;
  cutlass::half_t* fast_destination = nullptr;
  std::size_t const fast_workspace_bytes =
      std::size_t(8) * kFastColumns * sizeof(float);
  if (!cuda_ok(cudaMalloc(&fast_workspace, fast_workspace_bytes),
               "cudaMalloc(fast_workspace)") ||
      !cuda_ok(cudaMalloc(&fast_destination,
                          std::size_t(kFastColumns) * sizeof(cutlass::half_t)),
               "cudaMalloc(fast_destination)")) {
    return 4;
  }
  int fast_bad = 0;
  for (int partitions : {2, 4, 8}) {
    Reduction::Arguments fast_args{
        1, kFastColumns, partitions, fast_workspace,
        std::size_t(partitions) * kFastColumns * sizeof(float),
        fast_destination, kFastColumns};
    Reduction fast_reduction;
    if (fast_reduction.initialize(fast_args) != cutlass::Status::kSuccess ||
        !fast_reduction.fast_path_selected_for_diagnostics()) {
      std::fprintf(stderr,
                   "[l189] fast admission FAIL: S=%d did not select fast path\n",
                   partitions);
      ++failures;
      continue;
    }
    int const count = partitions * kFastColumns;
    if (!cuda_ok(cudaMemsetAsync(
                     fast_destination, 0xa5,
                     std::size_t(kFastColumns) * sizeof(cutlass::half_t),
                     stream),
                 "poison fast destination")) {
      ++failures;
      continue;
    }
    auto launch_producer = [&](hggcStream_t ordered_stream) {
      produce_fast_partials<<<(count + 127) / 128, 128, 0,
                               reinterpret_cast<cudaStream_t>(ordered_stream)>>>(
          fast_workspace, partitions, kFastColumns);
      return cudaGetLastError() == cudaSuccess
          ? cutlass::Status::kSuccess
          : cutlass::Status::kErrorInternal;
    };
    cutlass::Status const launch =
        splitk::launch_main_then_reduce_same_stream(
            launch_producer, fast_reduction,
            reinterpret_cast<hggcStream_t>(stream));
    if (launch != cutlass::Status::kSuccess ||
        !cuda_ok(cudaStreamSynchronize(stream),
                 "cudaStreamSynchronize(fast)")) {
      ++failures;
      continue;
    }
    std::vector<cutlass::half_t> fast_result(kFastColumns);
    if (!cuda_ok(cudaMemcpy(fast_result.data(), fast_destination,
                            fast_result.size() * sizeof(fast_result[0]),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(fast_result)")) {
      ++failures;
      continue;
    }
    for (int column = 0; column < kFastColumns; ++column) {
      volatile float expected = 0.0f;
      for (int split_k = 0; split_k < partitions; ++split_k) {
        expected = expected + fast_partial_value(split_k, column);
      }
      cutlass::half_t const expected_half{float(expected)};
      if (fast_result[std::size_t(column)].raw() != expected_half.raw()) {
        ++fast_bad;
      }
    }
  }

  // Three single-variable fallback cases must not merely classify correctly:
  // they execute the old tail-capable body against poisoned D and compare all
  // output bits.  This guards against turning admission into a silent no-op.
  auto verify_fallback = [&](char const* name, int columns,
                             float* fallback_workspace,
                             cutlass::half_t* fallback_destination) {
    std::size_t const fallback_bytes =
        std::size_t(kSlices) * columns * sizeof(float);
    Reduction::Arguments fallback_args{
        1, columns, kSlices, fallback_workspace, fallback_bytes,
        fallback_destination, columns};
    Reduction fallback_reduction;
    if (fallback_reduction.initialize(fallback_args) !=
            cutlass::Status::kSuccess ||
        fallback_reduction.fast_path_selected_for_diagnostics()) {
      std::fprintf(stderr, "[l189] fallback admission FAIL: %s\n", name);
      return false;
    }
    int const count = kSlices * columns;
    if (cudaMemsetAsync(fallback_destination, 0xa5,
                        std::size_t(columns) * sizeof(cutlass::half_t),
                        stream) != cudaSuccess) {
      std::fprintf(stderr, "[l189] fallback poison FAIL: %s\n", name);
      return false;
    }
    produce_fast_partials<<<(count + 127) / 128, 128, 0, stream>>>(
        fallback_workspace, kSlices, columns);
    if (cudaGetLastError() != cudaSuccess ||
        fallback_reduction.run(reinterpret_cast<hggcStream_t>(stream)) !=
            cutlass::Status::kSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      std::fprintf(stderr, "[l189] fallback launch FAIL: %s\n", name);
      return false;
    }
    std::vector<cutlass::half_t> observed(
        static_cast<std::size_t>(columns), cutlass::half_t{});
    if (cudaMemcpy(observed.data(), fallback_destination,
                   observed.size() * sizeof(observed[0]),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      return false;
    }
    int fallback_bad = 0;
    for (int column = 0; column < columns; ++column) {
      volatile float expected = 0.0f;
      for (int split = 0; split < kSlices; ++split) {
        expected = expected + fast_partial_value(split, column);
      }
      fallback_bad += observed[std::size_t(column)].raw() !=
          cutlass::half_t(float(expected)).raw();
    }
    if (fallback_bad != 0) {
      std::fprintf(stderr, "[l189] fallback value FAIL: %s bad=%d/%d\n",
                   name, fallback_bad, columns);
    }
    return fallback_bad == 0;
  };
  failures += !verify_fallback(
      "M1-N66-tail", kTailColumns, fast_workspace, fast_destination);
  failures += !verify_fallback(
      "workspace-16B-not-128B", 64, fast_workspace + 4,
      fast_destination);
  failures += !verify_fallback(
      "destination-half-aligned-not-16B", 64, fast_workspace,
      fast_destination + 1);

  // A shape whose one-warp CTA count exceeds dim3.x must never select the
  // fast dispatcher.  Fake, nonoverlapping ranges are sufficient because
  // initialize performs arithmetic only and this case is deliberately not run.
  {
    int64_t const huge_columns =
        (int64_t((std::numeric_limits<unsigned>::max)()) + 1) * 64;
    std::size_t huge_bytes = 0;
    bool const sized = splitk::fp32_workspace_size(
        1, huge_columns, 2, huge_bytes);
    Reduction::Arguments huge_args{
        1, huge_columns, 2, reinterpret_cast<float const*>(uintptr_t(0x10000)),
        huge_bytes,
        reinterpret_cast<cutlass::half_t*>(uintptr_t(0x400000000000)),
        huge_columns};
    Reduction huge_reduction;
    if (!sized || huge_reduction.initialize(huge_args) !=
                      cutlass::Status::kSuccess ||
        huge_reduction.fast_path_selected_for_diagnostics()) {
      std::fprintf(stderr,
                   "[l189] grid-overflow fallback FAIL\n");
      ++failures;
    }
  }
  failures += fast_bad;
  cudaFree(fast_destination);
  cudaFree(fast_workspace);

  cudaStreamDestroy(stream);
  cudaFree(destination);
  cudaFree(workspace);

  std::printf(
      "[l189] workspace=131072 tail=N10 same_stream=producer->reduce "
      "fixed_order=%s bad=%d fast_S2_S4_S8_bad=%d "
      "fallback=tail+workspace_alignment+D_alignment+grid_limit "
      "admission_red=7/7\n",
      result[0].raw() == cutlass::half_t(0.0f).raw() ? "PASS" : "FAIL",
      bad, fast_bad);
  if (failures != 0) {
    std::fprintf(stderr, "[l189] FAIL: failures=%d\n", failures);
    return 1;
  }
  std::printf("[l189] PASS: live reduction and all admission controls\n");
  return 0;
}
