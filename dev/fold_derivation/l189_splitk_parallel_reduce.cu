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
#include <vector>

#include "quactlize_extensions/cutlass/gemm/device/ppu_mixed_input_splitk_parallel.hpp"

namespace splitk = cutlass::gemm::device::splitk_parallel;

namespace {

constexpr int kRows = 3;
constexpr int kColumns = 10;
constexpr int kSlices = 4;
constexpr int kElementsPerAccess = 8;

using Reduction =
    splitk::PpuMixedInputSplitKParallelReduction<kElementsPerAccess>;

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

  cudaStreamDestroy(stream);
  cudaFree(destination);
  cudaFree(workspace);

  std::printf(
      "[l189] workspace=131072 tail=N10 same_stream=producer->reduce "
      "fixed_order=%s bad=%d admission_red=7/7\n",
      result[0].raw() == cutlass::half_t(0.0f).raw() ? "PASS" : "FAIL",
      bad);
  if (failures != 0) {
    std::fprintf(stderr, "[l189] FAIL: failures=%d\n", failures);
    return 1;
  }
  std::printf("[l189] PASS: live reduction and all admission controls\n");
  return 0;
}
