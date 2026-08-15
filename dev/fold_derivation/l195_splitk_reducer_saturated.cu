/***************************************************************************************************
 * L195: saturated body-throughput diagnostic for the production M=1 Split-K reducer.
 *
 * The shipping N=4096 problem is intentionally too small to saturate HBM: its 136 KiB S=8 body
 * is dominated by the second kernel launch.  This larger, otherwise identical one-row reduction
 * separates the vector memory body from that launch floor and compares the four legal vector
 * widths without changing arithmetic or workspace layout.
 **************************************************************************************************/
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#include "quactlize_extensions/cutlass/gemm/device/ppu_mixed_input_splitk_parallel.hpp"

namespace splitk = cutlass::gemm::device::splitk_parallel;

namespace {

constexpr int64_t kColumns = int64_t(1) << 22;
constexpr int kPartitions = 8;
constexpr int kWarmup = 16;
constexpr int kIterations = 128;

bool cuda_ok(cudaError_t status, char const* what) {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "[l195] CUDA FAIL: %s: %s\n", what,
               cudaGetErrorString(status));
  return false;
}

template <int EPA>
bool run_case(float* workspace, cutlass::half_t* output, double peak_gbs) {
  using Reduction = splitk::PpuMixedInputSplitKParallelM1FastReduction<EPA>;
  typename Reduction::Arguments args{
      1, kColumns, kPartitions, workspace,
      std::size_t(kPartitions) * std::size_t(kColumns) * sizeof(float),
      output, kColumns};
  Reduction reducer;
  if (reducer.initialize(args) != cutlass::Status::kSuccess ||
      !reducer.fast_path_selected_for_diagnostics()) {
    std::fprintf(stderr, "[l195] admission FAIL: EPA=%d\n", EPA);
    return false;
  }
  if (!cuda_ok(cudaMemset(output, 0xa5,
                          std::size_t(kColumns) * sizeof(cutlass::half_t)),
               "poison output")) {
    return false;
  }
  for (int i = 0; i < kWarmup; ++i) {
    if (reducer.run() != cutlass::Status::kSuccess) return false;
  }
  if (!cuda_ok(cudaDeviceSynchronize(), "warmup")) return false;
  cudaEvent_t begin{}, end{};
  if (!cuda_ok(cudaEventCreate(&begin), "event begin") ||
      !cuda_ok(cudaEventCreate(&end), "event end")) return false;
  cudaEventRecord(begin);
  for (int i = 0; i < kIterations; ++i) {
    if (reducer.run() != cutlass::Status::kSuccess) return false;
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float millis = 0;
  cudaEventElapsedTime(&millis, begin, end);
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  double const us = double(millis) * 1000.0 / kIterations;
  double const bytes = double(kColumns) *
      (double(kPartitions) * sizeof(float) + sizeof(cutlass::half_t));
  double const gbs = bytes / us / 1000.0;
  std::vector<cutlass::half_t> observed(
      static_cast<std::size_t>(kColumns), cutlass::half_t{});
  bool const copied = cuda_ok(
      cudaMemcpy(observed.data(), output,
                 observed.size() * sizeof(observed[0]), cudaMemcpyDeviceToHost),
      "copy complete output");
  std::size_t bad = 0;
  if (copied) {
    for (cutlass::half_t value : observed) bad += value.raw() != 0;
  } else {
    bad = observed.size();
  }
  bool const exact = bad == 0;
  std::printf(
      "L195_SATURATED EPA=%d columns=%lld grid=%lld block=32 logical_bytes=%.0f "
      "latency=%.6f_us logical=%.3f_GB/s nameplate=%.3f_GB/s utilization=%.3f%% "
      "complete_output_bad=%zu/%lld\n",
      EPA, static_cast<long long>(kColumns),
      static_cast<long long>(kColumns / (32 * EPA)), bytes, us, gbs,
      peak_gbs, peak_gbs > 0 ? 100.0 * gbs / peak_gbs : 0.0,
      bad, static_cast<long long>(kColumns));
  return exact && us > 0;
}

}  // namespace

int main() {
  int device = 0;
  cudaDeviceProp property{};
  if (!cuda_ok(cudaGetDevice(&device), "get device") ||
      !cuda_ok(cudaGetDeviceProperties(&property, device), "get properties")) {
    return 2;
  }
  double const peak_gbs =
      2.0 * double(property.memoryClockRate) * 1000.0 *
      (double(property.memoryBusWidth) / 8.0) / 1.0e9;
  float* workspace = nullptr;
  cutlass::half_t* output = nullptr;
  std::size_t const workspace_bytes =
      std::size_t(kPartitions) * std::size_t(kColumns) * sizeof(float);
  std::size_t const output_bytes = std::size_t(kColumns) * sizeof(cutlass::half_t);
  if (!cuda_ok(cudaMalloc(&workspace, workspace_bytes), "malloc workspace") ||
      !cuda_ok(cudaMalloc(&output, output_bytes), "malloc output") ||
      !cuda_ok(cudaMemset(workspace, 0, workspace_bytes), "zero workspace")) {
    return 2;
  }
  int failures = 0;
  failures += !run_case<1>(workspace, output, peak_gbs);
  failures += !run_case<2>(workspace, output, peak_gbs);
  failures += !run_case<4>(workspace, output, peak_gbs);
  failures += !run_case<8>(workspace, output, peak_gbs);
  cudaFree(output);
  cudaFree(workspace);
  if (failures) {
    std::fprintf(stderr, "[l195] FAIL: %d case(s)\n", failures);
    return 1;
  }
  std::printf("[l195] PASS: all production vector widths are exact; performance is diagnostic\n");
  return 0;
}
