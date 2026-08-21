/***************************************************************************************************
 * L194: local live sweep of the mature CUTLASS vector Split-K reducer at the M=1 geometry.
 *
 * This is a topology/code-path selection oracle, not a PPU performance claim.  It compares the
 * current checked reducer with Shape<1,32*EPA> vector reducers under one warm launch protocol,
 * verifies every output bit, and reports an empty-kernel floor for each launch geometry.
 **************************************************************************************************/
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/reduction/device/reduce_split_k.h"
#include "cutlass/reduction/kernel/reduce_split_k.h"
#include "cutlass/reduction/thread/reduction_operators.h"
#include "actlize_extensions/cutlass/gemm/device/ppu_mixed_input_splitk_parallel.hpp"

namespace splitk = cutlass::gemm::device::splitk_parallel;

namespace {

constexpr int kN = 4096;
constexpr int kMaxS = 8;
constexpr int kWarmup = 64;
constexpr int kIterations = 4000;

__host__ __device__ float partial_value(int split, int column) {
  if (column == 0) {
    constexpr float ordered[kMaxS] = {
        16777216.0f, 1.0f, -16777216.0f, 0.0f,
        8388608.0f, 1.0f, -8388608.0f, 0.0f};
    return ordered[split];
  }
  return float(((split * 29 + column * 17) % 31) - 15) * 0.25f;
}

__global__ void empty_kernel() {}

template <int Partitions>
__global__ void volatile_fixed_order_kernel(
    float const* workspace, cutlass::half_t* output) {
  constexpr int EPA = 2;
  int const column =
      (int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x)) * EPA;
  if (column >= kN) return;
  auto accumulator =
      splitk::reduce_fp32_volatile_fixed_partition_order<EPA, Partitions>(
          workspace, kN, column);
  cutlass::NumericArrayConverter<cutlass::half_t, float, EPA> convert;
  auto converted = convert(accumulator);
  using Output = cutlass::AlignedArray<cutlass::half_t, EPA>;
  Output packed;
  packed[0] = converted[0];
  packed[1] = converted[1];
  *reinterpret_cast<Output*>(output + column) = packed;
}

bool cuda_ok(cudaError_t status, char const* what) {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "[l194] CUDA FAIL: %s: %s\n", what,
               cudaGetErrorString(status));
  return false;
}

template <class Launch>
double time_launches(Launch&& launch, dim3 grid, dim3 block) {
  for (int i = 0; i < kWarmup; ++i) launch();
  if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
  cudaEvent_t begin{}, end{};
  if (cudaEventCreate(&begin) != cudaSuccess || cudaEventCreate(&end) != cudaSuccess)
    return -1.0;
  cudaEventRecord(begin);
  for (int i = 0; i < kIterations; ++i) launch();
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float millis = 0;
  cudaEventElapsedTime(&millis, begin, end);
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  (void)grid;
  (void)block;
  return double(millis) * 1000.0 / double(kIterations);
}

double time_empty(dim3 grid, dim3 block) {
  return time_launches(
      [=] { empty_kernel<<<grid, block>>>(); }, grid, block);
}

int check_output(cutlass::half_t const* device, std::vector<cutlass::half_t> const& expected) {
  std::vector<cutlass::half_t> got(expected.size());
  if (cudaMemcpy(got.data(), device, got.size() * sizeof(got[0]),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    return int(expected.size());
  }
  int bad = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    bad += got[i].raw() != expected[i].raw();
  }
  return bad;
}

template <int EPA>
bool run_vector_case(int partitions, float* workspace, cutlass::half_t* output,
                     std::vector<cutlass::half_t> const& expected) {
  using OutputOp = cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t, EPA, float, float,
      cutlass::epilogue::thread::ScaleType::Nothing>;
  using ReductionOp = cutlass::reduction::thread::ReduceAdd<float, float, EPA>;
  using Kernel = cutlass::reduction::kernel::ReduceSplitK<
      cutlass::MatrixShape<1, 32 * EPA>, OutputOp, ReductionOp>;
  using Device = cutlass::reduction::device::ReduceSplitK<Kernel>;

  static_assert(Kernel::Shape::kRow == 1);
  static_assert(Kernel::Shape::kColumn / EPA == 32);
  if (kN % (32 * EPA) != 0) return false;

  typename Kernel::WorkspaceTensorRef workspace_ref(workspace, kN);
  typename Kernel::OutputTensorRef output_ref(output, kN);
  typename Device::Arguments args{
      cutlass::MatrixCoord(1, kN), partitions, kN, workspace_ref,
      output_ref, output_ref, typename OutputOp::Params{},
      typename ReductionOp::Params{}};
  Device reducer;
  if (reducer.initialize(args) != cutlass::Status::kSuccess) return false;
  dim3 const grid = Kernel::grid_shape(cutlass::MatrixCoord(1, kN));
  dim3 const block = Kernel::block_shape();
  auto launch = [&] { reducer.run(); };
  launch();
  if (cudaDeviceSynchronize() != cudaSuccess) return false;
  int const bad = check_output(output, expected);
  double const us = time_launches(launch, grid, block);
  double const empty_us = time_empty(grid, block);
  double const bytes = double(partitions) * kN * sizeof(float) +
      double(kN) * sizeof(cutlass::half_t);
  std::printf(
      "L194_VECTOR S=%d EPA=%d grid=%u block=%u active_warps=%u "
      "latency=%.6f_us empty=%.6f_us excess=%.6f_us logical=%.3f_GB/s bad=%d\n",
      partitions, EPA, grid.x * grid.y, block.x * block.y,
      grid.x * grid.y * ((block.x * block.y + 31) / 32), us, empty_us,
      us - empty_us, bytes / us / 1000.0, bad);
  return bad == 0 && us > 0 && empty_us > 0;
}

bool run_legacy_case(int partitions, float* workspace, cutlass::half_t* output,
                     std::vector<cutlass::half_t> const& expected) {
  using Reduction = splitk::PpuMixedInputSplitKParallelReduction<8>;
  typename Reduction::Arguments args{
      1, kN, partitions, workspace,
      std::size_t(partitions) * kN * sizeof(float), output, kN};
  Reduction reducer;
  if (reducer.initialize(args) != cutlass::Status::kSuccess) return false;
  using Kernel = typename Reduction::Kernel;
  dim3 const grid = Kernel::grid_shape(1, kN);
  dim3 const block = Kernel::block_shape();
  auto launch = [&] { reducer.run(); };
  launch();
  if (cudaDeviceSynchronize() != cudaSuccess) return false;
  int const bad = check_output(output, expected);
  double const us = time_launches(launch, grid, block);
  double const empty_us = time_empty(grid, block);
  double const bytes = double(partitions) * kN * sizeof(float) +
      double(kN) * sizeof(cutlass::half_t);
  // Only one of the four y-warps has a valid row at M=1.
  unsigned const active_warps = grid.x * grid.y;
  std::printf(
      "L194_LEGACY S=%d EPA=8 grid=%u block=%u active_warps=%u "
      "dead_warps=%u latency=%.6f_us empty=%.6f_us excess=%.6f_us "
      "logical=%.3f_GB/s bad=%d\n",
      partitions, grid.x * grid.y, block.x * block.y, active_warps,
      active_warps * 3, us, empty_us, us - empty_us,
      bytes / us / 1000.0, bad);
  return bad == 0 && us > 0 && empty_us > 0;
}

bool run_production_fast_case(
    int partitions, float* workspace, cutlass::half_t* output,
    std::vector<cutlass::half_t> const& expected) {
  using Reduction =
      splitk::PpuMixedInputSplitKParallelM1FastReduction<2>;
  typename Reduction::Arguments args{
      1, kN, partitions, workspace,
      std::size_t(partitions) * kN * sizeof(float), output, kN};
  Reduction reducer;
  if (reducer.initialize(args) != cutlass::Status::kSuccess ||
      !reducer.fast_path_selected_for_diagnostics()) {
    return false;
  }
  using KernelS8 = typename Reduction::KernelS8;
  dim3 const grid = KernelS8::grid_shape(kN);
  dim3 const block = KernelS8::block_shape();
  auto launch = [&] { reducer.run(); };
  launch();
  if (cudaDeviceSynchronize() != cudaSuccess) return false;
  int const bad = check_output(output, expected);
  double const us = time_launches(launch, grid, block);
  double const empty_us = time_empty(grid, block);
  double const bytes = double(partitions) * kN * sizeof(float) +
      double(kN) * sizeof(cutlass::half_t);
  std::printf(
      "L194_PRODUCTION_FAST S=%d EPA=2 grid=%u block=%u active_warps=%u "
      "latency=%.6f_us empty=%.6f_us excess=%.6f_us logical=%.3f_GB/s "
      "fast=1 bad=%d\n",
      partitions, grid.x * grid.y, block.x * block.y,
      grid.x * grid.y * ((block.x * block.y + 31) / 32), us, empty_us,
      us - empty_us, bytes / us / 1000.0, bad);
  return bad == 0 && us > 0 && empty_us > 0;
}

template <int Partitions>
bool run_volatile_case(
    float* workspace, cutlass::half_t* output,
    std::vector<cutlass::half_t> const& expected) {
  constexpr int EPA = 2;
  dim3 const block(32, 1, 1);
  dim3 const grid(kN / (32 * EPA), 1, 1);
  volatile_fixed_order_kernel<Partitions><<<grid, block>>>(workspace, output);
  if (cudaDeviceSynchronize() != cudaSuccess) return false;
  int const bad = check_output(output, expected);
  std::printf(
      "L194_FUSED_VOLATILE S=%d EPA=2 grid=%u block=%u "
      "fixed_order=0..S-1 bad=%d\n",
      Partitions, grid.x, block.x, bad);
  return bad == 0;
}

}  // namespace

int main() {
  std::vector<float> host_workspace(std::size_t(kMaxS) * kN);
  for (int s = 0; s < kMaxS; ++s) {
    for (int n = 0; n < kN; ++n) {
      host_workspace[std::size_t(s) * kN + n] = partial_value(s, n);
    }
  }
  float* workspace = nullptr;
  cutlass::half_t* output = nullptr;
  if (!cuda_ok(cudaMalloc(&workspace, host_workspace.size() * sizeof(float)),
               "cudaMalloc(workspace)") ||
      !cuda_ok(cudaMalloc(&output, std::size_t(kN) * sizeof(cutlass::half_t)),
               "cudaMalloc(output)") ||
      !cuda_ok(cudaMemcpy(workspace, host_workspace.data(),
                          host_workspace.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "copy workspace")) {
    return 2;
  }

  int failures = 0;
  for (int partitions : {2, 4, 8}) {
    std::vector<cutlass::half_t> expected(kN);
    for (int n = 0; n < kN; ++n) {
      volatile float sum = 0;
      for (int s = 0; s < partitions; ++s) sum = sum + partial_value(s, n);
      expected[n] = cutlass::half_t(float(sum));
    }
    failures += !run_legacy_case(partitions, workspace, output, expected);
    failures += !run_vector_case<1>(partitions, workspace, output, expected);
    failures += !run_vector_case<2>(partitions, workspace, output, expected);
    failures += !run_vector_case<4>(partitions, workspace, output, expected);
    failures += !run_vector_case<8>(partitions, workspace, output, expected);
    failures +=
        !run_production_fast_case(partitions, workspace, output, expected);
    switch (partitions) {
      case 2: failures += !run_volatile_case<2>(workspace, output, expected); break;
      case 4: failures += !run_volatile_case<4>(workspace, output, expected); break;
      case 8: failures += !run_volatile_case<8>(workspace, output, expected); break;
      default: ++failures; break;
    }
  }

  cudaFree(output);
  cudaFree(workspace);
  if (failures != 0) {
    std::fprintf(stderr, "[l194] FAIL: %d case(s)\n", failures);
    return 1;
  }
  std::printf(
      "[l194] PASS: legacy, 12 vector topology, 3 production-fast and 3 "
      "fused-volatile fixed-order cases are raw-bit exact\n");
  return 0;
}
