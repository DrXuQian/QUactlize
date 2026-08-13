// One-launch standalone classic-Marlin profile harness for INBOX 149/156.
//
// bench_marlin.cu launches the same kernel seven times (correctness, warmups,
// timing), which makes an aggregate ACU instruction count ambiguous.  This
// harness deliberately performs allocation/memset setup followed by exactly
// one marlin_cuda() call for M=1,N=K=4096,gs=128.  ACU may also list runtime
// memset operations, but there is exactly one Marlin production kernel row.

#include "marlin_classic_ppu.cuh"

#include <cstdio>
#include <cstdlib>

namespace {
constexpr int kM = 1;
constexpr int kN = 4096;
constexpr int kK = 4096;
constexpr int kGroupSize = 128;
constexpr int kMaxPar = 128;
constexpr int kExpectedMma = 65536;
}

int main() {
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 2;

  size_t const a_i4 = size_t(kM) * kK / 8;
  size_t const b_i4 = size_t(kK / 16) * (kN * 16 / 32);
  size_t const c_i4 = size_t(kM) * kN / 8;
  size_t const scale_h = size_t(kK / kGroupSize) * kN;
  size_t const workspace_i32 = size_t(kN / 128 + 1) * kMaxPar;

  int4 *a = nullptr, *b = nullptr, *c = nullptr, *scales = nullptr;
  int* locks = nullptr;
  if (cudaMalloc(&a, a_i4 * 16) || cudaMalloc(&b, b_i4 * 16) ||
      cudaMalloc(&c, c_i4 * 16) || cudaMalloc(&scales, (scale_h / 8 + 1) * 16) ||
      cudaMalloc(&locks, workspace_i32 * sizeof(int))) {
    std::fprintf(stderr, "MARLIN156 FAIL: allocation\n");
    return 3;
  }
  cudaMemset(a, 1, a_i4 * 16);
  cudaMemset(b, 1, b_i4 * 16);
  cudaMemset(c, 0, c_i4 * 16);
  cudaMemset(scales, 0x3c, (scale_h / 8) * 16);
  cudaMemset(locks, 0, workspace_i32 * sizeof(int));
  if (cudaDeviceSynchronize() != cudaSuccess) return 4;

  int const q = kN / 128;
  int const grid = q >= prop.multiProcessorCount ? q : prop.multiProcessorCount;
  std::printf(
      "MARLIN156 identity=Marlin<256,1,8,8,4,8> shape=%dx%dx%d gs=%d "
      "threads=256 stages=4 min_blocks=2 Q=%d CU=%d expected_grid=%d "
      "expected_vmma=%d\n",
      kM, kN, kK, kGroupSize, q, prop.multiProcessorCount, grid, kExpectedMma);

  int const rc = marlin_classic_ppu::marlin_cuda(
      a, b, c, scales, kM, kN, kK, locks, kGroupSize,
      /*dev=*/0, /*stream=*/0, /*thread_k=*/-1, /*thread_n=*/-1,
      /*sms=*/-1, kMaxPar);
  cudaError_t const sync = cudaDeviceSynchronize();
  std::printf("MARLIN156 launch-count=1 rc=%d sync_code=%d sync=%s\n", rc,
              int(sync), cudaGetErrorString(sync));

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaFree(scales);
  cudaFree(locks);
  return rc || sync != cudaSuccess ? 5 : 0;
}
