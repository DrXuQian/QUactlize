// L142 -- instantiate the production classic-aligned two-source consumer.
//
// L138 proves the real CuTe delivery algebra.  This file closes the other
// half of the seam: the actual dense policy must instantiate the WK4 branch
// in quactlize_mma_mixed_input.hpp, not merely parse beside it.  It includes
// the benchmark's real Cfg stack so the collective, scheduler and epilogue
// types are exactly the ones the box target will use.
#define LOWBIT_DENSE_UNIT_BUILD 1
#define DENSE_MARLIN_AB 1
#define DENSE_STREAMK_AB 1
#define BENCH_GS 128
#define BENCH_TSK 64
#define DENSE_AB_BITS 4
#define DENSE_AB_ARTIFACT_TK 64

#include "test_lowbit_dense_bench.cu"

#if defined(L142_UNPROVED_WK2)
using L142Cfg = Cfg<128, 16, 128, 128, 16, 64, 4, 64>;
#else
using L142Cfg = Cfg<128, 16, 128, 128, 16, 64, 4, 32>;
#endif
using L142Kernel = typename L142Cfg::MarlinGemm::GemmKernel;

__global__ void l142_force(L142Kernel::Params params) {
  extern __shared__ char smem[];
  L142Kernel{}(params, smem);
}
