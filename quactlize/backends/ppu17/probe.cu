#include <cuda_runtime.h>
#include "target.cuh"

__global__ void q17_compiler_target(unsigned *target) {
#if defined(__CUDA_ARCH__)
  target[0] = __CUDA_ARCH__;
  target[1] = 1; // target.cuh requires compiler-provided SM90A feature macros.
#ifdef Q17_NATIVE
  target[2] = ACOMPUTE_VERSION;
#else
  target[2] = 0;
#endif
#endif
}
