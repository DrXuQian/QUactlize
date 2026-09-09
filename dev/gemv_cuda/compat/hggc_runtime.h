#pragma once
#include <cuda_runtime.h>
// Real CUDA calls, not the no-op stubs used by host-only layout oracles.
using hggcStream_t = cudaStream_t;
using hggcError_t = cudaError_t;
constexpr auto hggcSuccess = cudaSuccess;
inline hggcError_t hggcGetLastError() { return cudaGetLastError(); }
