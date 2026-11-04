#pragma once
// Compile the existing portable producer without changing its byte layout.
// These are runtime spelling aliases only; no architecture is impersonated.
#include <cuda_runtime.h>
using hggcStream_t = cudaStream_t;
using hggcError_t = cudaError_t;
constexpr auto hggcSuccess = cudaSuccess;
inline cudaError_t hggcGetLastError() { return cudaGetLastError(); }
