#pragma once

#include <cuda_runtime.h>
#include "target.cuh"

namespace quactlize::ppu17 {

#if defined(Q17_ENABLE_PDL)
inline constexpr bool pdl_compiled = true;
#else
inline constexpr bool pdl_compiled = false;
#endif

// Grid dependencies are not CTA-to-CTA data readiness. Every thread that
// accesses a predecessor's storage must pass this wait, including readers of
// grouped offsets. The runtime intrinsic also supplies the compiler memory
// clobber; a launch-completion hint by itself supplies neither guarantee.
template <bool Pdl> __device__ __forceinline__ void wait_for_predecessors() {
  if constexpr (Pdl) {
    static_assert(!Pdl || pdl_compiled, "PDL specialization requires Q17_ENABLE_PDL");
    cudaGridDependencySynchronize();
  }
}

template <bool Pdl> __device__ __forceinline__ void release_successors() {
  if constexpr (Pdl) {
    static_assert(!Pdl || pdl_compiled, "PDL specialization requires Q17_ENABLE_PDL");
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

// Ordinary specializations retain the ordinary launch and contain no GDC
// instructions. PDL is chosen once when preparing a plan, never by a device
// branch or a process-global environment variable.
template <bool Pdl, class Kernel, class... Args>
inline int enqueue_kernel(Kernel kernel, dim3 grid, dim3 block,
                          size_t shared_bytes, cudaStream_t stream, Args... args) {
  if constexpr (Pdl) {
    static_assert(!Pdl || pdl_compiled, "PDL launch requires Q17_ENABLE_PDL");
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = shared_bytes;
    config.stream = stream;
    config.attrs = &attribute;
    config.numAttrs = 1;
    auto status = cudaLaunchKernelEx(&config, kernel, args...);
    if (status != cudaSuccess) return int(status);
  } else {
    kernel<<<grid, block, shared_bytes, stream>>>(args...);
  }
  return int(cudaGetLastError());
}

// Reuse CUTLASS's own wait/release sites and cluster-aware launch adapter.
// This helper is for cooperative collectives with audited GDC hooks, not a
// way to enable PDL on an arbitrary third-party kernel.
template <class Gemm>
inline auto enqueue_cooperative(Gemm &gemm, cudaStream_t stream) {
#if defined(Q17_ENABLE_PDL)
# if !defined(CUTLASS_ENABLE_GDC_FOR_SM90)
#  error "PDL cooperative launch requires CUTLASS_ENABLE_GDC_FOR_SM90"
# endif
  return gemm.run(stream, nullptr, true);
#else
  return gemm.run(stream);
#endif
}

} // namespace quactlize::ppu17
