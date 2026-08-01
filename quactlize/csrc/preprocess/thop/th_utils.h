// SHIM for TensorRT-LLM's thop/th_utils.h. The port needs three things from it, all small: a typed pointer out of a
// torch Tensor, and the CHECK_CPU / CHECK_CONTIGUOUS argument guards. Everything else it uses -- TORCH_CHECK,
// at::ScalarType -- comes from torch itself.
//
// The guards are spelled out rather than reduced to TORCH_CHECK at each call site because their MESSAGE is the
// useful part: these functions rearrange bytes on the host, so a device tensor or a non-contiguous one is a caller
// mistake that should say which argument and why, not a segfault inside a memcpy.
#pragma once
#include <torch/extension.h>
// `half` and `__nv_bfloat16` unqualified: the ported sources name CUDA's host-visible half types directly. On a host
// build there is no CUDA header in scope, so they come from cuda_fp16.h when it is available and from torch's own
// scalar types otherwise -- the preprocessing only ever memcpys through these, never does arithmetic on them, so the
// only requirement is that the size is right.
#if defined(__has_include)
#  if __has_include(<cuda_fp16.h>)
#    include <cuda_fp16.h>
#    include <cuda_bf16.h>
#    define QUACTLIZE_HAVE_CUDA_FP16 1
#  endif
#endif
#if !defined(QUACTLIZE_HAVE_CUDA_FP16)
using half = c10::Half;
using __nv_bfloat16 = c10::BFloat16;
static_assert(sizeof(half) == 2 && sizeof(__nv_bfloat16) == 2, "the port memcpys through these; size must match");
#endif

namespace torch_ext {
template <typename T>
inline T* get_ptr(torch::Tensor const& t) { return reinterpret_cast<T*>(t.data_ptr()); }
}  // namespace torch_ext

#define CHECK_CPU(x)                                                                                   \
  TORCH_CHECK((x).device().is_cpu(), #x " must live on the CPU: this is a host-side byte rearrangement")
#define CHECK_CONTIGUOUS(x)                                                                            \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous: the rearrangement walks the buffer directly")
#define CHECK_CPU_INPUT(x, dt)                                                                         \
  CHECK_CPU(x); CHECK_CONTIGUOUS(x); TORCH_CHECK((x).scalar_type() == (dt), #x " has the wrong dtype")
#define CHECK_TH_CUDA(x)  TORCH_CHECK((x).is_cuda(), #x " must be a device tensor")
#define CHECK_INPUT(x, dt)                                                                             \
  CHECK_CONTIGUOUS(x); TORCH_CHECK((x).scalar_type() == (dt), #x " has the wrong dtype")
