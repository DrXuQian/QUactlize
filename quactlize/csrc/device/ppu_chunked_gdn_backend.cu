// Device-library entry for the pure C++ PPU chunked-GDN forward kernel.

#include <cstdint>

#include "cutlass/bfloat16.h"
#include "cutlass/device_kernel.h"
#include "gemv_lowbit/gemv_rt.hpp"
#include "quactlize_ppu_linear_attention.h"
#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_kernel.cuh"

namespace {

using Element = cutlass::bfloat16_t;
using Traits = cutlass::linear_attention::PpuChunkedGdnTraits<64, 128, 128>;
using Arguments = cutlass::linear_attention::PpuChunkedGdnArguments<Element, Element, float>;
using Kernel = cutlass::linear_attention::PpuChunkedGdnKernel<Arguments, Traits>;
using Admission = cutlass::linear_attention::PpuChunkedGdnStatus;

static_assert(int(Admission::kSuccess) == QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS &&
                  int(Admission::kNullPointer) == QUACTLIZE_PPU_CHUNKED_GDN_NULL_POINTER &&
                  int(Admission::kInvalidProblem) == QUACTLIZE_PPU_CHUNKED_GDN_INVALID_PROBLEM &&
                  int(Admission::kUnsupportedHeadDimension) ==
                      QUACTLIZE_PPU_CHUNKED_GDN_UNSUPPORTED_HEAD_DIMENSION &&
                  int(Admission::kUnsupportedChunkSize) ==
                      QUACTLIZE_PPU_CHUNKED_GDN_UNSUPPORTED_CHUNK_SIZE &&
                  int(Admission::kUnsupportedHeadMapping) ==
                      QUACTLIZE_PPU_CHUNKED_GDN_UNSUPPORTED_HEAD_MAPPING &&
                  int(Admission::kInvalidSequenceLayout) ==
                      QUACTLIZE_PPU_CHUNKED_GDN_INVALID_SEQUENCE_LAYOUT &&
                  int(Admission::kMisalignedPointer) ==
                      QUACTLIZE_PPU_CHUNKED_GDN_MISALIGNED_POINTER,
              "public C status values must match the CUTLASS admission ABI");

int configure_dynamic_shared_memory() {
  constexpr int bytes = int(sizeof(typename Kernel::SharedStorage));
  if (bytes < (48 << 10)) return QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS;
#if defined(__HGGCCC__)
  hggcError_t const result = hggcFuncSetAttribute(
      cutlass::device_kernel<Kernel>,
      hggcFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (result != hggcSuccess) {
    (void)hggcGetLastError();
    return QUACTLIZE_PPU_CHUNKED_GDN_RUNTIME_ERROR;
  }
#else
  cudaError_t const result = cudaFuncSetAttribute(
      cutlass::device_kernel<Kernel>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (result != cudaSuccess) {
    (void)cudaGetLastError();
    return QUACTLIZE_PPU_CHUNKED_GDN_RUNTIME_ERROR;
  }
#endif
  return QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS;
}

}  // namespace

extern "C" int quactlize_ppu_chunked_gdn_fwd_bf16_v1(
    uint16_t const* q,
    uint16_t const* k,
    uint16_t const* v,
    float const* gamma_log2_cumsum,
    float const* beta,
    float const* initial_state,
    uint16_t* output,
    float* final_state,
    quactlize_ppu_chunked_gdn_problem_v1 const* problem,
    float scale,
    void* stream) {
  if (problem == nullptr) return QUACTLIZE_PPU_CHUNKED_GDN_NULL_POINTER;
  if (problem->schema_version != QUACTLIZE_PPU_CHUNKED_GDN_SCHEMA_V1) {
    return QUACTLIZE_PPU_CHUNKED_GDN_INVALID_PROBLEM;
  }

  Arguments args{};
  args.q = reinterpret_cast<Element const*>(q);
  args.k = reinterpret_cast<Element const*>(k);
  args.v = reinterpret_cast<Element const*>(v);
  args.gamma_log2_cumsum = gamma_log2_cumsum;
  args.beta = beta;
  args.initial_state = initial_state;
  args.output = reinterpret_cast<Element*>(output);
  args.final_state = final_state;
  args.problem = {
      problem->total_tokens,
      problem->num_sequences,
      problem->sequence_length,
      problem->num_qk_heads,
      problem->num_v_heads,
      problem->head_size_k,
      problem->head_size_v,
      problem->chunk_size,
  };
  args.scale = scale;

  auto const status = Kernel::Collective::argument_status(args);
  if (status != cutlass::linear_attention::PpuChunkedGdnStatus::kSuccess) {
    return int(status);
  }
  if (!Kernel::can_implement(args)) {
    return QUACTLIZE_PPU_CHUNKED_GDN_INVALID_PROBLEM;
  }
  int const smem_status = configure_dynamic_shared_memory();
  if (smem_status != QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS) return smem_status;

  ppu_gemv::rt_clear_error();
  auto const params = Kernel::to_underlying_arguments(args, nullptr);
#if defined(__HGGCCC__)
  hggcStream_t const launch_stream = static_cast<hggcStream_t>(stream);
#else
  cudaStream_t const launch_stream = static_cast<cudaStream_t>(stream);
#endif
  cutlass::device_kernel<Kernel>
      <<<Kernel::get_grid_shape(params), Kernel::get_block_shape(),
         sizeof(typename Kernel::SharedStorage), launch_stream>>>(params);
  return ppu_gemv::rt_check_launch("PPU chunked GDN BF16 forward enqueue")
      ? QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS
      : QUACTLIZE_PPU_CHUNKED_GDN_RUNTIME_ERROR;
}
