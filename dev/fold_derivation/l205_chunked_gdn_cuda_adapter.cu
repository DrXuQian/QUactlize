// Local RTX correctness adapter for the exact production collective body.
//
// RTX 5090 exposes 101376 bytes of opt-in shared memory per block while the
// PPU specialization owns 139776 bytes.  A normal CUDA launch therefore
// cannot instantiate the exact body.  This test-only adapter places the
// unchanged SharedStorage object in global memory and passes it to the same
// PpuChunkedGdnKernel::operator().  Every scalar CUDA fallback instruction,
// synchronization point and BF16 materialization is unchanged; only the
// address space used by the scratch ledger differs.  Performance is out of
// scope.  The PPU ABI arm separately launches the shipping shared-memory path.

#include <cstdint>

#include <cuda_runtime.h>

#include "cutlass/bfloat16.h"
#include "quactlize_ppu_linear_attention.h"
#include "quactlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_kernel.cuh"

namespace {

using Element = cutlass::bfloat16_t;
using Traits = cutlass::linear_attention::PpuChunkedGdnTraits<64, 128, 128>;
using Arguments =
    cutlass::linear_attention::PpuChunkedGdnArguments<Element, Element, float>;
using Kernel = cutlass::linear_attention::PpuChunkedGdnKernel<Arguments, Traits>;

__global__ void chunked_gdn_global_scratch_kernel(
    Arguments args, typename Kernel::SharedStorage* scratch) {
  Kernel kernel;
  kernel(args, reinterpret_cast<char*>(&scratch[blockIdx.x]));
}

}  // namespace

extern "C" int quactlize_ppu_chunked_gdn_fwd_bf16_v1(
    std::uint16_t const* q,
    std::uint16_t const* k,
    std::uint16_t const* v,
    float const* gamma_log2_cumsum,
    float const* beta,
    float const* initial_state,
    std::uint16_t* output,
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

  auto const admission = Kernel::Collective::argument_status(args);
  if (admission != cutlass::linear_attention::PpuChunkedGdnStatus::kSuccess) {
    return int(admission);
  }

  int const blocks = Traits::ChunkSize == 64
      ? args.problem.num_sequences * args.problem.num_v_heads
      : 0;
  if (blocks <= 0) return QUACTLIZE_PPU_CHUNKED_GDN_INVALID_PROBLEM;
  typename Kernel::SharedStorage* scratch = nullptr;
  if (cudaMalloc(&scratch, std::size_t(blocks) * sizeof(*scratch)) != cudaSuccess) {
    return QUACTLIZE_PPU_CHUNKED_GDN_RUNTIME_ERROR;
  }
  cudaStream_t const cuda_stream = static_cast<cudaStream_t>(stream);
  chunked_gdn_global_scratch_kernel<<<blocks, Kernel::MaxThreadsPerBlock, 0, cuda_stream>>>(
      args, scratch);
  cudaError_t status = cudaGetLastError();
  if (status == cudaSuccess) status = cudaStreamSynchronize(cuda_stream);
  cudaError_t const free_status = cudaFree(scratch);
  if (status == cudaSuccess) status = free_status;
  return status == cudaSuccess ? QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS
                               : QUACTLIZE_PPU_CHUNKED_GDN_RUNTIME_ERROR;
}
