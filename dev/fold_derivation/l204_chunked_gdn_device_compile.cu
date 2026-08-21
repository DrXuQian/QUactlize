// L204: instantiate the complete chunked-GDN device body with the shipping
// BF16/C64/K128/V128 type.  It launches nothing; L203 owns algebra and the PPU
// box owns hardware opcodes.  This gate proves the CUDA/CUTLASS body and public
// launch geometry are one compilable type rather than disconnected headers.

#include <cstdio>
#include <cstdint>

#include "cutlass/bfloat16.h"
#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_kernel.cuh"

namespace {

#if defined(L204_PLANT_CHUNK)
using Traits = cutlass::linear_attention::PpuChunkedGdnTraits<32, 128, 128>;
#elif defined(L204_PLANT_HEAD)
using Traits = cutlass::linear_attention::PpuChunkedGdnTraits<64, 64, 128>;
#else
using Traits = cutlass::linear_attention::PpuChunkedGdnTraits<64, 128, 128>;
#endif
using Args = cutlass::linear_attention::PpuChunkedGdnArguments<
    cutlass::bfloat16_t, cutlass::bfloat16_t, float>;
using Kernel = cutlass::linear_attention::PpuChunkedGdnKernel<Args, Traits>;

static_assert(Kernel::Collective::kAllStagesConnected,
              "L204 requires the complete GDN dataflow");
static_assert(Kernel::Collective::kGlobalQkAndKkUseAiuOnPpu0010,
              "L204 requires the proved PPU global-dot AIU route");
static_assert(!Kernel::Collective::kAllMatrixProductsUseAiu,
              "L204 must not overclaim generated-operand AIU coverage");
static_assert(Kernel::MaxThreadsPerBlock == 128,
              "L204 launch geometry changed without a new proof");
static_assert(sizeof(typename Kernel::SharedStorage) <= 262144,
              "L204 exceeds the PPU per-CTA shared-memory budget");

// actlize's generic device_kernel ends in an hgcc-only synclog call, so plain
// nvcc cannot use that wrapper as a portability gate.  This minimal equivalent
// still instantiates the shipping operator and every reachable scalar phase;
// the hgcc backend separately launches the same operator through device_kernel.
__global__ void nvcc_device_kernel(typename Kernel::Params params) {
  extern __shared__ char smem[];
  Kernel{}(params, smem);
}

// The launch expression forces nvcc to instantiate the complete operator.
// main() deliberately does not call it.
void instantiate_device_body(typename Kernel::Params params) {
  nvcc_device_kernel
      <<<Kernel::get_grid_shape(params), Kernel::get_block_shape(),
         sizeof(typename Kernel::SharedStorage)>>>(params);
}

}  // namespace

int main() {
  alignas(16) std::uint16_t qkv[8]{};
  float scalar = 0.0f;
  Args args{};
  args.q = args.k = args.v =
      reinterpret_cast<cutlass::bfloat16_t const*>(qkv);
  args.gamma_log2_cumsum = args.beta = &scalar;
  args.output = reinterpret_cast<cutlass::bfloat16_t*>(qkv);
  args.problem = {65, 1, 65, 1, 1, 128, 128, 64};
  args.scale = 0.5f;

  bool const admitted = Kernel::can_implement(args);
  auto const params = Kernel::to_underlying_arguments(args, nullptr);
  dim3 const grid = Kernel::get_grid_shape(params);
  dim3 const block = Kernel::get_block_shape();
  bool const ok = admitted && grid.x == 1 && grid.y == 1 && grid.z == 1 &&
                  block.x == 128 && Kernel::get_workspace_size(args) == 0;
  std::printf(
      "[l204] %s: device-body=INSTANTIATED C=64 K=128 V=128 threads=%u "
      "shared=%zu all-stages=1 global-dot=PPU-AIU generated-products=SIMT-V1\n",
      ok ? "PASS" : "FAIL", unsigned(block.x),
      sizeof(typename Kernel::SharedStorage));
  return ok ? 0 : 1;
}
