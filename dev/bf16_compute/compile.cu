#include "quactlize/runtime/kernel_types.cuh"
#include "quactlize/decode/types.cuh"
#include "cutlass/device_kernel.h"

#ifndef BF16_PROBE_Q
#define BF16_PROBE_Q 12
#endif
#ifndef BF16_PROBE_TM
#define BF16_PROBE_TM 8
#endif
#ifndef BF16_PROBE_COMPUTE
#define BF16_PROBE_COMPUTE 1
#endif

using Compute = std::conditional_t<BF16_PROBE_COMPUTE, cutlass::bfloat16_t, cutlass::half_t>;
using Core = quactlize::runtime::DenseTypes<BF16_PROBE_Q, BF16_PROBE_TM, 64, 256,
    BF16_PROBE_TM, 16, 2, 0, 16, Compute>;
using Dense = quactlize::decode::DenseTypes<Core, float, float>;
using Grouped = quactlize::runtime::GroupedTypes<BF16_PROBE_Q, BF16_PROBE_TM, 64, 256,
    BF16_PROBE_TM, 16, 2, 16, false, Compute, true, Compute>;
static_assert(std::is_same_v<typename Core::Mainloop::TiledMma::ValTypeA, Compute>);
static_assert(std::is_same_v<typename Core::Mainloop::TiledMma::ValTypeB, Compute>);
static_assert(std::is_same_v<typename Grouped::Mainloop::TiledMma::ValTypeA, Compute>);
static_assert(std::is_same_v<typename Grouped::Mainloop::TiledMma::ValTypeB, Compute>);
static_assert(std::is_same_v<typename Grouped::Mainloop::ElementScale, cutlass::half_t>);
using HalfReader=cute::PPU0010_TSM_LD_SWZL_M8<cutlass::half_t,16,64,true,false,4>;
using Bf16Reader=cute::PPU0010_TSM_LD_SWZL_M8<cutlass::bfloat16_t,16,64,true,false,4>;
constexpr bool same_projection() {
  for(int lane=0;lane<32;++lane) for(int reg=0;reg<2;++reg)
    for(int col=0;col<64;col+=16)
      if(HalfReader::logical_word_offset(lane,reg,col,0)!=
         Bf16Reader::logical_word_offset(lane,reg,col,0)) return false;
  return HalfReader::kStagePitch==Bf16Reader::kStagePitch &&
         HalfReader::kCubePitch==Bf16Reader::kCubePitch;
}
static_assert(same_projection(), "BF16 changes values, not the b16 A-reader coordinate map");

extern "C" void compile_dense(typename Dense::Shipping::GemmKernel::Params params, hggcStream_t stream) {
  using Kernel = typename Dense::Shipping::GemmKernel;
  cutlass::device_kernel<Kernel><<<1, Kernel::MaxThreadsPerBlock,
      sizeof(typename Kernel::SharedStorage), stream>>>(params);
}
extern "C" void compile_grouped(typename Grouped::Kernel::Params params, hggcStream_t stream) {
  using Kernel = typename Grouped::Kernel;
  cutlass::device_kernel<Kernel><<<1, Kernel::MaxThreadsPerBlock,
      sizeof(typename Kernel::SharedStorage), stream>>>(params);
}
