// NVIDIA-only instruction-shape experiment. The PPU pair reader already has
// native f16x2 assembly; this does not predict an additional PPU speedup.
#include <cuda_fp16.h>
#include "reader.hpp"

namespace quactlize::execution {
template<KType T> struct CudaHalf2Reader : Reader<T> {
  CUTLASS_HOST_DEVICE static uint32_t weight_pair(int raw0,int raw1,gguf_scale::GroupScale s) {
#if defined(__CUDA_ARCH__)
    uint32_t const bits=uint32_t(raw0)|(uint32_t(raw1)<<16)|UINT32_C(0x64006400);
    __half2 codes=__halves2half2(__ushort_as_half(uint16_t(bits)),__ushort_as_half(uint16_t(bits>>16)));
    __half offset=__float2half_rn(float(1024+(Reader<T>::lo_bits==4 ? 8 : 0)));
    codes=__hsub2(codes,__halves2half2(offset,offset));
    __half scale=__ushort_as_half(s.scale.raw()),zero=__ushort_as_half(s.zero.raw());
    __half2 value=__hfma2(codes,__halves2half2(scale,scale),__halves2half2(zero,zero));
    return uint32_t(__half_as_ushort(__low2half(value)))|
        (uint32_t(__half_as_ushort(__high2half(value)))<<16);
#else
    return Reader<T>::weight_pair(raw0,raw1,s);
#endif
  }
};
}

// Reuse the exact production kernel, API, indexing, dot order and reducer.
// Only its Reader alias changes inside this development translation unit.
#define Reader CudaHalf2Reader
#include "../../quactlize/execution/gemv.cu"
#undef Reader
