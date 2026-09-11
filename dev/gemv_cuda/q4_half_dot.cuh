// Arithmetic control only: four-product FP16 partials within each Q4 group,
// then FP32 accumulation between groups. This is NOT the FP32 shipping path.
#include "q4_native.cuh"

namespace quactlize::dev::q4_native {
__device__ __forceinline__ __half2 half_activation_pair(void const* a, int64_t offset, int type) {
    if (type == 0) {
        auto p=static_cast<__half const*>(a)+offset;
        if (!(uintptr_t(p)&3)) return *reinterpret_cast<__half2 const*>(p);
        return __halves2half2(p[0],p[1]);
    }
    auto p=static_cast<float const*>(a)+offset;
    return __floats2half2_rn(p[0],p[1]);
}

template<int Slot>
__device__ __forceinline__ void half_dot_slot(void const* a, int64_t offset, int type,
    uint32_t const (&words)[8], __half2 scale, __half2 zero, __half2 (&partial)[8]) {
    #pragma unroll
    for (int r=0;r<8;r+=2) {
        __half2 const av=half_activation_pair(a,offset+Slot*8+r,type);
        partial[r]=__hfma2(__hfma2(codes<Slot>(words[r]),scale,zero),
                          __half2half2(__low2half(av)),partial[r]);
        partial[r+1]=__hfma2(__hfma2(codes<Slot>(words[r+1]),scale,zero),
                            __half2half2(__high2half(av)),partial[r+1]);
    }
}
} // namespace quactlize::dev::q4_native
