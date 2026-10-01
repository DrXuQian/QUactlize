#pragma once
#include "q4_s1_activation.cuh"
#include "simt_activation.cuh"

namespace quactlize::execution::q4_s1 {
template<int Input,int Compute>
using ComputeActivation = std::conditional_t<Compute==0,Activation<Input>,simt::Activation<Input,1>>;

template<int Compute>
__device__ __forceinline__ float activation_value(uint16_t bits) {
    static_assert(Compute==0 || Compute==1);
    if constexpr(Compute==0) return __half2float(__ushort_as_half(bits));
    else return __uint_as_float(uint32_t(bits)<<16);
}

template<int Input,int Compute,class A>
__device__ __forceinline__ uint16_t activation_bits(A a,int64_t i) {
    if constexpr(Compute==0) return __half_as_ushort(a[i]);
    else if constexpr(Input==2) return a.ptr[i];
    else {
        float value;
        if constexpr(Input==1) value=a.ptr[i];
        else value=__half2float(__ushort_as_half(a.ptr[i]));
        __ppu_bfloat16 const rounded=__float2bfloat16_rn(value);
        uint16_t bits;
        __builtin_memcpy(&bits,&rounded,sizeof(bits));
        return bits;
    }
}

template<int Ignored,int Input>
__device__ __forceinline__ float4 aligned_activation(simt::Activation<Input,1> a,int64_t i) {
    static_assert(Ignored==0);
    return a.values4(i);
}
} // namespace quactlize::execution::q4_s1
