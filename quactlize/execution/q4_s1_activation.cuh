#pragma once
#include <hggc_fp16.h>
#include <hggc_runtime.h>
#include <cstdint>
#include <type_traits>

namespace quactlize::execution::q4_s1 {
// Value-semantic register view. F16 retains the old load widths; F32 uses
// float vectors then rounds in registers. No expanded activation buffer.
template<int Input>
struct Activation {
    static_assert(Input==0 || Input==1);
    using Scalar=typename std::conditional<Input==0,__half,float>::type;
    Scalar const* ptr;
    __device__ __forceinline__ __half operator[](int64_t i) const {
        if constexpr(Input==0) return ptr[i];
        else return __float2half_rn(ptr[i]);
    }
    __device__ __forceinline__ uint2 load4(int64_t i) const {
        if constexpr(Input==0) return *reinterpret_cast<uint2 const*>(ptr+i);
        else {
            float4 const v=*reinterpret_cast<float4 const*>(ptr+i);
            __half2_raw lo=__floats2half2_rn(v.x,v.y), hi=__floats2half2_rn(v.z,v.w);
            return make_uint2(uint32_t(lo.x)|(uint32_t(lo.y)<<16),
                              uint32_t(hi.x)|(uint32_t(hi.y)<<16));
        }
    }
    __device__ __forceinline__ uint4 load8(int64_t i) const {
        if constexpr(Input==0) return *reinterpret_cast<uint4 const*>(ptr+i);
        else {
            uint2 const a=load4(i), b=load4(i+4);
            return make_uint4(a.x,a.y,b.x,b.y);
        }
    }
    __device__ __forceinline__ float4 values4(int64_t i) const {
        if constexpr(Input==0) {
            float2 const a=__half22float2(*reinterpret_cast<__half2 const*>(ptr+i));
            float2 const b=__half22float2(*reinterpret_cast<__half2 const*>(ptr+i+2));
            return make_float4(a.x,a.y,b.x,b.y);
        } else {
            float4 v=*reinterpret_cast<float4 const*>(ptr+i);
            v.x=__half2float(__float2half_rn(v.x)); v.y=__half2float(__float2half_rn(v.y));
            v.z=__half2float(__float2half_rn(v.z)); v.w=__half2float(__float2half_rn(v.w));
            return v;
        }
    }
};
template<int Ignored,int Input>
__device__ __forceinline__ float4 aligned_activation(Activation<Input> a,int64_t i) {
    static_assert(Ignored==0);
    return a.values4(i);
}
} // namespace quactlize::execution::q4_s1
