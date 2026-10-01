#pragma once
#include "q4_s1_activation.cuh"
#include <hggc_bf16.h>

namespace quactlize::execution::simt {
// Compute and storage are independent. Existing reader instantiations use
// Compute=0 and therefore keep their original F16 conversion instructions.
template<int Input, int Compute>
struct Activation;

template<int Input>
struct Activation<Input, 0> : q4_s1::Activation<Input> {
    using Base = q4_s1::Activation<Input>;
    using Scalar = typename Base::Scalar;
    __device__ __forceinline__ explicit Activation(Scalar const* p) : Base{p} {}
    __device__ __forceinline__ uint32_t load2(int64_t i) const {
        __half2_raw value;
        if constexpr(Input==0) value=*reinterpret_cast<__half2 const*>(this->ptr+i);
        else {
            float2 x=*reinterpret_cast<float2 const*>(this->ptr+i);
            value=__floats2half2_rn(x.x,x.y);
        }
        return uint32_t(value.x)|(uint32_t(value.y)<<16);
    }
};

template<int Input>
struct Activation<Input, 1> {
    static_assert(Input>=0 && Input<=2);
    using Scalar = std::conditional_t<Input==1,float,uint16_t>;
    Scalar const* ptr;
    __device__ __forceinline__ explicit Activation(Scalar const* p) : ptr(p) {}
    __device__ __forceinline__ static uint32_t rounded_pair(float x,float y) {
        __ppu_bfloat162 value=__floats2bfloat162_rn(x,y);
        return reinterpret_cast<uint32_t const&>(value);
    }
    __device__ __forceinline__ uint32_t load2(int64_t i) const {
        if constexpr(Input==2) return *reinterpret_cast<uint32_t const*>(ptr+i);
        else {
            float2 value;
            if constexpr(Input==1) value=*reinterpret_cast<float2 const*>(ptr+i);
            else value=__half22float2(*reinterpret_cast<__half2 const*>(ptr+i));
            return rounded_pair(value.x,value.y);
        }
    }
    __device__ __forceinline__ uint2 load4(int64_t i) const {
        if constexpr(Input==2) return *reinterpret_cast<uint2 const*>(ptr+i);
        else if constexpr(Input==1) {
            float4 value=*reinterpret_cast<float4 const*>(ptr+i);
            return make_uint2(rounded_pair(value.x,value.y),rounded_pair(value.z,value.w));
        } else return make_uint2(load2(i),load2(i+2));
    }
    __device__ __forceinline__ uint4 load8(int64_t i) const {
        if constexpr(Input==2) return *reinterpret_cast<uint4 const*>(ptr+i);
        else {
            uint2 lo=load4(i),hi=load4(i+4);
            return make_uint4(lo.x,lo.y,hi.x,hi.y);
        }
    }
    __device__ __forceinline__ float4 values4(int64_t i) const {
        uint2 bits=load4(i);
        return make_float4(__uint_as_float(bits.x<<16),__uint_as_float(bits.x&0xffff0000u),
                           __uint_as_float(bits.y<<16),__uint_as_float(bits.y&0xffff0000u));
    }
};
} // namespace quactlize::execution::simt
