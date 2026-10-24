// Contiguous load/store body adapted from llama.cpp convert-ppu.cuh,
// d30de5bbe45fabb23bcc7a6832d84f0046c16f23. No llama build dependency.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once
#include <hggc_runtime.h>
#include <hggc_fp16.h>
#include <hggc_bf16.h>
#include <type_traits>

namespace quactlize::prefill {
template<bool BF16> struct DenseConvert;
template<> struct DenseConvert<false> {
    using Scalar=__half;using Pair=__half2;
    __device__ static Scalar pack(float x) { return __float2half_rn(x); }
    __device__ static float unpack(Scalar x) { return __half2float(x); }
    __device__ static Pair pack2(float2 x) { return __float22half2_rn(x); }
    __device__ static float2 unpack2(Pair x) { return __half22float2(x); }
};
template<> struct DenseConvert<true> {
    using Scalar=__ppu_bfloat16;using Pair=__ppu_bfloat162;
    __device__ static Scalar pack(float x) { return __float2bfloat16_rn(x); }
    __device__ static float unpack(Scalar x) { return __bfloat162float(x); }
    __device__ static Pair pack2(float2 x) { return __float22bfloat162_rn(x); }
    __device__ static float2 unpack2(Pair x) { return __bfloat1622float2(x); }
};
// DenseConvert is the existing direct F16/BF16 intrinsic policy. CheckError is
// instantiated only for output with a non-null flag; never discard its contract
// to obtain the native unchecked fast path. Eligibility is checked on the host.
template<bool BF16,bool Input,bool CheckError>
__global__ void dense_convert_cont(
        std::conditional_t<Input,float,typename DenseConvert<BF16>::Scalar> const* __restrict__ x,
        std::conditional_t<Input,typename DenseConvert<BF16>::Scalar,float>* __restrict__ y,
        int n8,int const* error) {
    using D=DenseConvert<BF16>;
    int const c=blockIdx.x*int(blockDim.x)+threadIdx.x;
    if(c>=n8)return;
    if constexpr(!Input && CheckError) {
        if(*error) {
            float const nan=__int_as_float(0x7fc00000);
            reinterpret_cast<float4*>(y)[2*c]=make_float4(nan,nan,nan,nan);
            reinterpret_cast<float4*>(y)[2*c+1]=make_float4(nan,nan,nan,nan);
            return;
        }
    }
    if constexpr(Input) {
        float4 const a=reinterpret_cast<float4 const*>(x)[2*c];
        float4 const b=reinterpret_cast<float4 const*>(x)[2*c+1];
        alignas(16) typename D::Pair h[4]={
            D::pack2(make_float2(a.x,a.y)),D::pack2(make_float2(a.z,a.w)),
            D::pack2(make_float2(b.x,b.y)),D::pack2(make_float2(b.z,b.w))};
        reinterpret_cast<uint4*>(y)[c]=*reinterpret_cast<uint4 const*>(h);
    } else {
        uint4 const v=reinterpret_cast<uint4 const*>(x)[c];
        auto const h=reinterpret_cast<typename D::Pair const*>(&v);
        float2 const a=D::unpack2(h[0]),b=D::unpack2(h[1]);
        float2 const d=D::unpack2(h[2]),e=D::unpack2(h[3]);
        reinterpret_cast<float4*>(y)[2*c]=make_float4(a.x,a.y,b.x,b.y);
        reinterpret_cast<float4*>(y)[2*c+1]=make_float4(d.x,d.y,e.x,e.y);
    }
}
}
