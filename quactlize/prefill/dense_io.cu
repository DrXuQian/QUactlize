#include "dense_api.h"
#include "dense_io_layout.hpp"
#include <hggc_runtime.h>
#include <hggc_fp16.h>
#include <hggc_bf16.h>
#include <climits>
#include "../execution/validation.hpp"

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

template<bool BF16,bool Input,bool Flat>
__global__ void dense_convert_vec8(void const* src,void* dst,int width,int64_t stride,
                                  int64_t count,int const* error) {
    using D=DenseConvert<BF16>;
    int64_t const i=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)*8;
    int64_t const end=Flat ? count:width;
    if (i>=end) return;
    int64_t const contiguous=Flat ? i:int64_t(blockIdx.y)*width+i;
    int64_t const strided=Flat ? i:int64_t(blockIdx.y)*stride+i;
    bool const bad=!Input && error && *error;
    if (i+8<=end) {
        if constexpr(Input) {
            float4 const a=reinterpret_cast<float4 const*>(static_cast<float const*>(src)+strided)[0];
            float4 const b=reinterpret_cast<float4 const*>(static_cast<float const*>(src)+strided)[1];
            union Packed { uint4 bits;typename D::Pair p[4]; } v;
            v.p[0]=D::pack2(make_float2(a.x,a.y));v.p[1]=D::pack2(make_float2(a.z,a.w));
            v.p[2]=D::pack2(make_float2(b.x,b.y));v.p[3]=D::pack2(make_float2(b.z,b.w));
            *reinterpret_cast<uint4*>(static_cast<typename D::Scalar*>(dst)+contiguous)=v.bits;
        } else {
            union Packed { uint4 bits;typename D::Pair p[4]; } v;
            v.bits=*reinterpret_cast<uint4 const*>(static_cast<typename D::Scalar const*>(src)+contiguous);
            float2 a=D::unpack2(v.p[0]),b=D::unpack2(v.p[1]);
            float2 c=D::unpack2(v.p[2]),d=D::unpack2(v.p[3]);
            if (bad) a=b=c=d=make_float2(__int_as_float(0x7fc00000),__int_as_float(0x7fc00000));
            auto out=reinterpret_cast<float4*>(static_cast<float*>(dst)+strided);
            out[0]=make_float4(a.x,a.y,b.x,b.y);out[1]=make_float4(c.x,c.y,d.x,d.y);
        }
    } else {
        for (int j=0;i+j<end;++j) {
            if constexpr(Input) static_cast<typename D::Scalar*>(dst)[contiguous+j]=D::pack(static_cast<float const*>(src)[strided+j]);
            else static_cast<float*>(dst)[strided+j]=bad ? __int_as_float(0x7fc00000):D::unpack(static_cast<typename D::Scalar const*>(src)[contiguous+j]);
        }
    }
}
template<bool BF16,bool Input>
__global__ void dense_convert_scalar(void const* src,void* dst,int width,int64_t stride,int const* error) {
    using D=DenseConvert<BF16>;int64_t row=blockIdx.x;
    for (int col=threadIdx.x;col<width;col+=blockDim.x) {
        if constexpr(Input) static_cast<typename D::Scalar*>(dst)[row*width+col]=D::pack(static_cast<float const*>(src)[row*stride+col]);
        else static_cast<float*>(dst)[row*stride+col]=error && *error ? __int_as_float(0x7fc00000):D::unpack(static_cast<typename D::Scalar const*>(src)[row*width+col]);
    }
}
template<bool BF16,bool Input,bool Flat>
__global__ void dense_convert_stripe4(void const* src,void* dst,int width,int64_t stride,
                                    int64_t count,int const* error) {
    using D=DenseConvert<BF16>;
    uint64_t const end=Flat ? uint64_t(count):uint64_t(width);
    uint64_t const row=Flat ? 0:blockIdx.y;
    bool const bad=!Input && error && *error;
    // Load the two independent chunks before conversion/publication. No
    // shared staging or cross-lane shuffle. Tails never vector-read OOB.
    float4 floats[2];uint2 packed[2];
    #pragma unroll
    for(unsigned part=0;part<2;++part) {
        uint64_t i=DenseStripe4<256>::element(blockIdx.x,threadIdx.x,part);
        if(i+4<=end) {
            if constexpr(Input) floats[part]=*reinterpret_cast<float4 const*>(
                static_cast<float const*>(src)+row*stride+i);
            else packed[part]=*reinterpret_cast<uint2 const*>(
                static_cast<typename D::Scalar const*>(src)+row*width+i);
        }
    }
    #pragma unroll
    for(unsigned part=0;part<2;++part) {
        uint64_t i=DenseStripe4<256>::element(blockIdx.x,threadIdx.x,part);
        if(i+4<=end) {
            union Bits {uint2 bits;typename D::Pair pairs[2];} v;
            if constexpr(Input) {
                auto a=floats[part];v.pairs[0]=D::pack2(make_float2(a.x,a.y));
                v.pairs[1]=D::pack2(make_float2(a.z,a.w));
                *reinterpret_cast<uint2*>(static_cast<typename D::Scalar*>(dst)+row*width+i)=v.bits;
            } else {
                v.bits=packed[part];auto a=D::unpack2(v.pairs[0]),b=D::unpack2(v.pairs[1]);
                if(bad)a=b=make_float2(__int_as_float(0x7fc00000),__int_as_float(0x7fc00000));
                *reinterpret_cast<float4*>(static_cast<float*>(dst)+row*stride+i)=make_float4(a.x,a.y,b.x,b.y);
            }
        } else for(unsigned j=0;j<4 && i+j<end;++j) {
            if constexpr(Input) static_cast<typename D::Scalar*>(dst)[row*width+i+j]=D::pack(static_cast<float const*>(src)[row*stride+i+j]);
            else static_cast<float*>(dst)[row*stride+i+j]=bad?__int_as_float(0x7fc00000):D::unpack(static_cast<typename D::Scalar const*>(src)[row*width+i+j]);
        }
    }
}
template<bool BF16,bool Input>
void dense_convert(int vector,void const* src,void* dst,int rows,int width,int64_t stride,int const* error,hggcStream_t stream) {
    bool aligned=!(uintptr_t(src)%16 || uintptr_t(dst)%16);
    if (vector==2 && aligned && stride==width) {
        int64_t count=int64_t(rows)*width;
        dense_convert_stripe4<BF16,Input,true><<<(count+2047)/2048,256,0,stream>>>(src,dst,width,stride,count,error);
    } else if(vector==2 && aligned && width%4==0 && stride%4==0) {
        dense_convert_stripe4<BF16,Input,false><<<dim3((width+2047)/2048,rows),256,0,stream>>>(src,dst,width,stride,0,error);
    } else if (vector==1 && aligned && stride==width) {
        int64_t count=int64_t(rows)*width;
        dense_convert_vec8<BF16,Input,true><<<(count+2047)/2048,256,0,stream>>>(src,dst,width,stride,count,error);
    } else if (vector==1 && aligned && width%8==0 && stride%4==0) {
        dense_convert_vec8<BF16,Input,false><<<dim3((width+2047)/2048,rows),256,0,stream>>>(src,dst,width,stride,0,error);
    } else dense_convert_scalar<BF16,Input><<<rows,256,0,stream>>>(src,dst,width,stride,error);
}
}
extern "C" int quactlize_dense_io_v1(int storage,int input,int vector,void const* src,void* dst,
        int rows,int width,int64_t stride,int const* error,void* stream) {
    if ((storage!=0 && storage!=1) || (input!=0 && input!=1) || vector<0 || vector>2 ||
        !src || !dst || rows<=0 || rows>65535 || width<=0 || stride<width ||
        uint64_t(stride)>uint64_t(INT64_MAX)/4/rows ||
        (uint64_t(rows)*width+2047)/2048>INT_MAX ||
        uintptr_t(src)%(input?4:2) || uintptr_t(dst)%(input?2:4) || uintptr_t(error)%4) return QKG_INVALID;
    uint64_t dense=uint64_t(rows)*width,span=uint64_t(rows-1)*stride+width;
    uint64_t src_bytes=input?span*4:dense*2,dst_bytes=input?dense*2:span*4;
    using quactlize::execution::overlap;
    if (!quactlize::execution::span(uintptr_t(src),src_bytes) ||
        !quactlize::execution::span(uintptr_t(dst),dst_bytes) ||
        overlap(uintptr_t(src),src_bytes,uintptr_t(dst),dst_bytes) ||
        (error && (overlap(uintptr_t(error),4,uintptr_t(src),src_bytes) ||
                   overlap(uintptr_t(error),4,uintptr_t(dst),dst_bytes)))) return QKG_INVALID;
    using namespace quactlize::prefill;auto s=static_cast<hggcStream_t>(stream);
    if (storage) {
        if (input) dense_convert<true,true>(vector,src,dst,rows,width,stride,error,s);
        else dense_convert<true,false>(vector,src,dst,rows,width,stride,error,s);
    } else {
        if (input) dense_convert<false,true>(vector,src,dst,rows,width,stride,error,s);
        else dense_convert<false,false>(vector,src,dst,rows,width,stride,error,s);
    }
    return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME;
}
