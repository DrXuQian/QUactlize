#pragma once
#include "q4_native.cuh"

namespace quactlize::dev::q4_native {
__device__ __forceinline__ ScaleZero aligned_scale_zero(uint4 m,unsigned group) {
    uint64_t const run=(group&4) ? uint64_t(m.z>>16)|(uint64_t(m.w)<<16)
                                : uint64_t(m.y)|(uint64_t(m.z&0xffff)<<32);
    unsigned const shift=6*(group&3);
    unsigned const sc=unsigned(run>>shift)&63, mn=unsigned(run>>(24+shift))&63;
    __half2_raw codes_raw,header_raw;
    codes_raw.x=uint16_t(0x6400|sc);codes_raw.y=uint16_t(0x6400|mn);
    header_raw.x=uint16_t(m.x);header_raw.y=uint16_t(m.x>>16);
    __half2 const products=__hmul2(__hsub2(__half2(codes_raw),__float2half2_rn(1024.f)),__half2(header_raw));
    __half const scale=__low2half(products),zero=__hneg(__high2half(products));
    return {scale,__float2half_rn(__half2float(zero)+8.f*__half2float(scale))};
}
__device__ __forceinline__ uint4 aligned_unit(uint8_t const* p) {
    return *reinterpret_cast<uint4 const*>(p);
}
__device__ __forceinline__ uint32_t aligned_word(uint16_t const* p) {
    return *reinterpret_cast<uint32_t const*>(p);
}
template<int Type>
__device__ __forceinline__ float4 aligned_activation(void const* base,int64_t offset) {
    if constexpr (Type==0) {
        auto p=static_cast<__half const*>(base)+offset;
        float2 const a=__half22float2(*reinterpret_cast<__half2 const*>(p));
        float2 const b=__half22float2(*reinterpret_cast<__half2 const*>(p+2));
        return make_float4(a.x,a.y,b.x,b.y);
    } else {
        float4 v=*reinterpret_cast<float4 const*>(static_cast<float const*>(base)+offset);
        v.x=__half2float(__float2half_rn(v.x)); v.y=__half2float(__float2half_rn(v.y));
        v.z=__half2float(__float2half_rn(v.z)); v.w=__half2float(__float2half_rn(v.w));
        return v;
    }
}
template<int Slot,int Type>
__device__ __forceinline__ void aligned_dot_slot(void const* a,int64_t group_offset,
    uint32_t const (&words)[8],__half2 scale,__half2 zero,float (&x)[4],float (&y)[4]) {
    #pragma unroll
    for (int first=0;first<8;first+=4) {
        float4 const v=aligned_activation<Type>(a,group_offset+8*Slot+first);
        float const av[4]={v.x,v.y,v.z,v.w};
        #pragma unroll
        for (int i=0;i<4;++i) {
            float2 const w=__half22float2(__hfma2(codes<Slot>(words[first+i]),scale,zero));
            x[i]=fmaf(av[i],w.x,x[i]);y[i]=fmaf(av[i],w.y,y[i]);
        }
    }
}
} // namespace quactlize::dev::q4_native
