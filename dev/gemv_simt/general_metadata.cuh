#pragma once
#include "q6_sparse.cuh"

namespace quactlize::execution::general_metadata {
using Packet = q6_sparse::Packet;

// Every C4/C8 warp owns <=8 consecutive groups inside one phase; select
// fields for the receiving lane AFTER shuffles. No misaligned vector loads,
// padding or read past a 20/28/36-byte unit is permitted.
template<int Q, class Words>
__host__ __device__ __forceinline__ Packet load_words(Words const& p, unsigned g) {
    static_assert(Q==10 || Q==11 || Q==12 || Q==13 || Q==14);
    if constexpr(Q==10) {
        return (g&8u) ? Packet{p[0],p[3],p[4],0} : Packet{p[0],p[1],p[2],0};
    } else if constexpr(Q==11) {
        unsigned const phase=(g>>3)&3u;
        if(phase==0) return {p[0],p[0],p[1],0};
        if(phase==1) return {p[0],p[2],p[3],0};
        if(phase==2) return {p[3]>>16,p[4],p[5],0};
        return {p[3]>>16,p[5],p[6],0};
    } else if constexpr(Q==14) return q6_sparse::load_words(p,g);
    else return {p[0],p[1],p[2],p[3]};
}

template<int Q>
__device__ __forceinline__ Packet load(uint8_t const* p, unsigned g) {
    if constexpr(Q==12 || Q==13) {
        uint4 const v=*reinterpret_cast<uint4 const*>(p);
        return {v.x,v.y,v.z,v.w};
    } else return load_words<Q>(reinterpret_cast<uint32_t const*>(p),g);
}

__device__ __forceinline__ Packet share(Packet p,int owner) {
    p.x=__shfl_sync(0xffffffffu,p.x,owner);
    p.y=__shfl_sync(0xffffffffu,p.y,owner);
    p.z=__shfl_sync(0xffffffffu,p.z,owner);
    p.w=__shfl_sync(0xffffffffu,p.w,owner);
    return p;
}

template<int Q>
__host__ __device__ __forceinline__ float2 affine(Packet const& p,unsigned g) {
    if constexpr(Q==10) {
        unsigned const shift=4u*(g&7u);
        float const d=__half2float(__ushort_as_half(uint16_t(p.x)));
        float const dm=__half2float(__ushort_as_half(uint16_t(p.x>>16)));
        return make_float2(d*float((p.y>>shift)&15u),-dm*float((p.z>>shift)&15u));
    } else if constexpr(Q==11) {
        unsigned const phase=(g>>3)&3u;
        unsigned const bit=6u*(g&7u)+((phase==0 || phase==3)?16u:0u);
        // Fixed register operands; guard the shift by 32 explicitly.
        uint32_t code=bit<32 ? p.y>>bit : p.z>>(bit-32);
        if(bit<32 && bit>26) code|=p.z<<(32-bit);
        int const scale=int(code&63u)-32;
        float const d=__half2float(__ushort_as_half(uint16_t(p.x)));
        return make_float2(d*float(scale),0.f);
    } else if constexpr(Q==14) return q6_sparse::affine(p,g);
    else {
        static_assert(Q==12 || Q==13);
        return q4_s1::q4_affine_header32(make_uint4(p.x,p.y,p.z,p.w),g&7u);
    }
}
}
