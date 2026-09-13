#pragma once
#include "q4_s1_activation.cuh"

namespace quactlize::execution::q4_s1 {
struct ScaleZero { __half scale,zero; };

template<int Slot>
__device__ __forceinline__ __half2 codes(uint32_t words) {
    static_assert(Slot >= 0 && Slot < 4);
    // The two b16 halves own adjacent N columns, not adjacent K values.
    uint32_t const bits = ((words >> (4 * Slot)) & 0x000f000fu) | 0x64006400u;
    __half2_raw raw;
    raw.x = uint16_t(bits);
    raw.y = uint16_t(bits >> 16);
    return __hsub2(__half2(raw), __float2half2_rn(1032.f));
}

__device__ __forceinline__ uint4 aligned_unit(uint8_t const* p) {
    return *reinterpret_cast<uint4 const*>(p);
}

__device__ __forceinline__ float2 q4_affine_header(uint4 u,unsigned group) {
    uint64_t run=(group&4) ? uint64_t(u.z>>16)|(uint64_t(u.w)<<16)
                          : uint64_t(u.y)|(uint64_t(u.z&0xffff)<<32);
    unsigned shift=6*(group&3);
    float sc=float(unsigned(run>>shift)&63),mn=float(unsigned(run>>(24+shift))&63);
    return make_float2(__half2float(__ushort_as_half(uint16_t(u.x)))*sc,
                      -__half2float(__ushort_as_half(uint16_t(u.x>>16)))*mn);
}

// Included inside the unchanged K-pack kernel namespace by reader_reuse.py.
// Register transport only: canonical weight bytes and FP32 dot order stay fixed.
template<int Columns,int Input>
__device__ __forceinline__ uint4 q4_cooperative_a_chunk(Activation<Input> a,int group,int lane) {
    static_assert(Columns==4 || Columns==8);
    int64_t offset=int64_t(group)*32+(lane%Columns)*(32/Columns);
    if constexpr(Columns==4) return a.load8(offset);
    else {
        uint2 v=a.load4(offset);
        return make_uint4(v.x,v.y,0,0);
    }
}

template<int Columns>
__device__ __forceinline__ float4 q4_cooperative_a_read(uint4 chunk,int offset,int lane) {
    int const owner=(lane&~(Columns-1))+offset/(32/Columns);
    uint32_t lo=chunk.x,hi=chunk.y;
    if constexpr(Columns==4) {
        if(offset&4) {lo=chunk.z;hi=chunk.w;}
    }
    lo=__shfl_sync(0xffffffffu,lo,owner);
    hi=__shfl_sync(0xffffffffu,hi,owner);
    return make_float4(__half2float(__ushort_as_half(uint16_t(lo))),
                       __half2float(__ushort_as_half(uint16_t(lo>>16))),
                       __half2float(__ushort_as_half(uint16_t(hi))),
                       __half2float(__ushort_as_half(uint16_t(hi>>16))));
}

__device__ __forceinline__ uint4 q4_cooperative_unit_read(uint4 unit,int owner) {
    return make_uint4(__shfl_sync(0xffffffffu,unit.x,owner),
                      __shfl_sync(0xffffffffu,unit.y,owner),
                      __shfl_sync(0xffffffffu,unit.z,owner),
                      __shfl_sync(0xffffffffu,unit.w,owner));
}

__device__ __forceinline__ float2 q4_affine_header32(uint4 u,unsigned group) {
    // Two 24-bit scale/min streams. Build each using constant 32-bit shifts,
    // then extract its six-bit group; no variable 64-bit shift is required.
    uint32_t scales=(group&4) ? (u.z>>16)|(u.w<<16) : u.y;
    uint32_t mins=(group&4) ? u.w>>8 : (u.y>>24)|(u.z<<8);
    unsigned shift=6*(group&3);
    float sc=float((scales>>shift)&63),mn=float((mins>>shift)&63);
    return make_float2(__half2float(__ushort_as_half(uint16_t(u.x)))*sc,
                      -__half2float(__ushort_as_half(uint16_t(u.x>>16)))*mn);
}

// Development experiment. Included inside the frozen K-pack namespace.
// No output-format change, FP16 accumulator or inter-CTA synchronization.
__device__ __forceinline__ uint32_t latency_mux(uint32_t mask,uint32_t hi,uint32_t lo) {
    uint32_t value;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xca;" : "=r"(value) : "r"(mask),"r"(hi),"r"(lo));
    return value;
}

template<int Header>
__device__ __forceinline__ uint2 latency_fields(uint4 u,unsigned group) {
    uint32_t scales,mins;
    if constexpr(Header==0) {
        scales=(group&4) ? (u.z>>16)|(u.w<<16) : u.y;
        mins=(group&4) ? u.w>>8 : (u.y>>24)|(u.z<<8);
    } else {
        uint32_t mask=0u-((group>>2)&1);
        scales=latency_mux(mask,(u.z>>16)|(u.w<<16),u.y);
        mins=latency_mux(mask,u.w>>8,(u.y>>24)|(u.z<<8));
    }
    unsigned shift=6*(group&3);
    return make_uint2((scales>>shift)&63,(mins>>shift)&63);
}

template<int Header>
__device__ __forceinline__ quactlize::execution::q4_s1::ScaleZero latency_half_header(uint4 u,unsigned group) {
    uint2 fields=latency_fields<Header>(u,group);
    __half2_raw codes_raw,header_raw;
    codes_raw.x=uint16_t(0x6400|fields.x);codes_raw.y=uint16_t(0x6400|fields.y);
    header_raw.x=uint16_t(u.x);header_raw.y=uint16_t(u.x>>16);
    __half2 const products=__hmul2(__hsub2(__half2(codes_raw),__float2half2_rn(1024.f)),__half2(header_raw));
    __half const scale=__low2half(products),zero=__hneg(__high2half(products));
    return {scale,__float2half_rn(__half2float(zero)+8.f*__half2float(scale))};
}

template<int Header>
__device__ __forceinline__ float2 latency_affine_header(uint4 u,unsigned group) {
    uint2 fields=latency_fields<Header>(u,group);
    return make_float2(__half2float(__ushort_as_half(uint16_t(u.x)))*float(fields.x),
                     -__half2float(__ushort_as_half(uint16_t(u.x>>16)))*float(fields.y));
}

// These compiler-only joins put all named values on the input side of the
// decoding boundary. They emit no device barrier. The build records actual
// native load/wait order; source ordering alone is not an admission claim.
__device__ __forceinline__ void latency_join(uint4& u,uint4& b,uint4& a) {
    asm volatile("" : "+r"(u.x),"+r"(u.y),"+r"(u.z),"+r"(u.w),
                      "+r"(b.x),"+r"(b.y),"+r"(b.z),"+r"(b.w),
                      "+r"(a.x),"+r"(a.y),"+r"(a.z),"+r"(a.w) : : "memory");
}

template<int AMode,int Input>
__device__ __forceinline__ uint4 latency_residue_a(Activation<Input> a,unsigned g,unsigned residue) {
    if constexpr(AMode==0) {
        uint2 v=a.load4(g*32+residue*4);
        return make_uint4(v.x,v.y,0,0);
    } else {
        return make_uint4(__half_as_ushort(a[g*32+residue]),__half_as_ushort(a[g*32+residue+8]),
                          __half_as_ushort(a[g*32+residue+16]),__half_as_ushort(a[g*32+residue+24]));
    }
}

__device__ __forceinline__ uint32_t latency_swap_half(uint32_t value,unsigned lane,unsigned bit) {
    uint32_t other=__shfl_xor_sync(0xffffffffu,value,bit);
    return __byte_perm(value,other,(lane&bit) ? 0x3276 : 0x5410);
}

template<int AMode>
__device__ __forceinline__ float4 latency_residue_values(uint4 raw,unsigned residue) {
    if constexpr(AMode==0) {
        uint2 v=make_uint2(raw.x,raw.y);
        v.x=latency_swap_half(v.x,residue,1);
        v.y=latency_swap_half(v.y,residue,1);
        uint32_t exchange=__shfl_xor_sync(0xffffffffu,(residue&2) ? v.x : v.y,2);
        if(residue&2) v.x=exchange; else v.y=exchange;
        v.x=latency_swap_half(v.x,residue,4);
        v.y=latency_swap_half(v.y,residue,4);
        raw=make_uint4(v.x,v.y,v.x>>16,v.y>>16);
    }
    return make_float4(__half2float(__ushort_as_half(uint16_t(raw.x))),
                       __half2float(__ushort_as_half(uint16_t(raw.y))),
                       __half2float(__ushort_as_half(uint16_t(raw.z))),
                       __half2float(__ushort_as_half(uint16_t(raw.w))));
}

// Reduce K and scatter N ownership at the same time. Each exchange halves
// the number of live output columns, instead of doing an all-reduce for each.
template<int Count, int Stride, int Width>
__device__ __forceinline__ float q4_reduce_scatter_steps(float (&value)[Width], int lane) {
    if constexpr (Count > 1) {
        bool const odd = (lane & Stride) != 0;
        #pragma unroll
        for (int i = 0; i < Count / 2; ++i) {
            float keep = odd ? value[2*i+1] : value[2*i];
            float send = odd ? value[2*i] : value[2*i+1];
            value[i] = keep + __shfl_xor_sync(0xffffffffu, send, Stride);
        }
        return q4_reduce_scatter_steps<Count/2,Stride*2>(value, lane);
    } else {
        float v = value[0];
        #pragma unroll
        for (int d = Stride; d < 32; d *= 2) v += __shfl_xor_sync(0xffffffffu, v, d);
        return v;
    }
}

// Exact-order CTA fold for the bounded medium Q4 experiment.
// Called by the first warp only, after the existing CTA barrier.
// Host-callable as well, so the actual helper can be compared bitwise with
// the old loop without a device. It introduces no CUDA intrinsic or barrier.
template<int Round,int Warps,int TileN>
__host__ __device__ __forceinline__ void q4_medium_fold(
        float& sum,float const* partial,unsigned lane) {
    static_assert(TileN>0 && TileN<=32 && 32%TileN==0);
    constexpr unsigned Stripes=32/TileN;
    unsigned const first=(lane&31u)/TileN;
    unsigned const w=first+Round*Stripes;
    if constexpr((Round+1)*Stripes<=Warps) {
        sum+=partial[w*TileN+(lane&(TileN-1))];
    } else if(w<Warps) {
        sum+=partial[w*TileN+(lane&(TileN-1))];
    }
    if constexpr((Round+1)*Stripes<Warps)
        q4_medium_fold<Round+1,Warps,TileN>(sum,partial,lane);
}

}
