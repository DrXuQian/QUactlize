#pragma once
// Exact integer-code fragments for canonical K-pack; no public execution ABI dependency.
// PPU0010 A carries weight columns; B carries rotated (group, token) activation rows.
// This mapping differs from the activation-as-A mixed-input mainloop.
#include <hggc_runtime.h>
#include <hggc_fp16.h>
#include <hggc_bf16.h>
#include <type_traits>
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cutlass/arch/arch.h"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_exact_code_geometry.hpp"

namespace cutlass::gemm::collective::exact_code_ppu_detail {
template<int I,int N,class F>
__device__ __forceinline__ void static_for(F&& f) {
    if constexpr(I<N) { f(std::integral_constant<int,I>{}); static_for<I+1,N>(f); }
}

template<int Compute>
using MmaOperation = std::conditional_t<Compute==0,
    cute::PPU0010_8x16x16_F32F16F16F32_TN,
    cute::PPU0010_8x16x16_F32BF16BF16F32_TN>;

// Bind the handwritten canonical-code delivery to the vendor's actual atom
// traits. Merely retaining the m8n16k16 mnemonic does not prove lane ownership.
template<class Traits>
constexpr bool native_mapping_matches() {
    for(int lane=0;lane<32;++lane) {
        int const m=lane/4,t=lane%4;
        for(int v=0;v<4;++v) {
            if(typename Traits::ALayout{}(cute::make_coord(lane,v)) != m+8*(2*t+v%2+8*(v/2)))
                return false;
            if(typename Traits::CLayout{}(cute::make_coord(lane,v)) != m+8*(t+4*v))
                return false;
        }
        for(int v=0;v<8;++v)
            if(typename Traits::BLayout{}(cute::make_coord(lane,v)) !=
                    m+8*(v/4)+16*(2*t+v%2+8*((v/2)%2))) return false;
    }
    return true;
}

template<int Compute>
__device__ __forceinline__ void mma(float (&c)[4],uint32_t a0,uint32_t a1,uint32_t b0,uint32_t b1) {
    static_assert(Compute==0 || Compute==1);
    MmaOperation<Compute>::fma(c[0],c[1],c[2],c[3],a0,a1,b0,b1,0u,0u,c[0],c[1],c[2],c[3]);
}

// Canonical Q4_K/Q5_K unit: two 24-bit scale/min streams, FP32 affine.
// Keep this standalone decoder independent of execution/SIMT dispatch.
__device__ __forceinline__ float2 q4_affine_header32(uint4 u,unsigned group) {
    uint32_t scales=(group&4) ? (u.z>>16)|(u.w<<16) : u.y;
    uint32_t mins=(group&4) ? u.w>>8 : (u.y>>24)|(u.z<<8);
    unsigned shift=6*(group&3);
    float sc=float((scales>>shift)&63),mn=float((mins>>shift)&63);
    return make_float2(__half2float(__ushort_as_half(uint16_t(u.x)))*sc,
                      -__half2float(__ushort_as_half(uint16_t(u.x>>16)))*mn);
}

// (x & Mask) | Value with an immediate mask and value.
template<uint32_t Mask,uint32_t Value>
__device__ __forceinline__ uint32_t lop3_imm(uint32_t x) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(r) : "r"(x),"n"(Mask),"n"(Value));
    return r;
}
// (x & Mask) | c with an immediate mask.
template<uint32_t Mask>
__device__ __forceinline__ uint32_t lop3_or(uint32_t x,uint32_t c) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(r) : "r"(x),"n"(Mask),"r"(c));
    return r;
}
// Logical shift by a compile-time signed amount (positive = right).
template<int S>
__device__ __forceinline__ uint32_t shift(uint32_t x) {
    if constexpr(S>0) return x>>S;
    else if constexpr(S<0) return x<<(-S);
    else return x;
}
__device__ __forceinline__ uint32_t h2(__half2 v) { return reinterpret_cast<uint32_t const&>(v); }
__device__ __forceinline__ __half2 as_h2(uint32_t v) { return reinterpret_cast<__half2 const&>(v); }
__device__ __forceinline__ __ppu_bfloat162 as_b2(uint32_t v) { return reinterpret_cast<__ppu_bfloat162 const&>(v); }
__device__ __forceinline__ uint32_t b2(__ppu_bfloat162 v) { return reinterpret_cast<uint32_t const&>(v); }

// Exact code pair (one value per 16-bit half of x) from the 4-bit nibble at bits 4*Slot..+3,
// plus optional high bits already placed by the caller (Hi, at bit 4 for even F16 slots and
// BF16, bit 8 for odd F16 slots), minus Bias, in the compute type.
template<int Compute,int Slot,int Bias>
__device__ __forceinline__ uint32_t code_pair(uint32_t x,uint32_t hi=0) {
    static_assert(Slot>=0 && Slot<4);
    if constexpr(Compute==0) {
        uint32_t const s=Slot>=2 ? x>>8 : x;
        if constexpr(Slot%2==0) {
            uint32_t const v=lop3_imm<0x000f000fu,0x64006400u>(s)|hi;
            return h2(__hadd2(as_h2(v),__float2half2_rn(-1024.f-Bias)));
        } else {
            uint32_t const v=lop3_imm<0x00f000f0u,0x64006400u>(s)|hi;
            return h2(__hfma2(as_h2(v),__float2half2_rn(1.f/16),__float2half2_rn(-64.f-Bias)));
        }
    } else {
        uint32_t const v=lop3_imm<0x000f000fu,0x43004300u>(x>>(4*Slot))|hi;
        return b2(__hadd2(as_b2(v),__float2bfloat162_rn(-128.f-Bias)));
    }
}
// Bit position at which the caller must place a code's high bits for code_pair<Compute,Slot>.
template<int Compute,int Slot> constexpr int kHighAt=(Compute==0 && Slot%2) ? 8 : 4;

template<int Compute>
__device__ __forceinline__ uint32_t round_pair(float x,float y) {
    if constexpr(Compute==0) return h2(__floats2half2_rn(x,y));
    else return b2(__floats2bfloat162_rn(x,y));
}
template<int Compute>
__device__ __forceinline__ float pair_sum(uint32_t v) {
    if constexpr(Compute==0) { float2 f=__half22float2(as_h2(v)); return f.x+f.y; }
    else return __uint_as_float(v<<16)+__uint_as_float(v&0xffff0000u);
}

// Shared activation staging (used for 3..8 tokens): per token a K-major array of 32-bit
// pairs where 32-k block b, lane t and slot s occupy word b*16 + t*4 + s (pair k = 32b + 8s +
// 2t + {0,1}). SumK>0 also stores the sum of the ROUNDED activations of every SumK-k group.
template<int Compute,int SumK>
__device__ __forceinline__ void stage(uint32_t* act,float* sums,float const* a,int64_t token_stride,
                                      int tokens,int k_begin,int k_len) {
    constexpr int U=4;
    int const quads=k_len/4,words=k_len/2;
    for(int tok=0;tok<tokens;++tok) {
        float const* src=a+int64_t(tok)*token_stride+k_begin;
        for(int base=threadIdx.x;base<quads;base+=blockDim.x*U) {
            float4 v[U];
            #pragma unroll
            for(int u=0;u<U;++u) {
                int const q=base+u*blockDim.x;
                if(q<quads) v[u]=*reinterpret_cast<float4 const*>(src+4*q);
            }
            #pragma unroll
            for(int u=0;u<U;++u) {
                int const q=base+u*blockDim.x;
                // k_len is a multiple of 256: this guard is uniform per warp.
                if(q<quads) {
                    uint32_t const p0=round_pair<Compute>(v[u].x,v[u].y),p1=round_pair<Compute>(v[u].z,v[u].w);
                    int const b=q>>3,s=(q&7)>>1,t=2*(q&1);
                    act[tok*words+b*16+t*4+s]=p0;
                    act[tok*words+b*16+(t+1)*4+s]=p1;
                    if constexpr(SumK>0) {
                        constexpr int Lanes=SumK/4;
                        float sum=pair_sum<Compute>(p0)+pair_sum<Compute>(p1);
                        #pragma unroll
                        for(int d=1;d<Lanes;d*=2) sum+=__shfl_xor_sync(0xffffffffu,sum,d);
                        if((q&(Lanes-1))==0) sums[tok*(k_len/SumK)+q/Lanes]=sum;
                    }
                }
            }
        }
    }
}

struct Planes {
    uint16_t const* low;
    uint16_t const* high;
    uint8_t const* units;
    int n;
};

template<int T> struct Words;
template<> struct Words<2> {
    uint32_t w[1];
    __device__ __forceinline__ void load(uint16_t const* p) { w[0]=*reinterpret_cast<uint32_t const*>(p); }
};
template<> struct Words<4> {
    uint32_t w[2];
    __device__ __forceinline__ void load(uint16_t const* p) { uint2 v=*reinterpret_cast<uint2 const*>(p); w[0]=v.x;w[1]=v.y; }
};
template<> struct Words<8> {
    uint32_t w[4];
    __device__ __forceinline__ void load(uint16_t const* p) {
        uint4 v=*reinterpret_cast<uint4 const*>(p); w[0]=v.x;w[1]=v.y;w[2]=v.z;w[3]=v.w;
    }
};
// [row0 word of column i | row1 word of column i << 16]
template<int T>
__device__ __forceinline__ uint32_t column_pair(Words<T> const& r0,Words<T> const& r1,int i) {
    return __byte_perm(r0.w[i/2],r1.w[i/2],(i&1) ? 0x7632 : 0x5410);
}

struct NoLane { __device__ explicit NoLane(int) {} };

// Format kinds. A granule is BPG consecutive 32-k blocks of all planes. Group metadata is
// loaded once per 256-k superblock (Sb) and/or once per window for the lane's two groups (Win).
template<int Q> struct Kind;

// ---------------------------------------------------------------- Q4_K
template<> struct Kind<12> {
    static constexpr int GK=32,SumK=32,BPG=1;
    using Lane=NoLane;
    template<int T> struct Gran { Words<T> lo[2]; };
    template<int T>
    __device__ __forceinline__ static void load(Gran<T>& g,Planes const& p,int col,int gi) {
        unsigned const t=threadIdx.x%4;
        g.lo[0].load(p.low+uint64_t(gi*8+2*t)*p.n+col);
        g.lo[1].load(p.low+uint64_t(gi*8+2*t+1)*p.n+col);
    }
    template<int Compute,int T,int BG,int J>
    __device__ __forceinline__ static uint2 a(Gran<T> const& g,Lane const&,int i) {
        uint32_t const x=column_pair<T>(g.lo[0],g.lo[1],i);
        return make_uint2(code_pair<Compute,2*J,0>(x),code_pair<Compute,2*J+1,0>(x));
    }
    template<int T> struct Sb { uint4 u[T]; };
    template<int T>
    __device__ __forceinline__ static void load_sb(Sb<T>& s,Planes const& p,int col,int sb) {
        #pragma unroll
        for(int i=0;i<T;++i) s.u[i]=*reinterpret_cast<uint4 const*>(p.units+(uint64_t(sb)*p.n+col+i)*16);
    }
    template<int T> struct Win {};
    template<int T>
    __device__ __forceinline__ static void load_win(Win<T>&,Planes const&,int,int,int) {}
    template<int T,int Vi>
    __device__ __forceinline__ static float2 affine(Sb<T> const& s,Win<T> const&,int i,unsigned g) {
        return q4_affine_header32(s.u[i],g);
    }
};

// ---------------------------------------------------------------- Q5_K
// Low plane as Q4_K. The high plane interchanges N bit 3 and K bit 7: for a lane's T adjacent
// columns and block b, row (b/8)*16 + ((b/2)&1)*8 + k%8, column (col&~15)|(col&7)|((b/4)&1)*8,
// bit ((col/8)&1)*8 + 4*(b&1) + slot. A granule is a block pair (one high row pair).
template<> struct Kind<13> {
    static constexpr int GK=32,SumK=32,BPG=2;
    struct Lane {
        uint32_t sel;   // byte_perm offset choosing the lane's high byte of each 16-bit word
        __device__ explicit Lane(int col) : sel(((col>>3)&1) ? 0x0101u : 0u) {}
    };
    template<int T> struct Gran { Words<T> lo[4]; Words<T> hi[2]; };
    template<int T>
    __device__ __forceinline__ static void load(Gran<T>& g,Planes const& p,int col,int gi) {
        unsigned const t=threadIdx.x%4;
        int const b0=2*gi;
        #pragma unroll
        for(int r=0;r<4;++r) g.lo[r].load(p.low+uint64_t((b0+r/2)*8+2*t+(r&1))*p.n+col);
        int const pn=(col&~15)|(col&7)|(((b0>>2)&1)<<3);
        #pragma unroll
        for(int v0=0;v0<2;++v0)
            g.hi[v0].load(p.high+uint64_t((b0>>3)*16+((b0>>1)&1)*8+2*t+v0)*p.n+pn);
    }
    template<int Compute,int T,int BG,int J>
    __device__ __forceinline__ static uint2 a(Gran<T> const& g,Lane const& lane,int i) {
        uint32_t const x=column_pair<T>(g.lo[2*BG],g.lo[2*BG+1],i);
        // [row0 high byte | . | row1 high byte | .]: bit 4*BG+slot of each 16-bit half.
        uint32_t const hx=__byte_perm(g.hi[0].w[i/2],g.hi[1].w[i/2],(i&1 ? 0x0602u : 0x0400u)+lane.sel);
        constexpr int S0=2*J,S1=2*J+1;
        uint32_t const h0=lop3_or<0x00010001u<<kHighAt<Compute,S0>>(shift<4*BG+S0-kHighAt<Compute,S0>>(hx),0u);
        uint32_t const h1=lop3_or<0x00010001u<<kHighAt<Compute,S1>>(shift<4*BG+S1-kHighAt<Compute,S1>>(hx),0u);
        return make_uint2(code_pair<Compute,S0,0>(x,h0),code_pair<Compute,S1,0>(x,h1));
    }
    using Sb=Kind<12>::Sb<1>;
    template<int T> using SbT=Kind<12>::Sb<T>;
    template<int T> struct Win {};
    template<int T>
    __device__ __forceinline__ static void load_win(Win<T>&,Planes const&,int,int,int) {}
};

// ---------------------------------------------------------------- Q6_K
// Low plane as Q4_K; high plane Plane<2>: row (b/2)*8 + k%8, two bits at 8*(b&1) + 2*slot.
// Groups of 16 k: one int8 scale per group and one FP16 d per superblock, in the canonical
// 36-byte unit of two superblocks of the same column.
template<> struct Kind<14> {
    static constexpr int GK=16,SumK=0,BPG=2;
    using Lane=NoLane;
    template<int T> struct Gran { Words<T> lo[4]; Words<T> hi[2]; };
    template<int T>
    __device__ __forceinline__ static void load(Gran<T>& g,Planes const& p,int col,int gi) {
        unsigned const t=threadIdx.x%4;
        int const b0=2*gi;
        #pragma unroll
        for(int r=0;r<4;++r) g.lo[r].load(p.low+uint64_t((b0+r/2)*8+2*t+(r&1))*p.n+col);
        #pragma unroll
        for(int v0=0;v0<2;++v0) g.hi[v0].load(p.high+uint64_t(gi*8+2*t+v0)*p.n+col);
    }
    template<int Compute,int T,int BG,int J>
    __device__ __forceinline__ static uint2 a(Gran<T> const& g,Lane const&,int i) {
        uint32_t const x=column_pair<T>(g.lo[2*BG],g.lo[2*BG+1],i);
        uint32_t const hx=column_pair<T>(g.hi[0],g.hi[1],i);
        constexpr int S0=2*J,S1=2*J+1;
        uint32_t const h0=lop3_or<0x00030003u<<kHighAt<Compute,S0>>(shift<8*BG+2*S0-kHighAt<Compute,S0>>(hx),0u);
        uint32_t const h1=lop3_or<0x00030003u<<kHighAt<Compute,S1>>(shift<8*BG+2*S1-kHighAt<Compute,S1>>(hx),0u);
        return make_uint2(code_pair<Compute,S0,32>(x,h0),code_pair<Compute,S1,32>(x,h1));
    }
    __device__ __forceinline__ static uint8_t const* record(Planes const& p,int col,int sb) {
        return p.units+(uint64_t(sb>>1)*p.n+col)*36+(sb&1)*18;
    }
    template<int T> struct Sb { float d[T]; };
    template<int T>
    __device__ __forceinline__ static void load_sb(Sb<T>& s,Planes const& p,int col,int sb) {
        #pragma unroll
        for(int i=0;i<T;++i) s.d[i]=__half2float(*reinterpret_cast<__half const*>(record(p,col+i,sb)));
    }
    template<int T> struct Win { int sc[2][T]; };
    template<int T>
    __device__ __forceinline__ static void load_win(Win<T>& w,Planes const& p,int col,int g0,int g1) {
        #pragma unroll
        for(int i=0;i<T;++i) {
            w.sc[0][i]=reinterpret_cast<int8_t const*>(record(p,col+i,g0/16))[2+g0%16];
            w.sc[1][i]=reinterpret_cast<int8_t const*>(record(p,col+i,g1/16))[2+g1%16];
        }
    }
    template<int T,int Vi>
    __device__ __forceinline__ static float2 affine(Sb<T> const& s,Win<T> const& w,int i,unsigned) {
        return make_float2(s.d[i]*float(w.sc[Vi][i]),0.f);
    }
};

// ---------------------------------------------------------------- Q8_0
template<> struct Kind<8> {
    static constexpr int GK=32,SumK=0,BPG=1;
    using Lane=NoLane;
    template<int T> struct Gran { Words<T> lo[4]; };   // [2j+v0]: row 16b + 8j + 2t + v0
    template<int T>
    __device__ __forceinline__ static void load(Gran<T>& g,Planes const& p,int col,int gi) {
        unsigned const t=threadIdx.x%4;
        #pragma unroll
        for(int r=0;r<4;++r) g.lo[r].load(p.low+uint64_t(gi*16+(r/2)*8+2*t+(r&1))*p.n+col);
    }
    template<int Compute,int T,int BG,int J>
    __device__ __forceinline__ static uint2 a(Gran<T> const& g,Lane const&,int i) {
        static_assert(Compute==0,"Q8_0 tensor-core decode reader is F16-only");
        uint32_t const r0=g.lo[2*J].w[i/2],r1=g.lo[2*J+1].w[i/2];
        // [row0 slot0, row1 slot0, row0 slot1, row1 slot1] of column i
        uint32_t const p=__byte_perm(r0,r1,(i&1) ? 0x7362 : 0x5140);
        uint32_t const s0=__byte_perm(p,0x64646464u,0x4140),s1=__byte_perm(p,0x64646464u,0x4342);
        __half2 const bias=__float2half2_rn(-1152.f);
        return make_uint2(h2(__hadd2(as_h2(s0),bias)),h2(__hadd2(as_h2(s1),bias)));
    }
    template<int T> struct Sb {};
    template<int T>
    __device__ __forceinline__ static void load_sb(Sb<T>&,Planes const&,int,int) {}
    template<int T> struct Win { Words<T> d[2]; };
    // FP16 d of the lane's two window groups (absolute group indices) for its T columns.
    template<int T>
    __device__ __forceinline__ static void load_win(Win<T>& w,Planes const& p,int col,int g0,int g1) {
        auto d=reinterpret_cast<uint16_t const*>(p.units);
        w.d[0].load(d+uint64_t(g0)*p.n+col);
        w.d[1].load(d+uint64_t(g1)*p.n+col);
    }
    template<int T,int Vi>
    __device__ __forceinline__ static float2 affine(Sb<T> const&,Win<T> const& w,int i,unsigned) {
        return make_float2(__half2float(__ushort_as_half(uint16_t(w.d[Vi].w[i/2]>>(16*(i&1))))),0.f);
    }
};

// Superblock metadata access shared by Q4_K/Q5_K.
template<int Q,int T> struct SbOf { using type=typename Kind<Q>::template Sb<T>; };
template<int T> struct SbOf<13,T> { using type=Kind<12>::Sb<T>; };
template<int Q> struct MetaOf { using type=Kind<Q>; };
template<> struct MetaOf<13> { using type=Kind<12>; };

} // namespace cutlass::gemm::collective::exact_code_ppu_detail
