#pragma once
// Register-direct tensor-core reader for decode / small-M K-pack GEMV (rows <= 8).
//
// Same canonical planes, row semantics, F32 storage and arithmetic contract as the SIMT
// register-reuse readers: activations are rounded once to the compute type (RNE), every code
// is expanded EXACTLY into the MMA operand (no per-weight scale or rounding), the tensor core
// accumulates code*activation products in FP32, and the group affine (scale and minimum) is
// applied in FP32 to each group's accumulated dot product. Only the order of the FP32
// additions differs from the SIMT readers.
//
// PPU0010 m8n16k16 (fragment maps verified on device): A = 8 weight columns x 16 k,
// B = 16 rows x 16 k, C = 8 columns x 16 rows.
//   A: lane holds column m=lane/4, k = 2t+v0+8*v1 (t=lane%4)    -> 2 registers
//   B: lane holds row    n=lane/4, k = 2t+v0+8*v1                -> registers 0,1 (rows 8..15 zero)
//   C: lane holds column m=lane/4, row n = t + 4*v                -> v=0,1 used
// Inside one 32-k block the MMA k of (chunk j, v1, t, v0) is the physical k = 16j+8v1+2t+v0
// for every supported plane, so one activation layout serves all formats.
//
// Group rotation: B row n carries token tau's activation for window group g, n = g + WG*tau,
// WG = 8/tokens. A window is WG consecutive quantization groups; every MMA of group g leaves
// the other rows zero, so after the window each C row holds one (group, token) dot product
// and every lane owns useful values even for one token. The group affine is applied once
// per window to those values.
//
// A warp owns 8*T columns (lane row m holds columns base+T*m .. +T-1, one vector load per
// plane row) and a contiguous range of 256-k superblocks. A CTA has WN column warps and WK
// k-slice warps; k slices are folded in fixed warp order through shared memory. Plane rows
// are loaded in granules of BPG 32-k blocks, NB-1 granules ahead of their use.
#include "simt_common.cuh"
#include "../decode/reducer.cuh"
#include <type_traits>

namespace quactlize::execution::mma_reader {

template<int I,int N,class F>
__device__ __forceinline__ void static_for(F&& f) {
    if constexpr(I<N) { f(std::integral_constant<int,I>{}); static_for<I+1,N>(f); }
}

template<int Compute>
__device__ __forceinline__ void mma(float (&c)[4],uint32_t a0,uint32_t a1,uint32_t b0,uint32_t b1) {
    uint32_t const zero=0;
    if constexpr(Compute==0)
        asm("ppu.mma.sync.aligned.m8n16k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6,%7,%8,%9}, "
            "{%0,%1,%2,%3};\n"
            : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
            : "r"(a0),"r"(a1),"r"(b0),"r"(b1),"r"(zero),"r"(zero));
    else
        asm("ppu.mma.sync.aligned.m8n16k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6,%7,%8,%9}, "
            "{%0,%1,%2,%3};\n"
            : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
            : "r"(a0),"r"(a1),"r"(b0),"r"(b1),"r"(zero),"r"(zero));
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

// Window geometry for the token count (C row n = t + 4v of lane t).
template<int Tokens> struct Rot {
    static constexpr int TP=Tokens<=1 ? 1 : Tokens<=2 ? 2 : Tokens<=4 ? 4 : 8;
    static constexpr int WG=8/TP;          // groups per window
    static constexpr int V=TP==1 ? 1 : 2;  // accumulated tokens per column and lane
    static constexpr bool Direct=TP<=2;    // per-lane activation loads instead of shared staging
    __device__ __forceinline__ static int group(int t,int v) { return (t+4*v)%WG; }
    __device__ __forceinline__ static int token(int t,int v) { return (t+4*v)/WG; }
};

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
        return q4_s1::q4_affine_header32(s.u[i],g);
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

template<int Q,int Mode,int Compute,int T,int WN,int WK,int Tokens,int NB>
__global__ void __launch_bounds__(WN*WK*32) kernel(qkg_call_v1 call,int split) {
    using K=Kind<Q>;
    using M=typename MetaOf<Q>::type;   // superblock/window metadata reader
    using R=Rot<Tokens>;
    constexpr int TileN=8*T,CtaN=WN*TileN,TP=R::TP,WG=R::WG,V=R::V;
    constexpr int CPG=K::GK/16,GPS=256/K::GK,Windows=GPS/WG,BPG=K::BPG,GPSB=8/BPG;
    static_assert(GPSB%NB==0,"granule ring must divide a superblock");
    extern __shared__ uint32_t smem[];
    unsigned const tid=threadIdx.x,lane=tid%32,warp=tid/32,m=lane/4,t=lane%4;
    unsigned const wn=warp%WN,wk=warp/WN,row=blockIdx.y,partition=blockIdx.z;
    int const n_cta=blockIdx.x*CtaN,n_w=n_cta+wn*TileN,col=n_w+T*m;
    int tokens=call.rows,expert=0;
    int64_t a_offset=0,out_offset=0;
    if constexpr(Mode==QKG_INDEXED) {
        auto r=simt::common_locate<QKG_INDEXED>(call,row);
        expert=r.expert;a_offset=r.a;out_offset=r.output;tokens=1;
    }
    int const sb_total=call.k/256;
    int const sb_begin=int(int64_t(sb_total)*partition/split),sb_end=int(int64_t(sb_total)*(partition+1)/split);
    int const k_begin=sb_begin*256,k_len=(sb_end-sb_begin)*256;
    auto store=[&](int tok,int n,float value) {
        if(split==1) call.output[(Mode==QKG_INDEXED ? out_offset : int64_t(tok)*call.out_row_stride)+n]=value;
        else static_cast<float*>(call.workspace)[(int64_t(Mode==QKG_INDEXED ? int(row) : tok)*split+partition)*call.n+n]=value;
    };
    if(expert<0 || expert>=call.experts) {
        for(int idx=tid;idx<CtaN;idx+=WN*WK*32) store(0,n_cta+idx,__int_as_float(0x7fc00000));
        return;
    }
    uint64_t const nk=uint64_t(call.n)*call.k;
    using F=simt::Format<Q>;
    Planes planes{reinterpret_cast<uint16_t const*>(call.low+uint64_t(expert)*nk/8*F::low_bits),nullptr,
                  call.units+uint64_t(expert)*F::metadata_bytes(nk),call.n};
    if constexpr(F::high_bits)
        planes.high=reinterpret_cast<uint16_t const*>(call.high+uint64_t(expert)*nk/8*F::high_bits);
    typename K::Lane const ls(col);
    float const* a=static_cast<float const*>(call.a)+a_offset;
    int const words=k_len/2;
    uint32_t* act=smem;
    float* sums=reinterpret_cast<float*>(smem+TP*words);
    int const local=sb_end-sb_begin;
    int const w_begin=sb_begin+local*int(wk)/WK,w_end=sb_begin+local*(int(wk)+1)/WK;
    int const gran_end=w_end*GPSB;
    int const g_lane=m%WG,tau_lane=m/WG;
    bool const supply=tau_lane<tokens;
    typename K::template Gran<T> buf[NB];
    // The first granules are in flight while activations are staged or loaded.
    static_for<0,NB-1>([&](auto pc) {
        constexpr int d=decltype(pc)::value;
        if(w_begin*GPSB+d<gran_end) K::load(buf[d],planes,col,w_begin*GPSB+d);
    });
    // Direct mode: this lane's activation pairs for (token tau_lane, its window group).
    float2 pre[2*CPG];
    auto load_act=[&](int group_abs) {
        float const* src=a+int64_t(tau_lane)*call.a_row_stride+group_abs*K::GK+2*t;
        #pragma unroll
        for(int s=0;s<2*CPG;++s)
            pre[s]=supply ? *reinterpret_cast<float2 const*>(src+8*s) : make_float2(0.f,0.f);
    };
    if constexpr(R::Direct) {
        if(w_begin<w_end) load_act(w_begin*GPS+g_lane);
    } else {
        stage<Compute,K::SumK>(act,sums,a,call.a_row_stride,tokens,k_begin,k_len);
        __syncthreads();
    }

    float acc[T][V];
    #pragma unroll
    for(int i=0;i<T;++i)
        #pragma unroll
        for(int v=0;v<V;++v) acc[i][v]=0.f;
    for(int sb=w_begin;sb<w_end;++sb) {
        typename SbOf<Q,T>::type sbm;
        M::load_sb(sbm,planes,col,sb);
        static_for<0,Windows>([&](auto wc) {
            constexpr int W=decltype(wc)::value,G0=W*WG;
            int const rel=(sb-sb_begin)*GPS+G0;   // partition-relative group of the window start
            typename M::template Win<T> win;
            M::load_win(win,planes,col,sb*GPS+G0+R::group(t,0),sb*GPS+G0+R::group(t,1));
            uint32_t bw[2*CPG];
            float asum[2]={0.f,0.f};
            if constexpr(R::Direct) {
                float part=0.f;
                #pragma unroll
                for(int s=0;s<2*CPG;++s) {
                    bw[s]=round_pair<Compute>(pre[s].x,pre[s].y);
                    if constexpr(K::SumK>0) part+=pair_sum<Compute>(bw[s]);
                }
                // Prefetch the next window's pairs (also across superblocks).
                int const next=W+1<Windows ? sb*GPS+G0+WG+g_lane : (sb+1)*GPS+g_lane;
                if(W+1<Windows || sb+1<w_end) load_act(next);
                if constexpr(K::SumK>0) {
                    // Sum over the four lanes of B row m, then fetch the rows this lane owns in C.
                    part+=__shfl_xor_sync(0xffffffffu,part,1);
                    part+=__shfl_xor_sync(0xffffffffu,part,2);
                    asum[0]=__shfl_sync(0xffffffffu,part,4*int(t));
                    asum[1]=__shfl_sync(0xffffffffu,part,4*int(t)+16);
                }
            } else {
                int const kk=(rel+g_lane)*K::GK;
                uint32_t const* src=act+tau_lane*words+(kk/32)*16+t*4+((kk/16)%2)*2;
                if constexpr(CPG==2) {
                    uint4 v=supply ? *reinterpret_cast<uint4 const*>(src) : make_uint4(0,0,0,0);
                    bw[0]=v.x;bw[1]=v.y;bw[2]=v.z;bw[3]=v.w;
                } else {
                    uint2 v=supply ? *reinterpret_cast<uint2 const*>(src) : make_uint2(0,0);
                    bw[0]=v.x;bw[1]=v.y;
                }
                if constexpr(K::SumK>0) {
                    #pragma unroll
                    for(int v=0;v<2;++v) {
                        int const tok=R::token(t,v);
                        if(tok<tokens) asum[v]=sums[tok*(k_len/K::SumK)+rel+R::group(t,v)];
                    }
                }
            }
            float c[T][4];
            #pragma unroll
            for(int i=0;i<T;++i) c[i][0]=c[i][1]=c[i][2]=c[i][3]=0.f;
            static_for<0,WG>([&](auto gc) {
                constexpr int g=decltype(gc)::value;
                bool const mine=g_lane==g;
                static_for<0,CPG>([&](auto jc) {
                    constexpr int jj=decltype(jc)::value,ch=(G0+g)*CPG+jj,blk=ch/2,j=ch%2;
                    constexpr int gs=blk/BPG,bg=blk%BPG;
                    if constexpr(j==0 && bg==0) {
                        // Entering granule gs: prefetch NB-1 granules ahead (also across superblocks).
                        int const next=sb*GPSB+gs+NB-1;
                        if(next<gran_end) K::load(buf[(gs+NB-1)%NB],planes,col,next);
                    }
                    uint32_t const b0=mine ? bw[2*jj] : 0u,b1=mine ? bw[2*jj+1] : 0u;
                    #pragma unroll
                    for(int i=0;i<T;++i) {
                        uint2 const av=K::template a<Compute,T,bg,j>(buf[gs%NB],ls,i);
                        mma<Compute>(c[i],av.x,av.y,b0,b1);
                    }
                });
            });
            static_for<0,2>([&](auto vc) {
                constexpr int v=decltype(vc)::value;
                int const tok=R::token(t,v),g=R::group(t,v);
                if(tok<tokens) {
                    #pragma unroll
                    for(int i=0;i<T;++i) {
                        float2 const af=M::template affine<T,v>(sbm,win,i,unsigned(G0+g));
                        acc[i][V==1 ? 0 : v]+=fmaf(af.x,c[i][v],af.y*asum[v]);
                    }
                }
            });
        });
    }
    // Fold lanes that hold the same (column, token), then k-slice warps in fixed order.
    #pragma unroll
    for(int i=0;i<T;++i)
        #pragma unroll
        for(int v=0;v<V;++v) {
            float x=acc[i][v];
            if constexpr(TP<=2) { x+=__shfl_xor_sync(0xffffffffu,x,1); x+=__shfl_xor_sync(0xffffffffu,x,2); }
            else if constexpr(TP==4) x+=__shfl_xor_sync(0xffffffffu,x,1);
            acc[i][v]=x;
        }
    bool const owner=TP<=2 ? t==0 : TP==4 ? (t&1)==0 : true;
    auto owner_token=[&](int v) { return TP==1 ? 0 : TP==2 ? v : TP==4 ? int(t>>1)+2*v : int(t)+4*v; };
    if constexpr(WK==1) {
        if(owner) {
            #pragma unroll
            for(int v=0;v<V;++v) {
                int const tok=owner_token(v);
                if(tok<tokens) {
                    #pragma unroll
                    for(int i=0;i<T;++i) store(tok,col+i,acc[i][v]);
                }
            }
        }
    } else {
        if constexpr(!R::Direct) __syncthreads();
        float* partial=reinterpret_cast<float*>(smem);
        if(owner) {
            #pragma unroll
            for(int v=0;v<V;++v) {
                int const tok=owner_token(v);
                if(tok<tokens) {
                    #pragma unroll
                    for(int i=0;i<T;++i) partial[(wk*TP+tok)*CtaN+wn*TileN+T*m+i]=acc[i][v];
                }
            }
        }
        __syncthreads();
        for(int idx=tid;idx<tokens*CtaN;idx+=WN*WK*32) {
            int const tok=idx/CtaN,cn=idx%CtaN;
            float sum=0.f;
            #pragma unroll
            for(int w=0;w<WK;++w) sum+=partial[(w*TP+tok)*CtaN+cn];
            store(tok,n_cta+cn,sum);
        }
    }
}

template<int Q,int Tokens,int WK,int CtaN>
size_t shared_bytes(int k_len) {
    using R=Rot<Tokens>;
    size_t const staging=R::Direct ? 0 :
        size_t(R::TP)*(k_len/2)*4+(Kind<Q>::SumK ? size_t(R::TP)*(k_len/Kind<Q>::SumK)*4 : 0);
    size_t const reduce=WK>1 ? size_t(WK)*R::TP*CtaN*4 : 0;
    return staging>reduce ? staging : reduce;
}

template<int Q,int Mode,int Compute,int T,int WN,int WK,int Tokens,int NB>
int launch_typed(qkg_call_v1 const& c,int split) {
    using R=Rot<Tokens>;
    constexpr int CtaN=WN*8*T;
    int const rows=Mode==QKG_INDEXED ? c.rows : 1;
    int const tokens=Mode==QKG_INDEXED ? 1 : c.rows;
    if(c.n%CtaN || tokens>R::TP || (Mode==QKG_INDEXED && Tokens!=1) || c.input_type!=QKG_F32 ||
       c.k%(256*split) || ((uintptr_t(c.low)|uintptr_t(c.high)|uintptr_t(c.units))&15) ||
       (uintptr_t(c.a)&7) || (c.a_row_stride&1) || (Mode==QKG_INDEXED && (c.a_token_stride&1))) return QKG_SHAPE;
    size_t const bytes=shared_bytes<Q,Tokens,WK,CtaN>(c.k/split);
    auto stream=static_cast<hggcStream_t>(c.stream);
    auto fn=kernel<Q,Mode,Compute,T,WN,WK,Tokens,NB>;
    if(bytes>48*1024 && hggcFuncSetAttribute(reinterpret_cast<void const*>(fn),
            hggcFuncAttributeMaxDynamicSharedMemorySize,int(bytes))!=hggcSuccess) return QKG_RUNTIME;
    fn<<<dim3(c.n/CtaN,rows,split),WN*WK*32,bytes,stream>>>(c,split);
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    simt::launch_reduction<Q>(c,split,stream);
    return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
}
} // namespace quactlize::execution::mma_reader
