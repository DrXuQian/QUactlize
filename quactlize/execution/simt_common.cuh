#pragma once
#include "simt_kernel.cuh"
#include "simt_common_scope.hpp"

// Same producer body admitted by the common-reader Split-K A/B gate.
// Canonical bytes, ordered FP32 math and the existing reducer are unchanged.

// Canonical coordinates, not a new packed representation. All
// coordinates are validated nonnegative before launch. Signed expert IDs and
// signed scales must not use these types; address products remain 64 bit.
#if defined(__CUDACC__) || defined(__HGGC__)
#define QG_HD __host__ __device__
#else
#define QG_HD
#endif
namespace quactlize::execution::common_map {
template<unsigned Bits> struct Plane {
    static constexpr unsigned kPack = 16 / Bits;
    static constexpr unsigned kLogicalKPerDelivery = 8 * kPack;
    QG_HD static constexpr unsigned word_slot(unsigned k) {
        return (k % kLogicalKPerDelivery) / 8;
    }
    QG_HD static constexpr uint64_t word_index(unsigned n, unsigned k, unsigned extent) {
        return uint64_t((k / kLogicalKPerDelivery) * 8 + k % 8) * extent + n;
    }
};

// Q5's high plane interchanges N bit3 and K bit7; it is NOT Plane<1>.
struct Q5High {
    QG_HD static constexpr unsigned word_slot(unsigned n, unsigned k) {
        return (((n >> 3) & 1) << 3) | ((k >> 3) & 7);
    }
    QG_HD static constexpr uint64_t word_index(unsigned n, unsigned k, unsigned extent) {
        unsigned const pn = (n & ~15u) | (n & 7u) | (((k >> 7) & 1u) << 3);
        unsigned const kg = (k / 256) * 16 | (((k >> 6) & 1) << 3) | (k & 7);
        return uint64_t(kg) * extent + pn;
    }
};
}
#undef QG_HD


// C4 warp: eight consecutive groups, within one canonical 36-byte Q6 unit.
// All lanes have the same phase. Fields are selected for the RECEIVING group,
// not for the source lane's group. No format, rounding or dot-order change.
namespace quactlize::execution::common_q6_metadata {
struct Packet { uint32_t x, y, z, w; };

template<class Words>
__host__ __device__ __forceinline__ Packet load_words(Words const& p, unsigned group) {
    unsigned const phase=(group>>3)&3u;
    if (phase==0) return {p[0],p[1],p[2],0};
    if (phase==1) return {p[0],p[2],p[3],p[4]};
    if (phase==2) return {p[4],p[5],p[6],0};
    return {p[4],p[7],p[8],0};
}

__device__ __forceinline__ Packet load(uint8_t const* p, unsigned group) {
    return load_words(reinterpret_cast<uint32_t const*>(p),group);
}

__device__ __forceinline__ Packet share(Packet p, int owner, unsigned /* group */) {
    p.x=__shfl_sync(0xffffffffu,p.x,owner);
    p.y=__shfl_sync(0xffffffffu,p.y,owner);
    p.z=__shfl_sync(0xffffffffu,p.z,owner);
    // Fixed four exchanges avoid a masked shuffle and its reconvergence.
    // The unused fourth word is zero; it never requires an extra global load.
    p.w=__shfl_sync(0xffffffffu,p.w,owner);
    return p;
}

__host__ __device__ __forceinline__ float2 affine(Packet const& p, unsigned group) {
    bool const second=(group&16u)!=0;
    unsigned const byte=(group&7u)+(second ? 0u : 2u);
    bool const first=(group&24u)==0;
    uint32_t const a=first ? p.x : p.y;
    uint32_t const b=first ? p.y : p.z;
    uint32_t const c=first ? p.z : p.w;
    uint32_t const packed=byte<4 ? a : byte<8 ? b : c;
    uint32_t const header=second ? p.x>>16 : p.x;
    unsigned const code=(packed>>(8u*(byte&3u)))&255u;
    int const scale=int(code^128u)-128;
    float const d=__half2float(__ushort_as_half(uint16_t(header)));
    return make_float2(d*float(scale),0.f);
}
} // namespace quactlize::execution::common_q6_metadata


namespace quactlize::execution::common_metadata {
using Packet = common_q6_metadata::Packet;

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
    } else if constexpr(Q==14) return common_q6_metadata::load_words(p,g);
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
    } else if constexpr(Q==14) return common_q6_metadata::affine(p,g);
    else {
        static_assert(Q==12 || Q==13);
        return q4_s1::q4_affine_header32(make_uint4(p.x,p.y,p.z,p.w),g&7u);
    }
}
}


#if defined(__CUDACC__) || defined(__HGGC__)
#define QG_HD __host__ __device__ __forceinline__
#else
#define QG_HD inline
#endif
namespace quactlize::execution::simt {
template<int Mode> QG_HD q4_s1::Row common_locate(qkg_call_v1 const& c,unsigned row) {
    static_assert(Mode==QKG_DENSE || Mode==QKG_GROUPED || Mode==QKG_INDEXED);
    int expert=0;
    int64_t a=int64_t(row)*c.a_row_stride;
    if constexpr(Mode==QKG_INDEXED) {
        unsigned const token=row/unsigned(c.topk),slot=row%unsigned(c.topk);
        expert=c.ids[int64_t(token)*c.ids_stride+slot];
        a=int64_t(token)*c.a_token_stride+(slot%unsigned(c.channels))*c.a_row_stride;
    } else if constexpr(Mode==QKG_GROUPED) {
        unsigned lo=0,hi=unsigned(c.experts);
        while(lo+1<hi) {
            unsigned const mid=lo+(hi-lo)/2;
            if(c.offsets[mid]<=int(row))lo=mid;else hi=mid;
        }
        expert=int(lo);
    }
    return {expert,a,int64_t(row)*c.out_row_stride};
}
}
#undef QG_HD

namespace quactlize::execution::simt {
template<int Q> struct CommonFormat : Format<Q> {
    using F=Format<Q>;
    using Low=common_map::Plane<F::low_bits>;
    using High=std::conditional_t<Q==13,common_map::Q5High,common_map::Plane<(F::high_bits?F::high_bits:1)>>;
    __host__ __device__ static uint64_t unit_offset(unsigned n,unsigned col,unsigned g) {
        using U=typename F::Unit;
        return (uint64_t(g/(U::kGroups*U::kSbPerUnit))*n+col)*U::kUnitTotal;
    }
};

template<int Q>
__device__ __forceinline__ float2 common_codes(uint32_t low,uint32_t high,unsigned col,unsigned k) {
    using F=CommonFormat<Q>;
    constexpr uint32_t mask=(1u<<F::low_bits)-1;
    uint32_t raw=(low>>(F::low_bits*F::Low::word_slot(k)))&(mask|(mask<<16));
    if constexpr(F::high_bits) {
        unsigned slot;
        if constexpr(Q==13) slot=F::High::word_slot(col,k);
        else slot=F::High::word_slot(k);
        constexpr uint32_t hmask=(1u<<F::high_bits)-1;
        raw|=((high>>(F::high_bits*slot))&(hmask|(hmask<<16)))<<F::low_bits;
    }
    // Two exact integer conversions from the packed mantissas, no scalar
    // integer-to-FP32 conversion per code. Accumulators below remain F32.
    raw|=0x64006400u;
    __half2_raw h;h.x=uint16_t(raw);h.y=uint16_t(raw>>16);
    return __half22float2(__hsub2(__half2(h),__float2half2_rn(1024.f+F::bias)));
}
template<int Mode,int Q,int Input,int Variant,int Columns,int Warps,int P,int Compute=0,int Changes=0,
         class Finish=void,class Call=qkg_call_v1>
__device__ __forceinline__ void common_reuse_body(Call call,int split) {
    using F=CommonFormat<Q>;
    static_assert(Columns==4 || Columns==8);
    constexpr int TileN=Columns*P,Workers=Warps*32/Columns,Pairs=P/2;
    constexpr int Segments=(F::group+F::Low::kLogicalKPerDelivery-1)/F::Low::kLogicalKPerDelivery;
    using Index=unsigned;
    Index tid=threadIdx.x,lane=tid%32,worker=tid/Columns;
    Index tile=blockIdx.x,partition=blockIdx.z,row=blockIdx.y;
    auto r=common_locate<Mode>(call,row);
    bool valid=r.expert>=0 && r.expert<call.experts;
    if (!valid) {
        if constexpr(!std::is_void_v<Finish>) {
            Finish::template invalid<TileN>(call,row,tile,partition,split);
        } else if (tid<TileN) {
            int n=tile*TileN+tid;
            if (split==1) call.output[r.output+n]=__int_as_float(0x7fc00000);
            else static_cast<float*>(call.workspace)[(int64_t(row)*split+partition)*call.n+n]=__int_as_float(0x7fc00000);
        }
        return;
    }
    Index col=tile*TileN+(tid%Columns)*P;
    uint64_t nk=uint64_t(call.n)*call.k;
    auto low=reinterpret_cast<uint16_t const*>(call.low+uint64_t(r.expert)*nk/8*F::low_bits);
    uint16_t const* high=nullptr;
    if constexpr(F::high_bits) high=reinterpret_cast<uint16_t const*>(call.high+uint64_t(r.expert)*nk/8*F::high_bits);
    auto units=call.units+uint64_t(r.expert)*F::metadata_bytes(nk);
    Activation<Input,Compute> a{static_cast<typename Activation<Input,Compute>::Scalar const*>(call.a)+r.a};
    float2 total[Pairs]{};
    for (Index g=partition*Workers+worker;g<call.k/F::group;g+=split*Workers) {
        // K/group is divisible by the number of workers in a warp. Tail
        // warps are wholly inactive, so full-mask exchanges remain legal.
        common_metadata::Packet cooperative{};
        if constexpr(Variant&2) {
            if (lane<TileN) cooperative=common_metadata::load<Q>(units+F::unit_offset(call.n,tile*TileN+lane,g),g);
        }
        float2 metadata[P];
        #pragma unroll
        for (int p=0;p<P;++p) {
            if constexpr(Variant&2)
                metadata[p]=common_metadata::affine<Q>(common_metadata::share(cooperative,(lane%Columns)*P+p),g);
            else metadata[p]=common_metadata::affine<Q>(common_metadata::load<Q>(units+F::unit_offset(call.n,col+p,g),g),g);
        }
        uint4 packet{};
        if constexpr(Variant&1) packet=activation_packet<Input,Compute,Columns,F::group>(a,g,lane);
        float2 dot[Pairs]{};float a_sum=0.f;
        #pragma unroll
        for (int half=0;half<2;++half) {
            uint32_t lo[Segments][4][Pairs],hi[4][Pairs]{};
            #pragma unroll
            for (int segment=0;segment<Segments;++segment) {
                #pragma unroll
                for (int residue=0;residue<4;++residue) {
                    unsigned k=g*F::group+segment*F::Low::kLogicalKPerDelivery+half*4+residue;
                    load_words<P>(low+F::Low::word_index(col,k,call.n),lo[segment][residue]);
                }
            }
            if constexpr(F::high_bits) {
                #pragma unroll
                for (int residue=0;residue<4;++residue) {
                    unsigned k=g*F::group+half*4+residue;
                    load_words<P>(high+F::High::word_index(col,k,call.n),hi[residue]);
                }
            }
            #pragma unroll
            for (int slot=0;slot<F::group/8;++slot) {
                float4 av;
                if constexpr(Variant&1) {
                    float2 x=activation_pair<Compute,Columns,F::group>(packet,slot*8+half*4,lane);
                    float2 y=activation_pair<Compute,Columns,F::group>(packet,slot*8+half*4+2,lane);
                    av=make_float4(x.x,x.y,y.x,y.y);
                } else av=a.values4(g*F::group+slot*8+half*4);
                float values[4]={av.x,av.y,av.z,av.w};
                a_sum+=(av.x+av.y)+(av.z+av.w);
                #pragma unroll
                for (int residue=0;residue<4;++residue) {
                    unsigned k=g*F::group+slot*8+half*4+residue;
                    #pragma unroll
                    for (int p=0;p<Pairs;++p) {
                        float2 code=common_codes<Q>(lo[slot/F::Low::kPack][residue][p],hi[residue][p],col+2*p,k);
                        dot[p].x=fmaf(values[residue],code.x,dot[p].x);
                        dot[p].y=fmaf(values[residue],code.y,dot[p].y);
                    }
                }
            }
        }
        #pragma unroll
        for (int p=0;p<Pairs;++p) {
            total[p].x+=fmaf(metadata[2*p].x,dot[p].x,metadata[2*p].y*a_sum);
            total[p].y+=fmaf(metadata[2*p+1].x,dot[p].y,metadata[2*p+1].y*a_sum);
        }
    }
    float values[P];
    #pragma unroll
    for (int p=0;p<Pairs;++p) {values[2*p]=total[p].x;values[2*p+1]=total[p].y;}
    float value=q4_s1::q4_reduce_scatter_steps<P,Columns>(values,lane);
    __shared__ float partial[Warps*TileN];
    if (lane<TileN) partial[(tid/32)*TileN+(lane%Columns)*P+lane/Columns]=value;
    __syncthreads();
    if constexpr(!std::is_void_v<Finish>) {
        Finish::template finish<TileN,Warps>(call,row,tile,partition,split,partial);
    } else if (tid<32) {
        float sum=0;
        if constexpr(Changes&2) {
            q4_s1::q4_medium_fold<0,Warps,TileN>(sum,partial,unsigned(tid));
        } else {
            #pragma unroll
            for (int w=tid/TileN;w<Warps;w+=32/TileN) sum+=partial[w*TileN+tid%TileN];
        }
        #pragma unroll
        for (int d=TileN;d<32;d*=2) sum+=__shfl_xor_sync(0xffffffffu,sum,d);
        if (tid<TileN) {
            int n=tile*TileN+tid;
            if (split==1) call.output[r.output+n]=sum;
            else static_cast<float*>(call.workspace)[(int64_t(row)*split+partition)*call.n+n]=sum;
        }
    }
}

template<int Mode,int Q,int C,int W,int P,int Compute>
__global__ void common_reader(qkg_call_v1 call,int split) {
    constexpr int Changes=kBf16F32Changes<Q,3,C,W,P> * Compute;
    common_reuse_body<Mode,Q,QKG_F32,3,C,W,P,Compute,Changes>(call,split);
}

template<int Q,int C,int W,int P,int Compute>
int common_reader_launch_typed(qkg_call_v1 const& c,int split) {
    static_assert(kCommonReader<Q,3,C,W,P>);
    auto stream=static_cast<hggcStream_t>(c.stream);
    if(hggcGetLastError()!=hggcSuccess)return QKG_RUNTIME;
    dim3 grid(c.n/(C*P),c.rows,split);
    switch(c.mode) {
        case QKG_DENSE:common_reader<QKG_DENSE,Q,C,W,P,Compute><<<grid,W*32,0,stream>>>(c,split);break;
        case QKG_GROUPED:common_reader<QKG_GROUPED,Q,C,W,P,Compute><<<grid,W*32,0,stream>>>(c,split);break;
        case QKG_INDEXED:common_reader<QKG_INDEXED,Q,C,W,P,Compute><<<grid,W*32,0,stream>>>(c,split);break;
        default:return QKG_INVALID;
    }
    if(hggcGetLastError()!=hggcSuccess)return QKG_RUNTIME;
    launch_reduction<Q>(c,split,stream);
    return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME;
}

template<int Q,int C,int W,int P>
int common_reader_launch(qkg_simt_call_v2 const& d,int split) {
    if(d.compute_type==QKG_COMPUTE_F16)return common_reader_launch_typed<Q,C,W,P,0>(d.call,split);
    if(d.compute_type==QKG_COMPUTE_BF16)return common_reader_launch_typed<Q,C,W,P,1>(d.call,split);
    return QKG_INVALID;
}
}
