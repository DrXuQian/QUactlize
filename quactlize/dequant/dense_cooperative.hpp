#pragma once
#include "dense_reader.hpp"

namespace quactlize::dequant {

// The two planes have different element sizes. A word swizzle must not be
// reused blindly for halfwords. Keep each consumer's four adjacent residues
// aligned and contiguous, and spread the producer's N8 cohorts separately.
template<unsigned N,unsigned K> struct DenseCooperativeLayout {
    static_assert(N==16 || N==32 || N==64);
    static_assert(K==128 || K==256);
    static constexpr unsigned V=N/8,LogN=N==16?4:N==32?5:6;
    static constexpr unsigned Words=K/4,Cells=N/2*Words;
    CUTLASS_HOST_DEVICE static constexpr unsigned low(unsigned pair,unsigned word) {
        unsigned address=pair*Words+word;
        unsigned fold=(pair&((N/2-1)&~3u))<<(6-LogN);
        if constexpr(K==256)fold^=(word&32u)>>3;
        return address^fold;
    }
    CUTLASS_HOST_DEVICE static constexpr unsigned high(unsigned pair,unsigned word) {
        return (pair*Words+word)^((pair&((N/2-1)&~3u))<<(7-LogN));
    }
    // Low producer keeps contiguous N8 cohorts. The high producer moves
    // residue parity outside a warp so separate threads do not update the
    // two halves of one shared word in the same warp store instruction.
    struct Producer { unsigned col,residue,group; };
    CUTLASS_HOST_DEVICE static constexpr Producer producer(unsigned i,bool upper) {
        unsigned t=i/V;
        if(!upper)return {(i%V)*8,t%8,t/8};
        constexpr unsigned P=8-LogN,Mask=(1u<<(P-2))-1;
        return {(i%V)*8,((t&3u)<<1)|((t>>P)&1u),((t>>2)&Mask)|((t>>(P+1))<<(P-2))};
    }
};

template<unsigned Groups> struct DenseCooperativeAffine {
    static_assert(Groups==4 || Groups==8 || Groups==16);
    CUTLASS_HOST_DEVICE static constexpr unsigned offset(unsigned col,unsigned group) {
        return col*Groups+(group^((col/(16/Groups))&(Groups-1)));
    }
};

// Each owner expands one column's groups from a staged canonical unit.
// Byte selection is fixed-register arithmetic, not a dynamically indexed
// private byte array. Header F16->F32 conversion occurs once per column.
template<KType T,unsigned TileK> struct DenseRegisterAffine {
    static_assert(T==KType::Q4_K || T==KType::Q5_K || T==KType::Q6_K);
    using R=DenseUnsignedReader<T,false>;
    float d,dm;
    uint32_t x,y,z,w;
    CUTLASS_HOST_DEVICE static DenseRegisterAffine load(uint32_t const* p,unsigned k0) {
        DenseRegisterAffine r{};
        if constexpr(T==KType::Q6_K) {
            unsigned sb=(k0>>8)&1u;
            uint32_t a=p[sb*4],b=p[sb*4+1],c=p[sb*4+2],e=p[sb*4+3],f=p[sb*4+4];
            uint16_t header=uint16_t(sb?a>>16:a);
            r.d=float(cutlass::half_t::bitcast(header));
            r.x=sb?b:(a>>16)|(b<<16);r.y=sb?c:(b>>16)|(c<<16);
            r.z=sb?e:(c>>16)|(e<<16);r.w=sb?f:(e>>16)|(f<<16);
            if constexpr(TileK==128)if(k0&128u){r.x=r.z;r.y=r.w;}
        } else {
            uint32_t a=p[0],b=p[1],c=p[2],e=p[3];
            r.d=float(cutlass::half_t::bitcast(uint16_t(a)));
            r.dm=float(cutlass::half_t::bitcast(uint16_t(a>>16)));
            r.x=b;r.y=(b>>24)|(c<<8);r.z=(c>>16)|(e<<16);r.w=e>>8;
            if constexpr(TileK==128)if(k0&128u){r.x=r.z;r.y=r.w;}
        }
        return r;
    }
    CUTLASS_HOST_DEVICE typename R::Affine group(unsigned g) const {
        if constexpr(T==KType::Q6_K) {
            uint32_t code=g<4?x:g<8?y:g<12?z:w;
            int scale=int((code>>(8*(g&3)))&255u);scale=(scale^128)-128;
            return {d*float(scale),0.f};
        } else {
            unsigned scale=((g<4?x:z)>>(6*(g&3)))&63u;
            unsigned minimum=((g<4?y:w)>>(6*(g&3)))&63u;
            return {d*float(scale),dm*float(minimum)};
        }
    }
};
} // namespace quactlize::dequant
