#pragma once
#include "reader.hpp"
#include "q4_s1_helpers.cuh"

namespace quactlize::execution::simt {

template<int Q> struct Format {
    static constexpr auto type=gguf_scale::KType(Q-10);
    using Unit=gguf_scale::packed_unit::Unit<type>;
    using Scale=gguf_scale::Traits<type>;
    using Base=Reader<type>;
    using Low=typename Base::LowMap;
    using High=typename Base::HighMap;
    static constexpr int low_bits=Base::lo_bits, high_bits=Base::hi_bits;
    static constexpr int group=Base::group, words=Unit::kUnitTotal/4;
    static constexpr int bias=Q==11 ? 4 : Q==14 ? 32 : 0;

    __host__ __device__ static uint64_t metadata_bytes(uint64_t nk) {
        return nk/256*Unit::kSbBytes;
    }
    __host__ __device__ static uint64_t unit_offset(int n,int col,int g) {
        return (uint64_t(g/(Unit::kGroups*Unit::kSbPerUnit))*n+col)*Unit::kUnitTotal;
    }
};

template<> struct Format<8> {
    using Low=q8_kpack2::Map;
    static constexpr int low_bits=8, high_bits=0, group=32, words=1, bias=128;
    __host__ __device__ static uint64_t metadata_bytes(uint64_t nk) { return nk/16; }
    __host__ __device__ static uint64_t unit_offset(int n,int col,int g) {
        return (uint64_t(g)*n+col)*2;
    }
};

template<int Count> struct Meta { uint32_t word[Count]; };

template<int Q>
__device__ __forceinline__ auto load_meta(uint8_t const* units,int n,int col,int group) {
    using F=Format<Q>;
    Meta<F::words> result{};
    auto p=units+F::unit_offset(n,col,group);
    if constexpr(Q==8) result.word[0]=*reinterpret_cast<uint16_t const*>(p);
    else if constexpr(F::words==4) {
        uint4 v=*reinterpret_cast<uint4 const*>(p);
        result.word[0]=v.x;result.word[1]=v.y;result.word[2]=v.z;result.word[3]=v.w;
    } else {
        // Q2/Q3/Q6 units are 20/28/36 bytes: only four-byte alignment is
        // guaranteed. No uint4 overread or padding of the offline artifact.
        #pragma unroll
        for (int i=0;i<F::words;++i) result.word[i]=reinterpret_cast<uint32_t const*>(p)[i];
    }
    return result;
}

template<int Count>
__device__ __forceinline__ Meta<Count> share_meta(Meta<Count> value,int owner) {
    #pragma unroll
    for (int i=0;i<Count;++i) value.word[i]=__shfl_sync(0xffffffffu,value.word[i],owner);
    return value;
}

template<int Count>
__host__ __device__ uint32_t meta_bits(Meta<Count> const& m,int bit,int width) {
    int i=bit/32,shift=bit%32;
    uint32_t v=m.word[i]>>shift;
    if (shift+width>32) v|=m.word[i+1]<<(32-shift);
    return v&((1u<<width)-1);
}

template<int Q>
__device__ __forceinline__ float2 affine(Meta<Format<Q>::words> const& m,int group) {
    if constexpr(Q==8) return make_float2(__half2float(__ushort_as_half(uint16_t(m.word[0]))),0.f);
    else {
        using F=Format<Q>;using U=typename F::Unit;using T=typename F::Scale;
        int g=group%U::kGroups,sb=(group/U::kGroups)%U::kSbPerUnit;
        int bit=sb*U::kSbBytes*8;
        float d=__half2float(__ushort_as_half(uint16_t(meta_bits(m,bit,16))));
        int sc=meta_bits(m,bit+U::bit_of(g,0),U::kScaleBits);
        if constexpr(T::kSigned) sc=(sc^128)-128;
        sc-=T::kScaleBias;
        float zero=0.f;
        if constexpr(U::kHasMin) {
            float dm=__half2float(__ushort_as_half(uint16_t(meta_bits(m,bit+16,16))));
            zero=-dm*float(meta_bits(m,bit+U::bit_of(g,1),U::kMinBits));
        }
        return make_float2(d*float(sc),zero);
    }
}

template<int Q>
__device__ __forceinline__ float2 codes(uint32_t low,uint32_t high,int col,int k) {
    using F=Format<Q>;
    constexpr uint32_t mask=(1u<<F::low_bits)-1;
    uint32_t raw=(low>>(F::low_bits*F::Low::word_slot(k)))&(mask|(mask<<16));
    if constexpr(F::high_bits) {
        int slot;
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
} // namespace quactlize::execution::simt
