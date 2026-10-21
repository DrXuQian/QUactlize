#pragma once
#include "reader.hpp"

namespace quactlize::dequant {
template<int Groups> struct DenseAffineLayout {
    static_assert(Groups==4 || Groups==8 || Groups==16);
    // Preserve row-local groups, xor high row bits into group bits so N-fast
    // metadata producers do not all store to only one/two shared banks.
    CUTLASS_HOST_DEVICE static constexpr int offset(int row,int group) {
        return row*Groups+(group^(row/(32/Groups)));
    }
};
template<KType T,bool BF16> struct DenseReader : Reader<T> {
    using R=Reader<T>;using Affine=typename R::Affine;
    CUTLASS_HOST_DEVICE static uint16_t weight(int raw,Affine a) {
        constexpr int bias=T==KType::Q3_K?4:T==KType::Q6_K?32:0;
#if defined(__CUDA_ARCH__) || defined(__HGGC_ARCH__)
        float value=__fsub_rn(__fmul_rn(float(raw-bias),a.scale),a.minimum);
#else
        volatile float product=float(raw-bias)*a.scale;
        float value=product-a.minimum;
#endif
        if constexpr(BF16) return cutlass::bfloat16_t(value).raw();
        else return cutlass::half_t(value).raw();
    }
};
template<KType T> struct DenseCodes {
    CUTLASS_HOST_DEVICE static uint32_t combine(uint16_t low,uint16_t high,int col,int k) {
        uint32_t codes=0;
        CUTLASS_PRAGMA_UNROLL
        for(int s=0;s<4;++s)codes|=uint32_t(Reader<T>::raw_from_words(low,high,col,k+s*8))<<(s*8);
        return codes;
    }
    CUTLASS_HOST_DEVICE static int code(uint32_t word,int slot){return (word>>(slot*8))&255;}
};
}
