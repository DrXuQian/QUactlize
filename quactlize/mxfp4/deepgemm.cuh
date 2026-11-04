#pragma once
#include "e8m0.hpp"
#include "dequant_w4a16.cuh"

// Keep Sail's MMA, scheduler, FP4 decoder and scale permutation. Correct only
// the exponent endpoints: its generic E8M0 unpack maps byte 0 to zero and
// byte 255 to infinity. Specialize BEFORE the kernel header instantiates it.
namespace deep_gemm {
template <> struct DequantE8M0<8> {
    CUTLASS_DEVICE static void convert(uint8_t const* src,cutlass::bfloat16_t* dst) {
        auto out=reinterpret_cast<uint16_t*>(dst);
        CUTLASS_PRAGMA_UNROLL
        for(int g=0;g<2;++g) {
            out[4*g+0]=quactlize::mxfp4::scale_bits(src[4*g+0]);
            out[4*g+1]=quactlize::mxfp4::scale_bits(src[4*g+2]);
            out[4*g+2]=quactlize::mxfp4::scale_bits(src[4*g+1]);
            out[4*g+3]=quactlize::mxfp4::scale_bits(src[4*g+3]);
        }
    }
};
}
#include "w4a16_gemm_cutlass3.cuh"
