#pragma once
#include <cstdint>

#if defined(__CUDACC__) || defined(__HGGCCC__) || defined(__HGGC__)
#define QZ_MX_SCALE_HD __host__ __device__
#else
#define QZ_MX_SCALE_HD
#endif
namespace quactlize::mxfp4 {
// OCP E8M0: zero exponent is 2^-127, NOT zero; 255 is NaN.
// This conversion is exact in BF16 and does not change stored scale bytes.
QZ_MX_SCALE_HD inline uint16_t scale_bits(uint8_t e) {
    return e==0?0x0040:e==255?0x7fc0:uint16_t(e)<<7;
}
}
#undef QZ_MX_SCALE_HD
