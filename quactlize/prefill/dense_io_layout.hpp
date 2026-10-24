#pragma once
#include <cstdint>
#include <climits>

namespace quactlize::prefill {
// All int intermediates in the native chunk body, including 2*c+1, must fit.
// The generic entry validates spans/overlap independently of this fast path.
inline bool dense_cont_eligible(uintptr_t src,uintptr_t dst,int rows,int width,int64_t stride) {
    uint64_t const count=uint64_t(rows)*uint64_t(width);
    return rows>0 && width>0 && !(src%16 || dst%16) && stride==width &&
           count%8==0 && count/8<=uint64_t(INT_MAX)/2;
}
// Same eight elements/thread and CTA work as vec8. Each individual FP32
// instruction now spans contiguous float4s across the warp, not alternating
// 16-byte halves of 32-byte lane records. Both parts are independent.
template<unsigned Threads> struct DenseStripe4 {
#if defined(__CUDACC__) || defined(__HGGCC__)
    __host__ __device__
#endif
    static constexpr uint64_t element(uint64_t block,unsigned thread,unsigned part) {
        return block*(Threads*8u)+(thread+part*Threads)*4u;
    }
};
// Four independent, fully populated vec4 strips. The caller must check
// alignment, contiguous rows and an exact whole-CTA extent before launch.
template<unsigned Threads> struct DenseFull4 {
#if defined(__CUDACC__) || defined(__HGGCC__)
    __host__ __device__
#endif
    static constexpr uint64_t element(uint64_t block,unsigned thread,unsigned part) {
        return block*(Threads*16u)+(thread+part*Threads)*4u;
    }
};
}
