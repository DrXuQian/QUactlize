#pragma once
#include "quactlize/execution/q4_s1_validation.hpp"

#if defined(__CUDACC__) || defined(__HGGC__)
#define QG_HD __host__ __device__ __forceinline__
#else
#define QG_HD inline
#endif
namespace quactlize::execution::simt {
template<int Mode> QG_HD q4_s1::Row general_locate(qkg_call_v1 const& c,unsigned row) {
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
