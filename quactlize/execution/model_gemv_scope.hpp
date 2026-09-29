#pragma once
#include "api.h"

namespace quactlize::execution::model_gemv {
// SIMT partials are row-major [row,split,N], not TC [split,row,N].
inline bool vector_reduction(qkg_call_v1 const& c,int split) {
    return (split==2 || split==4 || split==8) && c.rows>0 && c.n>0 && !(c.n&1) &&
        c.out_row_stride>=c.n && !(c.out_row_stride&1) && c.workspace && c.output &&
        !((uintptr_t(c.workspace)|uintptr_t(c.output))&7);
}

inline bool dense_m1(qkg_call_v1 const& c) {
    return c.input_type==QKG_F32 && c.mode==QKG_DENSE && c.rows==1 &&
        c.experts==1 && c.topk==1 && c.channels==1 && c.n>0 && c.k>0;
}
inline bool dense_m1(qkg_call_v1 const& c, int n, int k) {
    return dense_m1(c) && c.n==n && c.k==k;
}
inline bool indexed_m1(qkg_call_v1 const& c, int n, int k, int channels) {
    return c.input_type==QKG_F32 && c.mode==QKG_INDEXED && c.rows==8 &&
        c.experts==256 && c.topk==8 && c.channels==channels && c.n==n && c.k==k;
}
}
