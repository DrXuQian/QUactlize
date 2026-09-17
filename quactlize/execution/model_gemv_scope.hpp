#pragma once
#include "api.h"

namespace quactlize::execution::model_gemv {
inline bool dense_m1(qkg_call_v1 const& c, int n, int k) {
    return c.input_type==QKG_F32 && c.mode==QKG_DENSE && c.rows==1 &&
        c.experts==1 && c.topk==1 && c.channels==1 && c.n==n && c.k==k;
}
inline bool indexed_m1(qkg_call_v1 const& c, int n, int k, int channels) {
    return c.input_type==QKG_F32 && c.mode==QKG_INDEXED && c.rows==8 &&
        c.experts==256 && c.topk==8 && c.channels==channels && c.n==n && c.k==k;
}
}
