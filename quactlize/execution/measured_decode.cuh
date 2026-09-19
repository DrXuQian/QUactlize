#pragma once
#include "simt_q8_vector.cuh"
#include "measured_decode.hpp"

namespace quactlize::execution::simt {
// Reuse the exact measured arithmetic bodies. Split is a compile-time value
// even for dynamic N/K; this was part of the experiment's native kernel.
template<int Q,int Mode,int N,int K,int E,int Top,int Ch,int Compute,
         int V,int C,int W,int P,int S,int Changes,bool Hoist,bool Fixed>
__global__ void measured_decode_kernel(qkg_call_v1 c) {
    if constexpr(Fixed) {
        c.n=N;c.k=K;c.experts=E;c.mode=Mode;c.channels=Ch;c.topk=Top;
    }
    if constexpr(Q==8)
        q8_vector::kernel_body<QKG_F32,Compute,V-4,C,W,P,Hoist>(c,S);
    else register_reuse_body<Q,QKG_F32,V,C,W,P,Compute,Changes>(c,S);
}

template<int Qtype>
int measured_decode_launch(qkg_simt_call_v2 const& d,qkg_simt_config_v1 const& f) {
    auto id=measured_decode(d,f);
    if(id==MeasuredDecode::None) return QKG_SHAPE;
    auto const& c=d.call;auto stream=static_cast<hggcStream_t>(c.stream);
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    switch(id) {
#define QDM(Id,Q,Mode,N,K,E,Top,Ch,Compute,V,C,W,P,S,Changes,Hoist,Fixed) \
    case MeasuredDecode::Id: \
        if constexpr(Qtype==Q) { \
            measured_decode_kernel<Q,Mode,N,K,E,Top,Ch,Compute,V,C,W,P,S,Changes,Hoist,Fixed> \
                <<<c.rows*S*(c.n/(C*P)),W*32,0,stream>>>(c); \
            if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME; \
            launch_reduction<Q>(c,S,stream); \
            return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME; \
        } \
        return QKG_SHAPE;
#include "measured_decode.inc"
#undef QDM
        default:return QKG_SHAPE;
    }
}
}
