#pragma once
#include "api.h"
#include "../moe/matmul_validation.hpp"

namespace quactlize::ffn {
inline auto const& weight(quactlize_ppu_ffn_params_v1 const& p,int stage) {
    return stage==0?p.gate:stage==1?p.up:p.down;
}
inline quactlize_ppu_dense_params_v1 dense_params(quactlize_ppu_ffn_params_v1 const& p,int stage) {
    auto const& w=weight(p,stage);
    return {1,sizeof(quactlize_ppu_dense_params_v1),w.qtype,w.n,w.k,p.tokens,
        stage==2?w.k:p.input_token_stride,stage==2?p.output_stride:w.n,w.arrangement,w.sizes};
}
inline quactlize_ppu_moe_matmul_params_v1 moe_params(quactlize_ppu_ffn_params_v1 const& p,int stage) {
    auto const& w=weight(p,stage);
    return {1,sizeof(quactlize_ppu_moe_matmul_params_v1),w.qtype,w.n,w.k,w.experts,p.tokens,p.topk,
        stage==2?p.topk:p.channels,0,stage==2?w.k:p.input_row_stride,
        stage==2?int64_t(w.k)*p.topk:p.input_token_stride,p.ids_stride,w.n,w.arrangement,w.sizes};
}
inline int admission(quactlize_ppu_ffn_params_v1 const* p) {
    if(!p || p->version!=1 || p->size!=sizeof(*p)) return -1;
    if(p->reserved || p->tokens<0 || p->kind<0 || p->kind>1 || p->merged<0 || p->merged>1 ||
       p->down.k<=0 || p->down.k>INT_MAX/2 || p->gate.n!=p->down.k*(1+p->merged) ||
       p->gate.experts!=p->down.experts || p->output_stride<p->down.n) return 0;
    if(p->merged) {
        quactlize_ppu_ffn_weight_v1 empty{};
        if(std::memcmp(&p->up,&empty,sizeof(empty))) return 0;
    } else if(p->up.n!=p->gate.n || p->up.k!=p->gate.k || p->up.experts!=p->gate.experts) return 0;
    if(p->kind==QUACTLIZE_FFN_DENSE) {
        if(p->topk!=1 || p->channels!=1 || p->gate.experts!=1 || p->ids_stride || p->weights_stride ||
           p->input_row_stride<p->gate.k || p->input_token_stride<p->input_row_stride) return 0;
    } else if(p->topk<1 || p->weights_stride<p->topk ||
              uint64_t(p->weights_stride)>uint64_t(INT64_MAX)/4/(p->tokens?p->tokens:1)) return 0;
    for(int i=0;i<3;++i) {
        if(i==1 && p->merged) continue;
        if(p->kind==QUACTLIZE_FFN_DENSE) {
            auto a=dense_params(*p,i);if(dense::admission(&a)!=1) return 0;
        } else {
            auto a=moe_params(*p,i);if(moe::matmul::admission(&a)!=1) return 0;
        }
    }
    uint64_t ignored=0;
    return moe::matrix_bytes(p->tokens,p->down.n,p->output_stride,QZM_F32,ignored)==QZR_OK;
}
inline bool planes(quactlize_ppu_ffn_weight_v1 const& w,moe::Spans& spans) {
    return (w.sizes.high_bytes || !w.high) &&
        !spans.add(w.low,w.sizes.low_bytes,16,false) &&
        !spans.add(w.high,w.sizes.high_bytes,16,false) &&
        !spans.add(w.units,w.sizes.units_bytes,16,false);
}
inline bool buffers(quactlize_ppu_ffn_params_v1 const& p,float const* input,int32_t const* ids,
    float const* weights,float* output,void* scratch,uint64_t available,uint64_t needed,bool allow_alias=false) {
    if(available<needed || (p.kind==QUACTLIZE_FFN_DENSE && (ids || weights))) return false;
    moe::Spans spans;
    for(int i=0;i<3;++i) if(!(i==1 && p.merged) && !planes(weight(p,i),spans)) return false;
    uint64_t a=(uint64_t(p.tokens-1)*p.input_token_stride+uint64_t(p.channels-1)*p.input_row_stride+p.gate.k)*4;
    uint64_t c=(uint64_t(p.tokens-1)*p.output_stride+p.down.n)*4;
    if(p.kind==QUACTLIZE_FFN_MOE) {
        uint64_t ib=(uint64_t(p.tokens-1)*p.ids_stride+p.topk)*4;
        uint64_t wb=(uint64_t(p.tokens-1)*p.weights_stride+p.topk)*4;
        if(spans.add(ids,ib,4,false) || spans.add(weights,wb,4,false)) return false;
    }
    if(spans.add(input,a,16,false) || spans.add(scratch,needed,256,true)) return false;
    if(!allow_alias) return !spans.add(output,c,16,true);
    // A and IDs are consumed before final output. Aliased routing weights
    // are retained by transition; immutable planes and scratch stay disjoint.
    moe::Spans writes;
    for(int i=0;i<3;++i) if(!(i==1 && p.merged) && !planes(weight(p,i),writes)) return false;
    return !writes.add(output,c,16,true) && !writes.add(scratch,needed,256,true);
}
}
