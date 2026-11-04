#pragma once
#include "matmul.h"
#include "validation.hpp"
#include "../dense/validation.hpp"

namespace quactlize::moe::matmul {
inline int admission(quactlize_ppu_moe_matmul_params_v1 const* p) {
    if(!p || p->version!=1 || p->size!=sizeof(*p)) return -1;
    if(p->reserved || (p->qtype!=8 && p->qtype!=39 && (p->qtype<10 || p->qtype>14)) ||
       p->tokens<0 || p->n<=0 || p->n>65535*16 || p->n%256 || p->k<=0 || p->k>INT_MAX/8 ||
       p->k%((p->qtype==11 || p->qtype==14)?512:256) || p->experts<1 || p->experts>1024 ||
       p->topk<1 || p->topk>32 || p->topk>p->experts || p->channels<1 || p->channels>p->topk ||
       p->topk%p->channels || p->input_row_stride<p->k || p->input_row_stride%4 || p->output_stride<p->n ||
       p->ids_stride<p->topk || p->input_token_stride<=0 || p->input_token_stride%p->input_row_stride ||
       p->input_token_stride/p->input_row_stride<p->channels) return 0;
    uint64_t m=p->tokens?p->tokens:1;
    uint64_t rows=m*p->topk;
    if(rows>uint64_t(INT_MAX)/(p->n>p->k?p->n:p->k) ||
       uint64_t(p->input_token_stride)>uint64_t(INT64_MAX)/4/rows ||
       uint64_t(p->output_stride)>uint64_t(INT64_MAX)/4/rows ||
       uint64_t(p->ids_stride)>uint64_t(INT64_MAX)/4/rows ||
       (m-1)*uint64_t(p->input_token_stride/p->input_row_stride)+p->channels>INT_MAX) return 0;
    quactlize_ppu_placed_arrangement_v2 arrangement{};
    quactlize_ppu_kpack_sizes_v1 sizes{};
    return !quactlize_ppu_kpack_canonical_arrangement_v1(p->qtype,&arrangement) &&
        !std::memcmp(&arrangement,&p->arrangement,sizeof(arrangement)) &&
        !quactlize_ppu_kpack_sizes_for_arrangement_v1(p->n,p->k,p->experts,p->qtype,&arrangement,&sizes) &&
        !std::memcmp(&sizes,&p->sizes,sizeof(sizes));
}
inline bool buffers(quactlize_ppu_moe_matmul_params_v1 const& p,void const* low,void const* high,
        void const* units,float const* a,int32_t const* ids,float* output,
        void* workspace,uint64_t available,uint64_t needed) {
    if(available<needed) return false;
    uint64_t ab=(uint64_t(p.tokens-1)*p.input_token_stride+uint64_t(p.channels-1)*p.input_row_stride+p.k)*4;
    uint64_t cb=(uint64_t(p.tokens*p.topk-1)*p.output_stride+p.n)*4;
    uint64_t ib=(uint64_t(p.tokens-1)*p.ids_stride+p.topk)*4;
    Spans spans;
    return !spans.add(low,p.sizes.low_bytes,16,false) && !spans.add(high,p.sizes.high_bytes,16,false) &&
        !spans.add(units,p.sizes.units_bytes,16,false) && !spans.add(a,ab,16,false) &&
        !spans.add(ids,ib,4,false) && !spans.add(output,cb,16,true) &&
        !spans.add(workspace,needed,256,true);
}
} // namespace quactlize::moe::matmul
