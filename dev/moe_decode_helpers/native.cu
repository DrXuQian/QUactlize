// Compile the unchanged public control and candidates into one helper-only
// carrier. No GEMM, actlize, llama or production route compilation is needed.
#include "../../quactlize/moe/scaffold.cu"
#include "kernels.cuh"

extern "C" int qzm_decode_finish(qzm_finish_v1 const* c,int arm) {
    using namespace quactlize::moe;
    if(!c || arm<0 || arm>3) return QZR_INVALID;
    int rc=finish(*c,true);if(rc || !c->tokens) return rc;
    if(arm==0) return quactlize_ppu_moe_finish_run_v1(c);
    if(arm==1) return finish_parallel(c,4);
    if(!decode_helpers::finish_fast(*c)) return quactlize_ppu_moe_finish_run_v1(c);
    return arm==2?decode_helpers::finish_launch<128>(*c):decode_helpers::finish_launch<64>(*c);
}
extern "C" int qzm_decode_transition(qzm_transition_v1 const* c,int arm,float const* weights,
                                      int64_t stride,float* retained) {
    using namespace quactlize::moe;
    if(!c || arm<0 || arm>2) return QZR_INVALID;
    uint32_t ops=0;int rc=transition(*c,ops,true);if(rc || ops==QZM_NOOP) return rc;
    bool snapshot=weights || retained;
    if(snapshot) {
        if(!(ops&QZM_SWIGLU)) return QZR_INVALID;
        uint64_t bytes=0;rc=matrix_bytes(c->tokens,c->topk,stride,QZM_F32,bytes);if(rc) return rc;
        Spans spans;
        if(spans.add(weights,bytes,4,false) || spans.add(retained,uint64_t(c->tokens)*c->topk*4,4,true)) return QZR_INVALID;
    }
    if(arm==0 || !decode_helpers::transition_fast(*c))
        return snapshot?transition_snapshot(c,weights,stride,retained):quactlize_ppu_moe_transition_run_v1(c);
    if(snapshot) return arm==1?decode_helpers::transition_launch<128,true>(*c,weights,stride,retained):
                              decode_helpers::transition_launch<256,true>(*c,weights,stride,retained);
    return arm==1?decode_helpers::transition_launch<128,false>(*c,nullptr,0,nullptr):
                  decode_helpers::transition_launch<256,false>(*c,nullptr,0,nullptr);
}
