// Same BF16 controls as shipping d8f9a260, plus explicit experimental entries.
#include "../../quactlize/moe/scaffold.cu"
#include "kernels.cuh"

extern "C" int qzm_prefill_fast(int phase,void const* c) {
    using namespace quactlize::moe::prefill_helpers;
    if(!c) return 0;
    if(phase==0) return prepare_fast(*static_cast<qzm_prepare_v1 const*>(c));
    if(phase==1) return transition_fast(*static_cast<qzm_transition_v1 const*>(c));
    return phase==2?finish_fast(*static_cast<qzm_finish_v1 const*>(c)):0;
}
extern "C" int qzm_prefill_prepare(qzm_prepare_v1 const* c,int arm,int stage) {
    using namespace quactlize::moe;
    if(!c || arm<0 || arm>2 || stage<-1 || stage>2) return QZR_INVALID;
    quactlize::routing::Layout l;int rc=prepare(*c,l,true);if(rc) return rc;
    if(arm==0 || !prefill_helpers::prepare_fast(*c)) return run_prepare(c,stage);
    if(stage<0) {
        for(int s=0;s<2;++s) {rc=run_prepare(c,s);if(rc) return rc;}
    } else if(stage<2) return run_prepare(c,stage);
    auto prefix=reinterpret_cast<int const*>(static_cast<uint8_t const*>(c->route.workspace)+l.prefix);
    auto ranks=reinterpret_cast<int const*>(static_cast<uint8_t const*>(c->route.workspace)+l.ranks);
    int rows=arm==2?c->route.tokens:l.rows;
    dim3 grid((int64_t(c->columns)+1023)/1024,rows>65535?65535:rows);
    auto stream=static_cast<hggcStream_t>(c->route.stream);
    if(arm==2) prefill_helpers::prepare_vec4<true><<<grid,256,0,stream>>>(*c,l.rows,prefix,ranks);
    else prefill_helpers::prepare_vec4<false><<<grid,256,0,stream>>>(*c,l.rows,prefix,ranks);
    return launch_status();
}
extern "C" int qzm_prefill_finish(qzm_finish_v1 const* c,int arm) {
    using namespace quactlize::moe;
    if(!c || arm<0 || arm>2) return QZR_INVALID;
    int rc=finish(*c,true);if(rc || !c->tokens) return rc;
    if(arm==0 || !prefill_helpers::finish_fast(*c)) return quactlize_ppu_moe_finish_run_v1(c);
    return arm==1?prefill_helpers::finish_launch<128>(*c):prefill_helpers::finish_launch<256>(*c);
}
extern "C" int qzm_prefill_transition(qzm_transition_v1 const* c,int arm,float const* weights,
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
    if(arm==0 || !prefill_helpers::transition_fast(*c))
        return snapshot?transition_snapshot(c,weights,stride,retained):quactlize_ppu_moe_transition_run_v1(c);
    if(snapshot) return arm==1?prefill_helpers::transition_launch<128,true>(*c,weights,stride,retained):
                              prefill_helpers::transition_launch<256,true>(*c,weights,stride,retained);
    return arm==1?prefill_helpers::transition_launch<128,false>(*c,nullptr,0,nullptr):
                  prefill_helpers::transition_launch<256,false>(*c,nullptr,0,nullptr);
}
