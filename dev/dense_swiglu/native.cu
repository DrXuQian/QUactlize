#include "../../quactlize/moe/scaffold.cu"
#include "../../quactlize/ffn/dense_swiglu.cu"
#include "../moe_decode_helpers/kernels.cuh"

// Test-only entry: compare the actual production dense endpoint with the
// existing generic transition. No copied math or extra public ABI.
extern "C" int qdf_swiglu(qzm_transition_v1 const* c,int arm) {
    using namespace quactlize;
    if(!c || (arm!=0 && arm!=1)) return QZR_INVALID;
    uint32_t ops=0;int rc=moe::transition(*c,ops,true);if(rc || ops==QZM_NOOP) return rc;
    if(!arm) {
        // This body is byte-equal to shipping7112's transition_projected.
        // Shared dense decode already has that fast control; do not regress it
        // or compare the new endpoint only against the older generic body.
        if(c->tokens>=1 && c->tokens<=8 && (c->hidden==512 || c->hidden==1024) &&
           (c->topk==1 || c->topk==8) && moe::decode_helpers::transition_fast(*c))
            return moe::decode_helpers::transition_launch<256,false>(*c,nullptr,0,nullptr);
        return quactlize_ppu_moe_transition_run_v1(c);
    }
    auto projected=[](qzm_value_v1 const& v) {return v.dtype==QZM_F32 && v.round_type==QZM_F32 &&
        v.state==QZM_PROJECTED && v.partial_count==1 && v.order==QZM_TOKEN_SLOT;};
    if(c->topk!=1 || !projected(c->gate) || (!c->merged && !projected(c->up)) ||
       c->output_dtype!=QZM_F32 || c->output_order!=QZM_TOKEN_SLOT) return QZR_INVALID;
    auto gate=static_cast<float const*>(c->gate.data);
    auto up=c->merged?gate+c->hidden:static_cast<float const*>(c->up.data);
    return ffn::dense_swiglu(gate,up,static_cast<float*>(c->output),c->tokens,c->hidden,
        c->gate.row_stride,c->merged?c->gate.row_stride:c->up.row_stride,c->output_stride,c->stream);
}
