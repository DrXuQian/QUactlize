#include <hggc_runtime.h>
#include "validation.hpp"
#include "store.cuh"

#define DECL(Q) \
extern "C" int qgu_simt_##Q(quactlize::fusion::DeviceCall const*,qkg_gate_up_config_v1 const*); \
extern "C" int qgu_tc_##Q(quactlize::fusion::DeviceCall const*,qkg_gate_up_config_v1 const*,qkg_sizes_v1 const*);
DECL(8) DECL(10) DECL(11) DECL(12) DECL(13) DECL(14)
#undef DECL

extern "C" int quactlize_gate_up_query_v1(qkg_gate_up_call_v1 const* d,qkg_gate_up_config_v1 const* f,
    qkg_gate_up_layout_v1 const* layout,qkg_sizes_v1* out) {
    if(!d || !f || !layout || !out) return QKG_INVALID;
    qkg_sizes_v1 result{};
    int rc=quactlize::fusion::query(*d,*f,*layout,result);
    if(!rc) *out=result;
    return rc;
}

static int run(qkg_gate_up_call_v1 const* d,qkg_gate_up_config_v1 const* f,
    qkg_gate_up_layout_v1 const* layout,int32_t const* input_rows,int32_t const* status) {
    qkg_sizes_v1 sizes{};
    int rc=quactlize_gate_up_query_v1(d,f,layout,&sizes);
    if(rc) return rc;
    rc=quactlize::fusion::buffers(*d,sizes);
    if(rc) return rc;
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    quactlize::fusion::DeviceCall c{};
    static_cast<qkg_call_v1&>(c)=d->input.call;
    c.n*=2; c.compute_type=d->input.compute_type;
    c.output_type=d->output_type; c.round_projection=d->round_projection;
    c.input_rows=input_rows; c.status=status;
#define CASE(Q) case Q: return f->backend==QKG_GATE_UP_SIMT ? qgu_simt_##Q(&c,f) : qgu_tc_##Q(&c,f,&sizes)
    switch(c.qtype) { CASE(8); CASE(10); CASE(11); CASE(12); CASE(13); CASE(14); }
#undef CASE
    return QKG_FORMAT;
}

extern "C" int quactlize_gate_up_run_v1(qkg_gate_up_call_v1 const* d,qkg_gate_up_config_v1 const* f,
    qkg_gate_up_layout_v1 const* layout) {
    return run(d,f,layout,nullptr,nullptr);
}

extern "C" int quactlize_gate_up_run_v2(qkg_gate_up_call_v2 const* d,qkg_gate_up_config_v1 const* f,
    qkg_gate_up_layout_v1 const* layout) {
    if(!d) return QKG_INVALID;
    qkg_sizes_v1 sizes{};
    int rc=quactlize_gate_up_query_v1(&d->call,f,layout,&sizes);
    if(rc) return rc;
    rc=quactlize::fusion::row_buffers(*d,sizes);
    if(rc) return rc;
    return run(&d->call,f,layout,d->input_rows,d->status);
}

extern "C" int quactlize_gate_up_select_v1(int q,int n,int k,int experts,
    int tokens,int compute,qkg_gate_up_config_v1* out) {
    return quactlize::fusion::select(q,n,k,experts,tokens,compute,out);
}
