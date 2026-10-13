#include "quactlize/runtime/abi.h"
#include "quactlize/decode/api.h"
#include <cstring>
#include <new>
#include "stub_identity.inc"
struct StubHandle {
    bool dynamic=false;
    qk_call_v1 shape{};
    int endpoint=0;
};

extern "C" qk_identity_v1 const* quactlize_kpack_identity_v1() { return &stub_identity; }
extern "C" int quactlize_kpack_device_v1(char* name,int capacity,int32_t* ordinal,int32_t* cu) {
    if (capacity<16) return QK_INVALID;
    std::strcpy(name,"PPU-ZW810"); *ordinal=0; *cu=72; return 0;
}
extern "C" int quactlize_kpack_query_v1(qk_call_v1 const* c,qk_recipe_v1 const* r,qk_resources_v1* out) {
    if (!c || !r || !out || c->rows_host || c->rows_device) return QK_INVALID;
    if (c->device!=0 || c->compute_units!=72 || r->split!=stub_expected_split) return QK_INVALID;
    if (stub_expected_n && c->n!=stub_expected_n) return QK_INVALID;
    uint64_t bytes=stub_expected_n ? uint64_t(c->m)*c->n*r->split*4 : 256;
    *out={1,sizeof(*out),bytes,4096,6,0,0}; return QK_OK;
}
extern "C" int quactlize_kpack_grouped_query_v2(qk_device_call_v2 const* d,qk_recipe_v1 const* r,qk_resources_v1* out) {
    if (!d || d->version!=2 || d->size!=sizeof(*d) || d->max_rows<=0) return QK_INVALID;
    return quactlize_kpack_query_v1(&d->call,r,out);
}
extern "C" int quactlize_kpack_prepare_v1(qk_call_v1 const* c,qk_recipe_v1 const* r,void** out) {
    qk_resources_v1 resources{};
    int rc=quactlize_kpack_query_v1(c,r,&resources);
    if (rc || !c->a || !c->low || !c->metadata || !c->output || !c->workspace) return QK_INVALID;
    *out=new StubHandle{}; return QK_OK;
}
extern "C" int quactlize_kpack_grouped_prepare_v2(qk_device_call_v2 const* d,qk_recipe_v1 const* r,void** out) {
    if (!d->call.offsets_device) return QK_INVALID;
    return quactlize_kpack_prepare_v1(&d->call,r,out);
}
extern "C" int quactlize_kpack_run_v1(void* h,void*) { return h && !static_cast<StubHandle*>(h)->dynamic ? QK_OK : QK_INVALID; }
extern "C" void quactlize_kpack_destroy_v1(void* h) { delete static_cast<StubHandle*>(h); }

#ifdef QK_TEST_DYNAMIC
#include "quactlize/runtime/dynamic.hpp"
extern "C" int quactlize_kpack_prepare_dynamic_v1(qk_call_v1 const* c,qk_recipe_v1 const* r,void** out) {
    if (!c || !out || !quactlize::runtime::address_free(*c)) return QK_INVALID;
    qk_resources_v1 resources{};int rc=quactlize_kpack_query_v1(c,r,&resources);if (rc) return rc;
    *out=new StubHandle{true,*c,0};return QK_OK;
}
extern "C" int quactlize_kpack_run_dynamic_v1(void* p,qk_call_v1 const* c) {
    if (!p || !c) return QK_INVALID;
    auto const& h=*static_cast<StubHandle*>(p);
    if (!h.dynamic || !quactlize::runtime::same_dense_shape(*c,h.shape) ||
        !quactlize::runtime::dense_buffers(*c,256,false,stub_identity.route==1 && stub_identity.qtype!=8)) return QK_INVALID;
    // Writes use the CURRENT launch descriptor, not the addresses from the
    // previous run. Host ABI test only, not a numerical device oracle.
    *static_cast<float*>(c->output)=*static_cast<float const*>(c->a)+float(uintptr_t(c->stream));
    *static_cast<int*>(c->workspace)=stub_expected_split;
    return QK_OK;
}
extern "C" int quactlize_kpack_decode_dense_prepare_dynamic_v1(qkd_dense_call_v2 const* c,qk_recipe_v1 const* r,void** out) {
    if (!c || c->version!=2 || c->compute_type!=0 || c->dense.input_type!=c->dense.output_type) return QK_INVALID;
    int rc=quactlize_kpack_prepare_dynamic_v1(&c->dense.call,r,out);
    if (!rc) static_cast<StubHandle*>(*out)->endpoint=c->dense.input_type;
    return rc;
}
extern "C" int quactlize_kpack_decode_dense_run_dynamic_v1(void* p,qkd_dense_call_v2 const* c) {
    if (!p || !c || c->version!=2 || c->compute_type!=0 ||
        c->dense.input_type!=static_cast<StubHandle*>(p)->endpoint || c->dense.output_type!=c->dense.input_type) return QK_INVALID;
    return quactlize_kpack_run_dynamic_v1(p,&c->dense.call);
}
#endif

#ifdef QK_TEST_TYPED
extern "C" qk_identity_v1 const* quactlize_kpack_decode_dense_identity_v1() {
    static qk_identity_v1 const typed=[] {
        auto value=stub_identity;
        value.build_key="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
        return value;
    }();
    return &typed;
}
extern "C" int quactlize_kpack_decode_dense_device_v1(char* n,int cap,int32_t* d,int32_t* cu) {
    return quactlize_kpack_device_v1(n,cap,d,cu);
}
extern "C" int quactlize_kpack_decode_dense_query_v1(qkd_dense_call_v1 const* d,qk_recipe_v1 const* r,qk_resources_v1* out) {
    if (!d || d->version!=1 || d->size!=sizeof(*d) || d->input_type!=d->output_type ||
        (d->input_type!=QKD_F32 && d->input_type!=QKD_BF16) || d->call.m>8) return QK_INVALID;
    int rc=quactlize_kpack_query_v1(&d->call,r,out);
    if (!rc) out->shared_bytes+=d->input_type*128;
    return rc;
}
extern "C" int quactlize_kpack_decode_dense_prepare_v1(qkd_dense_call_v1 const* d,qk_recipe_v1 const* r,void** out) {
    qk_resources_v1 resources{};
    int rc=quactlize_kpack_decode_dense_query_v1(d,r,&resources);
    return rc ? rc : quactlize_kpack_prepare_v1(&d->call,r,out);
}
extern "C" int quactlize_kpack_decode_dense_run_v1(void* h,void* stream) {return quactlize_kpack_run_v1(h,stream);}
extern "C" void quactlize_kpack_decode_dense_destroy_v1(void* h) {quactlize_kpack_destroy_v1(h);}
#endif
