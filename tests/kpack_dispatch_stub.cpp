#include "quactlize/runtime/abi.h"
#include <cstring>
#include <new>
#include "stub_identity.inc"

extern "C" qk_identity_v1 const* quactlize_kpack_identity_v1() { return &stub_identity; }
extern "C" int quactlize_kpack_device_v1(char* name,int capacity,int32_t* ordinal,int32_t* cu) {
    if (capacity<16) return QK_INVALID;
    std::strcpy(name,"PPU-ZW810"); *ordinal=0; *cu=72; return 0;
}
extern "C" int quactlize_kpack_query_v1(qk_call_v1 const* c,qk_recipe_v1 const* r,qk_resources_v1* out) {
    if (!c || !r || !out || c->rows_host || c->rows_device) return QK_INVALID;
    if (c->device!=0 || c->compute_units!=72 || r->split!=stub_expected_split) return QK_INVALID;
    *out={1,sizeof(*out),256,4096,6,0,0}; return QK_OK;
}
extern "C" int quactlize_kpack_grouped_query_v2(qk_device_call_v2 const* d,qk_recipe_v1 const* r,qk_resources_v1* out) {
    if (!d || d->version!=2 || d->size!=sizeof(*d) || d->max_rows<=0) return QK_INVALID;
    return quactlize_kpack_query_v1(&d->call,r,out);
}
extern "C" int quactlize_kpack_prepare_v1(qk_call_v1 const* c,qk_recipe_v1 const* r,void** out) {
    qk_resources_v1 resources{};
    int rc=quactlize_kpack_query_v1(c,r,&resources);
    if (rc || !c->a || !c->low || !c->metadata || !c->output || !c->workspace) return QK_INVALID;
    *out=new int(42); return QK_OK;
}
extern "C" int quactlize_kpack_grouped_prepare_v2(qk_device_call_v2 const* d,qk_recipe_v1 const* r,void** out) {
    if (!d->call.offsets_device) return QK_INVALID;
    return quactlize_kpack_prepare_v1(&d->call,r,out);
}
extern "C" int quactlize_kpack_run_v1(void* h,void*) { return h && *static_cast<int*>(h)==42 ? QK_OK : QK_INVALID; }
extern "C" void quactlize_kpack_destroy_v1(void* h) { delete static_cast<int*>(h); }
