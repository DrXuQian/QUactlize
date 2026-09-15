#include "quactlize/dispatch/decode.hpp"
#include "quactlize/execution/q4_decode.h"
#include <cstdio>

extern "C" int qkg_q4_decode_launch(qkg_call_v1 const&, qkg_q4_decode_config_v1 const&) {
    return 123; // A real GPU is deliberately not involved in host ABI tests.
}
extern "C" int qkg_q4_decode_launch_bf16(qkg_call_v1 const&, qkg_q4_decode_config_v1 const&) { return 124; }
extern "C" int q4_decode_tc(qks_request_v1 const* r,char* out,int bytes) {
    auto s=quactlize::dispatch::select_decode_tc(*r);
    if(!s.config) return QKS_MISS;
    auto f=s.config;
    auto recipe=quactlize::dispatch::recipe(*f,*r,12);
    std::snprintf(out,bytes,"tc:%s:s%d:b%d:g%d",f->symbol,recipe.split,f->grid_b,f->grid_mode);
    return s.policy;
}
