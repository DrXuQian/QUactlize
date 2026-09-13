#include "q4_s1_validation.hpp"
using namespace quactlize::execution::q4_s1;
extern "C" int host_query(qkg_call_v1 const* c,qkg_q4_s1_config_v1 const* f,
        quactlize_ppu_placed_arrangement_v2 const* a,qkg_sizes_v1* out) {
    return validate(*c,*f,a,*out);
}
extern "C" int host_buffers(qkg_call_v1 const* c,qkg_sizes_v1 const* s) { return buffers(*c,*s); }
extern "C" void host_locate(qkg_call_v1 const* c,int row,int64_t* output) {
    Row r=locate(*c,row); output[0]=r.expert; output[1]=r.a; output[2]=r.output;
}
