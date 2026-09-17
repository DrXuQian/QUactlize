#include <hggc_runtime.h>
#include "tc.cuh"

#define QGU_CAT_(A,B) A##B
#define QGU_CAT(A,B) QGU_CAT_(A,B)
namespace {
template<int TM>
int launch(quactlize::fusion::DeviceCall const& c,qkg_gate_up_config_v1 const& f,qkg_sizes_v1 const& sizes) {
    using namespace quactlize::fusion;
    if(c.compute_type==QKG_COMPUTE_F16) {
        if(c.input_type==QKG_F32) return tc_launch<QGU_QTYPE,TM,cutlass::half_t,float>(c,f,sizes);
        return tc_launch<QGU_QTYPE,TM,cutlass::half_t,cutlass::half_t>(c,f,sizes);
    }
    if(c.input_type==QKG_F32) return tc_launch<QGU_QTYPE,TM,cutlass::bfloat16_t,float>(c,f,sizes);
    return tc_launch<QGU_QTYPE,TM,cutlass::bfloat16_t,cutlass::bfloat16_t>(c,f,sizes);
}
}
extern "C" int QGU_CAT(qgu_tc_,QGU_QTYPE)(quactlize::fusion::DeviceCall const* c,
    qkg_gate_up_config_v1 const* f,qkg_sizes_v1 const* sizes) {
    return f->tile_m==8 ? launch<8>(*c,*f,*sizes) : launch<16>(*c,*f,*sizes);
}
