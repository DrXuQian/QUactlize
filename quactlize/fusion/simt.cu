#include <hggc_runtime.h>
#include "store.cuh"
#include "../execution/q4_s1_validation.hpp"

namespace quactlize::execution::q4_s1 {
CUTLASS_DEVICE Row locate(quactlize::fusion::DeviceCall const& c, int row) {
    int source = quactlize::fusion::input_row(c, row);
    if (source < 0) return {-1, 0, 0};
    return locate(static_cast<qkg_call_v1 const&>(c), source);
}
}
#include "../execution/simt_q8_vector.cuh"

namespace quactlize::fusion {
template<int Q,int Input,int Compute,int Warps>
__global__ void simt_gate_up(DeviceCall c,int split) {
    if constexpr(Q==8)
        execution::simt::q8_vector::kernel_body<Input,Compute,1,4,Warps,8,false,SimtFinish>(c,split);
    else
        execution::simt::register_reuse_body<Q,Input,3,4,Warps,8,Compute,0,SimtFinish>(c,split);
}

template<int Q,int Input,int Compute>
int simt_launch(DeviceCall const& c,qkg_gate_up_config_v1 const& f) {
    auto stream=static_cast<hggcStream_t>(c.stream);
    int blocks=c.rows*f.split*(c.n/32);
    if(f.warps==4) simt_gate_up<Q,Input,Compute,4><<<blocks,128,0,stream>>>(c,f.split);
    else simt_gate_up<Q,Input,Compute,8><<<blocks,256,0,stream>>>(c,f.split);
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    if(f.split>1) reduce_gate_up<<<(int64_t(c.rows)*(c.n/2)+127)/128,128,0,stream>>>(c,f.split);
    return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
}
}

#define QGU_CAT_(A,B) A##B
#define QGU_CAT(A,B) QGU_CAT_(A,B)
extern "C" int QGU_CAT(qgu_simt_,QGU_QTYPE)(quactlize::fusion::DeviceCall const* c,qkg_gate_up_config_v1 const* f) {
    using namespace quactlize::fusion;
    if(c->compute_type==QKG_COMPUTE_F16) {
        if(c->input_type==QKG_F32) return simt_launch<QGU_QTYPE,QKG_F32,QKG_COMPUTE_F16>(*c,*f);
        return simt_launch<QGU_QTYPE,QKG_F16,QKG_COMPUTE_F16>(*c,*f);
    }
    if(c->input_type==QKG_F32) return simt_launch<QGU_QTYPE,QKG_F32,QKG_COMPUTE_BF16>(*c,*f);
    return simt_launch<QGU_QTYPE,QKG_SIMT_BF16,QKG_COMPUTE_BF16>(*c,*f);
}
