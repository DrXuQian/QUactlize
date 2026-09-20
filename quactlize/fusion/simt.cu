#include <hggc_runtime.h>
#include "store.cuh"
#include "paired16.cuh"
#include "selection.hpp"
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
// Same dynamic-dimension, constant-S1 tile16 body as the confirmed paired
// winner. Do not replace its sequential inter-warp gate/up sum by XOR fold.
__global__ void simt_gate_up_q8_tile16(DeviceCall c);
#if QGU_QTYPE == 8
__global__ void simt_gate_up_q8_tile16(DeviceCall c) {
    execution::simt::q8_vector::kernel_body<1,0,1,4,8,4,true,SimtFinish>(c,1);
}
#endif

template<int Q,int Input,int Compute,int Warps>
__global__ void simt_gate_up(DeviceCall c,int split) {
    if constexpr(Q==8)
        execution::simt::q8_vector::kernel_body<Input,Compute,1,4,Warps,8,false,SimtFinish>(c,split);
    else
        execution::simt::register_reuse_body<Q,Input,3,4,Warps,8,Compute,0,SimtFinish>(c,split);
}

template<int Q,int Input,int Compute,int Warps,int N,int K>
__global__ void simt_gate_up_model(DeviceCall c) {
    static_assert(Input==QKG_F32 && Warps==8);
    static_assert(N>0 && N%32==0 && K>0 && K%256==0);
    c.n=N;c.k=K;
    if constexpr(Q==12) {
        static_assert(Compute==QKG_COMPUTE_BF16);
        c.experts=256;c.mode=QKG_INDEXED;c.channels=1;c.topk=8;
        execution::simt::register_reuse_body<12,1,3,4,8,8,1,1,SimtFinish>(c,1);
    } else {
        static_assert(Q==8 && Compute==QKG_COMPUTE_F16);
        execution::simt::q8_vector::kernel_body<1,0,1,4,8,4,true,SimtFinish>(c,1);
    }
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

template<int Warps,int P,int Changes,int N,int K>
__global__ void simt_gate_up_paired16(DeviceCall c) {
    c.n=N;c.k=K;c.experts=256;c.mode=QKG_INDEXED;c.channels=1;c.topk=8;
    execution::simt::register_reuse_body<12,1,3,4,Warps,P,1,Changes,Paired16Finish>(c,1);
}

template<int Q,int Reader,int Warps,int P,int Changes,int N,int K>
int special_launch(DeviceCall const& c) {
    auto stream=static_cast<hggcStream_t>(c.stream);
    if constexpr(Q==8 && Reader==Q8Tile16Reader)
        simt_gate_up_q8_tile16<<<c.rows*(c.n/16),256,0,stream>>>(c);
    else if constexpr(Q==12 && Reader==Q4Paired32Reader)
        simt_gate_up_model<12,1,1,Warps,N,K><<<c.rows*(c.n/32),Warps*32,0,stream>>>(c);
    else if constexpr(Q==12 && Reader==Q4Paired16Reader)
        simt_gate_up_paired16<Warps,P,Changes,N,K><<<c.rows*(c.n/16),Warps*32,0,stream>>>(c);
    else return QKG_SHAPE;
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

extern "C" int QGU_CAT(qgu_recipe_,QGU_QTYPE)(quactlize::fusion::DeviceCall const* c,qkg_gate_up_recipe_v2 const* r) {
    using namespace quactlize::fusion;
    if(r->reader==GenericReader) return QGU_CAT(qgu_simt_,QGU_QTYPE)(c,&r->config);
    switch(r->recipe_id) {
#define QGU_RECIPE(Name,Id,B,S,TM,W,R,V,C,P,Changes,Hoist,N,K) \
    case Id: if constexpr(B==0 && R!=GenericReader) \
        return special_launch<QGU_QTYPE,R,W,P,Changes,N,K>(*c); else return QKG_SHAPE;
#include "recipes.inc"
#undef QGU_RECIPE
    default:return QKG_INVALID;
    }
}
