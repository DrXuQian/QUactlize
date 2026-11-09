// Prototype router: qgemv_mma_run(call, compute, tile, warps_code, split)
// warps_code = wn*1000 + wk*10 + nb (e.g. 1042 -> WN=1, WK=4, NB=2; 1168 -> WN=1, WK=16, NB=8).
#include "api.h"

#define QGEMV_TU_DECL3(base) \
    extern "C" int base##_t2(qkg_call_v1 const*,int,int,int,int); \
    extern "C" int base##_t4(qkg_call_v1 const*,int,int,int,int); \
    extern "C" int base##_t8(qkg_call_v1 const*,int,int,int,int);
QGEMV_TU_DECL3(qgemv_mma_tu_q12_d_f16)
QGEMV_TU_DECL3(qgemv_mma_tu_q12_i_bf16)
QGEMV_TU_DECL3(qgemv_mma_tu_q13_d_f16)
QGEMV_TU_DECL3(qgemv_mma_tu_q13_i_bf16)
QGEMV_TU_DECL3(qgemv_mma_tu_q14_d_f16)
QGEMV_TU_DECL3(qgemv_mma_tu_q8_d_f16)

extern "C" int qgemv_mma_run(qkg_call_v1 const* c,int compute,int t,int code,int split) {
    if(!c) return QKG_INVALID;
    int const wn=code/1000,wk=(code/10)%100,nb=code%10;
    bool const dense=c->mode==QKG_DENSE,indexed=c->mode==QKG_INDEXED;
#define QGEMV_ROUTE(q,ok,cmp,base) \
    if(c->qtype==q && (ok) && compute==cmp) \
        return t==2 ? base##_t2(c,wn,wk,nb,split) : t==4 ? base##_t4(c,wn,wk,nb,split) : \
               t==8 ? base##_t8(c,wn,wk,nb,split) : QKG_INVALID;
    QGEMV_ROUTE(12,dense,0,qgemv_mma_tu_q12_d_f16)
    QGEMV_ROUTE(12,indexed,1,qgemv_mma_tu_q12_i_bf16)
    QGEMV_ROUTE(13,dense,0,qgemv_mma_tu_q13_d_f16)
    QGEMV_ROUTE(13,indexed,1,qgemv_mma_tu_q13_i_bf16)
    QGEMV_ROUTE(14,dense,0,qgemv_mma_tu_q14_d_f16)
    QGEMV_ROUTE(8,dense,0,qgemv_mma_tu_q8_d_f16)
#undef QGEMV_ROUTE
    return QKG_FORMAT;
}
