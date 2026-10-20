#pragma once
#include "simt.h"
#include "model_gemv_scope.hpp"
#include "measured_decode.hpp"

namespace quactlize::execution::simt {
// The measured common-reader geometries, not an N/K lookup or a new selector.
// Preserve the existing fixed-shape S1 bodies: the paired controls include a
// Q5 regression and a Q4 tie. Other recipes/storage types keep their fallback.
template<int Q,int V,int C,int W,int P>
inline constexpr bool kCommonReader = Q>=10 && Q<=14 && V==3 &&
    ((C==4 && P==8 && (W==2 || W==8)) || (C==8 && W==4 && P==4) ||
     (Q==12 && C==4 && W==4 && P==4));

inline bool common_reader_scope(qkg_simt_call_v2 const& d,qkg_simt_config_v1 const& f) {
    auto const& c=d.call;
    if(c.qtype<10 || c.qtype>14 || c.input_type!=QKG_F32 || f.variant!=3 ||
       (d.compute_type!=QKG_COMPUTE_F16 && d.compute_type!=QKG_COMPUTE_BF16) ||
       c.rows<=0 || c.rows>65535 || c.n<=0 || c.k<=0 ||
       c.mode<QKG_DENSE || c.mode>QKG_INDEXED ||
       (f.split!=1 && f.split!=2 && f.split!=4 && f.split!=8))return false;
    if(!((f.columns==4 && f.values==8 && (f.warps==2 || f.warps==8)) ||
         (f.columns==8 && f.warps==4 && f.values==4) ||
         (c.qtype==12 && f.columns==4 && f.warps==4 && f.values==4)))return false;
    if(d.compute_type==QKG_COMPUTE_BF16 && f.split==1 && f.columns==4) {
        if(c.qtype==12 && f.warps==4 && f.values==4 && model_gemv::indexed_m1(c,1024,2048,1))return false;
        if(c.qtype==13 && f.warps==2 && f.values==8 && model_gemv::indexed_m1(c,2048,512,8))return false;
    }
    // Keep fixed measured implementations ahead of the generic replacement.
    switch(measured_decode(d,f)) {
#define QDM(Id,Q,Mode,N,K,E,Top,Ch,Compute,V,C,W,P,S,Changes,Hoist,Fixed) \
        case MeasuredDecode::Id:if constexpr(Fixed)return false;break;
#include "measured_decode.inc"
#undef QDM
        default:break;
    }
    return true;
}
}
