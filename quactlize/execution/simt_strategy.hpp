#pragma once
#include "simt.h"
#include "model_gemv_scope.hpp"

namespace quactlize::execution::simt {
// Host launcher strategy, shared with planning/explain. These are not new
// algorithms: preserve the admitted body, shape guards and reduction order.
enum class Q8Strategy { Generic, Hoisted, S1Hoisted, S1Narrow, FixedCold, FixedHoisted };

inline Q8Strategy q8_strategy(qkg_simt_call_v2 const& d,int variant,int columns,int warps,int values,int split) {
    auto const& c=d.call;
    if(d.compute_type!=QKG_COMPUTE_F16 || !model_gemv::dense_m1(c) || variant!=5 || values!=4)
        return Q8Strategy::Generic;
    if(columns==8 && warps==4) {
        if(split==8 && c.n==2048 && c.k==4096)return Q8Strategy::FixedCold;
        if(split==1 && c.n==8192 && c.k==2048)return Q8Strategy::FixedHoisted;
        if(split==1)return Q8Strategy::Hoisted;
    } else if(columns==4 && split==1) {
        if(warps==2)return Q8Strategy::S1Hoisted;
        if(warps==8)return c.n==4096 && c.k==2048 ? Q8Strategy::S1Narrow : Q8Strategy::Hoisted;
    }
    return Q8Strategy::Generic;
}

template<int Q,int Variant,int Columns,int Warps,int Values>
inline constexpr int kBf16F32Changes = Q==13 && Variant==3 && Columns==4 && Warps==2 && Values==8 ? 3 : 0;

inline bool measured_reuse(qkg_simt_call_v2 const& d,int variant,int columns,int warps,int values,int split) {
    auto const& c=d.call;
    if(d.compute_type!=QKG_COMPUTE_BF16 || variant!=3 || columns!=4 || split!=1)return false;
    return (c.qtype==12 && warps==4 && values==4 && model_gemv::indexed_m1(c,1024,2048,1)) ||
           (c.qtype==13 && warps==2 && values==8 && model_gemv::indexed_m1(c,2048,512,8));
}

struct Implementation {
    char const* producer;
    char const* reduction;
    bool hoist=false, fixed=false;
    int changes=0;
};

inline Implementation implementation(qkg_simt_call_v2 const& d,qkg_simt_config_v1 const& f) {
    auto const& c=d.call;
    Implementation out{"register-reuse",f.split==1?"none":"ordered-row-float2-or-scalar"};
    if(c.qtype==8 && f.variant>=4) {
        switch(q8_strategy(d,f.variant,f.columns,f.warps,f.values,f.split)) {
            case Q8Strategy::Generic:out.producer="q8-vector";break;
            case Q8Strategy::Hoisted:out.producer="q8-vector-hoisted";out.hoist=true;break;
            case Q8Strategy::S1Hoisted:out.producer="q8-vector-s1-hoisted";out.hoist=true;break;
            case Q8Strategy::S1Narrow:out.producer="q8-vector-s1-narrow";break;
            case Q8Strategy::FixedCold:
                out.producer="q8-vector-fixed";out.fixed=true;out.reduction="ordered-flat-float2-or-row-fallback";break;
            case Q8Strategy::FixedHoisted:
                out.producer="q8-vector-fixed-hoisted";out.fixed=out.hoist=true;break;
        }
    } else if(d.compute_type==QKG_COMPUTE_BF16 && c.input_type==QKG_F32) {
        if(c.qtype==13 && f.variant==3 && f.columns==4 && f.warps==2 && f.values==8)out.changes=3;
        if(measured_reuse(d,f.variant,f.columns,f.warps,f.values,f.split)) {
            out.fixed=c.qtype==13;
            out.changes=c.qtype==13?3:1;
            if(out.fixed)out.producer="register-reuse-fixed";
        }
    }
    return out;
}
} // namespace quactlize::execution::simt
