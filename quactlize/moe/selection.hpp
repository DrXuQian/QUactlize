#pragma once
#include "validation.hpp"

namespace quactlize::moe {
// ZW810/SDK2.1.1, measured source 7169721d; archive SHA256
// 7cf59b73d62cd0f23d29d5a1ac186ed31e99abea1df50777a31e8c6e2ac257c7.
// Actual before/after clocks 1700MHz, Asys component means. This exact scope
// is not a universal M crossover or native llama performance certificate.
inline int measured_m(int m) {
    constexpr int values[]={1,2,3,4,5,6,7,8,16,32,64,128,256,512,1024,2048,4096,5120};
    for(int i=0;i<18;++i) if(values[i]==m) return i;
    return -1;
}
inline int prepare_choice(qzm_prepare_v1 const& c,routing::Layout const& l) {
    bool measured=l.algorithm==QZR_TILED && l.tile==512 && c.route.experts==256 &&
        c.route.topk==8 && c.route.channels==1 && c.route.source_stride==1 &&
        c.route.ids_stride==8 && c.dtype==QZM_BF16 && c.input_row_stride==c.columns &&
        (c.columns==2048 || c.columns==3072) && measured_m(c.route.tokens)>=0;
    return measured && c.route.tokens>=32?QZM_PREPARE_HYBRID:QZM_PREPARE_FUSED;
}
inline int finish_choice(qzm_finish_v1 const& c) {
    int m=measured_m(c.tokens),column=-1,n=c.columns==2048?0:c.columns==3072?1:-1;
    if(m<0 || n<0 || c.topk!=8 || c.weights_stride!=8 || c.output_stride!=c.columns ||
       c.down.row_stride!=c.columns || c.down.round_type!=QZM_BF16) return QZM_FINISH_SERIAL;
    if(c.down.state==QZM_PROJECTED) {
        if(c.down.order==QZM_GROUPED) column=c.down.dtype==QZM_BF16?0:c.down.dtype==QZM_F32?1:-1;
        else if(c.down.dtype==QZM_F32) column=5;
    } else if(c.down.state==QZM_PARTIALS && c.down.order==QZM_GROUPED &&
              c.down.partial_stride==int64_t(c.tokens)*c.topk*c.columns) {
        column=c.down.partial_count==2?2:c.down.partial_count==4?3:c.down.partial_count==8?4:-1;
    }
    if(column<0) return QZM_FINISH_SERIAL;
    constexpr int S=QZM_FINISH_SERIAL,P4=QZM_FINISH_SLOTS4,P8=QZM_FINISH_SLOTS8,D=QZM_FINISH_SEPARATE;
    // Columns: grouped final BF16/F32, grouped partials S2/S4/S8, token-slot F32.
    // For each host-known key, minimize mean across the measured route profiles
    // among arms no slower than decomposition in EVERY profile. No route-label
    // lookup/readback. 14 keys choose D (70 cases), not just the 30 per-profile
    // decomposition winners. A per-profile oracle is not an executable policy.
    constexpr int choices[2][18][6]={
        { // N2048, rows in measured_m order.
            {P8,P8,P8,P8,P8,P8}, {P8,P8,P8,P8,P8,P8},
            {P8,P8,P8,P8,P8,P8}, {P8,P8,P8,P8,P8,P8},
            {P8,P4,P8,P4,P8,P8}, {P4,P4,P4,P4,P4,P4},
            {P4,P4,P4,P4,P4,P4}, {P4,P4,P4,P4,P4,P4},
            {P4,P4,P4,P4,P4,P4}, {S,S,S,S,S,S}, {S,S,S,S,S,S},
            {S,S,S,S,S,S}, {S,S,S,S,S,S}, {S,S,S,S,S,S},
            {S,S,S,S,D,S}, {S,S,S,S,D,S}, {S,S,S,D,D,S}, {S,S,S,S,D,S}
        },
        { // N3072, rows in measured_m order.
            {P8,P8,P8,P8,P8,P8}, {P8,P8,P8,P8,P8,P8}, {P8,P8,P8,P8,P8,P8},
            {P4,P4,P4,P4,P4,P4}, {P4,P4,P4,P4,P4,P4},
            {P4,P4,P4,P4,P4,P4}, {P4,P4,P4,P4,P4,P4}, {P4,P4,P4,P4,P4,P4},
            {S,S,S,S,S,S}, {S,S,S,S,S,S}, {S,S,S,S,S,S},
            {S,S,S,S,S,S}, {S,S,S,S,S,S}, {S,S,S,S,D,S},
            {S,S,S,D,D,S}, {S,S,S,D,D,S}, {S,S,S,D,D,S}, {S,S,S,D,D,S}
        }
    };
    return choices[n][m][column];
}
inline int prepare_plan(qzm_prepare_v1 const& c,int implementation,qzm_plan_v1& out) {
    routing::Layout l;int rc=prepare(c,l,false);if(rc) return rc;
    if(implementation==QZM_AUTO) implementation=prepare_choice(c,l);
    if(implementation<QZM_PREPARE_FUSED || implementation>QZM_PREPARE_SEPARATE ||
       (implementation==QZM_PREPARE_HYBRID && l.algorithm!=QZR_TILED)) return QZR_INVALID;
    int stages=!l.rows?1:implementation==QZM_PREPARE_HYBRID?4:
        implementation==QZM_PREPARE_SEPARATE?l.stages+2:3;
    out={1,sizeof(out),implementation,stages,l.bytes};return QZR_OK;
}
inline int finish_plan(qzm_finish_v1 const& c,int implementation,qzm_plan_v1& out) {
    int rc=finish(c,false);if(rc) return rc;
    if(implementation==QZM_AUTO) implementation=finish_choice(c);
    if(implementation<QZM_FINISH_SERIAL || implementation>QZM_FINISH_SEPARATE) return QZR_INVALID;
    uint64_t bytes=0;
    if(c.tokens && implementation==QZM_FINISH_SEPARATE) {
        rc=matrix_bytes(c.tokens*c.topk,c.columns,c.columns,QZM_F32,bytes);if(rc) return rc;
        if(bytes>UINT64_MAX/2-15) return QZR_OVERFLOW;
        bytes=2*routing::aligned(bytes);
    }
    out={1,sizeof(out),implementation,c.tokens?(implementation==QZM_FINISH_SEPARATE?3:1):0,bytes};
    return QZR_OK;
}
inline bool same_plan(qzm_plan_v1 const& a,qzm_plan_v1 const& b) {
    return a.version==1 && a.size==sizeof(a) && a.version==b.version && a.size==b.size && a.implementation==b.implementation &&
        a.stages==b.stages && a.workspace_bytes==b.workspace_bytes;
}
inline int finish_workspace(qzm_finish_v1 const& c,uint64_t required,void* workspace,uint64_t bytes) {
    if(bytes<required) return QZR_CAPACITY;
    if(!required) return QZR_OK;
    uint64_t down=0,weights=0,output=0;int nr=c.tokens*c.topk;
    int rc=value(c.down,nr,c.columns,down);if(rc) return rc;
    rc=matrix_bytes(c.tokens,c.topk,c.weights_stride,QZM_F32,weights);if(rc) return rc;
    rc=matrix_bytes(c.tokens,c.columns,c.output_stride,QZM_F32,output);if(rc) return rc;
    Spans spans;
    if((rc=spans.add(c.down.data,down,width(c.down.dtype),false))) return rc;
    if((rc=spans.add(c.weights,weights,4,false))) return rc;
    if(c.down.order==QZM_GROUPED && (rc=spans.add(c.inverse_rows,uint64_t(nr)*4,4,false))) return rc;
    if((rc=spans.add(c.output,output,4,true))) return rc;
    return spans.add(workspace,required,16,true);
}
} // namespace quactlize::moe
