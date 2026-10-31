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
    if(measured_m(c.tokens)<0 || (c.columns!=2048 && c.columns!=3072) ||
       c.topk!=8 || c.weights_stride!=8 || c.output_stride!=c.columns ||
       c.down.row_stride!=c.columns || c.down.round_type!=QZM_BF16) return QZM_FINISH_SERIAL;
    bool measured=false;
    if(c.down.state==QZM_PROJECTED) {
        measured=c.down.order==QZM_GROUPED?(c.down.dtype==QZM_BF16 || c.down.dtype==QZM_F32):
                 c.down.order==QZM_TOKEN_SLOT && c.down.dtype==QZM_F32;
    } else if(c.down.state==QZM_PARTIALS && c.down.order==QZM_GROUPED &&
              c.down.partial_stride==int64_t(c.tokens)*c.topk*c.columns) {
        measured=c.down.partial_count==2 || c.down.partial_count==4 || c.down.partial_count==8;
    }
    if(!measured) return QZM_FINISH_SERIAL;
    // One small-output rule replaces 4/8-warp and dtype-specific exact ties.
    // Across both measured cohorts, SLOTS4 gives up at most 0.59us to the old
    // choice and still beats decomposition. SLOTS8 remains an explicit option.
    // Second cohort archive: 5a8896de1217f2d23c9078f9f0c5bf1426bda82ea0f30a281d2aae35e402bc3f.
    if(int64_t(c.tokens)*c.columns<=32768) return QZM_FINISH_SLOTS4;
    if(c.down.state==QZM_PARTIALS) {
        // Retain all 14 material large-partial exceptions: serial finish can
        // regress by hundreds of us for some route profiles. No router readback.
        // N2048/S4/M5120 favors serial unlike M4096; do not interpolate it.
        bool separate=c.columns==2048?
            (c.down.partial_count==8 && c.tokens>=1024) || (c.down.partial_count==4 && c.tokens==4096):
            (c.down.partial_count==8 && c.tokens>=512) || (c.down.partial_count==4 && c.tokens>=1024);
        if(separate) return QZM_FINISH_SEPARATE;
    }
    return QZM_FINISH_SERIAL;
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
inline int prepare_plan_v2(qzm_prepare_v1 const& c,qzm_prepare_plan_v2& out) {
    qzm_prepare_v1 selected=c;
    if(c.config.version!=1 || c.config.size!=sizeof(c.config)) return QZR_INVALID;
    int implementation=QZM_AUTO;
    if(c.config.algorithm==QZR_AUTO) {
        if(c.config.tile_rows) return QZR_INVALID;
        selected.config={1,sizeof(selected.config),QZR_TILED,512};
        bool measured=c.route.experts==256 && c.route.topk==8 && c.route.channels==1 &&
            c.route.source_stride==1 && c.route.ids_stride==8 && c.dtype==QZM_BF16 &&
            c.input_row_stride==c.columns && (c.columns==2048 || c.columns==3072);
        // ZW810/1700MHz; ffaecba3 real-chain capture archive SHA256
        // 5d005fdf70f4a3aa36a4beb831467197b0f8d11b2f64e660421f2d73bf82b12c.
        // Keep the material small-M gain (about 2..4us), not sub-microsecond
        // WARP/tile-size exact ties. No router readback or unmeasured crossover.
        int m=c.route.tokens;
        if(measured && ((m>=1 && m<=9) || m==16 || m==32)) {
            selected.config.algorithm=QZR_SCALAR;
            implementation=QZM_PREPARE_SEPARATE;
        }
    }
    qzm_plan_v1 helper{};int rc=prepare_plan(selected,implementation,helper);if(rc) return rc;
    out={2,sizeof(out),selected.config,helper};return QZR_OK;
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
