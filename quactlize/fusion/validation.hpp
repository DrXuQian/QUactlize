#pragma once
#include "gate_up.h"
#include "../execution/validation.hpp"

namespace quactlize::fusion {
inline int select(int q,int n,int k,int experts,int tokens,int compute,qkg_gate_up_config_v1* out) {
    if(!out) return QKG_INVALID;
    *out={};
    if(n!=512 || k!=2048 || tokens<1 || tokens>8) return QKG_SHAPE;
    if(q==8 && experts==1 && compute==QKG_COMPUTE_F16) {
        *out={1,sizeof(*out),QKG_GATE_UP_SIMT,1,0,8}; return QKG_OK;
    }
    if(q!=12 || experts!=256 || compute!=QKG_COMPUTE_BF16) return QKG_SHAPE;
    // M4's two finalists differ by <1%; retain the single-kernel SIMT arm.
    if(tokens<=2 || tokens==4) *out={1,sizeof(*out),QKG_GATE_UP_SIMT,1,0,tokens==2?4:8};
    else *out={1,sizeof(*out),QKG_GATE_UP_TC,tokens==3?2:1,tokens<=6?16:8,0};
    return QKG_OK;
}

inline int tile_k(int q) {
    return q==11 || q==13 ? 256 : q==10 || q==14 ? 128 : 64;
}

inline int query(qkg_gate_up_call_v1 const& d,qkg_gate_up_config_v1 const& f,
                 qkg_gate_up_layout_v1 const& layout,qkg_sizes_v1& sizes) {
    auto const& input=d.input;
    auto const& c=input.call;
    if (d.version!=1 || d.size!=sizeof(d) || input.version!=2 || input.size!=sizeof(input) ||
        (input.compute_type!=QKG_COMPUTE_F16 && input.compute_type!=QKG_COMPUTE_BF16) ||
        c.input_type<0 || c.input_type>QKG_SIMT_BF16 || d.output_type<0 || d.output_type>QKG_SIMT_BF16 ||
        (c.input_type==QKG_SIMT_BF16 && input.compute_type!=QKG_COMPUTE_BF16) ||
        (c.input_type==QKG_F16 && input.compute_type!=QKG_COMPUTE_F16) ||
        d.round_projection<0 || d.round_projection>1 ||
        f.version!=1 || f.size!=sizeof(f) || (f.backend!=QKG_GATE_UP_SIMT && f.backend!=QKG_GATE_UP_TC) ||
        (f.split!=1 && f.split!=2 && f.split!=4 && f.split!=8)) return QKG_INVALID;
    if (layout.version!=1 || layout.size!=sizeof(layout) || layout.layout_id!=QKG_GATE_UP_N4_V1)
        return QKG_ARRANGEMENT;
    if (c.n<=0 || c.n>INT32_MAX/2 || c.out_row_stride<c.n) return QKG_SHAPE;
    qkg_sizes_v1 single{};
    int rc=execution::sizes(c.qtype,c.n,c.k,c.experts,&layout.packing,single);
    if (rc) return rc;
    auto physical=c;
    physical.n*=2;
    physical.out_row_stride=physical.n;
    if (physical.input_type==QKG_SIMT_BF16) physical.input_type=QKG_F16;
    qkg_config_v1 geometry{1,sizeof(geometry),16,4,f.split};
    rc=execution::query(physical,geometry,&layout.packing,sizes,true);
    if (rc) return rc;
    if (uint64_t(c.out_row_stride)>uint64_t(INT64_MAX)/uint64_t(c.rows)/4) return QKG_OVERFLOW;
    int width=c.input_type==QKG_F32 ? 4 : 8;
    if (c.a_row_stride%width || (c.mode==QKG_INDEXED && c.a_token_stride%width)) return QKG_INVALID;
    if (f.backend==QKG_GATE_UP_SIMT) {
        if (f.tile_m!=0 || (f.warps!=4 && f.warps!=8) ||
            (c.mode==QKG_GROUPED ? c.rows>64 : c.rows/(c.mode==QKG_INDEXED ? c.topk : 1)>8)) return QKG_SHAPE;
    } else {
        if ((f.tile_m!=8 && f.tile_m!=16) || f.warps!=0 || c.k%(tile_k(c.qtype)*f.split)) return QKG_SHAPE;
        uint64_t groups=c.mode==QKG_INDEXED ? c.rows : c.experts;
        if (groups*f.split>65535 || c.n>65535*32 ||
            uint64_t(c.rows+int64_t(f.tile_m)-1)/f.tile_m>INT32_MAX)
            return QKG_OVERFLOW;
    }
    return QKG_OK;
}

inline int buffers(qkg_gate_up_call_v1 const& d,qkg_sizes_v1 const& s) {
    auto const& c=d.input.call;
    if (!c.a || !c.low || !c.units || !c.output ||
        (s.high_bytes!=0)!=(c.high!=nullptr) ||
        (c.mode==QKG_GROUPED)!=(c.offsets!=nullptr) ||
        (c.mode==QKG_INDEXED)!=(c.ids!=nullptr) ||
        ((uintptr_t(c.a)|uintptr_t(c.low)|uintptr_t(c.high))&15) ||
        (uintptr_t(c.units)&(c.qtype==8 ? 1 : (c.qtype==12 || c.qtype==13) ? 15 : 3)) ||
        ((uintptr_t(c.output)|uintptr_t(c.workspace)|uintptr_t(c.ids)|uintptr_t(c.offsets))&3)) return QKG_INVALID;
    if (s.workspace_bytes && (!c.workspace || c.workspace_bytes<s.workspace_bytes)) return QKG_CAPACITY;
    uint64_t a_elements=c.mode==QKG_INDEXED ?
        uint64_t(c.rows/c.topk-1)*c.a_token_stride+uint64_t(c.channels-1)*c.a_row_stride+c.k :
        uint64_t(c.rows-1)*c.a_row_stride+c.k;
    uintptr_t p[]={uintptr_t(c.a),uintptr_t(c.low),uintptr_t(c.high),uintptr_t(c.units),
        uintptr_t(c.ids),uintptr_t(c.offsets),uintptr_t(c.output),uintptr_t(c.workspace)};
    uint64_t bytes[]={a_elements*(c.input_type==QKG_F32 ? 4 : 2),s.low_bytes,s.high_bytes,s.units_bytes,
        c.ids ? (uint64_t(c.rows/c.topk-1)*c.ids_stride+c.topk)*4 : 0,
        c.offsets ? (uint64_t(c.experts)+1)*4 : 0,
        (uint64_t(c.rows-1)*c.out_row_stride+c.n)*(d.output_type==QKG_F32 ? 4 : 2),s.workspace_bytes};
    for (int i=0;i<8;++i) {
        if (!execution::span(p[i],bytes[i])) return QKG_OVERFLOW;
        for (int j=0;j<i;++j)
            if (i>=6 && execution::overlap(p[i],bytes[i],p[j],bytes[j])) return QKG_INVALID;
    }
    return QKG_OK;
}

inline int row_buffers(qkg_gate_up_call_v2 const& d,qkg_sizes_v1 const& sizes) {
    if(d.version!=2 || d.size!=sizeof(d) ||
        ((uintptr_t(d.input_rows)|uintptr_t(d.status))&3) ||
        (d.input_rows && d.call.input.call.mode!=QKG_INDEXED)) return QKG_INVALID;
    auto const& c=d.call.input.call;
    uint64_t output_bytes=(uint64_t(c.rows-1)*c.out_row_stride+c.n)*(d.call.output_type==QKG_F32?4:2);
    for(int i=0;i<2;++i) {
        uintptr_t p=uintptr_t(i?d.status:d.input_rows);
        uint64_t bytes=p?(i?4:uint64_t(c.rows)*4):0;
        if(!execution::span(p,bytes)) return QKG_OVERFLOW;
        if(execution::overlap(p,bytes,uintptr_t(c.output),output_bytes) ||
            execution::overlap(p,bytes,uintptr_t(c.workspace),sizes.workspace_bytes)) return QKG_INVALID;
    }
    return QKG_OK;
}
}
