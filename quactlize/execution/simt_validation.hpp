#pragma once
#include "simt.h"
#include "validation.hpp"

namespace quactlize::execution::simt {

inline bool config_valid(qkg_simt_config_v1 const& f) {
    return f.version==1 && f.size==sizeof(f) && f.variant>=0 && f.variant<4 &&
        (f.columns==4 || f.columns==8) && (f.warps==2 || f.warps==4 || f.warps==8) &&
        (f.values==2 || f.values==4 || f.values==8) && f.columns*f.values<=32 &&
        (f.split==1 || f.split==2 || f.split==4 || f.split==8);
}

inline int query(qkg_call_v1 const& c, qkg_simt_config_v1 const& f,
    quactlize_ppu_placed_arrangement_v2 const* arrangement, qkg_sizes_v1& out) {
    if (!config_valid(f)) return QKG_INVALID;
    if (c.qtype==8 && (f.variant&2)) return QKG_INVALID;
    qkg_config_v1 shape{1,sizeof(shape),16,4,1};
    int rc=execution::query(c,shape,arrangement,out);
    if (rc) return rc;
    int tokens=c.mode==QKG_INDEXED ? c.rows/c.topk : c.rows;
    if ((c.mode!=QKG_GROUPED && tokens>8) ||
        (c.mode==QKG_GROUPED && c.rows>64)) return QKG_SHAPE;
    int alignment=c.input_type==QKG_F16 ? 8 : 4;
    if (c.a_row_stride%alignment ||
        (c.mode==QKG_INDEXED && c.a_token_stride%alignment)) return QKG_INVALID;
    uint64_t blocks=uint64_t(c.rows)*f.split*(c.n/(f.columns*f.values));
    if (blocks>INT32_MAX) return QKG_OVERFLOW;
    out.workspace_bytes=f.split==1 ? 0 : uint64_t(c.rows)*c.n*f.split*4;
    return QKG_OK;
}

inline int buffers(qkg_call_v1 const& c, qkg_sizes_v1 const& s) {
    if (!c.a || !c.low || !c.units || !c.output ||
        (s.high_bytes!=0)!=(c.high!=nullptr) ||
        (c.mode==QKG_GROUPED)!=(c.offsets!=nullptr) ||
        (c.mode==QKG_INDEXED)!=(c.ids!=nullptr) ||
        ((uintptr_t(c.a)|uintptr_t(c.low)|uintptr_t(c.high))&15) ||
        (uintptr_t(c.units)&(c.qtype==8 ? 1 : (c.qtype==12 || c.qtype==13) ? 15 : 3)) ||
        ((uintptr_t(c.output)|uintptr_t(c.workspace)|uintptr_t(c.ids)|uintptr_t(c.offsets))&3))
        return QKG_INVALID;
    if (s.workspace_bytes && (!c.workspace || c.workspace_bytes<s.workspace_bytes)) return QKG_CAPACITY;
    uint64_t a_elements=c.mode==QKG_INDEXED ?
        uint64_t(c.rows/c.topk-1)*c.a_token_stride+uint64_t(c.channels-1)*c.a_row_stride+c.k :
        uint64_t(c.rows-1)*c.a_row_stride+c.k;
    uintptr_t p[]={uintptr_t(c.a),uintptr_t(c.low),uintptr_t(c.high),uintptr_t(c.units),
        uintptr_t(c.ids),uintptr_t(c.offsets),uintptr_t(c.output),uintptr_t(c.workspace)};
    uint64_t bytes[]={a_elements*(c.input_type==QKG_F16 ? 2 : 4),s.low_bytes,s.high_bytes,s.units_bytes,
        c.ids ? (uint64_t(c.rows/c.topk-1)*c.ids_stride+c.topk)*4 : 0,
        c.offsets ? (uint64_t(c.experts)+1)*4 : 0,
        (uint64_t(c.rows-1)*c.out_row_stride+c.n)*4,s.workspace_bytes};
    for (int i=0;i<8;++i) {
        if (!execution::span(p[i],bytes[i])) return QKG_OVERFLOW;
        for (int j=0;j<i;++j)
            if (i>=6 && execution::overlap(p[i],bytes[i],p[j],bytes[j])) return QKG_INVALID;
    }
    return QKG_OK;
}

inline int query_v2(qkg_simt_call_v2 const& d, qkg_simt_config_v1 const& f,
    quactlize_ppu_placed_arrangement_v2 const* arrangement, qkg_sizes_v1& out) {
    if(d.version!=2 || d.size!=sizeof(d) ||
       (d.compute_type!=QKG_COMPUTE_F16 && d.compute_type!=QKG_COMPUTE_BF16) ||
       d.call.input_type<QKG_F16 || d.call.input_type>QKG_SIMT_BF16 ||
       (d.call.input_type==QKG_SIMT_BF16 && d.compute_type!=QKG_COMPUTE_BF16))
        return QKG_INVALID;
    auto storage=d.call;
    // Shape/alignment/capacity depend on storage width, not its exponent bits.
    if(storage.input_type==QKG_SIMT_BF16) storage.input_type=QKG_F16;
    return query(storage,f,arrangement,out);
}

inline int buffers_v2(qkg_simt_call_v2 const& d, qkg_sizes_v1 const& sizes) {
    auto storage=d.call;
    if(storage.input_type==QKG_SIMT_BF16) storage.input_type=QKG_F16;
    return buffers(storage,sizes);
}
} // namespace quactlize::execution::simt
