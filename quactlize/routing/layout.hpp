#pragma once
#include "api.h"
#include <climits>
#include <cstddef>

namespace quactlize::routing {
constexpr uint64_t aligned(uint64_t n) { return (n+15)&~uint64_t(15); }
struct Layout {
    int rows=0,tiles=0,tile=0,algorithm=0,stages=0;
    uint64_t prefix=0,ranks=0,bytes=0;
};
inline int layout(qzr_call_v1 const& c,qzr_config_v1 const& p,Layout& l) {
    if(c.version!=1 || c.size!=sizeof(c) || p.version!=1 || p.size!=sizeof(p) ||
       c.tokens<0 || c.experts<1 || c.experts>1024 || c.topk<1 || c.topk>32 ||
       c.topk>c.experts || c.channels<1 || c.channels>c.topk ||
       c.inverse<0 || c.inverse>1 || c.reserved || c.ids_stride<c.topk ||
       c.source_stride<c.channels || p.algorithm<QZR_AUTO || p.algorithm>QZR_SCALAR ||
       (p.tile_rows && p.tile_rows!=256 && p.tile_rows!=512 && p.tile_rows!=1024)) return QZR_INVALID;
    int64_t rows=int64_t(c.tokens)*c.topk;
    if(rows>INT_MAX || (c.tokens>1 &&
       (c.ids_stride>(INT64_MAX/4-c.topk)/(c.tokens-1) ||
        c.source_stride>(INT_MAX-(c.channels-1))/(c.tokens-1)))) return QZR_OVERFLOW;
    l.rows=int(rows);l.tile=p.tile_rows?p.tile_rows:512;
    l.algorithm=p.algorithm?p.algorithm:(rows<=1024?QZR_WARP:QZR_TILED);
    if(l.algorithm==QZR_WARP && c.tokens>8192) return QZR_CAPACITY;
    if(l.algorithm==QZR_SCALAR && rows>512) return QZR_CAPACITY;
    l.tiles=int((rows+l.tile-1)/l.tile);
    l.stages=rows==0 || l.algorithm!=QZR_TILED?1:3;
    l.prefix=aligned(uint64_t(l.tiles)*c.experts*4);
    l.ranks=2*l.prefix;
    l.bytes=rows && l.algorithm==QZR_TILED?l.ranks+aligned(uint64_t(rows)*4):0;
    return QZR_OK;
}
inline bool overlap(void const* a,uint64_t na,void const* b,uint64_t nb) {
    uintptr_t x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    return na && nb && (x<=y ? y-x<na : x-y<nb);
}
inline int buffers(qzr_call_v1 const& c,Layout const& l) {
    uint64_t ids_bytes=c.tokens?(uint64_t(c.tokens-1)*uint64_t(c.ids_stride)+c.topk)*4:0;
    void const* ptrs[]={c.ids,c.src_rows,c.dst_rows,c.offsets,c.workspace};
    uint64_t bytes[]={ids_bytes,uint64_t(l.rows)*4,uint64_t(l.rows)*4,uint64_t(c.experts+1)*4,l.bytes};
    if(c.workspace_bytes<l.bytes) return QZR_CAPACITY;
    for(int i=0;i<5;++i) if(bytes[i]) {
        auto p=reinterpret_cast<uintptr_t>(ptrs[i]);
        if(!p || (p&(i==4?15:3)) || bytes[i]>UINTPTR_MAX-p) return QZR_INVALID;
        for(int j=0;j<i;++j) if(overlap(ptrs[i],bytes[i],ptrs[j],bytes[j])) return QZR_INVALID;
    }
    return QZR_OK;
}
} // namespace quactlize::routing
