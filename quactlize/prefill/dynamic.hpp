#pragma once
#include "layout.hpp"

namespace quactlize::prefill {
constexpr uint64_t blas_workspace_bytes=4u<<20;
inline uint64_t dynamic_bytes(qkp_call_v1 const& c,Layout const& l) {
    return l.bytes+(c.weight.experts==1 ? blas_workspace_bytes : 0);
}
inline bool address_free(qkp_call_v1 const& c) {
    auto const& w=c.weight;
    return !c.a && !c.output && !c.src_rows && !c.dst_rows && !c.offsets &&
        !c.workspace && !c.workspace_bytes && !w.low && !w.high && !w.units && !w.output && !w.zero &&
        !w.low_bytes && !w.high_bytes && !w.unit_bytes && !w.output_bytes;
}
inline bool same_shape(qkp_call_v1 const& a,qkp_call_v1 const& b) {
    auto const& x=a.weight;auto const& y=b.weight;
    return a.version==1 && a.size==sizeof(a) && x.version==1 && x.size==sizeof(x) &&
        a.m==b.m && a.device==b.device && a.a_rows==b.a_rows &&
        a.a_stride==b.a_stride && a.output_stride==b.output_stride &&
        x.qtype==y.qtype && x.n==y.n && x.k==y.k && x.experts==y.experts &&
        x.operation==y.operation && x.config==y.config && !x.output && !x.zero && !x.output_bytes &&
        (x.experts>1 || (!a.src_rows && !a.dst_rows && !a.offsets));
}
inline int validate_buffers(qkp_call_v1 const& c,quactlize_ppu_placed_arrangement_v2 const& arr,uint64_t bytes) {
    if (!c.a || !c.output || !c.workspace || c.workspace_bytes<bytes || uintptr_t(c.workspace)%256 ||
        (c.weight.experts>1 && (!c.src_rows || !c.dst_rows || !c.offsets))) return QKG_CAPACITY;
    qkg_sizes_v1 sizes{};
    int rc=execution::sizes(c.weight.qtype,c.weight.n,c.weight.k,c.weight.experts,&arr,sizes);
    if (rc) return rc;
    if (c.weight.low_bytes<sizes.low_bytes || c.weight.high_bytes<sizes.high_bytes ||
        c.weight.unit_bytes<sizes.units_bytes || !c.weight.low || !c.weight.units ||
        (sizes.high_bytes && !c.weight.high) || (!sizes.high_bytes && c.weight.high)) return QKG_CAPACITY;
    uintptr_t pointers[]={uintptr_t(c.workspace),uintptr_t(c.weight.low),uintptr_t(c.weight.high),
        uintptr_t(c.weight.units),uintptr_t(c.output),uintptr_t(c.src_rows),uintptr_t(c.dst_rows),
        uintptr_t(c.offsets),uintptr_t(c.a)};
    uint64_t lengths[]={bytes,sizes.low_bytes,sizes.high_bytes,sizes.units_bytes,
        (uint64_t(c.m-1)*c.output_stride+c.weight.n)*4,
        c.weight.experts>1 ? uint64_t(c.m)*4:0,c.weight.experts>1 ? uint64_t(c.m)*4:0,
        c.weight.experts>1 ? uint64_t(c.weight.experts+1)*4:0,
        (uint64_t(c.a_rows-1)*c.a_stride+c.weight.k)*4};
    for (int i=0;i<9;++i) if (lengths[i]) {
        if (!execution::span(pointers[i],lengths[i]) || pointers[i]%(i<4 ? 16:4)) return QKG_INVALID;
        for (int j=0;j<i;++j)
            if (execution::overlap(pointers[i],lengths[i],pointers[j],lengths[j])) return QKG_INVALID;
    }
    return QKG_OK;
}
} // namespace quactlize::prefill
