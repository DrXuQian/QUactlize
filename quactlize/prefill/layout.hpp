#pragma once
#include "api.h"
#include "../execution/validation.hpp"
#include <stdexcept>

namespace quactlize::prefill {
struct Layout {
    uint64_t weights, a, out, rows, ids, count, error, directory, bytes;
};
inline uint64_t reserve(uint64_t& cursor,uint64_t bytes) {
    if (bytes>uint64_t(INT64_MAX)-255-cursor) throw std::overflow_error("prefill workspace overflow");
    auto offset=cursor;cursor+=(bytes+255)&~uint64_t(255);return offset;
}
inline int layout(qkp_call_v1 const& c,quactlize_ppu_placed_arrangement_v2 const* arr,Layout& l) {
    auto w=c.weight;
    qkg_sizes_v1 s{};
    int rc=execution::sizes(w.qtype,w.n,w.k,w.experts,arr,s);
    if (rc) return rc;
    if (c.version!=1 || c.size!=sizeof(c) || w.version!=1 || w.size!=sizeof(w) ||
        w.qtype==8 || w.operation!=1 || w.zero || c.device<0 ||
        c.a_stride<w.k || c.output_stride<w.n || w.n>INT32_MAX-256 ||
        w.k>INT32_MAX-256 || c.m>32768 || c.a_rows<=0 ||
        (w.experts==1 ? c.m<128 || c.m>4096 || c.a_rows!=c.m || c.src_rows || c.dst_rows || c.offsets :
                       w.experts!=256 || c.m<1024 || c.m%8)) return QKG_SHAPE;
    if (w.experts>1 && !(w.config==4 || w.config==5 ||
        ((w.qtype==12 || w.qtype==13) && (w.config==10 || w.config==11)))) return QKG_INVALID;
    if (w.experts==1 && (w.config<0 || w.config>12 ||
        (w.config>=6 && w.qtype!=12 && w.qtype!=13))) return QKG_INVALID;
    if (uint64_t(w.n)*w.k*w.experts>uint64_t(INT64_MAX)/2 ||
        uint64_t(c.a_stride)>uint64_t(INT64_MAX)/4/c.a_rows ||
        uint64_t(c.output_stride)>uint64_t(INT64_MAX)/4/c.m) return QKG_OVERFLOW;
    uint64_t cursor=0;
    l.weights=reserve(cursor,uint64_t(w.n)*w.k*w.experts*2);
    l.a=reserve(cursor,uint64_t(c.m)*w.k*2);l.out=reserve(cursor,uint64_t(c.m)*w.n*2);
    l.rows=reserve(cursor,uint64_t(w.experts)*4);l.ids=reserve(cursor,uint64_t(w.experts)*4);
    l.count=reserve(cursor,4);l.error=reserve(cursor,4);
    l.directory=reserve(cursor,uint64_t(c.m+w.experts+1)*16);
    l.bytes=cursor;
    return 0;
}
} // namespace quactlize::prefill
