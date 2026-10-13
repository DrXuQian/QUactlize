#pragma once
#include "api.h"
#include <climits>
#include <cstring>
#include <stdexcept>

namespace quactlize::dense {
inline int admission(quactlize_ppu_dense_params_v1 const* p) {
    if (!p || p->version!=1 || p->size!=sizeof(*p)) return -1;
    if ((p->qtype!=8 && (p->qtype<10 || p->qtype>14)) || p->m<0 ||
        p->m>INT32_MAX-256 || p->n<=0 || p->n>INT32_MAX/8 || p->n%256 ||
        p->k<=0 || p->k>INT32_MAX/8 || p->k%((p->qtype==11 || p->qtype==14)?512:256) ||
        p->input_stride<p->k || p->output_stride<p->n) return 0;
    uint64_t m=p->m ? p->m : 1;
    if (m*uint64_t(p->n)>INT32_MAX ||
        uint64_t(p->input_stride)>uint64_t(INT64_MAX)/4/m ||
        uint64_t(p->output_stride)>uint64_t(INT64_MAX)/4/m) return 0;
    quactlize_ppu_placed_arrangement_v2 arrangement{};
    quactlize_ppu_kpack_sizes_v1 sizes{};
    if (quactlize_ppu_kpack_canonical_arrangement_v1(p->qtype,&arrangement) ||
        std::memcmp(&arrangement,&p->arrangement,sizeof(arrangement)) ||
        quactlize_ppu_kpack_sizes_for_arrangement_v1(p->n,p->k,1,p->qtype,&arrangement,&sizes) ||
        std::memcmp(&sizes,&p->sizes,sizeof(sizes))) return 0;
    return 1;
}
inline uint64_t reserve(uint64_t& cursor,uint64_t bytes) {
    if (bytes>uint64_t(INT64_MAX)-255-cursor) throw std::overflow_error("dense workspace overflow");
    auto offset=cursor;cursor+=(bytes+255)&~uint64_t(255);return offset;
}
inline bool overlaps(void const* p,uint64_t n,void const* q,uint64_t m) {
    auto a=uintptr_t(p),b=uintptr_t(q);
    if (!n || !m) return false;
    if (!p || !q || n>UINTPTR_MAX-a || m>UINTPTR_MAX-b) return true;
    return a<b+m && b<a+n;
}
inline bool buffers(quactlize_ppu_dense_params_v1 const& p,void const* low,void const* high,
    void const* units,void const* a,void* out,void* workspace,uint64_t available,uint64_t needed) {
    if (!a || !out || (uintptr_t(a)|uintptr_t(out))%16 || available<needed ||
        (needed && (!workspace || uintptr_t(workspace)%256))) return false;
    uint64_t ab=(uint64_t(p.m-1)*p.input_stride+p.k)*4;
    uint64_t cb=(uint64_t(p.m-1)*p.output_stride+p.n)*4;
    void const* ptrs[]={a,out,low,high,units,workspace};
    uint64_t sizes[]={ab,cb,p.sizes.low_bytes,p.sizes.high_bytes,p.sizes.units_bytes,needed};
    for (int i=0;i<6;++i) {
        if (sizes[i] && (!ptrs[i] || sizes[i]>UINTPTR_MAX-uintptr_t(ptrs[i]))) return false;
        for (int j=0;j<i;++j) if (overlaps(ptrs[i],sizes[i],ptrs[j],sizes[j])) return false;
    }
    return true;
}
} // namespace quactlize::dense
