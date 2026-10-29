#pragma once
#include "api.h"
#include "../routing/layout.hpp"

namespace quactlize::moe {
inline bool dtype(int type) { return type>=QZM_F32 && type<=QZM_BF16; }
inline int width(int type) { return type==QZM_F32?4:2; }
inline int rows(int tokens,int topk,int& out) {
    if(tokens<0 || topk<1 || topk>32) return QZR_INVALID;
    int64_t n=int64_t(tokens)*topk;
    if(n>INT_MAX) return QZR_OVERFLOW;
    out=int(n);return QZR_OK;
}
inline int matrix_bytes(int rows,int columns,int64_t stride,int type,uint64_t& bytes,
                        int planes=1,int64_t pitch=0) {
    if(rows<0 || columns<1 || stride<columns || !dtype(type) || planes<1 || pitch<0)
        return QZR_INVALID;
    bytes=0;
    if(!rows) return QZR_OK;
    uint64_t limit=uint64_t(INT64_MAX)/width(type);
    if(uint64_t(rows-1)>(limit-columns)/uint64_t(stride)) return QZR_OVERFLOW;
    uint64_t extent=uint64_t(rows-1)*stride+columns;
    if(planes>1) {
        if(uint64_t(pitch)<extent) return QZR_INVALID;
        if(uint64_t(planes-1)>(limit-extent)/uint64_t(pitch)) return QZR_OVERFLOW;
        extent+=uint64_t(planes-1)*pitch;
    }
    bytes=extent*width(type);return QZR_OK;
}
struct Spans {
    void const* pointers[16]{};
    uint64_t bytes[16]{};
    bool writes[16]{};
    int count=0;
    int add(void const* ptr,uint64_t n,int alignment,bool write) {
        if(!n) return QZR_OK;
        uintptr_t p=reinterpret_cast<uintptr_t>(ptr);
        if(!p || p%alignment || n>UINTPTR_MAX-p || count==16) return QZR_INVALID;
        for(int i=0;i<count;++i)
            if((write || writes[i]) && routing::overlap(ptr,n,pointers[i],bytes[i])) return QZR_INVALID;
        pointers[count]=ptr;bytes[count]=n;writes[count++]=write;return QZR_OK;
    }
};
inline int value(qzm_value_v1 const& v,int nr,int nc,uint64_t& bytes) {
    if(v.order<QZM_GROUPED || v.order>QZM_TOKEN_SLOT || v.state<QZM_PROJECTED ||
       v.state>QZM_ACTIVATED || !dtype(v.round_type) || v.reserved ||
       (v.state==QZM_PARTIALS ? v.dtype!=QZM_F32 : v.partial_count!=1) ||
       (v.state==QZM_ACTIVATED && v.round_type!=QZM_F32)) return QZR_INVALID;
    return matrix_bytes(nr,nc,v.row_stride,v.dtype,bytes,v.partial_count,v.partial_stride);
}
inline int prepare(qzm_prepare_v1 const& c,routing::Layout& l,bool pointers) {
    if(c.version!=1 || c.size!=sizeof(c) || c.route.inverse || c.columns<1 ||
       c.input_row_stride<c.columns || !dtype(c.dtype)) return QZR_INVALID;
    int rc=routing::layout(c.route,c.config,l);if(rc) return rc;
    // This opt-in API does not promote the provisional route AUTO threshold.
    if(c.config.algorithm==QZR_AUTO) return QZR_INVALID;
    uint64_t input_bytes=0,output_bytes=0;
    int64_t input_rows=c.route.tokens?int64_t(c.route.tokens-1)*c.route.source_stride+c.route.channels:0;
    if(input_rows>INT_MAX) return QZR_OVERFLOW;
    rc=matrix_bytes(int(input_rows),c.columns,c.input_row_stride,QZM_F32,input_bytes);if(rc) return rc;
    rc=matrix_bytes(l.rows,c.columns,c.columns,c.dtype,output_bytes);if(rc || !pointers) return rc;
    if(c.route.workspace_bytes<l.bytes) return QZR_CAPACITY;
    Spans s;
    uint64_t ids=c.route.tokens?(uint64_t(c.route.tokens-1)*c.route.ids_stride+c.route.topk)*4:0;
    void const* ptrs[]={c.route.ids,c.input,c.route.src_rows,c.route.dst_rows,c.route.offsets,
        c.route.workspace,c.output,c.inverse_rows,c.expert_rows,c.active_experts,c.active_count};
    uint64_t sizes[]={ids,input_bytes,uint64_t(l.rows)*4,uint64_t(l.rows)*4,uint64_t(c.route.experts+1)*4,
        l.bytes,output_bytes,uint64_t(l.rows)*4,uint64_t(c.route.experts)*4,uint64_t(c.route.experts)*4,4};
    for(int i=0;i<11;++i) {
        rc=s.add(ptrs[i],sizes[i],i==5?16:i==6?width(c.dtype):4,i>=2);if(rc) return rc;
    }
    return QZR_OK;
}
inline int transition(qzm_transition_v1 const& c,uint32_t& ops,bool pointers) {
    ops=0;
    if(c.version!=1 || c.size!=sizeof(c) || c.hidden<1 || c.hidden>INT_MAX/2 ||
       c.merged<0 || c.merged>1 || !dtype(c.output_dtype) || c.output_order<0 || c.output_order>1)
        return QZR_INVALID;
    int nr=0,rc=rows(c.tokens,c.topk,nr);if(rc) return rc;
    bool activated=c.gate.state==QZM_ACTIVATED;
    uint64_t gbytes=0,ubytes=0,obytes=0;
    rc=value(c.gate,nr,c.hidden*((c.merged && !activated)?2:1),gbytes);if(rc) return rc;
    if(!activated && !c.merged) {
        rc=value(c.up,nr,c.hidden,ubytes);if(rc) return rc;
        if(c.up.state==QZM_ACTIVATED) return QZR_INVALID;
    }
    rc=matrix_bytes(nr,c.hidden,c.output_stride,c.output_dtype,obytes);if(rc) return rc;
    if(activated) {
        bool same=c.gate.data && c.gate.data==c.output && c.gate.dtype==c.output_dtype &&
            c.gate.order==c.output_order && c.gate.row_stride==c.output_stride;
        ops=same || !nr?QZM_NOOP:QZM_COPY_ACTIVATED;
    } else {
        ops=QZM_SWIGLU | (c.gate.state==QZM_PARTIALS?QZM_REDUCE_GATE:0) |
            (!c.merged && c.up.state==QZM_PARTIALS?QZM_REDUCE_UP:0);
        if(!nr) ops=QZM_NOOP;
    }
    if(!pointers || !nr) return QZR_OK;
    Spans s;
    rc=s.add(c.gate.data,gbytes,width(c.gate.dtype),false);if(rc) return rc;
    if(ubytes) {rc=s.add(c.up.data,ubytes,width(c.up.dtype),false);if(rc) return rc;}
    bool reorder=c.gate.order!=c.output_order || (ubytes && c.up.order!=c.output_order);
    if(reorder) {
        auto map=c.output_order==QZM_GROUPED?c.row_ids:c.inverse_rows;
        rc=s.add(map,uint64_t(nr)*4,4,false);if(rc) return rc;
    }
    if(ops!=QZM_NOOP) return s.add(c.output,obytes,width(c.output_dtype),true);
    return QZR_OK;
}
inline int finish(qzm_finish_v1 const& c,bool pointers) {
    if(c.version!=1 || c.size!=sizeof(c) || c.reserved || c.columns<1 ||
       c.down.state==QZM_ACTIVATED) return QZR_INVALID;
    int nr=0,rc=rows(c.tokens,c.topk,nr);if(rc) return rc;
    uint64_t dbytes=0,wbytes=0,obytes=0;
    rc=value(c.down,nr,c.columns,dbytes);if(rc) return rc;
    rc=matrix_bytes(c.tokens,c.topk,c.weights_stride,QZM_F32,wbytes);if(rc) return rc;
    rc=matrix_bytes(c.tokens,c.columns,c.output_stride,QZM_F32,obytes);if(rc || !pointers || !nr) return rc;
    Spans s;
    rc=s.add(c.down.data,dbytes,width(c.down.dtype),false);if(rc) return rc;
    rc=s.add(c.weights,wbytes,4,false);if(rc) return rc;
    if(c.down.order==QZM_GROUPED) {
        rc=s.add(c.inverse_rows,uint64_t(nr)*4,4,false);if(rc) return rc;
    }
    return s.add(c.output,obytes,4,true);
}
} // namespace quactlize::moe
