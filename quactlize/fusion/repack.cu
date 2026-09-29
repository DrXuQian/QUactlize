#include <hggc_runtime.h>
#include "validation.hpp"

namespace {
__global__ void pair_planes(qkg_gate_up_repack_v1 c,uint64_t low_words,uint64_t unit_words,int unit_width) {
    uint64_t stride=uint64_t(blockDim.x)*gridDim.x;
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<low_words;i+=stride) {
        int physical=int(i%(2*c.n)),side=(physical%8)/4,col=(physical/8)*4+physical%4;
        uint64_t source=i/(2*c.n)*(c.merged?2*c.n:c.n)+col+(c.merged?side*c.n:0);
        auto input=reinterpret_cast<uint16_t const*>(side && !c.merged?c.up_low:c.gate_low);
        reinterpret_cast<uint16_t*>(c.low)[i]=input[source];
    }
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<unit_words;i+=stride) {
        int part=int(i%unit_width),physical=int(i/unit_width%(2*c.n));
        int side=(physical%8)/4,col=(physical/8)*4+physical%4;
        uint64_t source=(i/unit_width/(2*c.n)*(c.merged?2*c.n:c.n)+col+(c.merged?side*c.n:0))*unit_width+part;
        auto input=reinterpret_cast<uint16_t const*>(side && !c.merged?c.up_units:c.gate_units);
        reinterpret_cast<uint16_t*>(c.units)[i]=input[source];
    }
}
}

extern "C" int quactlize_gate_up_repack_v1(qkg_gate_up_repack_v1 const* c,
    qkg_gate_up_layout_v1 const* layout,void* opaque) {
    using namespace quactlize::execution;
    if(!c || !layout || c->version!=1 || c->size!=sizeof(*c) ||
        (c->qtype!=8 && c->qtype!=12) || c->n<=0 || c->n>INT32_MAX/2 ||
        (c->merged!=0 && c->merged!=1)) return QKG_INVALID;
    if(layout->version!=1 || layout->size!=sizeof(*layout) || layout->layout_id!=QKG_GATE_UP_N4_V1)
        return QKG_ARRANGEMENT;
    qkg_sizes_v1 single{};
    int rc=sizes(c->qtype,c->n,c->k,c->experts,&layout->packing,single);
    if(rc) return rc;
    if(single.low_bytes>UINT64_MAX/2 || single.units_bytes>UINT64_MAX/2) return QKG_OVERFLOW;
    uintptr_t pointers[]={uintptr_t(c->gate_low),uintptr_t(c->gate_units),
        uintptr_t(c->up_low),uintptr_t(c->up_units),uintptr_t(c->low),uintptr_t(c->units)};
    uint64_t factor=c->merged?2:1;
    uint64_t bytes[]={single.low_bytes*factor,single.units_bytes*factor,
        c->merged?0:single.low_bytes,c->merged?0:single.units_bytes,single.low_bytes*2,single.units_bytes*2};
    if(bool(c->up_low)!=!c->merged || bool(c->up_units)!=!c->merged) return QKG_INVALID;
    for(int i=0;i<6;++i) {
        if((bytes[i] && !pointers[i]) || (pointers[i]&1)) return QKG_INVALID;
        if(!span(pointers[i],bytes[i])) return QKG_OVERFLOW;
        for(int j=0;j<i;++j) if(i>=4 && overlap(pointers[i],bytes[i],pointers[j],bytes[j])) return QKG_INVALID;
    }
    auto stream=static_cast<hggcStream_t>(opaque);
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    uint64_t words=single.low_bytes;
    unsigned blocks=unsigned((words+255)/256);
    if(blocks>4096) blocks=4096;
    pair_planes<<<blocks,256,0,stream>>>(*c,words,single.units_bytes,c->qtype==8?1:8);
    return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME;
}
