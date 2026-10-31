#include "quactlize/routing/layout.hpp"
#include <cassert>
#include <cstdio>
#include <initializer_list>

using namespace quactlize::routing;
static_assert(sizeof(qzr_call_v1)==104 && sizeof(qzr_config_v1)==16 && sizeof(qzr_sizes_v1)==24);
static_assert(offsetof(qzr_call_v1,ids)==48 && offsetof(qzr_call_v1,workspace)==80 && offsetof(qzr_call_v1,stream)==96);

int main() {
    qzr_call_v1 c{};c.version=1;c.size=sizeof(c);c.tokens=5120;c.experts=256;
    c.topk=8;c.channels=8;c.ids_stride=8;c.source_stride=8;
    qzr_config_v1 p{1,sizeof(p),QZR_TILED,512};Layout l;
    assert(layout(c,p,l)==QZR_OK && l.rows==40960 && l.tiles==80 && l.stages==3);
    assert(l.bytes==2*80*256*4+40960*4 && l.ranks==2*l.prefix);
    for(int m: {0,1,3,8,17,129,513,4097,5121,16384,32769,65536,268435455}) {
        c.tokens=m;
        for(int tile: {256,512,1024}) {
            p.tile_rows=tile;
            assert(layout(c,p,l)==0);
            assert(l.rows==int64_t(m)*8 && l.stages==(m?3:1));
            assert(l.tiles==(int64_t(m)*8+tile-1)/tile);
        }
    }
    c.tokens=268435456;assert(layout(c,p,l)==QZR_OVERFLOW);
    c.tokens=8;c.source_stride=INT_MAX;assert(layout(c,p,l)==QZR_OVERFLOW);
    c.source_stride=8;c.ids_stride=INT64_MAX;assert(layout(c,p,l)==QZR_OVERFLOW);
    c.tokens=1;assert(layout(c,p,l)==0);
    c.ids_stride=8;c.tokens=8193;p.algorithm=QZR_WARP;assert(layout(c,p,l)==QZR_CAPACITY);
    c.tokens=8192;assert(layout(c,p,l)==0 && l.bytes==0);
    p.algorithm=QZR_SCALAR;
    for(int m:{0,1,3,8,9,16,32,63,64}) {
        c.tokens=m;assert(layout(c,p,l)==0 && l.algorithm==QZR_SCALAR && l.stages==1 && !l.bytes);
    }
    c.tokens=65;assert(layout(c,p,l)==QZR_CAPACITY);
    c.tokens=512;c.topk=c.channels=c.source_stride=c.ids_stride=1;
    assert(layout(c,p,l)==0);c.tokens=513;assert(layout(c,p,l)==QZR_CAPACITY);
    c.topk=c.channels=c.source_stride=c.ids_stride=8;
    p.algorithm=QZR_AUTO;c.tokens=129;assert(layout(c,p,l)==0 && l.algorithm==QZR_TILED);
    c.tokens=128;assert(layout(c,p,l)==0 && l.algorithm==QZR_WARP);
    p.algorithm=QZR_TILED;c.tokens=4;assert(layout(c,p,l)==0);
    c.ids=reinterpret_cast<int const*>(0x100000);
    c.src_rows=reinterpret_cast<int*>(0x200000);c.dst_rows=reinterpret_cast<int*>(0x300000);
    c.offsets=reinterpret_cast<int*>(0x400000);c.workspace=reinterpret_cast<void*>(0x500000);
    c.workspace_bytes=l.bytes;assert(buffers(c,l)==0);
    --c.workspace_bytes;assert(buffers(c,l)==QZR_CAPACITY);++c.workspace_bytes;
    c.src_rows=const_cast<int*>(c.ids);assert(buffers(c,l)==QZR_INVALID);
    c.src_rows=reinterpret_cast<int*>(0x200001);assert(buffers(c,l)==QZR_INVALID);
    c.src_rows=reinterpret_cast<int*>(UINTPTR_MAX-3);assert(buffers(c,l)==QZR_INVALID);
    c.tokens=0;assert(layout(c,p,l)==0);c.ids=nullptr;c.src_rows=c.dst_rows=nullptr;
    c.workspace=nullptr;c.workspace_bytes=0;assert(buffers(c,l)==0);
    c.offsets=nullptr;assert(buffers(c,l)==QZR_INVALID);
    puts("MOE_ROUTE_HOST PASS");
}
