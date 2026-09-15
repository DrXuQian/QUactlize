#include "quactlize/execution/simt_validation.hpp"
#include <cassert>
#include <cstdio>

int main() {
    using namespace quactlize::execution;
    for (int q : {8,10,11,12,13,14}) {
        auto a=q==8 ? q8_kpack2::arrangement() : ppu_arrangements::kquant_kpack_transpose_v1(q);
        if (q==12) a=ppu_arrangements::q4_kpack4_transpose_v1();
        qkg_call_v1 c{};
        c.version=1;c.size=sizeof(c);c.qtype=q;c.n=256;c.k=512;c.experts=1;
        c.rows=8;c.mode=QKG_DENSE;c.input_type=QKG_F32;c.channels=1;c.topk=1;
        c.a_row_stride=520;c.a_token_stride=520;c.ids_stride=1;c.out_row_stride=264;
        qkg_simt_config_v1 f{1,sizeof(f),0,4,4,4,8};
        qkg_sizes_v1 sizes{};
        assert(simt::query(c,f,&a,sizes)==QKG_OK);
        assert(sizes.workspace_bytes==8*256*8*4);
        c.rows=9;assert(simt::query(c,f,&a,sizes)==QKG_SHAPE);c.rows=8;
        f.columns=8;f.values=8;assert(simt::query(c,f,&a,sizes)==QKG_INVALID);f.values=4;
        if (q==8) {f.variant=2;assert(simt::query(c,f,&a,sizes)==QKG_INVALID);f.variant=0;}
        c.a_row_stride=513;assert(simt::query(c,f,&a,sizes)==QKG_INVALID);c.a_row_stride=520;
        c.mode=QKG_INDEXED;c.experts=16;c.rows=64;c.channels=8;c.topk=8;
        c.a_token_stride=8*520;c.ids_stride=11;
        assert(simt::query(c,f,&a,sizes)==QKG_OK);
        assert(sizes.workspace_bytes==64*256*8*4);
        c.rows=72;assert(simt::query(c,f,&a,sizes)==QKG_SHAPE);c.rows=64;
        assert(simt::query(c,f,&a,sizes)==QKG_OK);
        c.a=reinterpret_cast<void*>(0x10000000);
        c.low=reinterpret_cast<uint8_t*>(0x20000000);
        c.high=sizes.high_bytes ? reinterpret_cast<uint8_t*>(0x30000000) : nullptr;
        c.units=reinterpret_cast<uint8_t*>(0x40000000);
        c.ids=reinterpret_cast<int32_t*>(0x50000000);
        c.output=reinterpret_cast<float*>(0x60000000);
        c.workspace=reinterpret_cast<void*>(0x70000000);c.workspace_bytes=sizes.workspace_bytes;
        assert(simt::buffers(c,sizes)==QKG_OK);
        --c.workspace_bytes;assert(simt::buffers(c,sizes)==QKG_CAPACITY);++c.workspace_bytes;
        c.output=reinterpret_cast<float*>(0x20000000);assert(simt::buffers(c,sizes)==QKG_INVALID);
        c.output=reinterpret_cast<float*>(0x60000000);
        c.a=reinterpret_cast<void*>(0x10000004);assert(simt::buffers(c,sizes)==QKG_INVALID);
    }
    std::puts("SIMT_HOST PASS all formats, typed strides, high plane, bounds and aliases");
}
