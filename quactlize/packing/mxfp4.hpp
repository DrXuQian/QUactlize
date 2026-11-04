#pragma once
#include "quactlize_ppu_config.h"
#include <cstdint>

#if defined(__CUDACC__) || defined(__HGGCCC__) || defined(__HGGC__)
#define QZ_MX_HD __forceinline__ __host__ __device__
#else
#define QZ_MX_HD
#endif

namespace quactlize::mxfp4 {
constexpr int qtype=39;
inline quactlize_ppu_placed_arrangement_v2 arrangement() {
    return {2,QUACTLIZE_PPU_LAYOUT_MXFP4_W4FA16_V1,4,0,0,16,32,0,
            QUACTLIZE_PPU_MXFP4_W4FA16_MAPPING_ID};
}

// DeepGEMM-for-sail W4FA16: int32 [E,K/16,2N], E8M0 [E,K/32,N].
// Layout attribution: LICENSE-DeepGEMM-layout at the repository root.
// Coordinates below implement the tile/lane and scale permutations; they
// never interpret a code as an integer quantization level or rescale it.
struct Map {
    QZ_MX_HD static unsigned source_in_tile(unsigned out) {
        unsigned lane=out/32, tile=(out/8)%4, s=out%8;
        unsigned interleaved=(s%4)*2+s/4;
        unsigned row=2*(lane%4)+(interleaved%2)+8*((interleaved%4)/2);
        unsigned col=lane/4+8*(interleaved/4);
        return 256*tile+16*row+col;
    }
    QZ_MX_HD static unsigned scale_in_tile(unsigned out) {
        unsigned s=out%4, swapped=(s%2)*2+s/2;
        unsigned i=(out/4)*4+swapped;
        return i/8+8*(i%8);
    }
    template<class Source>
    QZ_MX_HD static uint32_t word(Source raw,int n,int k,uint64_t index) {
        uint64_t words=uint64_t(n)*k/8, expert=index/words, first=(index%words)*8;
        uint32_t result=0;
        for(unsigned s=0;s<8;++s) {
            uint64_t i=first+s, tiled=(i/1024)*1024+source_in_tile(unsigned(i%1024));
            uint64_t kt=tiled/(uint64_t(n)*16), rest=tiled%(uint64_t(n)*16);
            unsigned kk=unsigned(16*kt+(rest%256)/16), nn=unsigned((rest/256)*16+rest%16);
            uint64_t block=(expert*n+nn)*(k/32)+kk/32;
            uint8_t code=(raw[block*17+1+kk%16] >> (kk%32>=16?4:0))&15;
            result|=uint32_t(code)<<(4*s);
        }
        return result;
    }
    template<class Source>
    QZ_MX_HD static uint8_t scale(Source raw,int n,int k,uint64_t index) {
        uint64_t count=uint64_t(n)*(k/32), expert=index/count, i=index%count;
        uint64_t src=(i/64)*64+scale_in_tile(unsigned(i%64));
        return raw[((expert*n+src%n)*(k/32)+src/n)*17];
    }
};
} // namespace quactlize::mxfp4
#undef QZ_MX_HD
