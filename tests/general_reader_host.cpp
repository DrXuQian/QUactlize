#include <cstdint>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <array>

#define __host__
#define __device__
#define __forceinline__ inline
struct float2 { float x,y; };
struct uint4 { uint32_t x,y,z,w; };
float2 make_float2(float x,float y) { return {x,y}; }
uint4 make_uint4(uint32_t x,uint32_t y,uint32_t z,uint32_t w) { return {x,y,z,w}; }
uint16_t __ushort_as_half(uint16_t b) { return b; }
float __half2float(uint16_t b) {
    unsigned e=(b>>10)&31,f=b&1023;
    float v=e?std::ldexp(float(1024+f),int(e)-25):std::ldexp(float(f),-24);
    return b&0x8000 ? -v:v;
}
namespace quactlize::execution::q4_s1 { float2 q4_affine_header32(uint4,unsigned); }
uint32_t __shfl_sync(unsigned,uint32_t,int);
#include "dev/gemv_simt/general_map.hpp"
#include "dev/gemv_simt/general_metadata.cuh"
#include "quactlize/include/kquant_kpack_offline.hpp"

namespace gm=quactlize::execution::general_map;
namespace md=quactlize::execution::general_metadata;
void require(bool ok) { if(!ok) std::abort(); }
uint32_t bits(float v) { uint32_t b;std::memcpy(&b,&v,4);return b; }
uint32_t packets[32][4];unsigned receiving,shuffles;
uint32_t __shfl_sync(unsigned mask,uint32_t v,int owner) {
    require(mask==0xffffffffu && owner>=0 && owner<32 && shuffles<4);
    require(v==packets[receiving][shuffles]);return packets[owner][shuffles++];
}
uint64_t fields=0,maps=0,exchanges=0;unsigned negative=0;
unsigned bitfield(uint8_t const* b,unsigned bit,unsigned width) {
    unsigned r=0;for(unsigned j=0;j<width;++j) r|=((b[(bit+j)/8]>>((bit+j)%8))&1u)<<j;
    return r;
}
template<int Q> float2 expected(uint8_t const* b,unsigned g) {
    unsigned sb=Q==10?0:(g/16)%2,base=sb*(Q==11?14:18);
    unsigned h=b[base]+256u*b[base+1];float d=__half2float(h);
    if constexpr(Q==10) {
        float dm=__half2float(b[2]+256u*b[3]);
        unsigned bit=32+(g%16/8)*64+(g%8)*4;
        return {d*float(bitfield(b,bit,4)),-dm*float(bitfield(b,bit+32,4))};
    } else if constexpr(Q==11) {
        return {d*float(int(bitfield(b,base*8+16+(g%16)*6,6))-32),0.f};
    } else {
        int s=b[base+2+g%16];s=s<128?s:s-256;return {d*float(s),0.f};
    }
}
template<int Q> void metadata() {
    constexpr unsigned Words=Q==10?5:Q==11?7:9;
    uint32_t columns[32][Words];
    for(unsigned seed=0;seed<192;++seed) {
        for(unsigned lane=0;lane<32;++lane) {
            auto* b=reinterpret_cast<uint8_t*>(columns[lane]);
            for(unsigned i=0;i<Words*4;++i)b[i]=uint8_t(seed*23+lane*31+i*17);
            b[0]=uint8_t(seed);b[1]=0x30|(seed&0x80);
            unsigned second=Q==10?2:Q==11?14:18;
            b[second]=uint8_t(seed*3);b[second+1]=0x38|(seed&0x80);
            for(unsigned g=0;g<64;++g) {
                auto p=md::load_words<Q>(columns[lane],g);
                auto a=md::affine<Q>(p,g),e=expected<Q>(b,g);
                require(bits(a.x)==bits(e.x) && bits(a.y)==bits(e.y));++fields;
                auto wrong=md::affine<Q>(p,g^1u);
                if(bits(a.x)!=bits(wrong.x)||bits(a.y)!=bits(wrong.y))negative|=1;
            }
        }
        for(unsigned c: {4u,8u})for(unsigned phase=0;phase<8;++phase) {
            for(unsigned lane=0;lane<32;++lane) {
                auto p=md::load_words<Q>(columns[lane],phase*(32/c)+lane/c);
                std::memcpy(packets[lane],&p,16);
            }
            for(unsigned lane=0;lane<32;++lane)for(unsigned p=0;p<32/c;++p) {
                unsigned g=phase*(32/c)+lane/c,owner=lane%c*(32/c)+p;
                receiving=lane;shuffles=0;md::Packet packet;
                std::memcpy(&packet,packets[lane],16);
                auto a=md::affine<Q>(md::share(packet,owner),g);
                auto e=expected<Q>(reinterpret_cast<uint8_t*>(columns[owner]),g);
                require(shuffles==4 && bits(a.x)==bits(e.x) && bits(a.y)==bits(e.y));++exchanges;
            }
        }
    }
}
template<unsigned B> void address(unsigned n,unsigned col,unsigned k) {
    using Old=kquant_kpack::PlaneMap<B,16>;
    require(gm::Plane<B>::word_index(col,k,n)==Old::word_index(col,k,n));
    require(gm::Plane<B>::word_slot(k)==unsigned(Old::word_slot(k)));++maps;
}
int main() {
    metadata<10>();metadata<11>();metadata<14>();
    for(unsigned n:{256u,512u,1024u,5120u,151936u,unsigned(INT_MAX)&~255u})
        for(unsigned col:{0u,7u,8u,15u,16u,n-1})
            for(unsigned kk=0;kk<8192;++kk) {
                unsigned k=kk<4096?kk:unsigned(INT_MAX)-8191+kk;
                address<1>(n,col,k);address<2>(n,col,k);address<4>(n,col,k);address<8>(n,col,k);
                auto hi=gm::Q5High::word_index(col,k,n);
                require(hi==kquant_kpack::Q5HighPlaneMap::word_index(col,k,n));
                require(gm::Q5High::word_slot(col,k)==unsigned(kquant_kpack::Q5HighPlaneMap::word_slot(col,k)));
                if(hi!=gm::Plane<1>::word_index(col,k,n))negative|=2;
                if(hi>UINT32_MAX && hi!=uint32_t(hi))negative|=4;
            }
    require(negative==7);
    std::printf("GENERAL_READER_HOST PASS fields=%llu exchanges=%llu maps=%llu negatives=DETECTED\n",
        (unsigned long long)fields,(unsigned long long)exchanges,(unsigned long long)maps);
}
