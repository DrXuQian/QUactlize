#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define __host__
#define __device__
#define __forceinline__ inline
struct float2 { float x,y; };
float2 make_float2(float x,float y) { return {x,y}; }
uint16_t __ushort_as_half(uint16_t bits) { return bits; }
float __half2float(uint16_t bits) {
    unsigned const exponent=(bits>>10)&31, fraction=bits&1023;
    float value=exponent ? std::ldexp(float(1024+fraction),int(exponent)-25)
                         : std::ldexp(float(fraction),-24);
    return bits&0x8000 ? -value : value;
}
uint32_t __shfl_sync(unsigned mask,uint32_t value,int owner);
#include "dev/gemv_simt/q6_sparse.cuh"

namespace qs=quactlize::execution::q6_sparse;
void require(bool ok) { if(!ok) std::abort(); }
uint32_t bits(float f) { uint32_t b;std::memcpy(&b,&f,4);return b; }
uint32_t packets[32][4];
unsigned receiving_lane, shuffle_count;
uint32_t __shfl_sync(unsigned mask,uint32_t value,int owner) {
    require(mask==0xffffffffu && owner>=0 && owner<32 && shuffle_count<4);
    require(value==packets[receiving_lane][shuffle_count]);
    return packets[owner][shuffle_count++];
}

struct Words {
    uint32_t word[9];
    mutable unsigned reads=0, mask=0;
    uint32_t operator[](unsigned i) const {
        require(i<9);++reads;mask|=1u<<i;return word[i];
    }
};
float expected(unsigned char const* data,unsigned group) {
    // Independent canonical byte reader, no packet/word-selection logic.
    unsigned const base=((group/16)%2)*18;
    unsigned const header=data[base]+256u*data[base+1];
    int const byte=data[base+2+group%16];
    int const scale=byte<128 ? byte : byte-256;
    return __half2float(__ushort_as_half(uint16_t(header)))*float(scale);
}
qs::Packet packet(unsigned char const* data,unsigned group) {
    Words words;std::memcpy(words.word,data,36);
    auto p=qs::load_words(words,group);
    unsigned const masks[]={0x007,0x01d,0x070,0x190};
    require(words.mask==masks[(group/8)%4]);
    require(words.reads==((group/8)%4==1 ? 4u : 3u));
    auto direct=qs::load(reinterpret_cast<uint8_t const*>(words.word),group);
    require(std::memcmp(&p,&direct,sizeof(p))==0);
    return p;
}
void check(unsigned char const* data,unsigned group) {
    auto got=qs::affine(packet(data,group),group);
    require(bits(got.x)==bits(expected(data,group)) && bits(got.y)==0);
}
int main() {
    uint64_t fields=0, exchanges=0;unsigned negative=0;
    unsigned char data[36];
    for(unsigned i=0;i<36;++i)data[i]=uint8_t(i*37+13);
    for(unsigned h=0;h<65536;++h) {
        if((h&0x7c00)==0x7c00)continue;
        data[0]=data[18]=h&255;data[1]=data[19]=h>>8;
        for(unsigned g=0;g<32;++g) {check(data,g);++fields;}
    }
    data[0]=0;data[1]=0x3c;data[18]=0;data[19]=0x40;
    for(unsigned g=0;g<128;++g)for(unsigned value=0;value<256;++value) {
        data[((g/16)%2)*18+2+g%16]=value;
        check(data,g);++fields;
        auto p=packet(data,g);float good=qs::affine(p,g).x;
        if(bits(good)!=bits(qs::affine(p,g^1).x))negative|=1;
        if(bits(good)!=bits(qs::affine(packet(data,g^8),g).x))negative|=2;
        if(value>=128 && good!=float(value)*(g&16 ? 2.f : 1.f))negative|=4;
    }
    // Different column data per source lane. Receiver and source groups differ.
    unsigned char columns[32][36];
    for(unsigned iter=0;iter<5;++iter)for(unsigned warp=0;warp<8;++warp) {
        for(unsigned lane=0;lane<32;++lane) {
            auto* d=columns[lane];
            for(unsigned i=0;i<36;++i)d[i]=uint8_t(lane*19+i*37+iter*23+warp*3);
            d[0]=uint8_t(lane*7);d[1]=0x3c;d[18]=uint8_t(lane*11);d[19]=0x40;
            auto p=packet(d,iter*64+warp*8+lane/4);
            std::memcpy(packets[lane],&p,sizeof(p));
        }
        for(unsigned lane=0;lane<32;++lane)for(unsigned col=0;col<8;++col) {
            unsigned const g=iter*64+warp*8+lane/4, owner=(lane%4)*8+col;
            receiving_lane=lane;shuffle_count=0;
            qs::Packet p;std::memcpy(&p,packets[lane],sizeof(p));
            auto got=qs::affine(qs::share(p,owner,g),g);
            require(shuffle_count==4);
            require(bits(got.x)==bits(expected(columns[owner],g)) && bits(got.y)==0);
            if(bits(got.x)!=bits(expected(columns[owner^1],g)))negative|=8;
            if(bits(got.x)!=bits(expected(columns[owner],iter*64+warp*8+owner/4)))negative|=16;
            ++exchanges;
        }
    }
    require(negative==31);
    std::printf("Q6_SPARSE_HOST PASS fields=%llu exchanges=%llu negative=DETECTED\n",
                (unsigned long long)fields,(unsigned long long)exchanges);
}
