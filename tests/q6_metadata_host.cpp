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
#include "dev/gemv_simt/q6_metadata.cuh"

struct Words { uint32_t word[9]; };
uint32_t bits(float f) { uint32_t b;std::memcpy(&b,&f,4);return b; }
void require(bool ok) { if(!ok) std::abort(); }
void check(unsigned char const* data,unsigned group) {
    Words m;std::memcpy(&m,data,36);
    // Independent byte reader: no producer layout or register-word selection.
    unsigned const base=((group/16)%2)*18;
    unsigned const header=data[base]+256u*data[base+1];
    int const byte=data[base+2+group%16];
    int const scale=byte<128 ? byte : byte-256;
    float const expected=__half2float(__ushort_as_half(uint16_t(header)))*float(scale);
    auto got=quactlize::execution::q6_metadata::affine(m,group);
    require(bits(got.x)==bits(expected) && bits(got.y)==0);
}
int main() {
    uint64_t checked=0;unsigned negative=0;
    unsigned char data[36];
    for(unsigned i=0;i<36;++i)data[i]=uint8_t(i*37+13);
    for(unsigned h=0;h<65536;++h) {
        if((h&0x7c00)==0x7c00)continue; // finite superscales, both signs/subnormals
        data[0]=data[18]=h&255;data[1]=data[19]=h>>8;
        for(unsigned g=0;g<32;++g) {check(data,g);++checked;}
    }
    data[0]=0;data[1]=0x3c;data[18]=0;data[19]=0x40;
    for(unsigned g=0;g<128;++g)for(unsigned value=0;value<256;++value) {
        auto offset=((g/16)%2)*18+2+g%16;
        data[offset]=value;
        check(data,g);++checked;
        Words m;std::memcpy(&m,data,36);
        auto good=quactlize::execution::q6_metadata::affine(m,g);
        // Wrong K superblock, wrong group and signedness must be detectable.
        if(bits(good.x)!=bits(quactlize::execution::q6_metadata::affine(m,g^16).x))negative|=1;
        if(bits(good.x)!=bits(quactlize::execution::q6_metadata::affine(m,g^1).x))negative|=2;
        if(value>=128 && good.x!=float(value)*(g&16 ? 2.f : 1.f))negative|=4;
    }
    require(negative==7);
    std::printf("Q6_METADATA_HOST PASS cases=%llu negative=DETECTED\n",(unsigned long long)checked);
}
