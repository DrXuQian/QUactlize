// Enumerate the actual offline word maps before pairing adjacent N columns.
// The negative reproduces the erroneous Q5 assumption: N and N+1 share a
// high word. In fact Q5 shares N and N+8; adjacent even/odd N words are b32.
#include "kquant_kpack_offline.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>

template<class Map, bool Q5=false>
void check(char const* name) {
    constexpr int N=32,K=512;
    std::vector<uint8_t> data(Map::placed_bytes(N,K));
    auto code=[](int n,int k) {return (n+3*(k/8)+k%8)&((1<<Map::kBits)-1);};
    for(int n=0;n<N;++n) for(int k=0;k<K;++k)
        Map::placed_put(data.data(),n,k,N,code(n,k));
    int bad=0,legacy_bad=0;
    for(int n=0;n<N;n+=2) for(int k=0;k<K;++k) {
        size_t first=Map::word_index(n,k,N),second=Map::word_index(n+1,k,N);
        if ((first&1) || second!=first+1) throw std::runtime_error("N2 vector is not adjacent/aligned");
        uint32_t words=0;
        for(int b=0;b<4;++b) words|=uint32_t(data[2*first+b])<<(8*b);
        for(int lane=0;lane<2;++lane) {
            int slot;
            if constexpr(Q5) slot=Map::word_slot(n+lane,k);
            else slot=Map::word_slot(k);
            int got=(words>>(16*lane+Map::kBits*slot))&((1<<Map::kBits)-1);
            bad+=got!=code(n+lane,k);
            if constexpr(Q5) {
                int legacy=(uint16_t(words)>>slot)&1;
                legacy_bad+=legacy!=code(n+lane,k);
            }
        }
    }
    if(bad || (Q5 && legacy_bad!=N*K/2)) throw std::runtime_error("N2 code oracle/negative failed");
    std::printf("GEMV_N2_LAYOUT format=%s cells=%d bad=%d adjacent_aligned=PASS legacy_duplicate_high_bad=%d\n",
                name,N*K,bad,legacy_bad);
}
int main() {
    using namespace kquant_kpack;
    try {
        check<PlaneMap<2,16>>("Q2-low/Q3-low");
        check<PlaneMap<1,16>>("Q3-high");
        check<PlaneMap<4,32>>("Q4-low/Q5-low");
        check<Q5HighPlaneMap,true>("Q5-high");
        check<PlaneMap<4,16>>("Q6-low");
        check<PlaneMap<2,16>>("Q6-high");
        check<PlaneMap<8,32>>("Q8-low");
    } catch(std::exception const& e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
