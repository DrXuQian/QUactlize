#include <cstdint>
#include <climits>
#include <cstdio>
#include <cstdlib>

#define __host__
#define __device__
#include "dev/gemv_simt/q6_unsigned_map.hpp"
#include "quactlize/include/kquant_kpack_offline.hpp"

namespace qu=quactlize::execution::q6_unsigned;
void require(bool ok) { if(!ok) std::abort(); }
uint64_t checks=0;unsigned negative=0;
template<unsigned Bits> void check(unsigned n,unsigned col,unsigned k) {
    using Original=kquant_kpack::PlaneMap<Bits,16>;
    using Candidate=qu::Plane<Bits>;
    auto index=Candidate::word_index(col,k,n);
    auto slot=Candidate::word_slot(k);
    require(k<=INT_MAX && n<=INT_MAX && col<n);
    require(index==Original::word_index(int(col),int(k),int(n)));
    require(slot==unsigned(Original::word_slot(int(k))));
    require(index==uint64_t((k/(128u/Bits))*8u+k%8u)*n+col);
    require(slot==(k/8u)%(16u/Bits));
    uint32_t lo=k*1664525u+col*1013904223u;
    require(((lo>>(slot*Bits))&((1u<<Bits)-1u))==
            ((lo>>(Original::word_slot(int(k))*Bits))&((1u<<Bits)-1u)));
    unsigned const group=k/16u;
    auto offset=qu::unit_offset(n,col,group);
    require(offset==(uint64_t(group/32u)*n+col)*36u);
    if(index!=qu::Plane<6-Bits>::word_index(col,k,n))negative|=1;
    if(index!=Candidate::word_index(0,k,n))negative|=2;
    if(offset!=(uint64_t(group/16u)*n+col)*36u)negative|=4;
    if(index>UINT32_MAX && index!=uint32_t(index))negative|=8;
    ++checks;
}
int main() {
    unsigned const extents[]={256,512,1024,25600,151936,unsigned(INT_MAX)&~255u};
    for(auto n:extents) {
        unsigned const columns[]={0,1,31,n/2,n-2,n-1};
        for(auto col:columns) {
            for(unsigned k=0;k<65536;++k) {
                check<4>(n,col,k);check<2>(n,col,k);
            }
            for(unsigned tail=0;tail<1024;++tail) {
                check<4>(n,col,unsigned(INT_MAX)-tail);
                check<2>(n,col,unsigned(INT_MAX)-tail);
            }
        }
    }
    // Every live kernel K coordinate and the final increment fit uint32.
    // Query admits Q6 K multiples of512 up to INT_MAX; group steps are64.
    unsigned const largest_k=unsigned(INT_MAX)&~511u;
    unsigned const groups=largest_k/16u;
    require(groups+63u<UINT32_MAX);
    require((groups-1u)*16u+15u<largest_k);
    require(negative==15);
    std::printf("Q6_UNSIGNED_HOST PASS checks=%llu negative=DETECTED wide_address=PASS\n",
                (unsigned long long)checks);
}
