#pragma once
#include <cstdint>
#include <type_traits>
#include "cutlass/half.h"
#include "cutlass/bfloat16.h"

namespace quactlize::ppu17 {

// Existing packed-unit ABI, NOT the original GGUF block's interleaved scale
// bytes: [half d, half dmin, 4x6 scale, 4x6 min, 4x6 scale, 4x6 min].
// Units are [K/256,N,16 bytes]. The headers remain original GGUF FP16 bits.
struct alignas(16) GgufQ4Unit {
  uint32_t words[4];
  CUTLASS_HOST_DEVICE unsigned field(int group, bool minimum) const {
    int bit=32+(group/4)*48+(minimum?24:0)+(group%4)*6;
    int word=bit/32, shift=bit%32;
    // Register selection rather than a runtime-indexed local array. The
    // latter can spill a perfectly small four-word unit to local memory.
    unsigned current=word==1?words[1]:(word==2?words[2]:words[3]);
    unsigned next=word==1?words[2]:words[3];
    unsigned result=current>>shift;
    if (shift>26) result|=next<<(32-shift);
    return result&63;
  }
  template<class T> struct Affine { T scale, zero; };
  template<class T> CUTLASS_HOST_DEVICE Affine<T> affine(int group) const {
    float d=float(cutlass::half_t::bitcast(uint16_t(words[0])));
    float dm=float(cutlass::half_t::bitcast(uint16_t(words[0]>>16)));
    T scale(d*float(field(group,false)));
    T zero(-dm*float(field(group,true)));
    if constexpr(std::is_same_v<T,cutlass::bfloat16_t>) {
      T correction(8.f*float(scale));
      zero=T(float(zero)+float(correction));
    } else {
      zero=T(float(zero)+8.f*float(scale));
    }
    return {scale,zero};
  }
};
static_assert(sizeof(GgufQ4Unit)==16);

} // namespace quactlize::ppu17
