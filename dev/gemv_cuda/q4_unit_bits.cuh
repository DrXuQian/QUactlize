// Exact 6-bit fields from the two 48-bit packed-unit runs, without 64-bit
// variable shifts. Extra high bits are discarded by the final six-bit masks.
#include <cuda_runtime.h>
#include <cstdint>
__device__ __forceinline__ uint2 q4_unit_codes(uint4 u,unsigned group) {
    unsigned shift=6*(group&3);
    uint32_t scales=(group&4) ? (u.z>>16)|(u.w<<16) : u.y;
    uint32_t minima=(group&4) ? u.w>>8 : (u.y>>24)|(u.z<<8);
    return make_uint2((scales>>shift)&63,(minima>>shift)&63);
}
