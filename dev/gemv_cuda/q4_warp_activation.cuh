// Transpose eight lanes x four contiguous FP16 values entirely in registers.
// Input lane r owns A[4*r+j]; output lane r owns A[r+8*j]. No new A layout.
#include <cuda_fp16.h>
__device__ __forceinline__ uint32_t q4_swap_lane_half(uint32_t value,int lane,int bit) {
    uint32_t other=__shfl_xor_sync(0xffffffffu,value,bit);
    return __byte_perm(value,other,(lane&bit) ? 0x3276 : 0x5410);
}
__device__ __forceinline__ float4 q4_warp_activation(void const* a,int group,int residue) {
    auto ptr=static_cast<__half const*>(a)+group*32+residue*4;
    uint2 v=*reinterpret_cast<uint2 const*>(ptr);
    v.x=q4_swap_lane_half(v.x,residue,1);
    v.y=q4_swap_lane_half(v.y,residue,1);
    uint32_t exchange=__shfl_xor_sync(0xffffffffu,(residue&2) ? v.x : v.y,2);
    if(residue&2) v.x=exchange; else v.y=exchange;
    v.x=q4_swap_lane_half(v.x,residue,4);
    v.y=q4_swap_lane_half(v.y,residue,4);
    return make_float4(__half2float(__ushort_as_half(uint16_t(v.x))),
                       __half2float(__ushort_as_half(uint16_t(v.y))),
                       __half2float(__ushort_as_half(uint16_t(v.x>>16))),
                       __half2float(__ushort_as_half(uint16_t(v.y>>16))));
}
