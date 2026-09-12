// Included inside the unchanged K-pack kernel namespace by reader_reuse.py.
// Register transport only: canonical weight bytes and FP32 dot order stay fixed.
template<int Columns>
__device__ __forceinline__ uint4 q4_cooperative_a_chunk(void const* a,int group,int lane) {
    static_assert(Columns==4 || Columns==8);
    auto ptr=static_cast<__half const*>(a)+group*32+(lane%Columns)*(32/Columns);
    if constexpr(Columns==4) return *reinterpret_cast<uint4 const*>(ptr);
    else {
        uint2 v=*reinterpret_cast<uint2 const*>(ptr);
        return make_uint4(v.x,v.y,0,0);
    }
}

template<int Columns>
__device__ __forceinline__ float4 q4_cooperative_a_read(uint4 chunk,int offset,int lane) {
    int const owner=(lane&~(Columns-1))+offset/(32/Columns);
    uint32_t lo=chunk.x,hi=chunk.y;
    if constexpr(Columns==4) {
        if(offset&4) {lo=chunk.z;hi=chunk.w;}
    }
    lo=__shfl_sync(0xffffffffu,lo,owner);
    hi=__shfl_sync(0xffffffffu,hi,owner);
    return make_float4(__half2float(__ushort_as_half(uint16_t(lo))),
                       __half2float(__ushort_as_half(uint16_t(lo>>16))),
                       __half2float(__ushort_as_half(uint16_t(hi))),
                       __half2float(__ushort_as_half(uint16_t(hi>>16))));
}

__device__ __forceinline__ uint4 q4_cooperative_unit_read(uint4 unit,int owner) {
    return make_uint4(__shfl_sync(0xffffffffu,unit.x,owner),
                      __shfl_sync(0xffffffffu,unit.y,owner),
                      __shfl_sync(0xffffffffu,unit.z,owner),
                      __shfl_sync(0xffffffffu,unit.w,owner));
}

__device__ __forceinline__ float2 q4_affine_header32(uint4 u,unsigned group) {
    // Two 24-bit scale/min streams. Build each using constant 32-bit shifts,
    // then extract its six-bit group; no variable 64-bit shift is required.
    uint32_t scales=(group&4) ? (u.z>>16)|(u.w<<16) : u.y;
    uint32_t mins=(group&4) ? u.w>>8 : (u.y>>24)|(u.z<<8);
    unsigned shift=6*(group&3);
    float sc=float((scales>>shift)&63),mn=float((mins>>shift)&63);
    return make_float2(__half2float(__ushort_as_half(uint16_t(u.x)))*sc,
                      -__half2float(__ushort_as_half(uint16_t(u.x>>16)))*mn);
}
