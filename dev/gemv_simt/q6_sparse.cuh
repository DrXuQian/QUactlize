#pragma once

// C4 warp: eight consecutive groups, within one canonical 36-byte Q6 unit.
// All lanes have the same phase. Fields are selected for the RECEIVING group,
// not for the source lane's group. No format, rounding or dot-order change.
namespace quactlize::execution::q6_sparse {
struct Packet { uint32_t x, y, z, w; };

template<class Words>
__host__ __device__ __forceinline__ Packet load_words(Words const& p, unsigned group) {
    unsigned const phase=(group>>3)&3u;
    if (phase==0) return {p[0],p[1],p[2],0};
    if (phase==1) return {p[0],p[2],p[3],p[4]};
    if (phase==2) return {p[4],p[5],p[6],0};
    return {p[4],p[7],p[8],0};
}

__device__ __forceinline__ Packet load(uint8_t const* p, unsigned group) {
    return load_words(reinterpret_cast<uint32_t const*>(p),group);
}

__device__ __forceinline__ Packet share(Packet p, int owner, unsigned /* group */) {
    p.x=__shfl_sync(0xffffffffu,p.x,owner);
    p.y=__shfl_sync(0xffffffffu,p.y,owner);
    p.z=__shfl_sync(0xffffffffu,p.z,owner);
    // Fixed four exchanges avoid a masked shuffle and its reconvergence.
    // The unused fourth word is zero; it never requires an extra global load.
    p.w=__shfl_sync(0xffffffffu,p.w,owner);
    return p;
}

__host__ __device__ __forceinline__ float2 affine(Packet const& p, unsigned group) {
    bool const second=(group&16u)!=0;
    unsigned const byte=(group&7u)+(second ? 0u : 2u);
    bool const first=(group&24u)==0;
    uint32_t const a=first ? p.x : p.y;
    uint32_t const b=first ? p.y : p.z;
    uint32_t const c=first ? p.z : p.w;
    uint32_t const packed=byte<4 ? a : byte<8 ? b : c;
    uint32_t const header=second ? p.x>>16 : p.x;
    unsigned const code=(packed>>(8u*(byte&3u)))&255u;
    int const scale=int(code^128u)-128;
    float const d=__half2float(__ushort_as_half(uint16_t(header)));
    return make_float2(d*float(scale),0.f);
}
} // namespace quactlize::execution::q6_sparse
