#pragma once
#include "q4_s1_readers.cuh"
#include "q4_s1_validation.hpp"

namespace quactlize::execution::q4_s1 {
template<int Input,int Reader,int Variant,int Warps,int P,int N,int K>
__global__ void indexed_kernel(qkg_call_v1 c) {
    constexpr int TileN=Reader==QKG_Q4_META ? 8 : 4*P;
    int const row=blockIdx.x/(N/TileN), tile=blockIdx.x%(N/TileN);
    Row const r=locate(c,row);
    float* out=c.output+r.output;
    if (r.expert<0 || r.expert>=c.experts) {
        if (threadIdx.x<TileN) out[tile*TileN+threadIdx.x]=__int_as_float(0x7fc00000);
        return;
    }
    using A=Activation<Input>;
    A const a{static_cast<typename A::Scalar const*>(c.a)+r.a};
    uint8_t const* low=c.low+uint64_t(r.expert)*N*K/2;
    uint8_t const* units=c.units+uint64_t(r.expert)*N*K/16;
    if constexpr(Reader==QKG_Q4_META)
        row_meta<Input,Variant,1,1,Warps,N,K>(tile,a,low,units,out);
    else if constexpr(Reader==QKG_Q4_MEDIUM)
        row_medium<Input,Variant&1,(Variant>>1)&1,1,P==2,P==2,4,Warps,P,N,K>(tile,a,low,units,out);
    else {
        static_assert(Reader==QKG_Q4_REUSE && (Variant==6 || Variant==7));
        row_reuse<Input,Variant,4,Warps,P,N,K>(tile,a,low,units,out);
    }
}

template<int Reader,int Variant,int Warps,int P,int N,int K>
int launch(qkg_call_v1 const& c) {
    constexpr int TileN=Reader==QKG_Q4_META ? 8 : 4*P;
    int const grid=c.rows*(N/TileN);
    auto stream=static_cast<hggcStream_t>(c.stream);
    if (hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    if (c.input_type==QKG_F16)
        indexed_kernel<0,Reader,Variant,Warps,P,N,K><<<grid,Warps*32,0,stream>>>(c);
    else indexed_kernel<1,Reader,Variant,Warps,P,N,K><<<grid,Warps*32,0,stream>>>(c);
    return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
}
} // namespace quactlize::execution::q4_s1
