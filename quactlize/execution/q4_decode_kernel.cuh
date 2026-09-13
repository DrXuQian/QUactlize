#pragma once
#include "q4_s1_kernel.cuh"
#include "q4_decode.h"

namespace quactlize::execution::q4_decode {
using namespace q4_s1;
template<int Input,int Reader,int Variant,int Warps,int P,int Columns,int N,int K>
__global__ void kernel(qkg_call_v1 c) {
    constexpr int TileN=Reader==QKG_Q4_META ? 8 : Columns*P;
    static_assert(Columns*P<=32);
    int row=blockIdx.x/(N/TileN), tile=blockIdx.x%(N/TileN);
    Row r=locate(c,row);
    float* out=c.output+r.output;
    if(r.expert<0 || r.expert>=c.experts) {
        if(threadIdx.x<TileN) out[tile*TileN+threadIdx.x]=__int_as_float(0x7fc00000);
        return;
    }
    using A=Activation<Input>;
    A a{static_cast<typename A::Scalar const*>(c.a)+r.a};
    auto low=c.low+uint64_t(r.expert)*N*K/2;
    auto units=c.units+uint64_t(r.expert)*N*K/16;
    if constexpr(Reader==QKG_Q4_META) row_meta<Input,Variant,1,1,Warps,N,K>(tile,a,low,units,out);
    else if constexpr(Reader==QKG_Q4_MEDIUM)
        row_medium<Input,Variant&1,(Variant>>1)&1,1,P==2,P==2,Columns,Warps,P,N,K>(tile,a,low,units,out);
    else row_reuse<Input,Variant,Columns,Warps,P,N,K>(tile,a,low,units,out);
}

template<int R,int V,int W,int P,int Columns,int N,int K>
int launch(qkg_call_v1 const& c) {
    constexpr int TileN=R==QKG_Q4_META ? 8 : Columns*P;
    auto stream=static_cast<hggcStream_t>(c.stream);
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    kernel<1,R,V,W,P,Columns,N,K><<<c.rows*(N/TileN),W*32,0,stream>>>(c);
    return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
}
} // namespace quactlize::execution::q4_decode
