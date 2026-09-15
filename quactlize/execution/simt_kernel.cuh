#pragma once
#include "simt_format.cuh"
#include "q4_s1_validation.hpp"
#include "simt.h"

namespace quactlize::execution::simt {
using q4_s1::Activation;

template<int P>
__device__ __forceinline__ void load_words(uint16_t const* ptr,uint32_t (&out)[P/2]) {
    if constexpr(P==2) out[0]=*reinterpret_cast<uint32_t const*>(ptr);
    else if constexpr(P==4) {
        uint2 v=*reinterpret_cast<uint2 const*>(ptr);out[0]=v.x;out[1]=v.y;
    } else {
        uint4 v=*reinterpret_cast<uint4 const*>(ptr);
        out[0]=v.x;out[1]=v.y;out[2]=v.z;out[3]=v.w;
    }
}

template<int Input,int Columns,int Group>
__device__ __forceinline__ uint4 activation_packet(Activation<Input> a,int group,int lane) {
    constexpr int Count=Group/Columns;
    int64_t pos=int64_t(group)*Group+(lane%Columns)*Count;
    if constexpr(Count==8) return a.load8(pos);
    else if constexpr(Count==4) {
        uint2 v=a.load4(pos);return make_uint4(v.x,v.y,0,0);
    } else {
        __half2_raw v;
        if constexpr(Input==0) v=*reinterpret_cast<__half2 const*>(a.ptr+pos);
        else {
            float2 f=*reinterpret_cast<float2 const*>(a.ptr+pos);
            v=__floats2half2_rn(f.x,f.y);
        }
        return make_uint4(uint32_t(v.x)|(uint32_t(v.y)<<16),0,0,0);
    }
}

template<int Columns,int Group>
__device__ __forceinline__ float2 activation_pair(uint4 packet,int offset,int lane) {
    constexpr int Count=Group/Columns;
    int owner=(lane&~(Columns-1))+offset/Count;
    int index=(offset%Count)/2;
    uint32_t value=index==0 ? packet.x : index==1 ? packet.y : index==2 ? packet.z : packet.w;
    value=__shfl_sync(0xffffffffu,value,owner);
    return make_float2(__half2float(__ushort_as_half(uint16_t(value))),
                       __half2float(__ushort_as_half(uint16_t(value>>16))));
}

template<int Q,int Input,int Variant,int Columns,int Warps,int P>
__global__ void register_reuse(qkg_call_v1 call,int split) {
    using F=Format<Q>;
    constexpr int TileN=Columns*P,Workers=Warps*32/Columns,Pairs=P/2;
    constexpr int Segments=(F::group+F::Low::kLogicalKPerDelivery-1)/F::Low::kLogicalKPerDelivery;
    int tid=threadIdx.x,lane=tid%32,worker=tid/Columns;
    int tile=blockIdx.x%(call.n/TileN),outer=blockIdx.x/(call.n/TileN);
    int partition=outer%split,row=outer/split;
    auto r=q4_s1::locate(call,row);
    bool valid=r.expert>=0 && r.expert<call.experts;
    if (!valid) {
        if (tid<TileN) {
            int n=tile*TileN+tid;
            if (split==1) call.output[r.output+n]=__int_as_float(0x7fc00000);
            else static_cast<float*>(call.workspace)[(int64_t(row)*split+partition)*call.n+n]=__int_as_float(0x7fc00000);
        }
        return;
    }
    int col=tile*TileN+(tid%Columns)*P;
    uint64_t nk=uint64_t(call.n)*call.k;
    auto low=reinterpret_cast<uint16_t const*>(call.low+uint64_t(r.expert)*nk/8*F::low_bits);
    uint16_t const* high=nullptr;
    if constexpr(F::high_bits) high=reinterpret_cast<uint16_t const*>(call.high+uint64_t(r.expert)*nk/8*F::high_bits);
    auto units=call.units+uint64_t(r.expert)*F::metadata_bytes(nk);
    Activation<Input> a{static_cast<typename Activation<Input>::Scalar const*>(call.a)+r.a};
    float2 total[Pairs]{};
    for (int g=partition*Workers+worker;g<call.k/F::group;g+=split*Workers) {
        // K/group is divisible by the number of workers in a warp. Tail
        // warps are wholly inactive, so full-mask exchanges remain legal.
        Meta<F::words> cooperative{};
        if constexpr(Variant&2) {
            if (lane<TileN) cooperative=load_meta<Q>(units,call.n,tile*TileN+lane,g);
        }
        float2 metadata[P];
        #pragma unroll
        for (int p=0;p<P;++p) {
            if constexpr(Variant&2)
                metadata[p]=affine<Q>(share_meta(cooperative,(lane%Columns)*P+p),g);
            else metadata[p]=affine<Q>(load_meta<Q>(units,call.n,col+p,g),g);
        }
        uint4 packet{};
        if constexpr(Variant&1) packet=activation_packet<Input,Columns,F::group>(a,g,lane);
        float2 dot[Pairs]{};float a_sum=0.f;
        #pragma unroll
        for (int half=0;half<2;++half) {
            uint32_t lo[Segments][4][Pairs],hi[4][Pairs]{};
            #pragma unroll
            for (int segment=0;segment<Segments;++segment) {
                #pragma unroll
                for (int residue=0;residue<4;++residue) {
                    int k=g*F::group+segment*F::Low::kLogicalKPerDelivery+half*4+residue;
                    load_words<P>(low+F::Low::word_index(col,k,call.n),lo[segment][residue]);
                }
            }
            if constexpr(F::high_bits) {
                #pragma unroll
                for (int residue=0;residue<4;++residue) {
                    int k=g*F::group+half*4+residue;
                    load_words<P>(high+F::High::word_index(col,k,call.n),hi[residue]);
                }
            }
            #pragma unroll
            for (int slot=0;slot<F::group/8;++slot) {
                float4 av;
                if constexpr(Variant&1) {
                    float2 x=activation_pair<Columns,F::group>(packet,slot*8+half*4,lane);
                    float2 y=activation_pair<Columns,F::group>(packet,slot*8+half*4+2,lane);
                    av=make_float4(x.x,x.y,y.x,y.y);
                } else av=a.values4(g*F::group+slot*8+half*4);
                float values[4]={av.x,av.y,av.z,av.w};
                a_sum+=(av.x+av.y)+(av.z+av.w);
                #pragma unroll
                for (int residue=0;residue<4;++residue) {
                    int k=g*F::group+slot*8+half*4+residue;
                    #pragma unroll
                    for (int p=0;p<Pairs;++p) {
                        float2 code=codes<Q>(lo[slot/F::Low::kPack][residue][p],hi[residue][p],col+2*p,k);
                        dot[p].x=fmaf(values[residue],code.x,dot[p].x);
                        dot[p].y=fmaf(values[residue],code.y,dot[p].y);
                    }
                }
            }
        }
        #pragma unroll
        for (int p=0;p<Pairs;++p) {
            total[p].x+=fmaf(metadata[2*p].x,dot[p].x,metadata[2*p].y*a_sum);
            total[p].y+=fmaf(metadata[2*p+1].x,dot[p].y,metadata[2*p+1].y*a_sum);
        }
    }
    float values[P];
    #pragma unroll
    for (int p=0;p<Pairs;++p) {values[2*p]=total[p].x;values[2*p+1]=total[p].y;}
    float value=q4_s1::q4_reduce_scatter_steps<P,Columns>(values,lane);
    __shared__ float partial[Warps*TileN];
    if (lane<TileN) partial[(tid/32)*TileN+(lane%Columns)*P+lane/Columns]=value;
    __syncthreads();
    if (tid<32) {
        float sum=0;
        #pragma unroll
        for (int w=tid/TileN;w<Warps;w+=32/TileN) sum+=partial[w*TileN+tid%TileN];
        #pragma unroll
        for (int d=TileN;d<32;d*=2) sum+=__shfl_xor_sync(0xffffffffu,sum,d);
        if (tid<TileN) {
            int n=tile*TileN+tid;
            if (split==1) call.output[r.output+n]=sum;
            else static_cast<float*>(call.workspace)[(int64_t(row)*split+partition)*call.n+n]=sum;
        }
    }
}

template<int Q>
__global__ void register_reuse_reduce(qkg_call_v1 c,int split) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=int64_t(c.rows)*c.n) return;
    int64_t row=i/c.n,col=i%c.n;
    float sum=0;
    for (int s=0;s<split;++s) sum+=static_cast<float const*>(c.workspace)[(row*split+s)*c.n+col];
    c.output[row*c.out_row_stride+col]=sum;
}

template<int Q,int Variant,int Columns,int Warps,int P>
int launch(qkg_call_v1 const& c,int split) {
    auto stream=static_cast<hggcStream_t>(c.stream);
    if (hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    int blocks=c.rows*split*(c.n/(Columns*P));
    if (c.input_type==QKG_F32)
        register_reuse<Q,1,Variant,Columns,Warps,P><<<blocks,Warps*32,0,stream>>>(c,split);
    else register_reuse<Q,0,Variant,Columns,Warps,P><<<blocks,Warps*32,0,stream>>>(c,split);
    if (hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    if (split>1) register_reuse_reduce<Q><<<(int64_t(c.rows)*c.n+127)/128,128,0,stream>>>(c,split);
    return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
}
} // namespace quactlize::execution::simt
