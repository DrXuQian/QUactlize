#pragma once
#include "model_gemv_scope.hpp"
#include "simt_format.cuh"
#include "q4_s1_validation.hpp"
#include "simt.h"
#include "simt_activation.cuh"
#include <type_traits>

namespace quactlize::execution::simt {

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

template<int Input,int Compute,int Columns,int Group>
__device__ __forceinline__ uint4 activation_packet(Activation<Input,Compute> a,int group,int lane) {
    constexpr int Count=Group/Columns;
    int64_t pos=int64_t(group)*Group+(lane%Columns)*Count;
    if constexpr(Count==8) return a.load8(pos);
    else if constexpr(Count==4) {
        uint2 v=a.load4(pos);return make_uint4(v.x,v.y,0,0);
    } else return make_uint4(a.load2(pos),0,0,0);
}

template<int Compute,int Columns,int Group>
__device__ __forceinline__ float2 activation_pair(uint4 packet,int offset,int lane) {
    constexpr int Count=Group/Columns;
    int owner=(lane&~(Columns-1))+offset/Count;
    int index=(offset%Count)/2;
    uint32_t value=index==0 ? packet.x : index==1 ? packet.y : index==2 ? packet.z : packet.w;
    value=__shfl_sync(0xffffffffu,value,owner);
    if constexpr(Compute==1)
        return make_float2(__uint_as_float(value<<16),__uint_as_float(value&0xffff0000u));
    else return make_float2(__half2float(__ushort_as_half(uint16_t(value))),
                            __half2float(__ushort_as_half(uint16_t(value>>16))));
}

template<int Q,int Changes>
__device__ __forceinline__ float2 affine_selected(Meta<Format<Q>::words> const& m,int group) {
    // Q4/Q5 share the same four-word scale/min header. Decode its cross-word
    // fields with fixed register operands for every recipe, not only tuned
    // ones: the dynamic word-index path loses scale[6] bits 0..3 on PPU.
    // Keep Changes in the kernel identity for existing measured recipes.
    if constexpr(Q==12 || Q==13) {
        static_assert(Format<Q>::words==4 && Format<Q>::Unit::kGroups==8);
        return q4_s1::q4_affine_header32(make_uint4(m.word[0],m.word[1],m.word[2],m.word[3]),unsigned(group)&7u);
    } else return affine<Q>(m,group);
}

template<int Q,int Input,int Variant,int Columns,int Warps,int P,int Compute=0,int Changes=0,
         class Finish=void,class Call=qkg_call_v1>
__device__ __forceinline__ void register_reuse_body(Call call,int split) {
    using F=Format<Q>;
    constexpr int TileN=Columns*P,Workers=Warps*32/Columns,Pairs=P/2;
    constexpr int Segments=(F::group+F::Low::kLogicalKPerDelivery-1)/F::Low::kLogicalKPerDelivery;
    using Index=std::conditional_t<(Changes&2)!=0,unsigned,int>;
    Index tid=threadIdx.x,lane=tid%32,worker=tid/Columns;
    Index tile=blockIdx.x%(call.n/TileN),outer=blockIdx.x/(call.n/TileN);
    Index partition=outer%split,row=outer/split;
    auto r=q4_s1::locate(call,row);
    bool valid=r.expert>=0 && r.expert<call.experts;
    if (!valid) {
        if constexpr(!std::is_void_v<Finish>) {
            Finish::template invalid<TileN>(call,row,tile,partition,split);
        } else if (tid<TileN) {
            int n=tile*TileN+tid;
            if (split==1) call.output[r.output+n]=__int_as_float(0x7fc00000);
            else static_cast<float*>(call.workspace)[(int64_t(row)*split+partition)*call.n+n]=__int_as_float(0x7fc00000);
        }
        return;
    }
    Index col=tile*TileN+(tid%Columns)*P;
    uint64_t nk=uint64_t(call.n)*call.k;
    auto low=reinterpret_cast<uint16_t const*>(call.low+uint64_t(r.expert)*nk/8*F::low_bits);
    uint16_t const* high=nullptr;
    if constexpr(F::high_bits) high=reinterpret_cast<uint16_t const*>(call.high+uint64_t(r.expert)*nk/8*F::high_bits);
    auto units=call.units+uint64_t(r.expert)*F::metadata_bytes(nk);
    Activation<Input,Compute> a{static_cast<typename Activation<Input,Compute>::Scalar const*>(call.a)+r.a};
    float2 total[Pairs]{};
    for (Index g=partition*Workers+worker;g<call.k/F::group;g+=split*Workers) {
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
                metadata[p]=affine_selected<Q,Changes>(share_meta(cooperative,(lane%Columns)*P+p),g);
            else metadata[p]=affine_selected<Q,Changes>(load_meta<Q>(units,call.n,col+p,g),g);
        }
        uint4 packet{};
        if constexpr(Variant&1) packet=activation_packet<Input,Compute,Columns,F::group>(a,g,lane);
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
                    float2 x=activation_pair<Compute,Columns,F::group>(packet,slot*8+half*4,lane);
                    float2 y=activation_pair<Compute,Columns,F::group>(packet,slot*8+half*4+2,lane);
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
    if constexpr(!std::is_void_v<Finish>) {
        Finish::template finish<TileN,Warps>(call,row,tile,partition,split,partial);
    } else if (tid<32) {
        float sum=0;
        if constexpr(Changes&2) {
            q4_s1::q4_medium_fold<0,Warps,TileN>(sum,partial,unsigned(tid));
        } else {
            #pragma unroll
            for (int w=tid/TileN;w<Warps;w+=32/TileN) sum+=partial[w*TileN+tid%TileN];
        }
        #pragma unroll
        for (int d=TileN;d<32;d*=2) sum+=__shfl_xor_sync(0xffffffffu,sum,d);
        if (tid<TileN) {
            int n=tile*TileN+tid;
            if (split==1) call.output[r.output+n]=sum;
            else static_cast<float*>(call.workspace)[(int64_t(row)*split+partition)*call.n+n]=sum;
        }
    }
}

template<int Q,int Input,int Variant,int Columns,int Warps,int P,int Compute=0,int Changes=0>
__global__ void register_reuse(qkg_call_v1 call,int split) {
    register_reuse_body<Q,Input,Variant,Columns,Warps,P,Compute,Changes>(call,split);
}

template<int Q,int Input,int Variant,int Columns,int Warps,int P,int Compute,int Changes>
__global__ void register_reuse_model(qkg_call_v1 c) {
    static_assert(Q==13 && Input==QKG_F32 && Compute==QKG_COMPUTE_BF16 &&
                  Variant==3 && Columns==4 && Warps==2 && P==8 && Changes==3);
    c.n=2048;c.k=512;c.experts=256;c.mode=QKG_INDEXED;c.channels=8;c.topk=8;
    register_reuse_body<Q,Input,Variant,Columns,Warps,P,Compute,Changes>(c,1);
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

template<int Q,int Variant,int Columns,int Warps,int P>
int launch_v2(qkg_simt_call_v2 const& d,int split) {
    auto const& c=d.call;
    if(d.compute_type==QKG_COMPUTE_F16) return launch<Q,Variant,Columns,Warps,P>(c,split);
    auto stream=static_cast<hggcStream_t>(c.stream);
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    int blocks=c.rows*split*(c.n/(Columns*P));
    // Only the two independently measured BF16 M1 indexed incumbents.
    // Keep all other shapes, compute/storage types and recipes unchanged.
    if constexpr(Variant==3 && Columns==4 && ((Q==12 && Warps==4 && P==4) ||
                                            (Q==13 && Warps==2 && P==8))) {
        bool measured=c.input_type==QKG_F32 && c.mode==QKG_INDEXED && c.rows==8 &&
            c.topk==8 && c.experts==256 && split==1 &&
            (Q==12 ? c.n==1024 && c.k==2048 && c.channels==1 :
                     c.n==2048 && c.k==512 && c.channels==8);
        if(measured) {
            if constexpr(Q==13)
                register_reuse_model<Q,1,Variant,Columns,Warps,P,1,3>
                    <<<blocks,Warps*32,0,stream>>>(c);
            else
                register_reuse<Q,1,Variant,Columns,Warps,P,1,1>
                    <<<blocks,Warps*32,0,stream>>>(c,split);
            return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
        }
    }
    if(c.input_type==QKG_F32)
        register_reuse<Q,1,Variant,Columns,Warps,P,1><<<blocks,Warps*32,0,stream>>>(c,split);
    else if(c.input_type==QKG_SIMT_BF16)
        register_reuse<Q,2,Variant,Columns,Warps,P,1><<<blocks,Warps*32,0,stream>>>(c,split);
    else register_reuse<Q,0,Variant,Columns,Warps,P,1><<<blocks,Warps*32,0,stream>>>(c,split);
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    if(split>1) register_reuse_reduce<Q><<<(int64_t(c.rows)*c.n+127)/128,128,0,stream>>>(c,split);
    return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
}
} // namespace quactlize::execution::simt
