#pragma once
#include "simt_kernel.cuh"
#include "../decode/reducer.cuh"

// K-pack2 vector reader. Preserve the register-reuse FP32 affine/K order.
namespace quactlize::execution::simt::q8_vector {

template<int P>
__device__ __forceinline__ void scales(uint8_t const* p,float (&out)[P]) {
    uint32_t words[P/2];
    if ((uintptr_t(p)&(2*P-1))==0) {
        load_words<P>(reinterpret_cast<uint16_t const*>(p),words);
        #pragma unroll
        for (int i=0;i<P/2;++i) {
            __half2_raw h;h.x=uint16_t(words[i]);h.y=uint16_t(words[i]>>16);
            float2 v=__half22float2(__half2(h));out[2*i]=v.x;out[2*i+1]=v.y;
        }
    } else {
        // Q8's public metadata contract guarantees only two-byte alignment.
        #pragma unroll
        for(int i=0;i<P;++i) out[i]=__half2float(__ushort_as_half(reinterpret_cast<uint16_t const*>(p)[i]));
    }
}

template<int Slot>
__device__ __forceinline__ float2 code_pair(uint32_t word) {
    uint32_t bits;
    uint32_t source=Slot ? word>>8 : word;
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(bits)
        : "r"(source),"n"(0x00ff00ffu),"n"(0x64006400u));
    __half2_raw h;h.x=uint16_t(bits);h.y=uint16_t(bits>>16);
    return __half22float2(__hsub2(__half2(h),__float2half2_rn(1152.f)));
}

template<int Input,int Compute,int Variant,int Columns,int Warps,int P,bool Hoist=false,
         class Finish=void,class Call=qkg_call_v1>
__device__ __forceinline__ void kernel_body(Call c,int split) {
    constexpr int TileN=Columns*P,Workers=Warps*32/Columns,Pairs=P/2;
    static_assert(Variant==0 || Variant==1);
    int tid=threadIdx.x,lane=tid%32,worker=tid/Columns;
    int tile=blockIdx.x%(c.n/TileN),outer=blockIdx.x/(c.n/TileN);
    int partition=outer%split,row=outer/split;
    auto r=q4_s1::locate(c,row);
    if(r.expert<0 || r.expert>=c.experts) {
        if constexpr(!std::is_void_v<Finish>) {
            Finish::template invalid<TileN>(c,row,tile,partition,split);
        } else if(tid<TileN) {
            int n=tile*TileN+tid;
            if(split==1) c.output[r.output+n]=__int_as_float(0x7fc00000);
            else static_cast<float*>(c.workspace)[(int64_t(row)*split+partition)*c.n+n]=__int_as_float(0x7fc00000);
        }
        return;
    }
    int col=tile*TileN+(tid%Columns)*P;
    uint64_t nk=uint64_t(c.n)*c.k;
    auto low=reinterpret_cast<uint16_t const*>(c.low+uint64_t(r.expert)*nk);
    auto units=c.units+uint64_t(r.expert)*nk/16;
    Activation<Input,Compute> a{static_cast<typename Activation<Input,Compute>::Scalar const*>(c.a)+r.a};
    float2 total[Pairs]{};
    for(int g=partition*Workers+worker;g<c.k/32;g+=split*Workers) {
        float d[P];scales<P>(units+2*(uint64_t(g)*c.n+col),d);
        uint4 packet{};
        if constexpr(Variant) packet=activation_packet<Input,Compute,Columns,32>(a,g,lane);
        float2 dot[Pairs]{};
        if constexpr(Hoist) {
            // Expose independent loads without changing the dot-product order.
            // A separate specialization keeps the Split-K register window small.
            uint32_t words[2][2][4][Pairs];
            #pragma unroll
            for(int half=0;half<2;++half)
                #pragma unroll
                for(int segment=0;segment<2;++segment)
                    #pragma unroll
                    for(int residue=0;residue<4;++residue)
                        load_words<P>(low+(uint64_t(g)*16+segment*8+half*4+residue)*c.n+col,words[half][segment][residue]);
            #pragma unroll
            for(int half=0;half<2;++half)
                #pragma unroll
                for(int segment=0;segment<2;++segment)
                    #pragma unroll
                    for(int slot=0;slot<2;++slot) {
                        int offset=segment*16+slot*8+half*4;
                        float4 av;
                        if constexpr(Variant) {
                            float2 x=activation_pair<Compute,Columns,32>(packet,offset,lane);
                            float2 y=activation_pair<Compute,Columns,32>(packet,offset+2,lane);
                            av=make_float4(x.x,x.y,y.x,y.y);
                        } else av=a.values4(g*32+offset);
                        float values[4]={av.x,av.y,av.z,av.w};
                        #pragma unroll
                        for(int residue=0;residue<4;++residue)
                            #pragma unroll
                            for(int p=0;p<Pairs;++p) {
                                float2 q=slot==0?code_pair<0>(words[half][segment][residue][p]):code_pair<1>(words[half][segment][residue][p]);
                                dot[p].x=fmaf(values[residue],q.x,dot[p].x);
                                dot[p].y=fmaf(values[residue],q.y,dot[p].y);
                            }
                    }
        } else {
            // Keep the original narrow load window outside the hoist scope.
            #pragma unroll
            for(int half=0;half<2;++half)
                #pragma unroll
                for(int segment=0;segment<2;++segment) {
                    uint32_t words[4][Pairs];
                    #pragma unroll
                    for(int residue=0;residue<4;++residue)
                        load_words<P>(low+(uint64_t(g)*16+segment*8+half*4+residue)*c.n+col,words[residue]);
                    #pragma unroll
                    for(int slot=0;slot<2;++slot) {
                        int offset=segment*16+slot*8+half*4;
                        float4 av;
                        if constexpr(Variant) {
                            float2 x=activation_pair<Compute,Columns,32>(packet,offset,lane);
                            float2 y=activation_pair<Compute,Columns,32>(packet,offset+2,lane);
                            av=make_float4(x.x,x.y,y.x,y.y);
                        } else av=a.values4(g*32+offset);
                        float values[4]={av.x,av.y,av.z,av.w};
                        #pragma unroll
                        for(int residue=0;residue<4;++residue)
                            #pragma unroll
                            for(int p=0;p<Pairs;++p) {
                                float2 q=slot==0?code_pair<0>(words[residue][p]):code_pair<1>(words[residue][p]);
                                dot[p].x=fmaf(values[residue],q.x,dot[p].x);
                                dot[p].y=fmaf(values[residue],q.y,dot[p].y);
                            }
                    }
                }
        }
        #pragma unroll
        for(int p=0;p<Pairs;++p) {total[p].x+=d[2*p]*dot[p].x;total[p].y+=d[2*p+1]*dot[p].y;}
    }
    float values[P];
    #pragma unroll
    for(int p=0;p<Pairs;++p) {values[2*p]=total[p].x;values[2*p+1]=total[p].y;}
    float value=q4_s1::q4_reduce_scatter_steps<P,Columns>(values,lane);
    __shared__ float partial[Warps*TileN];
    if(lane<TileN) partial[(tid/32)*TileN+(lane%Columns)*P+lane/Columns]=value;
    __syncthreads();
    if constexpr(!std::is_void_v<Finish>) {
        Finish::template finish<TileN,Warps>(c,row,tile,partition,split,partial);
    } else if(tid<32) {
        float sum=0;q4_s1::q4_medium_fold<0,Warps,TileN>(sum,partial,tid);
        #pragma unroll
        for(int d=TileN;d<32;d*=2) sum+=__shfl_xor_sync(0xffffffffu,sum,d);
        if(tid<TileN) {
            int n=tile*TileN+tid;
            if(split==1) c.output[r.output+n]=sum;
            else static_cast<float*>(c.workspace)[(int64_t(row)*split+partition)*c.n+n]=sum;
        }
    }
}
template<int Input,int Compute,int Variant,int Columns,int Warps,int P,bool Hoist=false>
__global__ void kernel(qkg_call_v1 c,int split) {
    kernel_body<Input,Compute,Variant,Columns,Warps,P,Hoist>(c,split);
}

template<int Q,int Variant,int Columns,int Warps,int P>
int launch_v2(qkg_simt_call_v2 const& d,int split) {
    static_assert(Q==8 && (Variant==4 || Variant==5));
    if(d.call.input_type!=QKG_F32) return QKG_INVALID;
    auto stream=static_cast<hggcStream_t>(d.call.stream);
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    int blocks=d.call.rows*split*(d.call.n/(Columns*P));
    if(d.compute_type==QKG_COMPUTE_F16) {
        auto const& c=d.call;
        if constexpr(Variant==5 && P==4 && ((Columns==4 && Warps==8) || (Columns==8 && Warps==4))) {
            bool hoist=split==1 && c.mode==QKG_DENSE && c.rows==1 && c.experts==1 &&
                ((Columns==4 && c.n==512 && c.k==2048) || (Columns==8 && c.n==2048 && c.k==512));
            if(hoist)
                kernel<QKG_F32,QKG_COMPUTE_F16,Variant-4,Columns,Warps,P,true><<<blocks,Warps*32,0,stream>>>(c,split);
            else
                kernel<QKG_F32,QKG_COMPUTE_F16,Variant-4,Columns,Warps,P><<<blocks,Warps*32,0,stream>>>(c,split);
        } else kernel<QKG_F32,QKG_COMPUTE_F16,Variant-4,Columns,Warps,P><<<blocks,Warps*32,0,stream>>>(c,split);
    } else if(d.compute_type==QKG_COMPUTE_BF16) {
        kernel<QKG_F32,QKG_COMPUTE_BF16,Variant-4,Columns,Warps,P><<<blocks,Warps*32,0,stream>>>(d.call,split);
    } else return QKG_INVALID;
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    if(split>1) {
        auto const& c=d.call;
        // The admitted M1 S8 call includes this ordered float2 reducer.
        // Public buffers need only four-byte alignment: keep scalar fallback.
        bool paired=Variant==5 && Columns==8 && Warps==4 && P==4 && split==8 &&
            d.compute_type==QKG_COMPUTE_F16 && c.mode==QKG_DENSE && c.rows==1 &&
            c.n==2048 && c.k==4096 && c.experts==1 &&
            !((uintptr_t(c.output)|uintptr_t(c.workspace))&7);
        if(paired) quactlize::decode::reduce_decode<8><<<(c.n+63)/64,32,0,stream>>>(
            static_cast<float const*>(c.workspace),c.output,c.n);
        else register_reuse_reduce<8><<<(int64_t(c.rows)*c.n+127)/128,128,0,stream>>>(c,split);
    }
    return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
}

template<int Q,int Variant,int Columns,int Warps,int P>
int launch(qkg_call_v1 const& c,int split) {
    qkg_simt_call_v2 d{2,sizeof(d),c,QKG_COMPUTE_F16};
    return launch_v2<Q,Variant,Columns,Warps,P>(d,split);
}
} // namespace quactlize::execution::simt::q8_vector
