#pragma once
#include "quactlize/execution/simt_kernel.cuh"

// Q8-only experiment. Same K-pack2 bytes and F32 group-affine order as the
// shipping register-reuse reader. The shipping inventory is untouched.
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

template<int Input,int Compute,int Variant,int Columns,int Warps,int P>
__global__ void kernel(qkg_call_v1 c,int split) {
    constexpr int TileN=Columns*P,Workers=Warps*32/Columns,Pairs=P/2;
    static_assert(Variant==0 || Variant==1);
    int tid=threadIdx.x,lane=tid%32,worker=tid/Columns;
    int tile=blockIdx.x%(c.n/TileN),outer=blockIdx.x/(c.n/TileN);
    int partition=outer%split,row=outer/split;
    auto r=q4_s1::locate(c,row);
    if(r.expert<0 || r.expert>=c.experts) {
        if(tid<TileN) {
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
        #pragma unroll
        for(int half=0;half<2;++half) {
            // Only four packed rows are live, versus eight in the generic
            // Q8 path. Retire one K16 delivery before loading the next.
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
                    for(int residue=0;residue<4;++residue) {
                        #pragma unroll
                        for(int p=0;p<Pairs;++p) {
                            float2 q=slot==0?code_pair<0>(words[residue][p]):code_pair<1>(words[residue][p]);
                            dot[p].x=fmaf(values[residue],q.x,dot[p].x);
                            dot[p].y=fmaf(values[residue],q.y,dot[p].y);
                        }
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
    if(tid<32) {
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
} // namespace quactlize::execution::simt::q8_vector
