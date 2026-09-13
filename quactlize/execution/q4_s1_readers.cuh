#pragma once
#include "q4_s1_helpers.cuh"

namespace quactlize::execution::q4_s1 {
template<int Input,int Header,int Loading,int AMode,int Warps,int N,int K>
__device__ __forceinline__ void row_meta(int tile,Activation<Input> a_ptr,uint8_t const* low_ptr,uint8_t const* units_ptr,float* output) {
    constexpr unsigned Width=8;
    unsigned tid=threadIdx.x,lane=tid%32,warp=tid/32,residue=lane%8;
    unsigned first=tile*Width;
    auto low=reinterpret_cast<uint16_t const*>(low_ptr);
    auto a=a_ptr;
    static_assert(AMode==0 || AMode==1);

    float2 sums[4]{};
    #pragma unroll
    for(unsigned pass=0;pass<(K/128+Warps-1)/Warps;++pass) {
        unsigned chunk=pass*Warps+warp;
        if(chunk>=K/128) continue;
        unsigned g=chunk*4+lane/8;
        uint4 unit=aligned_unit(units_ptr+(size_t(g/8)*N+first+residue)*16);
        uint4 b,ab;
        ScaleZero sz;
        if constexpr(Loading==0) sz=latency_half_header<Header>(unit,g&7);
        b=*reinterpret_cast<uint4 const*>(low+size_t(g*8+residue)*N+first);
        ab=latency_residue_a<AMode>(a,g,residue);
        if constexpr(Loading==1) {
            latency_join(unit,b,ab);
            sz=latency_half_header<Header>(unit,g&7);
        }
        uint32_t meta=uint32_t(__half_as_ushort(sz.scale))|(uint32_t(__half_as_ushort(sz.zero))<<16);
        float4 av=latency_residue_values<AMode>(ab,residue);
        float act[4]={av.x,av.y,av.z,av.w};
        uint32_t words[4]={b.x,b.y,b.z,b.w};
        #pragma unroll
        for(int p=0;p<4;++p) {
            uint32_t s0=__shfl_sync(0xffffffffu,meta,(lane&~7)+2*p);
            uint32_t s1=__shfl_sync(0xffffffffu,meta,(lane&~7)+2*p+1);
            __half2 scale=__halves2half2(__ushort_as_half(uint16_t(s0)),__ushort_as_half(uint16_t(s1)));
            __half2 zero=__halves2half2(__ushort_as_half(uint16_t(s0>>16)),__ushort_as_half(uint16_t(s1>>16)));
            #pragma unroll
            for(int slot=0;slot<4;++slot) {
                __half2 q;
                if(slot==0) q=codes<0>(words[p]);
                else if(slot==1) q=codes<1>(words[p]);
                else if(slot==2) q=codes<2>(words[p]);
                else q=codes<3>(words[p]);
                float2 w=__half22float2(__hfma2(q,scale,zero));
                sums[p].x=fmaf(act[slot],w.x,sums[p].x);
                sums[p].y=fmaf(act[slot],w.y,sums[p].y);
            }
        }
    }
    float values[Width];
    #pragma unroll
    for(int p=0;p<4;++p) {values[2*p]=sums[p].x;values[2*p+1]=sums[p].y;}
    float value=q4_reduce_scatter_steps<Width,1>(values,lane);
    __shared__ float partial[Warps*Width];
    if(lane<Width) partial[warp*Width+lane]=value;
    __syncthreads();
    if(tid<32) {
        float sum=0;
        #pragma unroll
        for(unsigned w=tid/Width;w<Warps;w+=32/Width) sum+=partial[w*Width+tid%Width];
        #pragma unroll
        for(int d=Width;d<32;d*=2) sum+=__shfl_xor_sync(0xffffffffu,sum,d);
        if(tid<Width) output[first+tid]=sum;
    }
}

template<int Input,int Reduction,int Unsigned,int Header,int Loading,int AMode,int Columns,int Warps,int P,int N,int K>
__device__ __forceinline__ void row_medium(int tile,Activation<Input> a_ptr,uint8_t const* low_ptr,uint8_t const* units_ptr,float* out_ptr) {
    constexpr int Variant=4;
    constexpr bool Early=true;
    constexpr int Workers=Warps*32/Columns,Pairs=P/2;
    static_assert(((Columns==4 || Columns==8) || (Variant&3)==0) && Workers%8==0 && K%256==0);
    int const read_lane=threadIdx.x%32;
    using Index=typename std::conditional<Unsigned!=0,unsigned,int>::type;
    Index const tid=threadIdx.x,worker=tid/Columns,col=tile*(Columns*P)+(tid%Columns)*P;
    auto low=reinterpret_cast<uint16_t const*>(low_ptr);
    auto act=a_ptr;
    static_assert(AMode==0 || AMode==1);

    float2 total[Pairs]{};
    #pragma unroll
    for(Index pass=0;pass<(K/32+Workers-1)/Workers;++pass) {
        Index const g=pass*Workers+worker;
        if(g>=K/32) continue;
        uint4 metadata[Early ? P : 1];
        if constexpr(Variant&2) {
            uint4 unit{};
            if(read_lane<Columns*P) unit=aligned_unit(units_ptr+(size_t(g/8)*N+tile*(Columns*P)+read_lane)*16);
            #pragma unroll
            for(int p=0;p<P;++p) metadata[p]=q4_cooperative_unit_read(unit,(read_lane%Columns)*P+p);
        } else {
            #pragma unroll
            for(int p=0;p<P;++p) metadata[p]=aligned_unit(units_ptr+(size_t(g/8)*N+col+p)*16);
        }
        uint4 a_chunk{};
        if constexpr(Variant&1) a_chunk=q4_cooperative_a_chunk<Columns>(a_ptr,g,read_lane);
        uint32_t all_words[8][Pairs];
        float4 all_a[2][4];
        if constexpr(Loading==1) {
            #pragma unroll
            for(int r=0;r<8;++r) {
                auto ptr=low+size_t(g*8+r)*N+col;
                if constexpr(P==2) all_words[r][0]=*reinterpret_cast<uint32_t const*>(ptr);
                else {uint2 b=*reinterpret_cast<uint2 const*>(ptr);all_words[r][0]=b.x;all_words[r][1]=b.y;}
            }
            #pragma unroll
            for(int slot=0;slot<4;++slot) {
                if constexpr(AMode==1) {
                    uint4 raw=act.load8(g*32+slot*8);
                    uint32_t regs[4]={raw.x,raw.y,raw.z,raw.w};
                    #pragma unroll
                    for(int h=0;h<2;++h) {
                        float2 lo=__half22float2(__halves2half2(__ushort_as_half(uint16_t(regs[2*h])),__ushort_as_half(uint16_t(regs[2*h]>>16))));
                        float2 hi=__half22float2(__halves2half2(__ushort_as_half(uint16_t(regs[2*h+1])),__ushort_as_half(uint16_t(regs[2*h+1]>>16))));
                        all_a[h][slot]=make_float4(lo.x,lo.y,hi.x,hi.y);
                    }
                } else {
                    #pragma unroll
                    for(int h=0;h<2;++h) all_a[h][slot]=aligned_activation<0>(act,g*32+slot*8+h*4);
                }
            }
            // A compiler-only issue boundary. It does not synchronize threads.
            #pragma unroll
            for(int p=0;p<P;++p) asm volatile("" : "+r"(metadata[p].x),"+r"(metadata[p].y),"+r"(metadata[p].z),"+r"(metadata[p].w) : : "memory");
            #pragma unroll
            for(int r=0;r<8;++r) {
                #pragma unroll
                for(int p=0;p<Pairs;++p) asm volatile("" : "+r"(all_words[r][p]) : : "memory");
            }
            #pragma unroll
            for(int h=0;h<2;++h) {
                #pragma unroll
                for(int slot=0;slot<4;++slot) asm volatile("" : "+f"(all_a[h][slot].x),"+f"(all_a[h][slot].y),"+f"(all_a[h][slot].z),"+f"(all_a[h][slot].w) : : "memory");
            }
        }
        float2 dot[Pairs]{};
        float a_sum=0;
        #pragma unroll
        for(int half=0;half<2;++half) {
            uint32_t words[4][Pairs];
            #pragma unroll
            for(int r=0;r<4;++r) {
                if constexpr(Loading==1) {
                    #pragma unroll
                    for(int p=0;p<Pairs;++p) words[r][p]=all_words[half*4+r][p];
                } else {
                auto ptr=low+size_t(g*8+half*4+r)*N+col;
                if constexpr(P==2) words[r][0]=*reinterpret_cast<uint32_t const*>(ptr);
                else if constexpr(P==4) {
                    uint2 v=*reinterpret_cast<uint2 const*>(ptr);words[r][0]=v.x;words[r][1]=v.y;
                } else {
                    uint4 v=*reinterpret_cast<uint4 const*>(ptr);
                    words[r][0]=v.x;words[r][1]=v.y;words[r][2]=v.z;words[r][3]=v.w;
                }
                }
            }
            #pragma unroll
            for(int slot=0;slot<4;++slot) {
                float4 av;
                if constexpr(Variant&1) av=q4_cooperative_a_read<Columns>(a_chunk,slot*8+half*4,read_lane);
                else if constexpr(Loading==1) av=all_a[half][slot];
                else av=aligned_activation<0>(act,g*32+slot*8+half*4);
                float ax[4]={av.x,av.y,av.z,av.w};
                a_sum+=(av.x+av.y)+(av.z+av.w);
                #pragma unroll
                for(int r=0;r<4;++r) {
                    #pragma unroll
                    for(int p=0;p<Pairs;++p) {
                        __half2 q;
                        if(slot==0) q=codes<0>(words[r][p]);
                        else if(slot==1) q=codes<1>(words[r][p]);
                        else if(slot==2) q=codes<2>(words[r][p]);
                        else q=codes<3>(words[r][p]);
                        float2 v=__half22float2(q);
                        dot[p].x=fmaf(ax[r],v.x,dot[p].x);
                        dot[p].y=fmaf(ax[r],v.y,dot[p].y);
                    }
                }
            }
        }
        #pragma unroll
        for(int p=0;p<Pairs;++p) {
            uint4 u0,u1;
            if constexpr(Early) {u0=metadata[2*p];u1=metadata[2*p+1];}
            else {
                u0=aligned_unit(units_ptr+(size_t(g/8)*N+col+2*p)*16);
                u1=aligned_unit(units_ptr+(size_t(g/8)*N+col+2*p+1)*16);
            }
            float2 s0,s1;
            if constexpr(Variant&4) {s0=latency_affine_header<Header>(u0,g&7);s1=latency_affine_header<Header>(u1,g&7);}
            else {s0=q4_affine_header(u0,g&7);s1=q4_affine_header(u1,g&7);}
            total[p].x+=fmaf(s0.x,dot[p].x,s0.y*a_sum);
            total[p].y+=fmaf(s1.x,dot[p].y,s1.y*a_sum);
        }
    }
    constexpr int TileN=Columns*P;
    static_assert(TileN<=32);
    float values[P];
    #pragma unroll
    for(int p=0;p<Pairs;++p) {values[2*p]=total[p].x;values[2*p+1]=total[p].y;}
    int const lane=tid%32;
    float value=q4_reduce_scatter_steps<P,Columns>(values,lane);
    __shared__ float partial[Warps*TileN];
    if(lane<TileN) partial[(tid/32)*TileN+(lane%Columns)*P+lane/Columns]=value;
    __syncthreads();
    if(tid<32) {
        float sum=0;
        if constexpr(Reduction==0) {
        #pragma unroll
        for(int w=tid/TileN;w<Warps;w+=32/TileN) sum+=partial[w*TileN+tid%TileN];
        } else {
            q4_medium_fold<0,Warps,TileN>(sum,partial,unsigned(tid));
        }
        #pragma unroll
        for(int d=TileN;d<32;d*=2) sum+=__shfl_xor_sync(0xffffffffu,sum,d);
        if(tid<TileN) out_ptr[tile*TileN+tid]=sum;
    }
}

template<int Input,int Variant,int Columns,int Warps,int P,int N,int K>
__device__ __forceinline__ void row_reuse(int tile,Activation<Input> a_ptr,uint8_t const* low_ptr,uint8_t const* units_ptr,float* out_ptr) {
    constexpr bool Early=true;
    constexpr int Workers=Warps*32/Columns,Pairs=P/2;
    static_assert(((Columns==4 || Columns==8) || (Variant&3)==0) && Workers%8==0 && K%256==0);
    int const read_lane=threadIdx.x%32;
    int const tid=threadIdx.x,worker=tid/Columns,col=tile*(Columns*P)+(tid%Columns)*P;
    auto low=reinterpret_cast<uint16_t const*>(low_ptr);
    float2 total[Pairs]{};
    #pragma unroll
    for(int pass=0;pass<(K/32+Workers-1)/Workers;++pass) {
        int const g=pass*Workers+worker;
        if(g>=K/32) continue;
        uint4 metadata[Early ? P : 1];
        if constexpr(Variant&2) {
            uint4 unit{};
            if(read_lane<Columns*P) unit=aligned_unit(units_ptr+(size_t(g/8)*N+tile*(Columns*P)+read_lane)*16);
            #pragma unroll
            for(int p=0;p<P;++p) metadata[p]=q4_cooperative_unit_read(unit,(read_lane%Columns)*P+p);
        } else {
            #pragma unroll
            for(int p=0;p<P;++p) metadata[p]=aligned_unit(units_ptr+(size_t(g/8)*N+col+p)*16);
        }
        uint4 a_chunk{};
        if constexpr(Variant&1) a_chunk=q4_cooperative_a_chunk<Columns>(a_ptr,g,read_lane);
        float2 dot[Pairs]{};
        float a_sum=0;
        #pragma unroll
        for(int half=0;half<2;++half) {
            uint32_t words[4][Pairs];
            #pragma unroll
            for(int r=0;r<4;++r) {
                auto ptr=low+size_t(g*8+half*4+r)*N+col;
                if constexpr(P==2) words[r][0]=*reinterpret_cast<uint32_t const*>(ptr);
                else if constexpr(P==4) {
                    uint2 v=*reinterpret_cast<uint2 const*>(ptr);words[r][0]=v.x;words[r][1]=v.y;
                } else {
                    uint4 v=*reinterpret_cast<uint4 const*>(ptr);
                    words[r][0]=v.x;words[r][1]=v.y;words[r][2]=v.z;words[r][3]=v.w;
                }
            }
            #pragma unroll
            for(int slot=0;slot<4;++slot) {
                float4 av;
                if constexpr(Variant&1) av=q4_cooperative_a_read<Columns>(a_chunk,slot*8+half*4,read_lane);
                else av=aligned_activation<0>(a_ptr,g*32+slot*8+half*4);
                float ax[4]={av.x,av.y,av.z,av.w};
                a_sum+=(av.x+av.y)+(av.z+av.w);
                #pragma unroll
                for(int r=0;r<4;++r) {
                    #pragma unroll
                    for(int p=0;p<Pairs;++p) {
                        __half2 q;
                        if(slot==0) q=codes<0>(words[r][p]);
                        else if(slot==1) q=codes<1>(words[r][p]);
                        else if(slot==2) q=codes<2>(words[r][p]);
                        else q=codes<3>(words[r][p]);
                        float2 v=__half22float2(q);
                        dot[p].x=fmaf(ax[r],v.x,dot[p].x);
                        dot[p].y=fmaf(ax[r],v.y,dot[p].y);
                    }
                }
            }
        }
        #pragma unroll
        for(int p=0;p<Pairs;++p) {
            uint4 u0,u1;
            if constexpr(Early) {u0=metadata[2*p];u1=metadata[2*p+1];}
            else {
                u0=aligned_unit(units_ptr+(size_t(g/8)*N+col+2*p)*16);
                u1=aligned_unit(units_ptr+(size_t(g/8)*N+col+2*p+1)*16);
            }
            float2 s0,s1;
            if constexpr(Variant&4) {s0=q4_affine_header32(u0,g&7);s1=q4_affine_header32(u1,g&7);}
            else {s0=q4_affine_header(u0,g&7);s1=q4_affine_header(u1,g&7);}
            total[p].x+=fmaf(s0.x,dot[p].x,s0.y*a_sum);
            total[p].y+=fmaf(s1.x,dot[p].y,s1.y*a_sum);
        }
    }
    constexpr int TileN=Columns*P;
    static_assert(TileN<=32);
    float values[P];
    #pragma unroll
    for(int p=0;p<Pairs;++p) {values[2*p]=total[p].x;values[2*p+1]=total[p].y;}
    int const lane=tid%32;
    float value=q4_reduce_scatter_steps<P,Columns>(values,lane);
    __shared__ float partial[Warps*TileN];
    if(lane<TileN) partial[(tid/32)*TileN+(lane%Columns)*P+lane/Columns]=value;
    __syncthreads();
    if(tid<32) {
        float sum=0;
        #pragma unroll
        for(int w=tid/TileN;w<Warps;w+=32/TileN) sum+=partial[w*TileN+tid%TileN];
        #pragma unroll
        for(int d=TileN;d<32;d*=2) sum+=__shfl_xor_sync(0xffffffffu,sum,d);
        if(tid<TileN) out_ptr[tile*TileN+tid]=sum;
    }
}

}
