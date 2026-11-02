#pragma once
#include "../execution/simt_q8_vector.cuh"
#include "paired_n4.hpp"

// Canonical gate/up reader. Ordinary measured GEMV bodies remain unchanged.
namespace quactlize::fusion::canonical {
using namespace execution::simt;
namespace q4_s1 = execution::q4_s1;

template<class Map>
__device__ __forceinline__ int weight_column(int col,int n) {
    if constexpr(std::is_void_v<Map>) return col;
    else return Map::column(col,n);
}

template<int P,class Map,class Plane>
__device__ __forceinline__ void load_columns(uint16_t const* ptr,int col,int k,int n,uint32_t (&out)[P/2]) {
    if constexpr(std::is_void_v<Map>) load_words<P>(ptr+Plane::word_index(col,k,n),out);
    else {
        static_assert(P==4 || P==8);
        #pragma unroll
        for(int i=0;i<P;i+=4) {
            uint32_t words[2];load_words<4>(ptr+Plane::word_index(Map::column(col+i,n),k,n),words);
            out[i/2]=words[0];out[i/2+1]=words[1];
        }
    }
}

template<int Q,int Input,int Variant,int Columns,int Warps,int P,int Compute=0,int Changes=0,
         class Finish=void,class Call=qkg_call_v1,class WeightMap=void>
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
            if (lane<TileN) cooperative=load_meta<Q>(units,call.n,weight_column<WeightMap>(tile*TileN+lane,call.n),g);
        }
        float2 metadata[P];
        #pragma unroll
        for (int p=0;p<P;++p) {
            if constexpr(Variant&2)
                metadata[p]=affine_selected<Q,Changes>(share_meta(cooperative,(lane%Columns)*P+p),g);
            else metadata[p]=affine_selected<Q,Changes>(load_meta<Q>(units,call.n,weight_column<WeightMap>(col+p,call.n),g),g);
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
                    load_columns<P,WeightMap,typename F::Low>(low,col,k,call.n,lo[segment][residue]);
                }
            }
            if constexpr(F::high_bits) {
                #pragma unroll
                for (int residue=0;residue<4;++residue) {
                    int k=g*F::group+half*4+residue;
                    load_columns<P,WeightMap,typename F::High>(high,col,k,call.n,hi[residue]);
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
                        float2 code=codes<Q>(lo[slot/F::Low::kPack][residue][p],hi[residue][p],weight_column<WeightMap>(col+2*p,call.n),k);
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


namespace q8 {
using namespace execution::simt::q8_vector;

template<int Input,int Compute,int Variant,int Columns,int Warps,int P,bool Hoist=false,
         class Finish=void,class Call=qkg_call_v1,class WeightMap=void>
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
        float d[P];
        if constexpr(std::is_void_v<WeightMap>) scales<P>(units+2*(uint64_t(g)*c.n+col),d);
        else {
            #pragma unroll
            for(int p=0;p<P;p+=4) {
                float part[4];scales<4>(units+2*(uint64_t(g)*c.n+WeightMap::column(col+p,c.n)),part);
                #pragma unroll
                for(int j=0;j<4;++j) d[p+j]=part[j];
            }
        }
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
                    for(int residue=0;residue<4;++residue) {
                        if constexpr(std::is_void_v<WeightMap>)
                            load_words<P>(low+(uint64_t(g)*16+segment*8+half*4+residue)*c.n+col,words[half][segment][residue]);
                        else load_columns<P,WeightMap,typename Format<8>::Low>(low,col,g*32+segment*16+half*4+residue,c.n,words[half][segment][residue]);
                    }
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
                    for(int residue=0;residue<4;++residue) {
                        if constexpr(std::is_void_v<WeightMap>)
                            load_words<P>(low+(uint64_t(g)*16+segment*8+half*4+residue)*c.n+col,words[residue]);
                        else load_columns<P,WeightMap,typename Format<8>::Low>(low,col,g*32+segment*16+half*4+residue,c.n,words[residue]);
                    }
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
}
}
