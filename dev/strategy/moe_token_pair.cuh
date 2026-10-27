#pragma once
#include "quactlize/execution/simt_common.cuh"
#include "moe_token_pair.hpp"

namespace quactlize::execution::simt {

// Experiment only: indexed F32 endpoints, direct BF16 A, FP32 affine B and
// accumulation. Exactly the common reader's per-row math/reduction order.
// The pair shares code loads/decode/metadata, never A or different experts' B.
template<int Q,int Columns,int Warps,int P>
__global__ void moe_token_pair_reader(qkg_call_v1 call,int split) {
    static_assert(kCommonReader<Q,3,Columns,Warps,P>);
    using F=CommonFormat<Q>;
    constexpr int TileN=Columns*P,Workers=Warps*32/Columns,Pairs=P/2;
    constexpr int Changes=kBf16F32Changes<Q,3,Columns,Warps,P>;
    constexpr int Segments=(F::group+F::Low::kLogicalKPerDelivery-1)/F::Low::kLogicalKPerDelivery;
    unsigned const tid=threadIdx.x,lane=tid%32,worker=tid/Columns;
    unsigned const tile=blockIdx.x,partition=blockIdx.z;
    unsigned const col=tile*TileN+(tid%Columns)*P;
    int const row=blockIdx.y;
    int const expert=call.ids[int64_t(row/call.topk)*call.ids_stride+row%call.topk];
    // Each warp reads at most 64 IDs coalesced by token, with a ballot per
    // 32 rows. All lanes in all warps obtain the same schedule. No CTA barrier,
    // global routing workspace or host readback is needed. IDs remain dynamic.
    moe_pair::Rows pair{true,-1};
    if(expert>=0 && expert<call.experts) {
        uint64_t matches=0;
        for(int base=0;base<call.rows;base+=32) {
            int const r=base+lane;
            bool same=r<call.rows &&
                call.ids[int64_t(r/call.topk)*call.ids_stride+r%call.topk]==expert;
            matches|=uint64_t(__ballot_sync(0xffffffffu,same))<<base;
        }
        pair=moe_pair::locate(matches,row);
    }
    if(!pair.leader)return;
    // Share the singleton/invalid branch so the common body is inlined once.
    if(pair.peer<0) {
        common_reuse_body<QKG_INDEXED,Q,QKG_F32,3,Columns,Warps,P,1,Changes>(call,split);
        return;
    }
    int const rows[2]={row,pair.peer};
    int64_t const nk=int64_t(call.n)*call.k;
    auto const* low=reinterpret_cast<uint16_t const*>(call.low+expert*nk/8*F::low_bits);
    auto const* high=reinterpret_cast<uint16_t const*>(F::high_bits ? call.high+expert*nk/8*F::high_bits : nullptr);
    auto const* units=call.units+expert*F::metadata_bytes(nk);
    int64_t a_offset[2];
    #pragma unroll
    for(int t=0;t<2;++t)
        a_offset[t]=moe_pair::activation_row(rows[t],call.topk,call.channels,call.a_token_stride,call.a_row_stride);
    float2 total[2][Pairs]{};
    for(unsigned g=partition*Workers+worker;g<unsigned(call.k)/F::group;g+=split*Workers) {
        common_metadata::Packet cooperative{};
        if(lane<TileN)cooperative=common_metadata::load<Q>(units+F::unit_offset(call.n,tile*TileN+lane,g),g);
        float2 metadata[P];
        #pragma unroll
        for(int p=0;p<P;++p)
            metadata[p]=common_metadata::affine<Q>(common_metadata::share(cooperative,(lane%Columns)*P+p),g);
        uint4 packet[2];
        #pragma unroll
        for(int t=0;t<2;++t) {
            Activation<QKG_F32,1> a{static_cast<float const*>(call.a)+a_offset[t]};
            packet[t]=activation_packet<QKG_F32,1,Columns,F::group>(a,g,lane);
        }
        float2 dot[2][Pairs]{};float a_sum[2]{};
        #pragma unroll
        for(int half=0;half<2;++half) {
            uint32_t lo[Segments][4][Pairs],hi[4][Pairs]{};
            #pragma unroll
            for(int segment=0;segment<Segments;++segment) {
                #pragma unroll
                for(int residue=0;residue<4;++residue) {
                    unsigned const k=g*F::group+segment*F::Low::kLogicalKPerDelivery+half*4+residue;
                    load_words<P>(low+F::Low::word_index(col,k,call.n),lo[segment][residue]);
                }
            }
            if constexpr(F::high_bits) {
                #pragma unroll
                for(int residue=0;residue<4;++residue) {
                    unsigned const k=g*F::group+half*4+residue;
                    load_words<P>(high+F::High::word_index(col,k,call.n),hi[residue]);
                }
            }
            #pragma unroll
            for(int slot=0;slot<F::group/8;++slot) {
                float values[2][4];
                #pragma unroll
                for(int t=0;t<2;++t) {
                    float2 x=activation_pair<1,Columns,F::group>(packet[t],slot*8+half*4,lane);
                    float2 y=activation_pair<1,Columns,F::group>(packet[t],slot*8+half*4+2,lane);
                    values[t][0]=x.x;values[t][1]=x.y;values[t][2]=y.x;values[t][3]=y.y;
                    a_sum[t]+=(x.x+x.y)+(y.x+y.y);
                }
                #pragma unroll
                for(int residue=0;residue<4;++residue) {
                    unsigned const k=g*F::group+slot*8+half*4+residue;
                    #pragma unroll
                    for(int p=0;p<Pairs;++p) {
                        float2 code=common_codes<Q>(lo[slot/F::Low::kPack][residue][p],hi[residue][p],col+2*p,k);
                        #pragma unroll
                        for(int t=0;t<2;++t) {
                            dot[t][p].x=fmaf(values[t][residue],code.x,dot[t][p].x);
                            dot[t][p].y=fmaf(values[t][residue],code.y,dot[t][p].y);
                        }
                    }
                }
            }
        }
        #pragma unroll
        for(int t=0;t<2;++t) {
            #pragma unroll
            for(int p=0;p<Pairs;++p) {
                total[t][p].x+=fmaf(metadata[2*p].x,dot[t][p].x,metadata[2*p].y*a_sum[t]);
                total[t][p].y+=fmaf(metadata[2*p+1].x,dot[t][p].y,metadata[2*p+1].y*a_sum[t]);
            }
        }
    }
    __shared__ float partial[2][Warps*TileN];
    #pragma unroll
    for(int t=0;t<2;++t) {
        float values[P];
        #pragma unroll
        for(int p=0;p<Pairs;++p){values[2*p]=total[t][p].x;values[2*p+1]=total[t][p].y;}
        float value=q4_s1::q4_reduce_scatter_steps<P,Columns>(values,lane);
        if(lane<TileN)partial[t][(tid/32)*TileN+(lane%Columns)*P+lane/Columns]=value;
    }
    __syncthreads();
    if(tid<32) {
        #pragma unroll
        for(int t=0;t<2;++t) {
            float sum=0;
            if constexpr(Changes&2)q4_s1::q4_medium_fold<0,Warps,TileN>(sum,partial[t],unsigned(tid));
            else {
                #pragma unroll
                for(int w=tid/TileN;w<Warps;w+=32/TileN)sum+=partial[t][w*TileN+tid%TileN];
            }
            #pragma unroll
            for(int d=TileN;d<32;d*=2)sum+=__shfl_xor_sync(0xffffffffu,sum,d);
            if(tid<TileN) {
                int const n=tile*TileN+tid;
                if(split==1)call.output[int64_t(rows[t])*call.out_row_stride+n]=sum;
                else static_cast<float*>(call.workspace)[(int64_t(rows[t])*split+partition)*call.n+n]=sum;
            }
        }
    }
}

// ABI validation belongs to the generated carrier. M1 deliberately retains
// the old launcher (even if its router has duplicate IDs): no scan overhead.
template<int Q,int C,int W,int P,bool Pair=true>
int moe_token_pair_launch(qkg_simt_call_v2 const& d,int split) {
    auto const& c=d.call;
    auto stream=static_cast<hggcStream_t>(c.stream);
    if(hggcGetLastError()!=hggcSuccess)return QKG_RUNTIME;
    dim3 grid(c.n/(C*P),c.rows,split);
    if(Pair && c.rows!=c.topk)
        moe_token_pair_reader<Q,C,W,P><<<grid,W*32,0,stream>>>(c,split);
    else common_reader<QKG_INDEXED,Q,C,W,P,1><<<grid,W*32,0,stream>>>(c,split);
    if(hggcGetLastError()!=hggcSuccess)return QKG_RUNTIME;
    launch_reduction<Q>(c,split,stream);
    return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME;
}
} // namespace quactlize::execution::simt
