#pragma once
#include "simt_common.cuh"

namespace quactlize::execution::simt {

// Dense Q2--Q6 candidate: one N/K partition owns ALL rows. Keep the canonical
// planes and each row's original group/half/slot/residue FMA order. Only the
// token ownership changes; low/high loads, metadata and code extraction are
// shared across tokens. The public selector does not enable this candidate.
template<int Tokens,int Columns,int Warps,int P,int Q=14>
__global__ void token_cta_reader(qkg_call_v1 call,int split) {
    static_assert(Tokens>=2 && Tokens<=8);
    static_assert(kCommonReader<Q,3,Columns,Warps,P>);
    using F=CommonFormat<Q>;
    constexpr int TileN=Columns*P,Workers=Warps*32/Columns,Pairs=P/2;
    constexpr int Segments=(F::group+F::Low::kLogicalKPerDelivery-1)/F::Low::kLogicalKPerDelivery;
    unsigned const tid=threadIdx.x,lane=tid%32,worker=tid/Columns;
    unsigned const tile=blockIdx.x,partition=blockIdx.z;
    unsigned const col=tile*TileN+(tid%Columns)*P;
    auto const* low=reinterpret_cast<uint16_t const*>(call.low);
    auto const* high=reinterpret_cast<uint16_t const*>(call.high);
    float2 total[Tokens][Pairs]{};

    for(unsigned g=partition*Workers+worker;g<unsigned(call.k)/F::group;g+=split*Workers) {
        // Same all-or-none warp group tail as common_reader. No token lane
        // participates in a different group while full-mask shuffles execute.
        common_metadata::Packet cooperative{};
        if(lane<TileN)
            cooperative=common_metadata::load<Q>(call.units+F::unit_offset(call.n,tile*TileN+lane,g),g);
        float2 metadata[P];
        #pragma unroll
        for(int p=0;p<P;++p)
            metadata[p]=common_metadata::affine<Q>(common_metadata::share(cooperative,(lane%Columns)*P+p),g);
        uint4 packet[Tokens];
        #pragma unroll
        for(int t=0;t<Tokens;++t) {
            Activation<QKG_F32,0> a{static_cast<float const*>(call.a)+int64_t(t)*call.a_row_stride};
            packet[t]=activation_packet<QKG_F32,0,Columns,F::group>(a,g,lane);
        }
        float2 dot[Tokens][Pairs]{};
        float a_sum[Tokens]{};
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
                float values[Tokens][4];
                #pragma unroll
                for(int t=0;t<Tokens;++t) {
                    float2 x=activation_pair<0,Columns,F::group>(packet[t],slot*8+half*4,lane);
                    float2 y=activation_pair<0,Columns,F::group>(packet[t],slot*8+half*4+2,lane);
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
                        for(int t=0;t<Tokens;++t) {
                            dot[t][p].x=fmaf(values[t][residue],code.x,dot[t][p].x);
                            dot[t][p].y=fmaf(values[t][residue],code.y,dot[t][p].y);
                        }
                    }
                }
            }
        }
        #pragma unroll
        for(int t=0;t<Tokens;++t) {
            #pragma unroll
            for(int p=0;p<Pairs;++p) {
                total[t][p].x+=fmaf(metadata[2*p].x,dot[t][p].x,metadata[2*p].y*a_sum[t]);
                total[t][p].y+=fmaf(metadata[2*p+1].x,dot[t][p].y,metadata[2*p+1].y*a_sum[t]);
            }
        }
    }

    __shared__ float partial[Tokens][Warps*TileN];
    #pragma unroll
    for(int t=0;t<Tokens;++t) {
        float values[P];
        #pragma unroll
        for(int p=0;p<Pairs;++p) {values[2*p]=total[t][p].x;values[2*p+1]=total[t][p].y;}
        float value=q4_s1::q4_reduce_scatter_steps<P,Columns>(values,lane);
        if(lane<TileN)partial[t][(tid/32)*TileN+(lane%Columns)*P+lane/Columns]=value;
    }
    __syncthreads();
    if(tid<32) {
        #pragma unroll
        for(int t=0;t<Tokens;++t) {
            float sum=0;
            #pragma unroll
            for(int w=tid/TileN;w<Warps;w+=32/TileN)sum+=partial[t][w*TileN+tid%TileN];
            #pragma unroll
            for(int d=TileN;d<32;d*=2)sum+=__shfl_xor_sync(0xffffffffu,sum,d);
            if(tid<TileN) {
                unsigned const n=tile*TileN+tid;
                if(split==1)call.output[int64_t(t)*call.out_row_stride+n]=sum;
                else static_cast<float*>(call.workspace)[(int64_t(t)*split+partition)*call.n+n]=sum;
            }
        }
    }
}

// Query/buffer validation belongs to the enclosing ABI carrier, just as for
// common_reader_launch. Reject a mismatched token count instead of truncating.
template<int Tokens,int Columns,int Warps,int P,int Q=14>
int token_cta_launch(qkg_simt_call_v2 const& d,int split) {
    auto const& c=d.call;
    if(c.qtype!=Q || c.mode!=QKG_DENSE || c.rows!=Tokens || c.experts!=1 ||
       c.input_type!=QKG_F32 || d.compute_type!=QKG_COMPUTE_F16)return QKG_SHAPE;
    auto stream=static_cast<hggcStream_t>(c.stream);
    if(hggcGetLastError()!=hggcSuccess)return QKG_RUNTIME;
    token_cta_reader<Tokens,Columns,Warps,P,Q><<<dim3(c.n/(Columns*P),1,split),Warps*32,0,stream>>>(c,split);
    if(hggcGetLastError()!=hggcSuccess)return QKG_RUNTIME;
    launch_reduction<Q>(c,split,stream);
    return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME;
}
} // namespace quactlize::execution::simt
