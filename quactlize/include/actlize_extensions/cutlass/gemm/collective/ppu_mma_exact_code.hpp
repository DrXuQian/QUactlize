#pragma once
// Optional standalone actlize extension, NOT a replacement of CollectiveMma's
// mixed-input specializations. Include explicitly, like the fold/two-plane extensions.
//
// A is rounded once to ElementCompute. Integer codes enter MMA exactly; dot
// products and group (scale * dot + minimum * sum(A)) stay FP32. Do not substitute
// this for per-weight F16/BF16 reconstruction without numerical admission.
//
// PPU m8n16k16: native A = 8 weight columns, native B = 16 activation rows.
// Rows 0..7 rotate quantization groups over tokens; rows 8..15 are zero.
// Group masking keeps cross-group products out of each group's accumulator.
// The canonical low/high/metadata planes and FP32 Split-K layout do not change.
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_exact_code_reader.hpp"

namespace cutlass::gemm::collective {

struct ExactCodeTileArgumentsPPU {
    exact_code_ppu_detail::Planes planes; // one expert, original global N pitch
    float const* a;                       // F32, offset to the token/expert row
    int64_t a_row_stride;                 // elements, NOT bytes
    int tokens, n_begin, sb_begin, sb_end; // half-open 256-K superblock interval
};

template<int Q,class ElementCompute,int T,int WN,int WK,int Tokens,class Delivery>
struct ExactCodeCollectivePPU;

template<int Q,class ElementCompute_,int T,int WN,int WK,int Tokens>
struct ExactCodeCollectiveBasePPU : exact_code_ppu_detail::Geometry<Q,T,WN,WK,Tokens> {
    using Geometry=exact_code_ppu_detail::Geometry<Q,T,WN,WK,Tokens>;
    using ElementCompute=ElementCompute_;
    using ElementAccumulator=float;
    using ArchTag=cutlass::arch::PPU0010;
    using Arguments=ExactCodeTileArgumentsPPU;
    static_assert(std::is_same_v<ElementCompute,cutlass::half_t> ||
                  std::is_same_v<ElementCompute,cutlass::bfloat16_t>);
    static constexpr int Compute=std::is_same_v<ElementCompute,cutlass::half_t> ? 0 : 1;
    static_assert(Q!=8 || Compute==0, "Q8 exact-code reader currently implements F16 only");
    static constexpr int QType=Q, TokenCapacity=Tokens;
    using MmaOperation=exact_code_ppu_detail::MmaOperation<Compute>;
    using MmaAtom=cute::MMA_Atom<MmaOperation>;
    using MmaTraits=cute::MMA_Traits<MmaOperation>;
    static_assert(cute::is_same_v<typename MmaTraits::Shape_MNK,cute::Shape<cute::_8,cute::_16,cute::_16>>);
    static_assert(exact_code_ppu_detail::native_mapping_matches<MmaTraits>(),
                  "exact-code delivery no longer matches actlize's MMA lane/register layout");
};

template<int Q,class ElementCompute,int T,int WN,int WK,int Tokens,int NB>
struct ExactCodeCollectivePPU<Q,ElementCompute,T,WN,WK,Tokens,ExactCodeRegisterDirectPPU<NB>>
    : ExactCodeCollectiveBasePPU<Q,ElementCompute,T,WN,WK,Tokens> {
    using Base=ExactCodeCollectiveBasePPU<Q,ElementCompute,T,WN,WK,Tokens>;
    using Arguments=typename Base::Arguments;
    using DispatchPolicy=ExactCodeRegisterDirectPPU<NB>;
    static constexpr int Compute=Base::Compute, PartitionSuperblocks=1;
    static constexpr int GranulesPerSuperblock=8/exact_code_ppu_detail::Kind<Q>::BPG;
    static_assert((NB==2 || NB==4 || NB==8) && GranulesPerSuperblock%NB==0);
    static size_t shared_bytes(int k_len) { return Base::direct_shared_bytes(k_len); }

    // One whole CTA, all lanes participate; aligned canonical planes, N tile and
    // K partition are caller-validated. Store(token, global_n, value) publishes
    // final F32 values or the caller's row/split-major partials. No launch,
    // allocation, host synchronization, routing-ID readback or JIT here.
    template<class Store>
    __device__ __forceinline__ static void run(Arguments args,void* storage,Store store) {
        using namespace exact_code_ppu_detail;
        using K=Kind<Q>;
        using M=typename MetaOf<Q>::type;
        using R=Rot<Tokens>;
        constexpr int TileN=8*T,CtaN=WN*TileN,TP=R::TP,WG=R::WG,V=R::V;
        constexpr int CPG=K::GK/16,GPS=256/K::GK,Windows=GPS/WG,BPG=K::BPG,GPSB=8/BPG;
        uint32_t* smem=static_cast<uint32_t*>(storage);
        unsigned const tid=threadIdx.x,lane=tid%32,warp=tid/32,m=lane/4,t=lane%4;
        unsigned const wn=warp%WN,wk=warp/WN;
        int const n_cta=args.n_begin,n_w=n_cta+wn*TileN,col=n_w+T*m;
        int const tokens=args.tokens,sb_begin=args.sb_begin,sb_end=args.sb_end;
        int const k_begin=sb_begin*256,k_len=(sb_end-sb_begin)*256;
        Planes const& planes=args.planes;
        typename K::Lane const ls(col);
        float const* a=args.a;
        int const words=k_len/2;
        uint32_t* act=smem;
        float* sums=reinterpret_cast<float*>(smem+TP*words);
        int const local=sb_end-sb_begin;
        int const w_begin=sb_begin+local*int(wk)/WK,w_end=sb_begin+local*(int(wk)+1)/WK;
        int const gran_end=w_end*GPSB;
        int const g_lane=m%WG,tau_lane=m/WG;
        bool const supply=tau_lane<tokens;
        typename K::template Gran<T> buf[NB];
        // The first granules are in flight while activations are staged or loaded.
        static_for<0,NB-1>([&](auto pc) {
            constexpr int d=decltype(pc)::value;
            if(w_begin*GPSB+d<gran_end) K::load(buf[d],planes,col,w_begin*GPSB+d);
        });
        // Direct mode: this lane's activation pairs for (token tau_lane, its window group).
        float2 pre[2*CPG];
        auto load_act=[&](int group_abs) {
            float const* src=a+int64_t(tau_lane)*args.a_row_stride+group_abs*K::GK+2*t;
            #pragma unroll
            for(int s=0;s<2*CPG;++s)
                pre[s]=supply ? *reinterpret_cast<float2 const*>(src+8*s) : make_float2(0.f,0.f);
        };
        if constexpr(R::Direct) {
            if(w_begin<w_end) load_act(w_begin*GPS+g_lane);
        } else {
            stage<Compute,K::SumK>(act,sums,a,args.a_row_stride,tokens,k_begin,k_len);
            __syncthreads();
        }

        float acc[T][V];
        #pragma unroll
        for(int i=0;i<T;++i)
            #pragma unroll
            for(int v=0;v<V;++v) acc[i][v]=0.f;
        for(int sb=w_begin;sb<w_end;++sb) {
            typename SbOf<Q,T>::type sbm;
            M::load_sb(sbm,planes,col,sb);
            static_for<0,Windows>([&](auto wc) {
                constexpr int W=decltype(wc)::value,G0=W*WG;
                int const rel=(sb-sb_begin)*GPS+G0;   // partition-relative group of the window start
                typename M::template Win<T> win;
                M::load_win(win,planes,col,sb*GPS+G0+R::group(t,0),sb*GPS+G0+R::group(t,1));
                uint32_t bw[2*CPG];
                float asum[2]={0.f,0.f};
                if constexpr(R::Direct) {
                    float part=0.f;
                    #pragma unroll
                    for(int s=0;s<2*CPG;++s) {
                        bw[s]=round_pair<Compute>(pre[s].x,pre[s].y);
                        if constexpr(K::SumK>0) part+=pair_sum<Compute>(bw[s]);
                    }
                    // Prefetch the next window's pairs (also across superblocks).
                    int const next=W+1<Windows ? sb*GPS+G0+WG+g_lane : (sb+1)*GPS+g_lane;
                    if(W+1<Windows || sb+1<w_end) load_act(next);
                    if constexpr(K::SumK>0) {
                        // Sum over the four lanes of B row m, then fetch the rows this lane owns in C.
                        part+=__shfl_xor_sync(0xffffffffu,part,1);
                        part+=__shfl_xor_sync(0xffffffffu,part,2);
                        asum[0]=__shfl_sync(0xffffffffu,part,4*int(t));
                        asum[1]=__shfl_sync(0xffffffffu,part,4*int(t)+16);
                    }
                } else {
                    int const kk=(rel+g_lane)*K::GK;
                    uint32_t const* src=act+tau_lane*words+(kk/32)*16+t*4+((kk/16)%2)*2;
                    if constexpr(CPG==2) {
                        uint4 v=supply ? *reinterpret_cast<uint4 const*>(src) : make_uint4(0,0,0,0);
                        bw[0]=v.x;bw[1]=v.y;bw[2]=v.z;bw[3]=v.w;
                    } else {
                        uint2 v=supply ? *reinterpret_cast<uint2 const*>(src) : make_uint2(0,0);
                        bw[0]=v.x;bw[1]=v.y;
                    }
                    if constexpr(K::SumK>0) {
                        #pragma unroll
                        for(int v=0;v<2;++v) {
                            int const tok=R::token(t,v);
                            if(tok<tokens) asum[v]=sums[tok*(k_len/K::SumK)+rel+R::group(t,v)];
                        }
                    }
                }
                float c[T][4];
                #pragma unroll
                for(int i=0;i<T;++i) c[i][0]=c[i][1]=c[i][2]=c[i][3]=0.f;
                static_for<0,WG>([&](auto gc) {
                    constexpr int g=decltype(gc)::value;
                    bool const mine=g_lane==g;
                    static_for<0,CPG>([&](auto jc) {
                        constexpr int jj=decltype(jc)::value,ch=(G0+g)*CPG+jj,blk=ch/2,j=ch%2;
                        constexpr int gs=blk/BPG,bg=blk%BPG;
                        if constexpr(j==0 && bg==0) {
                            // Entering granule gs: prefetch NB-1 granules ahead (also across superblocks).
                            int const next=sb*GPSB+gs+NB-1;
                            if(next<gran_end) K::load(buf[(gs+NB-1)%NB],planes,col,next);
                        }
                        uint32_t const b0=mine ? bw[2*jj] : 0u,b1=mine ? bw[2*jj+1] : 0u;
                        #pragma unroll
                        for(int i=0;i<T;++i) {
                            uint2 const av=K::template a<Compute,T,bg,j>(buf[gs%NB],ls,i);
                            mma<Compute>(c[i],av.x,av.y,b0,b1);
                        }
                    });
                });
                static_for<0,2>([&](auto vc) {
                    constexpr int v=decltype(vc)::value;
                    int const tok=R::token(t,v),g=R::group(t,v);
                    if(tok<tokens) {
                        #pragma unroll
                        for(int i=0;i<T;++i) {
                            float2 const af=M::template affine<T,v>(sbm,win,i,unsigned(G0+g));
                            acc[i][V==1 ? 0 : v]+=fmaf(af.x,c[i][v],af.y*asum[v]);
                        }
                    }
                });
            });
        }
        // Fold lanes that hold the same (column, token), then k-slice warps in fixed order.
        #pragma unroll
        for(int i=0;i<T;++i)
            #pragma unroll
            for(int v=0;v<V;++v) {
                float x=acc[i][v];
                if constexpr(TP<=2) { x+=__shfl_xor_sync(0xffffffffu,x,1); x+=__shfl_xor_sync(0xffffffffu,x,2); }
                else if constexpr(TP==4) x+=__shfl_xor_sync(0xffffffffu,x,1);
                acc[i][v]=x;
            }
        bool const owner=TP<=2 ? t==0 : TP==4 ? (t&1)==0 : true;
        auto owner_token=[&](int v) { return TP==1 ? 0 : TP==2 ? v : TP==4 ? int(t>>1)+2*v : int(t)+4*v; };
        if constexpr(WK==1) {
            if(owner) {
                #pragma unroll
                for(int v=0;v<V;++v) {
                    int const tok=owner_token(v);
                    if(tok<tokens) {
                        #pragma unroll
                        for(int i=0;i<T;++i) store(tok,col+i,acc[i][v]);
                    }
                }
            }
        } else {
            if constexpr(!R::Direct) __syncthreads();
            float* partial=reinterpret_cast<float*>(smem);
            if(owner) {
                #pragma unroll
                for(int v=0;v<V;++v) {
                    int const tok=owner_token(v);
                    if(tok<tokens) {
                        #pragma unroll
                        for(int i=0;i<T;++i) partial[(wk*TP+tok)*CtaN+wn*TileN+T*m+i]=acc[i][v];
                    }
                }
            }
            __syncthreads();
            for(int idx=tid;idx<tokens*CtaN;idx+=WN*WK*32) {
                int const tok=idx/CtaN,cn=idx%CtaN;
                float sum=0.f;
                #pragma unroll
                for(int w=0;w<WK;++w) sum+=partial[(w*TP+tok)*CtaN+cn];
                store(tok,n_cta+cn,sum);
            }
        }
    }
};
} // namespace cutlass::gemm::collective
