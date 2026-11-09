#pragma once
// Optional row-contiguous cp.async delivery for the same exact-code collective.
// Preserve the verified stage lifetime: wait for writes before consumption AND
// CTA reader retirement before a ring slot is overwritten. Shared rows remain
// compressed codes, not reconstructed F16/BF16 weight tiles.
#include "actlize_extensions/cutlass/gemm/collective/ppu_mma_exact_code.hpp"

namespace cutlass::gemm::collective {
namespace exact_code_ppu_detail {
__device__ __forceinline__ void cp_async16(void* smem,void const* gmem) {
    uint32_t const s=uint32_t(__cvta_generic_to_shared(smem));
    asm volatile("ppu.cp.async.cg.shared.global.LLC::128B [%0], [%1], 16;\n" :: "r"(s),"l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("ppu.cp.async.commit_group;\n" ::: "memory"); }
template<int N>
__device__ __forceinline__ void cp_async_wait() {
    if constexpr(N==0) asm volatile("ppu.cp.async.wait_all;\n" ::: "memory");
    else asm volatile("ppu.cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

// Issue the copies of one stage (superblocks [sb0, sb0+count), count <= SB) with every thread.
template<int Q,int CtaN,int SB,int Threads>
__device__ __forceinline__ void copy_stage(uint8_t* st,Planes const& g,int n_cta,int sb0,int count,int tid) {
    using L=StageLayout<Q,CtaN,SB>;
    constexpr int CPR=CtaN/8;                              // 16-B chunks per staged row
    auto rows=[&](int off,uint16_t const* base,int row0,int nrows) {
        int const total=nrows*CPR;
        for(int c=tid;c<total;c+=Threads) {
            int const r=c/CPR,q=c%CPR;
            cp_async16(st+off+r*L::Pitch+q*16,base+uint64_t(row0+r)*g.n+n_cta+q*8);
        }
    };
    rows(L::LowOff,g.low,sb0*L::LowPerSb,count*L::LowPerSb);
    if constexpr(L::HighPerSb>0) rows(L::HighOff,g.high,sb0*L::HighPerSb,count*L::HighPerSb);
    if constexpr(L::MetaPerSb>0)
        rows(L::MetaOff,reinterpret_cast<uint16_t const*>(g.units),sb0*L::MetaPerSb,count*L::MetaPerSb);
    if constexpr(Q==12 || Q==13) {
        int const total=count*CtaN;                        // units [sb][n] x 16 B
        for(int c=tid;c<total;c+=Threads) {
            int const j=c/CtaN,q=c%CtaN;
            cp_async16(st+L::UnitOff+(j*CtaN+q)*16,g.units+(uint64_t(sb0+j)*g.n+n_cta+q)*16);
        }
    } else if constexpr(Q==14) {
        constexpr int CPP=CtaN*36/16;                      // units [sb/2][n] x 36 B
        int const total=((count+1)/2)*CPP;
        for(int c=tid;c<total;c+=Threads) {
            int const j=c/CPP,q=c%CPP;
            cp_async16(st+L::UnitOff+j*CtaN*36+q*16,g.units+(uint64_t(sb0/2+j)*g.n+n_cta)*36+q*16);
        }
    }
}

// Metadata of stage-local superblock j for T columns starting at local column lc.
template<int Q,int CtaN,int SB,int T>
__device__ __forceinline__ void staged_sb(typename SbOf<Q,T>::type& s,uint8_t const* st,int lc,int j) {
    using L=StageLayout<Q,CtaN,SB>;
    if constexpr(Q==12 || Q==13) {
        #pragma unroll
        for(int i=0;i<T;++i) s.u[i]=*reinterpret_cast<uint4 const*>(st+L::UnitOff+(j*CtaN+lc+i)*16);
    } else if constexpr(Q==14) {
        #pragma unroll
        for(int i=0;i<T;++i)
            s.d[i]=__half2float(*reinterpret_cast<__half const*>(st+L::UnitOff+((j>>1)*CtaN+lc+i)*36+(j&1)*18));
    }
}
// Window metadata for the lane's two window groups (superblock-relative groups g0, g1).
template<int Q,int CtaN,int SB,int T>
__device__ __forceinline__ void staged_win(typename MetaOf<Q>::type::template Win<T>& w,uint8_t const* st,int lc,
                                           int j,int g0,int g1) {
    using L=StageLayout<Q,CtaN,SB>;
    if constexpr(Q==14) {
        uint8_t const* rec=st+L::UnitOff+((j>>1)*CtaN+lc)*36+(j&1)*18+2;
        #pragma unroll
        for(int i=0;i<T;++i) {
            w.sc[0][i]=reinterpret_cast<int8_t const*>(rec+i*36)[g0];
            w.sc[1][i]=reinterpret_cast<int8_t const*>(rec+i*36)[g1];
        }
    } else if constexpr(Q==8) {
        w.d[0].load(reinterpret_cast<uint16_t const*>(st+L::MetaOff+(j*8+g0)*L::Pitch)+lc);
        w.d[1].load(reinterpret_cast<uint16_t const*>(st+L::MetaOff+(j*8+g1)*L::Pitch)+lc);
    }
}

} // namespace exact_code_ppu_detail

template<int Q,class ElementCompute,int T,int WN,int WK,int Tokens,int S>
struct ExactCodeCollectivePPU<Q,ElementCompute,T,WN,WK,Tokens,ExactCodeCpAsyncPPU<S>>
    : ExactCodeCollectiveBasePPU<Q,ElementCompute,T,WN,WK,Tokens> {
    using Base=ExactCodeCollectiveBasePPU<Q,ElementCompute,T,WN,WK,Tokens>;
    using Arguments=typename Base::Arguments;
    using DispatchPolicy=ExactCodeCpAsyncPPU<S>;
    static constexpr int Compute=Base::Compute;
    static constexpr int PartitionSuperblocks=exact_code_ppu_detail::kUnitSb<Q>;
    static constexpr int StageSuperblocks=exact_code_ppu_detail::kStageSb<Q,WK>;
    using Layout=exact_code_ppu_detail::StageLayout<Q,Base::CtaN,StageSuperblocks>;
    static_assert(Tokens<=2, "staged weights require direct (M<=2 per CTA) activations");
    static_assert(S>=2 && S<=4 && StageSuperblocks%WK==0);
    static constexpr size_t SharedBytes=size_t(S)*Layout::Bytes;
    static_assert(SharedBytes>=Base::reduction_bytes && SharedBytes<=256*1024);
    static constexpr size_t shared_bytes(int = 0) { return SharedBytes; }

    template<class Store>
    __device__ __forceinline__ static void run(Arguments args,void* storage,Store store) {
        using namespace exact_code_ppu_detail;
        using K=Kind<Q>;
        using M=typename MetaOf<Q>::type;
        using R=Rot<Tokens>;
        constexpr int SB=kStageSb<Q,WK>;
        constexpr int TileN=8*T,CtaN=WN*TileN,TP=R::TP,WG=R::WG,V=R::V,Threads=WN*WK*32;
        constexpr int CPG=K::GK/16,GPS=256/K::GK,Windows=GPS/WG,BPG=K::BPG,GPSB=8/BPG;
        using L=StageLayout<Q,CtaN,SB>;
        auto* smem8=static_cast<uint8_t*>(storage);
        unsigned const tid=threadIdx.x,lane=tid%32,warp=tid/32,m=lane/4,t=lane%4;
        unsigned const wn=warp%WN,wk=warp/WN;
        int const n_cta=args.n_begin,lc=int(wn)*TileN+T*int(m),col=n_cta+lc;
        int const tokens=args.tokens,sb_begin=args.sb_begin,sb_end=args.sb_end;
        Planes const& g=args.planes;
        int const stages=(sb_end-sb_begin+SB-1)/SB;
        auto issue=[&](int s) {
            if(s<stages) {
                int const sb0=sb_begin+s*SB;
                copy_stage<Q,CtaN,SB,Threads>(smem8+(s%S)*L::Bytes,g,n_cta,sb0,min(SB,sb_end-sb0),int(tid));
            }
            cp_async_commit();
        };
        #pragma unroll
        for(int s=0;s<S-1;++s) issue(s);

        typename K::Lane const ls(col);
        float const* a=args.a;
        int const g_lane=m%WG,tau_lane=m/WG;
        bool const supply=tau_lane<tokens;
        float2 pre[2*CPG];
        auto load_act=[&](int group_abs) {
            float const* src=a+int64_t(tau_lane)*args.a_row_stride+group_abs*K::GK+2*t;
            #pragma unroll
            for(int s=0;s<2*CPG;++s)
                pre[s]=supply ? *reinterpret_cast<float2 const*>(src+8*s) : make_float2(0.f,0.f);
        };
        int const first=sb_begin+int(wk);
        if(first<sb_end) load_act(first*GPS+g_lane);

        float acc[T][V];
        #pragma unroll
        for(int i=0;i<T;++i)
            #pragma unroll
            for(int v=0;v<V;++v) acc[i][v]=0.f;
        for(int s=0;s<stages;++s) {
            cp_async_wait<S-2>();
            __syncthreads();                // stage s visible; every warp is done with stage s-1
            issue(s+S-1);                   // refills the slot of stage s-1
            uint8_t const* st=smem8+(s%S)*L::Bytes;
            Planes const sp{reinterpret_cast<uint16_t const*>(st+L::LowOff),reinterpret_cast<uint16_t const*>(st+L::HighOff),
                            st+L::UnitOff,L::Pitch/2};
            #pragma unroll
            for(int j=int(wk);j<SB;j+=WK) {
                int const sb=sb_begin+s*SB+j;
                if(sb<sb_end) {
                    typename SbOf<Q,T>::type sbm;
                    staged_sb<Q,CtaN,SB,T>(sbm,st,lc,j);
                    static_for<0,Windows>([&](auto wc) {
                        constexpr int W=decltype(wc)::value,G0=W*WG;
                        typename M::template Win<T> win;
                        staged_win<Q,CtaN,SB,T>(win,st,lc,j,G0+R::group(t,0),G0+R::group(t,1));
                        uint32_t bw[2*CPG];
                        float asum[2]={0.f,0.f};
                        float part=0.f;
                        #pragma unroll
                        for(int q=0;q<2*CPG;++q) {
                            bw[q]=round_pair<Compute>(pre[q].x,pre[q].y);
                            if constexpr(K::SumK>0) part+=pair_sum<Compute>(bw[q]);
                        }
                        // The lane's next window: same superblock, else its next superblock (sb + WK).
                        int const next=W+1<Windows ? sb*GPS+G0+WG+g_lane : (sb+WK)*GPS+g_lane;
                        if(W+1<Windows || sb+WK<sb_end) load_act(next);
                        if constexpr(K::SumK>0) {
                            part+=__shfl_xor_sync(0xffffffffu,part,1);
                            part+=__shfl_xor_sync(0xffffffffu,part,2);
                            asum[0]=__shfl_sync(0xffffffffu,part,4*int(t));
                            asum[1]=__shfl_sync(0xffffffffu,part,4*int(t)+16);
                        }
                        float c[T][4];
                        #pragma unroll
                        for(int i=0;i<T;++i) c[i][0]=c[i][1]=c[i][2]=c[i][3]=0.f;
                        static_for<0,WG>([&](auto gc) {
                            constexpr int gg=decltype(gc)::value;
                            bool const mine=g_lane==gg;
                            static_for<0,CPG>([&](auto jc) {
                                constexpr int jj=decltype(jc)::value,ch=(G0+gg)*CPG+jj,blk=ch/2,hj=ch%2;
                                constexpr int gs=blk/BPG,bg=blk%BPG;
                                typename K::template Gran<T> gr;
                                K::load(gr,sp,lc,j*GPSB+gs);
                                uint32_t const b0=mine ? bw[2*jj] : 0u,b1=mine ? bw[2*jj+1] : 0u;
                                #pragma unroll
                                for(int i=0;i<T;++i) {
                                    uint2 const av=K::template a<Compute,T,bg,hj>(gr,ls,i);
                                    mma<Compute>(c[i],av.x,av.y,b0,b1);
                                }
                            });
                        });
                        static_for<0,2>([&](auto vc) {
                            constexpr int v=decltype(vc)::value;
                            int const tok=R::token(t,v),gq=R::group(t,v);
                            if(tok<tokens) {
                                #pragma unroll
                                for(int i=0;i<T;++i) {
                                    float2 const af=M::template affine<T,v>(sbm,win,i,unsigned(G0+gq));
                                    acc[i][V==1 ? 0 : v]+=fmaf(af.x,c[i][v],af.y*asum[v]);
                                }
                            }
                        });
                    });
                }
            }
        }
        cp_async_wait<0>();
        #pragma unroll
        for(int i=0;i<T;++i)
            #pragma unroll
            for(int v=0;v<V;++v) {
                float x=acc[i][v];
                x+=__shfl_xor_sync(0xffffffffu,x,1);
                x+=__shfl_xor_sync(0xffffffffu,x,2);
                acc[i][v]=x;
            }
        if constexpr(WK==1) {
            if(t==0) {
                #pragma unroll
                for(int v=0;v<V;++v) {
                    if(v<tokens) {
                        #pragma unroll
                        for(int i=0;i<T;++i) store(v,col+i,acc[i][v]);
                    }
                }
            }
        } else {
            __syncthreads();                // every warp has left the stages before they are reused
            float* partial=reinterpret_cast<float*>(smem8);
            if(t==0) {
                #pragma unroll
                for(int v=0;v<V;++v) {
                    if(v<tokens) {
                        #pragma unroll
                        for(int i=0;i<T;++i) partial[(wk*TP+v)*CtaN+lc+i]=acc[i][v];
                    }
                }
            }
            __syncthreads();
            for(int idx=tid;idx<tokens*CtaN;idx+=Threads) {
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
