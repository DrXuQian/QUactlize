#pragma once
#include "store.cuh"
#include "../decode/types.cuh"

namespace quactlize::fusion {
template<int Q,int TM,class Compute,class Source>
struct TcTypes {
    static constexpr int TK=Q==11 || Q==13 ? 256 : Q==10 || Q==14 ? 128 : 64;
    using Metadata=std::conditional_t<Q==8,cutlass::half_t,Compute>;
    using Core=runtime::GroupedTypes<Q,TM,64,TK,TM,16,2,16,false,float,false,Compute,Metadata>;
    using Mainloop=typename decode::InputCollective<typename Core::Mainloop,Source>::type;
    using Tile=typename Core::Tile;
    using Mma=typename Mainloop::TiledMma;
    static constexpr int Threads=cute::size(Mma{});
    struct Params {
        DeviceCall call;
        typename Mainloop::Params mainloop;
        uint64_t low_bytes,high_bytes,unit_bytes;
        int split;
    };
};

template<class Pointer>
CUTLASS_DEVICE Pointer advance_bytes(Pointer p,uint64_t bytes) {
    return reinterpret_cast<Pointer>(reinterpret_cast<uint8_t const*>(p)+bytes);
}

// Reuse the shipping collective; only row binding, split traversal and the
// paired-register epilogue differ. No gate/up global output for S1.
template<class T>
__global__ __launch_bounds__(T::Threads) void tc_gate_up(typename T::Params p) {
    using namespace cute;
    auto const& c=p.call;
    int batch=int(blockIdx.z)/p.split, slice=int(blockIdx.z)%p.split;
    int expert=batch, begin=0, rows=c.rows;
    int64_t a_offset=0;
    if(c.mode==QKG_INDEXED) {
        int source=input_row(c,batch);
        int token=source<0 ? 0 : source/c.topk,slot=source<0 ? 0 : source%c.topk;
        expert=source<0 ? -1 : c.ids[int64_t(token)*c.ids_stride+slot];
        begin=batch; rows=1;
        a_offset=int64_t(token)*c.a_token_stride+(slot%c.channels)*c.a_row_stride;
    } else if(c.mode==QKG_GROUPED) {
        begin=c.offsets[expert]; rows=c.offsets[expert+1]-begin;
        if(begin<0 || rows<0 || int64_t(begin)+rows>c.rows) return;
        a_offset=int64_t(begin)*c.a_row_stride;
    }
    if(c.status && *c.status) expert=-1;
    constexpr int TM=size<0>(typename T::Tile{});
    int m_base=int(blockIdx.x)*TM;
    if(m_base>=rows) return;
    if(expert<0 || expert>=c.experts) {
        for(int linear=threadIdx.x;linear<TM*32;linear+=blockDim.x) {
            int m=m_base+linear/32,col=int(blockIdx.y)*32+linear%32;
            if(m>=rows || col>=c.n/2) continue;
            if(p.split==1) output(c,begin+m,col,__int_as_float(0x7fc00000));
            else {
                auto dst=static_cast<float*>(c.workspace)+(int64_t(begin+m)*p.split+slice)*c.n;
                dst[PairedN4::gate(col)]=dst[PairedN4::up(col)]=__int_as_float(0x7fc00000);
            }
        }
        return;
    }
    auto ml=p.mainloop;
    ml.ptr_A+=a_offset+int64_t(m_base)*c.a_row_stride;
    ml.ptr_B=advance_bytes(ml.ptr_B,uint64_t(expert)*p.low_bytes);
    ml.ptr_S=advance_bytes(ml.ptr_S,uint64_t(expert)*p.unit_bytes);
    if constexpr(!std::is_void_v<typename T::Core::High>)
        ml.ptr_B2=advance_bytes(ml.ptr_B2,uint64_t(expert)*p.high_bytes);
    // The typed A copy predicates local rows against desc.dim_h. Rebase
    // each M tile so the final physical cube never reads the next tensor.
    int tile_rows=rows-m_base<TM ? rows-m_base : TM;
    auto problem_shape=make_shape(tile_rows,c.n,c.k,1);
    auto coord=make_coord(0,int(blockIdx.y),_,0);
    typename T::Mainloop mainloop;
    auto inputs=mainloop.load_init(problem_shape,coord,ml);
    auto gA=get<0>(inputs);
    typename T::Mma mma;
    auto accum=make_fragment_like<float>(partition_fragment_C(mma,take<0,2>(typename T::Tile{})));
    clear(accum);
    auto iterator=make_splitk_coord_iterator(shape<2>(gA),slice,p.split);
    extern __shared__ __align__(16) char storage[];
    mainloop(ml,inputs,accum,iterator,size<2>(gA)/p.split,int(threadIdx.x),storage);
    auto coordinates=mma.get_thread_slice(int(threadIdx.x)).partition_C(
        make_identity_tensor(take<0,2>(typename T::Tile{})));
    CUTE_STATIC_ASSERT_V(size(accum)==size(coordinates));
    // PPU0010 CLayout's consecutive value slots pair N and N+4 at equal M.
    #pragma unroll
    for(int i=0;i<size(accum);++i) {
        auto mn=coordinates(i);
        int m=m_base+int(get<0>(mn));
        int n=int(blockIdx.y)*64+int(get<1>(mn));
        if(m>=rows || n>=c.n) continue;
        if(p.split>1) {
            static_cast<float*>(c.workspace)[(int64_t(begin+m)*p.split+slice)*c.n+n]=accum(i);
        } else if((i&1)==0) {
            output(c,begin+m,PairedN4::channel(n),activate(c,accum(i),accum(i+1)));
        }
    }
}

template<int Q,int TM,class Compute,class Source>
int tc_launch(DeviceCall const& c,qkg_gate_up_config_v1 const& f,qkg_sizes_v1 const& sizes) {
    using namespace cute;
    using T=TcTypes<Q,TM,Compute,Source>;
    using Mainloop=typename T::Mainloop;
    using Policy=typename T::Core::Policy;
    using F=runtime::Format<Q>;
    static_assert(sizeof(typename Mainloop::SharedStorage)<=48*1024,
                  "larger candidates need explicit pre-capture shared-memory admission");
    typename Mainloop::Arguments a{};
    a.ptr_A=static_cast<Source const*>(c.a);
    a.dA=cutlass::make_cute_packed_stride(typename Mainloop::StrideA{},make_shape(c.rows,c.k,1));
    get<0>(a.dA)=c.a_row_stride;
    a.ptr_B=reinterpret_cast<typename Mainloop::ElementB const*>(c.low);
    a.dB=cutlass::make_cute_packed_stride(typename Mainloop::StrideB{},
        make_shape(c.n/Policy::ArtifactLowFold,c.k*Policy::ArtifactLowFold,1));
    a.ptr_S=reinterpret_cast<typename Mainloop::ElementScale const*>(c.units);
    a.dS=cutlass::make_cute_packed_stride(typename Mainloop::StrideScale{},make_shape(c.n,c.k/F::spec.group_size,1));
    a.group_size=F::spec.group_size;
    if constexpr(!std::is_void_v<typename T::Core::High>) {
        a.ptr_B2=reinterpret_cast<typename T::Core::High const*>(c.high);
        if constexpr(Policy::ArtifactHighFold!=Policy::ArtifactLowFold) {
            a.dB2=cutlass::make_cute_packed_stride(typename Mainloop::StrideB{},
                make_shape(c.n/Policy::ArtifactHighFold,c.k*Policy::ArtifactHighFold,1));
            a.dB2_valid=true;
        }
    }
    auto shape=make_shape(c.rows,c.n,c.k,1);
    if(!Mainloop::can_implement(shape,a)) return QKG_SHAPE;
    typename T::Params p{c,Mainloop::to_underlying_arguments(shape,a,nullptr),
        sizes.low_bytes/c.experts,sizes.high_bytes/c.experts,sizes.units_bytes/c.experts,f.split};
    int bound=c.mode==QKG_INDEXED ? 1 : c.rows;
    int groups=c.mode==QKG_INDEXED ? c.rows : c.experts;
    dim3 grid((bound+int64_t(TM)-1)/TM,c.n/64,groups*f.split);
    auto stream=static_cast<hggcStream_t>(c.stream);
    tc_gate_up<T><<<grid,T::Threads,sizeof(typename Mainloop::SharedStorage),stream>>>(p);
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    if(f.split>1) reduce_gate_up<<<(int64_t(c.rows)*(c.n/2)+127)/128,128,0,stream>>>(c,f.split);
    return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
}
}
