#pragma once
// Runtime adapter for the optional actlize exact-code collective. No production
// selector references this entry. The caller supplies canonical, validated planes.
#include "simt_common.cuh"
#include "actlize_extensions/cutlass/gemm/collective/ppu_mma_exact_code.hpp"

namespace quactlize::execution::mma_reader {
namespace ec = cutlass::gemm::collective;
template<int Q> using Kind=ec::exact_code_ppu_detail::Kind<Q>;
template<int Tokens> using Rot=ec::exact_code_ppu_detail::Rot<Tokens>;

template<int Mode,class Collective>
__device__ __forceinline__ void run_tile(qkg_call_v1 call,int split,void* storage) {
    using G=typename Collective::Geometry;
    constexpr int CtaN=Collective::CtaN,US=Collective::PartitionSuperblocks;
    unsigned const row=blockIdx.y,partition=blockIdx.z;
    int const n_cta=blockIdx.x*CtaN;
    int tokens=call.rows,expert=0;
    int64_t a_offset=0,out_offset=0;
    if constexpr(Mode==QKG_INDEXED) {
        auto r=simt::common_locate<QKG_INDEXED>(call,row);
        expert=r.expert;a_offset=r.a;out_offset=r.output;tokens=1;
    }
    // Snapshot scalar launch arguments; no address of the call crosses the
    // collective's epilogue seam (otherwise HGCC reloads them after stores).
    auto store=[=](int tok,int n,float value) {
        if(split==1) call.output[(Mode==QKG_INDEXED ? out_offset : int64_t(tok)*call.out_row_stride)+n]=value;
        else static_cast<float*>(call.workspace)[(int64_t(Mode==QKG_INDEXED ? int(row) : tok)*split+partition)*call.n+n]=value;
    };
    // Preserve the prototype's caller-validated-ID contract, including its NaN
    // diagnostic on invalid IDs; this is not the serving ABI's inactive-ID path.
    if(expert<0 || expert>=call.experts) {
        for(int idx=threadIdx.x;idx<CtaN;idx+=Collective::Threads) store(0,n_cta+idx,__int_as_float(0x7fc00000));
        return;
    }
    uint64_t const nk=uint64_t(call.n)*call.k;
    ec::exact_code_ppu_detail::Planes planes{
        reinterpret_cast<uint16_t const*>(call.low+uint64_t(expert)*nk/8*G::LowBits),nullptr,
        call.units+uint64_t(expert)*G::metadata_bytes(nk),call.n};
    if constexpr(G::HighBits)
        planes.high=reinterpret_cast<uint16_t const*>(call.high+uint64_t(expert)*nk/8*G::HighBits);
    int const units=call.k/(256*US);
    int const begin=US*int(int64_t(units)*partition/split),end=US*int(int64_t(units)*(partition+1)/split);
    typename Collective::Arguments args{planes,static_cast<float const*>(call.a)+a_offset,
        call.a_row_stride,tokens,n_cta,begin,end};
    Collective::run(args,storage,store);
}

template<int Mode,class Collective,class Kernel>
int launch(qkg_call_v1 const& c,int split,Kernel fn) {
    static_assert(Mode==QKG_DENSE || Mode==QKG_INDEXED);
    static_assert(Mode!=QKG_INDEXED || Collective::TokenCapacity==1);
    if(split!=1 && split!=2 && split!=4 && split!=8) return QKG_INVALID;
    if(c.version!=1 || c.size!=sizeof(c) || c.mode!=Mode || c.qtype!=Collective::QType ||
       c.rows<=0 || c.n<=0 || c.k<=0 || c.experts<=0 || c.input_type!=QKG_F32 ||
       !c.a || !c.low || !c.units || !c.output ||
       (Collective::HighBits!=0)!=(c.high!=nullptr)) return QKG_INVALID;
    int const rows=Mode==QKG_INDEXED ? c.rows : 1;
    int const tokens=Mode==QKG_INDEXED ? 1 : c.rows;
    constexpr int align=Collective::TokenCapacity<=2 ? 2 : 4; // F32 elements
    if(c.n%Collective::CtaN || tokens>Collective::TokenCapacity || rows>65535 ||
       c.k%(256*Collective::PartitionSuperblocks*split) ||
       ((uintptr_t(c.low)|uintptr_t(c.high)|uintptr_t(c.units))&15) ||
       (uintptr_t(c.a)&(align*4-1)) || c.a_row_stride<c.k || c.a_row_stride%align ||
       c.out_row_stride<c.n || (uintptr_t(c.output)&3)) return QKG_SHAPE;
    if constexpr(Mode==QKG_INDEXED) {
        if(!c.ids || (uintptr_t(c.ids)&3) || c.topk<=0 || c.topk>c.experts ||
           c.channels<=0 || c.channels>c.topk || c.topk%c.channels || c.rows%c.topk ||
           c.ids_stride<c.topk || c.a_token_stride%align || c.a_row_stride>INT64_MAX/c.channels ||
           c.a_token_stride<c.a_row_stride*c.channels) return QKG_INVALID;
    } else if(c.experts!=1) return QKG_INVALID;
    if(split>1 && (!c.workspace || (uintptr_t(c.workspace)&3) ||
       c.workspace_bytes<uint64_t(c.rows)*c.n*split*sizeof(float))) return QKG_CAPACITY;
    size_t const bytes=Collective::shared_bytes(c.k/split);
    if(bytes>256*1024) return QKG_CAPACITY;
    auto stream=static_cast<hggcStream_t>(c.stream);
    if(bytes>48*1024 && hggcFuncSetAttribute(reinterpret_cast<void const*>(fn),
            hggcFuncAttributeMaxDynamicSharedMemorySize,int(bytes))!=hggcSuccess) return QKG_RUNTIME;
    fn<<<dim3(c.n/Collective::CtaN,rows,split),Collective::Threads,bytes,stream>>>(c,split);
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    simt::launch_reduction<Collective::QType>(c,split,stream);
    return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
}

template<int Q,int Compute,int T,int WN,int WK,int Tokens,int NB>
using RegisterCollective=ec::ExactCodeCollectivePPU<Q,
    std::conditional_t<Compute==0,cutlass::half_t,cutlass::bfloat16_t>,
    T,WN,WK,Tokens,ec::ExactCodeRegisterDirectPPU<NB>>;

// Retain the prototype's kernel/config identity for the existing profiler tools.
template<int Q,int Mode,int Compute,int T,int WN,int WK,int Tokens,int NB>
__global__ void __launch_bounds__(WN*WK*32) kernel(qkg_call_v1 call,int split) {
    extern __shared__ __align__(128) uint8_t storage[];
    run_tile<Mode,RegisterCollective<Q,Compute,T,WN,WK,Tokens,NB>>(call,split,storage);
}

template<int Q,int Mode,int Compute,int T,int WN,int WK,int Tokens,int NB>
int launch_typed(qkg_call_v1 const& c,int split) {
    static_assert(Compute==0 || Compute==1);
    using Collective=RegisterCollective<Q,Compute,T,WN,WK,Tokens,NB>;
    return launch<Mode,Collective>(c,split,kernel<Q,Mode,Compute,T,WN,WK,Tokens,NB>);
}
} // namespace quactlize::execution::mma_reader
