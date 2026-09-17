#include "quactlize/fusion/store.cuh"
#include "quactlize/execution/q4_s1_validation.hpp"
namespace quactlize::execution::q4_s1 {
CUTLASS_DEVICE Row locate(quactlize::fusion::DeviceCall const& c,int row) {
    int source=quactlize::fusion::input_row(c,row);
    if(source<0) return {-1,0,0};
    return locate(static_cast<qkg_call_v1 const&>(c),source);
}}
#include "quactlize/execution/simt_q8_vector.cuh"
#include "quactlize/execution/simt_validation.hpp"
#include "quactlize/fusion/validation.hpp"
namespace quactlize::execution::model_gemv {
using namespace quactlize::execution::simt;
// A half-width physical tile contains two complete G4/U4 pairs. Keep
// the same sequential inter-warp sum, not the ordinary XOR fold order.
struct Paired16Finish : quactlize::fusion::SimtFinish {
    template<int TileN,int Warps>
    CUTLASS_DEVICE static void finish(quactlize::fusion::DeviceCall const& c,
        int row,int tile,int partition,int split,float const* partial) {
        static_assert(TileN==16, "half-width paired tile");
        int tid=int(threadIdx.x);
        if(tid<8) {
            int ng=quactlize::fusion::PairedN4::gate(tid);
            float gate=0.f,up=0.f;
            #pragma unroll
            for(int w=0;w<Warps;++w) {
                gate+=partial[w*16+ng];up+=partial[w*16+ng+4];
            }
            quactlize::fusion::output(c,row,tile*8+tid,quactlize::fusion::activate(c,gate,up));
        }
    }
};
template<int Q,int Input,int Variant,int Columns,int Warps,int P,int Compute=0,int Changes=0,
         class Finish=void,class Call=qkg_call_v1>
__device__ __forceinline__ void direct_metadata_body(Call call,int split) {
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
        static_assert(Q==14 && F::Unit::kSbBytes==18 && F::Unit::kSbPerUnit==2);
        float2 metadata[P];
        #pragma unroll
        for (int p=0;p<P;++p) {
            auto unit=units+F::unit_offset(call.n,col+p,g)+((g/16)&1)*18;
            float d=__half2float(__ushort_as_half(*reinterpret_cast<uint16_t const*>(unit)));
            int scale=int(*reinterpret_cast<int8_t const*>(unit+2+(g&15)));
            metadata[p]=make_float2(d*float(scale),0.f);
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
                    load_words<P>(low+F::Low::word_index(col,k,call.n),lo[segment][residue]);
                }
            }
            if constexpr(F::high_bits) {
                #pragma unroll
                for (int residue=0;residue<4;++residue) {
                    int k=g*F::group+half*4+residue;
                    load_words<P>(high+F::High::word_index(col,k,call.n),hi[residue]);
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
                        float2 code=codes<Q>(lo[slot/F::Low::kPack][residue][p],hi[residue][p],col+2*p,k);
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
__global__ void point_14_248320_2048_arm_0(qkg_call_v1 c) {
    register_reuse_body<14,1,3,8,4,4,0,0>(c,1);
}
__global__ void point_14_248320_2048_arm_1(qkg_call_v1 c) {
    direct_metadata_body<14,1,3,8,4,4,0,0>(c,1);
}
__global__ void point_14_248320_2048_arm_2(qkg_call_v1 c) {
    c.n=248320;c.k=2048;c.experts=1;c.mode=0;c.channels=1;c.topk=1;
    direct_metadata_body<14,1,3,8,4,4,0,0>(c,1);
}
__global__ void point_14_248320_2048_arm_3(qkg_call_v1 c) {
    register_reuse_body<14,1,3,4,4,4,0,0>(c,1);
}
__global__ void point_14_248320_2048_arm_4(qkg_call_v1 c) {
    direct_metadata_body<14,1,3,4,4,4,0,0>(c,1);
}
__global__ void point_14_248320_2048_arm_5(qkg_call_v1 c) {
    direct_metadata_body<14,1,3,4,4,8,0,0>(c,1);
}
__global__ void point_14_248320_2048_arm_6(qkg_call_v1 c) {
    direct_metadata_body<14,1,3,8,2,4,0,0>(c,1);
}
} // namespace
extern "C" int model_gemv_run(qkg_simt_call_v2 const* input,
    qkg_gate_up_call_v2 const* fusion,qkg_gate_up_layout_v1 const* layout,int arm) {
    using namespace quactlize::execution;
    if(!input || fusion || layout) return QKG_INVALID;
    auto const& d=*input;
    auto const& c=d.call;
    if(c.qtype!=14 || c.n!=248320 || c.k!=2048 || c.experts!=1 ||
       c.mode!=0 || c.channels!=1 || c.topk!=1 ||
       c.input_type!=QKG_F32 || d.compute_type!=0) return QKG_SHAPE;
    qkg_sizes_v1 sizes{};
    auto stream=static_cast<hggcStream_t>(c.stream);
    int rc=QKG_INVALID;
    switch(arm) {
    case 0: {
        qkg_simt_config_v1 cfg{1,sizeof(cfg),3,8,4,4,1};
        auto arrangement=ppu_arrangements::kquant_kpack_transpose_v1(14);
        rc=simt::query_v2(d,cfg,&arrangement,sizes);if(rc) return rc;
        rc=simt::buffers_v2(d,sizes);if(rc) return rc;
        auto call=c;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_14_248320_2048_arm_0<<<c.rows*1*(248320/32),128,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    case 1: {
        qkg_simt_config_v1 cfg{1,sizeof(cfg),3,8,4,4,1};
        auto arrangement=ppu_arrangements::kquant_kpack_transpose_v1(14);
        rc=simt::query_v2(d,cfg,&arrangement,sizes);if(rc) return rc;
        rc=simt::buffers_v2(d,sizes);if(rc) return rc;
        auto call=c;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_14_248320_2048_arm_1<<<c.rows*1*(248320/32),128,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    case 2: {
        qkg_simt_config_v1 cfg{1,sizeof(cfg),3,8,4,4,1};
        auto arrangement=ppu_arrangements::kquant_kpack_transpose_v1(14);
        rc=simt::query_v2(d,cfg,&arrangement,sizes);if(rc) return rc;
        rc=simt::buffers_v2(d,sizes);if(rc) return rc;
        auto call=c;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_14_248320_2048_arm_2<<<c.rows*1*(248320/32),128,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    case 3: {
        qkg_simt_config_v1 cfg{1,sizeof(cfg),3,4,4,4,1};
        auto arrangement=ppu_arrangements::kquant_kpack_transpose_v1(14);
        rc=simt::query_v2(d,cfg,&arrangement,sizes);if(rc) return rc;
        rc=simt::buffers_v2(d,sizes);if(rc) return rc;
        auto call=c;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_14_248320_2048_arm_3<<<c.rows*1*(248320/16),128,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    case 4: {
        qkg_simt_config_v1 cfg{1,sizeof(cfg),3,4,4,4,1};
        auto arrangement=ppu_arrangements::kquant_kpack_transpose_v1(14);
        rc=simt::query_v2(d,cfg,&arrangement,sizes);if(rc) return rc;
        rc=simt::buffers_v2(d,sizes);if(rc) return rc;
        auto call=c;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_14_248320_2048_arm_4<<<c.rows*1*(248320/16),128,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    case 5: {
        qkg_simt_config_v1 cfg{1,sizeof(cfg),3,4,4,8,1};
        auto arrangement=ppu_arrangements::kquant_kpack_transpose_v1(14);
        rc=simt::query_v2(d,cfg,&arrangement,sizes);if(rc) return rc;
        rc=simt::buffers_v2(d,sizes);if(rc) return rc;
        auto call=c;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_14_248320_2048_arm_5<<<c.rows*1*(248320/32),128,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    case 6: {
        qkg_simt_config_v1 cfg{1,sizeof(cfg),3,8,2,4,1};
        auto arrangement=ppu_arrangements::kquant_kpack_transpose_v1(14);
        rc=simt::query_v2(d,cfg,&arrangement,sizes);if(rc) return rc;
        rc=simt::buffers_v2(d,sizes);if(rc) return rc;
        auto call=c;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_14_248320_2048_arm_6<<<c.rows*1*(248320/32),64,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    default:return QKG_INVALID;
    }
}
