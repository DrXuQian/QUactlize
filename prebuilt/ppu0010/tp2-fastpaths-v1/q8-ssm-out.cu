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
__global__ void point_8_2048_4096_arm_0(qkg_call_v1 c) {
    c.n=2048;c.k=4096;c.experts=1;c.mode=0;c.channels=1;c.topk=1;
    q8_vector::kernel_body<1,0,1,8,4,4,false>(c,8);
}
__global__ void point_8_2048_4096_arm_1(qkg_call_v1 c) {
    q8_vector::kernel_body<1,0,1,8,4,4,false>(c,1);
}
__global__ void point_8_2048_4096_arm_2(qkg_call_v1 c) {
    q8_vector::kernel_body<1,0,1,8,4,4,true>(c,1);
}
__global__ void point_8_2048_4096_arm_3(qkg_call_v1 c) {
    c.n=2048;c.k=4096;c.experts=1;c.mode=0;c.channels=1;c.topk=1;
    q8_vector::kernel_body<1,0,1,8,4,4,true>(c,1);
}
__global__ void point_8_2048_4096_arm_4(qkg_call_v1 c) {
    q8_vector::kernel_body<1,0,1,4,8,4,true>(c,1);
}
__global__ void point_8_2048_4096_arm_5(qkg_call_v1 c) {
    q8_vector::kernel_body<1,0,1,8,4,4,false>(c,2);
}
__global__ void point_8_2048_4096_arm_6(qkg_call_v1 c) {
    q8_vector::kernel_body<1,0,1,8,4,4,false>(c,4);
}
__global__ void point_8_2048_4096_arm_7(qkg_call_v1 c) {
    q8_vector::kernel_body<1,0,1,8,4,4,false>(c,4);
}
__global__ void point_8_2048_4096_arm_8(qkg_call_v1 c) {
    q8_vector::kernel_body<1,0,1,8,4,4,false>(c,8);
}
} // namespace
extern "C" int model_gemv_run(qkg_simt_call_v2 const* input,
    qkg_gate_up_call_v2 const* fusion,qkg_gate_up_layout_v1 const* layout,int arm) {
    using namespace quactlize::execution;
    if(!input || fusion || layout) return QKG_INVALID;
    auto const& d=*input;
    auto const& c=d.call;
    if(c.qtype!=8 || c.n!=2048 || c.k!=4096 || c.experts!=1 ||
       c.mode!=0 || c.channels!=1 || c.topk!=1 ||
       c.input_type!=QKG_F32 || d.compute_type!=0) return QKG_SHAPE;
    qkg_sizes_v1 sizes{};
    auto stream=static_cast<hggcStream_t>(c.stream);
    int rc=QKG_INVALID;
    switch(arm) {
    case 0: {
        qkg_simt_config_v1 cfg{1,sizeof(cfg),5,8,4,4,8};
        auto arrangement=q8_kpack2::arrangement();
        rc=simt::query_v2(d,cfg,&arrangement,sizes);if(rc) return rc;
        rc=simt::buffers_v2(d,sizes);if(rc) return rc;
        auto call=c;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_8_2048_4096_arm_0<<<c.rows*8*(2048/32),128,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        if(c.rows==1 && !((uintptr_t(c.output)|uintptr_t(c.workspace))&7))
            quactlize::decode::reduce_decode<8><<<(c.n+63)/64,32,0,stream>>>(static_cast<float const*>(c.workspace),c.output,c.n);
        else
        simt::register_reuse_reduce<8><<<(int64_t(c.rows)*c.n+127)/128,128,0,stream>>>(c,8);
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    case 1: {
        qkg_simt_config_v1 cfg{1,sizeof(cfg),5,8,4,4,1};
        auto arrangement=q8_kpack2::arrangement();
        rc=simt::query_v2(d,cfg,&arrangement,sizes);if(rc) return rc;
        rc=simt::buffers_v2(d,sizes);if(rc) return rc;
        auto call=c;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_8_2048_4096_arm_1<<<c.rows*1*(2048/32),128,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    case 2: {
        qkg_simt_config_v1 cfg{1,sizeof(cfg),5,8,4,4,1};
        auto arrangement=q8_kpack2::arrangement();
        rc=simt::query_v2(d,cfg,&arrangement,sizes);if(rc) return rc;
        rc=simt::buffers_v2(d,sizes);if(rc) return rc;
        auto call=c;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_8_2048_4096_arm_2<<<c.rows*1*(2048/32),128,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    case 3: {
        qkg_simt_config_v1 cfg{1,sizeof(cfg),5,8,4,4,1};
        auto arrangement=q8_kpack2::arrangement();
        rc=simt::query_v2(d,cfg,&arrangement,sizes);if(rc) return rc;
        rc=simt::buffers_v2(d,sizes);if(rc) return rc;
        auto call=c;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_8_2048_4096_arm_3<<<c.rows*1*(2048/32),128,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    case 4: {
        qkg_simt_config_v1 cfg{1,sizeof(cfg),5,4,8,4,1};
        auto arrangement=q8_kpack2::arrangement();
        rc=simt::query_v2(d,cfg,&arrangement,sizes);if(rc) return rc;
        rc=simt::buffers_v2(d,sizes);if(rc) return rc;
        auto call=c;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_8_2048_4096_arm_4<<<c.rows*1*(2048/16),256,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    case 5: {
        qkg_simt_config_v1 cfg{1,sizeof(cfg),5,8,4,4,2};
        auto arrangement=q8_kpack2::arrangement();
        rc=simt::query_v2(d,cfg,&arrangement,sizes);if(rc) return rc;
        rc=simt::buffers_v2(d,sizes);if(rc) return rc;
        auto call=c;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_8_2048_4096_arm_5<<<c.rows*2*(2048/32),128,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        if(model_gemv::vector_reduction(c,2))
            quactlize::decode::reduce_decode_rows<2><<<(int64_t(c.rows)*(c.n/2)+31)/32,32,0,stream>>>(
                static_cast<float const*>(c.workspace),c.output,c.rows,c.n,c.out_row_stride);
        else
        simt::register_reuse_reduce<8><<<(int64_t(c.rows)*c.n+127)/128,128,0,stream>>>(c,2);
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    case 6: {
        qkg_simt_config_v1 cfg{1,sizeof(cfg),5,8,4,4,4};
        auto arrangement=q8_kpack2::arrangement();
        rc=simt::query_v2(d,cfg,&arrangement,sizes);if(rc) return rc;
        rc=simt::buffers_v2(d,sizes);if(rc) return rc;
        auto call=c;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_8_2048_4096_arm_6<<<c.rows*4*(2048/32),128,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        simt::register_reuse_reduce<8><<<(int64_t(c.rows)*c.n+127)/128,128,0,stream>>>(c,4);
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    case 7: {
        qkg_simt_config_v1 cfg{1,sizeof(cfg),5,8,4,4,4};
        auto arrangement=q8_kpack2::arrangement();
        rc=simt::query_v2(d,cfg,&arrangement,sizes);if(rc) return rc;
        rc=simt::buffers_v2(d,sizes);if(rc) return rc;
        auto call=c;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_8_2048_4096_arm_7<<<c.rows*4*(2048/32),128,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        if(model_gemv::vector_reduction(c,4))
            quactlize::decode::reduce_decode_rows<4><<<(int64_t(c.rows)*(c.n/2)+31)/32,32,0,stream>>>(
                static_cast<float const*>(c.workspace),c.output,c.rows,c.n,c.out_row_stride);
        else
        simt::register_reuse_reduce<8><<<(int64_t(c.rows)*c.n+127)/128,128,0,stream>>>(c,4);
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    case 8: {
        qkg_simt_config_v1 cfg{1,sizeof(cfg),5,8,4,4,8};
        auto arrangement=q8_kpack2::arrangement();
        rc=simt::query_v2(d,cfg,&arrangement,sizes);if(rc) return rc;
        rc=simt::buffers_v2(d,sizes);if(rc) return rc;
        auto call=c;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_8_2048_4096_arm_8<<<c.rows*8*(2048/32),128,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        if(model_gemv::vector_reduction(c,8))
            quactlize::decode::reduce_decode_rows<8><<<(int64_t(c.rows)*(c.n/2)+31)/32,32,0,stream>>>(
                static_cast<float const*>(c.workspace),c.output,c.rows,c.n,c.out_row_stride);
        else
        simt::register_reuse_reduce<8><<<(int64_t(c.rows)*c.n+127)/128,128,0,stream>>>(c,8);
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    default:return QKG_INVALID;
    }
}
