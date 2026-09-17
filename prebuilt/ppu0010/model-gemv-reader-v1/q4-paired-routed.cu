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
__global__ void point_12_1024_2048_arm_0(quactlize::fusion::DeviceCall c) {
    register_reuse_body<12,1,3,4,8,8,1,0,quactlize::fusion::SimtFinish>(c,1);
}
__global__ void point_12_1024_2048_arm_1(quactlize::fusion::DeviceCall c) {
    register_reuse_body<12,1,3,4,8,8,1,1,quactlize::fusion::SimtFinish>(c,1);
}
__global__ void point_12_1024_2048_arm_2(quactlize::fusion::DeviceCall c) {
    register_reuse_body<12,1,3,4,8,8,1,3,quactlize::fusion::SimtFinish>(c,1);
}
__global__ void point_12_1024_2048_arm_3(quactlize::fusion::DeviceCall c) {
    c.n=1024;c.k=2048;c.experts=256;c.mode=2;c.channels=1;c.topk=8;
    register_reuse_body<12,1,3,4,8,8,1,1,quactlize::fusion::SimtFinish>(c,1);
}
__global__ void point_12_1024_2048_arm_4(quactlize::fusion::DeviceCall c) {
    register_reuse_body<12,1,3,8,8,4,1,1,quactlize::fusion::SimtFinish>(c,1);
}
__global__ void point_12_1024_2048_arm_5(quactlize::fusion::DeviceCall c) {
    register_reuse_body<12,1,3,8,4,4,1,1,quactlize::fusion::SimtFinish>(c,1);
}
__global__ void point_12_1024_2048_arm_6(quactlize::fusion::DeviceCall c) {
    c.n=1024;c.k=2048;c.experts=256;c.mode=2;c.channels=1;c.topk=8;
    register_reuse_body<12,1,3,8,4,4,1,1,quactlize::fusion::SimtFinish>(c,1);
}
__global__ void point_12_1024_2048_arm_7(quactlize::fusion::DeviceCall c) {
    register_reuse_body<12,1,3,4,8,4,1,1,Paired16Finish>(c,1);
}
__global__ void point_12_1024_2048_arm_8(quactlize::fusion::DeviceCall c) {
    register_reuse_body<12,1,3,4,4,4,1,1,Paired16Finish>(c,1);
}
} // namespace
extern "C" int model_gemv_run(qkg_simt_call_v2 const* input,
    qkg_gate_up_call_v2 const* fusion,qkg_gate_up_layout_v1 const* layout,int arm) {
    using namespace quactlize::execution;
    if(!fusion || !layout || input) return QKG_INVALID;
    auto const& d=fusion->call.input;
    auto const& c=d.call;
    if(c.qtype!=12 || c.n!=512 || c.k!=2048 || c.experts!=256 ||
       c.mode!=2 || c.channels!=1 || c.topk!=8 ||
       c.input_type!=QKG_F32 || d.compute_type!=1) return QKG_SHAPE;
    qkg_sizes_v1 sizes{};
    auto stream=static_cast<hggcStream_t>(c.stream);
    int rc=QKG_INVALID;
    switch(arm) {
    case 0: {
        qkg_gate_up_config_v1 cfg{1,sizeof(cfg),QKG_GATE_UP_SIMT,1,0,8};
        if(fusion->call.output_type!=QKG_F32 || fusion->call.round_projection!=1) return QKG_INVALID;
        rc=quactlize::fusion::query(fusion->call,cfg,*layout,sizes);if(rc) return rc;
        rc=quactlize::fusion::buffers(fusion->call,sizes);if(rc) return rc;
        rc=quactlize::fusion::row_buffers(*fusion,sizes);if(rc) return rc;
        quactlize::fusion::DeviceCall call{};
        static_cast<qkg_call_v1&>(call)=c;
        call.n*=2;call.compute_type=d.compute_type;call.output_type=QKG_F32;
        call.round_projection=fusion->call.round_projection;
        call.input_rows=fusion->input_rows;call.status=fusion->status;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_12_1024_2048_arm_0<<<c.rows*1*(1024/32),256,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    case 1: {
        qkg_gate_up_config_v1 cfg{1,sizeof(cfg),QKG_GATE_UP_SIMT,1,0,8};
        if(fusion->call.output_type!=QKG_F32 || fusion->call.round_projection!=1) return QKG_INVALID;
        rc=quactlize::fusion::query(fusion->call,cfg,*layout,sizes);if(rc) return rc;
        rc=quactlize::fusion::buffers(fusion->call,sizes);if(rc) return rc;
        rc=quactlize::fusion::row_buffers(*fusion,sizes);if(rc) return rc;
        quactlize::fusion::DeviceCall call{};
        static_cast<qkg_call_v1&>(call)=c;
        call.n*=2;call.compute_type=d.compute_type;call.output_type=QKG_F32;
        call.round_projection=fusion->call.round_projection;
        call.input_rows=fusion->input_rows;call.status=fusion->status;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_12_1024_2048_arm_1<<<c.rows*1*(1024/32),256,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    case 2: {
        qkg_gate_up_config_v1 cfg{1,sizeof(cfg),QKG_GATE_UP_SIMT,1,0,8};
        if(fusion->call.output_type!=QKG_F32 || fusion->call.round_projection!=1) return QKG_INVALID;
        rc=quactlize::fusion::query(fusion->call,cfg,*layout,sizes);if(rc) return rc;
        rc=quactlize::fusion::buffers(fusion->call,sizes);if(rc) return rc;
        rc=quactlize::fusion::row_buffers(*fusion,sizes);if(rc) return rc;
        quactlize::fusion::DeviceCall call{};
        static_cast<qkg_call_v1&>(call)=c;
        call.n*=2;call.compute_type=d.compute_type;call.output_type=QKG_F32;
        call.round_projection=fusion->call.round_projection;
        call.input_rows=fusion->input_rows;call.status=fusion->status;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_12_1024_2048_arm_2<<<c.rows*1*(1024/32),256,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    case 3: {
        qkg_gate_up_config_v1 cfg{1,sizeof(cfg),QKG_GATE_UP_SIMT,1,0,8};
        if(fusion->call.output_type!=QKG_F32 || fusion->call.round_projection!=1) return QKG_INVALID;
        rc=quactlize::fusion::query(fusion->call,cfg,*layout,sizes);if(rc) return rc;
        rc=quactlize::fusion::buffers(fusion->call,sizes);if(rc) return rc;
        rc=quactlize::fusion::row_buffers(*fusion,sizes);if(rc) return rc;
        quactlize::fusion::DeviceCall call{};
        static_cast<qkg_call_v1&>(call)=c;
        call.n*=2;call.compute_type=d.compute_type;call.output_type=QKG_F32;
        call.round_projection=fusion->call.round_projection;
        call.input_rows=fusion->input_rows;call.status=fusion->status;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_12_1024_2048_arm_3<<<c.rows*1*(1024/32),256,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    case 4: {
        qkg_gate_up_config_v1 cfg{1,sizeof(cfg),QKG_GATE_UP_SIMT,1,0,8};
        if(fusion->call.output_type!=QKG_F32 || fusion->call.round_projection!=1) return QKG_INVALID;
        rc=quactlize::fusion::query(fusion->call,cfg,*layout,sizes);if(rc) return rc;
        rc=quactlize::fusion::buffers(fusion->call,sizes);if(rc) return rc;
        rc=quactlize::fusion::row_buffers(*fusion,sizes);if(rc) return rc;
        quactlize::fusion::DeviceCall call{};
        static_cast<qkg_call_v1&>(call)=c;
        call.n*=2;call.compute_type=d.compute_type;call.output_type=QKG_F32;
        call.round_projection=fusion->call.round_projection;
        call.input_rows=fusion->input_rows;call.status=fusion->status;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_12_1024_2048_arm_4<<<c.rows*1*(1024/32),256,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    case 5: {
        qkg_gate_up_config_v1 cfg{1,sizeof(cfg),QKG_GATE_UP_SIMT,1,0,4};
        if(fusion->call.output_type!=QKG_F32 || fusion->call.round_projection!=1) return QKG_INVALID;
        rc=quactlize::fusion::query(fusion->call,cfg,*layout,sizes);if(rc) return rc;
        rc=quactlize::fusion::buffers(fusion->call,sizes);if(rc) return rc;
        rc=quactlize::fusion::row_buffers(*fusion,sizes);if(rc) return rc;
        quactlize::fusion::DeviceCall call{};
        static_cast<qkg_call_v1&>(call)=c;
        call.n*=2;call.compute_type=d.compute_type;call.output_type=QKG_F32;
        call.round_projection=fusion->call.round_projection;
        call.input_rows=fusion->input_rows;call.status=fusion->status;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_12_1024_2048_arm_5<<<c.rows*1*(1024/32),128,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    case 6: {
        qkg_gate_up_config_v1 cfg{1,sizeof(cfg),QKG_GATE_UP_SIMT,1,0,4};
        if(fusion->call.output_type!=QKG_F32 || fusion->call.round_projection!=1) return QKG_INVALID;
        rc=quactlize::fusion::query(fusion->call,cfg,*layout,sizes);if(rc) return rc;
        rc=quactlize::fusion::buffers(fusion->call,sizes);if(rc) return rc;
        rc=quactlize::fusion::row_buffers(*fusion,sizes);if(rc) return rc;
        quactlize::fusion::DeviceCall call{};
        static_cast<qkg_call_v1&>(call)=c;
        call.n*=2;call.compute_type=d.compute_type;call.output_type=QKG_F32;
        call.round_projection=fusion->call.round_projection;
        call.input_rows=fusion->input_rows;call.status=fusion->status;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_12_1024_2048_arm_6<<<c.rows*1*(1024/32),128,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    case 7: {
        qkg_gate_up_config_v1 cfg{1,sizeof(cfg),QKG_GATE_UP_SIMT,1,0,8};
        if(fusion->call.output_type!=QKG_F32 || fusion->call.round_projection!=1) return QKG_INVALID;
        rc=quactlize::fusion::query(fusion->call,cfg,*layout,sizes);if(rc) return rc;
        rc=quactlize::fusion::buffers(fusion->call,sizes);if(rc) return rc;
        rc=quactlize::fusion::row_buffers(*fusion,sizes);if(rc) return rc;
        quactlize::fusion::DeviceCall call{};
        static_cast<qkg_call_v1&>(call)=c;
        call.n*=2;call.compute_type=d.compute_type;call.output_type=QKG_F32;
        call.round_projection=fusion->call.round_projection;
        call.input_rows=fusion->input_rows;call.status=fusion->status;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_12_1024_2048_arm_7<<<c.rows*1*(1024/16),256,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    case 8: {
        qkg_gate_up_config_v1 cfg{1,sizeof(cfg),QKG_GATE_UP_SIMT,1,0,4};
        if(fusion->call.output_type!=QKG_F32 || fusion->call.round_projection!=1) return QKG_INVALID;
        rc=quactlize::fusion::query(fusion->call,cfg,*layout,sizes);if(rc) return rc;
        rc=quactlize::fusion::buffers(fusion->call,sizes);if(rc) return rc;
        rc=quactlize::fusion::row_buffers(*fusion,sizes);if(rc) return rc;
        quactlize::fusion::DeviceCall call{};
        static_cast<qkg_call_v1&>(call)=c;
        call.n*=2;call.compute_type=d.compute_type;call.output_type=QKG_F32;
        call.round_projection=fusion->call.round_projection;
        call.input_rows=fusion->input_rows;call.status=fusion->status;
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_12_1024_2048_arm_8<<<c.rows*1*(1024/16),128,0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }
    default:return QKG_INVALID;
    }
}
