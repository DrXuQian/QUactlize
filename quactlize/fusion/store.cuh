#pragma once
#include "gate_up.h"
#include "paired_n4.hpp"
#include "cutlass/numeric_types.h"

namespace quactlize::fusion {
struct DeviceCall : qkg_call_v1 {
    int compute_type, output_type, round_projection;
    int32_t const* input_rows;
    int32_t const* status;
};

CUTLASS_DEVICE int input_row(DeviceCall const& c, int row) {
    if (c.status && *c.status) return -1;
    int source = c.input_rows ? c.input_rows[row] : row;
    return source >= 0 && source < c.rows ? source : -1;
}

CUTLASS_DEVICE float activate(DeviceCall const& c, float gate, float up) {
    if (c.round_projection) {
        if (c.compute_type == QKG_COMPUTE_BF16) {
            gate = float(cutlass::bfloat16_t(gate)); up = float(cutlass::bfloat16_t(up));
        } else {
            gate = float(cutlass::half_t(gate)); up = float(cutlass::half_t(up));
        }
    }
    return (gate / (1.f + expf(-gate))) * up;
}

CUTLASS_DEVICE void output(DeviceCall const& c, int row, int col, float value) {
    int64_t index = int64_t(row) * c.out_row_stride + col;
    if (c.output_type == QKG_F32) c.output[index] = value;
    else if (c.output_type == QKG_SIMT_BF16)
        reinterpret_cast<cutlass::bfloat16_t*>(c.output)[index] = cutlass::bfloat16_t(value);
    else reinterpret_cast<cutlass::half_t*>(c.output)[index] = cutlass::half_t(value);
}

struct SimtFinish {
    template<int TileN>
    CUTLASS_DEVICE static void invalid(DeviceCall const& c, int row, int tile, int partition, int split) {
        int tid = int(threadIdx.x);
        float nan = __int_as_float(0x7fc00000);
        if (split == 1) {
            if (tid < TileN/2) output(c,row,tile*TileN/2+tid,nan);
        } else if (tid < TileN) {
            static_cast<float*>(c.workspace)[(int64_t(row)*split+partition)*c.n+tile*TileN+tid] = nan;
        }
    }
    template<int TileN,int Warps>
    CUTLASS_DEVICE static void finish(DeviceCall const& c,int row,int tile,int partition,int split,float const* partial) {
        static_assert(TileN == 32, "paired SIMT keeps the full-warp reduction order");
        int tid = int(threadIdx.x);
        if (split > 1) {
            if (tid < TileN) {
                float sum = 0.f;
                #pragma unroll
                for (int w=0;w<Warps;++w) sum += partial[w*TileN+tid];
                static_cast<float*>(c.workspace)[(int64_t(row)*split+partition)*c.n+tile*TileN+tid] = sum;
            }
        } else if (tid < TileN/2) {
            int ng = PairedN4::gate(tid);
            float gate = 0.f, up = 0.f;
            #pragma unroll
            for (int w=0;w<Warps;++w) {
                gate += partial[w*TileN+ng]; up += partial[w*TileN+ng+4];
            }
            output(c,row,tile*TileN/2+tid,activate(c,gate,up));
        }
    }
};

static __global__ void reduce_gate_up(DeviceCall c, int split) {
    int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    int n = c.n/2;
    if (i >= int64_t(c.rows)*n) return;
    int row = int(i/n), col = int(i%n), ng = PairedN4::gate(col);
    float gate = 0.f, up = 0.f;
    for (int s=0;s<split;++s) {
        auto p = static_cast<float const*>(c.workspace)+(int64_t(row)*split+s)*c.n+ng;
        gate += p[0]; up += p[4];
    }
    output(c,row,col,activate(c,gate,up));
}
}
