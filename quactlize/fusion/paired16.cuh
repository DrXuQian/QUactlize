#pragma once
#include "store.cuh"

namespace quactlize::fusion {
// Keep the confirmed paired-16 sequential inter-warp reduction order.
struct Paired16Finish : SimtFinish {
    template<int TileN,int Warps>
    CUTLASS_DEVICE static void finish(DeviceCall const& c,int row,int tile,
        int partition,int split,float const* partial) {
        static_assert(TileN==16,"half-width paired tile");
        int tid=int(threadIdx.x);
        if(tid<8) {
            int ng=PairedN4::gate(tid);
            float gate=0.f,up=0.f;
            #pragma unroll
            for(int w=0;w<Warps;++w) {
                gate+=partial[w*16+ng];up+=partial[w*16+ng+4];
            }
            output(c,row,tile*8+tid,activate(c,gate,up));
        }
    }
};
}
