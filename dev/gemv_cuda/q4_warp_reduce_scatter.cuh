#pragma once
// Reduce K and scatter N ownership at the same time. Each exchange halves
// the number of live output columns, instead of doing an all-reduce for each.
template<int Count, int Stride, int Width>
__device__ __forceinline__ float q4_reduce_scatter_steps(float (&value)[Width], int lane) {
    if constexpr (Count > 1) {
        bool const odd = (lane & Stride) != 0;
        #pragma unroll
        for (int i = 0; i < Count / 2; ++i) {
            float keep = odd ? value[2*i+1] : value[2*i];
            float send = odd ? value[2*i] : value[2*i+1];
            value[i] = keep + __shfl_xor_sync(0xffffffffu, send, Stride);
        }
        return q4_reduce_scatter_steps<Count/2,Stride*2>(value, lane);
    } else {
        float v = value[0];
        #pragma unroll
        for (int d = Stride; d < 32; d *= 2) v += __shfl_xor_sync(0xffffffffu, v, d);
        return v;
    }
}
