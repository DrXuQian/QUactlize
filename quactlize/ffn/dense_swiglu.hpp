#pragma once
#include <cstdint>

namespace quactlize::ffn {
// Private dense FFN endpoint: projected F32 inputs, F32 activation output.
// The public FFN caller owns validation and disjoint scratch. Merged GateUp
// passes up=gate+hidden with the same row stride; no expert maps or rounding.
inline bool dense_swiglu_vectorized(float const* gate,float const* up,float const* output,
                                   int hidden,int64_t gate_stride,int64_t up_stride,int64_t output_stride) {
    return hidden%4==0 && gate_stride%4==0 && up_stride%4==0 && output_stride%4==0 &&
        (reinterpret_cast<uintptr_t>(gate)|reinterpret_cast<uintptr_t>(up)|
         reinterpret_cast<uintptr_t>(output))%16==0;
}
int dense_swiglu(float const* gate,float const* up,float* output,int rows,int hidden,
                int64_t gate_stride,int64_t up_stride,int64_t output_stride,void* stream);
}
