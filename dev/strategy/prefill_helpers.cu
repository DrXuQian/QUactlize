// Experiment-only launch wrappers around unchanged production adapters.
#include "quactlize/dense/copy.cu"
#include "quactlize/prefill/stages.cu"
#include "dev/strategy/prefill_capture.hpp"

extern "C" int strategy_prefill_io(int bf16, int input, void const* src, void* dst,
        int rows, int width, int64_t stride, int const* map, int source_rows,
        int* error, void* stream) {
    if (!bf16) {
        if (map) return 1;
        return quactlize_dense_convert_f16(input, src, dst, rows, width, stride, 0, stream);
    }
    auto s = static_cast<hggcStream_t>(stream);
    if (input) quactlize::prefill::gather_bf16<<<rows,256,0,s>>>(
        static_cast<float const*>(src), static_cast<__ppu_bfloat16*>(dst), map,
        width, stride, source_rows, error);
    else quactlize::prefill::finish_bf16<<<rows,256,0,s>>>(
        static_cast<__ppu_bfloat16 const*>(src), static_cast<float*>(dst), map,
        width, stride, rows, error);
    return int(hggcGetLastError());
}

extern "C" int strategy_prefill_active(int const* offsets, int* rows, int* ids,
        int* count, int* error, int m, void* stream) {
    quactlize::prefill::active_experts<<<1,256,0,static_cast<hggcStream_t>(stream)>>>(
        offsets, rows, ids, count, error, m);
    return int(hggcGetLastError());
}

// Clone before PyTorch ends capture (older versions destroy the original
// graph at capture_end). The Python CUDAGraph owner retains its allocation
// pool until all extracted graphs are destroyed. No private Torch ABI.
// CUDA's graph_out contract permits graph operations except destroy/removal
// during capture; PPU runtime support still requires the device execution gate.
// https://docs.nvidia.com/cuda/archive/12.4.1/cuda-runtime-api/group__CUDART__STREAM.html
extern "C" int strategy_prefill_clone_capture(void* stream, void** result) {
    return strategy_prefill::clone_capture(stream, result);
}
