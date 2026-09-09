// Development-only launch comparison. Both arms instantiate the production
// reduction bodies; no copied arithmetic and no PPU runtime emulation.
#include <cuda_runtime.h>
#include <hggc_runtime.h>
#include "actlize_extensions/cutlass/gemm/device/ppu_mixed_input_splitk_parallel.hpp"

namespace sp = cutlass::gemm::device::splitk_parallel;

template<int EPA, int S>
int flat(float const* partial, cutlass::half_t* output, int64_t count, cudaStream_t stream) {
    using Kernel = sp::PpuMixedInputSplitKParallelM1FastReductionKernel<EPA,S>;
    // Actual vector alignment, including grouped workspace offsets. This is
    // an experiment, not a relaxation of the production dispatcher's guards.
    if (count <= 0 || count % (32 * EPA) ||
        uintptr_t(partial) % (EPA * sizeof(float)) ||
        uintptr_t(output) % (EPA * sizeof(cutlass::half_t))) return 24;
    typename Kernel::Params params{partial, output, count};
    cutlass::Kernel<Kernel><<<Kernel::grid_shape(count), Kernel::block_shape(), 0, stream>>>(params);
    return int(cudaGetLastError());
}

template<int EPA>
int flat_s(float const* p, cutlass::half_t* o, int64_t count, int s, cudaStream_t stream) {
    switch (s) {
        case 2: return flat<EPA,2>(p,o,count,stream);
        case 4: return flat<EPA,4>(p,o,count,stream);
        case 8: return flat<EPA,8>(p,o,count,stream);
        default: return 24;
    }
}

extern "C" int qkg_cuda_reduce(float const* p, cutlass::half_t* o,
        int m, int n, int stride, int s, int arm, void* stream) {
    using Generic = sp::PpuMixedInputSplitKParallelReduction<8>;
    Generic::Arguments args{m,n,s,p,size_t(m)*n*s*sizeof(float),o,stride};
    Generic reducer;
    if (reducer.initialize(args) != cutlass::Status::kSuccess) return 24;
    auto st = static_cast<cudaStream_t>(stream);
    if (arm == 0) return int(reducer.run(st));
    if (arm == 8) {
        sp::PpuMixedInputSplitKParallelCompactReduction<2> compact;
        auto status=compact.initialize(args);
        return int(status == cutlass::Status::kSuccess ? compact.run(st) : status);
    }
    if (stride != n) return 24;
    switch (arm) {
        case 1: return flat_s<1>(p,o,int64_t(m)*n,s,st);
        case 2: return flat_s<2>(p,o,int64_t(m)*n,s,st);
        case 4: return flat_s<4>(p,o,int64_t(m)*n,s,st);
        default: return 24;
    }
}

extern "C" int qkg_cuda_reduce_fast(float const* p, cutlass::half_t* o,
        int m,int n,int stride,int s) {
    sp::PpuMixedInputSplitKParallelCompactReduction<2> compact;
    typename decltype(compact)::Arguments args{m,n,s,p,size_t(m)*n*s*4,o,stride};
    if (compact.initialize(args)!=cutlass::Status::kSuccess) return -1;
    return int(compact.fast_path_selected_for_diagnostics());
}

__global__ void empty() {}
extern "C" int qkg_cuda_empty(void* stream) {
    empty<<<1,32,0,static_cast<cudaStream_t>(stream)>>>();
    return int(cudaGetLastError());
}
