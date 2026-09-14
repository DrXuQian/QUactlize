// Diagnostic carrier of the production reducers; no replacement kernel body.
#include "hggc_runtime.h"
#include "actlize_extensions/cutlass/gemm/device/ppu_mixed_input_splitk_parallel.hpp"
#include <memory>

namespace {
namespace sk = cutlass::gemm::device::splitk_parallel;
using Dense = sk::PpuMixedInputSplitKParallelM1FastReduction<2>;
using Grouped = sk::PpuMixedInputSplitKParallelCompactReduction<2>;
struct Handle {
    Dense dense;
    Grouped grouped;
    bool compact;
};
}

extern "C" int prefill_reducer_prepare(int compact, int m, int n, int split,
    float const* partial, uint64_t bytes, cutlass::half_t* output, void** result) {
    if (!result || (compact != 0 && compact != 1)) return 1;
    *result = nullptr;
    auto h = std::make_unique<Handle>();
    h->compact = compact != 0;
    Dense::Arguments args{m, n, split, partial, size_t(bytes), output, n};
    auto status = h->compact ? h->grouped.initialize(args) : h->dense.initialize(args);
    if (status != cutlass::Status::kSuccess) return 2;
    *result = h.release();
    return 0;
}
extern "C" int prefill_reducer_run(void* handle, void* stream) {
    if (!handle) return 1;
    auto& h = *static_cast<Handle*>(handle);
    auto s = static_cast<hggcStream_t>(stream);
    return (h.compact ? h.grouped.run(s) : h.dense.run(s)) == cutlass::Status::kSuccess ? 0 : 3;
}
extern "C" int prefill_reducer_fast(void* handle) {
    if (!handle) return -1;
    auto& h = *static_cast<Handle*>(handle);
    return h.compact ? h.grouped.fast_path_selected_for_diagnostics()
                     : h.dense.fast_path_selected_for_diagnostics();
}
extern "C" void prefill_reducer_destroy(void* handle) { delete static_cast<Handle*>(handle); }
