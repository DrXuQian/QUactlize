#include "serving.h"

// Keep symbols stable without compiling PPU1.0-only native16 TC operands into
// a PPU1.5 package. No geometry, buffer or stream can admit this path.
#if !defined(QUACTLIZE_PPU_ARCH) || QUACTLIZE_PPU_ARCH != 150
#error "The unavailable Flash-Next implementation is only for the PPU1.5 package"
#endif

namespace {
constexpr char reason[] =
    "Flash-Next native16/serving APIs currently require a ppu_10 package; "
    "PPU1.5 supports the canonical dense/MoE/FFN APIs, not these entrypoints";
}

extern "C" int quactlize_ppu_matmul_query_v1(
    qpm_problem_v1 const *, quactlize_ppu_placed_arrangement_v2 const *, qpm_sizes_v1 *sizes) {
    if (sizes) *sizes = {};
    return QPM_INVALID;
}
extern "C" int quactlize_ppu_matmul_packed_query_v1(
    qpm_problem_v1 const *p, quactlize_ppu_placed_arrangement_v2 const *a, int, qpm_sizes_v1 *sizes) {
    return quactlize_ppu_matmul_query_v1(p, a, sizes);
}
extern "C" int quactlize_ppu_matmul_prepare_v1(
    qpm_problem_v1 const *, quactlize_ppu_placed_arrangement_v2 const *,
    qpm_buffers_v1 const *, void **handle) {
    if (handle) *handle = nullptr;
    return QPM_INVALID;
}
extern "C" int quactlize_ppu_matmul_packed_prepare_v1(
    qpm_problem_v1 const *p, quactlize_ppu_placed_arrangement_v2 const *a, int,
    qpm_buffers_v1 const *b, void **handle) {
    return quactlize_ppu_matmul_prepare_v1(p, a, b, handle);
}
extern "C" int quactlize_ppu_matmul_run_v1(void *, qpm_buffers_v1 const *) { return QPM_INVALID; }
extern "C" int quactlize_ppu_matmul_dequant_v1(
    void *, qpm_buffers_v1 const *, int32_t, int32_t, int32_t, void *, uint64_t) { return QPM_INVALID; }
extern "C" void quactlize_ppu_matmul_destroy_v1(void *) {}
extern "C" char const *quactlize_ppu_matmul_error_v1() { return reason; }
extern "C" int quactlize_ppu_router_run_v1(qpm_router_v1 const *) { return QPM_INVALID; }
extern "C" int quactlize_ppu_swiglu_run_v1(qpm_swiglu_v1 const *) { return QPM_INVALID; }
extern "C" int quactlize_ppu_merge_run_v1(qpm_merge_v1 const *) { return QPM_INVALID; }
extern "C" int quactlize_serving_prepare_v1(int32_t, void *) { return QPM_INVALID; }
extern "C" char const *quactlize_serving_error_v1() { return reason; }
extern "C" int quactlize_serving_router_v2(qpm_router_v2 const *) { return QPM_INVALID; }
extern "C" int quactlize_serving_swiglu_v2(qpm_swiglu_v2 const *) { return QPM_INVALID; }
extern "C" int quactlize_serving_gather_rows_v1(qpm_rows_v1 const *) { return QPM_INVALID; }
extern "C" int quactlize_serving_merge_v1(qpm_merge_v1 const *) { return QPM_INVALID; }
