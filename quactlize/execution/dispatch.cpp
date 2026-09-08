#include "validation.hpp"

extern "C" {
int qkg_launch_10(qkg_call_v1 const&, qkg_config_v1 const&);
int qkg_launch_11(qkg_call_v1 const&, qkg_config_v1 const&);
int qkg_launch_12(qkg_call_v1 const&, qkg_config_v1 const&);
int qkg_launch_13(qkg_call_v1 const&, qkg_config_v1 const&);
int qkg_launch_14(qkg_call_v1 const&, qkg_config_v1 const&);
}

extern "C" int quactlize_kpack_gemv_query_v1(qkg_call_v1 const* c, qkg_config_v1 const* f,
        quactlize_ppu_placed_arrangement_v2 const* arrangement, qkg_sizes_v1* out) {
    if (!c || !f || !out) return QKG_INVALID;
    qkg_sizes_v1 result{};
    int const rc = quactlize::execution::query(*c, *f, arrangement, result);
    if (!rc) *out = result;
    return rc;
}

extern "C" int quactlize_kpack_gemv_run_v1(qkg_call_v1 const* c, qkg_config_v1 const* f,
        quactlize_ppu_placed_arrangement_v2 const* arrangement) {
    using namespace quactlize::execution;
    qkg_sizes_v1 s{};
    int const rc = quactlize_kpack_gemv_query_v1(c, f, arrangement, &s);
    if (rc) return rc;
    if (!c->a || !c->low || !c->units || !c->output ||
        (s.high_bytes != 0) != (c->high != nullptr) ||
        (c->mode == QKG_GROUPED) != (c->offsets != nullptr) ||
        (c->mode == QKG_INDEXED) != (c->ids != nullptr) ||
        ((uintptr_t(c->a) | uintptr_t(c->low) | uintptr_t(c->high)) & 1) ||
        ((uintptr_t(c->output) | uintptr_t(c->workspace) | uintptr_t(c->ids) | uintptr_t(c->offsets)) & 3) ||
        (c->input_type == QKG_F32 && (uintptr_t(c->a) & 3))) return QKG_INVALID;
    if (s.workspace_bytes && (!c->workspace || c->workspace_bytes < s.workspace_bytes)) return QKG_CAPACITY;
    uint64_t const a_elements = c->mode == QKG_INDEXED
        ? uint64_t(c->rows / c->topk) * c->a_token_stride
        : uint64_t(c->rows) * c->a_row_stride;
    uint64_t const a_bytes = a_elements * (c->input_type == QKG_F16 ? 2 : 4);
    uintptr_t const p[] = {uintptr_t(c->low),uintptr_t(c->high),uintptr_t(c->units),uintptr_t(c->a),
        uintptr_t(c->ids),uintptr_t(c->offsets),uintptr_t(c->output),uintptr_t(c->workspace)};
    uint64_t const b[] = {s.low_bytes,s.high_bytes,s.units_bytes,a_bytes,
        c->ids ? uint64_t(c->rows / c->topk) * c->ids_stride * 4 : 0,
        c->offsets ? (uint64_t(c->experts) + 1) * 4 : 0,
        uint64_t(c->rows) * c->out_row_stride * 4,s.workspace_bytes};
    for (int i = 0; i < 8; ++i) {
        if (!span(p[i], b[i])) return QKG_OVERFLOW;
        for (int j = 0; j < i; ++j)
            if (i >= 6 && overlap(p[i], b[i], p[j], b[j])) return QKG_INVALID;
    }
    using Launch = int(*)(qkg_call_v1 const&, qkg_config_v1 const&);
    static Launch const launch[] = {qkg_launch_10,qkg_launch_11,qkg_launch_12,qkg_launch_13,qkg_launch_14};
    return launch[c->qtype - 10](*c, *f);
}
