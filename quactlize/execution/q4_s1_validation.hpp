#pragma once
#include "q4_s1_api.h"
#include "validation.hpp"

namespace quactlize::execution::q4_s1 {

inline int tile_n(qkg_q4_s1_config_v1 const& f) {
    if (f.version != 1 || f.size != sizeof(f) || f.warps < 1 || f.warps > 32)
        return 0;
    if (f.reader == QKG_Q4_META)
        return (f.variant == 0 || f.variant == 1) && f.values == 1 ? 8 : 0;
    if (f.reader == QKG_Q4_MEDIUM)
        return f.variant >= 0 && f.variant <= 3 && (f.values == 2 || f.values == 4)
            ? 4 * f.values : 0;
    if (f.reader == QKG_Q4_REUSE)
        return (f.variant == 6 || f.variant == 7) && (f.values == 4 || f.values == 8)
            ? 4 * f.values : 0;
    return 0;
}

inline int validate(qkg_call_v1 const& c, qkg_q4_s1_config_v1 const& f,
        quactlize_ppu_placed_arrangement_v2 const* arrangement, qkg_sizes_v1& out) {
    int const tile = tile_n(f);
    if (!tile) return QKG_INVALID;
    if (c.qtype != 12) return QKG_FORMAT;
    // Reuse the established call-shape/overflow contract, not its selector.
    qkg_config_v1 const shape_only{1, sizeof(qkg_config_v1), 16, 4, 1};
    int const rc = execution::query(c, shape_only, arrangement, out);
    if (rc) return rc;
    int const a_alignment = c.input_type == QKG_F16 ? 8 : 4;
    if (c.a_row_stride % a_alignment ||
        (c.mode == QKG_INDEXED && c.a_token_stride % a_alignment)) return QKG_INVALID;
    if (uint64_t(c.rows) * (c.n / tile) > INT32_MAX) return QKG_OVERFLOW;
    return QKG_OK;
}

inline int buffers(qkg_call_v1 const& c, qkg_sizes_v1 const& s) {
    if (!c.a || !c.low || !c.units || !c.output || c.high ||
        (c.mode == QKG_GROUPED) != (c.offsets != nullptr) ||
        (c.mode == QKG_INDEXED) != (c.ids != nullptr) ||
        ((uintptr_t(c.a) | uintptr_t(c.low) | uintptr_t(c.units)) & 15) ||
        ((uintptr_t(c.output) | uintptr_t(c.ids) | uintptr_t(c.offsets)) & 3))
        return QKG_INVALID;
    uint64_t const a_elements = c.mode == QKG_INDEXED
        ? uint64_t(c.rows/c.topk-1)*c.a_token_stride + uint64_t(c.channels-1)*c.a_row_stride + c.k
        : uint64_t(c.rows-1)*c.a_row_stride + c.k;
    uintptr_t const p[] = {uintptr_t(c.low), uintptr_t(c.units), uintptr_t(c.a),
        uintptr_t(c.ids), uintptr_t(c.offsets), uintptr_t(c.output)};
    uint64_t const b[] = {s.low_bytes, s.units_bytes, a_elements*(c.input_type==QKG_F16 ? 2 : 4),
        c.ids ? (uint64_t(c.rows/c.topk-1)*c.ids_stride+c.topk)*4 : 0,
        c.offsets ? (uint64_t(c.experts)+1)*4 : 0,
        (uint64_t(c.rows-1)*c.out_row_stride+c.n)*4};
    for (int i=0; i<6; ++i) if (!execution::span(p[i], b[i])) return QKG_OVERFLOW;
    for (int i=0; i<5; ++i)
        if (execution::overlap(p[5], b[5], p[i], b[i])) return QKG_INVALID;
    return QKG_OK;
}

// Host/device callable row contract, shared by the actual kernel and tests.
// ids/offsets point to device memory in production; no host probing occurs.
#if defined(__CUDACC__) || defined(__HGGC__)
#define QKG_S1_HD __host__ __device__
#else
#define QKG_S1_HD
#endif
struct Row { int expert; int64_t a, output; };
QKG_S1_HD inline Row locate(qkg_call_v1 const& c, int row) {
    int expert=0;
    int64_t a=int64_t(row)*c.a_row_stride;
    if (c.mode==QKG_INDEXED) {
        int const token=row/c.topk, slot=row%c.topk;
        expert=c.ids[int64_t(token)*c.ids_stride+slot];
        a=int64_t(token)*c.a_token_stride+(slot%c.channels)*c.a_row_stride;
    } else if (c.mode==QKG_GROUPED) {
        int lo=0, hi=c.experts;
        while (lo+1<hi) {
            int const mid=lo+(hi-lo)/2;
            if (c.offsets[mid]<=row) lo=mid; else hi=mid;
        }
        expert=lo;
    }
    return {expert,a,int64_t(row)*c.out_row_stride};
}
#undef QKG_S1_HD
} // namespace quactlize::execution::q4_s1
