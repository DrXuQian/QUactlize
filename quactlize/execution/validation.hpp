#pragma once
#include "api.h"
#include "ppu_placed_arrangement.hpp"
#include "q8_kpack2.hpp"
#include <limits>

namespace quactlize::execution {

inline int sizes(int q, int n, int k, int experts,
                 quactlize_ppu_placed_arrangement_v2 const* arrangement, qkg_sizes_v1& out) {
    if (q != 8 && (q < 10 || q > 14)) return QKG_FORMAT;
    if (q==8 ? !q8_kpack2::matches(arrangement) :
        !ppu_arrangements::matches_canonical_kpack(arrangement, q)) return QKG_ARRANGEMENT;
    if (n <= 0 || k <= 0 || experts <= 0 || n % 256 ||
        k % ((q == 11 || q == 14) ? 512 : 256)) return QKG_SHAPE;
    uint64_t const nk = uint64_t(n) * k;
    if (nk > uint64_t(INT64_MAX) / uint64_t(experts)) return QKG_OVERFLOW;
    uint64_t const count = nk * experts;
    if (q==8) {
        // Resident original FP16 d; no expanded scale/zero workspace.
        out={count,0,count/32*2,0,0};
        return QKG_OK;
    }
    constexpr int low[] = {2,2,4,4,4}, high[] = {0,1,0,1,2};
    constexpr int unit[] = {20,14,16,16,18}, group[] = {16,16,32,32,16};
    out = {count / 8 * low[q-10], count / 8 * high[q-10],
           count / 256 * unit[q-10], count / group[q-10] * 2, 0};
    return QKG_OK;
}

inline int query(qkg_call_v1 const& c, qkg_config_v1 const& f,
                 quactlize_ppu_placed_arrangement_v2 const* arrangement, qkg_sizes_v1& out,
                 bool pair = false) {
    if (c.version != 1 || c.size != sizeof(c) || f.version != 1 || f.size != sizeof(f) ||
        c.rows <= 0 || c.mode < QKG_DENSE || c.mode > QKG_INDEXED ||
        c.input_type < QKG_F16 || c.input_type > QKG_F32 ||
        (f.columns != 16 && f.columns != 32) ||
        (f.warps != 4 && f.warps != 8 && !(pair && f.warps == 2)) ||
        (f.split != 1 && f.split != 4 && !(pair && (f.split == 2 || f.split == 8))) ||
        c.a_row_stride < c.k || c.out_row_stride < c.n)
        return QKG_INVALID;
    int const rc = sizes(c.qtype, c.n, c.k, c.experts, arrangement, out);
    if (rc) return rc;
    if (c.mode == QKG_INDEXED) {
        if (c.topk <= 0 || c.topk > c.experts || c.channels <= 0 ||
            c.channels > c.topk || c.topk % c.channels || c.rows % c.topk ||
            c.ids_stride < c.topk || c.a_row_stride > INT64_MAX / c.channels ||
            c.a_token_stride < c.a_row_stride * c.channels) return QKG_INVALID;
    } else if (c.channels != 1 || c.topk != 1 || (c.mode == QKG_DENSE && c.experts != 1))
        return QKG_INVALID;
    uint64_t const a_extent = uint64_t(c.mode == QKG_INDEXED ? c.a_token_stride : c.a_row_stride);
    uint64_t const limit = uint64_t(INT64_MAX) / uint64_t(c.rows) / 4;
    if (a_extent > limit || uint64_t(c.out_row_stride) > limit ||
        (c.mode == QKG_INDEXED && uint64_t(c.ids_stride) > limit)) return QKG_OVERFLOW;
    // Flatten row/split into grid.x to avoid the grid.y=65535 ceiling.
    uint64_t const blocks = uint64_t(c.rows) * f.split * (c.n / f.columns);
    if (blocks > INT32_MAX) return QKG_OVERFLOW;
    if (f.split > 1) out.workspace_bytes = uint64_t(c.rows) * c.n * f.split * sizeof(float);
    return QKG_OK;
}

inline bool span(uintptr_t pointer, uint64_t bytes) {
    return bytes == 0 || (pointer && bytes <= UINTPTR_MAX - pointer);
}
inline bool overlap(uintptr_t a, uint64_t an, uintptr_t b, uint64_t bn) {
    return an && bn && a < b + bn && b < a + an;
}
} // namespace quactlize::execution
