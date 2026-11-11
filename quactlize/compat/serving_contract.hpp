#pragma once
#include "serving.h"
#include "contract.hpp"
#include <cmath>

namespace quactlize::compat::serving {
struct Range {
    void const *p;
    uint64_t bytes;
};
template <size_t N> inline void writes_disjoint(Range const (&r)[N], size_t first_write) {
    for (size_t i = first_write; i < N; ++i)
        for (size_t j = 0; j < i; ++j)
            disjoint(r[i].p, r[i].bytes, r[j].p, r[j].bytes);
}
inline void typed(int v) { require(v == QPM_F16 || v == QPM_BF16, "expected F16/BF16"); }
inline void validate(qpm_router_v2 const &p) {
    require(p.version == 2 && p.size == sizeof(p) && !p.reserved, "invalid router v2 ABI");
    require(p.stream, "serving requires explicit stream");
    require(p.capacity >= 0 && p.experts > 0 && p.experts <= 1024 && p.topk > 0 && p.topk <= 32 &&
                p.topk <= p.experts,
            "invalid router shape");
    require(p.dtype == QPM_BF16 || p.dtype == QPM_F32, "router logits must be BF16/F32");
    require(p.score_func == QPM_SCORE_SOFTMAX && p.renormalize == 1,
            "router supports selected-logit softmax with renormalize=True only");
    require(std::isfinite(p.multiplier) && p.multiplier >= 0, "invalid router multiplier");
    auto l = extent(p.capacity, p.logits_stride, p.experts, p.dtype == QPM_F32 ? 4 : 2);
    auto s = extent(p.capacity, p.selected_stride, p.topk, 4);
    auto i = extent(p.capacity, p.ids_stride, p.topk, 4);
    auto w = extent(p.capacity, p.weights_stride, p.topk, 4);
    span(p.logits, l, p.logits_bytes, p.dtype == QPM_F32 ? 4 : 2);
    span(p.selected_logits, s, p.selected_bytes, 4);
    span(p.ids, i, p.ids_bytes, 4);
    span(p.weights, w, p.weights_bytes, 4);
    span(p.device_error, 4, 4, 4);
    auto c = p.num_valid_tokens ? 4u : 0u;
    span(p.num_valid_tokens, c, c, 4);
    Range r[] = {{p.logits, l}, {p.num_valid_tokens, c}, {p.selected_logits, s},
                 {p.ids, i},    {p.weights, w},          {p.device_error, 4}};
    writes_disjoint(r, 2);
}
inline void validate(qpm_swiglu_v2 const &p) {
    require(p.version == 2 && p.size == sizeof(p) && !p.reserved, "invalid SwiGLU v2 ABI");
    require(p.stream, "serving requires explicit stream");
    typed(p.dtype);
    require(p.rows >= 0 && p.hidden > 0 && p.hidden <= INT_MAX / 2 && (p.merged == 0 || p.merged == 1) &&
                (p.round_silu == 0 || p.round_silu == 1),
            "invalid SwiGLU geometry/rounding");
    require(!p.merged || !p.up, "merged activation rows must not carry separate up");
    auto g = extent(p.rows, p.gate_stride, p.hidden * (p.merged ? 2 : 1), 2);
    auto u = p.merged ? 0 : extent(p.rows, p.up_stride, p.hidden, 2);
    auto o = extent(p.rows, p.output_stride, p.hidden, 2);
    span(p.gate, g, p.gate_bytes, 2);
    span(p.up, u, p.up_bytes, 2);
    span(p.output, o, p.output_bytes, 2);
    Range r[] = {{p.gate, g}, {p.up, u}, {p.output, o}};
    writes_disjoint(r, 2);
}
inline uint64_t row_bytes(int q, int columns) {
    int block = q == 42 ? 64 : 32;
    require((q == 8 || q == 20 || q == 42) && columns > 0 && columns % block == 0,
            "row lookup requires raw Q8_0/IQ4_NL (K%32) or Q2_0 (K%64)");
    return mul(columns / block, q == 8 ? 34 : 18);
}
inline void validate(qpm_rows_v1 const &p) {
    require(p.version == 1 && p.size == sizeof(p), "invalid row lookup ABI");
    require(p.stream, "serving requires explicit stream");
    typed(p.dtype);
    require(p.capacity >= 0 && p.table_rows >= 0, "invalid lookup capacity/table rows");
    auto r = mul(p.table_rows, row_bytes(p.qtype, p.columns));
    auto i = mul(p.capacity, 8), o = mul(mul(p.capacity, p.columns), 2);
    auto c = p.num_rows ? 4u : 0u;
    span(p.raw, r, p.raw_bytes, 1);
    span(p.ids, i, p.ids_bytes, 8);
    span(p.output, o, p.output_bytes, 2);
    span(p.num_rows, c, c, 4);
    span(p.device_error, 4, 4, 4);
    Range ranges[] = {{p.raw, r}, {p.ids, i}, {p.num_rows, c}, {p.output, o}, {p.device_error, 4}};
    writes_disjoint(ranges, 3);
}
inline void validate(qpm_merge_v1 const &p) {
    require(p.version == 1 && p.size == sizeof(p), "invalid merge ABI");
    require(p.stream, "serving requires explicit stream");
    typed(p.dtype);
    require(p.m >= 0 && p.topk > 0 && p.topk <= 32 && p.n > 0, "invalid merge geometry");
    auto i = extent(mul(p.m, p.topk), p.input_stride, p.n, 2);
    auto w = extent(p.m, p.weights_stride, p.topk, 4), o = extent(p.m, p.output_stride, p.n, 4);
    span(p.input, i, i, 2);
    span(p.weights, w, w, 4);
    span(p.output, o, o, 4);
    Range r[] = {{p.input, i}, {p.weights, w}, {p.output, o}};
    writes_disjoint(r, 2);
}
} // namespace quactlize::compat::serving
