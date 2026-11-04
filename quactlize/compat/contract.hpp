#pragma once
#include "api.h"
#include <algorithm>
#include <climits>
#include <stdexcept>

namespace quactlize::compat {
inline void require(bool ok, char const *message) {
    if (!ok)
        throw std::invalid_argument(message);
}
inline uint64_t mul(uint64_t a, uint64_t b) {
    require(!b || a <= uint64_t(INT64_MAX) / b, "byte extent overflow");
    return a * b;
}
inline uint64_t add(uint64_t a, uint64_t b) {
    require(a <= uint64_t(INT64_MAX) - b, "byte extent overflow");
    return a + b;
}
inline uint64_t align(uint64_t n) { return add(n, 255) & ~uint64_t(255); }
inline uint64_t extent(int64_t rows, int64_t stride, int64_t columns, int bytes) {
    require(rows >= 0 && columns > 0 && stride >= columns, "invalid stride/extent");
    return rows ? mul(add(mul(rows - 1, stride), columns), bytes) : 0;
}
inline int block_elements(int q) {
    require(q == 7 || q == 8 || (q >= 10 && q <= 14), "unsupported GGUF qtype (Q5_1 is 7, not Q5_K)");
    return q < 10 ? 32 : 256;
}
inline int block_bytes(int q) {
    switch (q) {
    case 7:
        return 24;
    case 8:
        return 34;
    case 10:
        return 84;
    case 11:
        return 110;
    case 12:
        return 144;
    case 13:
        return 176;
    case 14:
        return 210;
    default:
        throw std::invalid_argument("unsupported GGUF qtype");
    }
}
struct Layout {
    qpm_sizes_v1 sizes{};
    uint64_t counts = 0, cursors = 0, maps = 0, inverse = 0, a = 0, c = 0, w = 0, metadata = 0;
    int rows = 0, padded_n = 0;
};

// Shared by host tests and the device implementation. Shape checks cannot be
// bypassed by prepare/run; no device pointer is dereferenced here.
inline Layout layout(qpm_problem_v1 const &p, quactlize_ppu_placed_arrangement_v2 const *arr) {
    require(p.version == 1 && p.size == sizeof(p) && !p.reserved, "invalid matmul problem ABI");
    require(p.m >= 0 && p.n > 0 && p.n <= INT_MAX - 7 && p.k > 0 && p.device >= 0,
            "invalid matrix shape/device");
    require(p.dtype == QPM_F16 || p.dtype == QPM_BF16, "expected native F16 or BF16");
    require(p.mode == QPM_DENSE || p.mode == QPM_INDEXED, "invalid matmul mode");
    require(p.experts > 0 && p.experts <= 1024 && p.topk > 0 && p.topk <= 32 && p.topk <= p.experts,
            "experts/topk out of range");
    require(p.channels == 1 || p.channels == p.topk, "channels must be 1 or topk");
    require(p.mode != QPM_DENSE || (p.experts == 1 && p.topk == 1 && p.channels == 1),
            "dense requires one expert");
    require(p.k % block_elements(p.qtype) == 0, "incomplete GGUF block in K");
    Layout l;
    auto &s = l.sizes;
    s.version = 1;
    s.size = sizeof(s);
    uint64_t count = mul(mul(p.experts, p.n), p.k);
    if (p.layout == QPM_RAW_GGUF) {
        require(!arr, "raw GGUF must not carry a K-pack arrangement");
        s.raw_bytes = mul(count / block_elements(p.qtype), block_bytes(p.qtype));
    } else {
        require(p.layout == QPM_CANONICAL_KPACK && arr, "explicit canonical arrangement required");
        quactlize_ppu_kpack_sizes_v1 packed{};
        require(quactlize_ppu_kpack_sizes_for_arrangement_v1(p.n, p.k, p.experts, p.qtype, arr, &packed) == 0,
                "canonical producer rejects this format/shape/descriptor");
        s.low_bytes = packed.low_bytes;
        s.high_bytes = packed.high_bytes;
        s.units_bytes = packed.units_bytes;
    }
    require(p.input_row_stride >= p.k && p.input_token_stride > 0, "invalid input strides");
    uint64_t token_span = add(mul(p.channels - 1, p.input_row_stride), p.k);
    require(uint64_t(p.input_token_stride) >= token_span, "overlapping input tokens");
    s.input_bytes = extent(p.m, p.input_token_stride, token_span, 2);
    uint64_t rows = mul(p.m, p.topk);
    l.padded_n = (p.n + 7) / 8 * 8;
    require(rows <= INT_MAX && mul(rows, std::max(p.k, l.padded_n)) <= INT_MAX,
            "routed tensor exceeds int32 GEMM index domain");
    l.rows = int(rows);
    s.output_bytes = extent(rows, p.output_stride, p.n, 2);
    if (p.mode == QPM_INDEXED)
        s.ids_bytes = extent(p.m, p.ids_stride, p.topk, 4);
    else
        require(p.ids_stride == 0, "dense does not accept an IDs stride");
    s.route = p.m <= 8 ? QPM_SIMT : QPM_GROUPED_TC;
    if (!rows || s.route == QPM_SIMT)
        return l;
    s.expert_tile = std::min(p.experts, 4);
    s.column_tile = std::min(l.padded_n, 512);
    uint64_t next = 0;
    auto reserve = [&](uint64_t bytes) {
        auto offset = next;
        next = add(next, align(bytes));
        return offset;
    };
    l.counts = reserve(mul(p.experts + 1, 4));
    l.cursors = reserve(mul(p.experts, 4));
    l.maps = reserve(mul(rows, 4));
    l.inverse = reserve(mul(rows, 4));
    s.routing_bytes = next;
    // An extra M tile protects the existing AIU grouped input load's tail.
    s.grouped_input_bytes = mul(mul(add(rows, 32), p.k), 2);
    s.grouped_output_bytes = mul(mul(rows, l.padded_n), 2);
    s.expanded_weight_bytes = mul(mul(mul(s.expert_tile, s.column_tile), p.k), 2);
    l.a = reserve(s.grouped_input_bytes);
    l.c = reserve(s.grouped_output_bytes);
    l.w = reserve(s.expanded_weight_bytes);
    l.metadata = reserve(4096);
    s.workspace_bytes = next;
    return l;
}

inline void span(void const *pointer, uint64_t bytes, uint64_t capacity, int alignment) {
    require(capacity >= bytes, "buffer capacity too small");
    if (!bytes)
        return;
    auto p = reinterpret_cast<uintptr_t>(pointer);
    require(p && p % alignment == 0 && bytes <= UINTPTR_MAX - p, "null, unaligned or overflowing buffer");
}
inline void disjoint(void const *a, uint64_t na, void const *b, uint64_t nb) {
    auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    require(!na || !nb || x + na <= y || y + nb <= x, "overlapping live buffers");
}
inline void weights(qpm_sizes_v1 const &s, qpm_buffers_v1 const &b) {
    require(b.version == 1 && b.size == sizeof(b), "invalid matmul buffers ABI");
    span(b.raw, s.raw_bytes, b.raw_bytes, 1);
    span(b.low, s.low_bytes, b.low_bytes, 16);
    span(b.high, s.high_bytes, b.high_bytes, 16);
    span(b.units, s.units_bytes, b.units_bytes, 16);
    require((!b.raw) == (!s.raw_bytes) && (!b.low) == (!s.low_bytes) && (!b.high) == (!s.high_bytes) &&
                (!b.units) == (!s.units_bytes),
            "unused weight plane must be NULL");
}
inline void validate(qpm_problem_v1 const &p, Layout const &l, qpm_buffers_v1 const &b) {
    auto const &s = l.sizes;
    weights(s, b);
    span(b.input, s.input_bytes, b.input_bytes, 2);
    span(b.output, s.output_bytes, b.output_bytes, 2);
    span(b.ids, s.ids_bytes, b.ids_bytes, 4);
    span(b.workspace, s.workspace_bytes, b.workspace_bytes, 256);
    span(b.device_error, 4, 4, 4);
    require(p.mode == QPM_INDEXED || !b.ids, "dense IDs must be NULL");
    void const *pointers[] = {b.raw, b.low,    b.high,      b.units,       b.input,
                              b.ids, b.output, b.workspace, b.device_error};
    uint64_t bytes[] = {s.raw_bytes,    s.low_bytes,       s.high_bytes,
                        s.units_bytes,  s.input_bytes,     s.ids_bytes,
                        s.output_bytes, s.workspace_bytes, 4};
    // Read-only aliases are harmless; writes may alias neither inputs nor each other.
    for (int i = 6; i < 9; ++i)
        for (int j = 0; j < i; ++j)
            disjoint(pointers[i], bytes[i], pointers[j], bytes[j]);
}
} // namespace quactlize::compat
