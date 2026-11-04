#include "contract.hpp"
#include "reader.cuh"
#include "grouped.cuh"
#include <memory>
#include <string>

namespace quactlize::compat {
thread_local std::string last_error;
template <class F> int guarded(F f) {
    try {
        f();
        last_error.clear();
        return QPM_OK;
    } catch (std::invalid_argument const &e) {
        last_error = e.what();
        return QPM_INVALID;
    } catch (std::exception const &e) {
        last_error = e.what();
        return QPM_RUNTIME;
    }
}
void checked(hggcError_t code) {
    if (code != hggcSuccess)
        throw std::runtime_error(hggcGetErrorString(code));
}
void success(int code, char const *name) {
    if (code)
        throw std::runtime_error(std::string(name) + ": status=" + std::to_string(code));
}
struct Prepared {
    qpm_problem_v1 p;
    Layout l;
    int cus = 0;
};
Weight weight(Prepared const &h, qpm_buffers_v1 const &b) {
    auto const &s = h.l.sizes;
    int e = h.p.experts;
    return {static_cast<uint8_t const *>(b.raw),
            static_cast<uint8_t const *>(b.low),
            static_cast<uint8_t const *>(b.high),
            static_cast<uint8_t const *>(b.units),
            s.raw_bytes / e,
            s.low_bytes / e,
            s.high_bytes / e,
            s.units_bytes / e,
            h.p.n,
            h.p.k};
}
int grid(uint64_t count) { return int(std::min<uint64_t>((count + 255) / 256, 4096)); }
__device__ int expert(qpm_problem_v1 const &p, qpm_buffers_v1 const &b, int row) {
    return p.mode == QPM_DENSE ? 0 : b.ids[int64_t(row / p.topk) * p.ids_stride + row % p.topk];
}
__global__ void count_rows(qpm_problem_v1 p, qpm_buffers_v1 b, int rows, int *counts) {
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < rows; row += blockDim.x * gridDim.x) {
        int e = expert(p, b, row);
        if (e >= 0 && e < p.experts)
            atomicAdd(counts + e, 1);
        else if (e != -1)
            atomicOr(b.device_error, QPM_BAD_ID);
    }
}
__global__ void prefix_rows(int *counts, int *cursors, int experts) {
    if (threadIdx.x)
        return;
    int sum = 0;
    for (int e = 0; e < experts; ++e) {
        int n = counts[e];
        counts[e] = sum;
        cursors[e] = 0;
        sum += n;
    }
    counts[experts] = sum;
}
__global__ void map_rows(qpm_problem_v1 p, qpm_buffers_v1 b, int rows, int const *offsets, int *cursors,
                         int *maps, int *inverse) {
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < rows; row += blockDim.x * gridDim.x) {
        int e = expert(p, b, row), index = -1;
        if (e >= 0 && e < p.experts) {
            index = offsets[e] + atomicAdd(cursors + e, 1);
            maps[index] = row;
        }
        inverse[row] = index;
    }
}
template <class T>
__global__ void gather(qpm_problem_v1 p, qpm_buffers_v1 b, int const *offsets, int const *maps, T *a,
                       int capacity) {
    int valid = offsets[p.experts];
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < int64_t(capacity) * p.k;
         i += int64_t(blockDim.x) * gridDim.x) {
        int row = int(i / p.k), k = int(i % p.k);
        T value(0.f);
        if (row < valid) {
            int slot = maps[row];
            int64_t offset = int64_t(slot / p.topk) * p.input_token_stride +
                             (p.channels == 1 ? 0 : int64_t(slot % p.topk) * p.input_row_stride) + k;
            value = static_cast<T const *>(b.input)[offset];
        }
        a[i] = value;
    }
}
template <class T>
__global__ void scatter(qpm_problem_v1 p, qpm_buffers_v1 b, int rows, int stride, int const *inverse,
                        T const *c) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < int64_t(rows) * p.n;
         i += int64_t(blockDim.x) * gridDim.x) {
        int row = int(i / p.n), n = int(i % p.n), src = inverse[row];
        static_cast<T *>(b.output)[int64_t(row) * p.output_stride + n] =
            src < 0 ? T(0.f) : c[int64_t(src) * stride + n];
    }
}
template <int Q, class T, bool Packed>
__global__ void expand(Weight w, T *out, int first_expert, int groups, int first_column, int columns,
                       int const *offsets) {
    uint64_t count = uint64_t(groups) * columns * w.k;
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(blockDim.x) * gridDim.x) {
        int e = first_expert + int(i / (uint64_t(columns) * w.k));
        int n = first_column + int(i / w.k % columns), k = int(i % w.k);
        if (offsets && offsets[e] == offsets[e + 1])
            continue;
        out[i] = n < w.n ? T(value<Q, Packed>(w, e, n, k)) : T(0.f);
    }
}
template <int Q, class T, bool Packed>
__global__ void indexed_simt(qpm_problem_v1 p, qpm_buffers_v1 b, Weight w, int rows) {
    int lane = threadIdx.x % 32;
    for (int64_t cell = int64_t(blockIdx.x) * 4 + threadIdx.x / 32; cell < int64_t(rows) * p.n;
         cell += int64_t(gridDim.x) * 4) {
        int row = int(cell / p.n), n = int(cell % p.n), e = expert(p, b, row);
        float sum = 0.f;
        if (e >= 0 && e < p.experts) {
            auto a = static_cast<T const *>(b.input) + int64_t(row / p.topk) * p.input_token_stride +
                     (p.channels == 1 ? 0 : int64_t(row % p.topk) * p.input_row_stride);
            for (int k = lane; k < p.k; k += 32) {
                T w16(value<Q, Packed>(w, e, n, k));
                sum = fmaf(float(a[k]), float(w16), sum);
            }
        } else if (e != -1 && lane == 0)
            atomicOr(b.device_error, QPM_BAD_ID);
        for (int delta = 16; delta; delta /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, delta);
        if (lane == 0)
            static_cast<T *>(b.output)[int64_t(row) * p.output_stride + n] = T(sum);
    }
}

template <int Q, class T, bool Packed> void run(Prepared const &h, qpm_buffers_v1 const &b) {
    auto const &p = h.p;
    auto const &l = h.l;
    auto const &s = l.sizes;
    if (!l.rows)
        return;
    auto stream = static_cast<hggcStream_t>(b.stream);
    auto w = weight(h, b);
    if (s.route == QPM_SIMT) {
        int blocks = int(std::min<uint64_t>((uint64_t(l.rows) * p.n + 3) / 4, 4096));
        indexed_simt<Q, T, Packed><<<blocks, 128, 0, stream>>>(p, b, w, l.rows);
        checked(hggcGetLastError());
        return;
    }
    auto base = static_cast<uint8_t *>(b.workspace);
    auto counts = reinterpret_cast<int *>(base + l.counts),
         cursors = reinterpret_cast<int *>(base + l.cursors);
    auto maps = reinterpret_cast<int *>(base + l.maps), inverse = reinterpret_cast<int *>(base + l.inverse);
    auto a = reinterpret_cast<T *>(base + l.a), c = reinterpret_cast<T *>(base + l.c),
         expanded = reinterpret_cast<T *>(base + l.w);
    auto metadata = reinterpret_cast<typename Grouped<T>::Metadata *>(base + l.metadata);
    checked(hggcMemsetAsync(counts, 0, 4 * (p.experts + 1), stream));
    count_rows<<<grid(l.rows), 256, 0, stream>>>(p, b, l.rows, counts);
    prefix_rows<<<1, 32, 0, stream>>>(counts, cursors, p.experts);
    map_rows<<<grid(l.rows), 256, 0, stream>>>(p, b, l.rows, counts, cursors, maps, inverse);
    gather<T><<<grid(uint64_t(l.rows + 32) * p.k), 256, 0, stream>>>(p, b, counts, maps, a, l.rows + 32);
    checked(hggcGetLastError());
    for (int e = 0; e < p.experts; e += s.expert_tile) {
        int groups = std::min(s.expert_tile, p.experts - e);
        for (int n = 0; n < l.padded_n; n += s.column_tile) {
            int columns = std::min(s.column_tile, l.padded_n - n);
            expand<Q, T, Packed><<<grid(uint64_t(groups) * columns * p.k), 256, 0, stream>>>(
                w, expanded, e, groups, n, columns, counts);
            group_metadata<T><<<1, 32, 0, stream>>>(metadata, counts, a, expanded, c, e, groups, n, columns,
                                                    l.padded_n, p.k);
            checked(hggcGetLastError());
            success(Grouped<T>::launch(metadata, groups, p.device, h.cus, base, stream),
                    "native grouped GEMM");
        }
    }
    scatter<T><<<grid(uint64_t(l.rows) * p.n), 256, 0, stream>>>(p, b, l.rows, l.padded_n, inverse, c);
    checked(hggcGetLastError());
}
template <int Q, class T, bool Packed>
void dequant(Prepared const &h, qpm_buffers_v1 const &b, int e, int first, int rows, void *out) {
    expand<Q, T, Packed><<<grid(uint64_t(rows) * h.p.k), 256, 0, static_cast<hggcStream_t>(b.stream)>>>(
        weight(h, b), static_cast<T *>(out), e, 1, first, rows, nullptr);
    checked(hggcGetLastError());
}
// One dispatch for run/dequant keeps format and precision admission identical.
template <int Q, class T>
void dispatch_layout(Prepared const &h, qpm_buffers_v1 const &b, int e = -1, int first = 0, int rows = 0,
                     void *out = nullptr) {
    if (h.p.layout == QPM_RAW_GGUF) {
        if (e < 0)
            run<Q, T, false>(h, b);
        else
            dequant<Q, T, false>(h, b, e, first, rows, out);
    } else {
        if (e < 0)
            run<Q, T, true>(h, b);
        else
            dequant<Q, T, true>(h, b, e, first, rows, out);
    }
}
template <class T>
void dispatch(Prepared const &h, qpm_buffers_v1 const &b, int e = -1, int first = 0, int rows = 0,
              void *out = nullptr) {
#define QPM_CASE(Q)                                                                                          \
    case Q:                                                                                                  \
        dispatch_layout<Q, T>(h, b, e, first, rows, out);                                                    \
        break
    switch (h.p.qtype) {
        QPM_CASE(7);
        QPM_CASE(8);
        QPM_CASE(10);
        QPM_CASE(11);
        QPM_CASE(12);
        QPM_CASE(13);
        QPM_CASE(14);
    default:
        require(false, "unknown qtype");
    }
#undef QPM_CASE
}
void dispatch(Prepared const &h, qpm_buffers_v1 const &b, int e = -1, int first = 0, int rows = 0,
              void *out = nullptr) {
    if (h.p.dtype == QPM_F16)
        dispatch<cutlass::half_t>(h, b, e, first, rows, out);
    else
        dispatch<cutlass::bfloat16_t>(h, b, e, first, rows, out);
}
void device(Prepared const &h) {
    int current = -1;
    checked(hggcGetDevice(&current));
    require(current == h.p.device, "current device differs from prepared device");
}
int warm_helpers();
} // namespace quactlize::compat

extern "C" {
int quactlize_ppu_matmul_query_v1(qpm_problem_v1 const *p, quactlize_ppu_placed_arrangement_v2 const *arr,
                                  qpm_sizes_v1 *s) {
    using namespace quactlize::compat;
    return guarded([&] {
        require(p && s, "null query argument");
        *s = layout(*p, arr).sizes;
    });
}
int quactlize_ppu_matmul_prepare_v1(qpm_problem_v1 const *p, quactlize_ppu_placed_arrangement_v2 const *arr,
                                    qpm_buffers_v1 const *b, void **handle) {
    using namespace quactlize::compat;
    if (handle)
        *handle = nullptr;
    return guarded([&] {
        require(p && b && handle, "null prepare argument");
        auto h = std::make_unique<Prepared>();
        h->p = *p;
        h->l = layout(*p, arr);
        validate(h->p, h->l, *b);
        device(*h);
        hggcStreamCaptureStatus capture;
        auto stream = static_cast<hggcStream_t>(b->stream);
        checked(hggcStreamIsCapturing(stream, &capture));
        require(capture == hggcStreamCaptureStatusNone, "prepare during capture");
        hggcDeviceProp prop{};
        checked(hggcGetDeviceProperties(&prop, p->device));
        h->cus = prop.multiProcessorCount;
        require(h->cus > 0, "missing device compute units");
        if (h->l.sizes.route == QPM_GROUPED_TC) {
            if (p->dtype == QPM_F16)
                success(Grouped<cutlass::half_t>::configure(), "F16 grouped preparation");
            else
                success(Grouped<cutlass::bfloat16_t>::configure(), "BF16 grouped preparation");
        }
        success(warm_helpers(), "native helper module preparation");
        dispatch(*h, *b);
        checked(hggcStreamSynchronize(stream));
        *handle = h.release();
    });
}
int quactlize_ppu_matmul_run_v1(void *handle, qpm_buffers_v1 const *b) {
    using namespace quactlize::compat;
    return guarded([&] {
        require(handle && b, "null run argument");
        auto &h = *static_cast<Prepared *>(handle);
        validate(h.p, h.l, *b);
        device(h);
        dispatch(h, *b);
    });
}
void quactlize_ppu_matmul_destroy_v1(void *handle) {
    delete static_cast<quactlize::compat::Prepared *>(handle);
}
char const *quactlize_ppu_matmul_error_v1() { return quactlize::compat::last_error.c_str(); }
int quactlize_ppu_matmul_dequant_v1(void *handle, qpm_buffers_v1 const *b, int e, int first, int rows,
                                    void *out, uint64_t bytes) {
    using namespace quactlize::compat;
    return guarded([&] {
        require(handle && b, "null dequant argument");
        auto const &h = *static_cast<Prepared *>(handle);
        require(e >= 0 && e < h.p.experts && first >= 0 && rows > 0 && first <= h.p.n - rows,
                "invalid dequant row range");
        weights(h.l.sizes, *b);
        device(h);
        uint64_t length = mul(mul(rows, h.p.k), 2);
        span(out, length, bytes, 2);
        auto const &s = h.l.sizes;
        disjoint(out, length, b->raw, s.raw_bytes);
        disjoint(out, length, b->low, s.low_bytes);
        disjoint(out, length, b->high, s.high_bytes);
        disjoint(out, length, b->units, s.units_bytes);
        dispatch(h, *b, e, first, rows, out);
    });
}
}
