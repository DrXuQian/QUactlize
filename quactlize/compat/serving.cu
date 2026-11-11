#include "serving_contract.hpp"
#include <string>
#ifdef QPM_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define QPM_RT(name) cuda##name
using Half = __half;
using BFloat = __nv_bfloat16;
#else
#include <hggc_runtime.h>
#include "cutlass/numeric_types.h"
#define QPM_RT(name) hggc##name
using Half = cutlass::half_t;
using BFloat = cutlass::bfloat16_t;
#endif

namespace quactlize::compat::serving {
static thread_local std::string error;
template <class F> int guard(F f) {
    try {
        f();
        error.clear();
        return QPM_OK;
    } catch (std::invalid_argument const &e) {
        error = e.what();
        return QPM_INVALID;
    } catch (std::exception const &e) {
        error = e.what();
        return QPM_RUNTIME;
    }
}
static void checked(QPM_RT(Error_t) rc) {
    if (rc != QPM_RT(Success))
        throw std::runtime_error(QPM_RT(GetErrorString)(rc));
}
static int grid(uint64_t n) { return int(std::max<uint64_t>(1, std::min<uint64_t>((n + 255) / 256, 4096))); }

__device__ int live_count(int capacity, int32_t const *count, int32_t *status) {
    int live = count ? *count : capacity;
    if (live < 0 || live > capacity) {
        if (!threadIdx.x && !blockIdx.x)
            atomicOr(status, QPM_BAD_COUNT);
        return 0;
    }
    return live;
}
template <class T> __global__ void router(qpm_router_v2 p) {
    int live = live_count(p.capacity, p.num_valid_tokens, p.device_error);
    int row = blockIdx.x, tid = threadIdx.x;
    if (row >= p.capacity)
        return;
    if (row >= live) {
        if (tid < p.topk) {
            p.selected_logits[int64_t(row) * p.selected_stride + tid] = 0;
            p.ids[int64_t(row) * p.ids_stride + tid] = -1;
            p.weights[int64_t(row) * p.weights_stride + tid] = 0;
        }
        return;
    }
    __shared__ float scores[1024], values[256], selected[32];
    __shared__ int indices[256], bad;
    if (!tid)
        bad = 0;
    __syncthreads();
    for (int e = tid; e < p.experts; e += 256) {
        float x = float(static_cast<T const *>(p.logits)[int64_t(row) * p.logits_stride + e]);
        scores[e] = x;
        if (!isfinite(x))
            atomicExch(&bad, 1);
    }
    __syncthreads();
    if (bad) {
        if (!tid)
            atomicOr(p.device_error, QPM_BAD_LOGIT);
        if (tid < p.topk) {
            p.selected_logits[int64_t(row) * p.selected_stride + tid] = 0;
            p.ids[int64_t(row) * p.ids_stride + tid] = -1;
            p.weights[int64_t(row) * p.weights_stride + tid] = 0;
        }
        return;
    }
    for (int slot = 0; slot < p.topk; ++slot) {
        float best = -INFINITY;
        int index = -1;
        for (int e = tid; e < p.experts; e += 256)
            if (scores[e] > best || (scores[e] == best && e > index)) {
                best = scores[e];
                index = e;
            }
        values[tid] = best;
        indices[tid] = index;
        __syncthreads();
        for (int s = 128; s; s /= 2) {
            if (tid < s && (values[tid + s] > values[tid] ||
                            (values[tid + s] == values[tid] && indices[tid + s] > indices[tid]))) {
                values[tid] = values[tid + s];
                indices[tid] = indices[tid + s];
            }
            __syncthreads();
        }
        if (!tid) {
            selected[slot] = values[0];
            p.selected_logits[int64_t(row) * p.selected_stride + slot] = values[0];
            p.ids[int64_t(row) * p.ids_stride + slot] = indices[0];
            scores[indices[0]] = -INFINITY;
        }
        __syncthreads();
    }
    if (!tid) {
        float maximum = selected[0], sum = 0;
        for (int s = 0; s < p.topk; ++s) {
            selected[s] = expf(selected[s] - maximum);
            sum += selected[s];
        }
        for (int s = 0; s < p.topk; ++s)
            p.weights[int64_t(row) * p.weights_stride + s] = selected[s] / sum * p.multiplier;
    }
}
template <class T, bool Round> __global__ void swiglu(qpm_swiglu_v2 p) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < int64_t(p.rows) * p.hidden;
         i += int64_t(blockDim.x) * gridDim.x) {
        auto row = i / p.hidden, col = i % p.hidden;
        float g = float(static_cast<T const *>(p.gate)[row * p.gate_stride + col]);
        float u = p.merged ? float(static_cast<T const *>(p.gate)[row * p.gate_stride + p.hidden + col])
                           : float(static_cast<T const *>(p.up)[row * p.up_stride + col]);
        // An explicit native-16 conversion is an observable rounding boundary,
        // not a standalone cast kernel. Do not build with unsafe fast-math.
        float a = g * (1.f / (1.f + expf(-g)));
        if constexpr (Round)
            a = float(T(a));
        static_cast<T *>(p.output)[row * p.output_stride + col] = T(a * u);
    }
}
__device__ float half_bits(uint16_t bits) {
#ifdef QPM_CUDA
    return __half2float(__ushort_as_half(bits));
#else
    return float(cutlass::half_t::bitcast(bits));
#endif
}
template <class T, int Q> __global__ void rows(qpm_rows_v1 p) {
    int live = live_count(p.capacity, p.num_rows, p.device_error);
    constexpr int bytes = Q == 8 ? 34 : 18;
    constexpr int block = Q == 42 ? 64 : 32;
    constexpr int8_t codebook[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                     1,    13,   25,  38,  53,  69,  89,  113};
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < int64_t(p.capacity) * p.columns;
         i += int64_t(blockDim.x) * gridDim.x) {
        int64_t row = i / p.columns, col = i % p.columns;
        float value = 0;
        if (row < live) {
            int64_t id = p.ids[row];
            if (id >= 0 && id < p.table_rows) {
                auto b = static_cast<uint8_t const *>(p.raw) + (id * (p.columns / block) + col / block) * bytes;
                float scale = half_bits(uint16_t(b[0]) | (uint16_t(b[1]) << 8));
                int code;
                if constexpr (Q == 8)
                    code = int(int8_t(b[2 + col % 32]));
                else if constexpr (Q == 42)
                    code = int((b[2 + (col % 64) / 4] >> (2 * (col % 4))) & 3) - 1;
                else
                    code = codebook[(b[2 + col % 16] >> (col % 32 / 16 * 4)) & 15];
                value = scale * float(code);
            } else if (id != -1 && col == 0)
                atomicOr(p.device_error, QPM_BAD_ID);
        }
        static_cast<T *>(p.output)[i] = T(value);
    }
}
template <class T> __global__ void merge(qpm_merge_v1 p) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < int64_t(p.m) * p.n;
         i += int64_t(blockDim.x) * gridDim.x) {
        auto row = i / p.n, col = i % p.n;
        float sum = 0;
        for (int s = 0; s < p.topk; ++s)
            sum += float(static_cast<T const *>(p.input)[(row * p.topk + s) * p.input_stride + col]) *
                   p.weights[row * p.weights_stride + s];
        p.output[row * p.output_stride + col] = sum;
    }
}
} // namespace quactlize::compat::serving

extern "C" {
char const *quactlize_serving_error_v1(void) { return quactlize::compat::serving::error.c_str(); }
int quactlize_serving_prepare_v1(int32_t device, void *stream) {
    using namespace quactlize::compat;
    using namespace serving;
    return guard([&] {
        require(device >= 0 && stream, "prepare requires device and explicit stream");
        int current = -1;
        checked(QPM_RT(GetDevice)(&current));
        require(current == device, "prepare on wrong device");
        auto s = static_cast<QPM_RT(Stream_t)>(stream);
        QPM_RT(StreamCaptureStatus) status;
        checked(QPM_RT(StreamIsCapturing)(s, &status));
        require(status == QPM_RT(StreamCaptureStatusNone), "prepare before capture");
        // Real zero-work launches, not merely symbol lookup: first module
        // loading/lazy runtime initialization must finish BEFORE capture.
        qpm_router_v2 r{};
        qpm_swiglu_v2 a{};
        a.hidden = 1;
        qpm_rows_v1 g{};
        g.columns = 32;
        qpm_merge_v1 m{};
        m.n = 1;
        router<float><<<1, 256, 0, s>>>(r);
        router<BFloat><<<1, 256, 0, s>>>(r);
        swiglu<Half, false><<<1, 256, 0, s>>>(a);
        swiglu<Half, true><<<1, 256, 0, s>>>(a);
        swiglu<BFloat, false><<<1, 256, 0, s>>>(a);
        swiglu<BFloat, true><<<1, 256, 0, s>>>(a);
        rows<Half, 8><<<1, 256, 0, s>>>(g);
        rows<BFloat, 8><<<1, 256, 0, s>>>(g);
        rows<Half, 20><<<1, 256, 0, s>>>(g);
        rows<BFloat, 20><<<1, 256, 0, s>>>(g);
        rows<Half, 42><<<1, 256, 0, s>>>(g);
        rows<BFloat, 42><<<1, 256, 0, s>>>(g);
        merge<Half><<<1, 256, 0, s>>>(m);
        merge<BFloat><<<1, 256, 0, s>>>(m);
        checked(QPM_RT(GetLastError)());
        checked(QPM_RT(StreamSynchronize)(s));
    });
}
int quactlize_serving_router_v2(qpm_router_v2 const *p) {
    using namespace quactlize::compat;
    using namespace serving;
    return guard([&] {
        require(p, "null router call");
        validate(*p);
        auto s = static_cast<QPM_RT(Stream_t)>(p->stream);
        if (p->dtype == QPM_F32)
            router<float><<<std::max(1, p->capacity), 256, 0, s>>>(*p);
        else
            router<BFloat><<<std::max(1, p->capacity), 256, 0, s>>>(*p);
        checked(QPM_RT(GetLastError)());
    });
}
int quactlize_serving_swiglu_v2(qpm_swiglu_v2 const *p) {
    using namespace quactlize::compat;
    using namespace serving;
    return guard([&] {
        require(p, "null SwiGLU call");
        validate(*p);
        auto s = static_cast<QPM_RT(Stream_t)>(p->stream);
        int n = grid(mul(p->rows, p->hidden));
        if (p->dtype == QPM_F16) {
            if (p->round_silu)
                swiglu<Half, true><<<n, 256, 0, s>>>(*p);
            else
                swiglu<Half, false><<<n, 256, 0, s>>>(*p);
        } else {
            if (p->round_silu)
                swiglu<BFloat, true><<<n, 256, 0, s>>>(*p);
            else
                swiglu<BFloat, false><<<n, 256, 0, s>>>(*p);
        }
        checked(QPM_RT(GetLastError)());
    });
}
int quactlize_serving_gather_rows_v1(qpm_rows_v1 const *p) {
    using namespace quactlize::compat;
    using namespace serving;
    return guard([&] {
        require(p, "null row lookup call");
        validate(*p);
        auto s = static_cast<QPM_RT(Stream_t)>(p->stream);
        int n = grid(mul(p->capacity, p->columns));
        if (p->dtype == QPM_F16) {
            if (p->qtype == 8)
                rows<Half, 8><<<n, 256, 0, s>>>(*p);
            else if (p->qtype == 42)
                rows<Half, 42><<<n, 256, 0, s>>>(*p);
            else
                rows<Half, 20><<<n, 256, 0, s>>>(*p);
        } else {
            if (p->qtype == 8)
                rows<BFloat, 8><<<n, 256, 0, s>>>(*p);
            else if (p->qtype == 42)
                rows<BFloat, 42><<<n, 256, 0, s>>>(*p);
            else
                rows<BFloat, 20><<<n, 256, 0, s>>>(*p);
        }
        checked(QPM_RT(GetLastError)());
    });
}
int quactlize_serving_merge_v1(qpm_merge_v1 const *p) {
    using namespace quactlize::compat;
    using namespace serving;
    return guard([&] {
        require(p, "null merge call");
        validate(*p);
        auto s = static_cast<QPM_RT(Stream_t)>(p->stream);
        int n = grid(mul(p->m, p->n));
        if (p->dtype == QPM_F16)
            merge<Half><<<n, 256, 0, s>>>(*p);
        else
            merge<BFloat><<<n, 256, 0, s>>>(*p);
        checked(QPM_RT(GetLastError)());
    });
}
}
