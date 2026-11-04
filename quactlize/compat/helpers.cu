#include "contract.hpp"
#include <hggc_runtime.h>
#include "cutlass/numeric_types.h"
#include <cmath>
#include <string>

namespace quactlize::compat {
extern thread_local std::string last_error;
template <class F> int helper_guard(F f) {
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
void launch_checked() {
    auto rc = hggcGetLastError();
    if (rc != hggcSuccess)
        throw std::runtime_error(hggcGetErrorString(rc));
}
__global__ void router(qpm_router_v1 p) {
    int token = blockIdx.x, tid = threadIdx.x;
    __shared__ float scores[1024], reduce[256];
    __shared__ int indices[256], bad;
    if (!tid)
        bad = 0;
    __syncthreads();
    float local = -INFINITY;
    for (int e = tid; e < p.experts; e += 256) {
        float x = p.logits[int64_t(token) * p.logits_stride + e];
        scores[e] = x;
        if (!isfinite(x))
            atomicExch(&bad, 1);
        local = fmaxf(local, x);
    }
    reduce[tid] = local;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (tid < s)
            reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
        __syncthreads();
    }
    if (bad) {
        if (!tid)
            atomicOr(p.device_error, QPM_BAD_LOGIT);
        if (tid < p.topk) {
            p.ids[int64_t(token) * p.ids_stride + tid] = -1;
            p.weights[int64_t(token) * p.weights_stride + tid] = 0.f;
        }
        return;
    }
    float maximum = reduce[0], sum = 0.f;
    __syncthreads();
    for (int e = tid; e < p.experts; e += 256) {
        scores[e] = expf(scores[e] - maximum);
        sum += scores[e];
    }
    reduce[tid] = sum;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (tid < s)
            reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    sum = reduce[0];
    __syncthreads();
    for (int e = tid; e < p.experts; e += 256)
        scores[e] /= sum;
    __syncthreads();
    // Stable top-k, including equal probabilities after FP32 softmax rounding.
    for (int slot = 0; slot < p.topk; ++slot) {
        float value = -INFINITY;
        int index = INT_MAX;
        for (int e = tid; e < p.experts; e += 256) {
            if (scores[e] > value || (scores[e] == value && e < index)) {
                value = scores[e];
                index = e;
            }
        }
        reduce[tid] = value;
        indices[tid] = index;
        __syncthreads();
        for (int s = 128; s; s /= 2) {
            if (tid < s && (reduce[tid + s] > reduce[tid] ||
                            (reduce[tid + s] == reduce[tid] && indices[tid + s] < indices[tid]))) {
                reduce[tid] = reduce[tid + s];
                indices[tid] = indices[tid + s];
            }
            __syncthreads();
        }
        if (!tid) {
            p.ids[int64_t(token) * p.ids_stride + slot] = indices[0];
            p.weights[int64_t(token) * p.weights_stride + slot] = reduce[0];
            scores[indices[0]] = -INFINITY;
        }
        __syncthreads();
    }
    if (!tid) {
        float norm = 0.f;
        for (int slot = 0; slot < p.topk; ++slot)
            norm += p.weights[int64_t(token) * p.weights_stride + slot];
        for (int slot = 0; slot < p.topk; ++slot)
            p.weights[int64_t(token) * p.weights_stride + slot] =
                (p.weights[int64_t(token) * p.weights_stride + slot] / norm) * p.multiplier;
    }
}
template <class T> __global__ void swiglu(qpm_swiglu_v1 p) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < int64_t(p.rows) * p.hidden;
         i += int64_t(blockDim.x) * gridDim.x) {
        int64_t row = i / p.hidden, col = i % p.hidden;
        float gate = float(static_cast<T const *>(p.gate)[row * p.gate_stride + col]);
        float up = p.merged ? float(static_cast<T const *>(p.gate)[row * p.gate_stride + p.hidden + col])
                            : float(static_cast<T const *>(p.up)[row * p.up_stride + col]);
        static_cast<T *>(p.output)[row * p.output_stride + col] = T((gate / (1.f + expf(-gate))) * up);
    }
}
template <class T> __global__ void merge(qpm_merge_v1 p) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < int64_t(p.m) * p.n;
         i += int64_t(blockDim.x) * gridDim.x) {
        int64_t row = i / p.n, col = i % p.n;
        float sum = 0.f;
        for (int slot = 0; slot < p.topk; ++slot)
            sum += float(static_cast<T const *>(p.input)[(row * p.topk + slot) * p.input_stride + col]) *
                   p.weights[row * p.weights_stride + slot];
        p.output[row * p.output_stride + col] = sum;
    }
}
int warm_helpers() {
    hggcFuncAttributes a{};
    for (void const *f :
         {reinterpret_cast<void const *>(router), reinterpret_cast<void const *>(swiglu<cutlass::half_t>),
          reinterpret_cast<void const *>(swiglu<cutlass::bfloat16_t>),
          reinterpret_cast<void const *>(merge<cutlass::half_t>),
          reinterpret_cast<void const *>(merge<cutlass::bfloat16_t>)})
        if (hggcFuncGetAttributes(&a, f) != hggcSuccess)
            return QPM_RUNTIME;
    return QPM_OK;
}
int helper_grid(uint64_t n) { return int(std::min<uint64_t>((n + 255) / 256, 4096)); }
void dtype(int value) { require(value == QPM_F16 || value == QPM_BF16, "expected native F16 or BF16"); }
} // namespace quactlize::compat

extern "C" {
int quactlize_ppu_router_run_v1(qpm_router_v1 const *p) {
    using namespace quactlize::compat;
    return helper_guard([&] {
        require(p && p->version == 1 && p->size == sizeof(*p) && !p->reserved, "invalid router ABI");
        require(p->m >= 0 && p->experts > 0 && p->experts <= 1024 && p->topk > 0 && p->topk <= 32 &&
                    p->topk <= p->experts,
                "invalid router shape");
        require(std::isfinite(p->multiplier) && p->multiplier >= 0, "invalid router multiplier");
        uint64_t logits = extent(p->m, p->logits_stride, p->experts, 4);
        uint64_t ids = extent(p->m, p->ids_stride, p->topk, 4),
                 w = extent(p->m, p->weights_stride, p->topk, 4);
        span(p->logits, logits, logits, 4);
        span(p->ids, ids, ids, 4);
        span(p->weights, w, w, 4);
        span(p->device_error, 4, 4, 4);
        disjoint(p->ids, ids, p->logits, logits);
        disjoint(p->weights, w, p->logits, logits);
        disjoint(p->ids, ids, p->weights, w);
        disjoint(p->device_error, 4, p->logits, logits);
        disjoint(p->device_error, 4, p->ids, ids);
        disjoint(p->device_error, 4, p->weights, w);
        if (p->m) {
            router<<<p->m, 256, 0, static_cast<hggcStream_t>(p->stream)>>>(*p);
            launch_checked();
        }
    });
}
int quactlize_ppu_swiglu_run_v1(qpm_swiglu_v1 const *p) {
    using namespace quactlize::compat;
    return helper_guard([&] {
        require(p && p->version == 1 && p->size == sizeof(*p), "invalid SwiGLU ABI");
        dtype(p->dtype);
        require(p->rows >= 0 && p->hidden > 0 && p->hidden <= INT_MAX / 2 &&
                    (p->merged == 0 || p->merged == 1),
                "invalid SwiGLU shape");
        uint64_t g = extent(p->rows, p->gate_stride, p->hidden * (p->merged ? 2 : 1), 2);
        uint64_t u = p->merged ? 0 : extent(p->rows, p->up_stride, p->hidden, 2);
        uint64_t o = extent(p->rows, p->output_stride, p->hidden, 2);
        span(p->gate, g, g, 2);
        span(p->up, u, u, 2);
        span(p->output, o, o, 2);
        require(!p->merged || !p->up, "merged SwiGLU must not carry a separate up buffer");
        disjoint(p->output, o, p->gate, g);
        disjoint(p->output, o, p->up, u);
        if (!p->rows)
            return;
        int grid = helper_grid(mul(p->rows, p->hidden));
        auto stream = static_cast<hggcStream_t>(p->stream);
        if (p->dtype == QPM_F16)
            swiglu<cutlass::half_t><<<grid, 256, 0, stream>>>(*p);
        else
            swiglu<cutlass::bfloat16_t><<<grid, 256, 0, stream>>>(*p);
        launch_checked();
    });
}
int quactlize_ppu_merge_run_v1(qpm_merge_v1 const *p) {
    using namespace quactlize::compat;
    return helper_guard([&] {
        require(p && p->version == 1 && p->size == sizeof(*p), "invalid merge ABI");
        dtype(p->dtype);
        require(p->m >= 0 && p->topk > 0 && p->topk <= 32 && p->n > 0, "invalid merge shape");
        uint64_t i = extent(mul(p->m, p->topk), p->input_stride, p->n, 2);
        uint64_t w = extent(p->m, p->weights_stride, p->topk, 4), o = extent(p->m, p->output_stride, p->n, 4);
        span(p->input, i, i, 2);
        span(p->weights, w, w, 4);
        span(p->output, o, o, 4);
        disjoint(p->output, o, p->input, i);
        disjoint(p->output, o, p->weights, w);
        if (!p->m)
            return;
        int grid = helper_grid(mul(p->m, p->n));
        auto stream = static_cast<hggcStream_t>(p->stream);
        if (p->dtype == QPM_F16)
            merge<cutlass::half_t><<<grid, 256, 0, stream>>>(*p);
        else
            merge<cutlass::bfloat16_t><<<grid, 256, 0, stream>>>(*p);
        launch_checked();
    });
}
}
