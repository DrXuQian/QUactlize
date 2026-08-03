// CUDA-REPRESENTATIVE DEVICE GOLDENS AND MICROBENCHMARKS FOR THE GGUF PATH.
//
// This is built as a tiny shared library by tests/test_gguf_golden.py. Keeping torch out of the translation unit is
// deliberate: CUDA events surround only the launch, while Python supplies the independent host-op golden. H2D/D2H
// and cudaMalloc happen in the ctypes wrapper outside every timed interval.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "gguf_scale_prepass.hpp"
#include "gguf_vecdot.hpp"

namespace {

using cutlass::half_t;
using gguf_scale::KType;
using gguf_scale::prepass::BlockDesc;
using gguf_scale::prepass::PlaneDesc;

int cuda_ok(cudaError_t e) { return e == cudaSuccess ? 0 : int(e); }

__global__ void flush_l2_kernel(uint8_t const* p, size_t bytes, unsigned long long* sink) {
  size_t const i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 128;
  unsigned long long v = 0;
  for (size_t j = i; j < bytes; j += size_t(gridDim.x) * blockDim.x * 128) v += p[j];
  if (v) atomicAdd(sink, v);
}

struct Q4Device {
  uint8_t* blocks = nullptr;
  half_t *d = nullptr, *dmin = nullptr, *scale = nullptr, *zero = nullptr;
  uint8_t* flush = nullptr;
  unsigned long long* sink = nullptr;
  size_t total = 0, flush_bytes = 0;

  ~Q4Device() {
    cudaFree(blocks); cudaFree(d); cudaFree(dmin); cudaFree(scale); cudaFree(zero); cudaFree(flush); cudaFree(sink);
  }

  int allocate(int cols, int superblocks) {
    total = size_t(cols) * superblocks;
    cudaDeviceProp prop{};
    int dev = 0;
    if (cuda_ok(cudaGetDevice(&dev)) || cuda_ok(cudaGetDeviceProperties(&prop, dev))) return 1;
    flush_bytes = std::max<size_t>(size_t(prop.l2CacheSize) * 2, size_t(128) << 20);
    if (cuda_ok(cudaMalloc(&blocks, total * 12)) || cuda_ok(cudaMalloc(&d, total * sizeof(half_t))) ||
        cuda_ok(cudaMalloc(&dmin, total * sizeof(half_t))) || cuda_ok(cudaMalloc(&scale, total * 8 * sizeof(half_t))) ||
        cuda_ok(cudaMalloc(&zero, total * 8 * sizeof(half_t))) || cuda_ok(cudaMalloc(&flush, flush_bytes)) ||
        cuda_ok(cudaMalloc(&sink, sizeof(*sink)))) return 2;
    if (cuda_ok(cudaMemset(flush, 1, flush_bytes)) || cuda_ok(cudaMemset(sink, 0, sizeof(*sink)))) return 3;
    return 0;
  }

  BlockDesc src(int superblocks) const {
    return {blocks, d, dmin, int64_t(superblocks) * 12, 12, superblocks, 1};
  }
  PlaneDesc dst(int superblocks) const { return {scale, zero, int64_t(superblocks) * 8, 1}; }
  void flush_l2() const { flush_l2_kernel<<<4096, 256>>>(flush, flush_bytes, sink); }
};

template <bool Cooperative>
void launch_q4(Q4Device const& q, int cols, int superblocks) {
#ifndef GGUF_PROBE_THREADS
#define GGUF_PROBE_THREADS 256
#endif
  constexpr int kThreads = GGUF_PROBE_THREADS;
  if constexpr (Cooperative) {
    int const grid = gguf_scale::prepass::prepass_grid_size(cols, superblocks, kThreads);
    gguf_scale::prepass::prepass_kernel<KType::Q4_K, 8><<<grid, kThreads>>>(
        q.src(superblocks), q.dst(superblocks), cols, superblocks);
  } else {
    int const total = cols * superblocks;
    gguf_scale::prepass::prepass_kernel_serial<KType::Q4_K, 8><<<(total + kThreads - 1) / kThreads, kThreads>>>(
        q.src(superblocks), q.dst(superblocks), cols, superblocks);
  }
}

template <bool Cooperative>
int run_q4(uint8_t const* blocks, uint16_t const* d, uint16_t const* dmin, uint16_t* scale, uint16_t* zero,
           int cols, int superblocks) {
  Q4Device q;
  if (int e = q.allocate(cols, superblocks)) return 100 + e;
  size_t const total = q.total;
  if (cuda_ok(cudaMemcpy(q.blocks, blocks, total * 12, cudaMemcpyHostToDevice)) ||
      cuda_ok(cudaMemcpy(q.d, d, total * 2, cudaMemcpyHostToDevice)) ||
      cuda_ok(cudaMemcpy(q.dmin, dmin, total * 2, cudaMemcpyHostToDevice))) return 110;
  if constexpr (Cooperative) launch_q4<true>(q, cols, superblocks);
  else                       launch_q4<false>(q, cols, superblocks);
  if (cuda_ok(cudaGetLastError()) || cuda_ok(cudaDeviceSynchronize())) return 111;
  if (cuda_ok(cudaMemcpy(scale, q.scale, total * 8 * 2, cudaMemcpyDeviceToHost)) ||
      cuda_ok(cudaMemcpy(zero, q.zero, total * 8 * 2, cudaMemcpyDeviceToHost))) return 112;
  return 0;
}

template <bool Cooperative>
int bench_variant(Q4Device const& q, int cols, int superblocks, int reps, float* usec) {
  cudaEvent_t begin{}, end{};
  if (cuda_ok(cudaEventCreate(&begin)) || cuda_ok(cudaEventCreate(&end))) return 1;
  std::vector<float> samples;
  samples.reserve(size_t(reps));
  for (int r = 0; r < reps + 5; ++r) {
    q.flush_l2();
    cudaEventRecord(begin);
    launch_q4<Cooperative>(q, cols, superblocks);
    cudaEventRecord(end);
    if (cuda_ok(cudaEventSynchronize(end))) return 2;
    float ms = 0.f;
    if (cuda_ok(cudaEventElapsedTime(&ms, begin, end))) return 3;
    if (r >= 5) samples.push_back(ms * 1000.f);
  }
  cudaEventDestroy(begin); cudaEventDestroy(end);
  std::sort(samples.begin(), samples.end());
  *usec = samples[samples.size() / 2];
  return 0;
}

template <int Blocks, bool Physical>
void launch_q4_layout(uint8_t const* blocks, half_t* out) {
  // A compact per-superblock transpose: logical element i = lane + 32*v is stored at lane*8 + v. This is small
  // enough that physical ownership keeps a warp on one source block, but nontrivial enough that striping logical i
  // destroys lane-fast stores -- the exact distinction the destination API exists to preserve.
  using Layout = cute::Layout<cute::Shape<cute::Int<Blocks>, cute::Shape<cute::_32, cute::_8>>,
                              cute::Stride<cute::Int<256>, cute::Stride<cute::_8, cute::_1>>>;
  constexpr int kThreads = 256;
  constexpr int kWarpsPerCta = kThreads / 32;
  if constexpr (Physical) {
    gguf_scale::vecdot::dequantize_kernel_warp<KType::Q4_K>
        <<<(Blocks + kWarpsPerCta - 1) / kWarpsPerCta, kThreads>>>(blocks, 144, out, Blocks, Layout{});
  } else {
    gguf_scale::vecdot::dequantize_kernel_warp_logical<KType::Q4_K>
        <<<(Blocks + kWarpsPerCta - 1) / kWarpsPerCta, kThreads>>>(blocks, 144, out, Blocks, Layout{});
  }
}

template <int Blocks, bool Physical>
int bench_layout_variant(uint8_t const* blocks, half_t* out, uint8_t const* flush, size_t flush_bytes,
                         unsigned long long* sink, int reps, float* usec) {
  cudaEvent_t begin{}, end{};
  if (cuda_ok(cudaEventCreate(&begin)) || cuda_ok(cudaEventCreate(&end))) return 1;
  std::vector<float> samples;
  for (int r = 0; r < reps + 5; ++r) {
    flush_l2_kernel<<<4096, 256>>>(flush, flush_bytes, sink);
    cudaEventRecord(begin);
    launch_q4_layout<Blocks, Physical>(blocks, out);
    cudaEventRecord(end);
    if (cuda_ok(cudaEventSynchronize(end))) return 2;
    float ms = 0.f;
    if (cuda_ok(cudaEventElapsedTime(&ms, begin, end))) return 3;
    if (r >= 5) samples.push_back(ms * 1000.f);
  }
  cudaEventDestroy(begin); cudaEventDestroy(end);
  std::sort(samples.begin(), samples.end());
  *usec = samples[samples.size() / 2];
  return 0;
}

}  // namespace

extern "C" int quactlize_cuda_q4_prepass(uint8_t const* blocks, uint16_t const* d, uint16_t const* dmin,
                                          uint16_t* scale, uint16_t* zero, int cols, int superblocks,
                                          int cooperative) {
  return cooperative ? run_q4<true>(blocks, d, dmin, scale, zero, cols, superblocks)
                     : run_q4<false>(blocks, d, dmin, scale, zero, cols, superblocks);
}

extern "C" int quactlize_cuda_q4_prepass_bench(int scale_factor, int reps, float* serial_us, float* coop_us,
                                                double* bytes) {
  int const cols = 8 * 2048 * scale_factor;
  int const superblocks = 8;
  Q4Device q;
  if (int e = q.allocate(cols, superblocks)) return 200 + e;

  // Inputs are allocated and initialised before timing. Their particular values do not affect the access pattern;
  // correctness is a separate call supplied with non-degenerate data from the official Python golden.
  if (cuda_ok(cudaMemset(q.blocks, 0x5a, q.total * 12)) || cuda_ok(cudaMemset(q.d, 0x34, q.total * 2)) ||
      cuda_ok(cudaMemset(q.dmin, 0x30, q.total * 2))) return 210;
  launch_q4<false>(q, cols, superblocks);
  launch_q4<true>(q, cols, superblocks);
  if (cuda_ok(cudaDeviceSynchronize())) return 211;
  if (int e = bench_variant<false>(q, cols, superblocks, reps, serial_us)) return 220 + e;
  if (int e = bench_variant<true >(q, cols, superblocks, reps, coop_us)) return 230 + e;
  *bytes = double(q.total) * gguf_scale::prepass::bytes_per_column_superblock<KType::Q4_K>(true);
  return 0;
}

extern "C" int quactlize_cuda_q4_layout_check(uint8_t const* blocks, uint16_t* physical, uint16_t* logical) {
  constexpr int kBlocks = 512;
  uint8_t* db = nullptr;
  half_t* dout = nullptr;
  size_t constexpr kOut = size_t(kBlocks) * 256 * sizeof(half_t);
  if (cuda_ok(cudaMalloc(&db, size_t(kBlocks) * 144)) || cuda_ok(cudaMalloc(&dout, kOut))) return 300;
  if (cuda_ok(cudaMemcpy(db, blocks, size_t(kBlocks) * 144, cudaMemcpyHostToDevice))) return 301;
  cudaMemset(dout, 0xa5, kOut);
  launch_q4_layout<kBlocks, true>(db, dout);
  if (cuda_ok(cudaGetLastError()) || cuda_ok(cudaDeviceSynchronize()) ||
      cuda_ok(cudaMemcpy(physical, dout, kOut, cudaMemcpyDeviceToHost))) return 302;
  cudaMemset(dout, 0x5a, kOut);
  launch_q4_layout<kBlocks, false>(db, dout);
  if (cuda_ok(cudaGetLastError()) || cuda_ok(cudaDeviceSynchronize()) ||
      cuda_ok(cudaMemcpy(logical, dout, kOut, cudaMemcpyDeviceToHost))) return 303;
  cudaFree(db); cudaFree(dout);
  return 0;
}

extern "C" int quactlize_cuda_q4_layout_bench(int reps, float* physical_us, float* logical_us) {
  constexpr int kBlocks = 8 * 2048 * 8;
  size_t constexpr kBlockBytes = size_t(kBlocks) * 144;
  size_t constexpr kOutBytes = size_t(kBlocks) * 256 * sizeof(half_t);
  uint8_t *blocks = nullptr, *flush = nullptr;
  half_t* out = nullptr;
  unsigned long long* sink = nullptr;
  cudaDeviceProp prop{}; int dev = 0;
  cudaGetDevice(&dev); cudaGetDeviceProperties(&prop, dev);
  size_t const flush_bytes = std::max<size_t>(size_t(prop.l2CacheSize) * 2, size_t(128) << 20);
  if (cuda_ok(cudaMalloc(&blocks, kBlockBytes)) || cuda_ok(cudaMalloc(&out, kOutBytes)) ||
      cuda_ok(cudaMalloc(&flush, flush_bytes)) || cuda_ok(cudaMalloc(&sink, sizeof(*sink)))) return 310;
  cudaMemset(blocks, 0x33, kBlockBytes); cudaMemset(flush, 1, flush_bytes); cudaMemset(sink, 0, sizeof(*sink));
  launch_q4_layout<kBlocks, true>(blocks, out); launch_q4_layout<kBlocks, false>(blocks, out);
  if (cuda_ok(cudaDeviceSynchronize())) return 311;
  if (int e = bench_layout_variant<kBlocks, true>(blocks, out, flush, flush_bytes, sink, reps, physical_us))
    return 320 + e;
  if (int e = bench_layout_variant<kBlocks, false>(blocks, out, flush, flush_bytes, sink, reps, logical_us))
    return 330 + e;
  cudaFree(blocks); cudaFree(out); cudaFree(flush); cudaFree(sink);
  return 0;
}
