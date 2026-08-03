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

template <KType T>
struct PrepassDevice {
  using Tr = gguf_scale::Traits<T>;
  uint8_t* blocks = nullptr;
  half_t *d = nullptr, *dmin = nullptr, *scale = nullptr, *zero = nullptr;
  uint8_t* flush = nullptr;
  unsigned long long* sink = nullptr;
  size_t total = 0, flush_bytes = 0;

  ~PrepassDevice() {
    cudaFree(blocks); cudaFree(d); cudaFree(dmin); cudaFree(scale); cudaFree(zero); cudaFree(flush); cudaFree(sink);
  }

  int allocate(int cols, int superblocks) {
    total = size_t(cols) * superblocks;
    cudaDeviceProp prop{};
    int dev = 0;
    if (cuda_ok(cudaGetDevice(&dev)) || cuda_ok(cudaGetDeviceProperties(&prop, dev))) return 1;
    flush_bytes = std::max<size_t>(size_t(prop.l2CacheSize) * 2, size_t(128) << 20);
    if (cuda_ok(cudaMalloc(&blocks, total * Tr::kBlockBytes)) || cuda_ok(cudaMalloc(&d, total * sizeof(half_t))) ||
        cuda_ok(cudaMalloc(&dmin, total * sizeof(half_t))) ||
        cuda_ok(cudaMalloc(&scale, total * Tr::kGroups * sizeof(half_t))) ||
        cuda_ok(cudaMalloc(&zero, total * Tr::kGroups * sizeof(half_t))) || cuda_ok(cudaMalloc(&flush, flush_bytes)) ||
        cuda_ok(cudaMalloc(&sink, sizeof(*sink)))) return 2;
    if (cuda_ok(cudaMemset(flush, 1, flush_bytes)) || cuda_ok(cudaMemset(sink, 0, sizeof(*sink)))) return 3;
    return 0;
  }

  BlockDesc src(int superblocks) const {
    return {blocks, d, Tr::kHasMin ? dmin : nullptr,
            int64_t(superblocks) * Tr::kBlockBytes, Tr::kBlockBytes, superblocks, 1};
  }
  PlaneDesc dst(int superblocks) const { return {scale, zero, int64_t(superblocks) * Tr::kGroups, 1}; }
  void flush_l2() const { flush_l2_kernel<<<4096, 256>>>(flush, flush_bytes, sink); }
};

template <KType T, bool Cooperative>
void launch_prepass(PrepassDevice<T> const& q, int cols, int superblocks) {
#ifndef GGUF_PROBE_THREADS
#define GGUF_PROBE_THREADS 256
#endif
  constexpr int kThreads = GGUF_PROBE_THREADS;
  if constexpr (Cooperative) {
    int const grid = gguf_scale::prepass::prepass_grid_size(cols, superblocks, kThreads);
    gguf_scale::prepass::prepass_kernel<T, 8><<<grid, kThreads>>>(
        q.src(superblocks), q.dst(superblocks), cols, superblocks);
  } else {
    int const total = cols * superblocks;
    gguf_scale::prepass::prepass_kernel_serial<T, 8><<<(total + kThreads - 1) / kThreads, kThreads>>>(
        q.src(superblocks), q.dst(superblocks), cols, superblocks);
  }
}

template <KType T, bool Cooperative>
int run_prepass(uint8_t const* blocks, uint16_t const* d, uint16_t const* dmin, uint16_t* scale, uint16_t* zero,
                int cols, int superblocks) {
  using Tr = gguf_scale::Traits<T>;
  PrepassDevice<T> q;
  if (int e = q.allocate(cols, superblocks)) return 100 + e;
  size_t const total = q.total;
  if (cuda_ok(cudaMemcpy(q.blocks, blocks, total * Tr::kBlockBytes, cudaMemcpyHostToDevice)) ||
      cuda_ok(cudaMemcpy(q.d, d, total * 2, cudaMemcpyHostToDevice)) ||
      (Tr::kHasMin && cuda_ok(cudaMemcpy(q.dmin, dmin, total * 2, cudaMemcpyHostToDevice))) ||
      cuda_ok(cudaMemset(q.scale, 0xa5, total * Tr::kGroups * 2)) ||
      cuda_ok(cudaMemset(q.zero, 0x5a, total * Tr::kGroups * 2))) return 110;
  if constexpr (Cooperative) launch_prepass<T, true>(q, cols, superblocks);
  else                       launch_prepass<T, false>(q, cols, superblocks);
  if (cuda_ok(cudaGetLastError()) || cuda_ok(cudaDeviceSynchronize())) return 111;
  if (cuda_ok(cudaMemcpy(scale, q.scale, total * Tr::kGroups * 2, cudaMemcpyDeviceToHost)) ||
      cuda_ok(cudaMemcpy(zero, q.zero, total * Tr::kGroups * 2, cudaMemcpyDeviceToHost))) return 112;
  return 0;
}

template <bool Cooperative>
int bench_variant(PrepassDevice<KType::Q4_K> const& q, int cols, int superblocks, int reps, float* usec) {
  cudaEvent_t begin{}, end{};
  if (cuda_ok(cudaEventCreate(&begin)) || cuda_ok(cudaEventCreate(&end))) return 1;
  std::vector<float> samples;
  samples.reserve(size_t(reps));
  for (int r = 0; r < reps + 5; ++r) {
    q.flush_l2();
    cudaEventRecord(begin);
    launch_prepass<KType::Q4_K, Cooperative>(q, cols, superblocks);
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

template <KType T>
struct VecdotDevice {
  static constexpr int kBlockBytes = T == KType::Q2_K ? 84 : T == KType::Q3_K ? 110
                                         : T == KType::Q4_K ? 144 : T == KType::Q5_K ? 176 : 210;
  uint8_t *blocks = nullptr, *flush = nullptr;
  float *x = nullptr, *out = nullptr;
  unsigned long long* sink = nullptr;
  size_t block_bytes = 0, x_bytes = 0, out_bytes = 0, flush_bytes = 0;

  ~VecdotDevice() { cudaFree(blocks); cudaFree(x); cudaFree(out); cudaFree(flush); cudaFree(sink); }

  int allocate(int rows, int blocks_per_row) {
    cudaDeviceProp prop{};
    int dev = 0;
    if (cuda_ok(cudaGetDevice(&dev)) || cuda_ok(cudaGetDeviceProperties(&prop, dev))) return 1;
    block_bytes = size_t(rows) * blocks_per_row * kBlockBytes;
    x_bytes = size_t(blocks_per_row) * gguf_scale::vecdot::kQK * sizeof(float);
    out_bytes = size_t(rows) * sizeof(float);
    flush_bytes = std::max<size_t>(size_t(prop.l2CacheSize) * 2, size_t(128) << 20);
    if (cuda_ok(cudaMalloc(&blocks, block_bytes)) || cuda_ok(cudaMalloc(&x, x_bytes)) ||
        cuda_ok(cudaMalloc(&out, out_bytes)) || cuda_ok(cudaMalloc(&flush, flush_bytes)) ||
        cuda_ok(cudaMalloc(&sink, sizeof(*sink)))) return 2;
    if (cuda_ok(cudaMemset(flush, 1, flush_bytes)) || cuda_ok(cudaMemset(sink, 0, sizeof(*sink)))) return 3;
    return 0;
  }

  void flush_l2() const { flush_l2_kernel<<<4096, 256>>>(flush, flush_bytes, sink); }
};

template <KType T, int RowsPerWarp = gguf_scale::vecdot::vecdot_preferred_rows_per_warp<T>(), bool Serial = false>
void launch_vecdot(VecdotDevice<T> const& q, int rows, int blocks_per_row) {
  constexpr int kThreads = 256;
  if constexpr (Serial) {
    gguf_scale::vecdot::vecdot_rows_kernel_serial<T><<<(rows + kThreads - 1) / kThreads, kThreads>>>(
        q.blocks, VecdotDevice<T>::kBlockBytes, q.x, q.out, rows, blocks_per_row);
  } else {
    gguf_scale::vecdot::vecdot_rows_kernel<T, RowsPerWarp>
        <<<gguf_scale::vecdot::vecdot_grid_size<T, RowsPerWarp>(rows, kThreads), kThreads>>>(
            q.blocks, VecdotDevice<T>::kBlockBytes, q.x, q.out, rows, blocks_per_row);
  }
}

template <KType T>
int run_vecdot(uint8_t const* blocks, float const* x, float* out, int rows, int blocks_per_row) {
  VecdotDevice<T> q;
  if (int e = q.allocate(rows, blocks_per_row)) return 400 + e;
  if (cuda_ok(cudaMemcpy(q.blocks, blocks, q.block_bytes, cudaMemcpyHostToDevice)) ||
      cuda_ok(cudaMemcpy(q.x, x, q.x_bytes, cudaMemcpyHostToDevice))) return 410;
  launch_vecdot<T>(q, rows, blocks_per_row);
  if (cuda_ok(cudaGetLastError()) || cuda_ok(cudaDeviceSynchronize()) ||
      cuda_ok(cudaMemcpy(out, q.out, q.out_bytes, cudaMemcpyDeviceToHost))) return 411;
  return 0;
}

template <KType T, bool Cold, int RowsPerWarp = gguf_scale::vecdot::vecdot_preferred_rows_per_warp<T>(),
          bool Serial = false>
int bench_vecdot_variant(VecdotDevice<T> const& q, int rows, int blocks_per_row, int reps, float* usec) {
  cudaEvent_t begin{}, end{};
  if (cuda_ok(cudaEventCreate(&begin)) || cuda_ok(cudaEventCreate(&end))) return 1;
  std::vector<float> samples;
  samples.reserve(size_t(reps));
  if constexpr (!Cold) {
    // A 5090 idles at 180 MHz and these launches are too short to reach a stable boost state by themselves. Repeated
    // untimed launches both make the operand genuinely L2-resident and bring the SM clock up before warm samples.
    for (int warmup = 0; warmup < 100; ++warmup)
      launch_vecdot<T, RowsPerWarp, Serial>(q, rows, blocks_per_row);
    if (cuda_ok(cudaDeviceSynchronize())) return 2;
  }
  for (int r = 0; r < reps + 5; ++r) {
    // The flush and any dirty writeback it induces precede `begin` in the same stream. The events contain one
    // vecdot launch and nothing else; allocations, initialisation and transfers all happen outside this loop.
    if constexpr (Cold) q.flush_l2();
    if (cuda_ok(cudaEventRecord(begin))) return 3;
    launch_vecdot<T, RowsPerWarp, Serial>(q, rows, blocks_per_row);
    if (cuda_ok(cudaGetLastError()) || cuda_ok(cudaEventRecord(end)) || cuda_ok(cudaEventSynchronize(end))) return 4;
    float ms = 0.f;
    if (cuda_ok(cudaEventElapsedTime(&ms, begin, end))) return 5;
    if (r >= 5) samples.push_back(ms * 1000.f);
  }
  cudaEventDestroy(begin); cudaEventDestroy(end);
  std::sort(samples.begin(), samples.end());
  *usec = samples[samples.size() / 2];
  return 0;
}

template <KType T, int RowsPerWarp = gguf_scale::vecdot::vecdot_preferred_rows_per_warp<T>(), bool Serial = false>
int bench_vecdot(int rows, int blocks_per_row, int reps, float* cold_us, float* warm_us, double* bytes) {
  VecdotDevice<T> q;
  if (int e = q.allocate(rows, blocks_per_row)) return 420 + e;
  // 0x01 is finite in the half scale fields and 0x3f is finite as fp32 activation data. Values do not affect the
  // instruction or address trace, but avoiding NaNs makes the same allocation useful to profilers and sanitizers.
  if (cuda_ok(cudaMemset(q.blocks, 0x01, q.block_bytes)) || cuda_ok(cudaMemset(q.x, 0x3f, q.x_bytes)) ||
      cuda_ok(cudaMemset(q.out, 0, q.out_bytes))) return 430;
  launch_vecdot<T, RowsPerWarp, Serial>(q, rows, blocks_per_row);
  if (cuda_ok(cudaGetLastError()) || cuda_ok(cudaDeviceSynchronize())) return 431;
  if (int e = bench_vecdot_variant<T, true, RowsPerWarp, Serial>(q, rows, blocks_per_row, reps, cold_us))
    return 440 + e;
  if (int e = bench_vecdot_variant<T, false, RowsPerWarp, Serial>(q, rows, blocks_per_row, reps, warm_us))
    return 450 + e;
  *bytes = double(q.block_bytes + q.x_bytes + q.out_bytes);
  return 0;
}

template <KType T>
int bench_vecdot_config(int rows_per_warp, int rows, int blocks_per_row, int reps,
                        float* cold_us, float* warm_us, double* bytes) {
  switch (rows_per_warp) {
    case 0:  return bench_vecdot<T, 1, true>(rows, blocks_per_row, reps, cold_us, warm_us, bytes);
    case 1:  return bench_vecdot<T, 1>(rows, blocks_per_row, reps, cold_us, warm_us, bytes);
    case 2:  return bench_vecdot<T, 2>(rows, blocks_per_row, reps, cold_us, warm_us, bytes);
    case 4:  return bench_vecdot<T, 4>(rows, blocks_per_row, reps, cold_us, warm_us, bytes);
    case 8:  return bench_vecdot<T, 8>(rows, blocks_per_row, reps, cold_us, warm_us, bytes);
    case 16: return bench_vecdot<T, 16>(rows, blocks_per_row, reps, cold_us, warm_us, bytes);
    default: return 492;
  }
}

}  // namespace

extern "C" int quactlize_cuda_q4_prepass(uint8_t const* blocks, uint16_t const* d, uint16_t const* dmin,
                                          uint16_t* scale, uint16_t* zero, int cols, int superblocks,
                                          int cooperative) {
  return cooperative ? run_prepass<KType::Q4_K, true>(blocks, d, dmin, scale, zero, cols, superblocks)
                     : run_prepass<KType::Q4_K, false>(blocks, d, dmin, scale, zero, cols, superblocks);
}

extern "C" int quactlize_cuda_prepass(uint8_t const* blocks, uint16_t const* d, uint16_t const* dmin,
                                       uint16_t* scale, uint16_t* zero, int cols, int superblocks,
                                       int qtype, int cooperative) {
#define RUN_PREPASS(TYPE) (cooperative \
    ? run_prepass<KType::TYPE, true>(blocks, d, dmin, scale, zero, cols, superblocks) \
    : run_prepass<KType::TYPE, false>(blocks, d, dmin, scale, zero, cols, superblocks))
  switch (qtype) {
    case 10: return RUN_PREPASS(Q2_K);
    case 11: return RUN_PREPASS(Q3_K);
    case 12: return RUN_PREPASS(Q4_K);
    case 13: return RUN_PREPASS(Q5_K);
    case 14: return RUN_PREPASS(Q6_K);
    default: return 190;
  }
#undef RUN_PREPASS
}

extern "C" int quactlize_cuda_q4_prepass_bench(int scale_factor, int reps, float* serial_us, float* coop_us,
                                                double* bytes) {
  int const cols = 8 * 2048 * scale_factor;
  int const superblocks = 8;
  PrepassDevice<KType::Q4_K> q;
  if (int e = q.allocate(cols, superblocks)) return 200 + e;

  // Inputs are allocated and initialised before timing. Their particular values do not affect the access pattern;
  // correctness is a separate call supplied with non-degenerate data from the official Python golden.
  if (cuda_ok(cudaMemset(q.blocks, 0x5a, q.total * 12)) || cuda_ok(cudaMemset(q.d, 0x34, q.total * 2)) ||
      cuda_ok(cudaMemset(q.dmin, 0x30, q.total * 2))) return 210;
  launch_prepass<KType::Q4_K, false>(q, cols, superblocks);
  launch_prepass<KType::Q4_K, true>(q, cols, superblocks);
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

extern "C" int quactlize_cuda_vecdot(uint8_t const* blocks, float const* x, float* out,
                                      int rows, int blocks_per_row, int qtype) {
#define RUN_VECDOT(TYPE) run_vecdot<KType::TYPE>(blocks, x, out, rows, blocks_per_row)
  switch (qtype) {
    case 10: return RUN_VECDOT(Q2_K);
    case 11: return RUN_VECDOT(Q3_K);
    case 12: return RUN_VECDOT(Q4_K);
    case 13: return RUN_VECDOT(Q5_K);
    case 14: return RUN_VECDOT(Q6_K);
    default: return 490;
  }
#undef RUN_VECDOT
}

extern "C" int quactlize_cuda_vecdot_bench(int qtype, int rows, int blocks_per_row, int reps,
                                             float* cold_us, float* warm_us, double* bytes) {
#define BENCH_VECDOT(TYPE) bench_vecdot<KType::TYPE>(rows, blocks_per_row, reps, cold_us, warm_us, bytes)
  switch (qtype) {
    case 10: return BENCH_VECDOT(Q2_K);
    case 11: return BENCH_VECDOT(Q3_K);
    case 12: return BENCH_VECDOT(Q4_K);
    case 13: return BENCH_VECDOT(Q5_K);
    case 14: return BENCH_VECDOT(Q6_K);
    default: return 491;
  }
#undef BENCH_VECDOT
}

extern "C" int quactlize_cuda_vecdot_bench_config(int qtype, int rows_per_warp, int rows, int blocks_per_row,
                                                    int reps, float* cold_us, float* warm_us, double* bytes) {
#define BENCH_CONFIG(TYPE) \
  bench_vecdot_config<KType::TYPE>(rows_per_warp, rows, blocks_per_row, reps, cold_us, warm_us, bytes)
  switch (qtype) {
    case 10: return BENCH_CONFIG(Q2_K);
    case 11: return BENCH_CONFIG(Q3_K);
    case 12: return BENCH_CONFIG(Q4_K);
    case 13: return BENCH_CONFIG(Q5_K);
    case 14: return BENCH_CONFIG(Q6_K);
    default: return 493;
  }
#undef BENCH_CONFIG
}
