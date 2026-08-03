// Device half of quactlize's dlopen boundary. This file contains no torch types: every entry is a C ABI over raw
// host pointers, allocates the corresponding device buffers, launches the production kernel, synchronises, and
// copies the result back. The same source builds under plain nvcc for the independent local oracle and under hgcc
// through actlize's PPUToolchain.
//
// GEMV is CUDA-core-only. Keep this instantiation list narrow: k-quants use gs=16/32 and always provide an affine
// zero plane (Q3/Q6 use it for signed-code offset binary). No tensor-core or dp4a instruction is reachable here.
#define GEMV_GS_LIST(EMIT) EMIT(16) EMIT(32)
#define GEMV_QUANT_LIST(EMIT, G) EMIT(QuantOp::FinegrainedScaleZero, G)

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gguf_scale_prepass.hpp"
#include "gguf_vecdot.hpp"
#include "gemv_lowbit/gemv_launcher.hpp"
#include "gemv_lowbit/gemv_rt.hpp"

namespace {

using gguf_scale::KType;
using gguf_scale::vecdot::VecdotActivation;
using ppu_gemv::DevBuf;

template <KType T> constexpr int raw_block_bytes() {
  return T == KType::Q2_K ? 84 : T == KType::Q3_K ? 110 : T == KType::Q4_K ? 144
       : T == KType::Q5_K ? 176 : 210;
}

template <KType T, int RowsPerWarp, bool Grouped>
void launch_native_fixed(uint8_t const* blocks, VecdotActivation const* x, float* out,
                         int rows, int blocks_per_row, int const* offsets,
                         int max_rows, int experts) {
  constexpr int kThreads = 256;
  dim3 const grid(gguf_scale::vecdot::vecdot_grid_size<T, RowsPerWarp>(rows, kThreads),
                  Grouped ? max_rows : 1, Grouped ? experts : 1);
  gguf_scale::vecdot::vecdot_rows_kernel<T, RowsPerWarp, Grouped><<<grid, kThreads>>>(
      blocks, raw_block_bytes<T>(), x, out, rows, blocks_per_row, offsets);
}

template <KType T, bool Grouped>
void launch_native(uint8_t const* blocks, VecdotActivation const* x, float* out,
                   int rows, int blocks_per_row, int const* offsets,
                   int max_rows, int experts) {
  switch (gguf_scale::vecdot::vecdot_rows_per_warp<T>(rows, blocks_per_row)) {
    case 1: launch_native_fixed<T, 1, Grouped>(blocks, x, out, rows, blocks_per_row, offsets, max_rows, experts); break;
    case 2: launch_native_fixed<T, 2, Grouped>(blocks, x, out, rows, blocks_per_row, offsets, max_rows, experts); break;
    case 4: launch_native_fixed<T, 4, Grouped>(blocks, x, out, rows, blocks_per_row, offsets, max_rows, experts); break;
    case 8: launch_native_fixed<T, 8, Grouped>(blocks, x, out, rows, blocks_per_row, offsets, max_rows, experts); break;
  }
}

template <KType T>
int native_dense(uint8_t const* blocks, uint16_t const* x, float* out, int rows, int bpr) {
  DevBuf db(size_t(rows) * bpr * raw_block_bytes<T>());
  DevBuf dx(size_t(bpr) * 256 * sizeof(uint16_t));
  DevBuf dout(size_t(rows) * sizeof(float));
  db.from_host(blocks); dx.from_host(x);
  launch_native<T, false>(db.as<uint8_t>(), dx.as<VecdotActivation>(), dout.as<float>(), rows, bpr,
                          nullptr, 1, 1);
  ppu_gemv::rt_sync("native dense GEMV");
  dout.to_host(out);
  return 0;
}

template <KType T>
int native_moe(uint8_t const* blocks, uint16_t const* x, int const* offsets, float* out,
               int n, int bpr, int experts, int total_rows, int max_rows) {
  DevBuf db(size_t(experts) * n * bpr * raw_block_bytes<T>());
  DevBuf dx(size_t(total_rows) * bpr * 256 * sizeof(uint16_t));
  DevBuf doff(size_t(experts + 1) * sizeof(int));
  DevBuf dout(size_t(total_rows) * n * sizeof(float));
  db.from_host(blocks); dx.from_host(x); doff.from_host(offsets);
  launch_native<T, true>(db.as<uint8_t>(), dx.as<VecdotActivation>(), dout.as<float>(), n, bpr,
                         doff.as<int>(), max_rows, experts);
  ppu_gemv::rt_sync("native MoE GEMV");
  dout.to_host(out);
  return 0;
}

template <KType T>
__global__ void dequant_kernel(uint8_t const* blocks, cutlass::half_t* out, int count) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) gguf_scale::vecdot::dequantize_block<T>(blocks + int64_t(i) * raw_block_bytes<T>(), out + i * 256);
}

template <KType T>
int dequant(uint8_t const* blocks, uint16_t* out, int count) {
  DevBuf db(size_t(count) * raw_block_bytes<T>());
  DevBuf dout(size_t(count) * 256 * sizeof(uint16_t));
  db.from_host(blocks);
  dequant_kernel<T><<<(count + 127) / 128, 128>>>(db.as<uint8_t>(), dout.as<cutlass::half_t>(), count);
  ppu_gemv::rt_sync("GGUF dequantize");
  dout.to_host(out);
  return 0;
}

template <KType T, int ZMul>
int prepass(uint8_t const* blocks, uint16_t const* d, uint16_t const* dmin,
            int count, uint16_t* scale, uint16_t* zero) {
  using Tr = gguf_scale::Traits<T>;
  DevBuf db(size_t(count) * Tr::kBlockBytes), dd(size_t(count) * 2),
         dm(Tr::kHasMin ? size_t(count) * 2 : 0), ds(size_t(count) * Tr::kGroups * 2),
         dz(size_t(count) * Tr::kGroups * 2);
  db.from_host(blocks); dd.from_host(d); if constexpr (Tr::kHasMin) dm.from_host(dmin);
  gguf_scale::prepass::BlockDesc src{db.as<uint8_t>(), dd.as<cutlass::half_t>(),
      Tr::kHasMin ? dm.as<cutlass::half_t>() : nullptr, Tr::kBlockBytes, 0, 1, 0};
  gguf_scale::prepass::PlaneDesc dst{ds.as<cutlass::half_t>(), dz.as<cutlass::half_t>(), Tr::kGroups, 1};
  int const grid = gguf_scale::prepass::prepass_grid_size(count, 1, 256);
  gguf_scale::prepass::prepass_kernel<T, ZMul><<<grid, 256>>>(src, dst, count, 1);
  ppu_gemv::rt_sync("GGUF scale prepass");
  ds.to_host(scale); dz.to_host(zero);
  return 0;
}

template <ppu_gemv::WFormat F, int StepK, int Threads>
int lowbit(uint16_t const* act, uint8_t const* low, uint8_t const* high,
           uint16_t const* scale, uint16_t const* zero, uint16_t* out,
           int total_rows, int n, int k, int group_size, int experts,
           int const* offsets, int max_rows) {
  using D = ppu_gemv::KernelDetails<ppu_gemv::FP16DetailsA, F, ppu_gemv::WLayout::Native, StepK, Threads>;
  constexpr int LoBits = D::kLoBits, HiBits = D::kHiBits;
  int const weight_experts = experts > 0 ? experts : 1;
  int64_t const lo_per = int64_t(n) * k * LoBits / 8;
  int64_t const hi_per = HiBits ? int64_t(n) * k * HiBits / 8 : 0;
  int64_t const scale_per = int64_t(k / group_size) * n;
  DevBuf da(size_t(total_rows) * k * 2), dl(size_t(weight_experts) * lo_per),
         dh(size_t(weight_experts) * hi_per), ds(size_t(weight_experts) * scale_per * 2),
         dz(size_t(weight_experts) * scale_per * 2), dout(size_t(total_rows) * n * 2),
         doff(experts > 0 ? size_t(experts + 1) * sizeof(int) : 0);
  da.from_host(act); dl.from_host(low); if constexpr (HiBits != 0) dh.from_host(high);
  ds.from_host(scale); dz.from_host(zero); if (experts > 0) doff.from_host(offsets);

  ppu_gemv::Params p;
  p.act = da.p; p.weight = dl.p; p.weight_hi = dh.p; p.scales = ds.p; p.zeros = dz.p; p.out = dout.p;
  p.m = total_rows; p.n = n; p.k = k; p.groupsize = group_size;
  p.format = F; p.quant = ppu_gemv::QuantOp::FinegrainedScaleZero; p.layout = ppu_gemv::WLayout::Native;
  if (experts > 0) {
    p.num_experts = experts; p.row_offsets = doff.as<int>(); p.max_rows = max_rows;
    p.w_bytes_per_expert = lo_per; p.w_hi_bytes_per_expert = hi_per; p.scale_elems_per_expert = scale_per;
  }
  int const before = ppu_gemv::gemv_fail_count();
  bool const launched = ppu_gemv::launch_gemv<D, 8, 2>(p, 0);
  if (!launched || ppu_gemv::gemv_fail_count() != before) return 40;
  ppu_gemv::rt_sync("scale-first GEMV");
  dout.to_host(out);
  return 0;
}

}  // namespace

extern "C" int quactlize_ppu_vecdot(uint8_t const* b, int64_t block_bytes, uint16_t const* x, float* out,
                                      int rows, int bpr, int qtype) {
#define RUN(T) (block_bytes == raw_block_bytes<KType::T>() && rows > 0 && bpr > 0 \
                ? native_dense<KType::T>(b, x, out, rows, bpr) : 10)
  switch (qtype) { case 10: return RUN(Q2_K); case 11: return RUN(Q3_K); case 12: return RUN(Q4_K);
                   case 13: return RUN(Q5_K); case 14: return RUN(Q6_K); default: return 1; }
#undef RUN
}

extern "C" int quactlize_ppu_vecdot_moe(uint8_t const* b, int64_t block_bytes, uint16_t const* x,
                                          int const* offsets, float* out, int n, int bpr, int experts,
                                          int total_rows, int max_rows, int qtype) {
#define RUN(T) (block_bytes == raw_block_bytes<KType::T>() && n > 0 && bpr > 0 && experts > 0 && \
                total_rows > 0 && max_rows > 0 \
                ? native_moe<KType::T>(b, x, offsets, out, n, bpr, experts, total_rows, max_rows) : 11)
  switch (qtype) { case 10: return RUN(Q2_K); case 11: return RUN(Q3_K); case 12: return RUN(Q4_K);
                   case 13: return RUN(Q5_K); case 14: return RUN(Q6_K); default: return 2; }
#undef RUN
}

extern "C" int quactlize_ppu_dequantize(uint8_t const* b, int64_t block_bytes, uint16_t* out,
                                         int count, int qtype) {
#define RUN(T) (block_bytes == raw_block_bytes<KType::T>() && count > 0 ? dequant<KType::T>(b, out, count) : 12)
  switch (qtype) { case 10: return RUN(Q2_K); case 11: return RUN(Q3_K); case 12: return RUN(Q4_K);
                   case 13: return RUN(Q5_K); case 14: return RUN(Q6_K); default: return 3; }
#undef RUN
}

extern "C" int quactlize_ppu_prepass(uint8_t const* b, int64_t block_bytes, uint16_t const* d,
                                      uint16_t const* dm, int count, uint16_t* scale, uint16_t* zero,
                                      int groups, int qtype, int zmul) {
  (void)block_bytes; (void)groups;
#define RUN(T) (zmul == 8 ? prepass<KType::T, 8>(b, d, dm, count, scale, zero) \
                          : prepass<KType::T, 0>(b, d, dm, count, scale, zero))
  if (zmul != 0 && zmul != 8) return 4;
  switch (qtype) { case 10: return RUN(Q2_K); case 11: return RUN(Q3_K); case 12: return RUN(Q4_K);
                   case 13: return RUN(Q5_K); case 14: return RUN(Q6_K); default: return 5; }
#undef RUN
}

extern "C" int quactlize_ppu_gemv_lowbit(uint16_t const* a, uint8_t const* low, uint8_t const* high,
                                          uint16_t const* scale, uint16_t const* zero, uint16_t* out,
                                          int total_rows, int n, int k, int group_size, int qtype,
                                          int experts, int const* offsets, int max_rows) {
  switch (qtype) {
    case 10: return lowbit<ppu_gemv::WFormat::Int2,  16, 128>(a,low,high,scale,zero,out,total_rows,n,k,group_size,experts,offsets,max_rows);
    case 11: return lowbit<ppu_gemv::WFormat::Q3_21, 32,  64>(a,low,high,scale,zero,out,total_rows,n,k,group_size,experts,offsets,max_rows);
    case 12: return lowbit<ppu_gemv::WFormat::Int4,  16, 128>(a,low,high,scale,zero,out,total_rows,n,k,group_size,experts,offsets,max_rows);
    case 13: return lowbit<ppu_gemv::WFormat::Q5_41, 32,  64>(a,low,high,scale,zero,out,total_rows,n,k,group_size,experts,offsets,max_rows);
    case 14: return lowbit<ppu_gemv::WFormat::Q6_42, 16, 128>(a,low,high,scale,zero,out,total_rows,n,k,group_size,experts,offsets,max_rows);
    default: return 6;
  }
}
