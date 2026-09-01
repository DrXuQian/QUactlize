// Device half of quactlize's dlopen boundary. This file contains no torch types: every entry is a C ABI over raw
// host pointers, allocates the corresponding device buffers, launches the production kernel, synchronises, and
// copies the result back. The same source builds under plain nvcc for the independent local oracle and under hgcc
// through actlize's PPUToolchain.
//
// GEMV is CUDA-core-only. Keep this instantiation list narrow: k-quants use gs=16/32 and always provide an affine
// zero plane (Q3/Q6 use it for signed-code offset binary). No tensor-core or dp4a instruction is reachable here.
#define GEMV_GS_LIST(EMIT) EMIT(16) EMIT(32)
#define GEMV_QUANT_LIST(EMIT, G) EMIT(QuantOp::FinegrainedScaleZero, G)
#define GEMV_ENABLE_BIAS 0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "gguf_bc_vecdot.hpp"
#include "gguf_scale_prepass.hpp"
#include "gguf_vecdot.hpp"
#include "gemv_lowbit/gemv_launcher.hpp"
#include "gemv_lowbit/gemv_rt.hpp"
#include "ppu_dense_configs.inc"
#include "ppu_format_config.hpp"
#include "ppu_grouped_configs.inc"
#include "ppu_placed_arrangement.hpp"
#include "quactlize_ppu_device.h"


namespace {

using gguf_scale::KType;
using gguf_scale::vecdot::VecdotActivation;
using ppu_gemv::DevBuf;


template <ppu_gemv::WFormat F, int StepK, int Threads>
bool lowbit_dense_config_valid(int m, int n, int k, int group_size) {
  using D = ppu_gemv::KernelDetails<ppu_gemv::FP16DetailsA, F, ppu_gemv::WLayout::Native, StepK, Threads>;
  ppu_gemv::Params p;
  p.m = m; p.n = n; p.k = k; p.groupsize = group_size;
  p.format = F; p.quant = ppu_gemv::QuantOp::FinegrainedScaleZero; p.layout = ppu_gemv::WLayout::Native;
  // Validity is about the compiled problem shape. Non-null sentinels state the artifact contract that the caller
  // has already established; no pointer is dereferenced by gemv_config_invalid_reason.
  p.weight_hi = ppu_gemv::is_two_plane(F) ? reinterpret_cast<void const*>(uintptr_t(1)) : nullptr;
  p.zeros = reinterpret_cast<void const*>(uintptr_t(1));
  return ppu_gemv::gemv_config_invalid_reason<D, 8>(p) == nullptr;
}

template <KType T> constexpr int raw_block_bytes() {
  return T == KType::Q2_K ? 84 : T == KType::Q3_K ? 110 : T == KType::Q4_K ? 144
       : T == KType::Q5_K ? 176 : 210;
}

template <KType T, int RowsPerWarp, bool Grouped, bool PerRowActivation = false>
void launch_native_fixed(uint8_t const* blocks, VecdotActivation const* x, float* out,
                         int rows, int blocks_per_row, int const* offsets,
                         int max_rows, int experts, gemv_stream_t stream = nullptr) {
  constexpr int kThreads = 256;
  dim3 const grid(gguf_scale::vecdot::vecdot_grid_size<T, RowsPerWarp>(rows, kThreads),
                  Grouped ? max_rows : 1, Grouped ? experts : 1);
  gguf_scale::vecdot::vecdot_rows_kernel<T, RowsPerWarp, Grouped, PerRowActivation><<<grid, kThreads, 0, stream>>>(
      blocks, raw_block_bytes<T>(), x, out, rows, blocks_per_row, offsets);
}

template <KType T, bool Grouped, bool PerRowActivation = false>
void launch_native(uint8_t const* blocks, VecdotActivation const* x, float* out,
                   int rows, int blocks_per_row, int const* offsets,
                   int max_rows, int experts, gemv_stream_t stream = nullptr) {
  switch (gguf_scale::vecdot::vecdot_rows_per_warp<T>(rows, blocks_per_row)) {
    case 1: launch_native_fixed<T, 1, Grouped, PerRowActivation>(blocks, x, out, rows, blocks_per_row, offsets, max_rows, experts, stream); break;
    case 2: launch_native_fixed<T, 2, Grouped, PerRowActivation>(blocks, x, out, rows, blocks_per_row, offsets, max_rows, experts, stream); break;
    case 4: launch_native_fixed<T, 4, Grouped, PerRowActivation>(blocks, x, out, rows, blocks_per_row, offsets, max_rows, experts, stream); break;
    case 8: launch_native_fixed<T, 8, Grouped, PerRowActivation>(blocks, x, out, rows, blocks_per_row, offsets, max_rows, experts, stream); break;
  }
}

// gguf_vecdot's public block primitive pairs every raw block row with its own activation slice. This is not the
// dense GEMV contract below, where all output rows share one activation. Keeping separate functions at the ABI
// boundary makes the distinction impossible to erase with the coincidental blocks_per_row=1 shape.
template <KType T>
int native_rows(uint8_t const* blocks, uint16_t const* x, float* out, int rows, int bpr) {
  ppu_gemv::rt_clear_error();
  DevBuf db(size_t(rows) * bpr * raw_block_bytes<T>());
  DevBuf dx(size_t(rows) * bpr * 256 * sizeof(uint16_t));
  DevBuf dout(size_t(rows) * sizeof(float));
  db.from_host(blocks); dx.from_host(x);
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  launch_native<T, false, true>(db.as<uint8_t>(), dx.as<VecdotActivation>(), dout.as<float>(), rows, bpr,
                                nullptr, 1, 1);
  ppu_gemv::rt_sync("native GGUF block-row vecdot");
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  return ppu_gemv::rt_copy_output(dout, out, size_t(rows));
}

template <KType T>
int native_dense(uint8_t const* blocks, uint16_t const* x, float* out, int rows, int bpr) {
  ppu_gemv::rt_clear_error();
  DevBuf db(size_t(rows) * bpr * raw_block_bytes<T>());
  DevBuf dx(size_t(bpr) * 256 * sizeof(uint16_t));
  DevBuf dout(size_t(rows) * sizeof(float));
  db.from_host(blocks); dx.from_host(x);
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  launch_native<T, false>(db.as<uint8_t>(), dx.as<VecdotActivation>(), dout.as<float>(), rows, bpr,
                          nullptr, 1, 1);
  ppu_gemv::rt_sync("native dense GEMV");
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  return ppu_gemv::rt_copy_output(dout, out, size_t(rows));
}

template <KType T>
int native_dense_device(uint8_t const* blocks, uint16_t const* x, float* out,
                        int rows, int bpr, gemv_stream_t stream) {
  launch_native<T, false>(blocks, reinterpret_cast<VecdotActivation const*>(x), out,
                          rows, bpr, nullptr, 1, 1, stream);
  return ppu_gemv::rt_check_launch("native dense GEMV enqueue") ? 0 : ppu_gemv::kRuntimeError;
}

template <KType T>
int native_moe(uint8_t const* blocks, uint16_t const* x, int const* offsets, float* out,
               int n, int bpr, int experts, int total_rows, int max_rows) {
  ppu_gemv::rt_clear_error();
  DevBuf db(size_t(experts) * n * bpr * raw_block_bytes<T>());
  DevBuf dx(size_t(total_rows) * bpr * 256 * sizeof(uint16_t));
  DevBuf doff(size_t(experts + 1) * sizeof(int));
  DevBuf dout(size_t(total_rows) * n * sizeof(float));
  db.from_host(blocks); dx.from_host(x); doff.from_host(offsets);
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  launch_native<T, true>(db.as<uint8_t>(), dx.as<VecdotActivation>(), dout.as<float>(), n, bpr,
                         doff.as<int>(), max_rows, experts);
  ppu_gemv::rt_sync("native MoE GEMV");
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  return ppu_gemv::rt_copy_output(dout, out, size_t(total_rows) * n);
}

// The merged BC resident artifact: xplane-placed low/high codes plus byte-neutral packed scale units. The CUDA-core
// decode path consumes those exact bytes; unlike SCALE_FIRST it neither materialises nor reads fp16 affine planes.
// experts==0 is one dense decode batch with 1..7 rows. grid.y owns the batch
// row, so this remains one launch and one shared resident weight artifact.
// Grouped mode uses the same gathered-row/offset contract as native_moe.
template <KType T, int ArtifactTileK>
int bc_gemv_device(uint16_t const* x, uint8_t const* low, uint8_t const* high, uint8_t const* units,
                   int const* offsets, float* out, int total_rows, int n, int k, int experts, int max_rows,
                   gemv_stream_t stream) {
  int const bpr = k / 256;
  if (experts > 0) {
    gguf_scale::bc_vecdot::launch<T, ArtifactTileK, true>(low, high, units, reinterpret_cast<VecdotActivation const*>(x),
                                           offsets, out, n, bpr, experts, max_rows, stream);
  } else {
    gguf_scale::bc_vecdot::launch<T, ArtifactTileK, false>(low, high, units, reinterpret_cast<VecdotActivation const*>(x),
                                            nullptr, out, n, bpr, 1, total_rows, stream);
  }
  return ppu_gemv::rt_check_launch(experts > 0 ? "BC MoE GEMV enqueue" : "BC dense GEMV enqueue")
      ? 0 : ppu_gemv::kRuntimeError;
}

template <KType T, int ArtifactTileK>
int bc_gemv(uint16_t const* x, uint8_t const* low, uint8_t const* high, uint8_t const* units,
            int const* offsets, float* out, int total_rows, int n, int k, int experts, int max_rows) {
  ppu_gemv::rt_clear_error();
  using U = gguf_scale::packed_unit::Unit<T>;
  using BT = gguf_scale::bc_vecdot::Traits<T>;
  int const weight_experts = experts > 0 ? experts : 1;
  int const bpr = k / 256;
  int const num_units = bpr / U::kSbPerUnit;
  size_t const low_bytes = size_t(weight_experts) * n * k * BT::Lo / 8;
  size_t const high_bytes = size_t(weight_experts) * n * k * BT::Hi / 8;
  size_t const unit_bytes = size_t(weight_experts) * num_units * n * U::kUnitTotal;
  DevBuf dx(size_t(total_rows) * k * sizeof(uint16_t)), dl(low_bytes), dh(high_bytes), du(unit_bytes),
         doff(experts > 0 ? size_t(experts + 1) * sizeof(int) : 0),
         dout(size_t(total_rows) * n * sizeof(float));
  dx.from_host(x); dl.from_host(low); if constexpr (BT::Hi != 0) dh.from_host(high); du.from_host(units);
  if (experts > 0) doff.from_host(offsets);
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  int const launch_rc = bc_gemv_device<T, ArtifactTileK>(
      dx.as<uint16_t>(), dl.as<uint8_t>(), dh.as<uint8_t>(), du.as<uint8_t>(),
      doff.as<int>(), dout.as<float>(), total_rows, n, k, experts, max_rows, nullptr);
  if (launch_rc) return launch_rc;
  ppu_gemv::rt_sync(experts > 0 ? "BC MoE GEMV" : "BC dense GEMV");
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  return ppu_gemv::rt_copy_output(dout, out, size_t(total_rows) * n);
}

template <KType T, bool Device>
int bc_gemv_arrangement_dispatch(
    uint16_t const* x, uint8_t const* low, uint8_t const* high, uint8_t const* units,
    int const* offsets, float* out, int total_rows, int n, int k, int experts, int max_rows,
    int artifact_tile_k, gemv_stream_t stream = nullptr) {
#define QUACTLIZE_BC_A(A) case A:                                                       \
  if constexpr (gguf_scale::bc_vecdot::arrangement_supported_v<T, A>) {                 \
    if constexpr (Device)                                                               \
      return bc_gemv_device<T,A>(x,low,high,units,offsets,out,total_rows,n,k,experts,max_rows,stream); \
    else                                                                                 \
      return bc_gemv<T,A>(x,low,high,units,offsets,out,total_rows,n,k,experts,max_rows);  \
  } else return 23
  switch (artifact_tile_k) {
    QUACTLIZE_BC_A(32); QUACTLIZE_BC_A(64); QUACTLIZE_BC_A(128); QUACTLIZE_BC_A(256);
    default: return 23;
  }
#undef QUACTLIZE_BC_A
}

template <KType T>
__global__ void dequant_kernel(uint8_t const* blocks, cutlass::half_t* out, int count) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) gguf_scale::vecdot::dequantize_block<T>(blocks + int64_t(i) * raw_block_bytes<T>(), out + i * 256);
}

template <KType T>
int dequant(uint8_t const* blocks, uint16_t* out, int count) {
  ppu_gemv::rt_clear_error();
  DevBuf db(size_t(count) * raw_block_bytes<T>());
  DevBuf dout(size_t(count) * 256 * sizeof(uint16_t));
  db.from_host(blocks);
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  dequant_kernel<T><<<(count + 127) / 128, 128>>>(db.as<uint8_t>(), dout.as<cutlass::half_t>(), count);
  ppu_gemv::rt_sync("GGUF dequantize");
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  return ppu_gemv::rt_copy_output(dout, out, size_t(count) * 256);
}

template <KType T, int ZMul>
int prepass(uint8_t const* blocks, uint16_t const* d, uint16_t const* dmin,
            int count, uint16_t* scale, uint16_t* zero) {
  ppu_gemv::rt_clear_error();
  using Tr = gguf_scale::Traits<T>;
  DevBuf db(size_t(count) * Tr::kBlockBytes), dd(size_t(count) * 2),
         dm(Tr::kHasMin ? size_t(count) * 2 : 0), ds(size_t(count) * Tr::kGroups * 2),
         dz(size_t(count) * Tr::kGroups * 2);
  db.from_host(blocks); dd.from_host(d); if constexpr (Tr::kHasMin) dm.from_host(dmin);
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  gguf_scale::prepass::BlockDesc src{db.as<uint8_t>(), dd.as<cutlass::half_t>(),
      Tr::kHasMin ? dm.as<cutlass::half_t>() : nullptr, Tr::kBlockBytes, 0, 1, 0};
  gguf_scale::prepass::PlaneDesc dst{ds.as<cutlass::half_t>(), dz.as<cutlass::half_t>(), Tr::kGroups, 1};
  auto const args = gguf_scale::prepass::make_prepass_kernel_args(src, dst, count, 1);
  int const grid = gguf_scale::prepass::prepass_grid_size(count, 1, 256);
  gguf_scale::prepass::prepass_kernel<T, ZMul><<<grid, 256>>>(args);
  ppu_gemv::rt_sync("GGUF scale prepass");
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  return ppu_gemv::rt_copy_two_outputs(ds, scale, dz, zero, size_t(count) * Tr::kGroups);
}


template <KType T, int ZMul>
int prepass_unit(uint8_t const* units, uint16_t* scale, uint16_t* zero,
                 int n, int k, int experts) {
  ppu_gemv::rt_clear_error();
  using U = gguf_scale::packed_unit::Unit<T>;
  int const num_superblocks = k / 256;
  if (num_superblocks % U::kSbPerUnit) return 6;
  int const num_units = num_superblocks / U::kSbPerUnit;
  int64_t const plane_elems = int64_t(experts) * num_superblocks * U::kGroups * n;
  size_t const unit_bytes = size_t(experts) * num_units * n * U::kUnitTotal;
  DevBuf du(unit_bytes), ds(size_t(plane_elems) * 2), dz(size_t(plane_elems) * 2);
  du.from_host(units);
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  int64_t const expert_stride = int64_t(num_superblocks) * U::kGroups * n;
  gguf_scale::prepass::UnitPlaneDesc dst{
      ds.as<cutlass::half_t>(), dz.as<cutlass::half_t>(), expert_stride, n, 1};
  auto const args = gguf_scale::prepass::make_unit_prepass_kernel_args(
      du.as<uint8_t>(), dst, experts, n, num_superblocks);
  int const grid = gguf_scale::prepass::prepass_unit_grid_size<T>(experts, n, num_superblocks, 256);
  gguf_scale::prepass::prepass_unit_kernel<T, ZMul><<<grid, 256>>>(args);
  ppu_gemv::rt_sync("packed-unit scale prepass");
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  return ppu_gemv::rt_copy_two_outputs(ds, scale, dz, zero, size_t(plane_elems));
}

template <ppu_gemv::WFormat F, int StepK, int Threads>
int lowbit_device(uint16_t const* act, uint8_t const* low, uint8_t const* high,
                  uint16_t const* scale, uint16_t const* zero, uint16_t* out,
                  int total_rows, int n, int k, int group_size, int experts,
                  int const* offsets, int max_rows, gemv_stream_t stream) {
  using D = ppu_gemv::KernelDetails<ppu_gemv::FP16DetailsA, F, ppu_gemv::WLayout::Native, StepK, Threads>;
  constexpr int LoBits = D::kLoBits, HiBits = D::kHiBits;
  int64_t const lo_per = int64_t(n) * k * LoBits / 8;
  int64_t const hi_per = HiBits ? int64_t(n) * k * HiBits / 8 : 0;
  int64_t const scale_per = int64_t(k / group_size) * n;

  ppu_gemv::Params p;
  p.act = act; p.weight = low; p.weight_hi = high; p.scales = scale; p.zeros = zero; p.out = out;
  p.m = total_rows; p.n = n; p.k = k; p.groupsize = group_size;
  p.format = F; p.quant = ppu_gemv::QuantOp::FinegrainedScaleZero; p.layout = ppu_gemv::WLayout::Native;
  if (experts > 0) {
    p.num_experts = experts; p.row_offsets = offsets; p.max_rows = max_rows;
    p.w_bytes_per_expert = lo_per; p.w_hi_bytes_per_expert = hi_per; p.scale_elems_per_expert = scale_per;
  }
  int const before = ppu_gemv::gemv_fail_count();
  bool const launched = ppu_gemv::launch_gemv<D, 8, 2>(p, stream);
  if (!launched || ppu_gemv::gemv_fail_count() != before) return 40;
  return ppu_gemv::rt_check_launch("scale-first GEMV enqueue") ? 0 : ppu_gemv::kRuntimeError;
}

template <ppu_gemv::WFormat F, int StepK, int Threads>
int lowbit(uint16_t const* act, uint8_t const* low, uint8_t const* high,
           uint16_t const* scale, uint16_t const* zero, uint16_t* out,
           int total_rows, int n, int k, int group_size, int experts,
           int const* offsets, int max_rows) {
  ppu_gemv::rt_clear_error();
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
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;

  int const launch_rc = lowbit_device<F, StepK, Threads>(
      da.as<uint16_t>(), dl.as<uint8_t>(), dh.as<uint8_t>(), ds.as<uint16_t>(), dz.as<uint16_t>(),
      dout.as<uint16_t>(), total_rows, n, k, group_size, experts, doff.as<int>(), max_rows, nullptr);
  if (launch_rc) return launch_rc;
  ppu_gemv::rt_sync("scale-first GEMV");
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  return ppu_gemv::rt_copy_output(dout, out, size_t(total_rows) * n);
}

}  // namespace

// Build identity for the multi-library loader and bundle verifier. -1 is the
// non-packed/default library; FMT0..4 are the format-selected packed builds.
// Keep this a host-only C ABI function so inspecting the library never needs a
// PPU device or a kernel launch.
extern "C" int32_t quactlize_ppu_build_packed_format_v1() {
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT)
  return PPU_PACKED_FORMAT;
#else
  return -1;
#endif
}

// The loaded library owns exactly one canonical K-pack byte map.  Keep this
// query independent of device state: loaders use it before allocating or
// uploading a resident tensor.  In particular, never infer a map from qtype
// in the default/ScaleFirst library, because that library has no
// format-selected fully-quantized reader.
extern "C" int quactlize_ppu_canonical_arrangement_v2(
    int qtype, quactlize_ppu_placed_arrangement_v2* out) {
  if (!out) return 23;
  std::memset(out, 0, sizeof(*out));
  auto const& format = ppu_formats::for_qtype(qtype);
  if (format.qtype < 0) return 22;

#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT)
  auto const& selected = ppu_formats::for_packed_format(PPU_PACKED_FORMAT);
  if (selected.qtype < 0 || qtype != selected.qtype) return 29;

  *out = PPU_PACKED_FORMAT == 0
      ? ppu_arrangements::q4_kpack4_transpose_v1()
      : ppu_arrangements::kquant_kpack_transpose_v1(qtype);
  return 0;
#else
  (void)format;
  return 29;
#endif
}

extern "C" int quactlize_ppu_vecdot(uint8_t const* b, int64_t block_bytes, uint16_t const* x, float* out,
                                      int rows, int bpr, int qtype) {
#define RUN(T) (block_bytes == raw_block_bytes<KType::T>() && rows > 0 && bpr > 0 \
                ? native_rows<KType::T>(b, x, out, rows, bpr) : 10)
  switch (qtype) { case 10: return RUN(Q2_K); case 11: return RUN(Q3_K); case 12: return RUN(Q4_K);
                   case 13: return RUN(Q5_K); case 14: return RUN(Q6_K); default: return 1; }
#undef RUN
}

extern "C" int quactlize_ppu_vecdot_dense(uint8_t const* b, int64_t block_bytes, uint16_t const* x, float* out,
                                            int rows, int bpr, int qtype) {
#define RUN(T) (block_bytes == raw_block_bytes<KType::T>() && rows > 0 && bpr > 0 \
                ? native_dense<KType::T>(b, x, out, rows, bpr) : 12)
  switch (qtype) { case 10: return RUN(Q2_K); case 11: return RUN(Q3_K); case 12: return RUN(Q4_K);
                   case 13: return RUN(Q5_K); case 14: return RUN(Q6_K); default: return 3; }
#undef RUN
}

extern "C" int quactlize_ppu_vecdot_dense_dev_v1(uint8_t const* b, int64_t block_bytes,
                                                    uint16_t const* x, float* out,
                                                    int rows, int bpr, int qtype, void* stream) {
  if (!b || !x || !out || rows <= 0 || bpr <= 0) return 12;
  ppu_gemv::rt_clear_error();
  gemv_stream_t const s = static_cast<gemv_stream_t>(stream);
#define RUN(T) (block_bytes == raw_block_bytes<KType::T>() \
                ? native_dense_device<KType::T>(b, x, out, rows, bpr, s) : 12)
  switch (qtype) { case 10: return RUN(Q2_K); case 11: return RUN(Q3_K); case 12: return RUN(Q4_K);
                   case 13: return RUN(Q5_K); case 14: return RUN(Q6_K); default: return 3; }
#undef RUN
}

extern "C" int32_t quactlize_ppu_gemv_lowbit_config_valid_v1(
    int m, int n, int k, int group_size, int qtype, char const* config_name) {
  auto const& format = ppu_formats::for_qtype(qtype);
  bool const compiled_name = !config_name || !config_name[0] ||
                             std::strcmp(config_name, QUACTLIZE_PPU_DENSE_CUDA_CONFIG_NAME) == 0;
  // N%256 keeps this family inside the dense route whose compiled tensor default can serve a cache miss. M>=16 is
  // deliberately not rejected here: TRT-LLM's cutoff is a profiling-cost prune, while this predicate answers only
  // legality. The launcher tiles its largest exact specialization for those larger problems.
  // K%256 is the resident GGUF artifact's packed-superblock contract, not a GEMV scheduling restriction. The
  // predicated specialization below it accepts a final partial CtaK; the unpredicated specialization remains the
  // exact path when K fills every CTA iteration.
  if (!compiled_name || format.qtype != qtype || n <= 0 || n % 256 != 0 || k <= 0 || k % 256 != 0 ||
      group_size != format.group_size) return 0;
  switch (qtype) {
    case 10: return lowbit_dense_config_valid<ppu_gemv::WFormat::Int2,  16, 128>(m,n,k,group_size);
    case 11: return lowbit_dense_config_valid<ppu_gemv::WFormat::Q3_21, 32,  64>(m,n,k,group_size);
    case 12: return lowbit_dense_config_valid<ppu_gemv::WFormat::Int4,  16, 128>(m,n,k,group_size);
    case 13: return lowbit_dense_config_valid<ppu_gemv::WFormat::Q5_41, 32,  64>(m,n,k,group_size);
    case 14: return lowbit_dense_config_valid<ppu_gemv::WFormat::Q6_42, 16, 128>(m,n,k,group_size);
    default: return 0;
  }
}

extern "C" int32_t quactlize_ppu_vecdot_moe_config_valid_v1(
    int total_rows, int n, int k, int group_size, int experts, int max_rows,
    int qtype, char const* config_name) {
  int const expected_group = ppu_formats::for_qtype(qtype).group_size;
  bool const compiled_name = !config_name || !config_name[0] ||
                             std::strcmp(config_name, QUACTLIZE_PPU_GROUPED_CUDA_CONFIG_NAME) == 0;
  return compiled_name && total_rows > 0 && n > 0 && k > 0 && k % 256 == 0 &&
         experts > 0 && max_rows > 0 && group_size == expected_group;
}

extern "C" int32_t quactlize_ppu_list_valid_vecdot_moe_configs_v2(
    quactlize_ppu_config_v2* configs, int32_t capacity,
    int total_rows, int n, int k, int group_size, int experts, int max_rows, int qtype) {
  bool const valid = quactlize_ppu_vecdot_moe_config_valid_v1(
      total_rows, n, k, group_size, experts, max_rows, qtype,
      QUACTLIZE_PPU_GROUPED_CUDA_CONFIG_NAME);
  if (valid && configs && capacity > 0) {
    configs[0] = {true, QUACTLIZE_PPU_GROUPED_CUDA_CONFIG_NAME, 0, 0, 0, 0, 0, 0};
  }
  return valid ? 1 : 0;
}

extern "C" int quactlize_ppu_vecdot_moe_config_v1(
    uint8_t const* b, int64_t block_bytes, uint16_t const* x,
    int const* offsets, float* out, int n, int bpr, int experts,
    int total_rows, int max_rows, int qtype, char const* config_name) {
  if (config_name && config_name[0] &&
      std::strcmp(config_name, QUACTLIZE_PPU_GROUPED_CUDA_CONFIG_NAME) != 0) {
    std::fprintf(stderr,
                 "[quactlize_ppu] grouped CUDA config '%s' is not compiled in; declining to default '%s'\n",
                 config_name, QUACTLIZE_PPU_GROUPED_CUDA_CONFIG_NAME);
  }
#define RUN(T) (block_bytes == raw_block_bytes<KType::T>() && n > 0 && bpr > 0 && experts > 0 && \
                total_rows > 0 && max_rows > 0 \
                ? native_moe<KType::T>(b, x, offsets, out, n, bpr, experts, total_rows, max_rows) : 11)
  switch (qtype) { case 10: return RUN(Q2_K); case 11: return RUN(Q3_K); case 12: return RUN(Q4_K);
                   case 13: return RUN(Q5_K); case 14: return RUN(Q6_K); default: return 2; }
#undef RUN
}

extern "C" int quactlize_ppu_vecdot_moe(uint8_t const* b, int64_t block_bytes, uint16_t const* x,
                                          int const* offsets, float* out, int n, int bpr, int experts,
                                          int total_rows, int max_rows, int qtype) {
  return quactlize_ppu_vecdot_moe_config_v1(
      b, block_bytes, x, offsets, out, n, bpr, experts, total_rows, max_rows, qtype, nullptr);
}

extern "C" int quactlize_ppu_bc_gemv(uint16_t const* x, uint8_t const* low, uint8_t const* high,
                                       uint8_t const* units, int const* offsets, float* out,
                                       int total_rows, int n, int k, int experts, int max_rows, int qtype) {
  if (!x || !low || !units || !out || total_rows <= 0 || n <= 0 || n % 256 || k <= 0 || k % 256 ||
      experts < 0 || (experts > 0 && (!offsets || max_rows <= 0)) ||
      (experts == 0 && total_rows >= 8)) return 13;
#define RUN(T) (gguf_scale::bc_vecdot::Traits<KType::T>::Hi && !high ? 14 : \
                ((k / 256) % gguf_scale::packed_unit::Unit<KType::T>::kSbPerUnit ? 15 : \
                 bc_gemv<KType::T,gguf_scale::bc_vecdot::Traits<KType::T>::DefaultArtifactTileK>( \
                     x,low,high,units,offsets,out,total_rows,n,k,experts,max_rows)))
  switch (qtype) { case 10: return RUN(Q2_K); case 11: return RUN(Q3_K); case 12: return RUN(Q4_K);
                   case 13: return RUN(Q5_K); case 14: return RUN(Q6_K); default: return 16; }
#undef RUN
}

extern "C" int quactlize_ppu_bc_gemv_for_arrangement_v1(
    uint16_t const* x, uint8_t const* low, uint8_t const* high,
    uint8_t const* units, int const* offsets, float* out,
    int total_rows, int n, int k, int experts, int max_rows, int qtype,
    quactlize_ppu_placed_arrangement_v1 const* arrangement) {
  if (!x || !low || !units || !out || total_rows <= 0 || n <= 0 || n % 256 || k <= 0 || k % 256 ||
      experts < 0 || (experts > 0 && (!offsets || max_rows <= 0)) ||
      (experts == 0 && total_rows >= 8) ||
      !arrangement || arrangement->version != QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1 ||
      arrangement->artifact_tile_k <= 0 || k % arrangement->artifact_tile_k) return 13;
#define RUN(T) (arrangement->bits != gguf_scale::bc_vecdot::Traits<KType::T>::Lo || \
                arrangement->high_bits != gguf_scale::bc_vecdot::Traits<KType::T>::Hi ? 24 : \
               (gguf_scale::bc_vecdot::Traits<KType::T>::Hi && !high ? 14 : \
               ((k / 256) % gguf_scale::packed_unit::Unit<KType::T>::kSbPerUnit ? 15 : \
                bc_gemv_arrangement_dispatch<KType::T,false>( \
                    x,low,high,units,offsets,out,total_rows,n,k,experts,max_rows, \
                    arrangement->artifact_tile_k))))
  switch (qtype) { case 10: return RUN(Q2_K); case 11: return RUN(Q3_K); case 12: return RUN(Q4_K);
                   case 13: return RUN(Q5_K); case 14: return RUN(Q6_K); default: return 16; }
#undef RUN
}

extern "C" int quactlize_ppu_bc_gemv_dev_v1(uint16_t const* x,
                                               uint8_t const* low, uint8_t const* high, uint8_t const* units,
                                               int const* offsets, float* out,
                                               int total_rows, int n, int k, int experts, int max_rows, int qtype,
                                               void* stream) {
  if (!x || !low || !units || !out || total_rows <= 0 || n <= 0 || n % 256 || k <= 0 || k % 256 ||
      experts < 0 || (experts > 0 && (!offsets || max_rows <= 0)) ||
      (experts == 0 && total_rows >= 8)) return 13;
  // Q4's whole-word reader uses float4/uint4 global loads. Public device
  // pointers need not originate at an allocation base, so reject a sliced
  // pointer before launch rather than allowing an undefined vector load.
  if (qtype == 12 &&
      !gguf_scale::bc_vecdot::q4_reader::vector_load_contract(x, low, units)) return 25;
  ppu_gemv::rt_clear_error();
  gemv_stream_t const s = static_cast<gemv_stream_t>(stream);
#define RUN(T) (gguf_scale::bc_vecdot::Traits<KType::T>::Hi && !high ? 14 : \
                ((k / 256) % gguf_scale::packed_unit::Unit<KType::T>::kSbPerUnit ? 15 : \
                 bc_gemv_device<KType::T,gguf_scale::bc_vecdot::Traits<KType::T>::DefaultArtifactTileK>( \
                     x,low,high,units,offsets,out,total_rows,n,k,experts,max_rows,s)))
  switch (qtype) { case 10: return RUN(Q2_K); case 11: return RUN(Q3_K); case 12: return RUN(Q4_K);
                   case 13: return RUN(Q5_K); case 14: return RUN(Q6_K); default: return 16; }
#undef RUN
}

extern "C" int quactlize_ppu_bc_gemv_for_arrangement_dev_v1(
    uint16_t const* x, uint8_t const* low, uint8_t const* high, uint8_t const* units,
    int const* offsets, float* out, int total_rows, int n, int k, int experts, int max_rows, int qtype,
    quactlize_ppu_placed_arrangement_v1 const* arrangement, void* stream) {
  if (!x || !low || !units || !out || total_rows <= 0 || n <= 0 || n % 256 || k <= 0 || k % 256 ||
      experts < 0 || (experts > 0 && (!offsets || max_rows <= 0)) ||
      (experts == 0 && total_rows >= 8) ||
      !arrangement || arrangement->version != QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1 ||
      arrangement->artifact_tile_k <= 0 || k % arrangement->artifact_tile_k) return 13;
  if (qtype == 12 &&
      !gguf_scale::bc_vecdot::q4_reader::vector_load_contract(x, low, units)) return 25;
  ppu_gemv::rt_clear_error();
  gemv_stream_t const s = static_cast<gemv_stream_t>(stream);
#define RUN(T) (arrangement->bits != gguf_scale::bc_vecdot::Traits<KType::T>::Lo || \
                arrangement->high_bits != gguf_scale::bc_vecdot::Traits<KType::T>::Hi ? 24 : \
               (gguf_scale::bc_vecdot::Traits<KType::T>::Hi && !high ? 14 : \
               ((k / 256) % gguf_scale::packed_unit::Unit<KType::T>::kSbPerUnit ? 15 : \
                bc_gemv_arrangement_dispatch<KType::T,true>( \
                    x,low,high,units,offsets,out,total_rows,n,k,experts,max_rows, \
                    arrangement->artifact_tile_k,s))))
  switch (qtype) { case 10: return RUN(Q2_K); case 11: return RUN(Q3_K); case 12: return RUN(Q4_K);
                   case 13: return RUN(Q5_K); case 14: return RUN(Q6_K); default: return 16; }
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

extern "C" int quactlize_ppu_prepass_unit(uint8_t const* units, uint16_t* scale, uint16_t* zero,
                                           int n, int k, int experts, int qtype, int zmul) {
  if (!units || !scale || !zero || n <= 0 || k <= 0 || k % 256 || experts <= 0) return 7;
  if (zmul != -32 && zmul != -24 && zmul != -4 && zmul != 0 && zmul != 8) return 8;
#define RUN(T) (zmul == -32 ? prepass_unit<KType::T, -32>(units,scale,zero,n,k,experts) : \
                zmul == -24 ? prepass_unit<KType::T, -24>(units,scale,zero,n,k,experts) : \
                zmul ==  -4 ? prepass_unit<KType::T,  -4>(units,scale,zero,n,k,experts) : \
                zmul ==   8 ? prepass_unit<KType::T,   8>(units,scale,zero,n,k,experts) : \
                              prepass_unit<KType::T,   0>(units,scale,zero,n,k,experts))
  switch (qtype) { case 10: return RUN(Q2_K); case 11: return RUN(Q3_K); case 12: return RUN(Q4_K);
                   case 13: return RUN(Q5_K); case 14: return RUN(Q6_K); default: return 9; }
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

extern "C" int quactlize_ppu_gemv_lowbit_config_v1(
    uint16_t const* a, uint8_t const* low, uint8_t const* high,
    uint16_t const* scale, uint16_t const* zero, uint16_t* out,
    int m, int n, int k, int group_size, int qtype, char const* config_name) {
  if (config_name && config_name[0] &&
      std::strcmp(config_name, QUACTLIZE_PPU_DENSE_CUDA_CONFIG_NAME) != 0) {
    std::fprintf(stderr,
                 "[quactlize_ppu] dense CUDA config '%s' is not compiled in; declining to default '%s'\n",
                 config_name, QUACTLIZE_PPU_DENSE_CUDA_CONFIG_NAME);
  }
  return quactlize_ppu_gemv_lowbit(
      a, low, high, scale, zero, out, m, n, k, group_size, qtype, 0, nullptr, m);
}

extern "C" int quactlize_ppu_gemv_lowbit_dev_v1(
    uint16_t const* a, uint8_t const* low, uint8_t const* high,
    uint16_t const* scale, uint16_t const* zero, uint16_t* out,
    int m, int n, int k, int group_size, int qtype, void* stream, char const* config_name) {
  if (!a || !low || !scale || !zero || !out ||
      !quactlize_ppu_gemv_lowbit_config_valid_v1(m, n, k, group_size, qtype, config_name)) return 40;
  ppu_gemv::rt_clear_error();
  gemv_stream_t const s = static_cast<gemv_stream_t>(stream);
#define RUN(F, S, T) lowbit_device<ppu_gemv::WFormat::F, S, T>( \
    a,low,high,scale,zero,out,m,n,k,group_size,0,nullptr,m,s)
  switch (qtype) {
    case 10: return RUN(Int2,  16, 128);
    case 11: return RUN(Q3_21, 32,  64);
    case 12: return RUN(Int4,  16, 128);
    case 13: return RUN(Q5_41, 32,  64);
    case 14: return RUN(Q6_42, 16, 128);
    default: return 6;
  }
#undef RUN
}
