// hgcc half of SCALE_FIRST x DENSE: raw host pointers -> the dedicated fpA mixed-input launcher. The offline weight
// reorder lives in ppu_dense_layout.cu so the resident artifact crosses this ABI already in the kernel's layout.
#include <cstdint>
#include <algorithm>
#include <vector>

#include "fpA_intB_ppu.cuh"
#include "moe_grouped_ppu.cuh"
#include "gemv_lowbit/gemv_rt.hpp"

namespace {

using ppu_gemv::DevBuf;
using half_t = cutlass::half_t;
using QM = fpa_intb_ppu::QuantMode;
using GQM = moe_grouped_ppu::QuantMode;
using Q4PackedUnit = cutlass::gguf_packed::Unit<cutlass::gguf_packed::Fmt::Q4K>;
static_assert(Q4PackedUnit::kUnitBytes == 16, "Q4_K's byte-neutral packed unit is the shipped 16-byte unit");

template <class Low, class High = void, int GroupSize = 16, int TileK = 256, bool PackedScale = false>
int dense(uint16_t const* act, uint8_t const* low, uint8_t const* high,
          void const* scale, uint16_t const* zero, uint16_t* out,
          int m, int n, int k, int group_size) {
  constexpr int LowBits = cutlass::sizeof_bits<Low>::value;
  constexpr int HighBits = std::is_void_v<High> ? 0 : cutlass::sizeof_bits<High>::value;
  size_t const low_bytes = size_t(n) * k * LowBits / 8;
  size_t const high_bytes = size_t(n) * k * HighBits / 8;
  size_t const plane_elems = size_t(k / GroupSize) * n;
  // Q4_K's byte-neutral packed unit is 16 B per (superblock,column). At TileK=256 this is exactly the byte count
  // of one fp16 scale plane, but it is not a half tensor: the shared mainloop reinterprets ptr_S as raw unit bytes
  // when kPackedScaleOn is true and does not read ptr_Z. Keep the allocation distinction explicit so this entry
  // cannot grow into a second decode implementation beside the collective.
  size_t const scale_bytes = PackedScale ? size_t(k / 256) * n * Q4PackedUnit::kUnitBytes : plane_elems * 2;
  DevBuf da(size_t(m) * k * 2), dl(low_bytes), dh(high_bytes), ds(scale_bytes),
         dz(PackedScale ? 0 : plane_elems * 2),
         dout(size_t(m) * n * 2);
  // SplitKSerial's semaphore workspace is one int per output CTA for split_k=1. Size it from the actual fixed tile.
  size_t const ws_bytes = size_t(cutlass::ceil_div(m, 64)) * cutlass::ceil_div(n, 64) * sizeof(int);
  DevBuf ws(ws_bytes);
  da.from_host(act); dl.from_host(low); ds.from_host(scale);
  if constexpr (!PackedScale) dz.from_host(zero);
  if constexpr (HighBits != 0) dh.from_host(high);

  using Tile = cute::Shape<cute::_64, cute::_64, cute::C<TileK>>;
  using Warp = cute::Shape<cute::_32, cute::_32, cute::C<TileK>>;
  // n/k are 256-aligned at the ABI, so the artifact's xplane interleave and LayoutB are compile-time facts. Both
  // live k-quant group sizes reuse Gs32; ScaleTileShape carries the real 16 or 8 groups in this K tile.
  constexpr int ScaleGroups = TileK / GroupSize;
  bool const launched = fpa_intb_ppu::generic_launcher<QM::FinegrainedScaleZero,
      cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs32,
      Tile, cute::Shape<cute::_64, cute::C<ScaleGroups>>, Warp, 3, true, Low, High, PackedScale>(
          da.as<half_t>(), dl.as<Low>(), ds.as<half_t>(),
          PackedScale ? nullptr : dz.as<half_t>(), dout.as<half_t>(),
          m, n, k, GroupSize, 1, ws.as<char>(), ws_bytes, nullptr,
          [&]() {
            if constexpr (std::is_void_v<High>) return static_cast<High const*>(nullptr);
            else return dh.as<High>();
          }());
  if (!launched) return 31;
  ppu_gemv::rt_sync(PackedScale ? "fully-quantized dense GEMM" : "scale-first dense GEMM");
  dout.to_host(out);
  return 0;
}

}  // namespace

extern "C" int quactlize_ppu_dense_lowbit(uint16_t const* act, uint8_t const* low, uint8_t const* high,
                                            uint16_t const* scale, uint16_t const* zero, uint16_t* out,
                                            int m, int n, int k, int group_size, int qtype) {
  if (!act || !low || !scale || !zero || !out || m <= 0 || n <= 0 || k <= 0 || n % 256 || k % 256) return 30;
  switch (qtype) {
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 10
    case 10: return group_size == 16 ? dense<cutlass::uint2b_t,void,16>(act,low,high,scale,zero,out,m,n,k,group_size) : 32;
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 11
    case 11: return group_size == 16 ? dense<cutlass::uint2b_t,cutlass::uint1b_t,16>(act,low,high,scale,zero,out,m,n,k,group_size) : 32;
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 12
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
    // TileK=256 selects packed units in this build, so the SCALE_FIRST contract must not use that instantiation.
    // Its single low plane is tile-invariant; TileK=128 gives Scale_TileK=4, keeps kPackedScaleOn false, and lets one
    // flagged library run the existing independent scale-first oracle beside the new TileK=256 packed entry below.
    case 12: return group_size == 32 ? dense<cutlass::int4b_t,void,32,128>(act,low,high,scale,zero,out,m,n,k,group_size) : 32;
#else
    case 12: return group_size == 32 ? dense<cutlass::int4b_t,void,32>(act,low,high,scale,zero,out,m,n,k,group_size) : 32;
#endif
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 13
    case 13: return group_size == 32 ? dense<cutlass::int4b_t,cutlass::uint1b_t,32>(act,low,high,scale,zero,out,m,n,k,group_size) : 32;
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 14
    case 14: return group_size == 16 ? dense<cutlass::int4b_t,cutlass::uint2b_t,16,128>(act,low,high,scale,zero,out,m,n,k,group_size) : 32;
#endif
    default: return 33;
  }
}

// FULLY_QUANTIZED x DENSE, Q4_K. This is only a second ABI contract: it instantiates the SAME dense() wrapper and
// the SAME CollectiveBuilder as scale-first, with TileK=256 making Scale_TileK==PackedUnit::kGroups==8. The packed
// decode remains solely in the shared mainloop. Builds without PPU_PACKED_SCALE retain the symbol but return 34,
// so callers can distinguish "not in the default build" from a missing library.
extern "C" int quactlize_ppu_dense_fully_quantized(uint16_t const* act, uint8_t const* low,
                                                     uint8_t const* units, uint16_t* out,
                                                     int m, int n, int k, int qtype) {
  if (!act || !low || !units || !out || m <= 0 || n <= 0 || k <= 0 || n % 256 || k % 256) return 30;
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
#if !defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0
  if (qtype != 12) return 33;
  return dense<cutlass::int4b_t, void, 32, 256, true>(
      act, low, nullptr, units, nullptr, out, m, n, k, 32);
#else
  (void)qtype;
  return 35;  // this binary's packed-scale format is not Q4_K
#endif
#else
  (void)qtype;
  return 34;
#endif
}

// FULLY_QUANTIZED x GROUPED, Q4_K. The artifact is [E,N,K/2] low codes plus [E,K/256,N,16] units; activations and
// output are concatenated in expert order and rows_per_expert supplies the ragged boundaries. All tensor-core work
// remains in moe_grouped_ppu's existing grouped scheduler and shared packed-scale collective. This wrapper only
// materialises the raw-pointer arrays that the grouped CUTLASS interface requires.
extern "C" int quactlize_ppu_grouped_fully_quantized(
    uint16_t const* act, uint8_t const* low, uint8_t const* units, int const* rows_per_expert,
    uint16_t* out, int total_rows, int n, int k, int experts, int qtype) {
  if (!act || !low || !units || !rows_per_expert || !out || total_rows <= 0 || n <= 0 || k <= 0 ||
      experts <= 0 || n % 256 || k % 256) return 30;
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
#if !defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0
  if (qtype != 12) return 33;
  using GS = moe_grouped_ppu::GroupShape;
  using DS = moe_grouped_ppu::DStride;

  std::vector<int> rows(static_cast<size_t>(experts)), offsets(static_cast<size_t>(experts));
  int sum = 0, max_rows = 0;
  for (int e = 0; e < experts; ++e) {
    if (rows_per_expert[e] < 0) return 36;
    rows[size_t(e)] = rows_per_expert[e];
    offsets[size_t(e)] = sum;
    sum += rows_per_expert[e];
    max_rows = std::max(max_rows, rows_per_expert[e]);
  }
  if (sum != total_rows || max_rows <= 0) return 36;

  size_t const low_bytes = size_t(experts) * n * k / 2;
  size_t const unit_bytes = size_t(experts) * (k / 256) * n * Q4PackedUnit::kUnitBytes;
  DevBuf da(size_t(total_rows) * k * 2), dl(low_bytes), du(unit_bytes), dout(size_t(total_rows) * n * 2);
  da.from_host(act); dl.from_host(low); du.from_host(units);

  std::vector<GS> shapes(static_cast<size_t>(experts));
  std::vector<half_t*> out_ptrs(static_cast<size_t>(experts));
  std::vector<DS> out_strides(static_cast<size_t>(experts));
  for (int e = 0; e < experts; ++e) {
    shapes[size_t(e)] = cute::make_shape(rows[size_t(e)], n, k);
    out_ptrs[size_t(e)] = dout.as<half_t>() + size_t(offsets[size_t(e)]) * n;
    out_strides[size_t(e)] = cutlass::make_cute_packed_stride(
        DS{}, cute::make_shape(rows[size_t(e)], n, 1));
  }
  DevBuf d_shapes(sizeof(GS) * size_t(experts)), d_out_ptrs(sizeof(half_t*) * size_t(experts)),
         d_out_strides(sizeof(DS) * size_t(experts)), d_rows(sizeof(int) * size_t(experts)),
         d_offsets(sizeof(int) * size_t(experts));
  d_shapes.from_host(shapes.data()); d_out_ptrs.from_host(out_ptrs.data());
  d_out_strides.from_host(out_strides.data()); d_rows.from_host(rows.data()); d_offsets.from_host(offsets.data());

  size_t const ws_bytes = size_t(cutlass::ceil_div(max_rows, 16)) * cutlass::ceil_div(n, 64)
                        * size_t(experts) * 64;
  DevBuf ws(ws_bytes);
  int const failures = moe_grouped_ppu::moeg_fail_count();
  using GTile = cute::Shape<cute::_16, cute::_128, cute::C<256>>;
  using GScale = cute::Shape<cute::_128, cute::_8>;
  using GWarp = cute::Shape<cute::_16, cute::_16, cute::C<256>>;
  // Call the fixed gs32 instantiation directly. filter_and_run's runtime group-size ladder instantiates SK=2/4/8/16
  // together, so asserting packed selection there would correctly fail on its non-SK8 control branches.
  moe_grouped_ppu::launch<GQM::FinegrainedScaleZero,
                          cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs32,
                          GTile, GScale, GWarp, 2, true, cutlass::int4b_t, void, true>(
      da.as<half_t>(), dl.as<cutlass::int4b_t>(), reinterpret_cast<half_t const*>(du.p), nullptr,
      d_out_ptrs.as<half_t*>(), d_out_strides.as<DS>(), d_rows.as<int>(), max_rows, n, k, experts, 32,
      d_shapes.as<GS>(), shapes.data(), d_offsets.as<int>(), ws.as<char>(), ws_bytes, nullptr);
  if (moe_grouped_ppu::moeg_fail_count() != failures) return 31;
  ppu_gemv::rt_sync("fully-quantized grouped GEMM");
  dout.to_host(out);
  return 0;
#else
  (void)qtype;
  return 35;
#endif
#else
  (void)qtype;
  return 34;
#endif
}
