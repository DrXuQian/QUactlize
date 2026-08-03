// hgcc half of SCALE_FIRST x DENSE: raw host pointers -> the dedicated fpA mixed-input launcher. The offline weight
// reorder lives in ppu_dense_layout.cu so the resident artifact crosses this ABI already in the kernel's layout.
#include <cstdint>

#include "fpA_intB_ppu.cuh"
#include "gemv_lowbit/gemv_rt.hpp"

namespace {

using ppu_gemv::DevBuf;
using half_t = cutlass::half_t;
using QM = fpa_intb_ppu::QuantMode;
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
