// hgcc half of SCALE_FIRST x DENSE: raw host pointers -> the dedicated fpA mixed-input launcher. The offline weight
// reorder lives in ppu_dense_layout.cu so the resident artifact crosses this ABI already in the kernel's layout.
#include <cstdint>

#include "fpA_intB_ppu.cuh"
#include "gemv_lowbit/gemv_rt.hpp"

namespace {

using ppu_gemv::DevBuf;
using half_t = cutlass::half_t;
using QM = fpa_intb_ppu::QuantMode;

template <class Low, class High = void, int GroupSize = 16, int TileK = 256>
int dense(uint16_t const* act, uint8_t const* low, uint8_t const* high,
          uint16_t const* scale, uint16_t const* zero, uint16_t* out,
          int m, int n, int k, int group_size) {
  constexpr int LowBits = cutlass::sizeof_bits<Low>::value;
  constexpr int HighBits = std::is_void_v<High> ? 0 : cutlass::sizeof_bits<High>::value;
  size_t const low_bytes = size_t(n) * k * LowBits / 8;
  size_t const high_bytes = size_t(n) * k * HighBits / 8;
  size_t const plane_elems = size_t(k / GroupSize) * n;
  DevBuf da(size_t(m) * k * 2), dl(low_bytes), dh(high_bytes), ds(plane_elems * 2), dz(plane_elems * 2),
         dout(size_t(m) * n * 2);
  // SplitKSerial's semaphore workspace is one int per output CTA for split_k=1. Size it from the actual fixed tile.
  size_t const ws_bytes = size_t(cutlass::ceil_div(m, 64)) * cutlass::ceil_div(n, 64) * sizeof(int);
  DevBuf ws(ws_bytes);
  da.from_host(act); dl.from_host(low); ds.from_host(scale); dz.from_host(zero);
  if constexpr (HighBits != 0) dh.from_host(high);

  using Tile = cute::Shape<cute::_64, cute::_64, cute::C<TileK>>;
  using Warp = cute::Shape<cute::_32, cute::_32, cute::C<TileK>>;
  // n/k are 256-aligned at the ABI, so the artifact's xplane interleave and LayoutB are compile-time facts. Both
  // live k-quant group sizes reuse Gs32; ScaleTileShape carries the real 16 or 8 groups in this K tile.
  constexpr int ScaleGroups = TileK / GroupSize;
  bool const launched = fpa_intb_ppu::generic_launcher<QM::FinegrainedScaleZero,
      cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs32,
      Tile, cute::Shape<cute::_64, cute::C<ScaleGroups>>, Warp, 3, true, Low, High>(
          da.as<half_t>(), dl.as<Low>(), ds.as<half_t>(), dz.as<half_t>(), dout.as<half_t>(),
          m, n, k, GroupSize, 1, ws.as<char>(), ws_bytes, nullptr,
          [&]() {
            if constexpr (std::is_void_v<High>) return static_cast<High const*>(nullptr);
            else return dh.as<High>();
          }());
  if (!launched) return 31;
  ppu_gemv::rt_sync("scale-first dense GEMM");
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
    case 12: return group_size == 32 ? dense<cutlass::int4b_t,void,32>(act,low,high,scale,zero,out,m,n,k,group_size) : 32;
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
