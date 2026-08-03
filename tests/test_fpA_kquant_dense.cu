// SCALE_FIRST x DENSE, all five k-quant code shapes, through fpA_intB_ppu.cuh.
//
// ORACLE STRENGTH: SELF, DELIBERATELY. The production 64x64x256 tactic is compared with a 32x64x256 tactic over the
// same independently placed resident artifact. Stage 3 vs stage 2 catches launch/config plumbing and carries a
// failing exit status, but a wrong constant shared by both collectives moves both answers together. It is
// therefore evidence for IMPLEMENTED, never VALIDATED. The official-gguf artifact test covers the importer and affine
// boundary separately; an independent PPU output golden is still required to promote the cell.
//
// Build: TARGET=test_fpA_kquant_dense ./build.sh ; run: ./<bin>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "cutlass/util/device_memory.h"
#include "helper.h"
#include "fpA_intB_ppu.cuh"
#include "xplane_offline.hpp"

using half_t = cutlass::half_t;
using QM = fpa_intb_ppu::QuantMode;

template <int LowBits, int HighBits, class Low, class High = void, int GroupSize = 16, int TileK = 256>
int run_format(char const* name) {
  constexpr int M = 32, N = 256, K = 256, SK = K / GroupSize;
  std::vector<half_t> a(size_t(M) * K), scale(size_t(SK) * N), zero(size_t(SK) * N);
  for (size_t i = 0; i < a.size(); ++i) a[i] = half_t(float(int((i * 2654435761u >> 7) & 15) - 7) * .03125f);
  for (size_t i = 0; i < scale.size(); ++i) {
    scale[i] = half_t(float((i * 17) % 11 + 1) * .015625f);
    zero[i] = half_t(float(int((i * 29) % 9) - 4) * .0078125f);
  }
  std::vector<uint8_t> lo(size_t(K) * N), hi(size_t(K) * N);
  constexpr int CodeBits = LowBits + HighBits;
  for (size_t i = 0; i < lo.size(); ++i) {
    int const q = int((i * 104729u + i / N * 8191u) & ((1u << CodeBits) - 1));
    lo[i] = uint8_t(q & ((1 << LowBits) - 1));
    if constexpr (HighBits != 0) hi[i] = uint8_t(q >> LowBits);
  }
  std::vector<int8_t> blo(size_t(K) * N * LowBits / 8), bhi(size_t(K) * N * HighBits / 8);
  xplane::place_derived<LowBits, 64, 64, TileK, 32, 32, 1>(blo.data(), lo, N, K);
  if constexpr (HighBits != 0)
    xplane::place_hi<LowBits, HighBits, 64, 64, TileK, 32, 32, 1, 1>(bhi.data(), hi, N, K);

  cutlass::DeviceAllocation<half_t> da(size_t(M) * K), ds(size_t(SK) * N), dz(size_t(SK) * N),
                                      d0(size_t(M) * N), d1(size_t(M) * N);
  cutlass::DeviceAllocation<Low> db(size_t(K) * N);
  cutlass::DeviceAllocation<uint8_t> db2(std::max<size_t>(bhi.size(), 1));
  cutlass::DeviceAllocation<char> ws(2 * size_t(cutlass::ceil_div(M, 32)) * cutlass::ceil_div(N, 64) * sizeof(int));
  da.copy_from_host(a.data()); ds.copy_from_host(scale.data()); dz.copy_from_host(zero.data());
  db.copy_from_host(reinterpret_cast<Low const*>(blo.data()));
  if constexpr (HighBits != 0) db2.copy_from_host(reinterpret_cast<uint8_t const*>(bhi.data()));
  auto b2 = [&]() {
    if constexpr (HighBits == 0) return static_cast<High const*>(nullptr);
    else return reinterpret_cast<High const*>(db2.get());
  };

  using Tile0 = cute::Shape<cute::_64, cute::_64, cute::C<TileK>>;
  using Warp0 = cute::Shape<cute::_32, cute::_32, cute::C<TileK>>;
  using Tile1 = Tile0;
  using Warp1 = Warp0;
  using ScaleTile = cute::Shape<cute::_64, cute::C<TileK / GroupSize>>;
  using Schedule = cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs32;
  bool const ok0 = fpa_intb_ppu::generic_launcher<QM::FinegrainedScaleZero, Schedule,
      Tile0, ScaleTile, Warp0, 3, true, Low, High>(
          da.get(), db.get(), ds.get(), dz.get(), d0.get(), M, N, K, GroupSize, 1,
          ws.get(), ws.capacity, nullptr, b2());
  bool const ok1 = fpa_intb_ppu::generic_launcher<QM::FinegrainedScaleZero, Schedule,
      Tile1, ScaleTile, Warp1, 2, true, Low, High>(
          da.get(), db.get(), ds.get(), dz.get(), d1.get(), M, N, K, GroupSize, 1,
          ws.get(), ws.capacity, nullptr, b2());
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  if (!ok0 || !ok1) { std::printf("  %-4s launch failed (%d/%d)\n", name, int(ok0), int(ok1)); return 1; }

  std::vector<half_t> h0(size_t(M) * N), h1(size_t(M) * N); d0.copy_to_host(h0.data()); d1.copy_to_host(h1.data());
  int bad = 0; double worst = 0;
  for (size_t i = 0; i < h0.size(); ++i) {
    double const x = float(h0[i]), y = float(h1[i]);
    double const rel = std::abs(x - y) / (std::abs(y) + 1e-3);
    worst = std::max(worst, rel); if (rel > 2e-2) ++bad;
  }
  std::printf("  %-4s stage3 vs stage2: bad=%d/%zu max_rel=%.3e %s\n",
              name, bad, h0.size(), worst, bad ? "MISMATCH" : "MATCH");
  return bad ? 1 : 0;
}

int main() {
  std::printf("[fpA-kquant-dense] SELF oracle -- plumbing evidence, not independent validation\n");
  int fail = 0;
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 10
  fail += run_format<2,0,cutlass::uint2b_t,void,16>("Q2_K");
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 11
  fail += run_format<2,1,cutlass::uint2b_t,cutlass::uint1b_t,16>("Q3_K");
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 12
  fail += run_format<4,0,cutlass::int4b_t,void,32>("Q4_K");
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 13
  fail += run_format<4,1,cutlass::int4b_t,cutlass::uint1b_t,32>("Q5_K");
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 14
  fail += run_format<4,2,cutlass::int4b_t,cutlass::uint2b_t,16,128>("Q6_K");
#endif
  return fail ? 1 : 0;
}
