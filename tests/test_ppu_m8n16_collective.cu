// #112 G3/G4 numerical gate for the ppu001 m8n16 mixed-input collective.
//
// This target deliberately has two layers:
//
//   G3  invokes the real ScaleZero int4 collective mainloop directly, then
//       scatters its FP32 accumulator fragment with the TiledMma coordinate
//       tensor.  There is no formal epilogue in this arm.  Eight one-hot A
//       rows select K coordinates across all four TK64 tiles and both sides
//       of every gs=32 boundary, so a
//       failure is in A delivery, B delivery/dequant, metadata selection, or
//       the m8 atom -- not in the output epilogue.
//
//   G4  runs the production nonpersistent and persistent grouped kernels and
//       formal ptr-array epilogue for
//       M={1,2,3,7,8,9,15,16,17}.  Every logical D element starts as a qNaN and the D
//       allocation has distinct bit-exact canaries on both sides.  Each m8
//       result is checked against an independent host dequant/GEMM oracle and
//       against an exact m16-control launch on the same canonical A/Q/S/Z.
//
//   G5  is #108's real route: E=256, eight non-contiguous active IDs, every
//       expert given DIFFERENT W/S/Z and the 248 inactive ones poisoned.  G4
//       cannot stand in for it -- with L=1 there is no route at all: no
//       slot->expert map to invert, no row offset to be off by one, no expert
//       that should not be read.  Two negative controls give it power: the
//       neighbour-expert swap and the one-expert row shift must both be
//       rejected, or a green G5 would only mean "the numbers matched
//       something", which a fixture with indistinguishable experts always
//       does.  It runs in three configurations -- LOW / UNIFORM / REAL -- so a
//       red separates "the kernel's expert addressing" from "this fixture's
//       per-expert placement" without changing the kernel.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "m8n16_g5_contract.hpp"
#include "m8n16_g5_slot_map.hpp"
#include "moe_grouped_ppu.cuh"
#include "ppu_group_schedule.hpp"
#include "ppu_mixed_policy.hpp"
#include "xplane_offline.hpp"

namespace {

using half_t = cutlass::half_t;
using int4_t = cutlass::int4b_t;
using QM = moe_grouped_ppu::QuantMode;
using GS = moe_grouped_ppu::GroupShape;
using DStride = moe_grouped_ppu::DStride;

constexpr int kBits = m8n16_g5_contract::kBits;
constexpr int kN = m8n16_g5_contract::kN;
constexpr int kTacticK = m8n16_g5_contract::kTacticK;
// F=1/TK64 is stored in the AIU's 256-code resident-row layout.  The old
// K=64 fixture allocated the resulting padding but still advertised a packed
// K=64 StrideB to the collective, so the consumer advanced by 64 codes while
// the producer advanced by 256.  Four tactic tiles make those two strides the
// same without changing the tactic under test.
constexpr int kStoredRowK = m8n16_g5_contract::kStoredRowK;
constexpr int kK = m8n16_g5_contract::kK;
constexpr int kGs = m8n16_g5_contract::kGroupSize;
constexpr int kScaleK = m8n16_g5_contract::kScaleK;
constexpr std::size_t kHarnessBBytes = std::size_t(kN) * kK * kBits / 8;
constexpr std::size_t kPlacedBBytes =
    std::size_t(kN) * ((kK + kStoredRowK - 1) / kStoredRowK) * kStoredRowK * kBits / 8;
constexpr int kStages = m8n16_g5_contract::kStages;
constexpr int kMaxG4M = 17;
constexpr int kGuard = 64;
constexpr std::uint16_t kLeftCanary = 0x3555u;
constexpr std::uint16_t kRightCanary = 0xb555u;
constexpr std::uint16_t kOutputNaN = 0x7e01u;

using BaseSchedule = m8n16_g5_contract::Schedule;
using M8Tile = typename m8n16_g5_contract::M8::Tile;
using M8Warp = typename m8n16_g5_contract::M8::Warp;
using ScaleTile = typename m8n16_g5_contract::M8::Scale;
using M8Policy = typename m8n16_g5_contract::M8::Policy;
using M8Mainloop = typename M8Policy::CollectiveOp;

static_assert(cute::size<0>(typename M8Mainloop::TiledMma::AtomShape_MNK{}) == 8,
              "G3/G4 m8 arm must instantiate the ppu001 m8n16 atom");
static_assert(cute::size(typename M8Mainloop::TiledMma{}) == 32,
              "G3 is deliberately one warp / one CTA");
static_assert(cute::size<0>(typename M8Mainloop::SmemLayoutA{}) == 8,
              "m8 mainloop must expose eight logical A rows");
static_assert(cute::size<0>(typename M8Mainloop::SmemLayoutAPhysical{}) == 16,
              "m8 mainloop must retain the physical 16-row AIU cube");
static_assert(kK % kTacticK == 0 && kK / kTacticK == 4,
              "G3/G4 must exercise four TK64 mainloop tiles");
static_assert(kK % kStoredRowK == 0,
              "G3/G4 packed StrideB must span complete interleave-256 stored rows; "
              "a shorter problem K advertises a stride that disagrees with the artifact placement");
static_assert(kHarnessBBytes == kPlacedBBytes,
              "G3/G4 packed StrideB span disagrees with the F=1 interleave-256 artifact span");

half_t hbits(std::uint16_t bits) { return half_t::bitcast(bits); }

std::vector<std::uint8_t> make_codes() {
  std::vector<std::uint8_t> q(std::size_t(kK) * kN);
  for (int k = 0; k < kK; ++k) {
    for (int n = 0; n < kN; ++n) {
      // All 16 codes occur in every K neighbourhood, and neither K nor N is
      // a symmetry of the pattern.
      q[std::size_t(k) * kN + n] =
          std::uint8_t((11 * k + 7 * n + 3 * (k / kGs) + (k ^ n)) & 15);
    }
  }
  return q;
}

std::vector<half_t> make_scales() {
  std::vector<half_t> s(std::size_t(kScaleK) * kN);
  for (int g = 0; g < kScaleK; ++g) {
    for (int n = 0; n < kN; ++n) {
      // Positive dyadics keep the CPU oracle independent but exactly
      // representable through the device's fp16 multiply.
      s[std::size_t(g) * kN + n] = half_t(float(1 + ((5 * g + 3 * n) & 7)) / 32.0f);
    }
  }
  return s;
}

std::vector<half_t> make_zeros(std::vector<half_t> const& scales) {
  std::vector<half_t> z(scales.size());
  for (int g = 0; g < kScaleK; ++g) {
    for (int n = 0; n < kN; ++n) {
      // The int4 converter emits q-8.  8*scale cancels that representation
      // bias; a distinct signed dyadic offset keeps the zero plane
      // independently load-bearing instead of reducing the test to ScaleOnly.
      float const offset = float(((13 * n + 5 * g) % 7) - 3) / 16.0f;
      z[std::size_t(g) * kN + n] =
          half_t(8.0f * float(scales[std::size_t(g) * kN + n]) + offset);
    }
  }
  return z;
}

std::vector<half_t> make_dense_a() {
  std::vector<half_t> a(std::size_t(kMaxG4M) * kK);
  for (int m = 0; m < kMaxG4M; ++m) {
    for (int k = 0; k < kK; ++k) {
      a[std::size_t(m) * kK + k] =
          half_t(float(((17 * m + 9 * k + (m ^ k)) % 15) - 7) / 16.0f);
    }
  }
  return a;
}

std::vector<half_t> make_onehot_a() {
  // Cross gs=32 boundaries and all four TK64 tiles.  A gate confined to the
  // first tile would not prove that the packed K=256 stride advances to each
  // resident TK64 artifact correctly.
  constexpr int selected[8] = {31, 32, 95, 96, 159, 160, 223, 224};
  std::vector<half_t> a(std::size_t(8) * kK, half_t(0.0f));
  for (int m = 0; m < 8; ++m) {
    a[std::size_t(m) * kK + selected[m]] = half_t(float(m + 1) / 8.0f);
  }
  return a;
}

std::vector<half_t> make_row_tag_a(int M, bool replay_first_tile = false) {
  // K=33 has a nonzero dequantized value in every one of the 32 columns for
  // this fixture.  One nonzero half per row makes the FP32 MMA result one
  // exactly-representable half*half product.  The amplitude is the absolute
  // row coordinate, so rows 8 and 16 cannot alias rows 0 and 8 silently.
  constexpr int kTagK = 33;
  std::vector<half_t> a(std::size_t(M) * kK, half_t(0.0f));
  for (int m = 0; m < M; ++m) {
    int const tagged_m = replay_first_tile && m >= 8 ? m % 8 : m;
    a[std::size_t(m) * kK + kTagK] = half_t(float(tagged_m + 1));
  }
  return a;
}

// Reproduce the device's three fp16 steps independently: int4 -> fp16 q-8,
// fp16 multiply by scale, then fp16 add zero.  Keeping each rounding point
// prevents a host float FMA from becoming a self-invented oracle.
half_t dequant_one(std::uint8_t q, half_t scale, half_t zero) {
  half_t const centered(float(int(q) - 8));
  half_t const scaled(float(centered) * float(scale));
  return half_t(float(scaled) + float(zero));
}

std::vector<float> golden_fp32(
    std::vector<half_t> const& a, int M,
    std::vector<std::uint8_t> const& q,
    std::vector<half_t> const& scales,
    std::vector<half_t> const& zeros) {
  std::vector<float> d(std::size_t(M) * kN, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < kN; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < kK; ++k) {
        half_t const w = dequant_one(
            q[std::size_t(k) * kN + n],
            scales[std::size_t(k / kGs) * kN + n],
            zeros[std::size_t(k / kGs) * kN + n]);
        acc += float(a[std::size_t(m) * kK + k]) * float(w);
      }
      d[std::size_t(m) * kN + n] = acc;
    }
  }
  return d;
}

template <class Mainloop>
__global__ void g3_mainloop_only(
    typename Mainloop::Params params, float* output, int M) {
  extern __shared__ char smem[];
  int const tid = int(threadIdx.x);
  int const tile_m = int(blockIdx.x);
  if (blockIdx.y != 0 || blockIdx.z != 0 || tile_m * 8 >= M ||
      tid >= int(cute::size(typename Mainloop::TiledMma{}))) return;

  auto problem = cute::make_shape(M, kN, kK, 1);
  auto block = cute::make_coord(tile_m, 0, cute::_, 0);
  Mainloop mainloop;
  auto inputs = mainloop.load_init(problem, block, params);

  typename Mainloop::TiledMma tiled_mma;
  auto accum = cute::make_fragment_like<float>(
      cute::partition_fragment_C(tiled_mma, cute::take<0, 2>(typename Mainloop::TileShape{})));
  cute::clear(accum);
  auto gA = cute::get<0>(inputs);
  auto k_iter = cute::make_coord_iterator(cute::shape<2>(gA));
  mainloop(params, inputs, accum, k_iter, int(cute::size<2>(gA)), tid, smem);

  // This is the mainloop's own accumulator-coordinate map, not a second
  // transcription of the m8 register ABI (G1 already gates that ABI).
  auto cC = cute::make_identity_tensor(cute::make_shape(cute::_8{}, cute::_32{}));
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto tCcC = thr_mma.partition_C(cC);
  CUTE_STATIC_ASSERT_V(cute::size(tCcC) == cute::size(accum));
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < int(cute::size(accum)); ++i) {
    auto coord = tCcC(i);
    int const global_m = tile_m * 8 + int(cute::get<0>(coord));
    if (global_m < M) {
      output[global_m * kN + int(cute::get<1>(coord))] = accum(i);
    }
  }
}

int check_float_output(char const* tag, std::vector<float> const& got,
                       std::vector<float> const& want, float atol) {
  int bad = 0;
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < got.size(); ++i) {
    float const err = std::abs(got[i] - want[i]);
    max_abs = std::max(max_abs, err);
    if (!std::isfinite(got[i]) || err > atol) {
      if (bad < 6) {
        std::printf("    %s[%zu] got=%g want=%g err=%g\n",
                    tag, i, double(got[i]), double(want[i]), double(err));
      }
      ++bad;
    }
  }
  std::printf("  %-24s bad=%d/%zu max_abs=%.3e %s\n",
              tag, bad, got.size(), double(max_abs), bad ? "MISMATCH" : "MATCH");
  return bad;
}

std::uint32_t float_bits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

int float_bitdiff(std::vector<float> const& lhs,
                  std::vector<float> const& rhs) {
  if (lhs.size() != rhs.size()) return int(std::max(lhs.size(), rhs.size()));
  int bad = 0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    bad += float_bits(lhs[i]) != float_bits(rhs[i]);
  }
  return bad;
}

int run_g3(
    cutlass::DeviceAllocation<int4_t>& dB,
    cutlass::DeviceAllocation<half_t>& dScale,
    cutlass::DeviceAllocation<half_t>& dZero,
    std::vector<std::uint8_t> const& q,
    std::vector<half_t> const& scales,
    std::vector<half_t> const& zeros) {
  auto a = make_onehot_a();
  auto golden = golden_fp32(a, 8, q, scales, zeros);
  cutlass::DeviceAllocation<half_t> dA(a.size());
  cutlass::DeviceAllocation<float> dOut(golden.size());
  dA.copy_from_host(a.data());
  std::vector<float> init(golden.size(), std::numeric_limits<float>::quiet_NaN());
  dOut.copy_from_host(init.data());

  using StrideA = typename M8Mainloop::StrideA;
  using StrideB = typename M8Mainloop::StrideB;
  using StrideS = typename M8Mainloop::StrideScale;
  StrideA dA_stride = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(8, kK, 1));
  StrideB dB_stride = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape(kN, kK, 1));
  StrideS dS_stride = cutlass::make_cute_packed_stride(
      StrideS{}, cute::make_shape(kN, kScaleK, 1));
  typename M8Mainloop::Arguments args{
      dA.get(), dA_stride, dB.get(), dB_stride,
      dScale.get(), dS_stride, kGs, dZero.get(), nullptr};
  auto params = M8Mainloop::to_underlying_arguments(
      cute::make_shape(8, kN, kK, 1), args, nullptr);

  g3_mainloop_only<M8Mainloop><<<
      1, int(cute::size(typename M8Mainloop::TiledMma{})),
      sizeof(typename M8Mainloop::SharedStorage)>>>(params, dOut.get(), 8);
  CUTLASS_PPU_CHECK(hggcGetLastError());
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  std::vector<float> got(golden.size());
  dOut.copy_to_host(got.data());
  return check_float_output("G3 raw FP32 accum", got, golden, 1.0e-5f);
}

int run_g3_a_tag(
    int M, cutlass::DeviceAllocation<int4_t>& dB,
    cutlass::DeviceAllocation<half_t>& dScale,
    cutlass::DeviceAllocation<half_t>& dZero,
    std::vector<std::uint8_t> const& q,
    std::vector<half_t> const& scales,
    std::vector<half_t> const& zeros) {
  auto run = [&](std::vector<half_t> const& a) {
    cutlass::DeviceAllocation<half_t> dA(a.size());
    cutlass::DeviceAllocation<float> dOut(std::size_t(M) * kN);
    dA.copy_from_host(a.data());
    std::vector<float> init(std::size_t(M) * kN,
                            std::numeric_limits<float>::quiet_NaN());
    dOut.copy_from_host(init.data());

    using StrideA = typename M8Mainloop::StrideA;
    using StrideB = typename M8Mainloop::StrideB;
    using StrideS = typename M8Mainloop::StrideScale;
    StrideA dA_stride = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(M, kK, 1));
    StrideB dB_stride = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape(kN, kK, 1));
    StrideS dS_stride = cutlass::make_cute_packed_stride(
        StrideS{}, cute::make_shape(kN, kScaleK, 1));
    typename M8Mainloop::Arguments args{
        dA.get(), dA_stride, dB.get(), dB_stride,
        dScale.get(), dS_stride, kGs, dZero.get(), nullptr};
    auto params = M8Mainloop::to_underlying_arguments(
        cute::make_shape(M, kN, kK, 1), args, nullptr);
    int const tiles_m = (M + 7) / 8;
    g3_mainloop_only<M8Mainloop><<<
        tiles_m, int(cute::size(typename M8Mainloop::TiledMma{})),
        sizeof(typename M8Mainloop::SharedStorage)>>>(params, dOut.get(), M);
    CUTLASS_PPU_CHECK(hggcGetLastError());
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    std::vector<float> got(std::size_t(M) * kN);
    dOut.copy_to_host(got.data());
    return got;
  };

  auto const tagged_a = make_row_tag_a(M);
  auto const golden = golden_fp32(tagged_a, M, q, scales, zeros);
  auto const got = run(tagged_a);
  int const positive_bad = float_bitdiff(got, golden);
  std::printf("  G3 A-TAG M=%d raw-bitdiff=%d/%zu %s\n", M,
              positive_bad, golden.size(), positive_bad ? "MISMATCH" : "MATCH");

  // Exact negative: replay every later tile's local row 0..7.  The dedicated
  // K=33 fixture guarantees all N outputs differ, so the expected denominator
  // is known without consulting the device result.
  auto const replay_a = make_row_tag_a(M, true);
  auto const replay_golden = golden_fp32(replay_a, M, q, scales, zeros);
  int const fixture_red = float_bitdiff(replay_golden, golden);
  int const expected_red = (M - 8) * kN;
  auto const replay_got = run(replay_a);
  int const replay_oracle_bad = float_bitdiff(replay_got, replay_golden);
  int const observed_red = float_bitdiff(replay_got, golden);
  int const errors = positive_bad + replay_oracle_bad +
      (fixture_red != expected_red) + (observed_red != expected_red);
  std::printf("  G3 A-TAG-NEGATIVE M=%d replay-oracle-bitdiff=%d/%zu "
              "observed-red=%d expected-red=%d %s\n",
              M, replay_oracle_bad, replay_golden.size(), observed_red,
              expected_red, errors ? "FAIL" : "EXPECTED_RED");
  return errors;
}

template <class LaunchContract_>
struct G4EpilogueTypesBase {
  using LaunchContract = LaunchContract_;
  using Tile = typename LaunchContract::Tile;
  using Warp = typename LaunchContract::Warp;
  using Mainloop = typename LaunchContract::Mainloop;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
      Tile, Warp, cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      half_t, cutlass::layout::RowMajor*, 8,
      half_t, cutlass::layout::RowMajor*, 8,
      cutlass::epilogue::EpiloguePtrArraySimtVectorized,
      cutlass::epilogue::fusion::LinearCombination<half_t, float>>::CollectiveOp;
  static_assert(std::is_same_v<typename Epilogue::StrideD,
                               moe_grouped_ppu::DStride*>);
  static_assert(cute::size<0>(typename Epilogue::SmemLayout{}) ==
                cute::size<0>(typename Mainloop::TiledMma::AtomShape_MNK{}) *
                    cute::size<1>(
                        typename Mainloop::TiledMma::ThrLayoutVMNK{}));
};

template <int TM, int WM>
struct G4EpilogueTypes
    : G4EpilogueTypesBase<m8n16_g5_contract::Launch<TM, WM>> {};

CUTLASS_HOST_DEVICE
float g4_coordinate_tag(int m, int n);

// Epilogue-only topology closure for the exact grouped Q4 K-pack row that
// first failed at (local_m=8,n=32).  Both arms write the same logical Mx64
// tensor.  TN32/WN32 uses two one-warp CTAs along N; TN64/WN16 uses one CTA
// with the shipping row's four N warps.  Nothing below launches the mainloop
// or constructs A/B/metadata/scheduler state: its TiledMma is used solely as
// the production accumulator-coordinate contract consumed by the ptr-array
// epilogue.
template <int TN, int WN>
struct G4KpackEpilogueLaunchContract {
  using Schedule = ppu_group_schedule::FinegrainedSchedule<32>;
  using Tile = cute::Shape<cute::_8, cute::C<TN>, cute::_64>;
  using Scale = cute::Shape<cute::C<TN>, cute::_2>;
  using Warp = cute::Shape<cute::_8, cute::C<WN>, cute::_64>;
  using Policy = ppu_mixed_policy::Q4KPack4MainloopPolicy<
      ppu_mixed_policy::QuantMode::FinegrainedScaleZero,
      Schedule, Tile, Scale, Warp, 2, true, 0, 16,
      cutlass::gemm::SeparateHalfPlanes>;
  using Mainloop = typename Policy::CollectiveOp;
};

template <int TN, int WN>
struct G4KpackEpilogueTypes
    : G4EpilogueTypesBase<G4KpackEpilogueLaunchContract<TN, WN>> {
  using LaunchContract = G4KpackEpilogueLaunchContract<TN, WN>;
  using Policy = typename LaunchContract::Policy;
  using Mainloop = typename LaunchContract::Mainloop;
  using Epilogue = typename G4EpilogueTypesBase<LaunchContract>::Epilogue;
  using Descriptor = typename Policy::Descriptor;
  // Match the real grouped kernel's union carrier rather than allocating only
  // the epilogue member.  A malformed S2R map may address past that member yet
  // remain inside the production union; allocating only the member turns the
  // value/ownership failure under test into an unrelated TSM range exception.
  union SharedCarrier {
    typename Mainloop::SharedStorage mainloop;
    typename Epilogue::SharedStorage epilogue;
  };

  static_assert(TN == 32 || TN == 64);
  static_assert(TN % WN == 0 && (WN == 16 || WN == 32));
  static_assert(Descriptor::q4_kpack4_transpose &&
                Descriptor::kpack4_scheduled_delivery_n == 16 &&
                Descriptor::kpack4_resolved_delivery_n == 16 &&
                !Descriptor::interleaved_metadata);
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
  static_assert(Descriptor::packed_metadata,
                "the exact FQ topology runner must retain packed metadata");
#endif
  static_assert(std::is_same_v<typename Epilogue::StrideD,
                               moe_grouped_ppu::DStride*>);
  static_assert(cute::size<0>(typename Epilogue::SmemLayout{}) == 8);
  static_assert(cute::size<0>(typename Epilogue::SmemLayout{}) ==
                cute::size<0>(typename Mainloop::TiledMma::AtomShape_MNK{}) *
                    cute::size<1>(
                        typename Mainloop::TiledMma::ThrLayoutVMNK{}));
  static_assert(int(cute::size(typename Mainloop::TiledMma{})) ==
                32 * (TN / WN));
  static_assert(sizeof(SharedCarrier) >= sizeof(typename Epilogue::SharedStorage));
};

// Test-only reconstruction of the historical output-copy contract.  Keep the
// production Mainloop, TiledMma, shared layout and callbacks, and change only
// the S2R/R2G vector width back to the requested eight values.  This preserves
// the proved red without retaining a shipping workaround or legacy macro.
template <class Epilogue>
struct G4EpilogueScheduleOf;

template <class StrideC, class StrideD, class Callbacks, class SmemLayout,
          class CopyR2S, class CopyS2R, class CopyR2G, class Schedule,
          class CopyG2R>
struct G4EpilogueScheduleOf<cutlass::epilogue::collective::EpilogueEvt<
    StrideC, StrideD, Callbacks, SmemLayout, CopyR2S, CopyS2R, CopyR2G,
    Schedule, CopyG2R>> {
  using Type = Schedule;
};

template <int TN, int WN>
struct G4KpackLegacyEpilogueTypes : G4KpackEpilogueTypes<TN, WN> {
  using Base = G4KpackEpilogueTypes<TN, WN>;
  using Candidate = typename Base::Epilogue;
  static constexpr int ThreadNum = 32 * (TN / WN);
  static constexpr int Alignment = 8;
  static constexpr int EpiThreadN =
      ThreadNum < TN / Alignment ? ThreadNum : TN / Alignment;
  static constexpr int EpiThreadM = ThreadNum / EpiThreadN;
  using ThreadLayout = cute::Layout<
      cute::Shape<cute::C<EpiThreadM>, cute::C<EpiThreadN>>,
      cute::Stride<cute::C<EpiThreadN>, cute::_1>>;
  using ValueLayout = cute::Layout<cute::Shape<cute::_1, cute::C<Alignment>>>;
  using LegacyTiledCopyS2R = decltype(cute::make_tiled_copy(
      cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, float>{},
      ThreadLayout{}, ValueLayout{}));
  using LegacyCopyAtomR2G = cute::Copy_Atom<
      cute::AutoVectorizingCopyWithAssumedAlignment<sizeof(half_t) * Alignment * 8>,
      half_t>;
  using Epilogue = cutlass::epilogue::collective::EpilogueEvt<
      typename Candidate::StrideC, typename Candidate::StrideD,
      typename Candidate::FusionCallbacks, typename Candidate::SmemLayout,
      typename Candidate::CopyAtomR2S, LegacyTiledCopyS2R,
      LegacyCopyAtomR2G, typename G4EpilogueScheduleOf<Candidate>::Type>;
  using Mainloop = typename Base::Mainloop;
  union SharedCarrier {
    typename Mainloop::SharedStorage mainloop;
    typename Epilogue::SharedStorage epilogue;
  };
  static_assert(TN == 64 && WN == 16,
                "the retained legacy red is the exact failing topology");
  static_assert(cute::size(typename LegacyTiledCopyS2R::Tiler_MN{}) == 1024,
                "historical copy must retain the virtual 16x64 tile");
  static_assert(cute::size(typename Epilogue::SmemLayout{}) == 512,
                "legacy red must retain the real 8x64 shared tile");
};

using G4KpackCandidateSubject = G4KpackEpilogueTypes<64, 16>;
using G4KpackLegacySubject = G4KpackLegacyEpilogueTypes<64, 16>;
static_assert(cute::size(
                  typename G4KpackCandidateSubject::Epilogue::TiledCopyS2R::
                      Tiler_MN{}) == 512,
              "shipping TM8/TN64/WN16 copy must cover exactly 8x64");

constexpr int kG4TopologyN = 64;

template <class Types, bool ReplayFirstNWarp>
__global__ void g4_epilogue_topology_kernel(
    typename Types::Epilogue::Params params, int M, int N) {
  extern __shared__ char smem[];
  using Tile = typename Types::Tile;
  using Mainloop = typename Types::Mainloop;
  using Epilogue = typename Types::Epilogue;
  constexpr int TM = int(cute::size<0>(Tile{}));
  constexpr int TN = int(cute::size<1>(Tile{}));
  constexpr int WN = int(cute::size<1>(typename Types::Warp{}));
  int const tid = int(threadIdx.x);
  int const tile_m = int(blockIdx.x);
  int const tile_n = int(blockIdx.y);
  if (tile_m * TM >= M || tile_n * TN >= N ||
      tid >= int(cute::size(typename Mainloop::TiledMma{}))) return;

  // Production aliases the epilogue storage with a larger mainloop storage
  // union.  Give only the diagnostic tail outside the valid 8xTN epilogue
  // member a stable, finite FP32 tag.  This does not alter the production
  // epilogue mapping or its legal shared values; it only makes an invalid S2R
  // read deterministic instead of trapping or inheriting unspecified prior
  // mainloop bytes.
  static_assert(sizeof(typename Epilogue::SharedStorage) % sizeof(uint32_t) == 0);
  static_assert(sizeof(typename Types::SharedCarrier) % sizeof(uint32_t) == 0);
  auto* carrier_words = reinterpret_cast<uint32_t*>(smem);
  constexpr int kEpilogueWords =
      int(sizeof(typename Epilogue::SharedStorage) / sizeof(uint32_t));
  constexpr uint32_t kCarrierTagF32 = UINT32_C(0xc1500000);  // -13.0f
  for (int i = kEpilogueWords + tid;
       i < int(sizeof(typename Types::SharedCarrier) / sizeof(uint32_t));
       i += int(blockDim.x)) {
    carrier_words[i] = kCarrierTagF32;
  }
  __syncthreads();

  typename Mainloop::TiledMma tiled_mma;
  auto accum = cute::make_fragment_like<float>(cute::partition_fragment_C(
      tiled_mma, cute::take<0, 2>(Tile{})));
  auto cC = cute::make_identity_tensor(cute::take<0, 2>(Tile{}));
  auto tCcC = tiled_mma.get_thread_slice(tid).partition_C(cC);
  CUTE_STATIC_ASSERT_V(cute::size(tCcC) == cute::size(accum));
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < int(cute::size(accum)); ++i) {
    auto coord = tCcC(i);
    int const global_m = tile_m * TM + int(cute::get<0>(coord));
    int const global_n = tile_n * TN + int(cute::get<1>(coord));
    int const tagged_n = ReplayFirstNWarp ? global_n % WN : global_n;
    accum(i) = g4_coordinate_tag(global_m, tagged_n);
  }

  auto problem = cute::make_shape(M, N, kK, 1);
  auto block = cute::make_coord(tile_m, tile_n, cute::_, 0);
  auto residue = cute::make_tuple(M - tile_m * TM,
                                  N - tile_n * TN, 0);
  auto& storage = *reinterpret_cast<typename Epilogue::SharedStorage*>(smem);
  Epilogue epilogue{params, storage};
  auto& topology_store = epilogue;
  topology_store(problem, Tile{}, block, accum, tiled_mma, residue, tid,
                 reinterpret_cast<char*>(&storage));
}

CUTLASS_HOST_DEVICE
float g4_coordinate_tag(int m, int n) {
  // Integers stay exact through FP32 accumulator -> FP16 epilogue conversion.
  return float(1 + 64 * m + n);
}

template <class Types, bool ReplayFirstTile>
__global__ void g4_epilogue_tag_kernel(
    typename Types::Epilogue::Params params, int M) {
  extern __shared__ char smem[];
  using Tile = typename Types::Tile;
  using Mainloop = typename Types::Mainloop;
  using Epilogue = typename Types::Epilogue;
  constexpr int TM = int(cute::size<0>(Tile{}));
  constexpr int TN = int(cute::size<1>(Tile{}));
  int const tid = int(threadIdx.x);
  int const tile_m = int(blockIdx.x);
  int const tile_n = int(blockIdx.y);
  if (tile_m * TM >= M || tile_n * TN >= kN ||
      tid >= int(cute::size(typename Mainloop::TiledMma{}))) return;

  typename Mainloop::TiledMma tiled_mma;
  auto accum = cute::make_fragment_like<float>(cute::partition_fragment_C(
      tiled_mma, cute::take<0, 2>(Tile{})));
  auto cC = cute::make_identity_tensor(cute::take<0, 2>(Tile{}));
  auto tCcC = tiled_mma.get_thread_slice(tid).partition_C(cC);
  CUTE_STATIC_ASSERT_V(cute::size(tCcC) == cute::size(accum));
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < int(cute::size(accum)); ++i) {
    auto coord = tCcC(i);
    int const local_m = int(cute::get<0>(coord));
    int const global_m = tile_m * TM + local_m;
    int const global_n = tile_n * TN + int(cute::get<1>(coord));
    int const tagged_m = ReplayFirstTile && tile_m > 0 ? local_m : global_m;
    accum(i) = g4_coordinate_tag(tagged_m, global_n);
  }

  auto problem = cute::make_shape(M, kN, kK, 1);
  auto block = cute::make_coord(tile_m, tile_n, cute::_, 0);
  auto residue = cute::make_tuple(M - tile_m * TM,
                                  kN - tile_n * TN, 0);
  auto& storage = *reinterpret_cast<typename Epilogue::SharedStorage*>(smem);
  Epilogue epilogue{params, storage};
  epilogue(problem, Tile{}, block, accum, tiled_mma, residue, tid,
           reinterpret_cast<char*>(&storage));
}

template <int TM, int WM, bool UsePersistent = false>
bool launch_g4(
    half_t const* A, int4_t const* B, half_t const* scales, half_t const* zeros,
    half_t* D, int M, char* workspace, std::size_t workspace_bytes) {
  using Contract = m8n16_g5_contract::Launch<TM, WM>;
  using Tile = typename Contract::Tile;
  using Warp = typename Contract::Warp;
  using Scale = typename Contract::Scale;

  std::vector<GS> shapes{cute::make_shape(M, kN, kK)};
  cutlass::DeviceAllocation<GS> dShapes(1);
  dShapes.copy_from_host(shapes.data());
  auto stride = cutlass::make_cute_packed_stride(
      DStride{}, cute::make_shape(M, kN, 1));
  std::vector<half_t*> ptrs{D};
  std::vector<DStride> strides{stride};
  std::vector<int> group_m{M};
  cutlass::DeviceAllocation<half_t*> dPtrs(1);
  cutlass::DeviceAllocation<DStride> dStrides(1);
  cutlass::DeviceAllocation<int> dGroupM(1);
  dPtrs.copy_from_host(ptrs.data());
  dStrides.copy_from_host(strides.data());
  dGroupM.copy_from_host(group_m.data());

  bool const launched = moe_grouped_ppu::launch<
      Contract::QuantMode,
      typename Contract::BaseSchedule, Tile, Scale, Warp,
      Contract::Stages, Contract::AiuInterleaved,
      typename Contract::ElementB, void, false, false, false, 0,
      UsePersistent>(
          A, B, scales, zeros, dPtrs.get(), dStrides.get(), dGroupM.get(),
          M, kN, kK, 1, kGs, dShapes.get(), shapes.data(), nullptr,
          workspace, workspace_bytes, nullptr);
  // The pointer/stride/problem arrays above own the asynchronous launch's
  // arguments.  Do not let their RAII allocations go out of scope until the
  // kernel has consumed them.
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  return launched;
}

struct G4Result {
  std::vector<half_t> logical;
  int errors = 0;
  int guard_errors = 0;
  int finite_errors = 0;
};

template <bool ReplayFirstTile>
G4Result run_g4_epilogue_tag_arm(int M) {
  using Types = G4EpilogueTypes<8, 8>;
  using Epilogue = typename Types::Epilogue;
  using Mainloop = typename Types::Mainloop;
  std::size_t const logical_count = std::size_t(M) * kN;
  std::vector<half_t> host(kGuard + logical_count + kGuard);
  std::fill(host.begin(), host.begin() + kGuard, hbits(kLeftCanary));
  std::fill(host.begin() + kGuard, host.begin() + kGuard + logical_count,
            hbits(kOutputNaN));
  std::fill(host.begin() + kGuard + logical_count, host.end(),
            hbits(kRightCanary));

  cutlass::DeviceAllocation<half_t> dStorage(host.size());
  dStorage.copy_from_host(host.data());
  auto stride = cutlass::make_cute_packed_stride(
      DStride{}, cute::make_shape(M, kN, 1));
  std::vector<half_t*> ptrs{dStorage.get() + kGuard};
  std::vector<DStride> strides{stride};
  cutlass::DeviceAllocation<half_t*> dPtrs(1);
  cutlass::DeviceAllocation<DStride> dStrides(1);
  dPtrs.copy_from_host(ptrs.data());
  dStrides.copy_from_host(strides.data());

  typename Epilogue::Arguments args{};
  args.ptr_D = dPtrs.get();
  args.dD = dStrides.get();
  auto problem = cute::make_shape(M, kN, kK, 1);
  auto params = Epilogue::to_underlying_arguments(problem, args, nullptr);
  dim3 const grid((M + 7) / 8, (kN + 31) / 32, 1);
  g4_epilogue_tag_kernel<Types, ReplayFirstTile><<<
      grid, int(cute::size(typename Mainloop::TiledMma{})),
      sizeof(typename Epilogue::SharedStorage)>>>(params, M);
  CUTLASS_PPU_CHECK(hggcGetLastError());
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  dStorage.copy_to_host(host.data());

  G4Result result;
  for (int i = 0; i < kGuard; ++i) {
    result.guard_errors += host[i].raw() != kLeftCanary;
    result.guard_errors +=
        host[kGuard + logical_count + i].raw() != kRightCanary;
  }
  result.errors += result.guard_errors;
  result.logical.assign(host.begin() + kGuard,
                        host.begin() + kGuard + logical_count);
  for (half_t value : result.logical) {
    result.finite_errors += !std::isfinite(float(value));
  }
  result.errors += result.finite_errors;
  return result;
}

int half_bitdiff(std::vector<half_t> const& lhs,
                 std::vector<half_t> const& rhs) {
  if (lhs.size() != rhs.size()) return int(std::max(lhs.size(), rhs.size()));
  int bad = 0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    bad += lhs[i].raw() != rhs[i].raw();
  }
  return bad;
}

int half_row_bitdiff(std::vector<half_t> const& lhs,
                     std::vector<half_t> const& rhs, int row) {
  int bad = 0;
  for (int n = 0; n < kN; ++n) {
    std::size_t const i = std::size_t(row) * kN + n;
    bad += lhs[i].raw() != rhs[i].raw();
  }
  return bad;
}

int half_n_cohort_bitdiff(std::vector<half_t> const& lhs,
                          std::vector<half_t> const& rhs,
                          int M, int N, int cohort) {
  int bad = 0;
  int const begin = cohort * 16;
  int const end = std::min(begin + 16, N);
  for (int m = 0; m < M; ++m) {
    for (int n = begin; n < end; ++n) {
      std::size_t const i = std::size_t(m) * N + n;
      bad += lhs[i].raw() != rhs[i].raw();
    }
  }
  return bad;
}

template <class Types, bool ReplayFirstNWarp, bool FirstMTileOnly = false>
G4Result run_g4_epilogue_topology_arm(int M) {
  using Epilogue = typename Types::Epilogue;
  using Mainloop = typename Types::Mainloop;
  constexpr int TN = int(cute::size<1>(typename Types::Tile{}));
  constexpr int N = kG4TopologyN;
  std::size_t const logical_count = std::size_t(M) * N;
  std::vector<half_t> host(kGuard + logical_count + kGuard);
  std::fill(host.begin(), host.begin() + kGuard, hbits(kLeftCanary));
  std::fill(host.begin() + kGuard, host.begin() + kGuard + logical_count,
            hbits(kOutputNaN));
  std::fill(host.begin() + kGuard + logical_count, host.end(),
            hbits(kRightCanary));

  cutlass::DeviceAllocation<half_t> dStorage(host.size());
  dStorage.copy_from_host(host.data());
  auto stride = cutlass::make_cute_packed_stride(
      DStride{}, cute::make_shape(M, N, 1));
  std::vector<half_t*> ptrs{dStorage.get() + kGuard};
  std::vector<DStride> strides{stride};
  cutlass::DeviceAllocation<half_t*> dPtrs(1);
  cutlass::DeviceAllocation<DStride> dStrides(1);
  dPtrs.copy_from_host(ptrs.data());
  dStrides.copy_from_host(strides.data());

  typename Epilogue::Arguments args{};
  args.ptr_D = static_cast<half_t**>(dPtrs.get());
  args.dD = static_cast<DStride*>(dStrides.get());
  auto problem = cute::make_shape(M, N, kK, 1);
  auto params = Epilogue::to_underlying_arguments(problem, args, nullptr);
  dim3 const grid(FirstMTileOnly ? 1 : (M + 7) / 8,
                  (N + TN - 1) / TN, 1);
  g4_epilogue_topology_kernel<Types, ReplayFirstNWarp><<<
      grid, int(cute::size(typename Mainloop::TiledMma{})),
      sizeof(typename Types::SharedCarrier)>>>(params, M, N);
  CUTLASS_PPU_CHECK(hggcGetLastError());
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  dStorage.copy_to_host(host.data());

  G4Result result;
  for (int i = 0; i < kGuard; ++i) {
    result.guard_errors += host[i].raw() != kLeftCanary;
    result.guard_errors +=
        host[kGuard + logical_count + i].raw() != kRightCanary;
  }
  result.errors += result.guard_errors;
  result.logical.assign(host.begin() + kGuard,
                        host.begin() + kGuard + logical_count);
  std::size_t const written_count = FirstMTileOnly
      ? std::size_t(std::min(M, 8)) * N : logical_count;
  for (std::size_t i = 0; i < written_count; ++i) {
    result.finite_errors += !std::isfinite(float(result.logical[i]));
  }
  result.errors += result.finite_errors;
  return result;
}

template <class Types>
int check_g4_first_tile_ownership(char const* arm) {
  constexpr int M = 9;
  constexpr int N = kG4TopologyN;
  constexpr int TN = int(cute::size<1>(typename Types::Tile{}));
  constexpr int WN = int(cute::size<1>(typename Types::Warp{}));
  std::vector<half_t> expected(std::size_t(M) * N, hbits(kOutputNaN));
  for (int m = 0; m < 8; ++m) {
    for (int n = 0; n < N; ++n) {
      expected[std::size_t(m) * N + n] =
          half_t(g4_coordinate_tag(m, n));
    }
  }
  auto got = run_g4_epilogue_topology_arm<Types, false, true>(M);
  int first8_bad = 0;
  for (int m = 0; m < 8; ++m) {
    for (int n = 0; n < N; ++n) {
      std::size_t const i = std::size_t(m) * N + n;
      first8_bad += got.logical[i].raw() != expected[i].raw();
    }
  }
  int row8_written = 0;
  int cohort_written[4]{};
  for (int n = 0; n < N; ++n) {
    bool const written = got.logical[std::size_t(8) * N + n].raw() !=
                         kOutputNaN;
    row8_written += written;
    cohort_written[n / 16] += written;
  }
  int const ownership_bad = first8_bad + row8_written;
  bool const supporting_paths_clean =
      got.guard_errors == 0 && got.finite_errors == 0 && first8_bad == 0;
  char const* status = !supporting_paths_clean ? "VALUE_OR_GUARD_DIRTY" :
      (row8_written ? "ILLEGAL_ROW8_WRITE" : "EXACT_OWNER");
  int const errors = got.guard_errors + got.finite_errors + ownership_bad;
  std::printf("FQ_M8_EPILOGUE_FIRST_TILE_OWNERSHIP arm=%s M=9 "
              "TM=8 TN=%d WM=8 WN=%d ownership_bad=%d/%zu "
              "epilogue_smem=%zu carrier_smem=%zu "
              "first8_bad=%d/512 guard_bad=%d finite_bad=%d "
              "row8_written=%d/64 cohort_written=[%d,%d,%d,%d] %s\n",
              arm, TN, WN, ownership_bad, expected.size(),
              sizeof(typename Types::Epilogue::SharedStorage),
              sizeof(typename Types::SharedCarrier),
              first8_bad,
              got.guard_errors, got.finite_errors, row8_written,
              cohort_written[0], cohort_written[1], cohort_written[2],
              cohort_written[3], status);
  return errors;
}

template <class Types>
int check_g4_epilogue_topology(char const* arm, int M) {
  constexpr int N = kG4TopologyN;
  constexpr int TN = int(cute::size<1>(typename Types::Tile{}));
  constexpr int WN = int(cute::size<1>(typename Types::Warp{}));
  using TiledCopy = typename Types::Epilogue::TiledCopyS2R;
  constexpr int CtaThreads = 32 * (TN / WN);
  constexpr int FragmentSize = 8 * TN / CtaThreads;
  constexpr int CopyThreads = int(typename TiledCopy::TiledNumThr{});
  constexpr int OutputAlignment = int(typename TiledCopy::TiledNumVal{});
  constexpr int CopyTileValues =
      int(cute::size(typename TiledCopy::Tiler_MN{}));
  static_assert(CopyThreads == CtaThreads);
  static_assert(CopyTileValues == 8 * TN,
                "candidate output copy must cover exactly the logical 8xTN tile");
  static_assert(TN % OutputAlignment == 0);
  constexpr int EpiThreadN = TN / OutputAlignment;
  constexpr int EpiThreadM = CopyThreads / EpiThreadN;
  static_assert((TN == 32 && WN == 32 && FragmentSize == 8 &&
                 OutputAlignment == 8 && EpiThreadM == 8 &&
                 EpiThreadN == 4) ||
                (TN == 64 && WN == 16 && FragmentSize == 4 &&
                 OutputAlignment == 4 && EpiThreadM == 8 &&
                 EpiThreadN == 16));
  std::vector<half_t> golden(std::size_t(M) * N);
  std::vector<half_t> replay_golden(std::size_t(M) * N);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      golden[std::size_t(m) * N + n] =
          half_t(g4_coordinate_tag(m, n));
      replay_golden[std::size_t(m) * N + n] =
          half_t(g4_coordinate_tag(m, n % WN));
    }
  }

  auto positive = run_g4_epilogue_topology_arm<Types, false>(M);
  int const positive_bad = half_bitdiff(positive.logical, golden);
  auto replay = run_g4_epilogue_topology_arm<Types, true>(M);
  int const replay_oracle_bad =
      half_bitdiff(replay.logical, replay_golden);
  int const observed_red = half_bitdiff(replay.logical, golden);
  int const expected_red = M * (N - WN);
  int cohort_red[4]{};
  for (int c = 0; c < 4; ++c) {
    cohort_red[c] = half_n_cohort_bitdiff(replay.logical, golden, M, N, c);
  }
  int const expected_cohort[4] = {
      0, WN == 16 ? M * 16 : 0, M * 16, M * 16};
  int errors = positive.errors + positive_bad + replay.errors +
      replay_oracle_bad + (observed_red != expected_red);
  for (int c = 0; c < 4; ++c) {
    errors += cohort_red[c] != expected_cohort[c];
  }
  std::printf("FQ_M8_EPILOGUE_TOPOLOGY arm=%s M=%d TM=8 TN=%d WM=8 WN=%d "
              "cta_threads=%d fragment=%d output_alignment=%d "
              "epi_thread_map=%dx%d packed_metadata=%d positive_bad=%d/%zu "
              "negative_oracle_bad=%d/%zu observed_red=%d "
              "expected_red=%d cohort_red=[%d,%d,%d,%d] %s\n",
              arm, M, TN, WN, CtaThreads, FragmentSize, OutputAlignment,
              EpiThreadM, EpiThreadN,
              int(Types::Descriptor::packed_metadata),
              positive_bad, golden.size(),
              replay_oracle_bad, replay_golden.size(), observed_red,
              expected_red, cohort_red[0], cohort_red[1], cohort_red[2],
              cohort_red[3], errors ? "FAIL" : "EXPECTED_RED");
  return errors;
}

int run_g4_epilogue_topology() {
  using Control = G4KpackEpilogueTypes<32, 32>;
  using Candidate = G4KpackCandidateSubject;
  using Legacy = G4KpackLegacySubject;
  int errors = 0;
  errors += check_g4_first_tile_ownership<Control>("tn32-wn32-control");
  int const legacy_red =
      check_g4_first_tile_ownership<Legacy>("tn64-wn16-legacy");
  errors += legacy_red == 64 ? 0 : 1;
  errors += check_g4_first_tile_ownership<Candidate>("tn64-wn16-candidate");
  errors += check_g4_epilogue_topology<Control>("tn32-wn32-control", 9);
  errors += check_g4_epilogue_topology<Candidate>(
      "tn64-wn16-candidate-m8-control", 8);
  errors += check_g4_epilogue_topology<Candidate>(
      "tn64-wn16-candidate", 9);
  auto control = run_g4_epilogue_topology_arm<Control, false>(9);
  auto subject = run_g4_epilogue_topology_arm<Candidate, false>(9);
  int const cross_bad = half_bitdiff(control.logical, subject.logical);
  errors += control.errors + subject.errors + cross_bad;
  std::printf("FQ_M8_EPILOGUE_TOPOLOGY_AB M=%d N=%d cross_bad=%d/%zu "
              "verdict=%s\n", 9, kG4TopologyN, cross_bad,
              control.logical.size(),
              errors ? "DIRTY" : "CANDIDATE_MATCHES_CONTROL");
  return errors;
}

int run_g4_epilogue_tag(int M) {
  std::vector<half_t> golden(std::size_t(M) * kN);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < kN; ++n) {
      golden[std::size_t(m) * kN + n] = half_t(g4_coordinate_tag(m, n));
    }
  }
  auto positive = run_g4_epilogue_tag_arm<false>(M);
  int const positive_bad = half_bitdiff(positive.logical, golden);
  std::printf("  G4 EPILOGUE-TAG M=%d raw-bitdiff=%d/%zu %s\n", M,
              positive_bad, golden.size(),
              (positive.errors || positive_bad) ? "MISMATCH" : "MATCH");

  auto replay = run_g4_epilogue_tag_arm<true>(M);
  int const observed_red = half_bitdiff(replay.logical, golden);
  int const expected_red = (M - 8) * kN;
  int const row8_red = half_row_bitdiff(replay.logical, golden, 8);
  int const row16_red = M > 16 ?
      half_row_bitdiff(replay.logical, golden, 16) : -1;
  int const errors = positive.errors + positive_bad + replay.errors +
      (observed_red != expected_red) + (row8_red != kN) +
      (M > 16 && row16_red != kN);
  std::printf("  G4 EPILOGUE-TAG-NEGATIVE M=%d observed-red=%d "
              "expected-red=%d row8-red=%d row16-red=%d %s\n",
              M, observed_red, expected_red, row8_red, row16_red,
              errors ? "FAIL" : "EXPECTED_RED");
  return errors;
}

template <int TM, int WM, bool UsePersistent = false>
G4Result run_g4_arm(
    char const* family, int M, half_t const* dA, int4_t const* dB,
    half_t const* dScale, half_t const* dZero) {
  std::size_t const logical_count = std::size_t(M) * kN;
  std::vector<half_t> host(kGuard + logical_count + kGuard);
  std::fill(host.begin(), host.begin() + kGuard, hbits(kLeftCanary));
  std::fill(host.begin() + kGuard, host.begin() + kGuard + logical_count,
            hbits(kOutputNaN));
  std::fill(host.begin() + kGuard + logical_count, host.end(),
            hbits(kRightCanary));

  cutlass::DeviceAllocation<half_t> dStorage(host.size());
  // One CTA needs only a tiny scheduler workspace; overallocate deliberately
  // so this test does not duplicate scheduler-internal byte arithmetic.
  cutlass::DeviceAllocation<char> workspace(4096);
  dStorage.copy_from_host(host.data());
  int const fail_before = moe_grouped_ppu::moeg_fail_count();
  bool const launched = launch_g4<TM, WM, UsePersistent>(
      dA, dB, dScale, dZero, dStorage.get() + kGuard, M,
      workspace.get(), workspace.capacity);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  int const launch_failures = moe_grouped_ppu::moeg_fail_count() - fail_before;
  dStorage.copy_to_host(host.data());

  G4Result result;
  if (!launched || launch_failures != 0) {
    std::printf("    %s M=%d launch failed: returned=%d fail_delta=%d\n",
                family, M, int(launched), launch_failures);
    ++result.errors;
  }
  for (int i = 0; i < kGuard; ++i) {
    if (host[i].raw() != kLeftCanary) ++result.errors;
  }
  for (int i = 0; i < kGuard; ++i) {
    if (host[kGuard + logical_count + i].raw() != kRightCanary) ++result.errors;
  }
  result.logical.assign(host.begin() + kGuard,
                        host.begin() + kGuard + logical_count);
  for (half_t v : result.logical) {
    if (std::isnan(float(v)) || !std::isfinite(float(v))) ++result.errors;
  }
  if (result.errors) {
    std::printf("    %s M=%d canary/overwrite errors=%d\n",
                family, M, result.errors);
  }
  return result;
}

int check_g4_values(
    char const* family, int M, std::vector<half_t> const& got,
    std::vector<float> const& golden) {
  int bad = 0;
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < got.size(); ++i) {
    // The formal epilogue converts FP32 accumulator to fp16.
    half_t const want_h(golden[i]);
    float const err = std::abs(float(got[i]) - float(want_h));
    max_abs = std::max(max_abs, err);
    if (!std::isfinite(float(got[i])) || err > 1.0e-3f) {
      if (bad < 4) {
        std::printf("    %s M=%d i=%zu got=%g want=%g err=%g\n",
                    family, M, i, double(float(got[i])),
                    double(float(want_h)), double(err));
      }
      ++bad;
    }
  }
  std::printf("  G4 %-5s M=%d golden bad=%d/%zu max_abs=%.3e %s\n",
              family, M, bad, got.size(), double(max_abs),
              bad ? "MISMATCH" : "MATCH");
  return bad;
}

int check_m8_m16(int M, std::vector<half_t> const& m8,
                 std::vector<half_t> const& m16) {
  int bad = 0;
  for (std::size_t i = 0; i < m8.size(); ++i) {
    if (m8[i].raw() != m16[i].raw()) {
      if (bad < 4) {
        std::printf("    G4 A/B M=%d i=%zu m8=%g(0x%04x) m16=%g(0x%04x)\n",
                    M, i, double(float(m8[i])), unsigned(m8[i].raw()),
                    double(float(m16[i])), unsigned(m16[i].raw()));
      }
      ++bad;
    }
  }
  std::printf("  G4 m8-vs-m16 M=%d bitdiff=%d/%zu %s\n",
              M, bad, m8.size(), bad ? "MISMATCH" : "MATCH");
  return bad;
}

}  // namespace

// ===================================================================================================
// G5 -- the real route.  #108.
//
// WHAT G4 CANNOT TEST, AND WHY THAT IS NOT A MATTER OF SIZE.  G4 launches with `L = 1`: one group, one
// expert, offset zero.  Its oracle (`golden_fp32`) is genuinely independent -- a scalar (m,n,k) triple
// loop with explicit dequant, no CUTLASS layout in sight -- so it does catch an arithmetic or layout
// error.  What it cannot catch is anything about the ROUTE, because with one group there is no route:
// no slot->expert mapping to invert, no row offset to be off by one, and no expert that should not be
// read.  Growing M does not create one.  That is why an L=1 self-comparison "must not be presented as
// G5" (see the header): the missing coverage is structural, not statistical.
//
// The launch API already expresses the real thing -- `L = num_experts`, B/scales strided by the group
// index, `group_row_offsets` for the ragged A -- so this is a fixture, not an interface change.
//
// TWO PROPERTIES DO THE WORK, and neither is "more experts":
//
//   1. EVERY EXPERT'S DATA IS DIFFERENT.  Today's fixture gives all experts identical W/S/Z (both here
//      and in benchmarks/gemv_perf_common.hpp, which memcpys one packed plane L times).  Under that
//      fixture a kernel that uses `slot` where `real_expert_id` belongs computes the right answer, and
//      so does one that reads its neighbour.  Per-expert salting is what makes those two defects have
//      an observable consequence at all.
//   2. THE INACTIVE EXPERTS ARE POISON, NOT ABSENT.  All 256 experts are allocated and filled, the 248
//      inactive ones with a scale ~100x the active range.  Reading one is then a large, obvious error
//      rather than a plausible number.  Allocating only the active experts would turn the same defect
//      into an out-of-bounds read, which is not reliably observable.
//
// The active IDs are deliberately non-contiguous and include 255: an off-by-one at the top of the
// table is otherwise unreachable.
constexpr int kE = m8n16_g5_contract::kExperts;
constexpr int kActive = 8;
constexpr int kActiveIds[kActive] = {3, 17, 42, 88, 129, 190, 201, 255};

// Salted generators.  `salt == 0` reproduces the G3/G4 pattern EXACTLY -- the salt terms vanish -- so
// this addition cannot perturb the arms that are already proved.  The salt is mixed into the generator
// rather than added to its output, because a uniform offset would leave a wrong-expert read numerically
// close to the right one, and "close" is what this gate exists to reject.
std::vector<std::uint8_t> make_codes_salted(int salt) {
  std::vector<std::uint8_t> q(std::size_t(kK) * kN);
  for (int k = 0; k < kK; ++k)
    for (int n = 0; n < kN; ++n)
      q[std::size_t(k) * kN + n] = std::uint8_t(
          (11 * k + 7 * n + 3 * (k / kGs) + (k ^ n) + 29 * salt + salt * (k + n)) & 15);
  return q;
}

std::vector<half_t> make_scales_salted(int salt) {
  std::vector<half_t> s(std::size_t(kScaleK) * kN);
  for (int g = 0; g < kScaleK; ++g)
    for (int n = 0; n < kN; ++n)
      s[std::size_t(g) * kN + n] =
          half_t(float(1 + ((5 * g + 3 * n + 2 * salt) & 7)) / 32.0f);
  return s;
}

std::vector<half_t> make_zeros_salted(std::vector<half_t> const& scales, int salt) {
  std::vector<half_t> z(scales.size());
  for (int g = 0; g < kScaleK; ++g)
    for (int n = 0; n < kN; ++n) {
      float const offset = float(((13 * n + 5 * g + 3 * salt) % 7) - 3) / 16.0f;
      z[std::size_t(g) * kN + n] =
          half_t(8.0f * float(scales[std::size_t(g) * kN + n]) + offset);
    }
  return z;
}

// ~100x the active scale range (active is 1/32 .. 8/32).  A wrong-expert read is then off by two orders
// of magnitude, not by a plausible-looking amount.
std::vector<half_t> make_poison_scales(int e) {
  std::vector<half_t> s(std::size_t(kScaleK) * kN);
  for (std::size_t i = 0; i < s.size(); ++i)
    s[i] = half_t(4.0f + float((int(i) + e) & 7));
  return s;
}

std::vector<half_t> make_a_salted(int rows, int salt) {
  std::vector<half_t> a(std::size_t(rows) * kK);
  for (int m = 0; m < rows; ++m)
    for (int k = 0; k < kK; ++k)
      a[std::size_t(m) * kK + k] =
          half_t(float(1 + ((3 * m + 5 * k + 7 * salt) & 7)) / 8.0f);
  return a;
}

struct G5Fixture {
  std::vector<std::int8_t> B;                 // [kE] placed artifacts, expert-major
  std::vector<half_t> scales, zeros, A;
  std::vector<int> group_m, row_offsets;      // [kE], [kE + 1]
  std::vector<int> expert_for_slot, slot_for_expert;
  std::vector<std::vector<std::uint8_t>> q_by_slot;
  std::vector<std::vector<half_t>> s_by_slot, z_by_slot, a_by_slot;
  bool route_valid = false;
  int total_rows = 0;
};

void set_g5_route(G5Fixture& f, int rows_per_expert,
                  int const* ids, int n_active) {
  auto route = m8n16_g5_slot_map::make(
      kE, ids, n_active, rows_per_expert);
  f.route_valid = route.valid;
  f.expert_for_slot = std::move(route.expert_for_slot);
  f.slot_for_expert = std::move(route.slot_for_expert);
  f.group_m = std::move(route.group_m);
  f.row_offsets = std::move(route.row_offsets);
  f.total_rows = route.total_rows;
}

// TM/WM only select which placement to write; the artifact is byte-identical for m8 and m16 (proved
// above), so one buffer serves both arms and a divergence would already have failed the offline check.
G5Fixture make_g5_fixture(int rows_per_expert, int const* ids, int n_active, bool uniform) {
  G5Fixture f;
  f.B.assign(std::size_t(kE) * kPlacedBBytes, 0);
  f.scales.assign(std::size_t(kE) * kScaleK * kN, half_t(0.0f));
  f.zeros.assign(f.scales.size(), half_t(0.0f));
  set_g5_route(f, rows_per_expert, ids, n_active);
  f.q_by_slot.resize(n_active);
  f.s_by_slot.resize(n_active);
  f.z_by_slot.resize(n_active);
  f.a_by_slot.resize(n_active);

  for (int e = 0; e < kE; ++e) {
    int const slot = f.route_valid ? f.slot_for_expert[e] : -1;
    // salt 0 is reserved for the G3/G4 data; actives take 1..kActive so no two experts share a pattern.
    int const salt = (slot >= 0) ? (uniform ? 1 : slot % 64 + 1) : (n_active + 1 + e);
    auto q = make_codes_salted(salt);
    auto s = (slot >= 0) ? make_scales_salted(salt) : make_poison_scales(e);
    auto z = make_zeros_salted(s, salt);
    xplane::place_derived<4, 8, 32, 64, 8, 32, 1>(
        f.B.data() + std::size_t(e) * kPlacedBBytes, q, kN, kK);
    std::copy(s.begin(), s.end(), f.scales.begin() + std::size_t(e) * kScaleK * kN);
    std::copy(z.begin(), z.end(), f.zeros.begin() + std::size_t(e) * kScaleK * kN);
    if (slot >= 0) {
      f.q_by_slot[slot] = std::move(q);
      f.s_by_slot[slot] = std::move(s);
      f.z_by_slot[slot] = std::move(z);
      f.a_by_slot[slot] = make_a_salted(rows_per_expert, salt);
    }
  }
  f.A.assign(std::size_t(f.total_rows) * kK, half_t(0.0f));
  if (f.route_valid) {
    for (int slot = 0; slot < n_active; ++slot) {
      int const expert = f.expert_for_slot[slot];
      std::copy(f.a_by_slot[slot].begin(), f.a_by_slot[slot].end(),
                f.A.begin() + std::size_t(f.row_offsets[expert]) * kK);
    }
  }
  return f;
}

template <int TM, int WM>
bool launch_g5(G5Fixture const& f, half_t const* dA, int4_t const* dB,
               half_t const* dScale, half_t const* dZero, half_t* dD,
               int rows_per_expert, char* workspace, std::size_t workspace_bytes) {
  using Contract = m8n16_g5_contract::Launch<TM, WM>;
  using Tile = typename Contract::Tile;
  using Warp = typename Contract::Warp;
  using Scale = typename Contract::Scale;

  std::vector<GS> shapes(kE);
  std::vector<half_t*> ptrs(kE);
  std::vector<DStride> strides(kE);
  for (int e = 0; e < kE; ++e) {
    int const Me = f.group_m[e];
    shapes[e] = cute::make_shape(Me, kN, kK);
    ptrs[e] = dD + std::size_t(f.row_offsets[e]) * kN;
    strides[e] = cutlass::make_cute_packed_stride(
        DStride{}, cute::make_shape(Me > 0 ? Me : 1, kN, 1));
  }
  cutlass::DeviceAllocation<GS> dShapes(kE);
  cutlass::DeviceAllocation<half_t*> dPtrs(kE);
  cutlass::DeviceAllocation<DStride> dStrides(kE);
  cutlass::DeviceAllocation<int> dGroupM(kE);
  cutlass::DeviceAllocation<int> dOffs(kE + 1);
  dShapes.copy_from_host(shapes.data());
  dPtrs.copy_from_host(ptrs.data());
  dStrides.copy_from_host(strides.data());
  dGroupM.copy_from_host(f.group_m.data());
  dOffs.copy_from_host(f.row_offsets.data());

  bool const launched = moe_grouped_ppu::launch<
      Contract::QuantMode, typename Contract::BaseSchedule, Tile, Scale, Warp,
      Contract::Stages, Contract::AiuInterleaved, typename Contract::ElementB>(
          dA, dB, dScale, dZero, dPtrs.get(), dStrides.get(), dGroupM.get(),
          rows_per_expert, kN, kK, kE, kGs, dShapes.get(), shapes.data(),
          dOffs.get(), workspace, workspace_bytes, nullptr);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  return launched;
}

// Returns the device output, or an empty vector if the launch did not happen.
template <int TM, int WM>
std::vector<half_t> run_g5_arm(char const* family, G5Fixture const& f, int rows_per_expert,
                               half_t const* dA, int4_t const* dB,
                               half_t const* dScale, half_t const* dZero, int* errors) {
  std::size_t const count = std::size_t(f.total_rows) * kN;
  std::vector<half_t> host(kGuard + count + kGuard);
  std::fill(host.begin(), host.begin() + kGuard, hbits(kLeftCanary));
  std::fill(host.begin() + kGuard, host.begin() + kGuard + count, hbits(kOutputNaN));
  std::fill(host.begin() + kGuard + count, host.end(), hbits(kRightCanary));

  cutlass::DeviceAllocation<half_t> dStorage(host.size());
  cutlass::DeviceAllocation<char> workspace(1 << 20);
  dStorage.copy_from_host(host.data());
  int const fail_before = moe_grouped_ppu::moeg_fail_count();
  bool const launched = launch_g5<TM, WM>(
      f, dA, dB, dScale, dZero, dStorage.get() + kGuard, rows_per_expert,
      workspace.get(), workspace.size());
  int const failed = moe_grouped_ppu::moeg_fail_count() - fail_before;
  if (!launched || failed) {
    std::printf("  G5 %-4s LAUNCH FAILED (launched=%d init_failures=%d) -- FAIL\n",
                family, int(launched), failed);
    ++*errors;
    return {};
  }
  dStorage.copy_to_host(host.data());
  for (int i = 0; i < kGuard; ++i) {
    if (host[i] != hbits(kLeftCanary) || host[kGuard + count + i] != hbits(kRightCanary)) {
      std::printf("  G5 %-4s CANARY CLOBBERED at %d -- FAIL\n", family, i);
      ++*errors;
      return {};
    }
  }
  return std::vector<half_t>(host.begin() + kGuard, host.begin() + kGuard + count);
}

// Compare one slot's rows against a golden built from a NAMED expert's data.  `oracle_slot` is normally
// the slot itself; the negative controls pass a different one on purpose.
int g5_slot_mismatches(G5Fixture const& f, std::vector<half_t> const& got, int rows_per_expert,
                       int slot, int oracle_slot) {
  auto golden = golden_fp32(f.a_by_slot[slot], rows_per_expert,
                            f.q_by_slot[oracle_slot], f.s_by_slot[oracle_slot],
                            f.z_by_slot[oracle_slot]);
  int const expert = f.expert_for_slot[slot];
  std::size_t const base = std::size_t(f.row_offsets[expert]) * kN;
  int bad = 0;
  for (std::size_t i = 0; i < golden.size(); ++i) {
    float const g = golden[i], d = float(got[base + i]);
    float const tol = 3e-2f * std::max(1.0f, std::fabs(g));
    if (!(std::fabs(g - d) <= tol)) ++bad;
  }
  return bad;
}

int run_g5(char const* label, int rows_per_expert, int const* ids, int n_active, bool uniform);

int run_g5(char const* label, int rows_per_expert, int const* ids, int n_active, bool uniform) {
  int errors = 0;
  auto f = make_g5_fixture(rows_per_expert, ids, n_active, uniform);
  if (!f.route_valid) {
    std::printf("[G5:%s] invalid slot->expert map -- FAIL\n", label);
    return 1;
  }

  cutlass::DeviceAllocation<int4_t> dB(f.B.size());
  cutlass::DeviceAllocation<half_t> dScale(f.scales.size());
  cutlass::DeviceAllocation<half_t> dZero(f.zeros.size());
  cutlass::DeviceAllocation<half_t> dA(f.A.size());
  CUTLASS_PPU_CHECK(hggcMemcpy(dB.get(), f.B.data(), f.B.size(), hggcMemcpyHostToDevice));
  dScale.copy_from_host(f.scales.data());
  dZero.copy_from_host(f.zeros.data());
  dA.copy_from_host(f.A.data());

  std::printf("[G5:%s] E=%d active=%d rows/expert=%d total_rows=%d uniform=%d active_ids=",
              label, kE, n_active, rows_per_expert, f.total_rows, int(uniform));
  for (int i = 0; i < n_active && i < 8; ++i) std::printf("%d%s", ids[i], i + 1 < kActive ? "," : "\n");

  auto m8 = run_g5_arm<8, 8>("m8", f, rows_per_expert, dA.get(), dB.get(),
                             dScale.get(), dZero.get(), &errors);
  auto m16 = run_g5_arm<16, 16>("m16", f, rows_per_expert, dA.get(), dB.get(),
                                dScale.get(), dZero.get(), &errors);
  if (m8.empty() || m16.empty()) return errors ? errors : 1;

  int reported = 0;
  for (int slot = 0; slot < n_active; ++slot) {
    int const b8 = g5_slot_mismatches(f, m8, rows_per_expert, slot, slot);
    int const b16 = g5_slot_mismatches(f, m16, rows_per_expert, slot, slot);
    int const outputs = rows_per_expert * kN;
    if ((b8 || b16) ? reported++ < 8 : slot < 8)
    std::printf("  G5 slot=%d expert=%-3d m8 bad=%d/%d  m16 bad=%d/%d  %s\n",
                slot, ids[slot], b8, outputs, b16, outputs,
                (b8 || b16) ? "FAIL" : "MATCH");
    errors += b8 + b16;
  }

  // NEGATIVE CONTROLS.  Both are host-side: they re-run the oracle with the wrong data and require the
  // comparison to FAIL.  Without them a green G5 would only mean "the numbers matched something", and a
  // fixture whose experts are indistinguishable matches everything -- which is exactly the defect this
  // gate was built to remove, so it must be shown not to be present in the gate itself.
  // IN UNIFORM MODE THESE CONTROLS CANNOT FIRE, BY CONSTRUCTION: every active expert holds the same
  // W/S/Z, so swapping two of them changes nothing and shifting rows between them changes nothing.
  // Reporting that as a failure was my error, not a finding -- it made UNIFORM (a diagnostic config,
  // not a gate) print FAIL for a reason unrelated to what it measures.
  if (uniform) {
    std::printf("  G5 NEGATIVE controls SKIPPED: uniform data makes them vacuous by construction\n");
  } else {
  int wrong_expert_detected = 0;
  for (int slot = 0; slot + 1 < n_active; ++slot)
    if (g5_slot_mismatches(f, m8, rows_per_expert, slot, slot + 1) > 0) ++wrong_expert_detected;
  if (wrong_expert_detected != n_active - 1) {
    std::printf("  G5 NEGATIVE expert-identity: only %d/%d neighbour swaps were detected -- the experts "
                "are not distinguishable, so a passing G5 proves nothing -- FAIL\n",
                wrong_expert_detected, n_active - 1);
    ++errors;
  } else {
    std::printf("  G5 NEGATIVE expert-identity: %d/%d neighbour swaps rejected EXPECTED_RED\n",
                wrong_expert_detected, n_active - 1);
  }

  // An off-by-one in the row offsets shifts every slot's rows by one expert's worth.  If that still
  // compares equal, the row offsets are not load-bearing in this fixture.
  int shift_detected = 0;
  for (int slot = 0; slot + 1 < n_active; ++slot) {
    auto golden = golden_fp32(f.a_by_slot[slot], rows_per_expert,
                              f.q_by_slot[slot], f.s_by_slot[slot],
                              f.z_by_slot[slot]);
    int const next_expert = f.expert_for_slot[slot + 1];
    std::size_t const base = std::size_t(f.row_offsets[next_expert]) * kN;
    int bad = 0;
    for (std::size_t i = 0; i < golden.size(); ++i)
      if (!(std::fabs(golden[i] - float(m8[base + i])) <= 3e-2f * std::max(1.0f, std::fabs(golden[i]))))
        ++bad;
    if (bad > 0) ++shift_detected;
  }
  if (shift_detected != n_active - 1) {
    std::printf("  G5 NEGATIVE row-offset: only %d/%d one-expert shifts were detected -- FAIL\n",
                shift_detected, n_active - 1);
    ++errors;
  } else {
    std::printf("  G5 NEGATIVE row-offset: %d/%d one-expert shifts rejected EXPECTED_RED\n",
                shift_detected, n_active - 1);
  }

  }
  int const cross = check_m8_m16(f.total_rows, m8, m16);
  errors += cross;
  std::printf("[G5:%s] %s: E=%d/active=%d uniform=%d\n",
              label, errors ? "FAIL" : "PASS", kE, n_active, int(uniform));
  return errors;
}


// ===================================================================================================
// IDPROBE -- stop inferring which expert was read, and read it off the output.
//
// LOW/UNIFORM/DENSE narrowed the cause to "the expert used to fetch B/scales is not the one the slot
// owns", but every one of those configurations answers it by INFERENCE: a mismatch says "not this
// expert", never "that one instead".  Two hypotheses (off-by-one neighbour, and something specific to
// id >= 128) both survive, which is exactly when a controlled input beats another comparison.
//
// Construction: q == 8 everywhere so the int4 converter's q-8 term is exactly zero; the zero plane
// carries e/256, which is a dyadic and therefore exact in fp16 for every e in [0,256); A is all ones.
// Then every dequantised weight is exactly e/256 and the FP32 accumulation over K=256 is exactly e.
// The output IS the expert id the kernel used -- no oracle, no tolerance, no interpretation.
std::vector<std::uint8_t> make_probe_codes() {
  return std::vector<std::uint8_t>(std::size_t(kK) * kN, std::uint8_t(8));
}
std::vector<half_t> make_probe_zeros(int e) {
  return std::vector<half_t>(std::size_t(kScaleK) * kN, half_t(float(e) / 256.0f));
}

int run_g5_idprobe(int const* ids, int n_active) {
  G5Fixture f;
  f.B.assign(std::size_t(kE) * kPlacedBBytes, 0);
  f.scales.assign(std::size_t(kE) * kScaleK * kN, half_t(1.0f / 32.0f));
  f.zeros.assign(f.scales.size(), half_t(0.0f));
  set_g5_route(f, 1, ids, n_active);
  if (!f.route_valid) return 1;
  auto q = make_probe_codes();
  for (int e = 0; e < kE; ++e) {
    auto z = make_probe_zeros(e);
    xplane::place_derived<4, 8, 32, 64, 8, 32, 1>(
        f.B.data() + std::size_t(e) * kPlacedBBytes, q, kN, kK);
    std::copy(z.begin(), z.end(), f.zeros.begin() + std::size_t(e) * kScaleK * kN);
  }
  f.A.assign(std::size_t(f.total_rows) * kK, half_t(1.0f));

  cutlass::DeviceAllocation<int4_t> dB(f.B.size());
  cutlass::DeviceAllocation<half_t> dScale(f.scales.size());
  cutlass::DeviceAllocation<half_t> dZero(f.zeros.size());
  cutlass::DeviceAllocation<half_t> dA(f.A.size());
  CUTLASS_PPU_CHECK(hggcMemcpy(dB.get(), f.B.data(), f.B.size(), hggcMemcpyHostToDevice));
  dScale.copy_from_host(f.scales.data());
  dZero.copy_from_host(f.zeros.data());
  dA.copy_from_host(f.A.data());

  int errors = 0;
  auto got = run_g5_arm<8, 8>("m8", f, 1, dA.get(), dB.get(), dScale.get(), dZero.get(), &errors);
  if (got.empty()) return errors ? errors : 1;

  std::printf("[G5:IDPROBE] output value == the expert id actually read (want == slot's own id)\n");
  int wrong = 0;
  for (int slot = 0; slot < n_active; ++slot) {
    int const want = ids[slot];
    half_t const expected{float(want)};
    std::size_t const row = std::size_t(f.row_offsets[want]);
    half_t const first = got[row * kN];
    bool row_exact = true;
    for (int n = 0; n < kN; ++n)
      row_exact &= got[row * kN + n].raw() == expected.raw();
    bool const ok = row_exact;
    if (!ok && wrong++ < 16)
      std::printf("  IDPROBE slot=%-3d owns_expert=%-3d "
                  "first=%.3f/0x%04x want=%.3f/0x%04x%s\n",
                  slot, want, double(float(first)), unsigned(first.raw()),
                  double(float(expected)), unsigned(expected.raw()),
                  first.raw() == expected.raw()
                      ? "  (first matches; another column diverges)" : "");
    if (!ok) ++errors;
  }
  std::printf("[G5:IDPROBE] %s: %d/%d slots read an expert other than their own\n",
              errors ? "FAIL" : "PASS", errors, n_active);
  return errors;
}

// B-side companion to the zero-plane probe above.  Metadata is deliberately
// inert (scale=1, zero=0), and expert e stores q=9 for the first e K values of
// every column and q=8 for the rest.  The converter emits q-8, A is one, and
// K=256, so every output is the integer e exactly.  e<=255 is exactly
// representable in fp16: compare raw bits, not a tolerance or a rounded ID.
std::vector<std::uint8_t> make_b_probe_codes(int e) {
  std::vector<std::uint8_t> q(std::size_t(kK) * kN, std::uint8_t(8));
  for (int k = 0; k < e; ++k)
    for (int n = 0; n < kN; ++n)
      q[std::size_t(k) * kN + n] = std::uint8_t(9);
  return q;
}

int run_g5_b_idprobe(int const* ids, int n_active) {
  G5Fixture f;
  f.B.assign(std::size_t(kE) * kPlacedBBytes, 0);
  f.scales.assign(std::size_t(kE) * kScaleK * kN, half_t(1.0f));
  f.zeros.assign(f.scales.size(), half_t(0.0f));
  set_g5_route(f, 1, ids, n_active);
  if (!f.route_valid) return 1;

  for (int e = 0; e < kE; ++e) {
    auto const q = make_b_probe_codes(e);
    xplane::place_derived<4, 8, 32, 64, 8, 32, 1>(
        f.B.data() + std::size_t(e) * kPlacedBBytes, q, kN, kK);
  }
  f.A.assign(std::size_t(f.total_rows) * kK, half_t(1.0f));

  cutlass::DeviceAllocation<int4_t> dB(f.B.size());
  cutlass::DeviceAllocation<half_t> dScale(f.scales.size());
  cutlass::DeviceAllocation<half_t> dZero(f.zeros.size());
  cutlass::DeviceAllocation<half_t> dA(f.A.size());
  CUTLASS_PPU_CHECK(hggcMemcpy(
      dB.get(), f.B.data(), f.B.size(), hggcMemcpyHostToDevice));
  dScale.copy_from_host(f.scales.data());
  dZero.copy_from_host(f.zeros.data());
  dA.copy_from_host(f.A.data());

  int errors = 0;
  auto got = run_g5_arm<8, 8>(
      "m8-B-ID", f, 1, dA.get(), dB.get(), dScale.get(), dZero.get(), &errors);
  if (got.empty()) return errors ? errors : 1;

  std::printf("[G5:B-IDPROBE] A=1 scale=1 zero=0; expert e has e q9 codes "
              "and K-e q8 codes per column; output bits must equal fp16(e)\n");
  int bad_slots = 0;
  int bitdiff = 0;
  int reported = 0;
  for (int slot = 0; slot < n_active; ++slot) {
    int const expert = ids[slot];
    int const row = f.row_offsets[expert];
    half_t const want{float(expert)};
    int row_diff = 0;
    half_t const first = got[std::size_t(row) * kN];
    bool row_uniform = true;
    for (int n = 0; n < kN; ++n) {
      half_t const value = got[std::size_t(row) * kN + n];
      row_diff += value.raw() != want.raw();
      row_uniform &= value.raw() == first.raw();
    }
    if (row_diff && reported++ < 16) {
      std::printf("  B-IDPROBE slot=%-3d expert=%-3d first="
                  "%9.3f/0x%04x want=%9.3f/0x%04x row=%s "
                  "bitdiff=%d/%d\n",
                  slot, expert, double(float(first)), unsigned(first.raw()),
                  double(float(want)), unsigned(want.raw()),
                  row_uniform ? "uniform" : "SPLIT", row_diff, kN);
    }
    bitdiff += row_diff;
    bad_slots += row_diff != 0;
  }
  errors += bitdiff;
  std::printf("[G5:B-IDPROBE] %s: slot-mismatches=%d/%d "
              "output-bitdiff=%d/%d\n",
              errors ? "FAIL" : "PASS", bad_slots, n_active,
              bitdiff, n_active * kN);
  return errors;
}

int main(int argc, char** argv) {
  bool const idprobe_only = argc == 2 && std::strcmp(argv[1], "--idprobe-only") == 0;
  bool const second_tile_only =
      argc == 2 && std::strcmp(argv[1], "--second-tile-only") == 0;
  bool const epilogue_topology_only =
      argc == 2 && std::strcmp(argv[1], "--epilogue-topology-only") == 0;
  if (argc != 1 && !idprobe_only && !second_tile_only &&
      !epilogue_topology_only) {
    std::fprintf(stderr,
                 "usage: %s [--idprobe-only|--second-tile-only|"
                 "--epilogue-topology-only]\n", argv[0]);
    return 2;
  }

  if (epilogue_topology_only) {
    std::printf("== [112:EPILOGUE-TOPOLOGY] exact grouped ptr-array "
                "coordinate tags ==\n");
    int const errors = run_g4_epilogue_topology();
    std::printf("== [112:EPILOGUE-TOPOLOGY] %s errors=%d "
                "control=TN32/WN32 subject=TN64/WN16 ==\n",
                errors ? "FAIL" : "PASS", errors);
    return errors ? 1 : 0;
  }

  if (idprobe_only) {
    std::printf("== [112:IDPROBE-ONLY] zero-plane + B-plane expert identity ==\n");
    std::vector<int> all(kE);
    for (int e = 0; e < kE; ++e) all[e] = e;
    int errors = 0;
    errors += run_g5_idprobe(kActiveIds, kActive);
    errors += run_g5_idprobe(all.data(), kE);
    errors += run_g5_b_idprobe(kActiveIds, kActive);
    errors += run_g5_b_idprobe(all.data(), kE);
    std::printf("== [112:IDPROBE-ONLY] %s: errors=%d "
                "(zero active=8/256; B active=8/256) ==\n",
                errors ? "FAIL" : "PASS", errors);
    return errors ? 1 : 0;
  }

  std::printf("== [112] ppu001 m8n16 collective G3/G4/G5 ==\n");

  auto q = make_codes();
  auto scales = make_scales();
  auto zeros = make_zeros(scales);
  auto dense_a = make_dense_a();

  std::size_t const logical_bbytes = kHarnessBBytes;
  // Four TK64 artifacts exactly fill one interleave-256 row.  Unlike the old
  // K=64 gate, the physical artifact and the packed StrideB now describe the
  // same 4096-byte domain; the static_assert above makes a future shortened K
  // fail at compile time instead of silently reading every fourth column.
  std::size_t const artifact_bbytes = kPlacedBBytes;
  using HarnessStrideB = typename M8Mainloop::StrideB;
  HarnessStrideB const harness_b_stride = cutlass::make_cute_packed_stride(
      HarnessStrideB{}, cute::make_shape(kN, kK, 1));
  auto const harness_b_layout = cute::make_layout(
      cute::make_shape(kN, kK, 1), harness_b_stride);
  std::size_t const harness_bbytes =
      std::size_t(cute::cosize(harness_b_layout)) * kBits / 8;
  if (harness_bbytes != artifact_bbytes) {
    std::printf("[offline] packed StrideB span=%zu bytes but F=1 interleave-256 "
                "artifact span=%zu bytes -- FAIL\n",
                harness_bbytes, artifact_bbytes);
    return 1;
  }
  std::vector<std::int8_t> b8(artifact_bbytes, 0), b16(artifact_bbytes, 0);
  xplane::place_derived<4, 8, 32, 64, 8, 32, 1>(
      b8.data(), q, kN, kK);
  xplane::place_derived<4, 16, 32, 64, 16, 32, 1>(
      b16.data(), q, kN, kK);
  int errors = 0;
  if (b8 != b16) {
    int diff = 0;
    for (std::size_t i = 0; i < artifact_bbytes; ++i) diff += b8[i] != b16[i];
    std::printf("[offline] m8/m16 B artifacts differ at %d/%zu bytes -- FAIL\n",
                diff, artifact_bbytes);
    return 1;
  }
  std::vector<std::uint8_t> recovered;
  xplane::recover_derived<4, 8, 32, 64, 8, 32, 1>(
      b8.data(), recovered, kN, kK);
  int recovery_bad = 0;
  for (std::size_t i = 0; i < q.size(); ++i) recovery_bad += recovered[i] != q[i];
  if (recovery_bad != 0) {
    std::printf("[offline] m8 artifact round trip bad=%d/%zu -- FAIL\n",
                recovery_bad, q.size());
    return 1;
  }
  std::printf("[offline] m8/m16 B artifacts byte-identical: %zu physical bytes "
              "(%zu logical); roundtrip=0/%zu\n",
              artifact_bbytes, logical_bbytes, q.size());

  cutlass::DeviceAllocation<int4_t> dB(artifact_bbytes);
  cutlass::DeviceAllocation<half_t> dScale(scales.size());
  cutlass::DeviceAllocation<half_t> dZero(zeros.size());
  cutlass::DeviceAllocation<half_t> dDenseA(dense_a.size());
  // DeviceAllocation allocates sub-byte wrapper storage with sizeof(T), but
  // copy_from_host counts sizeof_bits<T>.  This allocation is intentionally
  // byte-sized, so use an explicitly byte-sized copy; otherwise only half of
  // an int4 artifact is transferred and the second K group stays uninitialised.
  CUTLASS_PPU_CHECK(hggcMemcpy(
      dB.get(), b8.data(), artifact_bbytes, hggcMemcpyHostToDevice));
  dScale.copy_from_host(scales.data());
  dZero.copy_from_host(zeros.data());
  dDenseA.copy_from_host(dense_a.data());

  errors += run_g3(dB, dScale, dZero, q, scales, zeros);

  constexpr int Ms[] = {1, 2, 3, 7, 8, 9, 15, 16, 17};
  for (int M : Ms) {
    if (M > 8) {
      errors += run_g3_a_tag(M, dB, dScale, dZero, q, scales, zeros);
      errors += run_g4_epilogue_tag(M);
    }
    auto golden = golden_fp32(dense_a, M, q, scales, zeros);
    auto m16 = run_g4_arm<16, 16>(
        "m16", M, dDenseA.get(), dB.get(), dScale.get(), dZero.get());
    auto m8 = run_g4_arm<8, 8>(
        "m8", M, dDenseA.get(), dB.get(), dScale.get(), dZero.get());
    errors += m8.errors + m16.errors;
    errors += check_g4_values("m8", M, m8.logical, golden);
    errors += check_g4_values("m16", M, m16.logical, golden);
    errors += check_m8_m16(M, m8.logical, m16.logical);
    if (M > 8) {
      auto m8p = run_g4_arm<8, 8, true>(
          "m8p", M, dDenseA.get(), dB.get(), dScale.get(), dZero.get());
      errors += m8p.errors;
      errors += check_g4_values("m8p", M, m8p.logical, golden);
      int const persistent_diff = half_bitdiff(m8p.logical, m8.logical);
      std::printf("  G4 m8p-vs-m8 M=%d bitdiff=%d/%zu %s\n", M,
                  persistent_diff, m8.logical.size(),
                  persistent_diff ? "MISMATCH" : "MATCH");
      errors += persistent_diff;
    }
  }

  if (second_tile_only) {
    std::printf("== [112:SECOND-TILE] %s: errors=%d "
                "M=9/15/16/17 seams=mainloop-A+ptr-array-epilogue+"
                "nonpersistent+persistent ==\n",
                errors ? "FAIL" : "PASS", errors);
    return errors ? 1 : 0;
  }

  // G5 runs at the two row counts that separate the m8 family from its control: one row is the decode
  // case the atom exists for, eight is the last M an m8 tile holds without a second tile.
  // THREE CONFIGURATIONS, because a brand-new gate's first red is more often the gate.  The negative
  // controls prove the experts are distinguishable; they do not prove that this fixture's per-expert
  // W/S/Z placement agrees with the stride the kernel derives from the group index.  These separate
  // the two candidate causes without touching the kernel:
  //   LOW     contiguous ids 0..7, distinct data  -- isolates "id >= 128" from "non-contiguous"
  //   UNIFORM the real ids, every active expert given IDENTICAL data (salt 1) -- if this passes, the
  //           addressing is right and the defect is expert-DEPENDENT; if it fails, my layout is wrong
  //   REAL    the ragged route this gate exists for
  constexpr int kLowIds[kActive] = {0, 1, 2, 3, 4, 5, 6, 7};
  // NO ZERO-ROW GROUPS.  248 groups with group_M == 0 is the one structural thing G4 (L=1) never
  // had, and it is where a slot->expert map would break.  If DENSE passes while the others fail,
  // the trigger is the zero-row group, not the expert index itself.
  for (int rows : {1, 8}) {
    errors += run_g5("LOW",     rows, kLowIds,    kActive, false);
    errors += run_g5("UNIFORM", rows, kActiveIds, kActive, true);
    errors += run_g5("REAL",    rows, kActiveIds, kActive, false);
    if (rows == 1) {
      std::vector<int> all(kE);
      for (int e = 0; e < kE; ++e) all[e] = e;
      errors += run_g5("DENSE", rows, all.data(), kE, false);
      errors += run_g5_idprobe(kActiveIds, kActive);
      errors += run_g5_idprobe(all.data(), kE);
      errors += run_g5_b_idprobe(kActiveIds, kActive);
      errors += run_g5_b_idprobe(all.data(), kE);
    }
  }

  std::printf("== [112] %s: errors=%d (G3/G4/G5) ==\n",
              errors ? "FAIL" : "PASS", errors);
  return errors ? 1 : 0;
}
