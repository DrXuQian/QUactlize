// Grouped mixed-input Stream-K phase-2 minimum-stripe gate [PPU box only].
//
// This target is deliberately isolated from production dispatch and the full
// tactic sweep.  Phase 1 proved the q-flattened scheduler/lock seam at the
// vendor-compatible Min=8 default.  This gate explicitly selects Min=2: the
// decode Kt=8 arm must now create four peers per output tile, while the host
// oracle retains Min=8 as the no-split negative baseline.
//
// Build: TARGET=test_moe_grouped_streamk ./build.sh
// Run:   timeout 180s ./build/.../test_moe_grouped_streamk
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "moe_grouped_ppu.cuh"
#include "moe_grouped_streamk_ppu.cuh"
#include "moe_router_fixture.hpp"
#include "xplane_offline.hpp"

// ArtifactTileK=64 with TacticTileK=256 selects the folded schedule even at
// F=1; the consumer must name its optional collective explicitly.
#include "quactlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_fold.hpp"

namespace {

using half_t = cutlass::half_t;
using int4_t = cutlass::int4b_t;
using GS = moe_grouped_streamk_ppu::GroupShape;
using DStride = moe_grouped_streamk_ppu::DStride;
using QM = moe_grouped_streamk_ppu::QuantMode;

constexpr int kTM = 16;
constexpr int kTN = 256;
constexpr int kWM = 16;
constexpr int kWN = 64;
constexpr int kStages = 3;
constexpr int kArtifactTileK = 64;
constexpr int kGs = 32;
constexpr uint32_t kMinSkIters = 2;
constexpr float kAlpha = 0.75f;
constexpr float kBeta = 0.5f;

template <int TK>
using Op = moe_grouped_streamk_ppu::Operation<
    QM::FinegrainedScaleZero,
    ppu_group_schedule::FinegrainedSchedule<kGs>,
    cute::Shape<cute::Int<kTM>, cute::Int<kTN>, cute::Int<TK>>,
    cute::Shape<cute::Int<kTN>, cute::Int<TK / kGs>>,
    cute::Shape<cute::Int<kWM>, cute::Int<kWN>, cute::Int<TK>>,
    kStages, true, int4_t, void, kArtifactTileK, kMinSkIters>;
static_assert(Op<64>::ExpectedGroupSize == kGs &&
              Op<256>::ExpectedGroupSize == kGs,
              "CPU golden and both compiled schedules must share gs");
static_assert(Op<64>::TileSchedulerParams::min_iters_per_sk_unit_ == 2 &&
              Op<256>::TileSchedulerParams::min_iters_per_sk_unit_ == 2,
              "phase 2 grouped Stream-K must explicitly select Min=2");

// Current decode champion.  Its 64-thread CTA was excluded only because the
// vendor fixup cohort was hard-wired to 128; keep this as a separate
// nonuniform-stripe device arm so the exact uniform oracle below remains an
// independent proof rather than being weakened to admit this case.
constexpr int kDecodeTM = 16;
constexpr int kDecodeTN = 32;
constexpr int kDecodeTK = 256;
constexpr int kDecodeWM = 16;
constexpr int kDecodeWN = 16;
using Decode64Op = moe_grouped_streamk_ppu::Operation<
    QM::FinegrainedScaleZero,
    ppu_group_schedule::FinegrainedSchedule<kGs>,
    cute::Shape<cute::Int<kDecodeTM>, cute::Int<kDecodeTN>,
                cute::Int<kDecodeTK>>,
    cute::Shape<cute::Int<kDecodeTN>, cute::Int<kDecodeTK / kGs>>,
    cute::Shape<cute::Int<kDecodeWM>, cute::Int<kDecodeWN>,
                cute::Int<kDecodeTK>>,
    kStages, true, int4_t, void, kArtifactTileK, 2u>;
static_assert(Decode64Op::Kernel::MaxThreadsPerBlock == 64);
static_assert(Decode64Op::Kernel::TileScheduler::FixupThreadCount == 64);
static_assert(Decode64Op::TileSchedulerParams::min_iters_per_sk_unit_ == 2);

struct Fixture {
  char const* name = nullptr;
  int experts = 0;
  int n = 0;
  int k = 0;
  std::vector<int> me;
  std::vector<int> offsets;
  std::vector<int> active;
  int total = 0;
  int mmax = 0;
  std::vector<half_t> a;
  std::vector<half_t> c;
  std::vector<half_t> scales;
  std::vector<half_t> zeros;
  std::vector<std::vector<uint8_t>> codes;
  std::vector<int8_t> packed;
};

half_t dequant_one(uint8_t q, half_t scale, half_t zero) {
  half_t const centered(float(int(q) - 8));
  half_t const scaled(float(centered) * float(scale));
  return half_t(float(scaled) + float(zero));
}

Fixture make_fixture(char const* name, std::vector<int> me, int n, int k) {
  Fixture f;
  f.name = name;
  f.experts = int(me.size());
  f.n = n;
  f.k = k;
  f.me = std::move(me);
  f.offsets.resize(f.experts);
  f.codes.resize(f.experts);
  for (int e = 0; e < f.experts; ++e) {
    f.offsets[e] = f.total;
    f.total += f.me[e];
    f.mmax = std::max(f.mmax, f.me[e]);
    if (f.me[e] > 0) f.active.push_back(e);
  }

  int const scale_k = k / kGs;
  f.a.resize(size_t(f.total) * k);
  f.c.resize(size_t(f.total) * n);
  f.scales.resize(size_t(f.experts) * scale_k * n);
  f.zeros.resize(size_t(f.experts) * scale_k * n);
  size_t const bytes_per_expert = size_t(k) * n * 4 / 8;
  f.packed.assign(size_t(f.experts) * bytes_per_expert, 0);

  for (int e = 0; e < f.experts; ++e) {
    for (int g = 0; g < scale_k; ++g) {
      for (int col = 0; col < n; ++col) {
        size_t const idx = (size_t(e) * scale_k + g) * n + col;
        float const s = float(1 + ((e + 3 * g + col) & 3)) / 16.0f;
        f.scales[idx] = half_t(s);
        f.zeros[idx] = half_t(float(((e + g + 2 * col) % 3) - 1) * s);
      }
    }
    if (f.me[e] == 0) continue;
    f.codes[e].resize(size_t(k) * n);
    for (int kk = 0; kk < k; ++kk) {
      for (int col = 0; col < n; ++col) {
        f.codes[e][size_t(kk) * n + col] =
            uint8_t((7 * e + 5 * kk + 3 * col + (kk ^ col)) & 15);
      }
    }
    // Pack the canonical ArtifactTileK=64 placement.  Both consumer TileK
    // arms below must read this exact same resident object.
    xplane::place_derived<4, kTM, kTN, 64, kWM, kWN, 1,
                          kArtifactTileK>(
        f.packed.data() + size_t(e) * bytes_per_expert, f.codes[e], n, k);
    for (int r = 0; r < f.me[e]; ++r) {
      size_t const row = size_t(f.offsets[e] + r);
      for (int kk = 0; kk < k; ++kk) {
        f.a[row * k + kk] = half_t(
            float(((11 * e + 7 * r + 3 * kk + (r ^ kk)) % 7) - 3) /
            8.0f);
      }
      for (int col = 0; col < n; ++col) {
        f.c[row * n + col] = half_t(
            float(((5 * e + 3 * r + col) % 9) - 4) / 8.0f);
      }
    }
  }
  return f;
}

std::vector<half_t> golden(Fixture const& f, float alpha, float beta) {
  std::vector<half_t> out(size_t(f.total) * f.n);
  int const scale_k = f.k / kGs;
  (void)scale_k;
  for (int e : f.active) {
    for (int r = 0; r < f.me[e]; ++r) {
      size_t const row = size_t(f.offsets[e] + r);
      for (int col = 0; col < f.n; ++col) {
        float accum = 0.0f;
        for (int kk = 0; kk < f.k; ++kk) {
          size_t const meta =
              (size_t(e) * (f.k / kGs) + kk / kGs) * f.n + col;
          half_t const w = dequant_one(
              f.codes[e][size_t(kk) * f.n + col],
              f.scales[meta], f.zeros[meta]);
          accum += float(f.a[row * f.k + kk]) * float(w);
        }
        out[row * f.n + col] =
            half_t(alpha * accum + beta * float(f.c[row * f.n + col]));
      }
    }
  }
  return out;
}

struct DeviceFixture {
  cutlass::DeviceAllocation<half_t> a;
  cutlass::DeviceAllocation<int4_t> b;
  cutlass::DeviceAllocation<half_t> scales;
  cutlass::DeviceAllocation<half_t> zeros;
  cutlass::DeviceAllocation<half_t> c;
  cutlass::DeviceAllocation<half_t> d;
  cutlass::DeviceAllocation<GS> shapes;
  cutlass::DeviceAllocation<int> row_offsets;
  cutlass::DeviceAllocation<int> group_m;
  cutlass::DeviceAllocation<half_t const*> ptr_c;
  cutlass::DeviceAllocation<half_t*> ptr_d;
  cutlass::DeviceAllocation<DStride> strides;
  std::vector<GS> host_shapes;

  explicit DeviceFixture(Fixture const& f)
      : a(size_t(f.total) * f.k),
        b(f.packed.size()),
        scales(f.scales.size()),
        zeros(f.zeros.size()),
        c(f.c.size()),
        d(f.c.size()),
        shapes(f.experts),
        row_offsets(f.experts),
        group_m(f.experts),
        ptr_c(f.experts),
        ptr_d(f.experts),
        strides(f.experts),
        host_shapes(f.experts) {
    a.copy_from_host(f.a.data());
    CUTLASS_PPU_CHECK(hggcMemcpy(b.get(), f.packed.data(), f.packed.size(),
                                hggcMemcpyHostToDevice));
    scales.copy_from_host(f.scales.data());
    zeros.copy_from_host(f.zeros.data());
    c.copy_from_host(f.c.data());
    row_offsets.copy_from_host(f.offsets.data());
    group_m.copy_from_host(f.me.data());

    std::vector<half_t const*> hc(f.experts);
    std::vector<half_t*> hd(f.experts);
    std::vector<DStride> hs(f.experts);
    for (int e = 0; e < f.experts; ++e) {
      host_shapes[e] = cute::make_shape(f.me[e], f.n, f.k);
      hc[e] = c.get() + size_t(f.offsets[e]) * f.n;
      hd[e] = d.get() + size_t(f.offsets[e]) * f.n;
      hs[e] = cutlass::make_cute_packed_stride(
          DStride{}, cute::make_shape(f.me[e], f.n, 1));
    }
    shapes.copy_from_host(host_shapes.data());
    ptr_c.copy_from_host(hc.data());
    ptr_d.copy_from_host(hd.data());
    strides.copy_from_host(hs.data());
  }
};

class EventBatch {
 public:
  struct Pair { hggcEvent_t start{}, stop{}; };
  explicit EventBatch(size_t n) : pairs_(n) {
    for (auto& p : pairs_) {
      CUTLASS_PPU_CHECK(hggcEventCreate(&p.start));
      CUTLASS_PPU_CHECK(hggcEventCreate(&p.stop));
    }
  }
  ~EventBatch() {
    for (auto& p : pairs_) {
      hggcEventDestroy(p.start);
      hggcEventDestroy(p.stop);
    }
  }
  Pair& operator[](size_t i) { return pairs_.at(i); }

 private:
  std::vector<Pair> pairs_;
};

int verify_output(Fixture const& f, DeviceFixture& d, float alpha, float beta,
                  char const* label) {
  std::vector<half_t> got(size_t(f.total) * f.n);
  d.d.copy_to_host(got.data());
  std::vector<half_t> want = golden(f, alpha, beta);
  int bad = 0, bitdiff = 0, nonfinite = 0, poison = 0;
  double absmax = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    float const gv = float(got[i]);
    if (!std::isfinite(gv)) ++nonfinite;
    if (got[i].raw() == uint16_t(0x7f7f)) ++poison;
    absmax = std::max(absmax, std::abs(double(gv)));
    if (got[i].raw() != want[i].raw()) {
      if (bad < 4) {
        std::printf("    %s i=%zu got=%g(0x%04x) want=%g(0x%04x)\n",
                    label, i, double(gv), unsigned(got[i].raw()),
                    double(float(want[i])), unsigned(want[i].raw()));
      }
      ++bad;
      ++bitdiff;
    }
  }
  std::printf("[%s numeric] outputs=%zu bad=%d bitdiff=%d nonfinite=%d "
              "poison_left=%d |got|max=%.6g %s\n",
              label, got.size(), bad, bitdiff, nonfinite, poison, absmax,
              bad == 0 && nonfinite == 0 && poison == 0 && absmax > 0.0
                  ? "PASS" : "FAIL");
  return (bad == 0 && nonfinite == 0 && poison == 0 && absmax > 0.0) ? 0 : 1;
}

template <class Params>
bool host_policy_line(char const* min_name, uint64_t q, uint32_t kt,
                      uint64_t workers, uint32_t expected_heuristic_tiles,
                      uint32_t expected_forced_tiles,
                      uint64_t expected_forced_units) {
  cutlass::gemm::GemmCoord cluster(1, 1, 1);
  uint32_t const ht = Params::get_num_sk_tiles(
      q, workers, 1, kt, Params::DecompositionMode::Heuristic);
  uint32_t const ft = Params::get_num_sk_tiles(
      q, workers, 1, kt, Params::DecompositionMode::StreamK);
  uint64_t const fu = Params::get_num_sk_units(cluster, workers, ft, kt);
  std::printf("[grouped streamk policy] %s Q=%llu Kt=%u W=%llu "
              "heuristic_tiles=%u forced_tiles=%u forced_units=%llu %s\n",
              min_name, static_cast<unsigned long long>(q), kt,
              static_cast<unsigned long long>(workers), ht, ft,
              static_cast<unsigned long long>(fu),
              ht == expected_heuristic_tiles && ft == expected_forced_tiles &&
                      fu == expected_forced_units
                  ? "POLICY-PASS" : "POLICY-FAIL");
  return ht == expected_heuristic_tiles && ft == expected_forced_tiles &&
         fu == expected_forced_units;
}

struct ArmResult {
  int errors = 0;
  moe_grouped_streamk_ppu::Plan plan{};
  int split_tiles = 0;
  uint64_t peer_excess = 0;
  uint32_t fixup_work_items = 0;
  uint32_t epilogue_work_items = 0;
  bool expected_supported = false;
  uint64_t expected_peer_excess = 0;
  uint64_t logical_fixup_elements = 0;
  uint64_t expected_logical_fixup_elements = 0;
};

struct Phase2Expectation {
  bool supported = false;
  uint32_t sk_tiles = 0;
  uint64_t sk_units = 0;
  int split_tiles = 0;
  uint64_t peer_excess = 0;
  uint32_t fixup_work_items = 0;
  uint32_t stripe_k_tiles = 0;
  uint32_t peers_per_tile = 0;
};

// The grouped scheduler flattens q with m-tile as the fast coordinate inside
// each expert, then n-tile.  Weight each q's peer excess by its *valid* output
// rectangle.  This is the logical FP32 workspace footprint of the predicated
// fixup path; allocation and cache-line/DRAM traffic remain full-tile concerns.
uint64_t valid_fixup_elements(Fixture const& f,
                              std::vector<uint32_t> const& peer_count,
                              int tile_m, int tile_n, bool* exact = nullptr) {
  uint64_t elements = 0;
  size_t q = 0;
  bool ok = tile_m > 0 && tile_n > 0;
  for (int e = 0; e < f.experts && ok; ++e) {
    int const mt = (f.me[e] + tile_m - 1) / tile_m;
    int const nt = (f.n + tile_n - 1) / tile_n;
    for (int n_idx = 0; n_idx < nt; ++n_idx) {
      int const valid_n = std::min(tile_n, f.n - n_idx * tile_n);
      for (int m_idx = 0; m_idx < mt; ++m_idx, ++q) {
        if (q >= peer_count.size()) {
          ok = false;
          break;
        }
        int const valid_m = std::min(tile_m, f.me[e] - m_idx * tile_m);
        uint64_t const excess = peer_count[q] > 0 ? peer_count[q] - 1 : 0;
        elements += excess * uint64_t(valid_m) * uint64_t(valid_n);
      }
    }
  }
  ok = ok && q == peer_count.size();
  if (exact != nullptr) *exact = ok;
  return ok ? elements : 0;
}

// Independent arithmetic oracle for this gate's single-cluster, sub-wave
// fixtures.  It intentionally does not call the vendor Params methods used by
// lowering.  Forced Stream-K owns every q tile, and Min=2 caps the worker
// population at floor(Q*Kt/2).  This first phase-2 device gate deliberately
// accepts only an exactly uniform stripe: a non-divisor worker cap is reported
// as unsupported instead of pretending that CTA count equals fixup work-item
// count when a stripe crosses an output-tile boundary.
Phase2Expectation phase2_expectation(int q, int kt, int workers) {
  Phase2Expectation out;
  if (q <= 0 || kt < int(kMinSkIters) || workers < 2 * q) return out;
  uint64_t const total_k_tiles = uint64_t(q) * uint64_t(kt);
  uint64_t const units_at_min_stripe = total_k_tiles / kMinSkIters;
  uint64_t const units = std::min<uint64_t>(uint64_t(workers),
                                            units_at_min_stripe);
  if (units < 2ull * uint64_t(q) || units > uint64_t(UINT32_MAX)) return out;
  if (total_k_tiles % units != 0) return out;
  uint64_t const stripe_k_tiles = total_k_tiles / units;
  if (stripe_k_tiles == 0 || uint64_t(kt) % stripe_k_tiles != 0) return out;
  uint64_t const peers_per_tile = uint64_t(kt) / stripe_k_tiles;
  if (peers_per_tile < 2 || peers_per_tile > uint64_t(UINT32_MAX)) return out;
  out.supported = true;
  out.sk_tiles = uint32_t(q);
  out.sk_units = units;
  out.split_tiles = q;
  out.peer_excess = uint64_t(q) * (peers_per_tile - 1);
  out.fixup_work_items = uint32_t(units);
  out.stripe_k_tiles = uint32_t(stripe_k_tiles);
  out.peers_per_tile = uint32_t(peers_per_tile);
  return out;
}

template <int TK>
ArmResult run_streamk_arm(Fixture const& f, DeviceFixture& d,
                          int device_id, int real_cu, bool time_arm) {
  using O = Op<TK>;
  using Gemm = typename O::Gemm;
  using Kernel = typename O::Kernel;
  ArmResult result;

  auto args = O::make_arguments(
      d.a.get(), d.b.get(), d.scales.get(), d.zeros.get(),
      d.ptr_c.get(), d.strides.get(), d.ptr_d.get(), d.strides.get(),
      f.mmax, f.n, f.k, f.experts, d.shapes.get(),
      d.host_shapes.data(), d.row_offsets.get(), kAlpha, kBeta);
  int const ctas_per_cu = Gemm::maximum_active_blocks();
  if (real_cu <= 0 || ctas_per_cu <= 0) {
    std::printf("[grouped streamk] invalid occupancy cu=%d ctas/cu=%d\n",
                real_cu, ctas_per_cu);
    result.errors = 1;
    return result;
  }
  O::configure_runtime(args, device_id, real_cu, ctas_per_cu);
  if (Gemm::can_implement(args) != cutlass::Status::kSuccess) {
    std::printf("[grouped streamk] TK=%d can_implement failed\n", TK);
    result.errors = 1;
    return result;
  }
  size_t const workspace_bytes = Gemm::get_workspace_size(args);
  cutlass::DeviceAllocation<uint8_t> workspace(workspace_bytes);
  result.plan = O::inspect(args, workspace.get());
  Phase2Expectation const expected = phase2_expectation(
      result.plan.q, result.plan.kt, result.plan.workers);
  result.expected_supported = expected.supported;
  result.expected_peer_excess = expected.peer_excess;
  dim3 const grid = Gemm::get_grid_shape(args, workspace.get());
  uint64_t const physical = uint64_t(grid.x) * grid.y * grid.z;
  size_t const expected_scheduler_workspace =
      size_t(result.plan.q) * kTM * kTN * sizeof(float) + 128;
  bool const plan_ok =
      result.plan.q > 0 && result.plan.kt == f.k / TK &&
      result.plan.workers == real_cu * ctas_per_cu &&
      physical == uint64_t(result.plan.workers) &&
      result.plan.splits == 1 && result.plan.separate_reduction_units == 0 &&
      result.plan.scheduler_workspace_bytes > 0 &&
      result.plan.scheduler_workspace_bytes == expected_scheduler_workspace &&
      result.plan.scheduler_barrier_bytes == 128 &&
      result.plan.q == int(result.plan.units_per_problem -
                           result.plan.sk_units + result.plan.sk_tiles);
  std::printf("[grouped streamk decomposition] fixture=%s "
              "tactic=i4_%dx%dx%d_w%dx%d_s%d requested=StreamK "
              "Min=%u actual=%s Q=%d Kt=%d W=%d scheduler_workers=%d "
              "sk_tiles=%u sk_units=%llu workspace=%zu scheduler/reset=%zu/%zu "
              "grid=(%u,%u,%u) %s\n",
              f.name, kTM, kTN, TK, kWM, kWN, kStages, kMinSkIters,
              result.plan.sk_tiles ? "StreamK" : "DataParallel",
              result.plan.q, result.plan.kt, real_cu * ctas_per_cu,
              result.plan.workers, result.plan.sk_tiles,
              static_cast<unsigned long long>(result.plan.sk_units),
              workspace_bytes, result.plan.scheduler_workspace_bytes,
              result.plan.scheduler_barrier_bytes, grid.x, grid.y, grid.z,
              plan_ok ? "PLAN-PASS" : "PLAN-FAIL");
  if (!plan_ok) ++result.errors;
  if (!expected.supported) {
    std::printf("[grouped streamk phase2 oracle] fixture=%s TK=%d Min=%u "
                "Q=%d Kt=%d W=%d ORACLE-UNSUPPORTED/FAIL\n",
                f.name, TK, kMinSkIters, result.plan.q, result.plan.kt,
                result.plan.workers);
    ++result.errors;
    return result;
  }
  std::printf("[grouped streamk phase2 oracle] fixture=%s TK=%d Min=%u "
              "stripe_k_tiles=%u peers_per_tile=%u ORACLE-PASS\n",
              f.name, TK, kMinSkIters, expected.stripe_k_tiles,
              expected.peers_per_tile);

  cutlass::DeviceAllocation<uint32_t> peer(result.plan.q);
  cutlass::DeviceAllocation<uint32_t> visits(
      size_t(result.plan.q) * result.plan.kt);
  cutlass::DeviceAllocation<uint32_t> totals(6);
  CUTLASS_PPU_CHECK(hggcMemset(peer.get(), 0,
                              sizeof(uint32_t) * result.plan.q));
  CUTLASS_PPU_CHECK(hggcMemset(visits.get(), 0,
                              sizeof(uint32_t) * size_t(result.plan.q) *
                                  result.plan.kt));
  CUTLASS_PPU_CHECK(hggcMemset(totals.get(), 0, 6 * sizeof(uint32_t)));
  typename O::Census census{peer.get(), visits.get(), totals.get(),
                            uint32_t(result.plan.q),
                            uint64_t(result.plan.q) * result.plan.kt};
  O::configure_runtime(args, device_id, real_cu, ctas_per_cu, census);
  Gemm gemm;
  CUTLASS_PPU_CHECK(hggcMemset(d.d.get(), 0x7f,
                              sizeof(half_t) * size_t(f.total) * f.n));
  if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess ||
      gemm.run() != cutlass::Status::kSuccess) {
    std::printf("[grouped streamk] TK=%d census launch failed\n", TK);
    ++result.errors;
    return result;
  }
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());

  std::vector<uint32_t> hp(result.plan.q), hv(size_t(result.plan.q) * result.plan.kt);
  uint32_t ht[6] = {};
  peer.copy_to_host(hp.data());
  visits.copy_to_host(hv.data());
  totals.copy_to_host(ht);
  int split_tiles = 0;
  uint64_t peer_excess = 0;
  int holes = 0;
  for (uint32_t p : hp) {
    split_tiles += p > 1;
    peer_excess += p > 0 ? p - 1 : 0;
    holes += p == 0;
  }
  int missing_k = 0, duplicate_k = 0;
  for (uint32_t v : hv) {
    missing_k += v == 0;
    duplicate_k += v > 1;
  }
  bool const census_identity =
      ht[2] == 0 && ht[1] == uint32_t(result.plan.q) &&
      split_tiles == int(ht[3]) && peer_excess == uint64_t(ht[0] - ht[3]) &&
      holes == 0 && missing_k == 0 && duplicate_k == 0 && ht[4] == 0 &&
      ht[5] == 0;
  result.split_tiles = split_tiles;
  result.peer_excess = peer_excess;
  result.fixup_work_items = ht[0];
  result.epilogue_work_items = ht[1];
  bool logical_exact = false;
  result.logical_fixup_elements =
      valid_fixup_elements(f, hp, kTM, kTN, &logical_exact);
  std::vector<uint32_t> expected_peers(
      size_t(result.plan.q), expected.peers_per_tile);
  bool expected_logical_exact = false;
  result.expected_logical_fixup_elements = valid_fixup_elements(
      f, expected_peers, kTM, kTN, &expected_logical_exact);
  if (!logical_exact || !expected_logical_exact) ++result.errors;
  std::printf("[grouped streamk census] fixture=%s TK=%d Min=%u Q=%d Kt=%d "
              "split_tiles=%d peer_excess=%llu requires_fixup=%u "
              "fixup_work_items=%u epilogue=%u separate=%u "
              "fixup_final=%u q_oob=%u "
              "empty_decode=%u holes=%d missing_k=%d duplicate_k=%d %s\n",
              f.name, TK, kMinSkIters, result.plan.q, result.plan.kt, split_tiles,
              static_cast<unsigned long long>(peer_excess), unsigned(ht[0] != 0),
              ht[0], ht[1], ht[2], ht[3], ht[4], ht[5], holes, missing_k,
              duplicate_k,
              census_identity ? "CENSUS-PASS" : "CENSUS-FAIL");
  if (!census_identity) ++result.errors;

  int const expected_q = f.name == std::string("S068") ? 16 : 6;
  bool const exact = expected.supported && result.plan.q == expected_q &&
                     result.plan.sk_tiles == expected.sk_tiles &&
                     result.plan.sk_units == expected.sk_units &&
                     split_tiles == expected.split_tiles &&
                     peer_excess == expected.peer_excess &&
                     ht[0] == expected.fixup_work_items && logical_exact &&
                     expected_logical_exact &&
                     result.logical_fixup_elements ==
                         result.expected_logical_fixup_elements;
  std::printf("[grouped streamk exact] fixture=%s TK=%d Min=%u expected "
              "Q/units/split/excess/fixup=%d/%llu/%d/%llu/%u %s\n",
              f.name, TK, kMinSkIters, expected_q,
              static_cast<unsigned long long>(expected.sk_units),
              expected.split_tiles,
              static_cast<unsigned long long>(expected.peer_excess),
              expected.fixup_work_items, exact ? "PASS" : "FAIL");
  if (!exact) ++result.errors;
  std::printf("[grouped streamk result] fixture=%s TK=%d Min=%u Q=%d Kt=%d W=%d "
              "sk_tiles=%u sk_units=%llu split_tiles=%d peer_excess=%llu "
              "requires_fixup=%u fixup_work_items=%u\n",
              f.name, TK, kMinSkIters, result.plan.q, result.plan.kt,
              result.plan.workers,
              result.plan.sk_tiles,
              static_cast<unsigned long long>(result.plan.sk_units),
              split_tiles, static_cast<unsigned long long>(peer_excess),
              unsigned(ht[0] != 0), ht[0]);

  // Relower without census before correctness/performance. Normal initialize
  // reinstalls P+shape once; the explicitly named timed reset below then
  // clears only the vendor barrier tail without touching scratch or P+shape.
  O::configure_runtime(args, device_id, real_cu, ctas_per_cu);
  CUTLASS_PPU_CHECK(hggcMemset(d.d.get(), 0x7f,
                              sizeof(half_t) * size_t(f.total) * f.n));
  if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess ||
      gemm.run() != cutlass::Status::kSuccess) {
    std::printf("[grouped streamk] TK=%d clean launch failed\n", TK);
    ++result.errors;
    return result;
  }
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  result.errors += verify_output(
      f, d, kAlpha, kBeta,
      TK == 256 ? "streamk-min2-TK256" : "streamk-min2-TK64");

  if (time_arm) {
    constexpr int kTimed = 20;
    EventBatch events(kTimed + 1);
    // Instrumented warmup owns its own event pair.
    CUTLASS_CHECK(Kernel::reset_scheduler_workspace_after_prefix_install(
        gemm.params(), workspace.get(), nullptr));
    CUTLASS_PPU_CHECK(hggcEventRecord(events[0].start, nullptr));
    CUTLASS_CHECK(gemm.run());
    CUTLASS_PPU_CHECK(hggcEventRecord(events[0].stop, nullptr));
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());

    auto const wall_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kTimed; ++i) {
      // barrier-tail reset is on the same stream before event start;
      // this reset-only verb does not touch the installed P/shape mirror.
      CUTLASS_CHECK(Kernel::reset_scheduler_workspace_after_prefix_install(
          gemm.params(), workspace.get(), nullptr));
      CUTLASS_PPU_CHECK(hggcEventRecord(events[size_t(i) + 1].start, nullptr));
      CUTLASS_CHECK(gemm.run());
      CUTLASS_PPU_CHECK(hggcEventRecord(events[size_t(i) + 1].stop, nullptr));
    }
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    auto const wall_stop = std::chrono::high_resolution_clock::now();
    std::vector<double> us(kTimed);
    for (int i = 0; i < kTimed; ++i) {
      float ms = 0.0f;
      CUTLASS_PPU_CHECK(hggcEventElapsedTime(
          &ms, events[size_t(i) + 1].start, events[size_t(i) + 1].stop));
      us[i] = double(ms) * 1000.0;
    }
    double const mean = std::accumulate(us.begin(), us.end(), 0.0) / us.size();
    std::sort(us.begin(), us.end());
    double const median = 0.5 * (us[9] + us[10]);
    double const spread = mean > 0 ? 100.0 * (us.back() - us.front()) / mean : 0.0;
    double const wall_us =
        std::chrono::duration<double, std::micro>(wall_stop - wall_start).count() /
        kTimed;
    bool const timing_ok =
        std::isfinite(us.front()) && std::isfinite(median) &&
        std::isfinite(mean) && std::isfinite(us.back()) &&
        std::isfinite(wall_us) && us.front() > 0.0 && median > 0.0 &&
        mean > 0.0 && us.back() > 0.0 && wall_us > 0.0;
    std::printf("[grouped streamk kernel-span-upper] fixture=%s TK=%d Min=%u n=20 "
                "median=%.3f us mean=%.3f us min=%.3f us max=%.3f us "
                "spread=(max-min)/mean=%.2f%% wall=%.3f us "
                "distinct-event-pairs=20 warmup-event-pairs=1 "
                "barrier-reset-before-start=%zuB prefix-shape-copy-before-timing=1 "
                "census-disabled=1 includes-launch-idle=1 %s\n",
                f.name, TK, kMinSkIters, median, mean, us.front(), us.back(), spread,
                wall_us, result.plan.scheduler_barrier_bytes,
                timing_ok ? "TIMING-PASS" : "TIMING-FAIL");
    if (!timing_ok) ++result.errors;
  }
  return result;
}

// The decode champion produces a genuinely nonuniform Stream-K partition:
// Q*Kt=1024 K tiles over 432 units.  The existing oracle above deliberately
// rejects that shape rather than inventing a uniform peer count.  This arm
// instead proves what the device can observe exactly (all K tiles visited
// once, one final epilogue per q, no bad q/expert decode), then performs a
// census-free launch against the independent CPU golden.  Per-tile traffic is
// left N/A until an independent nonuniform peer oracle exists.
int run_decode64_nonuniform(Fixture const& f, DeviceFixture& d,
                            int device_id, int real_cu) {
  using O = Decode64Op;
  using Gemm = typename O::Gemm;
  using Kernel = typename O::Kernel;
  int errors = 0;

  auto args = O::make_arguments(
      d.a.get(), d.b.get(), d.scales.get(), d.zeros.get(),
      d.ptr_c.get(), d.strides.get(), d.ptr_d.get(), d.strides.get(),
      f.mmax, f.n, f.k, f.experts, d.shapes.get(),
      d.host_shapes.data(), d.row_offsets.get(), kAlpha, kBeta);
  int const ctas_per_cu = Gemm::maximum_active_blocks();
  if (real_cu <= 0 || ctas_per_cu <= 0) {
    std::printf("[grouped streamk decode64] invalid occupancy cu=%d ctas/cu=%d\n",
                real_cu, ctas_per_cu);
    return 1;
  }
  O::configure_runtime(args, device_id, real_cu, ctas_per_cu);
  if (Gemm::can_implement(args) != cutlass::Status::kSuccess) {
    std::printf("[grouped streamk decode64] can_implement failed\n");
    return 1;
  }

  size_t const workspace_bytes = Gemm::get_workspace_size(args);
  cutlass::DeviceAllocation<uint8_t> workspace(workspace_bytes);
  auto const plan = O::inspect(args, workspace.get());
  dim3 const grid = Gemm::get_grid_shape(args, workspace.get());
  uint64_t const physical = uint64_t(grid.x) * grid.y * grid.z;
  constexpr int kExpectedQ = 128;
  constexpr int kExpectedKt = 8;
  constexpr int kExpectedWorkers = 432;
  constexpr size_t kReductionBytes =
      size_t(kExpectedQ) * kDecodeTM * kDecodeTN * sizeof(float);
  constexpr size_t kBarrierBytes = 512;
  constexpr size_t kSchedulerBytes = kReductionBytes + kBarrierBytes;
  Phase2Expectation const uniform = phase2_expectation(
      plan.q, plan.kt, plan.workers);
  bool const plan_ok =
      plan.q == kExpectedQ && plan.kt == kExpectedKt &&
      plan.workers == real_cu * ctas_per_cu &&
      plan.workers == kExpectedWorkers && physical == uint64_t(plan.workers) &&
      plan.sk_tiles == kExpectedQ && plan.sk_units == kExpectedWorkers &&
      plan.splits == 1 && plan.separate_reduction_units == 0 &&
      plan.scheduler_workspace_bytes == kSchedulerBytes &&
      plan.scheduler_barrier_bytes == kBarrierBytes &&
      plan.q == int(plan.units_per_problem - plan.sk_units + plan.sk_tiles) &&
      !uniform.supported;
  std::printf(
      "[grouped streamk decode64 decomposition] fixture=%s "
      "tactic=i4_%dx%dx%d_w%dx%d_s%d Min=%u threads=%u cohort=%u "
      "Q=%d Kt=%d W=%d sk_tiles=%u sk_units=%llu "
      "scheduler/reset=%zu/%zu grid=(%u,%u,%u) "
      "uniform_oracle_supported=%u %s\n",
      f.name, kDecodeTM, kDecodeTN, kDecodeTK, kDecodeWM, kDecodeWN,
      kStages, kMinSkIters, Kernel::MaxThreadsPerBlock,
      Kernel::TileScheduler::FixupThreadCount, plan.q, plan.kt, plan.workers,
      plan.sk_tiles, static_cast<unsigned long long>(plan.sk_units),
      plan.scheduler_workspace_bytes, plan.scheduler_barrier_bytes,
      grid.x, grid.y, grid.z, unsigned(uniform.supported),
      plan_ok ? "PLAN-PASS" : "PLAN-FAIL");
  if (!plan_ok) return 1;

  cutlass::DeviceAllocation<uint32_t> peer(plan.q);
  cutlass::DeviceAllocation<uint32_t> visits(size_t(plan.q) * plan.kt);
  cutlass::DeviceAllocation<uint32_t> totals(6);
  CUTLASS_PPU_CHECK(hggcMemset(peer.get(), 0,
                              sizeof(uint32_t) * plan.q));
  CUTLASS_PPU_CHECK(hggcMemset(visits.get(), 0,
                              sizeof(uint32_t) * size_t(plan.q) * plan.kt));
  CUTLASS_PPU_CHECK(hggcMemset(totals.get(), 0, 6 * sizeof(uint32_t)));
  typename O::Census census{peer.get(), visits.get(), totals.get(),
                            uint32_t(plan.q),
                            uint64_t(plan.q) * plan.kt};
  O::configure_runtime(args, device_id, real_cu, ctas_per_cu, census);
  Gemm gemm;
  CUTLASS_PPU_CHECK(hggcMemset(d.d.get(), 0x7f,
                              sizeof(half_t) * size_t(f.total) * f.n));
  if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess ||
      gemm.run() != cutlass::Status::kSuccess) {
    std::printf("[grouped streamk decode64] census launch failed\n");
    return 1;
  }
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());

  std::vector<uint32_t> hp(plan.q), hv(size_t(plan.q) * plan.kt);
  uint32_t ht[6] = {};
  peer.copy_to_host(hp.data());
  visits.copy_to_host(hv.data());
  totals.copy_to_host(ht);
  int split_tiles = 0, holes = 0;
  uint64_t peer_sum = 0;
  for (uint32_t p : hp) {
    split_tiles += p > 1;
    holes += p == 0;
    peer_sum += p;
  }
  int missing_k = 0, duplicate_k = 0;
  for (uint32_t v : hv) {
    missing_k += v == 0;
    duplicate_k += v > 1;
  }
  uint64_t const peer_excess =
      peer_sum >= uint64_t(plan.q) ? peer_sum - uint64_t(plan.q) : 0;
  bool const census_ok =
      split_tiles == plan.q && holes == 0 && missing_k == 0 &&
      duplicate_k == 0 && ht[0] == peer_sum && ht[1] == uint32_t(plan.q) &&
      ht[2] == 0 && ht[3] == uint32_t(plan.q) && ht[4] == 0 && ht[5] == 0;
  std::printf(
      "[grouped streamk decode64 census] fixture=%s Q=%d Kt=%d "
      "split_tiles=%d peer_sum=%llu peer_excess=%llu "
      "fixup_work_items=%u epilogue=%u separate=%u fixup_final=%u "
      "q_oob=%u empty_decode=%u holes=%d missing_k=%d duplicate_k=%d %s\n",
      f.name, plan.q, plan.kt, split_tiles,
      static_cast<unsigned long long>(peer_sum),
      static_cast<unsigned long long>(peer_excess), ht[0], ht[1], ht[2],
      ht[3], ht[4], ht[5], holes, missing_k, duplicate_k,
      census_ok ? "CENSUS-PASS" : "CENSUS-FAIL");
  errors += !census_ok;

  // Relower with census disabled.  The process-level timeout is also the
  // no-deadlock gate for the exact 64-thread named-barrier cohort.
  O::configure_runtime(args, device_id, real_cu, ctas_per_cu);
  CUTLASS_PPU_CHECK(hggcMemset(d.d.get(), 0x7f,
                              sizeof(half_t) * size_t(f.total) * f.n));
  if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess ||
      gemm.run() != cutlass::Status::kSuccess) {
    std::printf("[grouped streamk decode64] clean launch failed\n");
    return errors + 1;
  }
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  errors += verify_output(f, d, kAlpha, kBeta,
                          "streamk-min2-decode64-nonuniform");
  bool logical_exact = false;
  uint64_t const logical_elements = valid_fixup_elements(
      f, hp, kDecodeTM, kDecodeTN, &logical_exact);
  uint64_t const logical_workspace_rw =
      2ull * logical_elements * sizeof(float);
  uint64_t const full_tile_workspace_rw =
      2ull * uint64_t(kDecodeTM) * kDecodeTN * sizeof(float) * peer_excess;
  errors += !logical_exact;
  std::printf("[grouped streamk decode64 C-traffic] "
              "valid_accumulator_elements=%llu logical_workspace_RW=%llu "
              "old_full_tile_logical_RW=%llu allocation_unchanged=%zu "
              "MODEL-ONLY/not-a-DRAM-counter %s\n",
              static_cast<unsigned long long>(logical_elements),
              static_cast<unsigned long long>(logical_workspace_rw),
              static_cast<unsigned long long>(full_tile_workspace_rw),
              kReductionBytes, logical_exact ? "TRAFFIC-PASS" : "TRAFFIC-FAIL");
  return errors;
}

bool print_c_traffic(Fixture const& f, ArmResult const& arm, int tile_k) {
  uint64_t const output_d = 2ull * f.total * f.n;
  uint64_t const accumulator_tile = uint64_t(kTM) * kTN * sizeof(float);
  uint64_t const old_full_tile_workspace_rw =
      2ull * accumulator_tile * arm.peer_excess;
  uint64_t const logical_workspace_rw =
      2ull * arm.logical_fixup_elements * sizeof(float);
  uint64_t const production_beta0 =
      output_d + logical_workspace_rw;
  // This correctness gate deliberately uses beta=0.5. The shipping benchmark
  // uses beta=0, whose agreed C term is the line above; the gate itself also
  // reads one fp16 C input plane of the same logical size as D.
  uint64_t const gate_c_input = kBeta == 0.0f ? 0ull : output_d;
  uint64_t const gate_path = production_beta0 + gate_c_input;
  uint64_t const expected_logical_workspace_rw =
      2ull * arm.expected_logical_fixup_elements * sizeof(float);
  uint64_t const expected_production_beta0 =
      output_d + expected_logical_workspace_rw;
  uint64_t const expected_gate_path = expected_production_beta0 + gate_c_input;
  bool const traffic_ok = production_beta0 == expected_production_beta0 &&
                          gate_path == expected_gate_path &&
                          arm.expected_supported &&
                          arm.peer_excess == arm.expected_peer_excess &&
                          arm.logical_fixup_elements ==
                              arm.expected_logical_fixup_elements;
  std::printf("[grouped streamk C-traffic] fixture=%s TK=%d Min=%u "
              "output_D=%llu accumulator_Wtile=%llu peer_excess=%llu "
              "old_full_tile_logical_RW=%llu valid_accumulator_elements=%llu "
              "logical_workspace_RW=%llu production_beta0_C=%llu "
              "gate_beta=%.3g gate_C_read=%llu gate_C_path=%llu "
              "MODEL-ONLY/not-a-DRAM-counter %s\n",
              f.name, tile_k, kMinSkIters,
              static_cast<unsigned long long>(output_d),
              static_cast<unsigned long long>(accumulator_tile),
              static_cast<unsigned long long>(arm.peer_excess),
              static_cast<unsigned long long>(old_full_tile_workspace_rw),
              static_cast<unsigned long long>(arm.logical_fixup_elements),
              static_cast<unsigned long long>(logical_workspace_rw),
              static_cast<unsigned long long>(production_beta0),
              double(kBeta), static_cast<unsigned long long>(gate_c_input),
              static_cast<unsigned long long>(gate_path),
              traffic_ok ? "TRAFFIC-PASS" : "TRAFFIC-FAIL");
  return traffic_ok;
}

int run_legacy_control(Fixture const& f, DeviceFixture& d) {
  std::vector<int> gm = f.me;
  cutlass::DeviceAllocation<int> dgm(f.experts);
  dgm.copy_from_host(gm.data());
  size_t const wsb = size_t(f.experts + 1) * sizeof(int) + 4096;
  cutlass::DeviceAllocation<char> ws(wsb);
  CUTLASS_PPU_CHECK(hggcMemset(d.d.get(), 0x7f,
                              sizeof(half_t) * size_t(f.total) * f.n));
  int const failures0 = moe_grouped_ppu::moeg_fail_count();
  moe_grouped_ppu::filter_and_run<
      QM::FinegrainedScaleZero, kTM, kTN, 64, kWM, kWN, kStages,
      int4_t, void, false, kArtifactTileK>(
      d.a.get(), d.b.get(), d.scales.get(), d.zeros.get(), d.ptr_d.get(),
      d.strides.get(), dgm.get(), f.mmax, f.n, f.k, f.experts, kGs,
      d.shapes.get(), d.host_shapes.data(), d.row_offsets.get(), ws.get(),
      wsb, nullptr);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  if (moe_grouped_ppu::moeg_fail_count() != failures0) {
    std::printf("[legacy grouped control] launch failed\n");
    return 1;
  }
  return verify_output(f, d, 1.0f, 0.0f, "legacy-grouped-TK64");
}

}  // namespace

int main() {
  int device_id = 0;
  CUTLASS_PPU_CHECK(hggcGetDevice(&device_id));
  int const real_cu =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
  std::printf("== grouped Stream-K phase 2 min2: device=%d cu=%d ==\n", device_id,
              real_cu);

  using P8 = cutlass::gemm::kernel::detail::
      PersistentTileSchedulerPPUStreamKParamsT<8>;
  using P2 = cutlass::gemm::kernel::detail::
      PersistentTileSchedulerPPUStreamKParamsT<2>;
  int errors = 0;
  errors += !host_policy_line<P8>("min8", 128, 8, 432, 0, 128, 128);
  errors += !host_policy_line<P2>("min2", 128, 8, 432, 128, 128, 432);

  moe_router_fixture::Rows routed;
  char why[160] = "";
  if (!moe_router_fixture::route(1, 8, 256, routed, why, sizeof why)) {
    std::printf("S068 router refused: %s\n", why);
    return 1;
  }
  Fixture s068 = make_fixture("S068", routed.per_expert, 512, 2048);
  std::vector<int> const expected_active{7, 11, 35, 77, 127, 128, 218, 224};
  std::vector<int> local_lock_ids;
  int output_tiles = 0;
  for (int e : s068.active) {
    int const mt = (s068.me[e] + kTM - 1) / kTM;
    int const nt = (s068.n + kTN - 1) / kTN;
    output_tiles += mt * nt;
    for (int n_idx = 0; n_idx < nt; ++n_idx)
      for (int m_idx = 0; m_idx < mt; ++m_idx)
        local_lock_ids.push_back(m_idx + n_idx * mt);
  }
  std::sort(local_lock_ids.begin(), local_lock_ids.end());
  int const unique_local_locks = int(std::unique(
      local_lock_ids.begin(), local_lock_ids.end()) - local_lock_ids.begin());
  int const local_lock_collisions = output_tiles - unique_local_locks;
  std::vector<int> decode_local_lock_ids;
  int decode_output_tiles = 0;
  for (int e : s068.active) {
    int const mt = (s068.me[e] + kDecodeTM - 1) / kDecodeTM;
    int const nt = (s068.n + kDecodeTN - 1) / kDecodeTN;
    decode_output_tiles += mt * nt;
    for (int n_idx = 0; n_idx < nt; ++n_idx)
      for (int m_idx = 0; m_idx < mt; ++m_idx)
        decode_local_lock_ids.push_back(m_idx + n_idx * mt);
  }
  std::sort(decode_local_lock_ids.begin(), decode_local_lock_ids.end());
  int const decode_unique_local_locks = int(std::unique(
      decode_local_lock_ids.begin(), decode_local_lock_ids.end()) -
      decode_local_lock_ids.begin());
  int const decode_local_lock_collisions =
      decode_output_tiles - decode_unique_local_locks;
  bool const router_ok = s068.active == expected_active &&
                         s068.experts == 256 && s068.total == 8 &&
                         output_tiles == 16 && local_lock_collisions == 14 &&
                         decode_output_tiles == 128 &&
                         decode_local_lock_collisions == 112;
  std::printf("[grouped streamk plan] fixture=S068 router=%s E=%d active=%zu "
              "empty=%d active_ids=",
              moe_router_fixture::kName, s068.experts, s068.active.size(),
              s068.experts - int(s068.active.size()));
  for (size_t i = 0; i < s068.active.size(); ++i)
    std::printf("%s%d", i ? "," : "", s068.active[i]);
  // q has two n tiles per active expert.  Replacing q by expert-local (m,n)
  // would collapse 16 lock ids to two: the fixture actively exposes 14 aliases.
  std::printf(" Q=%d local_lock_collisions=%d "
              "decode64_Q=%d decode64_local_lock_collisions=%d %s\n",
              output_tiles, local_lock_collisions, decode_output_tiles,
              decode_local_lock_collisions,
              router_ok ? "ROUTER-PASS" : "ROUTER-FAIL");
  errors += !router_ok;

  // Canonical placement must not change when only the consumer TileK changes.
  std::vector<int8_t> packed256(s068.packed.size(), 0);
  std::vector<int8_t> packed_decode64(s068.packed.size(), 0);
  size_t const per = size_t(s068.k) * s068.n * 4 / 8;
  for (int e : s068.active) {
    xplane::place_derived<4, kTM, kTN, 256, kWM, kWN, 1,
                          kArtifactTileK>(
        packed256.data() + size_t(e) * per, s068.codes[e], s068.n, s068.k);
    xplane::place_derived<4, kDecodeTM, kDecodeTN, kDecodeTK,
                          kDecodeWM, kDecodeWN, 1, kArtifactTileK>(
        packed_decode64.data() + size_t(e) * per, s068.codes[e],
        s068.n, s068.k);
  }
  int const placement_diff = int(std::inner_product(
      s068.packed.begin(), s068.packed.end(), packed256.begin(), size_t(0),
      std::plus<size_t>{}, [](int8_t a, int8_t b) { return a != b; }));
  int const decode_placement_diff = int(std::inner_product(
      s068.packed.begin(), s068.packed.end(), packed_decode64.begin(), size_t(0),
      std::plus<size_t>{}, [](int8_t a, int8_t b) { return a != b; }));
  bool const artifact_ok = placement_diff == 0 && decode_placement_diff == 0;
  std::printf("[grouped streamk artifact] TK64/TK256 byte_diff=%d/%zu "
              "decode64 byte_diff=%d/%zu %s\n",
              placement_diff, s068.packed.size(), decode_placement_diff,
              s068.packed.size(), artifact_ok ? "PASS" : "FAIL");
  errors += !artifact_ok;

  DeviceFixture ds068(s068);
  errors += run_legacy_control(s068, ds068);
  auto tk256 = run_streamk_arm<256>(s068, ds068, device_id, real_cu, true);
  auto tk64 = run_streamk_arm<64>(s068, ds068, device_id, real_cu, true);
  errors += tk256.errors + tk64.errors;
  errors += run_decode64_nonuniform(s068, ds068, device_id, real_cu);

  errors += !print_c_traffic(s068, tk256, 256);
  errors += !print_c_traffic(s068, tk64, 64);

  Fixture ragged = make_fixture("ragged-0,1,17,0,33", {0, 1, 17, 0, 33},
                                256, 2048);
  DeviceFixture dragged(ragged);
  auto rag = run_streamk_arm<64>(ragged, dragged, device_id, real_cu, false);
  errors += rag.errors;
  errors += !print_c_traffic(ragged, rag, 64);

  std::printf("== grouped Stream-K phase 2 min2 %s: errors=%d ==\n",
              errors ? "FAIL" : "PASS", errors);
  return errors ? 1 : 0;
}
