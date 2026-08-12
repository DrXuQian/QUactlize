// Shared harness for the GEMV perf sweep. The sweep itself lives in one generated .cu per config, NOT here.
//
// WHY SPLIT AT ALL -- the same reason lowbit_moe_bench.hpp gives: one .cu is ONE compiler invocation, so a
// 42-row sweep in a single translation unit compiles strictly serially. cutlass_build_dev_kernels emits one
// add_custom_command per .cu, so N sources become N independent commands and `make -j` runs them
// concurrently. Splitting is the only way to instantiate MORE and wait LESS.
//
// make_bufs is deliberately NOT a template: the weight packing depends on (format, layout, TileSizeK) only, so
// the buffers are built once per GROUP in the caller's translation unit and passed to every config unit of
// that group. Packing per unit would repeat an O(N*K) pass 42 times.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <utility>

#include "gemv_perf_fixture.hpp"

// NARROW THE INSTANTIATION SET TO WHAT THE SHAPE LIST ACTUALLY CALLS. Every (gs, quant) pair is a full set of
// kernels per unit; the shapes below use gs 32 and 128 and only the two finegrained quant ops, so paying for
// gs {0,16,64} and PerColScaleOnly would triple the per-unit compile for coverage nothing exercises.
#define GEMV_GS_LIST(EMIT) EMIT(32) EMIT(128)
#define GEMV_QUANT_LIST(EMIT, G) \
  EMIT(QuantOp::FinegrainedScaleOnly, G) EMIT(QuantOp::FinegrainedScaleZero, G)
#define GEMV_ENABLE_BIAS 0
#include "gemv_lowbit/gemv_launcher.hpp"
#include "gemv_lowbit/gemv_rt.hpp"

using namespace ppu_gemv;

#define GEMV_PERF_REV 2

// ppu001. A CHECKED MIRROR OF bench_select.hpp's kHbmGBPerSecond, not a second source. The comment here used to
// read "same constants the MoE bench uses, so the percentages are comparable" -- an assertion a copy cannot make
// good on, and this file was absent from ci/check_bench_measurement.py's list while the dense and MoE benches
// were on it. It is now checked: that gate parses this literal and fails if it differs from the shared one.
//
// WHY NOT JUST INCLUDE bench_select.hpp, which was tried and reverted: it defines `Best` at global scope and so
// does this file, and the clash reaches further than either -- quactlize/csrc/CMakeLists.txt.in emits
// `void <fn>(const Shape&, const Bufs&, Best&)` for every generated GEMV unit, so the rename would have to
// travel through the generator. And bench_select.hpp's `Best` is the LEGACY selection machinery its own header
// says is scheduled for deletion; dragging it into a third bench is the wrong direction regardless.
static constexpr double HBM_GBS = 2766.0;
static constexpr int    CU      = 72;

inline const char* only_filter() { return std::getenv("GEMV_ONLY"); }
inline bool acu_mode() { return std::getenv("GEMV_ACU") != nullptr; }
// GEMV_ONLY is a substring of the whole tag, which means counting the spaces a "%-7s %-6s" pads out -- fragile
// for an acu capture, where selecting the wrong number of rows wastes the whole run. GEMV_FMT and GEMV_CFG match
// the two halves independently, so `GEMV_FMT=int4 GEMV_CFG="s16/t128 N2 C2"` picks exactly one row.
inline bool row_selected(const char* tag) {
  const char* f = only_filter();
  if (f && std::strstr(tag, f) == nullptr) return false;
  const char* fmt = std::getenv("GEMV_FMT");
  if (fmt && std::strncmp(tag, fmt, std::strlen(fmt)) != 0) return false;   // format name starts the tag
  const char* cfg = std::getenv("GEMV_CFG");
  if (cfg && std::strstr(tag, cfg) == nullptr) return false;
  return true;
}

// chrono + a device sync: the same timing shape lowbit_moe_bench.hpp uses, so it works under both runtimes.
template <typename F>
inline double time_it(F&& f, int iters) {
  if (iters == 0) { f(); rt_sync("cold launch"); return 0.0; }
  for (int i = 0; i < 5; ++i) f();
  rt_sync("warmup");
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) f();
  rt_sync("timed");
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

struct Best { char tag[96] = ""; double us = 1e30; double pct = 0; };
inline void upd(Best& b, const char* t, double us, double pct) {
  if (us > 0 && us < b.us) { std::snprintf(b.tag, sizeof(b.tag), "%s", t); b.us = us; b.pct = pct; }
}

// ---------------------------------------------------------------------------------------------------
// One benchmark problem. `experts == 0` is dense.
struct Shape {
  const char* name;
  int experts;      // 0 = dense
  int rows;         // dense: m. MoE: global routed token count.
  int N, K;
  int gs;
  QuantOp quant;
  int topk = 0;     // grouped: route every token to top-k distinct experts
  int active = 0;   // grouped: expected distinct active experts; independent of E
};

struct OutputWitness {
  std::size_t index = 0;
  float want = 0.0f;
  int expert = 0;
  int column = 0;
};

struct Bufs {
  DevBuf A, W, Wh, S, Z, O, Off;
  std::vector<int> offs;
  std::vector<int> rows_per_expert;
  std::vector<int> active_ids;
  std::vector<OutputWitness> witnesses;
  int total_rows = 0;
  int max_rows = 0;
};

// Pack a plane the way gemv_wformat.hpp defines the layout. Deliberately the same bit-position expression as
// the correctness gate's packer -- there is one convention and it lives in one form.
inline uint32_t plane_code(int bits, int n, int k, uint32_t seed) {
  return gemv_perf_fixture::plane_code(bits, n, k, seed);
}

inline std::vector<uint8_t> pack_plane(WLayout lay, int bits, int TS, int N, int K, uint32_t seed) {
  std::vector<uint8_t> out(size_t(N) * K * bits / 8, 0);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      size_t bitpos = (lay == WLayout::Native)
          ? (size_t(n) * K + size_t(k)) * bits
          : ((size_t(k / TS) * N * TS) + size_t(n) * TS + size_t(k % TS)) * bits;
      uint32_t const v = plane_code(bits, n, k, seed);
      out[bitpos >> 3] |= uint8_t(v << (bitpos & 7));
    }
  return out;
}

inline Bufs make_bufs(WFormat fmt, WLayout Lay, int TS, Shape const& sh) {
  int const LoBits = lo_bits_of(fmt), HiBits = hi_bits_of(fmt);
  bool const TwoPlane = is_two_plane(fmt);

  int const experts = sh.experts > 0 ? sh.experts : 1;
  Bufs b;
  std::vector<int> active_slot(std::size_t(experts), -1);
  if (sh.experts > 0) {
    auto route = gemv_perf_fixture::make_route(sh.experts, sh.rows, sh.topk);
    if (!route.valid) {
      std::fprintf(stderr, "GEMV perf router refused E=%d tokens=%d topk=%d\n",
                   sh.experts, sh.rows, sh.topk);
      std::abort();
    }
    if (int(route.active_ids.size()) != sh.active) {
      std::fprintf(stderr,
                   "GEMV perf active-expert authority drift: E=%d tokens=%d topk=%d got=%zu want=%d\n",
                   sh.experts, sh.rows, sh.topk, route.active_ids.size(), sh.active);
      std::abort();
    }
    b.offs = std::move(route.row_offsets);
    b.rows_per_expert = std::move(route.rows_per_expert);
    b.active_ids = std::move(route.active_ids);
    active_slot = std::move(route.active_slot_for_expert);
    b.total_rows = route.total_rows;
    b.max_rows = route.max_rows;
  } else {
    b.offs = {0, sh.rows};
    b.rows_per_expert = {sh.rows};
    b.active_ids = {0};
    active_slot[0] = 0;
    b.total_rows = sh.rows;
    b.max_rows = sh.rows;
  }

  int const sk = (sh.gs == 0) ? 1 : sh.K / sh.gs;

  using half_t = cutlass::half_t;
  std::vector<half_t> hA(size_t(b.total_rows) * sh.K);
  std::vector<half_t> hS(size_t(experts) * sk * sh.N);
  std::vector<half_t> hZ(size_t(experts) * sk * sh.N);
  std::size_t const lo_bytes = std::size_t(
      gemv_perf_fixture::packed_plane_bytes(sh.N, sh.K, LoBits));
  std::size_t const hi_bytes = HiBits ? std::size_t(
      gemv_perf_fixture::packed_plane_bytes(sh.N, sh.K, HiBits)) : 0;
  std::vector<uint8_t> wl(std::size_t(experts) * lo_bytes);
  std::vector<uint8_t> wh(std::size_t(experts) * hi_bytes);

  for (int e = 0; e < experts; ++e) {
    bool const active = active_slot[std::size_t(e)] >= 0;
    uint32_t const lo_seed = gemv_perf_fixture::plane_seed(e, active, false);
    uint32_t const hi_seed = gemv_perf_fixture::plane_seed(e, active, true);
    auto const plo = pack_plane(Lay, LoBits, TS, sh.N, sh.K, lo_seed);
    std::memcpy(wl.data() + gemv_perf_fixture::packed_plane_expert_offset(
                    e, sh.N, sh.K, LoBits), plo.data(), lo_bytes);
    if (TwoPlane) {
      auto const phi = pack_plane(Lay, HiBits, TS, sh.N, sh.K, hi_seed);
      std::memcpy(wh.data() + gemv_perf_fixture::packed_plane_expert_offset(
                      e, sh.N, sh.K, HiBits), phi.data(), hi_bytes);
    }
    for (int g = 0; g < sk; ++g)
      for (int n = 0; n < sh.N; ++n) {
        std::size_t const i = (std::size_t(e) * sk + g) * sh.N + n;
        hS[i] = half_t(gemv_perf_fixture::scale_value(e, g, n, active));
        hZ[i] = half_t(gemv_perf_fixture::zero_value(e, g, n, active));
      }
    for (int r = 0; r < b.rows_per_expert[std::size_t(e)]; ++r) {
      int const row = b.offs[std::size_t(e)] + r;
      half_t const av(gemv_perf_fixture::activation_value(e, r));
      std::fill(hA.begin() + std::size_t(row) * sh.K,
                hA.begin() + std::size_t(row + 1) * sh.K, av);
    }
  }

  // Three columns per real row are enough to make real-expert W/S/Z identity
  // observable without copying or golden-checking the full output matrix.
  int const witness_columns[] = {0, sh.N / 2, sh.N - 1};
  for (int e : b.active_ids) {
    bool const active = true;
    uint32_t const lo_seed = gemv_perf_fixture::plane_seed(e, active, false);
    uint32_t const hi_seed = gemv_perf_fixture::plane_seed(e, active, true);
    for (int r = 0; r < b.rows_per_expert[std::size_t(e)]; ++r) {
      int const row = b.offs[std::size_t(e)] + r;
      float const av = float(hA[std::size_t(row) * sh.K]);
      for (int n : witness_columns) {
        float acc = 0.0f;
        for (int k = 0; k < sh.K; ++k) {
          uint32_t q = plane_code(LoBits, n, k, lo_seed);
          if (TwoPlane) q |= plane_code(HiBits, n, k, hi_seed) << LoBits;
          int const g = sh.gs == 0 ? 0 : k / sh.gs;
          std::size_t const si = (std::size_t(e) * sk + g) * sh.N + n;
          float const z = has_zero(sh.quant) ? float(hZ[si]) : 0.0f;
          acc += av * (float(q) * float(hS[si]) + z);
        }
        b.witnesses.push_back({std::size_t(row) * sh.N + n, acc, e, n});
      }
    }
  }

  b.A = DevBuf(hA.size() * 2);  b.A.from_host(hA.data());
  b.S = DevBuf(hS.size() * 2);  b.S.from_host(hS.data());
  if (has_zero(sh.quant)) { b.Z = DevBuf(hZ.size() * 2); b.Z.from_host(hZ.data()); }
  b.W = DevBuf(wl.size());      b.W.from_host(wl.data());
  if (TwoPlane) {
    b.Wh = DevBuf(wh.size());   b.Wh.from_host(wh.data());
  }
  b.O = DevBuf(size_t(b.total_rows) * sh.N * 2);
  rt_memset0(b.O.p, b.O.bytes);
  if (sh.experts > 0) { b.Off = DevBuf(b.offs.size() * 4); b.Off.from_host(b.offs.data()); }
  return b;
}

inline bool verify_witnesses(Bufs const& b, int n, char const* tag) {
  using half_t = cutlass::half_t;
  std::vector<half_t> got(std::size_t(b.total_rows) * n);
  rt_d2h(got.data(), b.O.p, got.size() * sizeof(half_t));
  int bad = 0;
  for (auto const& w : b.witnesses) {
    float const value = float(got[w.index]);
    float const want = float(half_t(w.want));
    float const tol = 0.02f * std::max(1.0f, std::fabs(want));
    if (!std::isfinite(value) || std::fabs(value - want) > tol) {
      if (bad++ < 4)
        std::printf("    witness expert=%d n=%d got=%.6g want=%.6g\n",
                    w.expert, w.column, double(value), double(want));
    }
  }
  if (bad) std::printf("  %-34s %10s | WRONG EXPERT DATA (%d/%zu witnesses) -- excluded\n",
                       tag, "-", bad, b.witnesses.size());
  return bad == 0;
}

// ---------------------------------------------------------------------------------------------------
template <typename Details, int CtaN, int Chunk>
inline void run_row(Shape const& sh, Bufs const& b, Best& best) {
  constexpr int LoBits = Details::kLoBits, HiBits = Details::kHiBits;
  constexpr int TotalBits = LoBits + HiBits;
  constexpr int StepK = Details::kStepK, Threads = Details::kThreads;

  char tag[96];
  std::snprintf(tag, sizeof(tag), "%-7s %-6s s%-2d/t%-3d N%d C%d", Details::format_name(),
                name_of(Details::kLayout), StepK, Threads, CtaN, Chunk);
  if (!row_selected(tag)) return;

  int const experts = sh.experts > 0 ? sh.experts : 1;
  int const sk = (sh.gs == 0) ? 1 : sh.K / sh.gs;

  Params p;
  p.act = b.A.p; p.weight = b.W.p; p.weight_hi = b.Wh.p; p.scales = b.S.p; p.zeros = b.Z.p; p.out = b.O.p;
  p.m = b.total_rows; p.n = sh.N; p.k = sh.K; p.groupsize = sh.gs;
  p.format = Details::kFormat; p.quant = sh.quant; p.layout = Details::kLayout;
  if (sh.experts > 0) {
    p.num_experts = sh.experts; p.row_offsets = b.Off.as<int>(); p.max_rows = b.max_rows;
    p.w_bytes_per_expert = int64_t(gemv_perf_fixture::packed_plane_bytes(sh.N, sh.K, LoBits));
    p.w_hi_bytes_per_expert = HiBits ?
        int64_t(gemv_perf_fixture::packed_plane_bytes(sh.N, sh.K, HiBits)) : 0;
    p.scale_elems_per_expert = int64_t(sk) * sh.N;
  }

  int const f0 = gemv_fail_count();
  std::vector<cutlass::half_t> output_poison(
      std::size_t(b.total_rows) * sh.N, cutlass::half_t::bitcast(uint16_t(0x7f7f)));
  rt_h2d(b.O.p, output_poison.data(), output_poison.size() * sizeof(cutlass::half_t));
  auto go = [&] { launch_gemv<Details, CtaN, Chunk>(p, 0); };
  double const us = acu_mode() ? (time_it(go, 0), 0.0) : time_it(go, 100);
  if (acu_mode()) { std::printf("  [acu] ONE COLD launch (not a timing): %s\n", tag); return; }
  if (gemv_fail_count() != f0) {
    std::printf("  %-34s %10s | DID NOT RUN (launch refused) -- excluded\n", tag, "-");
    return;
  }
  if (!verify_witnesses(b, sh.N, tag)) return;

  // Compulsory traffic. B counted grid.x times because every m-tile re-reads it; A counted once (it is tiny
  // at decode and should be served by L2 across the n-tiles -- if the measured rate exceeds this model, that
  // assumption is what broke).
  int const ctam = std::min(b.max_rows, GEMV_CTAM_MAX);
  int64_t grid_m_sum = 0;
  for (int e : b.active_ids)
    grid_m_sum += (b.rows_per_expert[std::size_t(e)] + ctam - 1) / ctam;
  double const wb  = double(sh.N) * sh.K * TotalBits / 8.0;
  double const sb  = double(sk) * sh.N * 2.0 * (has_zero(sh.quant) ? 2 : 1);
  double const ab  = double(b.total_rows) * sh.K * 2.0;
  double const db  = double(b.total_rows) * sh.N * 2.0;
  double const bytes = double(grid_m_sum) * (wb + sb) + ab + db;
  double const gbs = bytes / (us * 1e-6) / 1e9;
  double const pct = 100.0 * gbs / HBM_GBS;

  int64_t const work_ctas = grid_m_sum * (sh.N / CtaN);
  int64_t const launch_ctas = int64_t((b.max_rows + ctam - 1) / ctam) *
                              (sh.N / CtaN) * experts;
  // WARPS OF WORK PER CU, not achieved occupancy -- the same quantity the MoE bench prints as grid_wrp/CU.
  // Naming it "wrp/CU" invited exactly the misreading that cost rounds earlier: 14.2 there was the TOTAL work,
  // and the reason occupancy could not exceed it. `wave` divides by the 64-warp/CU hardware maximum, so it is
  // a LOWER bound on the wave count (real occupancy is below 64, so real waves are more).
  double const wkwrp_cu = double(work_ctas) * (Threads / 32.0) / CU;

  // A ROW OVER THE NAMEPLATE INDICTS THE TRAFFIC MODEL, NOT THE MEASUREMENT -- and this used to DROP it from
  // the winner. The comment above already says why the rate can exceed 2766: the model charges the weight once
  // per grid_m and assumes L2 serves the n-tiles, so a run where that assumption holds better than modelled
  // reads high. Excluding those rows removed exactly the FASTEST configurations from `best`, which is worse
  // than the MoE bench's version of this bug (#52 item 1) -- there it mislabelled a row, here it deleted the
  // winner. The row is now retained and the MODEL is flagged.
  std::printf("  %-34s %8.2f us | %7.1f GB/s | %5.1f%% of %.0f nameplate | cta launch/work %6lld/%-6lld | wkwrp/CU %6.1f | "
              "wave>=%5.1f%s\n",
              tag, us, gbs, pct, HBM_GBS, (long long)launch_ctas,
              (long long)work_ctas, wkwrp_cu, wkwrp_cu / 64.0,
              gbs > HBM_GBS ? "  <-- MODEL BROKE: over nameplate, so the once-per-grid_m weight charge is "
                              "wrong here; row RETAINED" : "");
  upd(best, tag, us, pct);
}

// ---------------------------------------------------------------------------------------------------
