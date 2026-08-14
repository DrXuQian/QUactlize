// Same-machine, same-logical-Q4_K benchmark for INBOX 176.
//
// The external arm reads the user-supplied native GGUF block_q4_K bytes.  The
// production arm reads the exact merged resident artifact shared by prefill and
// decode: xplane-placed Q4 codes plus byte-neutral packed metadata units.  The
// offline placement/packing completes before timing and is never charged to the
// production GEMV.  No raw-GGUF reader is allowed to stand in for that arm.

#include "q4k_pdf_vs_bc_reference_nv.cuh"
#include "q4k_pdf_ab_fixture.hpp"

// actlize's helper macros intentionally recognize hgcc/PPU but not stock nvcc;
// without this benchmark-only compatibility seam its CUTLASS_DEVICE functions
// become host-only inline functions and the real BC kernel cannot be instantiated
// on sm_120.  Define the ordinary CUDA meanings before the production headers
// are parsed.  __HGGCCC__ remains unset, so target dispatch still selects the
// NVIDIA instruction arm rather than pretending nvcc is hgcc.
#include "cutlass/cutlass.h"
#undef CUTLASS_HOST_DEVICE
#undef CUTLASS_DEVICE
#define CUTLASS_HOST_DEVICE __forceinline__ __host__ __device__
#define CUTLASS_DEVICE __forceinline__ __device__

#include "gguf_bc_vecdot.hpp"
#include "gguf_unit_pack.hpp"
#include "xplane_offline.hpp"
#include "gguf_bc_q4_gemv.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kM = 1;
constexpr int kN = 4096;
constexpr int kK = 4096;
// The shipping Python producer uses scale_first_tile_k=64 for Q4 when tile_k
// is not explicitly requested.  The reader template must name that artifact;
// Traits::DefaultArtifactTileK=256 is only the legacy unversioned C ABI.
constexpr int kArtifactTileK = 64;
constexpr int kQtype = 12;
using RawBlock = q4k_pdf_reconstruction::block_q4_K;
using ExactRawBlock = q4k_gemv::block_q4_K;
using BCUnit = gguf_scale::packed_unit::Unit<gguf_scale::KType::Q4_K>;

static_assert(sizeof(RawBlock) == sizeof(ExactRawBlock),
              "fixture and exact PDF reference must share the GGUF Q4_K ABI");
static_assert(sizeof(RawBlock) == 144, "Q4_K raw block ABI changed");
static_assert(BCUnit::kUnitTotal == 16 && BCUnit::kSbPerUnit == 1,
              "shipping Q4_K packed unit must remain byte-neutral");

[[noreturn]] void fail(std::string const& message) {
  throw std::runtime_error(message);
}

void cuda_ok(cudaError_t status, char const* where) {
  if (status != cudaSuccess) fail(std::string(where) + ": " + cudaGetErrorString(status));
}

struct Buffer {
  void* ptr = nullptr;
  std::size_t bytes = 0;
  Buffer() = default;
  explicit Buffer(std::size_t n) : bytes(n) {
    if (n) cuda_ok(cudaMalloc(&ptr, n), "cudaMalloc");
  }
  Buffer(Buffer const&) = delete;
  Buffer& operator=(Buffer const&) = delete;
  Buffer(Buffer&& other) noexcept : ptr(other.ptr), bytes(other.bytes) {
    other.ptr = nullptr; other.bytes = 0;
  }
  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      if (ptr) cudaFree(ptr);
      ptr = other.ptr; bytes = other.bytes;
      other.ptr = nullptr; other.bytes = 0;
    }
    return *this;
  }
  ~Buffer() { if (ptr) cudaFree(ptr); }
  template <class T> T* as() const { return static_cast<T*>(ptr); }
};

template <class T>
Buffer upload_repeated(std::vector<T> const& host, int copies) {
  std::size_t const one = host.size() * sizeof(T);
  Buffer out(one * std::size_t(copies));
  for (int i = 0; i < copies; ++i)
    cuda_ok(cudaMemcpy(static_cast<std::uint8_t*>(out.ptr) + std::size_t(i) * one,
                       host.data(), one, cudaMemcpyHostToDevice), "upload repeated operand");
  return out;
}

struct Artifact {
  std::vector<std::uint8_t> low;
  std::vector<std::uint8_t> units;
  std::vector<std::uint8_t> logical_q_kn;
};

Artifact make_artifact(std::vector<RawBlock> const& raw, bool plant_bad_code) {
  int const bpr = kK / 256;
  Artifact a;
  a.logical_q_kn.resize(std::size_t(kK) * kN);
  for (int n = 0; n < kN; ++n) {
    for (int sb = 0; sb < bpr; ++sb) {
      RawBlock const& block = raw[std::size_t(n) * bpr + sb];
      for (int i = 0; i < 256; ++i) {
        int q = q4k_pdf_ab::get_q(block, i);
        if (plant_bad_code) q ^= 8;  // flip the int4 sign bit: must exceed the fixed accuracy gate
        a.logical_q_kn[std::size_t(sb * 256 + i) * kN + n] = std::uint8_t(q);
      }
    }
  }

  a.low.assign(std::size_t(kN) * kK / 2, 0);
  xplane::place_derived<4, 64, 64, kArtifactTileK, 32, 32, 1, kArtifactTileK>(
      reinterpret_cast<int8_t*>(a.low.data()), a.logical_q_kn, kN, kK);

  std::vector<std::uint8_t> recovered;
  xplane::recover_derived<4, 64, 64, kArtifactTileK, 32, 32, 1, kArtifactTileK>(
      reinterpret_cast<int8_t const*>(a.low.data()), recovered, kN, kK);
  if (recovered != a.logical_q_kn) fail("xplane place/recover roundtrip is not exact");

  std::int64_t const unit_bytes = gguf_scale::unit_pack::bytes<gguf_scale::KType::Q4_K>(kN, kK);
  if (unit_bytes <= 0) fail("Q4_K packed-unit byte query rejected the benchmark shape");
  a.units.resize(std::size_t(unit_bytes));
  gguf_scale::unit_pack::pack<gguf_scale::KType::Q4_K>(
      reinterpret_cast<std::uint8_t const*>(raw.data()), a.units.data(), kN, kK, 1);

  std::size_t bad_units = 0;
  for (int n = 0; n < kN; ++n) {
    for (int sb = 0; sb < bpr; ++sb) {
      RawBlock const& block = raw[std::size_t(n) * bpr + sb];
      std::uint8_t const* unit = a.units.data() + (std::int64_t(sb) * kN + n) * BCUnit::kUnitTotal;
      for (int g = 0; g < 8; ++g) {
        int sc = 0, mn = 0;
        q4k_pdf_ab::get_scale_min(block, g, sc, mn);
        auto const got = gguf_scale::packed_unit::unit_group_sb<gguf_scale::KType::Q4_K, 0>(unit, 0, g);
        float const want_scale = __half2float(block.d) * float(sc);
        float const want_zero = -__half2float(block.dmin) * float(mn);
        bad_units += float(got.scale) != float(cutlass::half_t(want_scale));
        bad_units += float(got.zero) != float(cutlass::half_t(want_zero));
      }
    }
  }
  if (bad_units) fail("packed-unit decode disagrees with raw Q4_K metadata: " + std::to_string(bad_units));

  std::size_t const raw_bytes = raw.size() * sizeof(RawBlock);
  if (a.low.size() + a.units.size() != raw_bytes)
    fail("merged artifact is not byte-neutral against native Q4_K");
  return a;
}

enum class InputMode { Positive, Signed };

std::vector<half> make_activation(InputMode mode) {
  std::vector<half> a(kK);
  for (int k = 0; k < kK; ++k) {
    float value = 0.f;
    if (mode == InputMode::Positive) {
      value = float(1 + ((3 * k + 1) & 7)) / 64.f;
    } else {
      // Deliberately non-periodic against the fixture's affine q(k) pattern.
      // A short modular sequence made the whole signed dot exactly zero and
      // therefore measured no accumulation precision at all.
      std::uint32_t h = std::uint32_t(k) * UINT32_C(747796405) + UINT32_C(2891336453);
      h ^= h >> 16; h *= UINT32_C(2246822519); h ^= h >> 13;
      int const signed_code = int(h % 31u) - 15;
      value = float(signed_code) / 128.f;
    }
    a[std::size_t(k)] = __float2half_rn(value);
  }
  return a;
}

std::vector<float> official_golden(std::vector<RawBlock> const& raw,
                                   std::vector<half> const& activation) {
  int const bpr = kK / 256;
  std::vector<float> out(kN);
  for (int n = 0; n < kN; ++n) {
    double sum = 0.0;
    for (int sb = 0; sb < bpr; ++sb) {
      RawBlock const& block = raw[std::size_t(n) * bpr + sb];
      for (int i = 0; i < 256; ++i) {
        int sc = 0, mn = 0;
        q4k_pdf_ab::get_scale_min(block, i / 32, sc, mn);
        float const w = float(q4k_pdf_ab::get_q(block, i)) * __half2float(block.d) * float(sc)
                      - __half2float(block.dmin) * float(mn);
        sum += double(__half2float(activation[std::size_t(sb * 256 + i)])) * double(w);
      }
    }
    out[std::size_t(n)] = float(sum);
  }
  return out;
}

struct DeviceData {
  int copies = 0;
  std::size_t raw_one = 0;
  std::size_t low_one = 0;
  std::size_t units_one = 0;
  Buffer raw;
  Buffer low;
  Buffer units;
  Buffer activation;

  DeviceData(std::vector<RawBlock> const& host_raw, Artifact const& artifact,
             std::vector<half> const& act, int count)
      : copies(count), raw_one(host_raw.size() * sizeof(RawBlock)),
        low_one(artifact.low.size()), units_one(artifact.units.size()),
        raw(upload_repeated(host_raw, count)), low(upload_repeated(artifact.low, count)),
        units(upload_repeated(artifact.units, count)), activation(upload_repeated(act, 1)) {}

  ExactRawBlock const* raw_at(int copy) const {
    return reinterpret_cast<ExactRawBlock const*>(
        static_cast<std::uint8_t const*>(raw.ptr) + std::size_t(copy) * raw_one);
  }
  std::uint8_t const* low_at(int copy) const {
    return static_cast<std::uint8_t const*>(low.ptr) + std::size_t(copy) * low_one;
  }
  std::uint8_t const* units_at(int copy) const {
    return static_cast<std::uint8_t const*>(units.ptr) + std::size_t(copy) * units_one;
  }
  void set_activation(std::vector<half> const& act) const {
    cuda_ok(cudaMemcpy(activation.ptr, act.data(), act.size() * sizeof(half),
                       cudaMemcpyHostToDevice), "upload activation mode");
  }
};

enum class ArmKind { Pdf, LegacyBC, ShippingBC };

struct Arm {
  ArmKind kind{};
  std::string name;
  std::string config;
  int grid_x = 0, grid_y = 0, threads = 0;
  int barriers = 0;
  int registers = 0;
  std::size_t local_bytes = 0;
  Buffer output;
  std::function<void(DeviceData const&, int, void*)> launch;
  std::vector<double> samples;
};

template <int CTA_N, int WARPS_N, int WARPS_K>
void add_pdf(std::vector<Arm>& arms) {
  Arm a;
  a.kind = ArmKind::Pdf;
  a.config = "CtaN" + std::to_string(CTA_N) + "-Wn" + std::to_string(WARPS_N)
           + "-Wk" + std::to_string(WARPS_K);
  a.name = "pdf/" + a.config;
  a.grid_x = kM; a.grid_y = kN / (CTA_N * WARPS_N);
  a.threads = WARPS_N * WARPS_K * 32;
  a.barriers = WARPS_K == 1 ? 1 : 2;
  a.output = Buffer(std::size_t(kN) * sizeof(half));
  cudaFuncAttributes attr{};
  cuda_ok(cudaFuncGetAttributes(&attr,
      reinterpret_cast<void const*>(q4k_gemv::q4k_gemv_kernel<CTA_N, WARPS_N, WARPS_K>)),
      "cudaFuncGetAttributes PDF");
  a.registers = attr.numRegs; a.local_bytes = attr.localSizeBytes;
  a.launch = [](DeviceData const& d, int copy, void* output) {
    q4k_gemv::launch_q4k_gemv<CTA_N, WARPS_N, WARPS_K>(
        d.activation.as<half>(), d.raw_at(copy), static_cast<half*>(output), kM, kN, kK);
  };
  arms.push_back(std::move(a));
}

template <int RowsPerWarp, int Threads>
void add_bc(std::vector<Arm>& arms) {
  Arm a;
  a.kind = ArmKind::LegacyBC;
  a.config = "A64-RPW" + std::to_string(RowsPerWarp) + "-T" + std::to_string(Threads);
  a.name = "bc/" + a.config;
  a.grid_x = gguf_scale::vecdot::vecdot_grid_size<gguf_scale::KType::Q4_K, RowsPerWarp>(kN, Threads);
  a.grid_y = 1; a.threads = Threads; a.barriers = 0;
  a.output = Buffer(std::size_t(kN) * sizeof(float));
  cudaFuncAttributes attr{};
  cuda_ok(cudaFuncGetAttributes(&attr,
      reinterpret_cast<void const*>(gguf_scale::bc_vecdot::rows_kernel<
          gguf_scale::KType::Q4_K, kArtifactTileK, RowsPerWarp, false>)),
      "cudaFuncGetAttributes BC");
  a.registers = attr.numRegs; a.local_bytes = attr.localSizeBytes;
  a.launch = [](DeviceData const& d, int copy, void* output) {
    constexpr int bpr = kK / 256;
    int const grid = gguf_scale::vecdot::vecdot_grid_size<gguf_scale::KType::Q4_K, RowsPerWarp>(kN, Threads);
    gguf_scale::bc_vecdot::rows_kernel<gguf_scale::KType::Q4_K, kArtifactTileK, RowsPerWarp, false>
        <<<grid, Threads>>>(d.low_at(copy), nullptr, d.units_at(copy),
                            reinterpret_cast<gguf_scale::vecdot::VecdotActivation const*>(
                                d.activation.ptr),
                            static_cast<float*>(output), kN, bpr, nullptr);
  };
  arms.push_back(std::move(a));
}

template <int CTA_N, int WARPS_N>
void add_shipping(std::vector<Arm>& arms) {
  Arm a;
  a.kind = ArmKind::ShippingBC;
  a.config = "A64-CtaN" + std::to_string(CTA_N) + "-Wn" + std::to_string(WARPS_N) + "-Wk1";
  a.name = "shipping/" + a.config;
  a.grid_x = kM; a.grid_y = (kN + CTA_N * WARPS_N - 1) / (CTA_N * WARPS_N);
  a.threads = WARPS_N * 32; a.barriers = 1;
  a.output = Buffer(std::size_t(kN) * sizeof(float));
  cudaFuncAttributes attr{};
  cuda_ok(cudaFuncGetAttributes(&attr,
      reinterpret_cast<void const*>(gguf_scale::bc_q4_gemv::kernel<CTA_N, WARPS_N, 1>)),
      "cudaFuncGetAttributes shipping BC");
  a.registers = attr.numRegs; a.local_bytes = attr.localSizeBytes;
  a.launch = [](DeviceData const& d, int copy, void* output) {
    if constexpr (CTA_N == 2 && WARPS_N == 4) {
      // Exercise the public shipping dispatch, not merely the new kernel
      // symbol.  A lost Q4/A64 seam would fall back to the legacy RPW kernel
      // and become a measured regression in this exact arm.
      gguf_scale::bc_vecdot::launch<gguf_scale::KType::Q4_K,
                                     kArtifactTileK, false>(
          d.low_at(copy), nullptr, d.units_at(copy),
          reinterpret_cast<gguf_scale::vecdot::VecdotActivation const*>(
              d.activation.ptr),
          nullptr, static_cast<float*>(output), kN, kK / 256, 1, 1, nullptr);
    } else {
      gguf_scale::bc_q4_gemv::launch<CTA_N, WARPS_N, 1>(
          d.activation.as<half>(), d.low_at(copy), d.units_at(copy),
          static_cast<float*>(output), kM, kN, kK);
    }
  };
  arms.push_back(std::move(a));
}

struct Options {
  std::string arm = "all";
  std::string expect_fail;
  int samples = 11;
  int copies = 0;
  bool plant_bad_bc = false;
  bool correctness_only = false;
};

Options parse_options(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    auto value = [&](char const* flag) -> char const* {
      if (i + 1 >= argc) fail(std::string("missing value after ") + flag);
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--arm")) o.arm = value(argv[i]);
    else if (!std::strcmp(argv[i], "--samples")) o.samples = std::atoi(value(argv[i]));
    else if (!std::strcmp(argv[i], "--copies")) o.copies = std::atoi(value(argv[i]));
    else if (!std::strcmp(argv[i], "--expect-correctness-fail")) o.expect_fail = value(argv[i]);
    else if (!std::strcmp(argv[i], "--plant-bad-bc-artifact")) o.plant_bad_bc = true;
    else if (!std::strcmp(argv[i], "--correctness-only")) o.correctness_only = true;
    else fail(std::string("unknown argument: ") + argv[i]);
  }
  if (o.arm != "all" && o.arm != "both" && o.arm != "pdf" &&
      o.arm != "bc" && o.arm != "shipping")
    fail("--arm must be all/both/pdf/bc/shipping");
  if (o.samples <= 0 || o.copies < 0) fail("samples must be positive and copies nonnegative");
  if (!o.expect_fail.empty() && o.expect_fail != "pdf" &&
      o.expect_fail != "bc" && o.expect_fail != "shipping")
    fail("--expect-correctness-fail must be pdf, bc, or shipping");
  return o;
}

std::vector<Arm> make_arms(Options const& o) {
  std::vector<Arm> arms;
  if (o.arm == "all" || o.arm == "both" || o.arm == "pdf") {
    add_pdf<1, 4, 1>(arms); add_pdf<2, 4, 1>(arms); add_pdf<4, 4, 1>(arms);
    add_pdf<8, 4, 1>(arms); add_pdf<1, 8, 1>(arms); add_pdf<2, 8, 1>(arms);
    add_pdf<4, 8, 1>(arms); add_pdf<8, 8, 1>(arms);
    add_pdf<1, 4, 2>(arms); add_pdf<2, 4, 2>(arms); add_pdf<4, 4, 2>(arms);
    add_pdf<8, 4, 2>(arms); add_pdf<1, 8, 2>(arms); add_pdf<2, 8, 2>(arms);
    add_pdf<4, 8, 2>(arms); add_pdf<8, 8, 2>(arms);
  }
  if (o.arm == "all" || o.arm == "both" || o.arm == "bc") {
    add_bc<1, 64>(arms);  add_bc<1, 128>(arms); add_bc<1, 256>(arms);
    add_bc<2, 64>(arms);  add_bc<2, 128>(arms); add_bc<2, 256>(arms);
    add_bc<4, 64>(arms);  add_bc<4, 128>(arms); add_bc<4, 256>(arms);
    add_bc<8, 64>(arms);  add_bc<8, 128>(arms); add_bc<8, 256>(arms);
  }
  if (o.arm == "all" || o.arm == "both" || o.arm == "shipping") {
    // Keep CTA_N*WARPS_N divisors of this preregistered N.  The PDF topology
    // intentionally omits a tail-column load guard, so a non-divisor would be
    // a different kernel rather than another point on this axis.
    add_shipping<1, 2>(arms); add_shipping<2, 2>(arms);
    add_shipping<4, 2>(arms); add_shipping<8, 2>(arms);
    add_shipping<1, 4>(arms); add_shipping<2, 4>(arms);
    add_shipping<4, 4>(arms); add_shipping<8, 4>(arms);
    add_shipping<1, 8>(arms); add_shipping<2, 8>(arms);
    add_shipping<4, 8>(arms); add_shipping<8, 8>(arms);
  }
  if (o.arm == "pdf") {
    std::printf("SKIP arm=bc reason=operator-selected-pdf-only\n");
    std::printf("SKIP arm=shipping reason=operator-selected-pdf-only\n");
  }
  if (o.arm == "bc") {
    std::printf("SKIP arm=pdf reason=operator-selected-bc-only\n");
    std::printf("SKIP arm=shipping reason=operator-selected-bc-only\n");
  }
  if (o.arm == "shipping") {
    std::printf("SKIP arm=pdf reason=operator-selected-shipping-only\n");
    std::printf("SKIP arm=bc reason=operator-selected-shipping-only\n");
  }
  return arms;
}

struct ErrorStats {
  double rel_l2 = 0.0;
  double max_abs = 0.0;
  double max_conditioned = 0.0;
  std::size_t nonfinite = 0;
};

ErrorStats error_stats(std::vector<float> const& got, std::vector<float> const& want) {
  long double diff2 = 0.0L, want2 = 0.0L;
  ErrorStats s;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (!std::isfinite(got[i])) { ++s.nonfinite; continue; }
    double const d = std::fabs(double(got[i]) - double(want[i]));
    diff2 += static_cast<long double>(d) * static_cast<long double>(d);
    want2 += static_cast<long double>(want[i]) * static_cast<long double>(want[i]);
    s.max_abs = std::max(s.max_abs, d);
    s.max_conditioned = std::max(s.max_conditioned,
        d / std::max(1.0e-6, std::fabs(double(want[i]))));
  }
  s.rel_l2 = std::sqrt(double(diff2 / std::max(want2, 1.0e-30L)));
  return s;
}

std::vector<float> download_output(Arm const& arm) {
  std::vector<float> got(kN);
  if (arm.kind == ArmKind::Pdf) {
    std::vector<half> h(kN);
    cuda_ok(cudaMemcpy(h.data(), arm.output.ptr, h.size() * sizeof(half), cudaMemcpyDeviceToHost),
            "download PDF output");
    for (int i = 0; i < kN; ++i) got[std::size_t(i)] = __half2float(h[std::size_t(i)]);
  } else {
    cuda_ok(cudaMemcpy(got.data(), arm.output.ptr, got.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "download BC output");
  }
  return got;
}

bool correctness_gate(DeviceData const& d, std::vector<RawBlock> const& raw,
                      std::vector<Arm>& arms, std::string const& expected_failure) {
  std::size_t expected_checks = 0;
  std::size_t expected_red = 0;
  std::set<std::string> expected_configs;
  std::set<std::string> red_configs;
  for (InputMode mode : {InputMode::Positive, InputMode::Signed}) {
    std::vector<half> const act = make_activation(mode);
    std::vector<float> const golden = official_golden(raw, act);
    d.set_activation(act);
    for (Arm& arm : arms) {
      cuda_ok(cudaMemset(arm.output.ptr, 0xa5, arm.output.bytes), "poison correctness output");
      arm.launch(d, 0, arm.output.ptr);
      cuda_ok(cudaGetLastError(), "correctness launch");
      cuda_ok(cudaDeviceSynchronize(), "correctness synchronize");
      ErrorStats const e = error_stats(download_output(arm), golden);
      double const limit = mode == InputMode::Positive ? 0.01 : 0.125;
      bool const pass = e.nonfinite == 0 && e.rel_l2 <= limit;
      char const* input = mode == InputMode::Positive ? "positive" : "signed";
      std::printf("ACCURACY arm=%s input=%s rel_l2=%.9g max_abs=%.9g max_conditioned=%.9g "
                  "nonfinite=%zu limit=%.6g verdict=%s\n",
                  arm.name.c_str(), input, e.rel_l2, e.max_abs, e.max_conditioned,
                  e.nonfinite, limit, pass ? "PASS" : "FAIL");
      std::string const kind = arm.kind == ArmKind::Pdf ? "pdf" :
                               arm.kind == ArmKind::LegacyBC ? "bc" : "shipping";
      if (kind == expected_failure) {
        ++expected_checks;
        expected_red += !pass;
        expected_configs.insert(arm.name);
        if (!pass) red_configs.insert(arm.name);
      } else if (!pass) {
        fail("unexpected correctness failure in " + arm.name + "/" + input);
      }
    }
  }
  if (!expected_failure.empty() && (expected_checks == 0 ||
      red_configs.size() != expected_configs.size()))
    fail("planted " + expected_failure + " correctness defect escaped in " +
         std::to_string(expected_configs.size() - red_configs.size()) + "/" +
         std::to_string(expected_configs.size()) + " configs");
  if (!expected_failure.empty()) {
    std::printf("NEGATIVE-CONTROL target=%s checks=%zu red=%zu configs=%zu red_configs=%zu "
                "verdict=EXPECTED-RED/PASS\n", expected_failure.c_str(), expected_checks,
                expected_red, expected_configs.size(), red_configs.size());
    return false;
  }
  return true;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  std::size_t const mid = values.size() / 2;
  return values.size() & 1 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

void time_arms(DeviceData const& d, std::vector<Arm>& arms, int samples) {
  std::vector<half> const act = make_activation(InputMode::Positive);
  d.set_activation(act);
  for (Arm& arm : arms) {
    for (int i = 0; i < 5; ++i) arm.launch(d, 0, arm.output.ptr);
  }
  cuda_ok(cudaDeviceSynchronize(), "timing warmup");
  for (int pass = 0; pass < samples; ++pass) {
    for (int rank = 0; rank < int(arms.size()); ++rank) {
      int const ai = (pass & 1) ? int(arms.size()) - 1 - rank : rank;
      Arm& arm = arms[std::size_t(ai)];
      cudaEvent_t start{}, stop{};
      cuda_ok(cudaEventCreate(&start), "create start event");
      cuda_ok(cudaEventCreate(&stop), "create stop event");
      cuda_ok(cudaEventRecord(start), "record start event");
      for (int copy = 0; copy < d.copies; ++copy) arm.launch(d, copy, arm.output.ptr);
      cuda_ok(cudaEventRecord(stop), "record stop event");
      cuda_ok(cudaEventSynchronize(stop), "synchronize stop event");
      float ms = 0.f;
      cuda_ok(cudaEventElapsedTime(&ms, start, stop), "elapsed time");
      arm.samples.push_back(double(ms) * 1000.0 / d.copies);
      cudaEventDestroy(start); cudaEventDestroy(stop);
    }
  }
}

void report(std::vector<Arm> const& arms, double peak_gbs,
            std::size_t weight_bytes, std::size_t total_bytes) {
  Arm const* pdf_winner = nullptr;
  Arm const* bc_winner = nullptr;
  Arm const* shipping_winner = nullptr;
  double pdf_us = std::numeric_limits<double>::infinity();
  double bc_us = std::numeric_limits<double>::infinity();
  double shipping_us = std::numeric_limits<double>::infinity();
  for (Arm const& arm : arms) {
    double const us = median(arm.samples);
    // Primary numerator matches the user-supplied PDF target: stored Q4_K
    // weight bytes only. A+D are printed separately and never smuggled into
    // the 1218 GB/s comparison.
    double const gbs = double(weight_bytes) / us / 1000.0;
    double const total_gbs = double(total_bytes) / us / 1000.0;
    double const pct = 100.0 * gbs / peak_gbs;
    std::printf("RESULT arm=%s median_us=%.6f weight_gbs=%.3f total_gbs=%.3f pct_hbm=%.3f regs=%d "
                "local_bytes_per_thread=%zu barriers=%d grid=(%d,%d,1) threads=%d samples=%zu\n",
                arm.name.c_str(), us, gbs, total_gbs, pct, arm.registers, arm.local_bytes,
                arm.barriers, arm.grid_x, arm.grid_y, arm.threads, arm.samples.size());
    if (arm.kind == ArmKind::Pdf && us < pdf_us) { pdf_us = us; pdf_winner = &arm; }
    if (arm.kind == ArmKind::LegacyBC && us < bc_us) { bc_us = us; bc_winner = &arm; }
    if (arm.kind == ArmKind::ShippingBC && us < shipping_us) {
      shipping_us = us; shipping_winner = &arm;
    }
  }
  if (pdf_winner)
    std::printf("WINNER family=pdf arm=%s median_us=%.6f\n", pdf_winner->name.c_str(), pdf_us);
  if (bc_winner)
    std::printf("WINNER family=bc arm=%s median_us=%.6f\n", bc_winner->name.c_str(), bc_us);
  if (shipping_winner)
    std::printf("WINNER family=shipping arm=%s median_us=%.6f\n",
                shipping_winner->name.c_str(), shipping_us);
  if (pdf_winner && bc_winner) {
    std::printf("VERDICT legacy_bc_vs_pdf ratio=%.6f winner=%s target_pdf_us=7.748\n",
                bc_us / pdf_us, bc_us < pdf_us ? "legacy-bc" : "pdf-reference");
  }
  if (pdf_winner && shipping_winner) {
    std::printf("VERDICT production_bc_vs_pdf ratio=%.6f winner=%s target_pdf_us=7.748\n",
                shipping_us / pdf_us,
                shipping_us < pdf_us ? "shipping-bc" : "pdf-reference");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options const options = parse_options(argc, argv);
    int device = 0;
    cuda_ok(cudaSetDevice(device), "cudaSetDevice");
    cudaDeviceProp prop{};
    cuda_ok(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    if (prop.major != 12 || prop.minor != 0)
      fail("benchmark is preregistered for sm_120 / RTX 5090");
    int driver = 0;
    cuda_ok(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
    double const peak_gbs = 2.0 * double(prop.memoryClockRate) * 1000.0
                          * double(prop.memoryBusWidth) / 8.0 / 1.0e9;
    if (!(peak_gbs > 0.0)) fail("CUDA reported no usable memory clock/bus width");

    q4k_pdf_ab::Shape const shape{"D-176", 1, kN, kK, 2, 8, 1, true};
    q4k_pdf_ab::HostProblem problem = q4k_pdf_ab::make_problem(shape);
    Artifact const artifact = make_artifact(problem.raw, options.plant_bad_bc);
    std::size_t const raw_bytes = problem.raw.size() * sizeof(RawBlock);
    std::size_t const artifact_bytes = artifact.low.size() + artifact.units.size();
    std::size_t const bytes_per_workload = raw_bytes + std::size_t(kK) * sizeof(half)
                                         + std::size_t(kN) * sizeof(float);
    int copies = options.copies;
    if (!copies) {
      double const need = 2.16 * double(prop.l2CacheSize);
      copies = std::max(3, int(std::ceil(need / double(std::max(raw_bytes, artifact_bytes)))));
    }
    if (std::size_t(copies) * std::max(raw_bytes, artifact_bytes)
        <= 2 * std::size_t(prop.l2CacheSize))
      fail("cold rotation does not exceed 2x L2");

    std::vector<half> const initial_act = make_activation(InputMode::Positive);
    DeviceData device_data(problem.raw, artifact, initial_act, copies);
    std::vector<Arm> arms = make_arms(options);
    std::printf("META schema=q4k-pdf-vs-bc-v1 device=%s cc=%d.%d driver=%d "
                "memory_clock_khz=%d memory_bus_bits=%d peak_gbs=%.3f l2_bytes=%d "
                "copies=%d cold_raw_bytes=%zu cold_artifact_bytes=%zu cold_multiple=%.6f "
                "M=%d N=%d K=%d artifact_tile_k=%d qtype=%d\n",
                prop.name, prop.major, prop.minor, driver, prop.memoryClockRate,
                prop.memoryBusWidth, peak_gbs, prop.l2CacheSize, copies, raw_bytes,
                artifact_bytes, double(copies) * std::max(raw_bytes, artifact_bytes) / prop.l2CacheSize,
                kM, kN, kK, kArtifactTileK, kQtype);
    std::printf("ARTIFACT low_bytes=%zu packed_unit_bytes=%zu total_bytes=%zu "
                "raw_q4k_bytes=%zu pack_outside_timing=1 roundtrip=exact byte_neutral=1\n",
                artifact.low.size(), artifact.units.size(), artifact_bytes, raw_bytes);

    if (!correctness_gate(device_data, problem.raw, arms, options.expect_fail)) return 0;
    if (options.correctness_only) {
      std::printf("CORRECTNESS-ONLY verdict=PASS\n");
      return 0;
    }
    time_arms(device_data, arms, options.samples);
    report(arms, peak_gbs, raw_bytes, bytes_per_workload);
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "q4k_pdf_vs_bc_5090: %s\n", e.what());
    return 1;
  }
}
