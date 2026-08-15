/***************************************************************************************************
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * One-row PPU benchmark for the dense fixed Split-K parallel path.
 *
 * This executable is deliberately independent of the dense tactic registry.  It instantiates one
 * explicit M==1 packed-A proof row, runs S={1,2,4,8}, and compares every result bit against one
 * exact-by-construction host golden.  It is not the unresolved winner behind the historical 17 us
 * measurement.  S==1 reaches the historical shipping launcher through
 * dense_splitk_parallel_ppu::generic_launcher; S>1 reaches the FP32 partial producer and ordered
 * reduction through that same public handle.
 **************************************************************************************************/

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

#include "cutlass/util/device_memory.h"
#include "dense_splitk_parallel_ppu.cuh"
#include "helper.h"
#include "ppu_group_schedule.hpp"
#include "unfused_weight_dequantize.hpp"
#include "xplane_offline.hpp"

namespace {

using namespace cute;
using half_t = cutlass::half_t;
using int4_t = cutlass::int4b_t;
using QuantMode = fpa_intb_ppu::QuantMode;

constexpr int kM = 1;
constexpr int kN = 4096;
constexpr int kK = 4096;
constexpr int kGroupSize = 128;
constexpr int kScaleK = kK / kGroupSize;
constexpr int kDefaultIterations = 20;
constexpr int kWarmups = 3;
constexpr int kCorrectnessRepeats = 8;
constexpr std::size_t kWorkspaceGuardBytes = 256;
constexpr unsigned char kWorkspaceCanary = 0xa5;
constexpr std::size_t kOutputGuardElements = 64;
constexpr uint16_t kOutputLeftCanary = 0x3555;
constexpr uint16_t kOutputRightCanary = 0x3aaa;
constexpr uint16_t kOutputPoison = 0x7e00;

using BaseSchedule = ppu_group_schedule::FinegrainedSchedule<kGroupSize>;
using Tile = Shape<_8, _128, _128>;
using ScaleTile = Shape<_128, _1>;
using Warp = Shape<_8, _32, _128>;
using Shipping = fpa_intb_ppu::DensePackedAKernelTypes<
    1, QuantMode::FinegrainedScaleZero, BaseSchedule, Tile, ScaleTile, Warp,
    3, true, int4_t, 128>;
using Split = dense_splitk_parallel_ppu::KernelTypes<Shipping, Tile, Warp>;
using ExpectedShippingKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, typename Shipping::CollectiveMainloop,
    typename Shipping::CollectiveEpilogue,
    cutlass::gemm::SplitKSerialScheduler>;

static_assert(std::is_same_v<typename Shipping::GemmKernel,
                             ExpectedShippingKernel>,
              "S=1 must retain the historical M==1 packed-A kernel type");
static_assert(std::is_same_v<typename Split::CollectiveMainloop,
                             typename Shipping::CollectiveMainloop>,
              "parallel Split-K must reuse the shipping mainloop verbatim");
static_assert(size<0>(typename Shipping::CollectiveMainloop::TiledMma::AtomShape_MNK{}) == 8 &&
                  size<1>(typename Shipping::CollectiveMainloop::TiledMma::AtomShape_MNK{}) == 16 &&
                  size<2>(typename Shipping::CollectiveMainloop::TiledMma::AtomShape_MNK{}) == 16,
              "the fixed benchmark must compile the M==1 m8n16k16 atom");
static_assert(Split::BlockM == 8 && Split::BlockN == 128 &&
                  Shipping::MainloopPolicy::TacticTileK == 128 &&
                  Shipping::MainloopPolicy::ArtifactTileK == 128 &&
                  Shipping::CollectiveMainloop::DispatchPolicy::StaticGroupSize == kGroupSize,
              "the benchmark type drifted from its explicit packed-A proof row");
static_assert(Shipping::MainloopPolicy::PackedARows == 1,
              "the M==1 canary must exercise the production packed-A provider");
static_assert(std::is_same_v<typename Split::CollectivePartialEpilogue::ElementD,
                             float>,
              "parallel peers must store FP32 partial planes");
static_assert(kK % 128 == 0 && (kK / 128) % 8 == 0,
              "the fixed shape must admit every registered split count without a ragged K stripe");

struct Fixture {
  std::vector<half_t> a;
  std::vector<int8_t> resident_b;
  std::vector<half_t> scales;
  std::vector<half_t> zeros;
  std::vector<half_t> golden;
  int max_nonzeros = 0;
  int max_abs_q = 0;
  int max_scale = 0;
  int max_abs_zero = 0;
  int max_output = 0;
  bool order_independent = false;
  bool fp16_exact = false;
  bool artifact_ok = false;
};

struct EventPair {
  hggcEvent_t start{};
  hggcEvent_t stop{};

  EventPair() {
    CUTLASS_PPU_CHECK(hggcEventCreate(&start));
    CUTLASS_PPU_CHECK(hggcEventCreate(&stop));
  }
  ~EventPair() {
    if (start) hggcEventDestroy(start);
    if (stop) hggcEventDestroy(stop);
  }
  EventPair(EventPair const&) = delete;
  EventPair& operator=(EventPair const&) = delete;
};

uint64_t fnv1a_half_bits(std::vector<half_t> const& values) {
  uint64_t hash = 1469598103934665603ull;
  for (half_t const value : values) {
    uint16_t const bits = value.raw();
    hash ^= uint8_t(bits);
    hash *= 1099511628211ull;
    hash ^= uint8_t(bits >> 8);
    hash *= 1099511628211ull;
  }
  return hash;
}

int signed_q_at(int k, int n) {
  // The full logical plane is asymmetric.  Only one K in each gs128 group is
  // consumed by A, but changing any K/N placement still changes the resident
  // bytes and therefore keeps the xplane writer in the tested path.
  return ((5 * k + 3 * n + 1) & 15) - 8;
}

Fixture make_fixture() {
  Fixture fixture;
  fixture.a.assign(std::size_t(kM) * kK, half_t(0.0f));
  fixture.scales.resize(std::size_t(kScaleK) * kN);
  fixture.zeros.resize(std::size_t(kScaleK) * kN);
  fixture.golden.resize(std::size_t(kM) * kN);

  std::vector<int> active_k(kScaleK);
  for (int kg = 0; kg < kScaleK; ++kg) {
    int const k = kg * kGroupSize + ((29 * kg + 17) % kGroupSize);
    active_k[kg] = k;
    fixture.a[std::size_t(k)] = half_t(1.0f);
  }

  for (int kg = 0; kg < kScaleK; ++kg) {
    for (int n = 0; n < kN; ++n) {
      int const scale = 1 << ((5 * kg + 3 * n) % 3);  // 1, 2, or 4.
      fixture.scales[std::size_t(kg) * kN + n] = half_t(float(scale));
      fixture.max_scale = std::max(fixture.max_scale, scale);
      int const zero = 3 * (((7 * kg + 5 * n) % 3) - 1);  // -3, 0, or 3.
      fixture.zeros[std::size_t(kg) * kN + n] = half_t(float(zero));
      fixture.max_abs_zero = std::max(fixture.max_abs_zero, std::abs(zero));
    }
  }

  // The explicit type-owned writer consumes biased canonical [K,N] codes.
  // Independently, preprocess_weights_for_mixed_gemm consumes packed [N,K]
  // signed-int4 nibbles and applies the shipping +8 bias.  Require the two
  // resident byte maps to agree, then require the type-owned inverse to recover
  // every canonical code.  This binds the canary to both its declared
  // placement and the shipping preprocessing entry point; it does not claim
  // that this row won a sweep.
  std::vector<uint8_t> canonical(std::size_t(kK) * kN);
  std::vector<int8_t> packed(std::size_t(kK) * kN / 2, int8_t(0));
  for (int n = 0; n < kN; ++n) {
    for (int k = 0; k < kK; ++k) {
      int const q = signed_q_at(k, n);
      fixture.max_abs_q = std::max(fixture.max_abs_q, std::abs(q));
      canonical[std::size_t(k) * kN + n] = uint8_t((q + 8) & 0xf);
      std::size_t const linear = std::size_t(n) * kK + k;
      uint8_t* byte = reinterpret_cast<uint8_t*>(packed.data()) + linear / 2;
      *byte |= uint8_t(q & 0xf) << (4 * (linear & 1));
    }
  }
  fixture.resident_b.resize(packed.size());
  xplane::place_derived<4, 8, 128, 128, 8, 32, 1, 128>(
      fixture.resident_b.data(), canonical, kN, kK);
  std::vector<int8_t> shipping_resident(packed.size());
  preprocess_weights_for_mixed_gemm<false, 256>(
      shipping_resident.data(), packed.data(),
      {std::size_t(kK), std::size_t(kN)},
      QuantTypeClass::PACKED_INT4_WEIGHT_ONLY);
  std::size_t shipping_byte_diff = 0;
  for (std::size_t i = 0; i < shipping_resident.size(); ++i) {
    shipping_byte_diff += shipping_resident[i] != fixture.resident_b[i];
  }
  std::vector<uint8_t> recovered(canonical.size(), uint8_t(0xff));
  xplane::recover_derived<4, 8, 128, 128, 8, 32, 1, 128>(
      fixture.resident_b.data(), recovered, kN, kK);
  std::size_t roundtrip_bad = 0;
  for (std::size_t i = 0; i < canonical.size(); ++i) {
    roundtrip_bad += canonical[i] != recovered[i];
  }
  fixture.artifact_ok = shipping_byte_diff == 0 && roundtrip_bad == 0;
  std::printf(
      "[splitk artifact] placement=xplane-int4-tm8-tn128-tk128-wm8-wn32-f1 "
      "bytes=%zu shipping_byte_diff=%zu/%zu roundtrip_bad=%zu/%zu -> %s\n",
      fixture.resident_b.size(), shipping_byte_diff, shipping_resident.size(),
      roundtrip_bad, canonical.size(), fixture.artifact_ok ? "PASS" : "FAIL");

  fixture.max_nonzeros = kScaleK;
  for (int n = 0; n < kN; ++n) {
    int sum = 0;
    for (int kg = 0; kg < kScaleK; ++kg) {
      int const q = signed_q_at(active_k[kg], n);
      std::size_t const metadata = std::size_t(kg) * kN + n;
      int const scale = int(float(fixture.scales[metadata]));
      int const zero = int(float(fixture.zeros[metadata]));
      sum += q * scale + zero;
    }
    fixture.max_output = std::max(fixture.max_output, std::abs(sum));
    fixture.golden[std::size_t(n)] = half_t(float(sum));
  }

  int const conservative_bound = fixture.max_nonzeros *
      (fixture.max_abs_q * fixture.max_scale + fixture.max_abs_zero);
  fixture.order_independent = conservative_bound < (1 << 24);
  fixture.fp16_exact = fixture.order_independent && conservative_bound < (1 << 11);
  std::printf(
      "[splitk fixture exactness] fixture=dense-m1-int4-gs128 shape=%dx%dx%d "
      "nonzeros/row=%d integer_A=1 integer_weights=1 max|q|=%d max_scale=%d max|zero|=%d "
      "max|D|=%d conservative_bound=%d vs fp32=2^24 fp16=2^11 -> %s\n",
      kM, kN, kK, fixture.max_nonzeros, fixture.max_abs_q,
      fixture.max_scale, fixture.max_abs_zero, fixture.max_output,
      conservative_bound,
      fixture.order_independent && fixture.fp16_exact
          ? "ORDER-INDEPENDENT+FP16-EXACT"
          : "ROUNDS/INVALID");
  return fixture;
}

bool parse_iterations(int argc, char** argv, int& iterations) {
  iterations = kDefaultIterations;
  if (argc == 1) return true;
  if (argc != 2) return false;
  char const prefix[] = "--iterations=";
  if (std::strncmp(argv[1], prefix, sizeof(prefix) - 1) != 0) return false;
  char* end = nullptr;
  long const value = std::strtol(argv[1] + sizeof(prefix) - 1, &end, 10);
  if (end == argv[1] + sizeof(prefix) - 1 || *end != '\0' || value <= 0 ||
      value > 10000) {
    return false;
  }
  iterations = int(value);
  return true;
}

bool launch(int splits, half_t const* a, uint8_t const* resident_b,
            half_t const* scales, half_t const* zeros, half_t* d,
            char* workspace, std::size_t workspace_bytes) {
  return dense_splitk_parallel_ppu::generic_launcher<
      QuantMode::FinegrainedScaleZero, BaseSchedule, Tile, ScaleTile, Warp, 3,
      true, int4_t, void, 128, Shipping>(
          a, reinterpret_cast<int4_t const*>(resident_b), scales, zeros, d,
          kM, kN, kK, kGroupSize, splits,
          workspace, workspace_bytes, nullptr, nullptr);
}

bool workspace_outside_plan_is_canary(
    std::vector<char> const& bytes, std::size_t plan_bytes) {
  if (bytes.size() < 2 * kWorkspaceGuardBytes + plan_bytes) return false;
  auto is_canary = [](char value) {
    return static_cast<unsigned char>(value) == kWorkspaceCanary;
  };
  return std::all_of(bytes.begin(),
                     bytes.begin() + std::ptrdiff_t(kWorkspaceGuardBytes),
                     is_canary) &&
      std::all_of(bytes.begin() +
                      std::ptrdiff_t(kWorkspaceGuardBytes + plan_bytes),
                  bytes.end(), is_canary);
}

std::vector<half_t> poisoned_output_storage() {
  std::vector<half_t> storage(
      2 * kOutputGuardElements + std::size_t(kM) * kN,
      half_t::bitcast(kOutputPoison));
  std::fill(storage.begin(),
            storage.begin() + std::ptrdiff_t(kOutputGuardElements),
            half_t::bitcast(kOutputLeftCanary));
  std::fill(storage.end() - std::ptrdiff_t(kOutputGuardElements),
            storage.end(), half_t::bitcast(kOutputRightCanary));
  return storage;
}

bool output_guards_are_intact(std::vector<half_t> const& storage) {
  if (storage.size() !=
      2 * kOutputGuardElements + std::size_t(kM) * kN) {
    return false;
  }
  for (std::size_t i = 0; i < kOutputGuardElements; ++i) {
    if (storage[i].raw() != kOutputLeftCanary ||
        storage[storage.size() - kOutputGuardElements + i].raw() !=
            kOutputRightCanary) {
      return false;
    }
  }
  return true;
}

bool run_one_split(
    int splits, int iterations, Fixture const& fixture,
    cutlass::DeviceAllocation<half_t>& d_a,
    cutlass::DeviceAllocation<uint8_t>& d_b,
    cutlass::DeviceAllocation<half_t>& d_scales,
    cutlass::DeviceAllocation<half_t>& d_zeros,
    cutlass::DeviceAllocation<half_t>& d_output_storage,
    cutlass::DeviceAllocation<char>& d_workspace,
    std::size_t max_workspace_bytes, uint64_t golden_fingerprint,
    uint64_t& output_fingerprint) {
  dense_splitk_parallel_ppu::WorkspacePlan plan;
  if (!dense_splitk_parallel_ppu::query_workspace_plan(
          kM, kN, splits, plan)) {
    std::fprintf(stderr, "[splitk S=%d] workspace query rejected a registered split\n", splits);
    return false;
  }
  std::size_t const expected_workspace =
      splits == 1 ? 0 : std::size_t(splits) * kM * kN * sizeof(float);
  if (plan.partial_bytes != expected_workspace || plan.alignment != 16 ||
      plan.partial_bytes > max_workspace_bytes) {
    std::fprintf(stderr,
                 "[splitk S=%d] workspace contract drifted: got=%zu align=%zu expected=%zu\n",
                 splits, plan.partial_bytes, plan.alignment, expected_workspace);
    return false;
  }

  constexpr uint32_t kTiles = kK / 128;
  auto const partition = cutlass::gemm::kernel::fixed_splitk::make_params(
      uint64_t((kM + Split::BlockM - 1) / Split::BlockM) *
          uint64_t((kN + Split::BlockN - 1) / Split::BlockN),
      kTiles, uint32_t(splits));
  if (!partition.is_valid()) {
    std::fprintf(stderr, "[splitk S=%d] fixed partition is invalid\n", splits);
    return false;
  }

  char* workspace = splits == 1
      ? nullptr
      : d_workspace.get() + kWorkspaceGuardBytes;
  std::vector<half_t> output_storage = poisoned_output_storage();
  d_output_storage.copy_from_host(output_storage.data());
  half_t* const output_ptr = d_output_storage.get() + kOutputGuardElements;
  CUTLASS_PPU_CHECK(hggcMemset(d_workspace.get(), kWorkspaceCanary,
                               d_workspace.size() * sizeof(char)));
  if (!launch(splits, d_a.get(), d_b.get(), d_scales.get(), d_zeros.get(),
              output_ptr, workspace, plan.partial_bytes)) {
    std::fprintf(stderr, "[splitk S=%d] production launcher rejected the fixed row\n", splits);
    return false;
  }
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());

  d_output_storage.copy_to_host(output_storage.data());
  std::vector<half_t> output(
      output_storage.begin() + std::ptrdiff_t(kOutputGuardElements),
      output_storage.begin() +
          std::ptrdiff_t(kOutputGuardElements + std::size_t(kM) * kN));
  bool output_guard_ok = output_guards_are_intact(output_storage);
  uint64_t raw_bad = 0;
  for (std::size_t i = 0; i < output.size(); ++i) {
    if (output[i].raw() != fixture.golden[i].raw()) {
      if (raw_bad < 8) {
        std::fprintf(stderr,
                     "[splitk S=%d mismatch] out=%zu got=%g/0x%04x want=%g/0x%04x\n",
                     splits, i, double(float(output[i])), unsigned(output[i].raw()),
                     double(float(fixture.golden[i])),
                     unsigned(fixture.golden[i].raw()));
      }
      ++raw_bad;
    }
  }
  output_fingerprint = fnv1a_half_bits(output);

  std::vector<char> workspace_host(d_workspace.size());
  d_workspace.copy_to_host(workspace_host.data());
  bool workspace_guard_ok = workspace_outside_plan_is_canary(
      workspace_host, plan.partial_bytes);

  // Correctness is checked after every launch, not inferred from the last
  // launch in a timing loop.  Re-poisoning both allocations makes a skipped
  // producer/reducer unable to inherit a previous correct result.
  bool repeated_ok = raw_bad == 0 && output_guard_ok && workspace_guard_ok &&
      output_fingerprint == golden_fingerprint;
  for (int repeat = 1; repeat < kCorrectnessRepeats; ++repeat) {
    output_storage = poisoned_output_storage();
    d_output_storage.copy_from_host(output_storage.data());
    CUTLASS_PPU_CHECK(hggcMemset(d_workspace.get(), kWorkspaceCanary,
                                 d_workspace.size() * sizeof(char)));
    if (!launch(splits, d_a.get(), d_b.get(), d_scales.get(), d_zeros.get(),
                output_ptr, workspace, plan.partial_bytes)) {
      std::fprintf(stderr, "[splitk S=%d] correctness repeat %d failed to launch\n",
                   splits, repeat);
      return false;
    }
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    d_output_storage.copy_to_host(output_storage.data());
    std::vector<half_t> repeat_output(
        output_storage.begin() + std::ptrdiff_t(kOutputGuardElements),
        output_storage.begin() +
            std::ptrdiff_t(kOutputGuardElements + std::size_t(kM) * kN));
    uint64_t repeat_bad = 0;
    for (std::size_t i = 0; i < repeat_output.size(); ++i) {
      repeat_bad += repeat_output[i].raw() != fixture.golden[i].raw();
    }
    uint64_t const repeat_fingerprint = fnv1a_half_bits(repeat_output);
    d_workspace.copy_to_host(workspace_host.data());
    bool const repeat_workspace_guard = workspace_outside_plan_is_canary(
        workspace_host, plan.partial_bytes);
    bool const repeat_output_guard = output_guards_are_intact(output_storage);
    bool const this_repeat_ok = repeat_bad == 0 && repeat_workspace_guard &&
        repeat_output_guard && repeat_fingerprint == output_fingerprint;
    if (!this_repeat_ok) {
      std::fprintf(
          stderr,
          "[splitk S=%d] correctness repeat=%d bad=%llu fingerprint=%016llx "
          "workspace_redzone=%d output_redzone=%d\n",
          splits, repeat, static_cast<unsigned long long>(repeat_bad),
          static_cast<unsigned long long>(repeat_fingerprint),
          int(repeat_workspace_guard), int(repeat_output_guard));
    }
    repeated_ok = repeated_ok && this_repeat_ok;
    workspace_guard_ok = workspace_guard_ok && repeat_workspace_guard;
    output_guard_ok = output_guard_ok && repeat_output_guard;
  }

  std::vector<double> samples;
  samples.reserve(std::size_t(iterations));
  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    if (!launch(splits, d_a.get(), d_b.get(), d_scales.get(), d_zeros.get(),
                output_ptr, workspace, plan.partial_bytes)) {
      std::fprintf(stderr, "[splitk S=%d] warmup launch %d failed\n", splits, warmup);
      return false;
    }
  }
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  for (int iteration = 0; iteration < iterations; ++iteration) {
    EventPair events;
    CUTLASS_PPU_CHECK(hggcEventRecord(events.start, nullptr));
    if (!launch(splits, d_a.get(), d_b.get(), d_scales.get(), d_zeros.get(),
                output_ptr, workspace, plan.partial_bytes)) {
      std::fprintf(stderr, "[splitk S=%d] timed launch %d failed\n", splits, iteration);
      return false;
    }
    CUTLASS_PPU_CHECK(hggcEventRecord(events.stop, nullptr));
    CUTLASS_PPU_CHECK(hggcEventSynchronize(events.stop));
    float elapsed_ms = 0.0f;
    CUTLASS_PPU_CHECK(hggcEventElapsedTime(&elapsed_ms, events.start, events.stop));
    if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f) {
      std::fprintf(stderr, "[splitk S=%d] invalid event duration %g ms\n",
                   splits, double(elapsed_ms));
      return false;
    }
    samples.push_back(double(elapsed_ms) * 1000.0);
  }
  std::sort(samples.begin(), samples.end());
  double const median = samples.size() & 1
      ? samples[samples.size() / 2]
      : 0.5 * (samples[samples.size() / 2 - 1] +
               samples[samples.size() / 2]);
  double const mean =
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();

  d_output_storage.copy_to_host(output_storage.data());
  output_guard_ok = output_guard_ok && output_guards_are_intact(output_storage);
  std::vector<half_t> post_timing_output(
      output_storage.begin() + std::ptrdiff_t(kOutputGuardElements),
      output_storage.begin() +
          std::ptrdiff_t(kOutputGuardElements + std::size_t(kM) * kN));
  uint64_t post_timing_raw_bad = 0;
  for (std::size_t i = 0; i < post_timing_output.size(); ++i) {
    post_timing_raw_bad +=
        post_timing_output[i].raw() != fixture.golden[i].raw();
  }
  uint64_t const post_timing_fingerprint =
      fnv1a_half_bits(post_timing_output);
  d_workspace.copy_to_host(workspace_host.data());
  bool const post_timing_workspace_guard_ok = workspace_outside_plan_is_canary(
      workspace_host, plan.partial_bytes);
  bool const all_guards_ok =
      workspace_guard_ok && post_timing_workspace_guard_ok && output_guard_ok;

  bool const correct = raw_bad == 0 && post_timing_raw_bad == 0 &&
      repeated_ok && all_guards_ok && output_fingerprint == golden_fingerprint &&
      post_timing_fingerprint == output_fingerprint;
  std::printf(
      "[splitk S=%d] grid=%ux%ux%u output_tiles=%llu work_units=%llu "
      "Kt=%u k_tiles_per_peer=%u workspace=%zu raw_half_bad=%llu/%zu "
      "fingerprint=%016llx golden=%016llx post_timing_raw_half_bad=%llu/%zu "
      "post_timing_fingerprint=%016llx correctness_repeats=%d repeated=%s "
      "workspace_redzone=%s output_redzone=%s "
      "e2e_median=%.3f_us mean=%.3f_us min=%.3f_us max=%.3f_us samples=%zu "
      "main_us=NA reduce_us=NA split_timing_seam=not-exposed-by-production-handle -> %s\n",
      splits,
      unsigned((kM + Split::BlockM - 1) / Split::BlockM),
      unsigned((kN + Split::BlockN - 1) / Split::BlockN),
      unsigned(splits),
      static_cast<unsigned long long>(partition.output_tiles),
      static_cast<unsigned long long>(partition.work_units),
      partition.k_tiles_per_output, partition.k_tiles_per_split,
      plan.partial_bytes, static_cast<unsigned long long>(raw_bad), output.size(),
      static_cast<unsigned long long>(output_fingerprint),
      static_cast<unsigned long long>(golden_fingerprint),
      static_cast<unsigned long long>(post_timing_raw_bad),
      post_timing_output.size(),
      static_cast<unsigned long long>(post_timing_fingerprint),
      kCorrectnessRepeats, repeated_ok ? "PASS" : "FAIL",
      (workspace_guard_ok && post_timing_workspace_guard_ok) ? "PASS" : "FAIL",
      output_guard_ok ? "PASS" : "FAIL",
      median, mean, samples.front(), samples.back(),
      samples.size(), correct ? "PASS" : "FAIL");
  return correct;
}

}  // namespace

int main(int argc, char** argv) {
  int iterations = 0;
  if (!parse_iterations(argc, argv, iterations)) {
    std::fprintf(stderr, "usage: %s [--iterations=N]\n", argv[0]);
    return 2;
  }

  Fixture const fixture = make_fixture();
  if (!fixture.artifact_ok || !fixture.order_independent || !fixture.fp16_exact) {
    std::fprintf(stderr,
                 "[splitk] exact fixture does not establish its own arithmetic bounds\n");
    return 1;
  }
  uint64_t const golden_fingerprint = fnv1a_half_bits(fixture.golden);

  cutlass::DeviceAllocation<half_t> d_a(std::size_t(kM) * kK);
  // The resident plane is a byte allocation. DeviceAllocation<int4_t>(codes)
  // allocates by sizeof(T) while typed copies count sizeof_bits<T>; spelling
  // bytes here prevents that historical logical-code/physical-byte ambiguity.
  cutlass::DeviceAllocation<uint8_t> d_b(fixture.resident_b.size());
  cutlass::DeviceAllocation<half_t> d_scales(std::size_t(kScaleK) * kN);
  cutlass::DeviceAllocation<half_t> d_zeros(std::size_t(kScaleK) * kN);
  cutlass::DeviceAllocation<half_t> d_output_storage(
      2 * kOutputGuardElements + std::size_t(kM) * kN);
  dense_splitk_parallel_ppu::WorkspacePlan max_plan;
  if (!dense_splitk_parallel_ppu::query_workspace_plan(kM, kN, 8, max_plan)) {
    std::fprintf(stderr, "[splitk] failed to query the S=8 workspace authority\n");
    return 1;
  }
  cutlass::DeviceAllocation<char> d_workspace(
      2 * kWorkspaceGuardBytes + max_plan.partial_bytes);

  d_a.copy_from_host(fixture.a.data());
  d_b.copy_from_host(reinterpret_cast<uint8_t const*>(fixture.resident_b.data()));
  d_scales.copy_from_host(fixture.scales.data());
  d_zeros.copy_from_host(fixture.zeros.data());

  std::printf(
      "[splitk] target=m1-packedA-canary-row winner_binding=UNRESOLVED "
      "measurement_scope=warm-single-artifact type=ordinary-int4-gs128 "
      "tile=8x128x128 warp=8x32x128 stages=3 artifact_tk=128 "
      "S=1,2,4,8 correctness_repeats=%d iterations=%d warmups=%d "
      "golden_fingerprint=%016llx\n",
      kCorrectnessRepeats, iterations, kWarmups,
      static_cast<unsigned long long>(golden_fingerprint));

  bool all_ok = true;
  uint64_t s1_fingerprint = 0;
  for (int const splits : std::array<int, 4>{1, 2, 4, 8}) {
    uint64_t fingerprint = 0;
    bool const ok = run_one_split(
        splits, iterations, fixture, d_a, d_b, d_scales, d_zeros,
        d_output_storage, d_workspace, max_plan.partial_bytes,
        golden_fingerprint,
        fingerprint);
    if (splits == 1) {
      s1_fingerprint = fingerprint;
    } else if (fingerprint != s1_fingerprint) {
      std::fprintf(stderr,
                   "[splitk S=%d] cross-S fingerprint differs from S=1: %016llx != %016llx\n",
                   splits, static_cast<unsigned long long>(fingerprint),
                   static_cast<unsigned long long>(s1_fingerprint));
      all_ok = false;
    }
    all_ok = all_ok && ok;
  }

  std::printf(
      "[splitk] %s: S=1 M1 packed-A provider and S=2/4/8 parallel paths "
      "share one exact fixture, repeated correctness, and raw-half fingerprint\n",
      all_ok ? "PASS" : "FAIL");
  std::puts(
      "[splitk perf] UNADJUDICATED: this canary row is not the unresolved "
      "17us sweep winner; timings are preliminary warm single-artifact observations");
  return all_ok ? 0 : 1;
}
