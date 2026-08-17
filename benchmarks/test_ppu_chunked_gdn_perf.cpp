/***************************************************************************************************
 * Copyright (c) 2026 quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Standalone PPU performance harness for the shipping chunked-GDN C ABI.
 * Correctness belongs to tests/test_ppu_chunked_gdn_abi.cpp; this executable
 * deliberately owns only fixture allocation, warmup and a device-event span
 * around calls through that ABI. The span includes launch idle introduced by
 * per-call runtime checks; it is reported as an upper bound, not naked kernel
 * time.
 **************************************************************************************************/

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <hggc_runtime.h>

#include "quactlize_ppu_linear_attention.h"

namespace {

constexpr int kChunk = 64;
constexpr int kHeadK = 128;
constexpr int kHeadV = 128;
constexpr int kThreads = 128;
constexpr std::size_t kSharedBytes = 139776;
constexpr float kScale = 0.5f;
constexpr std::uint16_t kBf16Poison = 0x7fc1u;
// Mathematical work performed by one full C64 / Dk128 / Dv128 value-head
// chunk. This is an algorithmic count, not an executed-instruction count.
constexpr std::uint64_t kLogicalFlopsPerFullChunk = 11276288ull;

struct Options {
  int sequences = 4;
  int sequence_length = 256;
  int qk_heads = 16;
  int v_heads = 32;
  int warmup = 5;
  int iterations = 20;
  int samples = 7;
  bool initial_state = true;
  bool final_state = true;
  bool acu = false;
};

bool runtime_ok(hggcError_t status, char const* where) {
  if (status == hggcSuccess) return true;
  std::fprintf(stderr, "[GDN perf] %s failed: %s\n", where,
               hggcGetErrorString(status));
  return false;
}

struct DeviceBuffer {
  void* pointer = nullptr;
  std::size_t bytes = 0;

  explicit DeviceBuffer(std::size_t n) : bytes(n) {
    if (n != 0 && !runtime_ok(hggcMalloc(&pointer, n), "device allocation")) {
      pointer = nullptr;
    }
  }
  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;
  ~DeviceBuffer() {
    if (pointer != nullptr) (void)hggcFree(pointer);
  }

  template <class T>
  T* as() const {
    return static_cast<T*>(pointer);
  }

  bool upload(void const* source) const {
    return pointer != nullptr && runtime_ok(
        hggcMemcpy(pointer, source, bytes, hggcMemcpyHostToDevice), "H2D");
  }
  bool download(void* destination) const {
    return pointer != nullptr && runtime_ok(
        hggcMemcpy(destination, pointer, bytes, hggcMemcpyDeviceToHost), "D2H");
  }
};

struct Events {
  hggcEvent_t start{};
  hggcEvent_t stop{};
  bool valid = false;

  Events() {
    valid = runtime_ok(hggcEventCreate(&start), "start-event create") &&
            runtime_ok(hggcEventCreate(&stop), "stop-event create");
  }
  Events(Events const&) = delete;
  Events& operator=(Events const&) = delete;
  ~Events() {
    if (start != nullptr) (void)hggcEventDestroy(start);
    if (stop != nullptr) (void)hggcEventDestroy(stop);
  }
};

std::uint16_t float_to_bf16(float x) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &x, sizeof(bits));
  std::uint32_t const lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return std::uint16_t(bits >> 16);
}

bool checked_mul(std::size_t a, std::size_t b, std::size_t& out) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
  out = a * b;
  return true;
}

bool elements3(int a, int b, int c, std::size_t& out) {
  if (a <= 0 || b <= 0 || c <= 0) return false;
  std::size_t ab = 0;
  return checked_mul(std::size_t(a), std::size_t(b), ab) &&
         checked_mul(ab, std::size_t(c), out);
}

bool elements4(int a, int b, int c, int d, std::size_t& out) {
  std::size_t abc = 0;
  return elements3(a, b, c, abc) && checked_mul(abc, std::size_t(d), out);
}

bool parse_int(char const* text, int& value) {
  if (text == nullptr || *text == '\0') return false;
  errno = 0;
  char* end = nullptr;
  long const parsed = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) {
    return false;
  }
  value = int(parsed);
  return true;
}

bool consume_int(char const* arg, char const* name, int& value) {
  std::string const prefix = std::string("--") + name + "=";
  if (std::strncmp(arg, prefix.c_str(), prefix.size()) != 0) return false;
  if (!parse_int(arg + prefix.size(), value)) {
    std::fprintf(stderr, "invalid %s: %s\n", name, arg + prefix.size());
    std::exit(2);
  }
  return true;
}

void usage(char const* argv0) {
  std::printf(
      "usage: %s [--sequences=N] [--length=N] [--qk-heads=N] [--v-heads=N] "
      "[--warmup=N] [--iterations=N] [--samples=N] "
      "[--initial-state=0|1] [--final-state=0|1] [--acu]\n",
      argv0);
}

bool parse_options(int argc, char** argv, Options& o) {
  for (int i = 1; i < argc; ++i) {
    char const* arg = argv[i];
    if (std::strcmp(arg, "--help") == 0) {
      usage(argv[0]);
      std::exit(0);
    }
    if (std::strcmp(arg, "--acu") == 0) {
      o.acu = true;
      continue;
    }
    int flag = 0;
    if (consume_int(arg, "sequences", o.sequences) ||
        consume_int(arg, "length", o.sequence_length) ||
        consume_int(arg, "qk-heads", o.qk_heads) ||
        consume_int(arg, "v-heads", o.v_heads) ||
        consume_int(arg, "warmup", o.warmup) ||
        consume_int(arg, "iterations", o.iterations) ||
        consume_int(arg, "samples", o.samples)) {
      continue;
    }
    if (consume_int(arg, "initial-state", flag)) {
      if (flag > 1) return false;
      o.initial_state = flag != 0;
      continue;
    }
    if (consume_int(arg, "final-state", flag)) {
      if (flag > 1) return false;
      o.final_state = flag != 0;
      continue;
    }
    std::fprintf(stderr, "unknown argument: %s\n", arg);
    return false;
  }
  if (o.sequences <= 0 || o.sequence_length <= 0 || o.qk_heads <= 0 ||
      o.v_heads <= 0 || o.v_heads % o.qk_heads != 0 || o.warmup < 0 ||
      o.iterations <= 0 || o.samples <= 0) {
    return false;
  }
  return true;
}

std::uint64_t fnv1a64(void const* data, std::size_t bytes) {
  auto const* p = static_cast<unsigned char const*>(data);
  std::uint64_t hash = 14695981039346656037ull;
  for (std::size_t i = 0; i < bytes; ++i) {
    hash ^= p[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

std::size_t count_nonfinite_bf16(std::vector<std::uint16_t> const& values) {
  std::size_t count = 0;
  for (std::uint16_t value : values) {
    count += (value & 0x7f80u) == 0x7f80u;
  }
  return count;
}

std::size_t count_nonfinite_fp32(std::vector<float> const& values) {
  std::size_t count = 0;
  for (float value : values) count += !std::isfinite(value);
  return count;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  std::size_t const n = values.size();
  return n & 1 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

int run(Options const& o) {
  std::int64_t const tokens64 =
      std::int64_t(o.sequences) * std::int64_t(o.sequence_length);
  if (tokens64 <= 0 || tokens64 > std::numeric_limits<std::int32_t>::max()) {
    std::fprintf(stderr, "[GDN perf] token count does not fit the v1 ABI\n");
    return 2;
  }
  int const tokens = int(tokens64);

  std::size_t qk_elements = 0, vo_elements = 0, gate_elements = 0,
              state_elements = 0;
  if (!elements3(tokens, o.qk_heads, kHeadK, qk_elements) ||
      !elements3(tokens, o.v_heads, kHeadV, vo_elements) ||
      !checked_mul(std::size_t(tokens), std::size_t(o.v_heads), gate_elements) ||
      !elements4(o.sequences, o.v_heads, kHeadK, kHeadV, state_elements)) {
    std::fprintf(stderr, "[GDN perf] fixture extent overflow\n");
    return 2;
  }

  std::vector<std::uint16_t> q(qk_elements), k(qk_elements), v(vo_elements);
  std::vector<float> gamma(gate_elements), beta(gate_elements);
  std::vector<float> initial(state_elements);
  std::vector<std::uint16_t> output(vo_elements, kBf16Poison);
  std::vector<float> final(
      state_elements, std::numeric_limits<float>::quiet_NaN());

  for (std::size_t i = 0; i < q.size(); ++i) {
    q[i] = float_to_bf16(float(int(i % 7) - 3) * 0.015625f);
    k[i] = float_to_bf16(float(int(i % 5) - 2) * 0.03125f);
  }
  for (std::size_t i = 0; i < v.size(); ++i) {
    v[i] = float_to_bf16(float(int(i % 9) - 4) * 0.015625f);
  }
  for (int t = 0; t < tokens; ++t) {
    // gamma is chunk-local within each independent sequence.  Global t%C is
    // wrong whenever sequence_length is not a multiple of C.
    int const sequence_local = t % o.sequence_length;
    int const local = sequence_local % kChunk;
    for (int h = 0; h < o.v_heads; ++h) {
      std::size_t const i = std::size_t(t) * o.v_heads + h;
      gamma[i] = -float(local) * 0.015625f;
      beta[i] = 0.5f;
    }
  }
  for (std::size_t i = 0; i < initial.size(); ++i) {
    initial[i] = float(int(i % 5) - 2) * 0.0009765625f;
  }

  DeviceBuffer dq(q.size() * sizeof(q[0]));
  DeviceBuffer dk(k.size() * sizeof(k[0]));
  DeviceBuffer dv(v.size() * sizeof(v[0]));
  DeviceBuffer dgamma(gamma.size() * sizeof(gamma[0]));
  DeviceBuffer dbeta(beta.size() * sizeof(beta[0]));
  DeviceBuffer dinitial(initial.size() * sizeof(initial[0]));
  DeviceBuffer doutput(output.size() * sizeof(output[0]));
  DeviceBuffer dfinal(final.size() * sizeof(final[0]));
  if (dq.pointer == nullptr || dk.pointer == nullptr || dv.pointer == nullptr ||
      dgamma.pointer == nullptr || dbeta.pointer == nullptr ||
      dinitial.pointer == nullptr || doutput.pointer == nullptr ||
      dfinal.pointer == nullptr) {
    return 1;
  }
  if (!dq.upload(q.data()) || !dk.upload(k.data()) || !dv.upload(v.data()) ||
      !dgamma.upload(gamma.data()) || !dbeta.upload(beta.data()) ||
      !dinitial.upload(initial.data()) || !doutput.upload(output.data()) ||
      !dfinal.upload(final.data())) {
    return 1;
  }

  quactlize_ppu_chunked_gdn_problem_v1 problem{
      QUACTLIZE_PPU_CHUNKED_GDN_SCHEMA_V1,
      tokens,
      o.sequences,
      o.sequence_length,
      o.qk_heads,
      o.v_heads,
      kHeadK,
      kHeadV,
      kChunk,
  };
  auto launch = [&]() {
    return quactlize_ppu_chunked_gdn_fwd_bf16_v1(
        dq.as<std::uint16_t>(), dk.as<std::uint16_t>(), dv.as<std::uint16_t>(),
        dgamma.as<float>(), dbeta.as<float>(),
        o.initial_state ? dinitial.as<float>() : nullptr,
        doutput.as<std::uint16_t>(), o.final_state ? dfinal.as<float>() : nullptr,
        &problem, kScale, nullptr);
  };

  int current_device = 0;
  int cu = 0;
  if (!runtime_ok(hggcGetDevice(&current_device), "device ordinal query") ||
      !runtime_ok(hggcDeviceGetAttribute(
          &cu, hggcDevAttrMultiProcessorCount, current_device),
          "device CU query") ||
      cu <= 0) {
    std::fprintf(stderr, "[GDN perf] device API returned invalid CU count %d\n", cu);
    return 1;
  }
  std::int64_t const grid = std::int64_t(o.sequences) * o.v_heads;
  int const chunks = o.sequence_length / kChunk +
                     (o.sequence_length % kChunk != 0);
  std::int64_t const work_units = grid * chunks;
  std::printf(
      "[GDN perf config] implementation=qk+kk-aiu/generated-simt "
      "shape=B%d,T%d,H%d,HV%d,K128,V128,C64 GVA=%d:%d "
      "tokens=%d token_heads=%lld chunks_per_sequence=%d work_units=%lld "
      "grid=(%lld,1,1) threads=%d shared_bytes=%zu device=%d cu=%d "
      "logical_flops_per_full_chunk=%llu "
      "occupancy_api=UNAVAILABLE(reason=shipping-kernel-symbol-not-public) "
      "initial_state=%d final_state=%d\n",
      o.sequences, o.sequence_length, o.qk_heads, o.v_heads,
      o.qk_heads, o.v_heads, tokens,
      static_cast<long long>(std::int64_t(tokens) * o.v_heads), chunks,
      static_cast<long long>(work_units), static_cast<long long>(grid),
      kThreads, kSharedBytes, current_device, cu,
      static_cast<unsigned long long>(kLogicalFlopsPerFullChunk),
      int(o.initial_state), int(o.final_state));

  auto snapshot = [&](std::uint64_t& output_hash, std::uint64_t& state_hash,
                      std::size_t& output_nonfinite,
                      std::size_t& state_nonfinite) {
    if (!doutput.download(output.data()) ||
        (o.final_state && !dfinal.download(final.data()))) {
      return false;
    }
    output_hash = fnv1a64(output.data(), output.size() * sizeof(output[0]));
    state_hash = o.final_state
        ? fnv1a64(final.data(), final.size() * sizeof(final[0])) : 0;
    output_nonfinite = count_nonfinite_bf16(output);
    state_nonfinite = o.final_state ? count_nonfinite_fp32(final) : 0;
    return true;
  };

  if (o.acu) {
    int const rc = launch();
    bool const ok = rc == QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS &&
                    runtime_ok(hggcDeviceSynchronize(), "ACU launch synchronize");
    std::uint64_t output_hash = 0, state_hash = 0;
    std::size_t output_nonfinite = output.size(), state_nonfinite = final.size();
    bool const writeback_ok = ok && snapshot(
        output_hash, state_hash, output_nonfinite, state_nonfinite) &&
        output_nonfinite == 0 && state_nonfinite == 0;
    std::printf(
        "[GDN perf] protocol=acu-single-launch launches=1 warmup=0 "
        "timing=NOT_TIMING rc=%d output_nonfinite=%zu state_nonfinite=%zu "
        "output_fnv=%016llx state_fnv=%016llx writeback=%s %s\n",
        rc, output_nonfinite, state_nonfinite,
        static_cast<unsigned long long>(output_hash),
        static_cast<unsigned long long>(state_hash),
        writeback_ok ? "FINITE-NONPOISON" : "INVALID",
        writeback_ok ? "PASS" : "FAIL");
    return writeback_ok ? 0 : 1;
  }

  // One untimed launch establishes that the shipping ABI actually writes a
  // finite result over the uploaded NaN sentinels.  Requiring the same hashes
  // after timing proves deterministic replay for this fixture; L205 remains
  // the independent numerical oracle.
  int const preflight_rc = launch();
  if (preflight_rc != QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS ||
      !runtime_ok(hggcDeviceSynchronize(), "preflight synchronize")) {
    std::fprintf(stderr, "[GDN perf] preflight launch returned %d\n", preflight_rc);
    return 1;
  }
  std::uint64_t pre_output_hash = 0, pre_state_hash = 0;
  std::size_t pre_output_nonfinite = output.size();
  std::size_t pre_state_nonfinite = final.size();
  if (!snapshot(pre_output_hash, pre_state_hash, pre_output_nonfinite,
                pre_state_nonfinite) ||
      pre_output_nonfinite != 0 || pre_state_nonfinite != 0) {
    std::fprintf(
        stderr,
        "[GDN perf] preflight writeback invalid: output_nonfinite=%zu "
        "state_nonfinite=%zu\n",
        pre_output_nonfinite, pre_state_nonfinite);
    return 1;
  }
  std::printf(
      "[GDN perf preflight] launches=1 output_nonfinite=0 state_nonfinite=0 "
      "output_fnv=%016llx state_fnv=%016llx "
      "numerical_authority=L205-exact-ABI stability_anchor=RECORDED/PASS\n",
      static_cast<unsigned long long>(pre_output_hash),
      static_cast<unsigned long long>(pre_state_hash));

  for (int i = 0; i < o.warmup; ++i) {
    int const rc = launch();
    if (rc != QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS) {
      std::fprintf(stderr, "[GDN perf] warmup launch %d returned %d\n", i, rc);
      return 1;
    }
  }
  if (!runtime_ok(hggcDeviceSynchronize(), "warmup synchronize")) return 1;

  Events events;
  if (!events.valid) return 1;
  std::vector<double> per_launch_us;
  per_launch_us.reserve(std::size_t(o.samples));
  for (int sample = 0; sample < o.samples; ++sample) {
    if (!runtime_ok(hggcEventRecord(events.start, nullptr), "start-event record")) {
      return 1;
    }
    for (int iter = 0; iter < o.iterations; ++iter) {
      int const rc = launch();
      if (rc != QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS) {
        std::fprintf(stderr,
                     "[GDN perf] timed launch sample=%d iter=%d returned %d\n",
                     sample, iter, rc);
        return 1;
      }
    }
    if (!runtime_ok(hggcEventRecord(events.stop, nullptr), "stop-event record") ||
        !runtime_ok(hggcEventSynchronize(events.stop), "stop-event synchronize")) {
      return 1;
    }
    float elapsed_ms = 0.0f;
    if (!runtime_ok(
            hggcEventElapsedTime(&elapsed_ms, events.start, events.stop),
            "event elapsed time")) {
      return 1;
    }
    double const us = double(elapsed_ms) * 1000.0 / double(o.iterations);
    if (!std::isfinite(us) || us <= 0.0) {
      std::fprintf(stderr, "[GDN perf] invalid aggregate sample: %.9g us\n", us);
      return 1;
    }
    per_launch_us.push_back(us);
  }

  std::uint64_t output_hash = 0, state_hash = 0;
  std::size_t output_nonfinite = output.size(), state_nonfinite = final.size();
  if (!snapshot(output_hash, state_hash, output_nonfinite, state_nonfinite)) {
    return 1;
  }
  bool const stable = output_hash == pre_output_hash &&
                      state_hash == pre_state_hash &&
                      output_nonfinite == 0 && state_nonfinite == 0;
  if (!stable) {
    std::fprintf(
        stderr,
        "[GDN perf] post-timing replay changed: output=%016llx/%016llx "
        "state=%016llx/%016llx nonfinite=%zu/%zu\n",
        static_cast<unsigned long long>(pre_output_hash),
        static_cast<unsigned long long>(output_hash),
        static_cast<unsigned long long>(pre_state_hash),
        static_cast<unsigned long long>(state_hash), output_nonfinite,
        state_nonfinite);
    return 1;
  }
  double const med = median(per_launch_us);
  double const mean = std::accumulate(
      per_launch_us.begin(), per_launch_us.end(), 0.0) /
      double(per_launch_us.size());
  auto const [lo, hi] = std::minmax_element(
      per_launch_us.begin(), per_launch_us.end());
  double const us_per_token = med / double(tokens);
  double const us_per_token_head =
      med / double(std::int64_t(tokens) * o.v_heads);
  double const tokens_per_second = double(tokens) * 1.0e6 / med;
  if (o.sequence_length % kChunk == 0) {
    long double const logical_flops =
        static_cast<long double>(work_units) * kLogicalFlopsPerFullChunk;
    long double const effective_tflops = logical_flops / med / 1.0e6L;
    std::printf(
        "[GDN perf] protocol=device-event-launch-span-upper "
        "includes_launch_idle=1 untimed_preflight_launches=1 warmup=%d "
        "samples=%d launches_per_sample=%d "
        "median_us=%.6f mean_us=%.6f min_us=%.6f max_us=%.6f "
        "us_per_token=%.9f us_per_token_head=%.9f tokens_per_second=%.3f "
        "logical_flops=%.0Lf effective_tflops=%.6Lf flops_scope=full-chunks "
        "output_fnv=%016llx state_fnv=%016llx "
        "replay_fingerprint=STABLE numerical_authority=L205-exact-ABI PASS\n",
        o.warmup, o.samples, o.iterations, med, mean, *lo, *hi,
        us_per_token, us_per_token_head, tokens_per_second, logical_flops,
        effective_tflops, static_cast<unsigned long long>(output_hash),
        static_cast<unsigned long long>(state_hash));
  } else {
    std::printf(
        "[GDN perf] protocol=device-event-launch-span-upper "
        "includes_launch_idle=1 untimed_preflight_launches=1 warmup=%d "
        "samples=%d launches_per_sample=%d "
        "median_us=%.6f mean_us=%.6f min_us=%.6f max_us=%.6f "
        "us_per_token=%.9f us_per_token_head=%.9f tokens_per_second=%.3f "
        "logical_flops=UNAVAILABLE effective_tflops=UNAVAILABLE "
        "flops_scope=tail-model-not-registered output_fnv=%016llx "
        "state_fnv=%016llx replay_fingerprint=STABLE "
        "numerical_authority=L205-exact-ABI PASS\n",
        o.warmup, o.samples, o.iterations, med, mean, *lo, *hi,
        us_per_token, us_per_token_head, tokens_per_second,
        static_cast<unsigned long long>(output_hash),
        static_cast<unsigned long long>(state_hash));
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_options(argc, argv, options)) {
    usage(argv[0]);
    return 2;
  }
  return run(options);
}
