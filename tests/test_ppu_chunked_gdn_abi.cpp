// Standalone device correctness harness for the public PPU chunked-GDN C ABI.
//
// This is deliberately a host C++ executable linked against libquactlize_ppu:
// no private kernel header is visible here.  The fixture implements the token
// recurrence independently, while the library implements the chunk/WY path.
// A match therefore crosses both the public ABI and the algebra boundary.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(QZ_GDN_CUDA_RUNTIME)
#include <cuda_runtime.h>
#else
#include <hggc_runtime.h>
#endif

#include "quactlize_ppu_linear_attention.h"

namespace {

#if defined(QZ_GDN_CUDA_RUNTIME)
using DeviceError = cudaError_t;
constexpr DeviceError kDeviceSuccess = cudaSuccess;
constexpr auto kHostToDevice = cudaMemcpyHostToDevice;
constexpr auto kDeviceToHost = cudaMemcpyDeviceToHost;
DeviceError device_malloc(void** p, std::size_t n) { return cudaMalloc(p, n); }
DeviceError device_free(void* p) { return cudaFree(p); }
DeviceError device_copy(void* dst, void const* src, std::size_t n, cudaMemcpyKind kind) {
  return cudaMemcpy(dst, src, n, kind);
}
DeviceError device_synchronize() { return cudaDeviceSynchronize(); }
char const* device_error_string(DeviceError e) { return cudaGetErrorString(e); }
#else
using DeviceError = hggcError_t;
constexpr DeviceError kDeviceSuccess = hggcSuccess;
constexpr auto kHostToDevice = hggcMemcpyHostToDevice;
constexpr auto kDeviceToHost = hggcMemcpyDeviceToHost;
DeviceError device_malloc(void** p, std::size_t n) { return hggcMalloc(p, n); }
DeviceError device_free(void* p) { return hggcFree(p); }
DeviceError device_copy(void* dst, void const* src, std::size_t n, hggcMemcpyKind kind) {
  return hggcMemcpy(dst, src, n, kind);
}
DeviceError device_synchronize() { return hggcDeviceSynchronize(); }
char const* device_error_string(DeviceError e) { return hggcGetErrorString(e); }
#endif

constexpr int kTokens = 65;
constexpr int kSequences = 1;
constexpr int kQkHeads = 1;
constexpr int kVHeads = 2;
constexpr int kHeadK = 128;
constexpr int kHeadV = 128;
constexpr int kChunk = 64;
constexpr float kScale = 0.5f;

std::uint16_t float_to_bf16(float x) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &x, sizeof(bits));
  // Round-to-nearest-even.  Every input in the exact fixture is already BF16,
  // but output conversion must match the device store rather than truncate.
  std::uint32_t const lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return std::uint16_t(bits >> 16);
}

float bf16_to_float(std::uint16_t x) {
  std::uint32_t bits = std::uint32_t(x) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

struct Fixture {
  std::vector<std::uint16_t> q;
  std::vector<std::uint16_t> k;
  std::vector<std::uint16_t> v;
  std::vector<float> gamma;
  std::vector<float> beta;
  std::vector<float> initial;
};

enum class KeyPattern {
  kDistinct,
  kPaired,
};

struct Reference {
  std::vector<std::uint16_t> output;
  std::vector<float> final_state;
};

struct ExactnessReport {
  bool ok = true;
  std::size_t a = 0;
  std::size_t w = 0;
  std::size_t u = 0;
  std::size_t p = 0;
  std::size_t vnew = 0;
  std::size_t h_cast = 0;
  std::size_t a_bad = 0;
  std::size_t w_bad = 0;
  std::size_t u_bad = 0;
  std::size_t p_bad = 0;
  std::size_t vnew_bad = 0;
  std::size_t h_cast_bad = 0;
  std::size_t strict_lower_nonzero = 0;
  std::size_t inverse_offdiag_nonzero = 0;
  std::size_t causal_offdiag_nonzero = 0;
};

std::size_t qk_index(int token, int head, int d) {
  return (std::size_t(token) * kQkHeads + head) * kHeadK + d;
}

std::size_t v_index(int token, int head, int d) {
  return (std::size_t(token) * kVHeads + head) * kHeadV + d;
}

std::size_t gate_index(int token, int head) {
  return std::size_t(token) * kVHeads + head;
}

std::size_t state_index(int sequence, int head, int k, int v) {
  return (((std::size_t(sequence) * kVHeads + head) * kHeadK + k) * kHeadV + v);
}

Fixture make_fixture(bool nonzero_initial_state,
                     KeyPattern pattern = KeyPattern::kDistinct) {
  Fixture f;
  f.q.assign(std::size_t(kTokens) * kQkHeads * kHeadK, 0);
  f.k.assign(std::size_t(kTokens) * kQkHeads * kHeadK, 0);
  f.v.resize(std::size_t(kTokens) * kVHeads * kHeadV);
  f.gamma.resize(std::size_t(kTokens) * kVHeads);
  f.beta.resize(std::size_t(kTokens) * kVHeads);
  f.initial.assign(std::size_t(kSequences) * kVHeads * kHeadK * kHeadV, 0.0f);

  for (int t = 0; t < kTokens; ++t) {
    // One-hot Q/K removes reduction-order ambiguity without weakening the
    // 128-wide address/layout test.  The paired arm repeats each coordinate
    // twice inside C64 and token 64 revisits row zero in the tail chunk, so it
    // exercises both nonzero WY coupling and recurrent state across chunks.
    int const kd = pattern == KeyPattern::kDistinct
        ? t % kHeadK
        : ((t % kChunk) / 2) % kHeadK;
    f.q[qk_index(t, 0, kd)] = float_to_bf16(0.25f);
    f.k[qk_index(t, 0, kd)] = float_to_bf16(0.5f);

    int const local = t % kChunk;
    for (int h = 0; h < kVHeads; ++h) {
      // gamma is CHUNK-LOCAL CUMULATIVE log2 decay, not a raw gate.  The
      // recurrence below differences it within each chunk.  All exp2 values
      // are exact powers of two, so any mismatch is structural, not tolerance.
      float cumulative = 0.0f;
      if (pattern == KeyPattern::kDistinct) {
        if (local >= 17) cumulative -= 1.0f;
        if (local >= 48) cumulative -= 1.0f;
        if (t == 64) cumulative = -1.0f;
      }
      f.gamma[gate_index(t, h)] = cumulative;
      f.beta[gate_index(t, h)] = pattern == KeyPattern::kDistinct
          ? 0.25f * float(1 + ((t + h) % 3))
          : 0.5f;
      for (int d = 0; d < kHeadV; ++d) {
        int tag = ((t / 2) * 3 + d * 5 + h * 7) % 9 - 4;
        float value = 0.0f;
        if (pattern == KeyPattern::kDistinct) {
          tag = (t * 3 + d * 5 + h * 7) % 9 - 4;
          value = float(tag) * 0.03125f;
        } else if (t < kChunk) {
          // With beta=1/2, K=1/2, and one repeated coordinate, L(1,0)=1/8
          // and A=I-L.  V1=-7/8*V0 makes Vnew0+Vnew1=0 for zero H, while
          // retaining nonzero U/W and exact binary fractions.  For nonzero H
          // the pair maps H -> 49/64 H, also exactly representable in BF16.
          float const pair_value = float(tag) * 0.03125f;
          value = (t & 1) == 0 ? pair_value : -0.875f * pair_value;
        }
        // The single tail token revisits coordinate zero with V=0.  It still
        // consumes the nonzero recurrent H left by the first chunk, without
        // introducing a ninth significand bit at the vnew BF16 boundary.
        f.v[v_index(t, h, d)] = float_to_bf16(value);
      }
    }
  }

  if (nonzero_initial_state) {
    for (int h = 0; h < kVHeads; ++h) {
      for (int k = 0; k < kHeadK; ++k) {
        for (int v = 0; v < kHeadV; ++v) {
          // A head-distinct power-of-two value keeps every later state and
          // residual within BF16's exact significand.  Thus the production
          // collective's documented BF16 QH/WH boundary cannot turn this
          // structural test into a tolerance test.
          f.initial[state_index(0, h, k, v)] = float(h + 1) * 0.015625f;
        }
      }
    }
  }
  return f;
}

Reference token_recurrence(Fixture const& f, bool nonzero_initial_state) {
  Reference r;
  r.output.resize(std::size_t(kTokens) * kVHeads * kHeadV);
  r.final_state = nonzero_initial_state
      ? f.initial
      : std::vector<float>(std::size_t(kSequences) * kVHeads * kHeadK * kHeadV, 0.0f);
  std::vector<float> residual(kHeadV, 0.0f);

  for (int h = 0; h < kVHeads; ++h) {
    int const qk_head = h / (kVHeads / kQkHeads);
    float previous_gamma = 0.0f;
    for (int t = 0; t < kTokens; ++t) {
      if ((t % kChunk) == 0) previous_gamma = 0.0f;
      float const gamma = f.gamma[gate_index(t, h)];
      float const decay = std::exp2(gamma - previous_gamma);
      previous_gamma = gamma;
      for (int k = 0; k < kHeadK; ++k) {
        for (int v = 0; v < kHeadV; ++v) {
          r.final_state[state_index(0, h, k, v)] *= decay;
        }
      }

      float const beta = f.beta[gate_index(t, h)];
      for (int v = 0; v < kHeadV; ++v) {
        float prediction = 0.0f;
        for (int k = 0; k < kHeadK; ++k) {
          prediction += bf16_to_float(f.k[qk_index(t, qk_head, k)]) *
                        r.final_state[state_index(0, h, k, v)];
        }
        residual[v] = beta * (bf16_to_float(f.v[v_index(t, h, v)]) - prediction);
      }
      for (int k = 0; k < kHeadK; ++k) {
        float const key = bf16_to_float(f.k[qk_index(t, qk_head, k)]);
        for (int v = 0; v < kHeadV; ++v) {
          r.final_state[state_index(0, h, k, v)] += key * residual[v];
        }
      }
      for (int v = 0; v < kHeadV; ++v) {
        float out = 0.0f;
        for (int k = 0; k < kHeadK; ++k) {
          out += bf16_to_float(f.q[qk_index(t, qk_head, k)]) *
                 r.final_state[state_index(0, h, k, v)];
        }
        r.output[v_index(t, h, v)] = float_to_bf16(kScale * out);
      }
    }
  }
  return r;
}

bool bf16_exact(float x) {
  return bf16_to_float(float_to_bf16(x)) == x;
}

// Prove that RAW-BIT is valid under every BF16 materialization in the shipping
// collective. This follows its declared A/W/U/P/vnew/H rounding boundaries,
// but does not produce the golden result (token_recurrence remains independent).
ExactnessReport check_bf16_boundaries(Fixture const& f, bool nonzero_initial_state) {
  ExactnessReport r;
  auto require_exact = [&](float x, std::size_t& count, std::size_t& bad) {
    ++count;
    if (!bf16_exact(x)) {
      ++bad;
      r.ok = false;
    }
  };

  for (int head = 0; head < kVHeads; ++head) {
    std::vector<float> state(std::size_t(kHeadK) * kHeadV, 0.0f);
    if (nonzero_initial_state) {
      for (int k = 0; k < kHeadK; ++k) {
        for (int v = 0; v < kHeadV; ++v) {
          state[std::size_t(k) * kHeadV + v] = f.initial[state_index(0, head, k, v)];
        }
      }
    }

    for (int begin = 0; begin < kTokens; begin += kChunk) {
      int const valid = std::min(kChunk, kTokens - begin);
      float const gamma_last = f.gamma[gate_index(begin + valid - 1, head)];

      // The collective casts H once per QH/WH use.  Checking every unique H
      // value is sufficient; h_cast records the full dynamic cast count.
      for (float x : state) {
        if (!bf16_exact(x)) {
          ++r.h_cast_bad;
          r.ok = false;
        }
      }
      r.h_cast += std::size_t(valid) * kHeadK * kHeadV;

      std::vector<float> lower(std::size_t(kChunk) * kChunk, 0.0f);
      std::vector<float> inverse_fp32(std::size_t(kChunk) * kChunk, 0.0f);
      std::vector<float> inverse_bf16(std::size_t(kChunk) * kChunk, 0.0f);
      for (int row = 0; row < kChunk; ++row) {
        for (int col = 0; col < kChunk; ++col) {
          float dot_kk = 0.0f;
          float dot_qk = 0.0f;
          if (row < valid && col < valid) {
            for (int d = 0; d < kHeadK; ++d) {
              dot_kk += bf16_to_float(f.k[qk_index(begin + row, 0, d)]) *
                        bf16_to_float(f.k[qk_index(begin + col, 0, d)]);
              dot_qk += bf16_to_float(f.q[qk_index(begin + row, 0, d)]) *
                        bf16_to_float(f.k[qk_index(begin + col, 0, d)]);
            }
          }
          if (row < valid && col < row) {
            lower[std::size_t(row) * kChunk + col] =
                f.beta[gate_index(begin + row, head)] * dot_kk *
                std::exp2(f.gamma[gate_index(begin + row, head)] -
                          f.gamma[gate_index(begin + col, head)]);
            if (lower[std::size_t(row) * kChunk + col] != 0.0f) {
              ++r.strict_lower_nonzero;
            }
          }
          float const p = row < valid && col < valid && row >= col
              ? dot_qk * std::exp2(f.gamma[gate_index(begin + row, head)] -
                                   f.gamma[gate_index(begin + col, head)])
              : 0.0f;
          require_exact(p, r.p, r.p_bad);
          if (row > col && p != 0.0f) ++r.causal_offdiag_nonzero;
        }
      }

      // Same row dependency as solve_inverse.  For the paired fixture each
      // nonzero is an isolated (odd,even) edge, hence L^2=0 and A=I-L.
      for (int row = 0; row < kChunk; ++row) {
        for (int col = 0; col < kChunk; ++col) {
          float a = row == col ? 1.0f : 0.0f;
          for (int k = 0; k < row; ++k) {
            a -= lower[std::size_t(row) * kChunk + k] *
                 inverse_fp32[std::size_t(k) * kChunk + col];
          }
          inverse_fp32[std::size_t(row) * kChunk + col] = a;
          require_exact(a, r.a, r.a_bad);
          inverse_bf16[std::size_t(row) * kChunk + col] =
              bf16_to_float(float_to_bf16(a));
          if (row != col && a != 0.0f) ++r.inverse_offdiag_nonzero;
        }
      }

      std::vector<float> w_bf16(std::size_t(kChunk) * kHeadK, 0.0f);
      for (int row = 0; row < kChunk; ++row) {
        for (int d = 0; d < kHeadK; ++d) {
          float w = 0.0f;
          for (int j = 0; j < kChunk; ++j) {
            float const kb = j < valid
                ? f.beta[gate_index(begin + j, head)] *
                      std::exp2(f.gamma[gate_index(begin + j, head)]) *
                      bf16_to_float(f.k[qk_index(begin + j, 0, d)])
                : 0.0f;
            w += inverse_bf16[std::size_t(row) * kChunk + j] * kb;
          }
          require_exact(w, r.w, r.w_bad);
          w_bf16[std::size_t(row) * kHeadK + d] =
              bf16_to_float(float_to_bf16(w));
        }
      }

      std::vector<float> vnew_scaled(std::size_t(kChunk) * kHeadV, 0.0f);
      for (int row = 0; row < kChunk; ++row) {
        bool const live = row < valid;
        for (int v = 0; v < kHeadV; ++v) {
          float u = 0.0f;
          for (int j = 0; j < kChunk; ++j) {
            float const vb = j < valid
                ? f.beta[gate_index(begin + j, head)] *
                      bf16_to_float(f.v[v_index(begin + j, head, v)])
                : 0.0f;
            u += inverse_bf16[std::size_t(row) * kChunk + j] * vb;
          }
          require_exact(u, r.u, r.u_bad);
          float wh = 0.0f;
          if (live) {
            for (int d = 0; d < kHeadK; ++d) {
              float const h = bf16_to_float(
                  float_to_bf16(state[std::size_t(d) * kHeadV + v]));
              wh += w_bf16[std::size_t(row) * kHeadK + d] * h;
            }
          }
          float const vn = live ? bf16_to_float(float_to_bf16(u)) - wh : 0.0f;
          require_exact(vn, r.vnew, r.vnew_bad);
          float const scaled = live
              ? std::exp2(gamma_last - f.gamma[gate_index(begin + row, head)]) * vn
              : 0.0f;
          require_exact(scaled, r.vnew, r.vnew_bad);
          vnew_scaled[std::size_t(row) * kHeadV + v] =
              bf16_to_float(float_to_bf16(scaled));
        }
      }

      float const state_decay = std::exp2(gamma_last);
      for (float& x : state) x *= state_decay;
      for (int row = 0; row < valid; ++row) {
        for (int d = 0; d < kHeadK; ++d) {
          float const key = bf16_to_float(f.k[qk_index(begin + row, 0, d)]);
          for (int v = 0; v < kHeadV; ++v) {
            state[std::size_t(d) * kHeadV + v] +=
                key * vnew_scaled[std::size_t(row) * kHeadV + v];
          }
        }
      }
    }
  }
  return r;
}

bool runtime_ok(DeviceError status, char const* where) {
  if (status == kDeviceSuccess) return true;
  std::fprintf(stderr, "[GDN device] %s failed: %s\n", where, device_error_string(status));
  return false;
}

struct DeviceBuffer {
  void* pointer = nullptr;
  std::size_t bytes = 0;

  explicit DeviceBuffer(std::size_t n) : bytes(n) {
    if (n && !runtime_ok(device_malloc(&pointer, n), "device allocation")) pointer = nullptr;
  }
  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;
  ~DeviceBuffer() {
    if (pointer) (void)device_free(pointer);
  }

  template <class T>
  T* as() const { return static_cast<T*>(pointer); }

  bool upload(void const* source) const {
    return pointer && runtime_ok(
        device_copy(pointer, source, bytes, kHostToDevice), "H2D");
  }
  bool download(void* destination) const {
    return pointer && runtime_ok(
        device_copy(destination, pointer, bytes, kDeviceToHost), "D2H");
  }
};

quactlize_ppu_chunked_gdn_problem_v1 valid_problem() {
  return {
      QUACTLIZE_PPU_CHUNKED_GDN_SCHEMA_V1,
      kTokens,
      kSequences,
      kTokens,
      kQkHeads,
      kVHeads,
      kHeadK,
      kHeadV,
      kChunk,
  };
}

bool check_admission() {
  alignas(16) std::uint16_t dummy_bf16[16]{};
  float dummy_float = 0.0f;
  auto call = [&](quactlize_ppu_chunked_gdn_problem_v1 problem,
                  std::uint16_t const* q = nullptr) {
    if (q == nullptr) q = dummy_bf16;
    return quactlize_ppu_chunked_gdn_fwd_bf16_v1(
        q, dummy_bf16, dummy_bf16, &dummy_float, &dummy_float,
        nullptr, dummy_bf16, &dummy_float, &problem, kScale, nullptr);
  };
  struct Case {
    char const* name;
    quactlize_ppu_chunked_gdn_problem_v1 problem;
    int expected;
  };
  std::vector<Case> cases;
  {
    auto p = valid_problem();
    p.schema_version = QUACTLIZE_PPU_CHUNKED_GDN_SCHEMA_V1 + 1;
    cases.push_back({"schema", p, QUACTLIZE_PPU_CHUNKED_GDN_INVALID_PROBLEM});
  }
  {
    auto p = valid_problem();
    p.chunk_size = 32;
    cases.push_back({"chunk", p, QUACTLIZE_PPU_CHUNKED_GDN_UNSUPPORTED_CHUNK_SIZE});
  }
  {
    auto p = valid_problem();
    p.head_size_k = 64;
    cases.push_back({"head", p, QUACTLIZE_PPU_CHUNKED_GDN_UNSUPPORTED_HEAD_DIMENSION});
  }
  {
    auto p = valid_problem();
    p.num_qk_heads = 2;
    p.num_v_heads = 3;
    cases.push_back({"head-map", p, QUACTLIZE_PPU_CHUNKED_GDN_UNSUPPORTED_HEAD_MAPPING});
  }
  {
    auto p = valid_problem();
    p.total_tokens = 46341;
    p.num_sequences = 46341;
    p.sequence_length = 1;
    p.num_qk_heads = 1;
    p.num_v_heads = 46341;
    cases.push_back({"grid-overflow", p, QUACTLIZE_PPU_CHUNKED_GDN_INVALID_PROBLEM});
  }
  {
    auto p = valid_problem();
    p.total_tokens = std::numeric_limits<std::int32_t>::max();
    p.num_sequences = 1;
    p.sequence_length = std::numeric_limits<std::int32_t>::max();
    p.num_qk_heads = std::numeric_limits<std::int32_t>::max();
    p.num_v_heads = std::numeric_limits<std::int32_t>::max();
    cases.push_back({"extent-overflow", p, QUACTLIZE_PPU_CHUNKED_GDN_INVALID_PROBLEM});
  }

  bool ok = true;
  for (auto const& c : cases) {
    int const got = call(c.problem);
    bool const pass = got == c.expected;
    ok &= pass;
    std::printf("[GDN admission] plant=%s got=%d expected=%d %s\n",
                c.name, got, c.expected, pass ? "EXPECTED_RED/PASS" : "FAIL");
  }
  {
    auto p = valid_problem();
    int const got = call(p, dummy_bf16 + 1);
    bool const pass = got == QUACTLIZE_PPU_CHUNKED_GDN_MISALIGNED_POINTER;
    ok &= pass;
    std::printf("[GDN admission] plant=misaligned-q got=%d expected=%d %s\n",
                got, QUACTLIZE_PPU_CHUNKED_GDN_MISALIGNED_POINTER,
                pass ? "EXPECTED_RED/PASS" : "FAIL");
  }
  return ok;
}

bool check_exactness_negative_control() {
  Fixture poisoned = make_fixture(true);
  // 0.1f is not representable in BF16.  If the boundary proof ever stops
  // inspecting the H cast, this plant goes falsely green immediately.
  poisoned.initial[state_index(0, 0, 0, 0)] = 0.1f;
  bool const red = !check_bf16_boundaries(poisoned, true).ok;
  std::printf("[GDN fixture exactness plant] non-bf16-H %s\n",
              red ? "EXPECTED_RED/PASS" : "FAIL");
  return red;
}

char const* pattern_name(KeyPattern pattern) {
  return pattern == KeyPattern::kPaired ? "paired" : "distinct";
}

bool check_device_case(bool nonzero_initial_state, KeyPattern pattern) {
  Fixture const f = make_fixture(nonzero_initial_state, pattern);
  Reference const ref = token_recurrence(f, nonzero_initial_state);
  ExactnessReport const exact = check_bf16_boundaries(f, nonzero_initial_state);
  constexpr std::size_t kExpectedPairedEdges =
      std::size_t(kChunk / 2) * kVHeads;
  bool const coverage_ok = pattern == KeyPattern::kPaired
      ? exact.strict_lower_nonzero == kExpectedPairedEdges &&
            exact.inverse_offdiag_nonzero == kExpectedPairedEdges &&
            exact.causal_offdiag_nonzero == kExpectedPairedEdges
      : exact.strict_lower_nonzero == 0 &&
            exact.inverse_offdiag_nonzero == 0 &&
            exact.causal_offdiag_nonzero == 0;
  std::printf(
      "[GDN fixture exactness] pattern=%s state=%s A=%zu W=%zu U=%zu P=%zu "
      "vnew+scaled=%zu H-casts=%zu strict_lower_nonzero=%zu "
      "inverse_offdiag_nonzero=%zu causal_offdiag_nonzero=%zu "
      "boundary_bad=A:%zu/W:%zu/U:%zu/P:%zu/vnew:%zu/H:%zu %s\n",
      pattern_name(pattern), nonzero_initial_state ? "nonzero" : "zero",
      exact.a, exact.w, exact.u, exact.p, exact.vnew, exact.h_cast,
      exact.strict_lower_nonzero, exact.inverse_offdiag_nonzero,
      exact.causal_offdiag_nonzero, exact.a_bad, exact.w_bad, exact.u_bad,
      exact.p_bad, exact.vnew_bad, exact.h_cast_bad,
      exact.ok && coverage_ok ? "BF16-BOUNDARIES-EXACT/PASS" : "FAIL");
  std::printf(
      "[GDN WY coverage] pattern=%s strict_lower_nonzero=%zu "
      "inverse_offdiag_nonzero=%zu causal_offdiag_nonzero=%zu expected=%zu %s\n",
      pattern_name(pattern), exact.strict_lower_nonzero,
      exact.inverse_offdiag_nonzero, exact.causal_offdiag_nonzero,
      pattern == KeyPattern::kPaired ? kExpectedPairedEdges : 0,
      coverage_ok ? "EXACT/PASS" : "FAIL");
  std::vector<std::uint16_t> got_output(ref.output.size(), 0xffffu);
  std::vector<float> got_state(ref.final_state.size(), std::numeric_limits<float>::quiet_NaN());

  DeviceBuffer dq(f.q.size() * sizeof(f.q[0]));
  DeviceBuffer dk(f.k.size() * sizeof(f.k[0]));
  DeviceBuffer dv(f.v.size() * sizeof(f.v[0]));
  DeviceBuffer dgamma(f.gamma.size() * sizeof(f.gamma[0]));
  DeviceBuffer dbeta(f.beta.size() * sizeof(f.beta[0]));
  DeviceBuffer dinitial(f.initial.size() * sizeof(f.initial[0]));
  DeviceBuffer doutput(got_output.size() * sizeof(got_output[0]));
  DeviceBuffer dstate(got_state.size() * sizeof(got_state[0]));
  bool ok = exact.ok && coverage_ok && dq.pointer && dk.pointer && dv.pointer &&
            dgamma.pointer && dbeta.pointer &&
            dinitial.pointer && doutput.pointer && dstate.pointer;
  ok &= dq.upload(f.q.data()) && dk.upload(f.k.data()) && dv.upload(f.v.data());
  ok &= dgamma.upload(f.gamma.data()) && dbeta.upload(f.beta.data());
  ok &= dinitial.upload(f.initial.data());
  if (!ok) return false;

  auto problem = valid_problem();
  int const launch = quactlize_ppu_chunked_gdn_fwd_bf16_v1(
      dq.as<std::uint16_t>(), dk.as<std::uint16_t>(), dv.as<std::uint16_t>(),
      dgamma.as<float>(), dbeta.as<float>(),
      nonzero_initial_state ? dinitial.as<float>() : nullptr,
      doutput.as<std::uint16_t>(), dstate.as<float>(), &problem, kScale, nullptr);
  ok &= launch == QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS;
  ok &= runtime_ok(device_synchronize(), "device synchronize");
  ok &= doutput.download(got_output.data()) && dstate.download(got_state.data());

  std::size_t output_bad = 0, state_bad = 0;
  float max_output_abs = 0.0f, max_state_abs = 0.0f;
  std::size_t first_output = got_output.size(), first_state = got_state.size();
  for (std::size_t i = 0; i < got_output.size(); ++i) {
    float const diff = std::abs(bf16_to_float(got_output[i]) - bf16_to_float(ref.output[i]));
    max_output_abs = std::max(max_output_abs, diff);
    if (got_output[i] != ref.output[i]) {
      if (first_output == got_output.size()) first_output = i;
      ++output_bad;
    }
  }
  for (std::size_t i = 0; i < got_state.size(); ++i) {
    float const diff = std::abs(got_state[i] - ref.final_state[i]);
    max_state_abs = std::max(max_state_abs, diff);
    std::uint32_t got_bits = 0, ref_bits = 0;
    std::memcpy(&got_bits, &got_state[i], sizeof(got_bits));
    std::memcpy(&ref_bits, &ref.final_state[i], sizeof(ref_bits));
    if (got_bits != ref_bits) {
      if (first_state == got_state.size()) first_state = i;
      ++state_bad;
    }
  }
  ok &= output_bad == 0 && state_bad == 0;
  std::printf(
      "[GDN device] pattern=%s state=%s T=65 C=64 K=128 V=128 GVA=1:2 "
      "output_raw_bad=%zu/%zu state_raw_bad=%zu/%zu max_output_abs=%g "
      "max_state_abs=%g launch_rc=%d %s\n",
      pattern_name(pattern), nonzero_initial_state ? "nonzero" : "zero",
      output_bad, got_output.size(), state_bad, got_state.size(), max_output_abs,
      max_state_abs, launch, ok ? "RAW-BIT/PASS" : "FAIL");
  if (first_output != got_output.size()) {
    std::printf("  first output mismatch i=%zu got=0x%04x/%g want=0x%04x/%g\n",
                first_output, unsigned(got_output[first_output]),
                bf16_to_float(got_output[first_output]), unsigned(ref.output[first_output]),
                bf16_to_float(ref.output[first_output]));
  }
  if (first_state != got_state.size()) {
    std::printf("  first state mismatch i=%zu got=%g want=%g\n",
                first_state, got_state[first_state], ref.final_state[first_state]);
  }
  return ok;
}

}  // namespace

int main() {
  bool const admission = check_admission();
  bool const exactness_plant = check_exactness_negative_control();
  bool const distinct_zero = check_device_case(false, KeyPattern::kDistinct);
  bool const distinct_nonzero = check_device_case(true, KeyPattern::kDistinct);
  bool const paired_zero = check_device_case(false, KeyPattern::kPaired);
  bool const paired_nonzero = check_device_case(true, KeyPattern::kPaired);
  bool const ok = admission && exactness_plant && distinct_zero && distinct_nonzero &&
                  paired_zero && paired_nonzero;
  std::printf("[GDN ABI] %s: public C ABI + T65 tail + GVA 1:2 + "
              "distinct/paired WY + zero/nonzero state\n",
              ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
