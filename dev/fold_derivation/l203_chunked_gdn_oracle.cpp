// L203: independent host authority for PPU chunked GDN.
//
// This file deliberately contains both decompositions:
//   (1) the token recurrence from the GDN definition, and
//   (2) the chunk/WY decomposition implemented by the fused kernel.
// They share only flat input arrays.  A match therefore proves the chunk
// algebra rather than comparing an implementation with itself.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_inverse.hpp"
#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_types.hpp"

namespace {

using cutlass::linear_attention::PpuChunkedGdnArguments;
using cutlass::linear_attention::PpuChunkedGdnProblem;
using cutlass::linear_attention::PpuChunkedGdnScheduler;
using cutlass::linear_attention::PpuChunkedGdnStatus;
using cutlass::linear_attention::PpuChunkedGdnTraits;
using cutlass::linear_attention::can_implement_ppu_chunked_gdn;
using cutlass::linear_attention::detail::invert_unit_lower_blocked;
using cutlass::linear_attention::detail::unit_lower_inverse_residual;

struct Fixture {
  int t = 0;
  int dk = 0;
  int dv = 0;
  int chunk = 0;
  float scale = 1.0f;
  std::vector<float> q, k, v, delta_log2, beta, initial;
};

struct Result {
  std::vector<float> output;
  std::vector<float> state;
};

float tagged(int a, int b, int c, int modulus, float quantum) {
  int x = (a * 131 + b * 37 + c * 17 + 19) % modulus;
  return float(x - modulus / 2) * quantum;
}

Fixture make_fixture(int t, int dk, int dv, int chunk, bool nonzero_state) {
  Fixture f;
  f.t = t;
  f.dk = dk;
  f.dv = dv;
  f.chunk = chunk;
  f.scale = 0.5f;
  f.q.resize(t * dk);
  f.k.resize(t * dk);
  f.v.resize(t * dv);
  f.delta_log2.resize(t);
  f.beta.resize(t);
  f.initial.resize(dk * dv);
  for (int i = 0; i < t; ++i) {
    f.delta_log2[i] = -float((i * 5 + 1) % 4) * 0.25f;
    f.beta[i] = float(1 + (i * 3) % 3) * 0.25f;
    for (int d = 0; d < dk; ++d) {
      f.q[i * dk + d] = tagged(i, d, 0, 9, 0.0625f);
      f.k[i * dk + d] = tagged(i, d, 1, 7, 0.0625f);
    }
    for (int d = 0; d < dv; ++d) {
      f.v[i * dv + d] = tagged(i, d, 2, 11, 0.0625f);
    }
  }
  for (int i = 0; i < dk; ++i) {
    for (int j = 0; j < dv; ++j) {
      f.initial[i * dv + j] =
          nonzero_state ? tagged(i, j, 3, 7, 0.015625f) : 0.0f;
    }
  }
  return f;
}

Result token_recurrence(Fixture const& f) {
  Result r{{}, f.initial};
  r.output.assign(f.t * f.dv, 0.0f);
  std::vector<float> residual(f.dv);
  for (int t = 0; t < f.t; ++t) {
    float const decay = std::exp2(f.delta_log2[t]);
    for (float& x : r.state) x *= decay;
    for (int v = 0; v < f.dv; ++v) {
      float prediction = 0.0f;
      for (int k = 0; k < f.dk; ++k) {
        prediction += f.k[t * f.dk + k] * r.state[k * f.dv + v];
      }
      residual[v] = f.beta[t] * (f.v[t * f.dv + v] - prediction);
    }
    for (int k = 0; k < f.dk; ++k) {
      for (int v = 0; v < f.dv; ++v) {
        r.state[k * f.dv + v] += f.k[t * f.dk + k] * residual[v];
      }
    }
    for (int v = 0; v < f.dv; ++v) {
      float out = 0.0f;
      for (int k = 0; k < f.dk; ++k) {
        out += f.q[t * f.dk + k] * r.state[k * f.dv + v];
      }
      r.output[t * f.dv + v] = f.scale * out;
    }
  }
  return r;
}

struct Plant {
  bool exclude_causal_diagonal = false;
  bool omit_beta_from_w = false;
  bool reverse_decay_difference = false;
  bool advance_tail_to_full_chunk = false;
};

Result chunk_wy(Fixture const& f, Plant plant = {}) {
  Result r{{}, f.initial};
  r.output.assign(f.t * f.dv, 0.0f);

  int const C = f.chunk;
  std::vector<float> gamma(C), lower(C * C), inv(C * C);
  std::vector<float> u(C * f.dv), w(C * f.dk), residual(C * f.dv);

  for (int begin = 0; begin < f.t; begin += C) {
    int const valid = std::min(C, f.t - begin);
    std::fill(gamma.begin(), gamma.end(), 0.0f);
    std::fill(lower.begin(), lower.end(), 0.0f);
    gamma[0] = f.delta_log2[begin];
    for (int i = 1; i < valid; ++i) {
      gamma[i] = gamma[i - 1] + f.delta_log2[begin + i];
    }
    // Invalid rows/columns are an identity extension.  They neither advance
    // the recurrent decay nor contribute K/V data.
    for (int i = 0; i < valid; ++i) {
      for (int j = 0; j < i; ++j) {
        float kk = 0.0f;
        for (int d = 0; d < f.dk; ++d) {
          kk += f.k[(begin + i) * f.dk + d] * f.k[(begin + j) * f.dk + d];
        }
        float exponent = gamma[i] - gamma[j];
        if (plant.reverse_decay_difference) exponent = -exponent;
        lower[i * C + j] = f.beta[begin + i] * std::exp2(exponent) * kk;
      }
    }
    if (!invert_unit_lower_blocked(lower.data(), inv.data(), C, 16)) {
      std::fprintf(stderr, "L203 internal error: inverse rejected C=%d\n", C);
      std::exit(2);
    }

    std::fill(u.begin(), u.end(), 0.0f);
    std::fill(w.begin(), w.end(), 0.0f);
    for (int i = 0; i < valid; ++i) {
      for (int j = 0; j < valid; ++j) {
        float const tij = inv[i * C + j];
        for (int d = 0; d < f.dv; ++d) {
          u[i * f.dv + d] += tij * f.beta[begin + j] * f.v[(begin + j) * f.dv + d];
        }
        float const w_beta = plant.omit_beta_from_w ? 1.0f : f.beta[begin + j];
        for (int d = 0; d < f.dk; ++d) {
          w[i * f.dk + d] += tij * w_beta * std::exp2(gamma[j]) *
                             f.k[(begin + j) * f.dk + d];
        }
      }
    }

    for (int i = 0; i < valid; ++i) {
      for (int d = 0; d < f.dv; ++d) {
        float ws = 0.0f;
        for (int k = 0; k < f.dk; ++k) ws += w[i * f.dk + k] * r.state[k * f.dv + d];
        residual[i * f.dv + d] = u[i * f.dv + d] - ws;
      }
    }

    for (int i = 0; i < valid; ++i) {
      for (int d = 0; d < f.dv; ++d) {
        float inter = 0.0f;
        for (int k = 0; k < f.dk; ++k) {
          inter += f.q[(begin + i) * f.dk + k] * r.state[k * f.dv + d];
        }
        inter *= f.scale * std::exp2(gamma[i]);
        float intra = 0.0f;
        int const last = plant.exclude_causal_diagonal ? i - 1 : i;
        for (int j = 0; j <= last; ++j) {
          float qk = 0.0f;
          for (int k = 0; k < f.dk; ++k) {
            qk += f.q[(begin + i) * f.dk + k] * f.k[(begin + j) * f.dk + k];
          }
          intra += f.scale * qk * std::exp2(gamma[i] - gamma[j]) *
                   residual[j * f.dv + d];
        }
        r.output[(begin + i) * f.dv + d] = inter + intra;
      }
    }

    float gamma_last = gamma[valid - 1];
    if (plant.advance_tail_to_full_chunk && valid < C) {
      gamma_last += float(C - valid) * -0.25f;
    }
    for (float& x : r.state) x *= std::exp2(gamma_last);
    for (int i = 0; i < valid; ++i) {
      float const key_scale = std::exp2(gamma_last - gamma[i]);
      for (int k = 0; k < f.dk; ++k) {
        for (int d = 0; d < f.dv; ++d) {
          r.state[k * f.dv + d] += key_scale * f.k[(begin + i) * f.dk + k] *
                                   residual[i * f.dv + d];
        }
      }
    }
  }
  return r;
}

float max_diff(std::vector<float> const& a, std::vector<float> const& b) {
  if (a.size() != b.size()) return std::numeric_limits<float>::infinity();
  float out = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) out = std::max(out, std::abs(a[i] - b[i]));
  return out;
}

bool check_inverse() {
  bool ok = true;
  for (int n : {16, 32, 64}) {
    std::vector<float> lower(n * n, 0.0f), inverse(n * n, 0.0f);
    for (int i = 1; i < n; ++i) {
      for (int j = 0; j < i; ++j) lower[i * n + j] = tagged(i, j, n, 11, 0.0005f);
    }
    ok &= invert_unit_lower_blocked(lower.data(), inverse.data(), n, 16);
    float const residual = unit_lower_inverse_residual(lower.data(), inverse.data(), n);
    std::printf("[L203 inverse] n=%d base=16 residual=%g %s\n", n, residual,
                residual < 2.0e-6f ? "PASS" : "FAIL");
    ok &= residual < 2.0e-6f;
  }
  return ok;
}

bool check_scheduler_and_admission() {
  using Traits = PpuChunkedGdnTraits<64, 128, 128>;
  using Scheduler = PpuChunkedGdnScheduler<Traits>;
  static_assert(
      Scheduler::ceil_div(std::numeric_limits<std::int32_t>::max(), 64) ==
          33554432,
      "scheduler ceil-div must not overflow at INT32_MAX");
  std::uint16_t dummy16 = 0;
  float dummy32 = 0.0f;
  PpuChunkedGdnArguments<std::uint16_t> a{};
  a.q = a.k = a.v = &dummy16;
  a.output = &dummy16;
  a.gamma_log2_cumsum = a.beta = &dummy32;
  a.problem = PpuChunkedGdnProblem{3 * 129, 3, 129, 2, 6, 128, 128, 64};
  bool ok = can_implement_ppu_chunked_gdn<Traits>(a) == PpuChunkedGdnStatus::kSuccess;
  std::array<int, 18> seen{};
  for (int block = 0; block < Scheduler::grid_size(a.problem); ++block) {
    auto const w = Scheduler::work(block, a.problem);
    ok &= w.valid && w.sequence_idx == block / 6 && w.v_head_idx == block % 6 &&
          w.qk_head_idx == (block % 6) / 3 && w.token_begin == (block / 6) * 129 &&
          w.token_count == 129 && w.chunk_count == 3;
    if (w.valid) ++seen[w.sequence_idx * 6 + w.v_head_idx];
  }
  for (int x : seen) ok &= x == 1;

  a.cu_seqlens = reinterpret_cast<std::int32_t const*>(&dummy32);
  ok &= can_implement_ppu_chunked_gdn<Traits>(a) == PpuChunkedGdnStatus::kInvalidSequenceLayout;
  a.cu_seqlens = nullptr;
  a.problem.num_v_heads = 5;
  ok &= can_implement_ppu_chunked_gdn<Traits>(a) == PpuChunkedGdnStatus::kUnsupportedHeadMapping;
  a.problem = PpuChunkedGdnProblem{
      std::numeric_limits<std::int32_t>::max(), 1,
      std::numeric_limits<std::int32_t>::max(),
      std::numeric_limits<std::int32_t>::max(),
      std::numeric_limits<std::int32_t>::max(), 128, 128, 64};
  ok &= can_implement_ppu_chunked_gdn<Traits>(a) == PpuChunkedGdnStatus::kInvalidProblem;
  std::printf("[L203 scheduler] work_tiles=18 exact_once=%d gva_map=3:1 fail_closed=%d %s\n",
              ok ? 1 : 0, ok ? 1 : 0, ok ? "PASS" : "FAIL");
  return ok;
}

bool check_chunk_equivalence() {
  bool ok = true;
  int cases = 0;
  float worst_output = 0.0f, worst_state = 0.0f;
  for (int chunk : {16, 32, 64}) {
    for (int t : {1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 79, 127, 129}) {
      for (bool initial : {false, true}) {
        Fixture const f = make_fixture(t, 8, 8, chunk, initial);
        Result const token = token_recurrence(f);
        Result const tiled = chunk_wy(f);
        float const od = max_diff(token.output, tiled.output);
        float const sd = max_diff(token.state, tiled.state);
        worst_output = std::max(worst_output, od);
        worst_state = std::max(worst_state, sd);
        ok &= od < 2.0e-4f && sd < 2.0e-4f;
        ++cases;
      }
    }
  }
  // Bind the algebra proof to the first shipping specialization as well as to
  // the cheap exhaustive tail grid above.  This is intentionally a complete
  // 128x128 state, not an 8x8 proxy that could hide a head-dimension stride.
  {
    Fixture const f = make_fixture(65, 128, 128, 64, true);
    Result const token = token_recurrence(f);
    Result const tiled = chunk_wy(f);
    float const od = max_diff(token.output, tiled.output);
    float const sd = max_diff(token.state, tiled.state);
    worst_output = std::max(worst_output, od);
    worst_state = std::max(worst_state, sd);
    ok &= od < 2.0e-4f && sd < 2.0e-4f;
    ++cases;
  }
  std::printf("[L203 algebra] cases=%d chunks=16,32,64 tails=EXHAUSTIVE-LIST "
              "production=64x128x128 worst_output=%g worst_state=%g %s\n",
              cases, worst_output, worst_state, ok ? "PASS" : "FAIL");
  return ok;
}

bool check_plants() {
  Fixture const f = make_fixture(23, 8, 8, 16, true);
  Result const good = token_recurrence(f);
  std::array<Plant, 4> plants{{
      Plant{true, false, false, false},
      Plant{false, true, false, false},
      Plant{false, false, true, false},
      Plant{false, false, false, true},
  }};
  std::array<char const*, 4> names{{
      "causal-diagonal-removed", "beta-omitted-from-W", "decay-difference-reversed",
      "tail-advances-through-padding"}};
  bool ok = true;
  for (std::size_t i = 0; i < plants.size(); ++i) {
    Result const bad = chunk_wy(f, plants[i]);
    float const diff = std::max(max_diff(good.output, bad.output), max_diff(good.state, bad.state));
    bool const red = diff > 1.0e-4f;
    ok &= red;
    std::printf("[L203 plant] name=%s diff=%g %s\n", names[i], diff,
                red ? "EXPECTED_RED/PASS" : "FAIL");
  }
  return ok;
}

}  // namespace

int main() {
  bool const inverse = check_inverse();
  bool const scheduler = check_scheduler_and_admission();
  bool const algebra = check_chunk_equivalence();
  bool const plants = check_plants();
  bool const ok = inverse && scheduler && algebra && plants;
  std::printf("[L203] %s: pure-C++ GDN recurrence == chunk/WY; inverse/scheduler/tails/plants closed\n",
              ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
