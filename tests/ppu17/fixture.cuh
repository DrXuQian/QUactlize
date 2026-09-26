#pragma once

// Shared CPU fixture and typed arithmetic oracle; no implicit GPU launches.
#include <cuda_runtime.h>
#include "io.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

using namespace quactlize::ppu17;

inline void check(int rc, char const *where) {
  if (rc) throw std::runtime_error(std::string(where) + ": " + q17_last_error_v1());
}
inline void gpu(cudaError_t rc, char const *where) {
  if (rc != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(rc));
}
struct DeviceMemory {
  void *p = nullptr;
  explicit DeviceMemory(size_t bytes, void const *host) {
    if (!bytes) return;
    gpu(cudaMalloc(&p, bytes), "allocation");
    auto rc = cudaMemcpy(p, host, bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) { cudaFree(p); p = nullptr; gpu(rc, "upload"); }
  }
  ~DeviceMemory() { if (p) cudaFree(p); }
  DeviceMemory(DeviceMemory const &) = delete;
};
struct Options {
  int q = 12, compute = Q17_BF16, input = Q17_BF16, output = Q17_F32;
  int config = 5, m = 9, n = 256, k = 512;
  bool grouped = false, sf = false, expansion = false, cpu_only = false;
  std::vector<int> rows;
};

template <int Q, class T> struct Fixture {
  using F = Format<Q>;
  static constexpr int cbias = Q == 8 ? 128 : (F::low_bits == 4 ? 8 : 0);
  static constexpr int qbias = Q == 8 ? 128 : Q == 11 ? 4 : Q == 14 ? 32 : 0;
  int n, k, experts;
  std::vector<uint8_t> low, high, units;
  std::vector<T> scale, zero;

  static int raw(int e, int n, int k) {
    return (e * 11 + n * 3 + k * 7) & ((1 << (F::low_bits + F::high_bits)) - 1);
  }
  static Affine<T> oracle_affine(int e, int col, int group) {
    float d = std::ldexp(float(1 + (col + e) % 3), -10);
    int sc = Q == 8 ? 1 : 1 + (col + group + e) % 7;
    if constexpr (Q == 11 || Q == 14) sc -= 4;
    T s(d * float(sc)), z(0.f);
    if constexpr (Q == 10 || Q == 12 || Q == 13)
      z = T(-std::ldexp(float((col + group + e) % 4), -11));
    if constexpr (cbias != qbias) {
      if constexpr (std::is_same_v<T, cutlass::bfloat16_t>)
        z = T(float(z) + float(T(float(cbias - qbias) * float(s))));
      else z = T(float(z) + float(cbias - qbias) * float(s));
    }
    return {s, z};
  }
  static T oracle_weight(int e, int col, int k) {
    auto a = oracle_affine(e, col, k / F::group_size);
    T product(float(raw(e, col, k) - cbias) * float(a.scale));
    return T(float(product) + float(a.zero));
  }
  Fixture(int n_, int k_, int e_) : n(n_), k(k_), experts(e_),
      low(size_t(n) * k * experts * F::low_bits / 8),
      high(size_t(n) * k * experts * F::high_bits / 8),
      units(F::unit_bytes(n, k) * experts),
      scale(size_t(experts) * n * (k / F::group_size)), zero(scale.size()) {
    for (int e = 0; e < experts; ++e) {
      auto l = low.data() + size_t(e) * n * k * F::low_bits / 8;
      auto u = units.data() + size_t(e) * F::unit_bytes(n, k);
      for (int col = 0; col < n; ++col) for (int kk = 0; kk < k; ++kk) {
        int value = raw(e, col, kk);
        F::Low::placed_put(l, col, kk, n, value & ((1 << F::low_bits) - 1));
        if constexpr (F::high_bits)
          F::High::placed_put(high.data() + size_t(e) * n * k * F::high_bits / 8,
                             col, kk, n, value >> F::low_bits);
      }
      for (int col = 0; col < n; ++col) for (int g = 0; g < k / F::group_size; ++g) {
        auto affine = oracle_affine(e, col, g);
        size_t index = (size_t(e) * (k / F::group_size) + g) * n + col;
        scale[index] = affine.scale; zero[index] = affine.zero;
        auto d = cutlass::half_t(std::ldexp(float(1 + (col + e) % 3), -10));
        if constexpr (Q == 8) {
          size_t off = 2 * (size_t(g) * n + col);
          u[off] = d.raw() & 255; u[off + 1] = d.raw() >> 8;
        } else {
          auto p = u + F::unit_offset(n, col, g);
          p[0] = d.raw() & 255; p[1] = d.raw() >> 8;
          int sc = 1 + (col + g + e) % 7;
          if constexpr (Q == 11 || Q == 14) sc -= 4;
          sc += F::Traits::kScaleBias;
          auto put = [&](int which, int bits, int value) {
            int bit = F::field_bit(g % F::Traits::kGroups, which);
            value &= (1 << bits) - 1;
            for (int b = 0; b < bits; ++b) if (value & (1 << b)) p[(bit + b) / 8] |= 1 << ((bit + b) & 7);
          };
          put(0, F::Traits::kScaleBits, sc);
          if constexpr (F::Traits::kHasMin) {
            auto dm = cutlass::half_t(std::ldexp(1.f, -11));
            p[2] = dm.raw() & 255; p[3] = dm.raw() >> 8;
            put(1, F::Traits::kMinBits, (col + g + e) % 4);
          }
        }
      }
    }
  }
};
