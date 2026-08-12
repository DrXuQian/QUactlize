// One logical Q4_K problem, packed into both operands used by the INBOX 132B A/B.
// The raw block is the PDF arm's [N][K/256] representation.  The affine arm is
// gemv_lowbit's production Native representation: an [N][K] int4 code plane
// plus fp16 S/Z in [K/32][N].  Both are decoded independently below before a
// CUDA kernel is allowed to enter timing.
#pragma once

#include "q4k_pdf_reconstruction.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace q4k_pdf_ab {

using q4k_pdf_reconstruction::block_q4_K;

struct Shape {
  char const* id;
  int l;
  int n;
  int k;
  int pdf_cta_n;
  int pdf_warps_n;
  int pdf_warps_k;
  bool document_winner;
};

inline constexpr Shape kShapes[] = {
    {"D-EXT-O", 1, 5120, 8192, 2, 8, 1, true},
    {"D-EXT-K1024", 1, 5120, 1024, 2, 8, 1, true},
    {"D-EXT-Q", 1, 8192, 5120, 4, 8, 1, true},
    // The document contains no result for this shape.  It uses the documented
    // default rather than pretending a new configuration is a paper winner.
    {"H-G8-2048", 8, 2048, 2048, 2, 8, 1, false},
};

inline std::uint16_t half_bits(half h) {
  std::uint16_t out;
  std::memcpy(&out, &h, sizeof(out));
  return out;
}

inline float half_float(half h) { return __half2float(h); }

inline half make_half(float x) { return __float2half_rn(x); }

inline void set_scale_min(block_q4_K& b, int g, int sc, int mn) {
  if (g < 4) {
    b.scales[g] = std::uint8_t((b.scales[g] & 0xc0u) | (sc & 63));
    b.scales[4 + g] = std::uint8_t((b.scales[4 + g] & 0xc0u) | (mn & 63));
  } else {
    int const t = g - 4;
    b.scales[8 + t] = std::uint8_t((sc & 15) | ((mn & 15) << 4));
    b.scales[t] = std::uint8_t((b.scales[t] & 0x3fu) | ((sc >> 4) << 6));
    b.scales[4 + t] = std::uint8_t((b.scales[4 + t] & 0x3fu) | ((mn >> 4) << 6));
  }
}

inline void get_scale_min(block_q4_K const& b, int g, int& sc, int& mn) {
  int const t = g & 3;
  if (g < 4) {
    sc = b.scales[t] & 63;
    mn = b.scales[4 + t] & 63;
  } else {
    sc = (b.scales[8 + t] & 15) | ((b.scales[t] >> 6) << 4);
    mn = (b.scales[8 + t] >> 4) | ((b.scales[4 + t] >> 6) << 4);
  }
}

inline void set_q(block_q4_K& b, int i, int q) {
  int const byte = (i / 64) * 32 + (i % 32);
  int const shift = 4 * ((i % 64) / 32);
  b.qs[byte] = std::uint8_t((b.qs[byte] & ~(15u << shift)) | ((q & 15) << shift));
}

inline int get_q(block_q4_K const& b, int i) {
  int const byte = (i / 64) * 32 + (i % 32);
  int const shift = 4 * ((i % 64) / 32);
  return (b.qs[byte] >> shift) & 15;
}

struct HostProblem {
  Shape shape{};
  std::vector<half> act;
  std::vector<block_q4_K> raw;
  std::vector<std::uint8_t> low;
  std::vector<half> scales;
  std::vector<half> zeros;
  std::vector<half> golden;
};

inline HostProblem make_problem(Shape shape) {
  HostProblem p;
  p.shape = shape;
  int const bpr = shape.k / 256;
  int const groups = shape.k / 32;
  p.act.resize(std::size_t(shape.l) * shape.k);
  p.raw.resize(std::size_t(shape.l) * shape.n * bpr);
  p.low.assign(std::size_t(shape.l) * shape.n * shape.k / 2, 0);
  p.scales.resize(std::size_t(shape.l) * groups * shape.n);
  p.zeros.resize(std::size_t(shape.l) * groups * shape.n);
  p.golden.resize(std::size_t(shape.l) * shape.n);

  for (int e = 0; e < shape.l; ++e)
    for (int k = 0; k < shape.k; ++k)
      p.act[std::size_t(e) * shape.k + k] = make_half(float(1 + ((5 * e + 3 * k) & 3)) / 64.f);

  for (int e = 0; e < shape.l; ++e) {
    for (int n = 0; n < shape.n; ++n) {
      for (int sb = 0; sb < bpr; ++sb) {
        block_q4_K& b = p.raw[(std::size_t(e) * shape.n + n) * bpr + sb];
        std::memset(&b, 0, sizeof(b));
        // Powers of two keep the representation comparison numerically well
        // conditioned while the n/sb/group salts still exercise every address.
        b.d = make_half(((e + n + sb) & 1) ? 1.f / 128.f : 1.f / 256.f);
        b.dmin = make_half(((3 * e + n + sb) & 1) ? 1.f / 512.f : 1.f / 1024.f);
        for (int g = 0; g < 8; ++g) {
          int const sc = 1 + ((e + 3 * n + 5 * sb + g) & 3);
          int const mn = (7 * e + n + sb + 3 * g) & 3;
          set_scale_min(b, g, sc, mn);
          std::size_t const si = (std::size_t(e) * groups + sb * 8 + g) * shape.n + n;
          p.scales[si] = make_half(half_float(b.d) * float(sc));
          p.zeros[si] = make_half(-half_float(b.dmin) * float(mn));
        }
        for (int i = 0; i < 256; ++i) {
          int const k = sb * 256 + i;
          int const q = (11 * e + 7 * n + 3 * k + 5 * (k / 32)) & 15;
          set_q(b, i, q);
          std::size_t const linear = (std::size_t(e) * shape.n + n) * shape.k + k;
          p.low[linear >> 1] |= std::uint8_t(q << (4 * (linear & 1)));
        }
      }
    }
  }

  // Independent decode from raw blocks.  Do not reuse low/S/Z: this is the
  // anchor that catches a packer which is internally self-consistent but wrong.
  for (int e = 0; e < shape.l; ++e) {
    for (int n = 0; n < shape.n; ++n) {
      double sum = 0.0;
      for (int sb = 0; sb < bpr; ++sb) {
        block_q4_K const& b = p.raw[(std::size_t(e) * shape.n + n) * bpr + sb];
        for (int i = 0; i < 256; ++i) {
          int sc = 0, mn = 0;
          get_scale_min(b, i / 32, sc, mn);
          float const w = float(get_q(b, i)) * half_float(b.d) * float(sc)
                        - half_float(b.dmin) * float(mn);
          sum += double(half_float(p.act[std::size_t(e) * shape.k + sb * 256 + i])) * double(w);
        }
      }
      p.golden[std::size_t(e) * shape.n + n] = make_half(float(sum));
    }
  }
  return p;
}

inline bool verify_representation(HostProblem const& p, std::string& why) {
  Shape const& s = p.shape;
  int const bpr = s.k / 256;
  int const groups = s.k / 32;
  std::size_t bad_q = 0, bad_affine = 0;
  for (int e = 0; e < s.l; ++e) {
    for (int n = 0; n < s.n; ++n) {
      for (int sb = 0; sb < bpr; ++sb) {
        block_q4_K const& b = p.raw[(std::size_t(e) * s.n + n) * bpr + sb];
        for (int i = 0; i < 256; ++i) {
          int const k = sb * 256 + i;
          std::size_t const linear = (std::size_t(e) * s.n + n) * s.k + k;
          int const packed = (p.low[linear >> 1] >> (4 * (linear & 1))) & 15;
          bad_q += packed != get_q(b, i);
          int sc = 0, mn = 0;
          get_scale_min(b, i / 32, sc, mn);
          std::size_t const si = (std::size_t(e) * groups + k / 32) * s.n + n;
          float const raw = get_q(b, i) * half_float(b.d) * float(sc)
                          - half_float(b.dmin) * float(mn);
          float const affine = packed * half_float(p.scales[si]) + half_float(p.zeros[si]);
          bad_affine += raw != affine;
        }
      }
    }
  }
  if (bad_q || bad_affine) {
    why = "raw/native representation mismatch q=" + std::to_string(bad_q)
        + " affine=" + std::to_string(bad_affine);
    return false;
  }
  return true;
}

inline std::uint64_t fnv1a_half(std::vector<half> const& v) {
  std::uint64_t h = UINT64_C(1469598103934665603);
  for (half x : v) {
    std::uint16_t bits = half_bits(x);
    h ^= bits & 0xffu; h *= UINT64_C(1099511628211);
    h ^= bits >> 8;    h *= UINT64_C(1099511628211);
  }
  return h;
}

}  // namespace q4k_pdf_ab
