// L141 -- turn L138's two-source 2N x 4K proof into an offline artifact API.
//
// This is deliberately narrower than the topology oracle.  It proves exactly
// what a benchmark/producer is allowed to consume:
//   * WarpK==TileK calls the shipping placement and is byte-identical;
//   * the one admitted WK4 map is the L138 16,384-entry bijection/hash;
//   * place/recover through that map is an exact inverse;
//   * decoding a stale WK1 artifact as WK4 is observably wrong.
//
// The negative is load-bearing.  WK is an artifact descriptor axis (like
// TileK and fold), not a new quantization format, but it still changes bytes.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "xplane_offline.hpp"

namespace {

constexpr int kBits = 4;
constexpr int kTM = 16;
constexpr int kTN = 128;
constexpr int kTK = 128;
constexpr int kWM = 16;
constexpr int kWN = 64;
constexpr int kFold = 1;
constexpr int kArtifactTK = 64;
constexpr int kWarpK1 = 128;
constexpr int kWarpK4 = 32;
constexpr int kN = 128;
// F=1 uses the shipping interleave-256 resident row.  Use two tactic tiles so
// the allocation is the real 256-code physical pitch rather than an invalid
// compact K=128 buffer (the exact harness bug caught in #112).
constexpr int kK = 256;

#if defined(L141_NEGATIVE_UNPROVED_FORMAT)
// A smaller WarpK is not a license to reuse this int4 converter proof for a
// different emission order.  The runner requires this arm to fail at compile
// time, before an artifact can be written.
auto unproved_int2 = xplane::plane_map_warp_k<
    2, kTM, kTN, kTK, kWM, kWN, kFold, kWarpK4, kArtifactTK>();
#endif

uint64_t hash_map(std::vector<int> const& map) {
  uint64_t h = UINT64_C(1469598103934665603);
  auto word = [&](uint64_t x) {
    for (int i = 0; i < 8; ++i) {
      h ^= (x >> (8 * i)) & 0xffu;
      h *= UINT64_C(1099511628211);
    }
  };
  for (int i = 0; i < int(map.size()); ++i) {
    word(uint64_t(i));
    word(uint64_t(map[i] + 1));
  }
  return h;
}

struct MapMetric {
  int holes = 0;
  int duplicates = 0;
  int out_of_range = 0;
};

MapMetric metric(std::vector<int> const& map) {
  MapMetric m;
  std::vector<int> hits(map.size(), 0);
  for (int x : map) {
    if (x < 0) {
      ++m.holes;
    } else if (x >= int(map.size())) {
      ++m.out_of_range;
    } else {
      ++hits[x];
    }
  }
  for (int n : hits) m.duplicates += std::max(0, n - 1);
  return m;
}

template <class T>
int differences(std::vector<T> const& a, std::vector<T> const& b) {
  if (a.size() != b.size()) return -1;
  int result = 0;
  for (size_t i = 0; i < a.size(); ++i) result += a[i] != b[i];
  return result;
}

} // namespace

int main() {
  auto shipping_map = xplane::plane_map<
      kBits, kTM, kTN, kTK, kWM, kWN, kFold, kArtifactTK>();
  auto default_map = xplane::plane_map_warp_k<
      kBits, kTM, kTN, kTK, kWM, kWN, kFold, kWarpK1,
      kArtifactTK>();
  auto wk4_map = xplane::plane_map_warp_k<
      kBits, kTM, kTN, kTK, kWM, kWN, kFold, kWarpK4,
      kArtifactTK>();

  auto mm = metric(wk4_map);
  int default_map_diff = differences(shipping_map, default_map);
  int wk4_map_diff = differences(shipping_map, wk4_map);
  uint64_t wk4_hash = hash_map(wk4_map);

  // Mix all coordinate bits.  A small-period pattern can alias a real
  // displacement (L61 documents exactly that false green), so use an
  // avalanche hash and then retain the low nibble.
  std::vector<uint8_t> q(size_t(kN) * kK);
  for (int k = 0; k < kK; ++k)
    for (int n = 0; n < kN; ++n) {
      uint32_t x = uint32_t(k * kN + n) + UINT32_C(0x9e3779b9);
      x ^= x >> 16;
      x *= UINT32_C(0x7feb352d);
      x ^= x >> 15;
      x *= UINT32_C(0x846ca68b);
      x ^= x >> 16;
      q[size_t(k) * kN + n] = uint8_t(x & 15);
    }

  std::vector<int8_t> shipping(size_t(kN) * kK * kBits / 8);
  std::vector<int8_t> default_wk1(shipping.size());
  std::vector<int8_t> wk4(shipping.size());
  xplane::place_derived<kBits, kTM, kTN, kTK, kWM, kWN, kFold,
                        kArtifactTK>(shipping.data(), q, kN, kK);
  xplane::place_derived_warp_k<
      kBits, kTM, kTN, kTK, kWM, kWN, kFold, kWarpK1,
      kArtifactTK>(default_wk1.data(), q, kN, kK);
  xplane::place_derived_warp_k<
      kBits, kTM, kTN, kTK, kWM, kWN, kFold, kWarpK4,
      kArtifactTK>(wk4.data(), q, kN, kK);

  std::vector<uint8_t> recovered;
  std::vector<uint8_t> stale_recovered;
  xplane::recover_derived_warp_k<
      kBits, kTM, kTN, kTK, kWM, kWN, kFold, kWarpK4,
      kArtifactTK>(wk4.data(), recovered, kN, kK);
  xplane::recover_derived_warp_k<
      kBits, kTM, kTN, kTK, kWM, kWN, kFold, kWarpK4,
      kArtifactTK>(shipping.data(), stale_recovered, kN, kK);

  int default_byte_diff = differences(shipping, default_wk1);
  int wk4_byte_diff = differences(shipping, wk4);
  int roundtrip_bad = differences(q, recovered);
  int stale_bad = differences(q, stale_recovered);

  std::printf("L141 map default-diff=%d WK4-diff=%d entries=%zu "
              "holes=%d duplicates=%d out-of-range=%d hash=%016llx\n",
              default_map_diff, wk4_map_diff, wk4_map.size(), mm.holes,
              mm.duplicates, mm.out_of_range,
              (unsigned long long)wk4_hash);
  std::printf("L141 bytes default-diff=%d WK4-diff=%d/%zu "
              "roundtrip-bad=%d/%zu stale-WK1-as-WK4-bad=%d/%zu\n",
              default_byte_diff, wk4_byte_diff, wk4.size(), roundtrip_bad,
              q.size(), stale_bad, q.size());

  bool ok = default_map_diff == 0 && default_byte_diff == 0 &&
            wk4_map.size() == size_t(kTN) * kTK && mm.holes == 0 &&
            mm.duplicates == 0 && mm.out_of_range == 0 &&
            wk4_hash == UINT64_C(0x17dfe6248fc38143) &&
            wk4_map_diff > 0 && wk4_byte_diff > 0 &&
            roundtrip_bad == 0 && stale_bad > 0;
  std::printf("L141 WK1=SHIPPING-BYTE-IDENTICAL WK4=BIJECTIVE+ROUNDTRIP "
              "stale-WK1=%s result=%s\n",
              stale_bad > 0 ? "EXPECTED-RED" : "UNEXPECTED-GREEN",
              ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
