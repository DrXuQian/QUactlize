// L241 -- independent host ABI closure for the canonical Q4 N16 x K64 direct
// checkpoint layout.
//
// The expected byte stream below deliberately does not call the production
// forward/inverse address functions.  It repeats L240's committed semantic
// bit permutation and outer [K/16][2*N] u32 composition as the independent
// oracle, then compares the production header at N32xK128 and N64xK256.

#include "q4_n16k64_direct_offline.hpp"
#include "q4_kpack4_offline.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

struct LogicalCoordinate {
  int n;
  int k;
};

constexpr int oracle_atom_physical_nibble(int n, int k) {
#if defined(L241_PLANT_WRONG_BITPERM) && L241_PLANT_WRONG_BITPERM
  // Bijective and byte-count preserving, but swaps p0:k3 with p1:k4.
  return ((k >> 4) & 1) * 1 +
         ((k >> 3) & 1) * 2 +
#else
  return ((k >> 3) & 1) * 1 +
         ((k >> 4) & 1) * 2 +
#endif
         ((k >> 0) & 1) * 4 +
         ((k >> 5) & 1) * 8 +
         ((n >> 3) & 1) * 16 +
         ((k >> 1) & 1) * 32 +
         ((k >> 2) & 1) * 64 +
         ((n >> 0) & 1) * 128 +
         ((n >> 1) & 1) * 256 +
         ((n >> 2) & 1) * 512;
}

constexpr std::size_t oracle_direct_physical_nibble(int n, int k,
                                                     int n_extent) {
  int const local_p = oracle_atom_physical_nibble(n % 16, k % 64);
  int const local_word = local_p / 8;
  int const word_nibble = local_p % 8;
  int const local_n_word = local_word % 32;
  int const local_k_row = local_word / 32;
  int const physical_n_word = (n / 16) * 32 + local_n_word;
  int const physical_k_row = (k / 64) * 4 + local_k_row;
  std::size_t const word =
      std::size_t(physical_k_row) * std::size_t(2 * n_extent) +
      std::size_t(physical_n_word);
  return 8 * word + std::size_t(word_nibble);
}

constexpr std::size_t oracle_layout1_physical_nibble(int n, int k,
                                                      int n_extent) {
  return std::size_t(q4_kpack4::kPack) *
             q4_kpack4::word_index(n, k, n_extent) +
         std::size_t(q4_kpack4::nibble(k));
}

constexpr std::size_t oracle_physical_nibble(int n, int k, int n_extent) {
#if defined(L241_PLANT_OLD_LAYOUT1) && L241_PLANT_OLD_LAYOUT1
  return oracle_layout1_physical_nibble(n, k, n_extent);
#else
  return oracle_direct_physical_nibble(n, k, n_extent);
#endif
}

void put_nibble(std::vector<std::uint8_t>& bytes, std::size_t physical,
                std::uint8_t value) {
  unsigned const shift = unsigned(4 * (physical & 1));
  std::uint8_t const mask = std::uint8_t(0xfu << shift);
  bytes[physical >> 1] = std::uint8_t(
      (bytes[physical >> 1] & std::uint8_t(~mask)) |
      ((value & 0xfu) << shift));
}

std::uint8_t get_nibble(std::vector<std::uint8_t> const& bytes,
                        std::size_t physical) {
  return std::uint8_t(
      (bytes[physical >> 1] >> (4 * (physical & 1))) & 0xfu);
}

int bit_count(std::uint8_t value) {
  int count = 0;
  for (int bit = 0; bit < 8; ++bit) count += (value >> bit) & 1u;
  return count;
}

struct Metrics {
  int map_equal = 0;
  int total = 0;
  int production_bijection_bad = 0;
  int oracle_bijection_bad = 0;
  int inverse_bad = 0;
  int placed_bit_bad = 0;
  int get_bad = 0;
  int roundtrip_bit_bad = 0;
  int layout1_equal = 0;
  std::size_t bytes = 0;
  bool api_exact = false;
};

template <int N, int K>
Metrics prove_shape() {
  static_assert(N % 16 == 0 && K % 64 == 0);
  constexpr int Codes = N * K;
  constexpr std::size_t Bytes = std::size_t(Codes) / 2;

  Metrics m{};
  m.bytes = q4_n16k64_direct::placed_bytes(N, K);
  std::vector<int> production_hits(Codes, 0);
  std::vector<int> oracle_hits(Codes, 0);
  std::vector<std::uint8_t> native(Bytes, 0);
  std::vector<std::uint8_t> expected(Bytes, 0);
  std::vector<std::uint8_t> placed(Bytes, 0xcd);
  std::vector<std::uint8_t> recovered(Bytes, 0xab);

  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      std::uint8_t const code = std::uint8_t(
          (13 * n + 7 * k + 5 * (k / 8) + (n ^ k) + 1) & 0xf);
      put_nibble(native, std::size_t(n) * K + k, code);

      std::size_t const production =
          q4_n16k64_direct::physical_nibble(n, k, N);
      std::size_t const oracle = oracle_physical_nibble(n, k, N);
      std::size_t const layout1 =
          oracle_layout1_physical_nibble(n, k, N);
      m.map_equal += production == oracle;
      m.layout1_equal += production == layout1;
      ++m.total;
      if (production < std::size_t(Codes))
        ++production_hits[production];
      else
        ++m.production_bijection_bad;
      if (oracle < std::size_t(Codes))
        ++oracle_hits[oracle];
      else
        ++m.oracle_bijection_bad;
      if (oracle < std::size_t(Codes)) put_nibble(expected, oracle, code);

      if (production < std::size_t(Codes)) {
        q4_n16k64_direct::LogicalCoordinate const inverse =
            q4_n16k64_direct::logical_coordinate(production, N);
        m.inverse_bad += inverse.n != n || inverse.k != k;
      }
    }
  }
  for (int h : production_hits) m.production_bijection_bad += h != 1;
  for (int h : oracle_hits) m.oracle_bijection_bad += h != 1;

  int const prepare_rc = q4_n16k64_direct::prepare(
      native.data(), placed.data(), N, K);
  for (std::size_t i = 0; i < Bytes; ++i)
    m.placed_bit_bad += bit_count(std::uint8_t(placed[i] ^ expected[i]));
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      m.get_bad += q4_n16k64_direct::placed_get(
                       placed.data(), n, k, N) !=
                   get_nibble(native, std::size_t(n) * K + k);
  int const recover_rc = q4_n16k64_direct::recover(
      placed.data(), recovered.data(), N, K);
  for (std::size_t i = 0; i < Bytes; ++i)
    m.roundtrip_bit_bad +=
        bit_count(std::uint8_t(recovered[i] ^ native[i]));

  m.api_exact = q4_n16k64_direct::shape_supported(N, K) &&
                m.bytes == Bytes && prepare_rc == 0 && recover_rc == 0;
  return m;
}

bool exact(Metrics const& m) {
  return m.api_exact && m.map_equal == m.total &&
         m.production_bijection_bad == 0 && m.oracle_bijection_bad == 0 &&
         m.inverse_bad == 0 && m.placed_bit_bad == 0 && m.get_bad == 0 &&
         m.roundtrip_bit_bad == 0 && m.layout1_equal != m.total;
}

template <int N, int K>
bool report() {
  Metrics const m = prove_shape<N, K>();
  bool const ok = exact(m);
  std::printf(
      "L241 OFFLINE shape=%dx%d bytes=%zu map=%d/%d "
      "production_bijection_bad=%d oracle_bijection_bad=%d inverse_bad=%d "
      "placed_bit_bad=%d get_bad=%d roundtrip_bit_bad=%d "
      "layout1_equal=%d/%d "
      "result=%s\n",
      N, K, m.bytes, m.map_equal, m.total,
      m.production_bijection_bad, m.oracle_bijection_bad, m.inverse_bad,
      m.placed_bit_bad, m.get_bad, m.roundtrip_bit_bad,
      m.layout1_equal, m.total,
      ok ? "PASS" : "FAIL");
  return ok;
}

}  // namespace

int main() {
  bool ok = report<32, 128>();
  ok &= report<64, 256>();
  ok &= q4_n16k64_direct::placed_bytes(31, 128) == 0;
  ok &= q4_n16k64_direct::placed_bytes(32, 96) == 0;
  std::printf(
      "L241 Q4_N16K64_DIRECT_OFFLINE %s layout=%u mapping=0x%016llx "
      "atom=0x%016llx shapes=2 reds=2\n",
      ok ? "PASS" : "FAIL", unsigned(q4_n16k64_direct::kLayoutId),
      static_cast<unsigned long long>(q4_n16k64_direct::kMappingId),
      static_cast<unsigned long long>(
          q4_n16k64_direct::kAtomMappingFingerprint));
  return ok ? 0 : 1;
}
