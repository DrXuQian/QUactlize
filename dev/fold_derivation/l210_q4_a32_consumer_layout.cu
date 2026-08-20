// Q4/A32 producer -> consumer byte-layout compatibility.
//
// A producer self-roundtrip cannot prove that a different tactic reads the
// same bytes.  This probe makes one canonical A32/FoldN=2 artifact, then asks
// the real xplane maps for representative points from every TN/WN equivalence
// class exposed by the finite tactic axes.  The exact device failure
// (TN32/TK64/WN16) is retained as a negative witness.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "xplane_offline.hpp"

namespace {

constexpr int N = 256;
constexpr int K = 256;
constexpr std::size_t Codes = std::size_t(N) * K;

struct Result {
  std::size_t byte_bad = 0;
  std::size_t code_bad = 0;
};

template <int TM, int TN, int TK, int WM, int WN>
Result cross(std::vector<std::uint8_t> const& logical,
             std::vector<std::int8_t> const& canonical) {
  std::vector<std::int8_t> candidate(Codes / 2);
  std::vector<std::uint8_t> recovered;
  xplane::place_derived<4, TM, TN, TK, WM, WN, 2, 32>(
      candidate.data(), logical, N, K);
  xplane::recover_derived<4, TM, TN, TK, WM, WN, 2, 32>(
      canonical.data(), recovered, N, K);
  Result result;
  for (std::size_t i = 0; i < candidate.size(); ++i)
    result.byte_bad += candidate[i] != canonical[i];
  for (std::size_t i = 0; i < logical.size(); ++i)
    result.code_bad += logical[i] != recovered[i];
  return result;
}

template <int TM, int TN, int TK, int WM, int WN>
bool row(char const* name, bool expect_compatible,
         std::vector<std::uint8_t> const& logical,
         std::vector<std::int8_t> const& canonical,
         Result* observed = nullptr) {
  Result const result = cross<TM, TN, TK, WM, WN>(logical, canonical);
  bool const rule = WN >= 32 && TN == 2 * WN;
  bool const measured = result.byte_bad == 0 && result.code_bad == 0;
  if (observed) *observed = result;
  std::printf("[l210 row] name=%s tm=%d tn=%d tk=%d wm=%d wn=%d "
              "rule=%d compatible=%d byte_bad=%zu/%zu code_bad=%zu/%zu\n",
              name, TM, TN, TK, WM, WN, int(rule), int(measured),
              result.byte_bad, Codes / 2, result.code_bad, Codes);
  return rule == expect_compatible && measured == expect_compatible;
}

}  // namespace

int main() {
  std::vector<std::uint8_t> logical(Codes);
  for (int k = 0; k < K; ++k)
    for (int n = 0; n < N; ++n)
      logical[std::size_t(k) * N + n] =
          std::uint8_t(((13 * n + 7 * k + 3) % 15) + 1);
  std::vector<std::int8_t> canonical(Codes / 2);
  xplane::place_derived<4, 64, 64, 32, 32, 32, 2, 32>(
      canonical.data(), logical, N, K);

  int rows = 0, positives = 0, negatives = 0;
  bool ok = true;
#define POS(NAME, TM, TN, TK, WM, WN)                                      \
  do { ++rows; ++positives; ok &= row<TM, TN, TK, WM, WN>(                 \
      NAME, true, logical, canonical); } while (0)
#define NEG(NAME, TM, TN, TK, WM, WN)                                      \
  do { ++rows; ++negatives; ok &= row<TM, TN, TK, WM, WN>(                 \
      NAME, false, logical, canonical); } while (0)

  // Both compatible reader classes, crossed with the smallest/largest
  // TacticTileK and different M-warp tilings.
  POS("canonical-a", 64, 64, 32, 32, 32);
  POS("canonical-large-t", 16, 64, 256, 16, 32);
  POS("scaled-a", 64, 128, 32, 32, 64);
  POS("scaled-large-t", 128, 128, 256, 64, 64);

  Result exact_device;
  ++rows; ++negatives;
  ok &= row<32, 32, 64, 16, 16>("exact-device-failure", false,
                                 logical, canonical, &exact_device);
  NEG("physical-n-too-small", 64, 32, 64, 32, 32);
  NEG("too-many-n-warps", 64, 64, 64, 32, 16);
  NEG("too-few-n-warps", 64, 64, 64, 32, 64);
  NEG("second-reader-instance", 64, 128, 64, 32, 32);
  NEG("larger-tile-same-wn", 64, 256, 64, 32, 64);

#undef POS
#undef NEG

  // A one-bit physical corruption must also be visible through the positive
  // reader.  Otherwise byte equality above could be a constant-output test.
  auto corrupt = canonical;
  corrupt[0] ^= 1;
  std::vector<std::uint8_t> corrupt_back;
  xplane::recover_derived<4, 64, 64, 32, 32, 32, 2, 32>(
      corrupt.data(), corrupt_back, N, K);
  std::size_t corruption_bad = 0;
  for (std::size_t i = 0; i < logical.size(); ++i)
    corruption_bad += logical[i] != corrupt_back[i];
  ok &= corruption_bad > 0;
  ok &= rows == 10 && positives == 4 && negatives == 6;
  ok &= exact_device.byte_bad == Codes / 2 &&
        exact_device.code_bad == Codes;

  std::printf("[l210] %s rows=%d positives=%d negatives=%d "
              "exact_device_byte_bad=%zu/%zu exact_device_code_bad=%zu/%zu "
              "one_bit_corruption_bad=%zu\n",
              ok ? "PASS" : "FAIL", rows, positives, negatives,
              exact_device.byte_bad, Codes / 2,
              exact_device.code_bad, Codes, corruption_bad);
  return ok ? 0 : 1;
}
