// L208 -- Q8_0's resident code-byte layout, before any benchmark or runner can claim support.
//
// There are three separate properties here, kept separate so agreement cannot be manufactured by one model:
//
//   1. EMISSION ANCHOR.  The source of truth is actlize's shipping
//        MixGemmNumericArrayConverter<half_t, int8_t, 16>.
//      Its int8x4 leaf emits source bytes (0,2,1,3), and its x16 wrapper places source groups (0,2,1,3).
//      `converter_emit()` transcribes those two explicit assignments, not MixGemmEmit's formula, and exhaustively
//      compares all 16 source positions.
//   2. PLACEMENT BIJECTION.  One Q8_0 K delivery is 32 signed bytes, hence ArtifactTileK=32 and FoldN=1.
//      Every row in the benchmark's bounded performance-envelope authority must own every (n,k) exactly once,
//      round-trip a nonzero, aperiodic byte fixture, and produce bytes identical to one independent A32/F1 anchor.
//      The authority deliberately includes WON=TileN/max(WarpN,16) values 1, 2 and 4; their anchors are compared
//      pairwise. Equality is measured evidence that WON remains a reader axis here, not an assumed invariant.
//   3. VALUE SEMANTICS.  GGUF Q8_0 stores signed q and fp16 d with W=d*q.  The resident converter consumes the
//      byte representation of q+128 and subtracts 128.  All 256 q values and four exactly representable fp16 scales
//      are checked, so signed-char reinterpretation cannot silently change the contract.
//
// L208_PLANT_WRONG_PERM flips one output-index bit for exactly one source position.  It must make the emission
// anchor red with one mismatch, one hole and one duplicate; the runner checks that exact signature.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "cutlass/half.h"
#include "actlize_extensions/cutlass/quactlize_mix_gemm_convert.h"
#include "xplane_offline.hpp"

namespace {

constexpr int permute_0213(int x) {
  return x == 1 ? 2 : x == 2 ? 1 : x;
}

// Independent transcription of the two assignment levels in actlize's converter:
//   int8x4: h[0] <- bytes 0,2; h[1] <- bytes 1,3
//   int8x16: result groups [0,2,1,3] <- source groups [0,1,2,3].
constexpr int converter_emit(int source_position) {
  int const source_group = source_position / 4;
  int const source_in_group = source_position % 4;
  int result = 4 * permute_0213(source_group) + permute_0213(source_in_group);
#if defined(L208_PLANT_WRONG_PERM) && L208_PLANT_WRONG_PERM
  if (source_position == 1) result ^= 1;  // one wrong permutation bit: 2 -> 3
#endif
  return result;
}

bool check_emission() {
  int mismatch = 0;
  int holes = 0;
  int duplicates = 0;
  int visits[16] = {};
  for (int source = 0; source < 16; ++source) {
    int const anchored = converter_emit(source);
    int const modeled = cutlass::MixGemmEmit<8>::index(source % 4, source / 4);
    if (anchored != modeled) {
      ++mismatch;
      std::printf("[l208 emit mismatch] source=%d converter=%d MixGemmEmit=%d\n",
                  source, anchored, modeled);
    }
    if (anchored >= 0 && anchored < 16) ++visits[anchored];
  }
  for (int output = 0; output < 16; ++output) {
    holes += visits[output] == 0;
    duplicates += visits[output] > 1;
  }
  std::printf("[l208 emit] inputs=16 mismatch=%d holes=%d duplicates=%d map=", mismatch, holes, duplicates);
  for (int source = 0; source < 16; ++source)
    std::printf("%s%d", source ? "," : "", converter_emit(source));
  std::printf("\n");
  return mismatch == 0 && holes == 0 && duplicates == 0;
}

struct PlacementTotals {
  int candidates = 0;
  int map_unset = 0;
  int map_out_of_range = 0;
  int holes = 0;
  int duplicates = 0;
  int roundtrip_bad = 0;
  long long within_class_byte_diff = 0;
  int won1 = 0, won2 = 0, won4 = 0;
};

template <int TM, int TN, int TK, int WM, int WN, int Stages>
bool check_candidate(std::vector<uint8_t> const& source,
                     std::vector<int8_t> const& canonical,
                     int N, int K, PlacementTotals& totals) {
  constexpr int FoldN = 1, ArtifactTileK = 32;
  constexpr int WON = TN / (WN < 16 ? 16 : WN);
  static_assert(WON == 1 || WON == 2 || WON == 4,
                "the bounded Q8 envelope must name one of its proved WON layout classes");
  auto const map = xplane::plane_map<8, TM, TN, TK, WM, WN, FoldN, ArtifactTileK>();
  std::vector<int> owners(size_t(TN) * TK, 0);
  int map_out_of_range = 0;
  int map_unset = 0;
  for (int logical : map) {
    if (logical < 0) {
      ++map_unset;
    } else if (logical >= TN * TK) {
      ++map_out_of_range;
    } else {
      ++owners[size_t(logical)];
    }
  }
  int holes = 0, duplicates = 0;
  for (int count : owners) {
    holes += count == 0;
    duplicates += count > 1;
  }

  std::vector<int8_t> resident(size_t(N) * K, int8_t(0));
  std::vector<uint8_t> recovered;
  xplane::place_derived<8, TM, TN, TK, WM, WN, FoldN, ArtifactTileK>(
      resident.data(), source, N, K);
  xplane::recover_derived<8, TM, TN, TK, WM, WN, FoldN, ArtifactTileK>(
      resident.data(), recovered, N, K);
  int roundtrip_bad = 0;
  for (size_t i = 0; i < source.size(); ++i) roundtrip_bad += source[i] != recovered[i];
  long long within_class_byte_diff = 0;
  for (size_t i = 0; i < resident.size(); ++i)
    within_class_byte_diff += resident[i] != canonical[i];

  std::printf(
      "[l208 placement-row] geometry=%dx%dx%d_w%dx%d_s%d artifact_tk=%d fold_n=%d "
      "map_entries=%zu logical=%d unset=%d out_of_range=%d holes=%d duplicates=%d "
      "roundtrip_bad=%d/%zu won=%d byte_diff_within_class=%lld/%zu\n",
      TM, TN, TK, WM, WN, Stages, ArtifactTileK, FoldN, map.size(), TN * TK,
      map_unset, map_out_of_range, holes, duplicates, roundtrip_bad, source.size(),
      WON, within_class_byte_diff, resident.size());
  ++totals.candidates;
  totals.map_unset += map_unset;
  totals.map_out_of_range += map_out_of_range;
  totals.holes += holes;
  totals.duplicates += duplicates;
  totals.roundtrip_bad += roundtrip_bad;
  totals.within_class_byte_diff += within_class_byte_diff;
  if constexpr (WON == 1) ++totals.won1;
  if constexpr (WON == 2) ++totals.won2;
  if constexpr (WON == 4) ++totals.won4;
  return map_unset == 0 && map_out_of_range == 0 && holes == 0 && duplicates == 0 &&
         roundtrip_bad == 0 && within_class_byte_diff == 0;
}

long long byte_diff(std::vector<int8_t> const& a, std::vector<int8_t> const& b) {
  long long result = 0;
  for (size_t i = 0; i < a.size(); ++i) result += a[i] != b[i];
  return result;
}

bool check_placement() {
  // This fixture must cover the largest emitted TN/TK. The former N=128
  // made every TN256 row vacuous (zero N tiles), which is exactly the kind of
  // internally green but unexercised proof this oracle exists to prevent.
  constexpr int N = 512, K = 512;
  // Avoid zero: place_from_map deliberately skips zero bits, so a zero-valued hole is not a useful witness.
  std::vector<uint8_t> source(size_t(N) * K);
  uint32_t state = 0x243f6a88u;
  for (size_t i = 0; i < source.size(); ++i) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    source[i] = uint8_t(1 + state % 255u);
  }

  // Build one independent A32/F1 anchor per emitted WON value. The candidate
  // rows are checked against won2 (the original shipping witness), while the
  // pairwise comparison below independently decides whether WON changes bytes.
  std::vector<int8_t> canonical_won1(size_t(N) * K, int8_t(0));
  std::vector<int8_t> canonical_won2(size_t(N) * K, int8_t(0));
  std::vector<int8_t> canonical_won4(size_t(N) * K, int8_t(0));
  xplane::place_derived<8, 8, 16, 32, 8, 16, 1, 32>(canonical_won1.data(), source, N, K);
  xplane::place_derived<8, 16, 64, 32, 16, 32, 1, 32>(canonical_won2.data(), source, N, K);
  xplane::place_derived<8, 16, 64, 32, 16, 16, 1, 32>(canonical_won4.data(), source, N, K);
  long long const diff12 = byte_diff(canonical_won1, canonical_won2);
  long long const diff14 = byte_diff(canonical_won1, canonical_won4);
  long long const diff24 = byte_diff(canonical_won2, canonical_won4);

  PlacementTotals totals;
  bool ok = true;
#define PREFILL_Q8_CANDIDATE(TM,TN,TK,WM,WN,S) \
  ok &= check_candidate<TM,TN,TK,WM,WN,S>(source, canonical_won2, N, K, totals);
#include "prefill_q8_candidates.inc"

  std::printf(
      "[l208 placement] candidates=%d canonical=A32/F1 fixture=%dx%d won_classes=3 "
      "resident_layout_classes=1 won1=%d won2=%d won4=%d class_pair_byte_diff=%lld/%lld/%lld "
      "unset=%d out_of_range=%d holes=%d duplicates=%d roundtrip_bad=%d/%zu "
      "within_class_byte_diff=%lld/%zu\n",
      totals.candidates, N, K, totals.won1, totals.won2, totals.won4,
      diff12, diff14, diff24, totals.map_unset, totals.map_out_of_range,
      totals.holes, totals.duplicates, totals.roundtrip_bad,
      source.size() * size_t(totals.candidates), totals.within_class_byte_diff,
      source.size() * size_t(totals.candidates));
  return ok && totals.candidates == PREFILL_Q8_EXPECTED_ROWS &&
         totals.won1 > 0 && totals.won2 > 0 && totals.won4 > 0 &&
         diff12 == 0 && diff14 == 0 && diff24 == 0;
}

bool check_q8_value_semantics() {
  // Powers of two keep d*q exactly representable over the full int8 domain, so raw equality needs no tolerance.
  cutlass::half_t const scales[] = {
      cutlass::half_t(0.5f), cutlass::half_t(-0.25f),
      cutlass::half_t(0.03125f), cutlass::half_t(2.0f)};
  int bias_bad = 0;
  int value_bad = 0;
  int checks = 0;
  for (int q = -128; q <= 127; ++q) {
    uint8_t const biased = uint8_t(q + 128);
    // The converter's source type is int8_t, but its prmt reads the four raw bytes.  Round-trip through int8_t here
    // so the test covers the q>=0 half whose biased representation has the sign bit set.
    int8_t const resident = static_cast<int8_t>(biased);
    int const decoded = int(static_cast<uint8_t>(resident)) - 128;
    bias_bad += decoded != q;
    for (cutlass::half_t d : scales) {
      float const got = float(d) * float(decoded);
      float const want = float(d) * float(q);  // GGUF Q8_0 definition: W = d*q
      value_bad += got != want;
      ++checks;
    }
  }
  std::printf("[l208 q8-value] signed_codes=256 scales=4 checks=%d bias_bad=%d value_bad=%d semantics=W=d*q\n",
              checks, bias_bad, value_bad);
  return bias_bad == 0 && value_bad == 0;
}

}  // namespace

int main() {
  bool const emit_ok = check_emission();
  bool const placement_ok = check_placement();
  bool const value_ok = check_q8_value_semantics();
  bool const ok = emit_ok && placement_ok && value_ok;
  std::printf("[l208] %s emit=%s placement=%s q8_value=%s\n",
              ok ? "PASS" : "FAIL", emit_ok ? "PASS" : "FAIL",
              placement_ok ? "PASS" : "FAIL", value_ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
