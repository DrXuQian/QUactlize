// L127 -- A CALLER-PROVIDED METADATA STRIDE MUST BE SEMANTIC.
//
// The test holds pointer, shape, coordinate and payload fixed and changes only
// dS.  Both CuTe maps are anchored to independent int64 address formulae and
// a unique value at every physical offset.  Agreement between two CuTe
// objects is therefore insufficient: an implementation that accepts dS but
// silently rebuilds the compact layout fails the planted ignored-dS arm.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "cute/tensor.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_metadata_policy.hpp"

namespace md = cutlass::gemm::collective::detail;
using namespace cute;

constexpr int kN = 32;
constexpr int kScaleK = 8;
constexpr int kExperts = 256;
constexpr int kTightGroupStride = kN;
constexpr int kTightExpertStride = kN * kScaleK;
// Metadata is copied with 16-byte cp.async atoms.  Exercise a genuinely
// non-compact layout without turning this stride test into an invalid-alignment
// test: eight fp16 elements are one legal source-address quantum.
constexpr int kPaddedGroupStride = kN + 8;
constexpr int kPaddedExpertStride = kPaddedGroupStride * kScaleK + 8;
constexpr int kGuard = 128;

using ScaleTile = Shape<Int<kN>, _2>;
using MetadataStride = Stride<_1, int64_t, int64_t>;

static_assert(md::kStridedMetadataTileApi == 2);

uint32_t payload(int64_t offset) {
  // Odd multiplication is bijective modulo 2^32 over this address range.
  return uint32_t(offset) * 2654435761u ^ 0xa5c31f27u;
}

int main() {
  MetadataStride const tight = make_stride(
      _1{}, int64_t(kTightGroupStride), int64_t(kTightExpertStride));
  MetadataStride const padded = make_stride(
      _1{}, int64_t(kPaddedGroupStride), int64_t(kPaddedExpertStride));
  MetadataStride const lowered = md::lower_metadata_stride(padded);
  bool const lowering_ok = int64_t(get<0>(lowered)) == 1 &&
                           int64_t(get<1>(lowered)) == kPaddedGroupStride &&
                           int64_t(get<2>(lowered)) == kPaddedExpertStride;
  int64_t const storage = int64_t(kExperts) * kPaddedExpertStride + kGuard;
  std::vector<uint32_t> source(static_cast<std::size_t>(storage), uint32_t{0});
  for (int64_t i = 0; i < storage; ++i) source[std::size_t(i)] = payload(i);

  int tight_address_bad = 0;
  int padded_address_bad = 0;
  int tight_tag_bad = 0;
  int padded_tag_bad = 0;
  int changed_addresses = 0;
  int changed_values = 0;
  int ignored_stride_mismatches = 0;

  for (int expert = 0; expert < kExperts; ++expert) {
    auto g_tight = md::make_metadata_tile<ScaleTile>(
        source.data(), tight, kN, int64_t(kScaleK), kExperts, expert, 0);
    auto g_padded = md::make_metadata_tile<ScaleTile>(
        source.data(), padded, kN, int64_t(kScaleK), kExperts, expert, 0);
    // Negative control: this is the historical defect -- caller asks for the
    // padded layout, implementation silently substitutes the compact one.
    auto g_ignored = md::make_metadata_tile<ScaleTile>(
        source.data(), tight, kN, int64_t(kScaleK), kExperts, expert, 0);

    for (int group = 0; group < kScaleK; ++group) {
      int const group_in_tile = group % size<1>(ScaleTile{});
      int const metadata_tile = group / size<1>(ScaleTile{});
      for (int n = 0; n < kN; ++n) {
        int64_t const tight_expected = int64_t(n) +
            int64_t(group) * kTightGroupStride +
            int64_t(expert) * kTightExpertStride;
        int64_t const padded_expected = int64_t(n) +
            int64_t(group) * kPaddedGroupStride +
            int64_t(expert) * kPaddedExpertStride;
        auto const* tight_ptr = raw_pointer_cast(
            &g_tight(n, group_in_tile, metadata_tile));
        auto const* padded_ptr = raw_pointer_cast(
            &g_padded(n, group_in_tile, metadata_tile));
        auto const* ignored_ptr = raw_pointer_cast(
            &g_ignored(n, group_in_tile, metadata_tile));
        int64_t const tight_actual = tight_ptr - source.data();
        int64_t const padded_actual = padded_ptr - source.data();
        int64_t const ignored_actual = ignored_ptr - source.data();

        tight_address_bad += tight_actual != tight_expected;
        padded_address_bad += padded_actual != padded_expected;
        tight_tag_bad += *tight_ptr != payload(tight_expected);
        padded_tag_bad += *padded_ptr != payload(padded_expected);
        changed_addresses += tight_actual != padded_actual;
        changed_values += *tight_ptr != *padded_ptr;
        ignored_stride_mismatches += ignored_actual != padded_expected;
      }
    }
  }

  constexpr int kCoordinates = kExperts * kScaleK * kN;
  // Only expert=0, group=0 has identical compact and padded origins; all 32
  // columns in that group are equal and every other coordinate must move.
  constexpr int kExpectedChanged = kCoordinates - kN;
  bool const positive = tight_address_bad == 0 && padded_address_bad == 0 &&
      tight_tag_bad == 0 && padded_tag_bad == 0 &&
      changed_addresses == kExpectedChanged && changed_values == kExpectedChanged;
  bool const negative = ignored_stride_mismatches == kExpectedChanged;

  std::printf(
      "L127 lowering=shared-production-seam Arguments.dS=(1,40,328) "
      "Params.dS=(%lld,%lld,%lld) -> %s\n",
      static_cast<long long>(get<0>(lowered)),
      static_cast<long long>(get<1>(lowered)),
      static_cast<long long>(get<2>(lowered)), lowering_ok ? "PASS" : "FAIL");
  std::printf(
      "L127 addresses=%d tight_bad=%d padded_bad=%d tight_tag_bad=%d "
      "padded_tag_bad=%d changed_addr=%d changed_value=%d expected_changed=%d -> %s\n",
      kCoordinates, tight_address_bad, padded_address_bad, tight_tag_bad,
      padded_tag_bad, changed_addresses, changed_values, kExpectedChanged,
      positive ? "PASS" : "FAIL");
  std::printf(
      "L127 anchor=explicit-int64-stride-formula+unique-physical-offset-tags "
      "ignored-dS-mismatches=%d expected=%d -> %s\n",
      ignored_stride_mismatches, kExpectedChanged,
      negative ? "EXPECTED-RED" : "FAIL");
  std::printf(
      "L127 caller-X=dS implementation-Y=dS result=%s scope=S-and-Z-shared-stride\n",
      positive && negative ? "PASS" : "FAIL");
  return lowering_ok && positive && negative ? 0 : 1;
}
