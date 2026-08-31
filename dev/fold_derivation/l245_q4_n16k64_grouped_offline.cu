// L245 -- grouped Q4 N16xK64 offline producer/recover closure.
//
// This links the production ppu_dense_layout.cu and ppu_unit_pack.cpp rather
// than reimplementing either producer.  The legacy fully-quantized producer is
// used only to decode official GGUF blocks into the canonical Native Q4 code
// plane; each expert is then placed/recovered through the layout-3 C ABI with
// the same byte strides owned by gguf_prepass_ops.cpp.

#include "ppu_placed_arrangement.hpp"
#include "q4_n16k64_direct_offline.hpp"
#include "quactlize_ppu_config.h"
#include "quactlize_ppu_packed.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" int quactlize_ppu_recover_dense_for_tile(
    std::uint8_t const* low_layout, std::uint8_t const* high_layout,
    std::uint8_t* low_native, std::uint8_t* high_native,
    int n, int k, int qtype, int artifact_tile_k);

namespace {

constexpr int kExperts = 2;
constexpr int kN = 256;
constexpr int kK = 256;
constexpr int kQtype = 12;
constexpr int kRawBytes = 144;
constexpr int kSuperblocks = kK / 256;
constexpr std::size_t kCodeBytesPerExpert =
    std::size_t(kN) * std::size_t(kK) / 2;
constexpr std::size_t kRawBytesPerExpert =
    std::size_t(kN) * kSuperblocks * kRawBytes;
constexpr std::size_t kGuardBytes = 64;

std::uint64_t fingerprint(std::uint8_t const* data, std::size_t bytes) {
  std::uint64_t h = UINT64_C(1469598103934665603);
  for (std::size_t i = 0; i < bytes; ++i) {
    h ^= data[i];
    h *= UINT64_C(1099511628211);
  }
  return h;
}

void make_distinct_blocks(std::vector<std::uint8_t>& blocks) {
  for (int e = 0; e < kExperts; ++e) {
    for (int n = 0; n < kN; ++n) {
      auto* block = blocks.data() +
          (std::size_t(e) * kN + std::size_t(n)) * kRawBytes;
      for (int b = 0; b < kRawBytes; ++b) {
        block[b] = std::uint8_t(
            1 + ((e + 1) * 67 + n * 29 + b * 43 + (n ^ b) * 7) % 251);
      }
      // Finite, expert-distinct fp16 d/dmin headers.  Their exact numeric
      // values are irrelevant to code placement but they make packed units a
      // nonzero, independently fingerprintable metadata payload.
      std::uint16_t const d = e == 0 ? 0x3c00u : 0x4000u;
      std::uint16_t const dmin = e == 0 ? 0x3800u : 0x3c00u;
      block[0] = std::uint8_t(d & 0xffu);
      block[1] = std::uint8_t(d >> 8);
      block[2] = std::uint8_t(dmin & 0xffu);
      block[3] = std::uint8_t(dmin >> 8);
    }
  }
}

bool all_equal(std::uint8_t const* p, std::size_t bytes,
               std::uint8_t value) {
  return std::all_of(p, p + bytes,
                     [value](std::uint8_t x) { return x == value; });
}

}  // namespace

int main() {
  int bad = 0;
  int code_clean = 0;
  int recover_clean = 0;
  int stride_bad = 0;
  int units_bad = 0;

  auto const arrangement = ppu_arrangements::q4_n16k64_direct_v1();
  if (arrangement.layout != QUACTLIZE_PPU_LAYOUT_Q4_N16K64_DIRECT_V1 ||
      arrangement.mapping_id != QUACTLIZE_PPU_Q4_N16K64_DIRECT_MAPPING_ID)
    ++bad;

  std::vector<std::uint8_t> blocks(
      std::size_t(kExperts) * kRawBytesPerExpert);
  make_distinct_blocks(blocks);
  auto const blocks_before = blocks;

  std::int64_t const units_per_expert_i64 =
      quactlize_ppu_units_bytes(kN, kK, kQtype);
  if (units_per_expert_i64 <= 0) {
    std::printf("L245 Q4_N16K64_GROUPED_OFFLINE FAIL reason=units-size\n");
    return 1;
  }
  std::size_t const units_per_expert =
      std::size_t(units_per_expert_i64);

  // Decode real official Q4_K blocks once through the production producer.
  // Its output is the established Xplane resident code plane plus the same
  // grouped packed-unit payload used by the torch route.
  std::vector<std::uint8_t> xplane(
      std::size_t(kExperts) * kCodeBytesPerExpert, 0x91);
  std::vector<std::uint8_t> units_reference(
      std::size_t(kExperts) * units_per_expert, 0x92);
  bad += quactlize_ppu_prepare_fully_quantized_v1(
      blocks.data(), xplane.data(), nullptr, units_reference.data(),
      kN, kK, kExperts, kQtype) != 0;

  std::vector<std::uint8_t> native(
      std::size_t(kExperts) * kCodeBytesPerExpert, 0x93);
  for (int e = 0; e < kExperts; ++e) {
    bad += quactlize_ppu_recover_dense_for_tile(
        xplane.data() + std::size_t(e) * kCodeBytesPerExpert, nullptr,
        native.data() + std::size_t(e) * kCodeBytesPerExpert, nullptr,
        kN, kK, kQtype, 256) != 0;
  }

  // The grouped metadata producer and two independent dense calls must agree
  // byte-for-byte.  This makes expert metadata base arithmetic observable.
  std::vector<std::uint8_t> units_grouped(
      std::size_t(kExperts) * units_per_expert, 0x94);
  bad += quactlize_ppu_prepare_units_grouped(
      blocks.data(), units_grouped.data(), kN, kK, kExperts, kQtype) != 0;
  units_bad += units_grouped != units_reference;
  for (int e = 0; e < kExperts; ++e) {
    std::vector<std::uint8_t> dense_units(units_per_expert, 0x95);
    bad += quactlize_ppu_prepare_units(
        blocks.data() + std::size_t(e) * kRawBytesPerExpert,
        dense_units.data(), kN, kK, kQtype) != 0;
    units_bad += !std::equal(
        dense_units.begin(), dense_units.end(),
        units_grouped.begin() + std::size_t(e) * units_per_expert);
  }
  auto const units_before = units_reference;

  // Put the packed metadata immediately after the placed plane.  A wrong
  // expert stride or an overrun in the layout transform therefore mutates a
  // live artifact field instead of harmless spare capacity.
  std::size_t const placed_total =
      std::size_t(kExperts) * kCodeBytesPerExpert;
  std::size_t const units_total =
      std::size_t(kExperts) * units_per_expert;
  std::vector<std::uint8_t> artifact(
      kGuardBytes + placed_total + units_total + kGuardBytes, 0xa5);
  auto* placed = artifact.data() + kGuardBytes;
  auto* adjacent_units = placed + placed_total;
  std::copy(units_reference.begin(), units_reference.end(), adjacent_units);

  for (int e = 0; e < kExperts; ++e) {
    int source_expert = e;
#if defined(L245_PLANT_EXPERT_BASE_REUSE)
    if (e == 1) source_expert = 0;
#endif
    bad += quactlize_ppu_prepare_dense_for_arrangement_v2(
        native.data() + std::size_t(source_expert) * kCodeBytesPerExpert,
        nullptr,
        placed + std::size_t(e) * kCodeBytesPerExpert,
        nullptr, kN, kK, kQtype, &arrangement) != 0;
  }

#if defined(L245_PLANT_METADATA_MUTATION)
  adjacent_units[units_per_expert] ^= 0x1u;
#endif

  std::uint64_t placed_fp[kExperts] = {};
  std::uint64_t units_fp[kExperts] = {};
  for (int e = 0; e < kExperts; ++e) {
    auto const* native_e =
        native.data() + std::size_t(e) * kCodeBytesPerExpert;
    auto const* placed_e =
        placed + std::size_t(e) * kCodeBytesPerExpert;
    bool expert_codes_clean = true;
    for (int n = 0; n < kN; ++n) {
      for (int k = 0; k < kK; ++k) {
        if (q4_n16k64_direct::placed_get(placed_e, n, k, kN) !=
            q4_n16k64_direct::native_get(native_e, n, k, kK)) {
          expert_codes_clean = false;
        }
      }
    }
    code_clean += expert_codes_clean;
    placed_fp[e] = fingerprint(placed_e, kCodeBytesPerExpert);
    units_fp[e] = fingerprint(
        adjacent_units + std::size_t(e) * units_per_expert,
        units_per_expert);
  }

  std::vector<std::uint8_t> recovered(
      std::size_t(kExperts) * kCodeBytesPerExpert, 0x96);
  for (int e = 0; e < kExperts; ++e) {
    bad += quactlize_ppu_recover_dense_for_arrangement_v2(
        placed + std::size_t(e) * kCodeBytesPerExpert, nullptr,
        recovered.data() + std::size_t(e) * kCodeBytesPerExpert, nullptr,
        kN, kK, kQtype, &arrangement) != 0;
    bool const clean = std::equal(
        recovered.begin() + std::size_t(e) * kCodeBytesPerExpert,
        recovered.begin() + std::size_t(e + 1) * kCodeBytesPerExpert,
        native.begin() + std::size_t(e) * kCodeBytesPerExpert);
    recover_clean += clean;
  }

  stride_bad += !all_equal(artifact.data(), kGuardBytes, 0xa5);
  stride_bad += !all_equal(
      adjacent_units + units_total, kGuardBytes, 0xa5);
  stride_bad += placed_fp[0] == placed_fp[1];
  stride_bad += units_fp[0] == units_fp[1];
  stride_bad += placed_fp[0] == 0 || placed_fp[1] == 0;
  stride_bad += units_fp[0] == 0 || units_fp[1] == 0;
  units_bad += !std::equal(
      adjacent_units, adjacent_units + units_total, units_before.begin());
  int const raw_unchanged = blocks == blocks_before;
  int const units_unchanged = units_bad == 0;

  bad += code_clean != kExperts;
  bad += recover_clean != kExperts;
  bad += stride_bad != 0;
  bad += units_bad != 0;
  bad += !raw_unchanged;

  std::printf(
      "L245 Q4_N16K64_GROUPED_OFFLINE %s experts=%d N=%d K=%d "
      "placed_bytes_per_expert=%zu units_bytes_per_expert=%zu "
      "code=%d/%d recover=%d/%d expert_stride=%s "
      "raw_unchanged=%d units_unchanged=%d fingerprints=%s reds=2\n",
      bad ? "FAIL" : "PASS", kExperts, kN, kK,
      kCodeBytesPerExpert, units_per_expert,
      code_clean, kExperts, recover_clean, kExperts,
      stride_bad ? "FAIL" : "PASS", raw_unchanged, units_unchanged,
      (placed_fp[0] != placed_fp[1] && units_fp[0] != units_fp[1])
          ? "DISTINCT" : "ALIASED");
  return bad ? 1 : 0;
}
