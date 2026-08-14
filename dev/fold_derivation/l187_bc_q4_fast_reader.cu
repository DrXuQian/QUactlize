// L187 -- bind the shipping Q4 BC whole-word reader to the production xplane bytes.
//
// This is deliberately a host-exhaustive oracle plus a device compile probe.  The host half covers every
// (n,k) in a 512x512 plane for every Q4 arrangement admitted by the production reader.  It compares:
//
//   xplane::place_derived bytes -> scalar xplane_physical_code (the code_at address oracle)
//                                -> Q4WordPlan + q4_group_byte_offset (the shipping word reader)
//
// recover_derived is the independent inverse anchor: agreement between two readers is not accepted unless the
// bytes also recover to the original logical codes.  The device kernel below instantiates the same shipping plan,
// actual code_at, and dequantize_word on sm_120; it is compiled and inspected but never executed locally.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "fold_traits.hpp"
#include "gguf_bc_vecdot.hpp"
#include "gguf_bc_q4_reader.hpp"
#include "xplane_offline.hpp"

namespace {

using gguf_scale::KType;
namespace bc = gguf_scale::bc_vecdot;
namespace q4 = gguf_scale::bc_vecdot::q4_reader;

constexpr int kN = 512;
constexpr int kK = 512;
constexpr int kCodes = kN * kK;

uint8_t logical_code(int n, int k) {
  // Deliberately asymmetric in both coordinates; adjacent logical codes are not aliases.
  return uint8_t((13 * n + 7 * k + (n >> 1) + 3 * (k >> 2)) & 15);
}

uint8_t nibble_at(std::vector<int8_t> const& bytes, int64_t physical_code) {
  uint8_t const packed = uint8_t(bytes.at(size_t(physical_code >> 1)));
  return uint8_t((packed >> (4 * (physical_code & 1))) & 15);
}

uint64_t fnv1a(std::vector<int8_t> const& bytes) {
  uint64_t hash = UINT64_C(1469598103934665603);
  for (int8_t value : bytes) {
    hash ^= uint8_t(value);
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

uint64_t check_native_metadata() {
  using U = gguf_scale::packed_unit::Unit<KType::Q4_K>;
  static_assert(U::kUnitTotal == 16 && U::kSbPerUnit == 1 && U::kGroups == 8 &&
                U::kScaleBits == 6 && U::kMinBits == 6,
                "L187 native metadata domain drifted from the Q4 packed-unit ABI");
  uint64_t bad = 0;
  for (unsigned group = 0; group < 8; ++group) {
    for (unsigned sc = 0; sc < 64; ++sc) {
      for (unsigned mn = 0; mn < 64; ++mn) {
        alignas(16) uint8_t unit[16] = {};
        unit[0] = 0x00; unit[1] = 0x3c;  // d = 1
        unit[2] = 0x00; unit[3] = 0x3c;  // dmin = 1
        gguf_scale::packed_unit::put_code<KType::Q4_K>(unit, int(group), 0, int(sc));
        gguf_scale::packed_unit::put_code<KType::Q4_K>(unit, int(group), 1, int(mn));
        uint32_t words[4];
        std::memcpy(words, unit, sizeof(words));
        q4::Q4PackedMetadata const metadata{words[0], words[1], words[2], words[3]};
        auto const native = q4::decode_scale_zero(metadata, group);
        uint16_t const scale_bits = reinterpret_cast<uint16_t const&>(native.scale);
        uint16_t const zero_bits = reinterpret_cast<uint16_t const&>(native.zero);
        uint16_t const expect_scale = cutlass::half_t(float(sc)).raw();
        uint16_t const expect_zero = cutlass::half_t(-float(mn)).raw();
        bad += metadata.scale_code(group) != sc || metadata.min_code(group) != mn ||
               metadata.d_bits() != 0x3c00u || metadata.dmin_bits() != 0x3c00u ||
               scale_bits != expect_scale || zero_bits != expect_zero;
      }
    }
  }
  return bad;
}

template <int ArtifactTileK>
int check_arrangement(bool plant_wrong_permutation, uint64_t& checked, uint64_t& address_bad,
                      uint64_t& value_bad, uint64_t& alignment_bad,
                      uint64_t& roundtrip_bad, uint64_t& byte_hash) {
  static_assert(bc::arrangement_supported_v<KType::Q4_K, ArtifactTileK>,
                "L187 must instantiate every production-supported Q4 arrangement");
  static_assert(ArtifactTileK == 32 || ArtifactTileK == 64 ||
                ArtifactTileK == 128 || ArtifactTileK == 256);
  constexpr int Fold = fold::delivery_fold_v<4, ArtifactTileK>;
  using Plan = q4::Q4WordPlan<ArtifactTileK>;

  std::vector<uint8_t> logical_codes(static_cast<size_t>(kCodes), uint8_t(0));
  for (int k = 0; k < kK; ++k)
    for (int n = 0; n < kN; ++n)
      logical_codes[size_t(k) * kN + n] = logical_code(n, k);

  std::vector<int8_t> resident(size_t(kCodes) / 2, int8_t(0x5a));
  xplane::place_derived<4, 64, 64, ArtifactTileK, 32, 32, Fold, ArtifactTileK>(
      resident.data(), logical_codes, kN, kK);
  byte_hash = fnv1a(resident);

  std::vector<uint8_t> recovered(size_t(kCodes), uint8_t(0xff));
  xplane::recover_derived<4, 64, 64, ArtifactTileK, 32, 32, Fold, ArtifactTileK>(
      resident.data(), recovered, kN, kK);
  for (int k = 0; k < kK; ++k)
    for (int n = 0; n < kN; ++n)
      roundtrip_bad += recovered[size_t(k) * kN + n] != logical_codes[size_t(k) * kN + n];

  for (int n = 0; n < kN; ++n) {
    for (int k = 0; k < kK; ++k) {
      int const k0 = k & ~31;
      int const logical_k = k - k0;
      int physical_nibble = Plan::physical_nibble_from_logical_k(logical_k);
      if (plant_wrong_permutation) physical_nibble ^= 1;  // bijective, in-range, and wrong at every coordinate
      int64_t const group_byte = bc::q4_group_byte_offset<ArtifactTileK>(n, k0, kN);
      alignment_bad += (group_byte & (alignof(uint32_t) - 1)) != 0;
      int64_t const fast_physical = 2 * group_byte + physical_nibble;
      int64_t const scalar_physical =
          bc::xplane_physical_code<KType::Q4_K, false, ArtifactTileK>(n, k, kN);
      int const within_word = Plan::nibble_in_word(physical_nibble);
      uint32_t packed_word = 0;
      std::memcpy(&packed_word,
                  resident.data() + group_byte + 4 * Plan::word_index(physical_nibble),
                  sizeof(packed_word));
      unsigned const fast_code = Plan::code_from_pair_lane(
          packed_word, within_word & 3, within_word >> 2);
      unsigned const scalar_code = nibble_at(resident, scalar_physical);
      address_bad += fast_physical != scalar_physical;
      value_bad += fast_code != scalar_code;
      ++checked;
    }
  }
  return 0;
}

template <int ArtifactTileK>
__global__ void device_binding_probe(uint8_t const* low, int* out) {
  int const n = int(blockIdx.x), k = int(threadIdx.x);
  if (k >= 32) return;
  int const scalar = bc::code_at<KType::Q4_K, ArtifactTileK>(low, nullptr, n, k, kN);
  int64_t const byte = bc::q4_group_byte_offset<ArtifactTileK>(n, 0, kN);
  uint32_t const word = *reinterpret_cast<uint32_t const*>(low + byte +
      4 * q4::Q4WordPlan<ArtifactTileK>::word_index(
          q4::Q4WordPlan<ArtifactTileK>::physical_nibble_from_logical_k(k)));
  auto const decoded = q4::dequantize_word(word);
  int const p = q4::Q4WordPlan<ArtifactTileK>::physical_nibble_from_logical_k(k);
  half2 const pair = decoded.pair[q4::Q4WordPlan<ArtifactTileK>::nibble_in_word(p) & 3];
  half const selected = (q4::Q4WordPlan<ArtifactTileK>::nibble_in_word(p) >> 2) ? pair.y : pair.x;
  // If this compile-only probe is promoted to a device runtime, zero is already the exact semantic verdict.
  out[n * 32 + k] = scalar - int(__half2float(selected));
}

template __global__ void device_binding_probe<32>(uint8_t const*, int*);
template __global__ void device_binding_probe<64>(uint8_t const*, int*);
template __global__ void device_binding_probe<128>(uint8_t const*, int*);
template __global__ void device_binding_probe<256>(uint8_t const*, int*);

constexpr int production_q4_denominator() {
  return int(bc::arrangement_supported_v<KType::Q4_K, 32>) +
         int(bc::arrangement_supported_v<KType::Q4_K, 64>) +
         int(bc::arrangement_supported_v<KType::Q4_K, 128>) +
         int(bc::arrangement_supported_v<KType::Q4_K, 256>);
}

}  // namespace

int main(int argc, char** argv) {
  char const* mode = argc == 2 ? argv[1] : "live";
  bool const wrong = !std::strcmp(mode, "wrong-permutation");
  bool const missing = !std::strcmp(mode, "missing-denominator");
  if (std::strcmp(mode, "live") && !wrong && !missing) {
    std::fprintf(stderr, "usage: %s [live|wrong-permutation|missing-denominator]\n", argv[0]);
    return 2;
  }

  static_assert(bc::Traits<KType::Q4_K>::DefaultArtifactTileK == 256,
                "the existing unversioned Q4 reader must retain its A256 byte interpretation");
  static_assert(production_q4_denominator() == 4,
                "L187's finite Q4 domain must change together with arrangement_supported_v");

  uint64_t checked = 0, address_bad = 0, value_bad = 0, alignment_bad = 0, roundtrip_bad = 0;
  uint64_t const metadata_bad = check_native_metadata();
  std::array<uint64_t, 4> hashes{};
  int observed = 0;
  check_arrangement<32>(wrong, checked, address_bad, value_bad, alignment_bad, roundtrip_bad, hashes[0]); ++observed;
  check_arrangement<64>(wrong, checked, address_bad, value_bad, alignment_bad, roundtrip_bad, hashes[1]); ++observed;
  check_arrangement<128>(wrong, checked, address_bad, value_bad, alignment_bad, roundtrip_bad, hashes[2]); ++observed;
  if (!missing) {
    check_arrangement<256>(wrong, checked, address_bad, value_bad, alignment_bad, roundtrip_bad, hashes[3]); ++observed;
  }

  int const denominator = production_q4_denominator();
  bool const coverage_ok = observed == denominator;
  std::printf("L187 mode=%s arrangements=%d/%d coordinates=%llu address_bad=%llu value_bad=%llu "
              "alignment_bad=%llu metadata_bad=%llu/32768 roundtrip_bad=%llu "
              "hashes=%016llx,%016llx,%016llx,%016llx\n",
              mode, observed, denominator, static_cast<unsigned long long>(checked),
              static_cast<unsigned long long>(address_bad), static_cast<unsigned long long>(value_bad),
              static_cast<unsigned long long>(alignment_bad),
              static_cast<unsigned long long>(metadata_bad),
              static_cast<unsigned long long>(roundtrip_bad),
              static_cast<unsigned long long>(hashes[0]), static_cast<unsigned long long>(hashes[1]),
              static_cast<unsigned long long>(hashes[2]), static_cast<unsigned long long>(hashes[3]));

  if (wrong) {
    bool const red = coverage_ok && alignment_bad == 0 && metadata_bad == 0 && roundtrip_bad == 0 &&
                     address_bad == checked && address_bad != 0;
    std::printf("PLANTED_RED wrong-permutation %s\n", red ? "DETECTED" : "ESCAPED");
    return red ? 1 : 0;
  }
  if (missing) {
    bool const red = !coverage_ok;
    std::printf("PLANTED_RED missing-denominator %s\n", red ? "DETECTED" : "ESCAPED");
    return red ? 1 : 0;
  }
  bool const pass = coverage_ok && address_bad == 0 && value_bad == 0 && alignment_bad == 0 && metadata_bad == 0 &&
                    roundtrip_bad == 0 &&
                    checked == uint64_t(denominator) * kCodes;
  std::printf("L187 %s: shipping Q4 whole-word reader equals code_at/production writer over the full domain\n",
              pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
