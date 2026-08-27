#pragma once

#include "ppu_format_config.hpp"
#include "q4_kpack4_offline.hpp"
#include "quactlize_ppu_config.h"

namespace ppu_arrangements {

// Legacy fully-quantized C readers predate the descriptor and historically consume the fully-quantized placement.
// Do not reuse the no-tile Python producer's scale-first default here: changing this map would silently reinterpret
// existing A256 Q2/Q4 artifacts. New Python artifacts always carry an explicit descriptor.
constexpr quactlize_ppu_placed_arrangement_v1 legacy_fully_quantized_default(int qtype) {
  auto const& format = ppu_formats::for_qtype(qtype);
  return {QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1, format.low_bits,
          format.fully_quantized_tile_k, format.high_bits};
}

// This is the one runtime artifact-descriptor predicate.  Inventory and launch both call it; neither may rederive
// fold compatibility from qtype or from a tactic row.  K is included because a well-formed descriptor still cannot
// describe a resident tensor whose logical K ends in a partial artifact tile.
constexpr bool matches_compiled_tactic(
    quactlize_ppu_placed_arrangement_v1 const* arrangement,
    int qtype, int k, int tactic_tile_k) {
  if (!arrangement || arrangement->version != QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1) return false;
  auto const& format = ppu_formats::for_qtype(qtype);
  return format.qtype >= 0 && arrangement->bits == format.low_bits &&
         arrangement->high_bits == format.high_bits &&
         ppu_formats::artifact_tile_k_supported(format, arrangement->artifact_tile_k) &&
         arrangement->artifact_tile_k > 0 && k > 0 && k % arrangement->artifact_tile_k == 0 &&
         arrangement->artifact_tile_k <= tactic_tile_k &&
         tactic_tile_k % arrangement->artifact_tile_k == 0;
}

constexpr quactlize_ppu_placed_arrangement_v2 q4_kpack4_transpose_v1() {
  return {QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V2,
          QUACTLIZE_PPU_LAYOUT_Q4_KPACK4_TRANSPOSE_V1,
          4, 0, 0, q4_kpack4::kTransportK, q4_kpack4::kGroupK, 0,
          q4_kpack4::kMappingId};
}

// v2 is intentionally a separate predicate rather than a templated reinterpret
// of v1.  Xplane and K-pack4 have different physical axes even though both
// contain the same number of low-code bytes.
constexpr bool matches_compiled_tactic(
    quactlize_ppu_placed_arrangement_v2 const* arrangement,
    int qtype, int k, int tactic_tile_k) {
  if (!arrangement ||
      arrangement->version != QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V2 ||
      arrangement->reserved != 0)
    return false;
  if (arrangement->layout == QUACTLIZE_PPU_LAYOUT_XPLANE_V1) {
    if (arrangement->transport_tile_k != 0 || arrangement->group_size != 0 ||
        arrangement->mapping_id != 0)
      return false;
    quactlize_ppu_placed_arrangement_v1 const legacy{
        QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1,
        arrangement->bits, arrangement->artifact_tile_k,
        arrangement->high_bits};
    return matches_compiled_tactic(&legacy, qtype, k, tactic_tile_k);
  }
  if (arrangement->layout !=
      QUACTLIZE_PPU_LAYOUT_Q4_KPACK4_TRANSPOSE_V1)
    return false;
  return qtype == 12 && arrangement->bits == 4 &&
         arrangement->high_bits == 0 &&
         arrangement->artifact_tile_k == 0 &&
         arrangement->transport_tile_k == q4_kpack4::kTransportK &&
         arrangement->group_size == q4_kpack4::kGroupK &&
         arrangement->mapping_id == q4_kpack4::kMappingId &&
         k > 0 && k % q4_kpack4::kGroupK == 0 &&
         tactic_tile_k >= q4_kpack4::kTransportK &&
         tactic_tile_k % q4_kpack4::kTransportK == 0;
}

// Packed metadata support is a property of the exact resident reader, not merely of the low-bit width.  Two-plane
// readers own their general packed channel; unfolded single-plane readers retain the 32-byte floor.  Q4/A32 is the
// one proved folded single-plane exception: its F2 collective reads each physical N/2 x 2K weight run directly and
// decodes the unchanged logical-N Q4_K unit when the shipping tactic consumes one 256-code superblock.
constexpr bool packed_tensor_reader_supported(
    quactlize_ppu_placed_arrangement_v1 const* arrangement, int qtype, int k, int tactic_tile_k) {
  if (!matches_compiled_tactic(arrangement, qtype, k, tactic_tile_k)) return false;
  auto const& format = ppu_formats::for_qtype(qtype);
  int const low_bytes = format.low_bits * arrangement->artifact_tile_k / 8;
  bool const q4_a32_fold2 = qtype == 12 &&
                            arrangement->artifact_tile_k == 32 &&
                            tactic_tile_k == 256;
  return format.high_bits != 0 || low_bytes >= 32 || q4_a32_fold2;
}

constexpr bool packed_tensor_reader_supported(
    quactlize_ppu_placed_arrangement_v2 const* arrangement,
    int qtype, int k, int tactic_tile_k) {
  if (!matches_compiled_tactic(arrangement, qtype, k, tactic_tile_k))
    return false;
  if (arrangement->layout ==
      QUACTLIZE_PPU_LAYOUT_Q4_KPACK4_TRANSPOSE_V1)
    return true;
  quactlize_ppu_placed_arrangement_v1 const legacy{
      QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1,
      arrangement->bits, arrangement->artifact_tile_k,
      arrangement->high_bits};
  return packed_tensor_reader_supported(&legacy, qtype, k, tactic_tile_k);
}

template <int QType, int TacticTileK, int ArtifactTileK>
constexpr bool static_matches_compiled_tactic() {
  constexpr auto descriptor = quactlize_ppu_placed_arrangement_v1{
      QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1,
      ppu_formats::for_qtype(QType).low_bits, ArtifactTileK,
      ppu_formats::for_qtype(QType).high_bits};
  return matches_compiled_tactic(&descriptor, QType, TacticTileK, TacticTileK);
}

template <int QType, int TacticTileK, int ReaderArtifactTileK>
constexpr bool matches_exact_reader(
    quactlize_ppu_placed_arrangement_v1 const* arrangement, int k) {
  return matches_compiled_tactic(arrangement, QType, k, TacticTileK) &&
         arrangement->artifact_tile_k == ReaderArtifactTileK &&
         static_matches_compiled_tactic<QType, TacticTileK, ReaderArtifactTileK>();
}

template <int QType, int TacticTileK, int ReaderArtifactTileK>
constexpr bool packed_tensor_matches_exact_reader(
    quactlize_ppu_placed_arrangement_v1 const* arrangement, int k) {
  return packed_tensor_reader_supported(arrangement, QType, k, TacticTileK) &&
         arrangement->artifact_tile_k == ReaderArtifactTileK &&
         static_matches_compiled_tactic<QType, TacticTileK, ReaderArtifactTileK>();
}

template <int QType, int TacticTileK, int ArtifactTileK>
constexpr bool static_packed_tensor_reader_supported() {
  constexpr auto descriptor = quactlize_ppu_placed_arrangement_v1{
      QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1,
      ppu_formats::for_qtype(QType).low_bits, ArtifactTileK,
      ppu_formats::for_qtype(QType).high_bits};
  return packed_tensor_reader_supported(&descriptor, QType, TacticTileK, TacticTileK);
}

static_assert(static_matches_compiled_tactic<12, 256, 64>(),
              "a Q4 TK64 artifact must be consumable by the shipping TK256 tactic");
static_assert(!static_matches_compiled_tactic<11, 256, 32>(),
              "a descriptor cannot name the absent Q3 TK32 producer");
static_assert(!static_matches_compiled_tactic<14, 128, 256>(),
              "a descriptor cannot name Q6's incomplete TK256 producer/map");
inline constexpr quactlize_ppu_placed_arrangement_v1 kFold2Q3Control{
    QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1, 2, 64, 1};
static_assert(packed_tensor_matches_exact_reader<11, 256, 64>(&kFold2Q3Control, 4096));
static_assert(!packed_tensor_matches_exact_reader<11, 256, 256>(&kFold2Q3Control, 4096),
              "an F=2 artifact must be rejected by an F=1 reader, not silently decoded");
inline constexpr quactlize_ppu_placed_arrangement_v1 kNativeFold2Q4Control{
    QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1, 4, 32, 0};
static_assert(packed_tensor_reader_supported(&kNativeFold2Q4Control, 12, 4096, 256),
              "Q4/A32 must retain its native folded packed-metadata reader");
static_assert(!packed_tensor_reader_supported(&kNativeFold2Q4Control, 12, 4096, 128),
              "Q4/A32 packed metadata requires the proved one-superblock tactic");
inline constexpr auto kQ4KPack4Control = q4_kpack4_transpose_v1();
static_assert(matches_compiled_tactic(&kQ4KPack4Control, 12, 5120, 64));
static_assert(matches_compiled_tactic(&kQ4KPack4Control, 12, 5120, 256));
static_assert(!matches_compiled_tactic(&kQ4KPack4Control, 12, 5120, 32),
              "TK32 needs a separately proved K64-stage/two-consume mainloop");
static_assert(!matches_compiled_tactic(&kQ4KPack4Control, 13, 5120, 256),
              "the first K-pack4 ABI is Q4_K-only");
static_assert(packed_tensor_reader_supported(&kQ4KPack4Control, 12, 5120, 256));

}  // namespace ppu_arrangements
