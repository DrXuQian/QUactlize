#pragma once

#include "ppu_format_config.hpp"
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

// The single-plane N-fold collective predates native packed metadata and cannot yet stage/decode a Q2/Q4 unit.
// Two-plane packed collectives already own that channel.  Keep the missing seam explicit and fail closed: merely
// instantiating F>1 under PPU_PACKED_SCALE otherwise dies at CollectiveMainloop::is_packed_scale, while pretending it
// is supported would reinterpret raw unit bytes as fp16 metadata.  This predicate can be relaxed only with a real
// flag-on compile control for the folded single-plane collective.
constexpr bool packed_tensor_reader_supported(
    quactlize_ppu_placed_arrangement_v1 const* arrangement, int qtype, int k, int tactic_tile_k) {
  if (!matches_compiled_tactic(arrangement, qtype, k, tactic_tile_k)) return false;
  auto const& format = ppu_formats::for_qtype(qtype);
  int const low_bytes = format.low_bits * arrangement->artifact_tile_k / 8;
  return format.high_bits != 0 || low_bytes >= 32;
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
inline constexpr quactlize_ppu_placed_arrangement_v1 kUnsupportedFold2Q4Control{
    QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1, 4, 32, 0};
static_assert(!packed_tensor_reader_supported(&kUnsupportedFold2Q4Control, 12, 4096, 256),
              "single-plane F>1 stays fail-closed until its packed metadata staging exists");

}  // namespace ppu_arrangements
