#pragma once
#include <stdint.h>

#define QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V2 2
#define QUACTLIZE_PPU_LAYOUT_Q4_KPACK4_TRANSPOSE_V1 1
#define QUACTLIZE_PPU_LAYOUT_KQUANT_KPACK_TRANSPOSE_V1 2
#define QUACTLIZE_PPU_LAYOUT_Q8_KPACK2_TRANSPOSE_V1 4
#define QUACTLIZE_PPU_Q4_KPACK4_MAPPING_ID UINT64_C(0x51344b5034540001)
#define QUACTLIZE_PPU_KQUANT_KPACK_MAPPING_ID UINT64_C(0x514b504b54000001)
#define QUACTLIZE_PPU_Q8_KPACK2_MAPPING_ID UINT64_C(0x51384b5032540001)

// Identity of resident bytes, independent of the chosen compute tactic.
// All fields are mandatory. Reserved and artifact_tile_k must be zero.
// Producers and consumers reject unknown layouts rather than guessing a map.
typedef struct quactlize_ppu_placed_arrangement_v2 {
  int32_t version;
  int32_t layout;
  int32_t bits;
  int32_t high_bits;
  int32_t artifact_tile_k;
  int32_t transport_tile_k;
  int32_t group_size;
  int32_t reserved;
  uint64_t mapping_id;
} quactlize_ppu_placed_arrangement_v2;
