#include <cstdint>

#include "quactlize_ppu_config.h"

#ifndef L140_BACKEND_MARKER
#error L140_BACKEND_MARKER must identify this fake binary
#endif

extern "C" {

int quactlize_ppu_vecdot(uint8_t const*, int64_t, uint16_t const*, float*, int, int, int) {
  return L140_BACKEND_MARKER;
}

int quactlize_ppu_vecdot_moe(uint8_t const*, int64_t, uint16_t const*, int const*, float*,
                             int, int, int, int, int, int) {
  return L140_BACKEND_MARKER;
}

int quactlize_ppu_dequantize(uint8_t const*, int64_t, uint16_t*, int, int) {
  return L140_BACKEND_MARKER;
}

int quactlize_ppu_prepass(uint8_t const*, int64_t, uint16_t const*, uint16_t const*,
                          int, uint16_t*, uint16_t*, int, int, int) {
  return L140_BACKEND_MARKER;
}

int quactlize_ppu_gemv_lowbit(uint16_t const*, uint8_t const*, uint8_t const*,
                              uint16_t const*, uint16_t const*, uint16_t*,
                              int, int, int, int, int, int, int const*, int) {
  return L140_BACKEND_MARKER;
}

int32_t quactlize_ppu_dense_fully_quantized_config_valid_for_arrangement_v1(
    int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v1 const* arrangement, char const*) {
  if (!arrangement || arrangement->version != QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1 ||
      m <= 0 || n <= 0 || k <= 0 || group_size <= 0) return 0;
#if defined(L140_ACCEPT_QTYPE) && defined(L140_ACCEPT_BITS) && defined(L140_ACCEPT_TILE_K) && \
    defined(L140_ACCEPT_HIGH_BITS)
  return qtype == L140_ACCEPT_QTYPE && arrangement->bits == L140_ACCEPT_BITS &&
         arrangement->artifact_tile_k == L140_ACCEPT_TILE_K &&
         arrangement->high_bits == L140_ACCEPT_HIGH_BITS;
#else
  (void)qtype;
  return 0;
#endif
}

#if !defined(L140_OMIT_ARRANGEMENT_READER)
int quactlize_ppu_dense_fully_quantized_for_arrangement_v1(
    uint16_t const*, uint8_t const*, uint8_t const*, uint8_t const*, uint16_t*,
    int, int, int, int, quactlize_ppu_placed_arrangement_v1 const*, char const*) {
  // Zero-M must prove this symbol exists but must return before calling it.  A nonempty invocation of the fake
  // library is outside l140's contract and deliberately cannot masquerade as a working device reader.
  return 140;
}
#endif

#if defined(L140_ACCEPT_V2)
int32_t quactlize_ppu_dense_fully_quantized_config_valid_for_arrangement_v2(
    int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement, char const*) {
  if (!arrangement || m <= 0 || n <= 0 || k <= 0 || group_size <= 0) return 0;
  return qtype == 12 && arrangement->version == QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V2 &&
         arrangement->layout == QUACTLIZE_PPU_LAYOUT_Q4_KPACK4_TRANSPOSE_V1 &&
         arrangement->bits == 4 && arrangement->high_bits == 0 &&
         arrangement->artifact_tile_k == 0 && arrangement->transport_tile_k == 64 &&
         arrangement->group_size == 32 && arrangement->reserved == 0 &&
         arrangement->mapping_id == QUACTLIZE_PPU_Q4_KPACK4_MAPPING_ID;
}

#if !defined(L140_OMIT_ARRANGEMENT_READER)
int quactlize_ppu_dense_fully_quantized_for_arrangement_v2(
    uint16_t const*, uint8_t const*, uint8_t const*, uint8_t const*, uint16_t*,
    int, int, int, int, quactlize_ppu_placed_arrangement_v2 const*, char const*) {
  return 140;
}
#endif

int32_t quactlize_ppu_grouped_fully_quantized_config_valid_for_arrangement_v2(
    int total_rows, int n, int k, int group_size, int experts, int max_rows,
    int qtype, quactlize_ppu_placed_arrangement_v2 const* arrangement,
    char const* config_name) {
  return (!config_name || !*config_name) && experts > 0 && max_rows > 0 &&
         max_rows <= total_rows &&
         quactlize_ppu_dense_fully_quantized_config_valid_for_arrangement_v2(
             total_rows, n, k, group_size, qtype, arrangement, nullptr);
}

#if !defined(L140_OMIT_ARRANGEMENT_READER)
int quactlize_ppu_grouped_fully_quantized_for_arrangement_v2(
    uint16_t const*, uint8_t const*, uint8_t const* high, uint8_t const*,
    int const*, uint16_t*, int total_rows, int n, int k, int experts,
    int qtype, quactlize_ppu_placed_arrangement_v2 const* arrangement,
    char const* config_name) {
  return !high &&
                 quactlize_ppu_grouped_fully_quantized_config_valid_for_arrangement_v2(
                     total_rows, n, k, 32, experts, total_rows, qtype,
                     arrangement, config_name)
             ? 0
             : 140;
}
#endif
#endif

}  // extern "C"
