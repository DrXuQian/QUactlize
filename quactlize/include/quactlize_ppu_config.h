#pragma once
// Runtime inventory of tactics compiled into libquactlize_ppu.so. The array and its strings have static lifetime;
// callers neither allocate nor free them.

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct quactlize_ppu_config_v1 {
  // Compare the family discriminator first. When true, the tile fields are meaningless and must be ignored; this
  // lets a CUDA-core GEMV live in the same candidate array and profiling loop as tensor-core tile configurations.
  bool enable_cuda_kernel;
  char const* name;
  int32_t tile_m;
  int32_t tile_n;
  int32_t warp_m;
  int32_t warp_n;
  int32_t stages;
} quactlize_ppu_config_v1;

// Per-problem tactic description. Unlike the legacy geometry-only v1 inventory, this record carries TileK: TileK
// selects the resident weight arrangement and therefore must agree with the offline artifact. The scheme-specific
// filtered queries below are the authoritative inventory for new tuners. name has static lifetime. As in v1, every
// tile field (including tile_k) is meaningless and zero for a CUDA-core family record.
typedef struct quactlize_ppu_config_v2 {
  bool enable_cuda_kernel;
  char const* name;
  int32_t tile_m;
  int32_t tile_n;
  int32_t tile_k;
  int32_t warp_m;
  int32_t warp_n;
  int32_t stages;
} quactlize_ppu_config_v2;

// Stores a static config-array address in *configs when configs is non-null and returns its element count.
// No CUDA/PPU context is required. Dense and grouped are separate operators and therefore separate inventories;
// each array also contains its CUDA-core GEMV tactic, discriminated before its meaningless tile fields.
int32_t quactlize_ppu_list_configs(quactlize_ppu_config_v1 const** configs);
int32_t quactlize_ppu_list_grouped_configs(quactlize_ppu_config_v1 const** configs);

// Writes up to capacity valid records and returns the full valid-record count, so a caller may first pass
// (NULL, 0), allocate exactly that many records, and query again. A negative capacity writes nothing. These are
// host-only queries and require no CUDA/PPU context. Dense/grouped tensor records report the scheme-specific TileK
// selected by ppu_format_config.inc; CUDA records report their family with zero/meaningless tile fields.
int32_t quactlize_ppu_list_valid_dense_lowbit_configs_v2(
    quactlize_ppu_config_v2* configs, int32_t capacity,
    int m, int n, int k, int group_size, int qtype);
int32_t quactlize_ppu_list_valid_dense_fully_quantized_configs_v2(
    quactlize_ppu_config_v2* configs, int32_t capacity,
    int m, int n, int k, int group_size, int qtype);
int32_t quactlize_ppu_list_valid_grouped_fully_quantized_configs_v2(
    quactlize_ppu_config_v2* configs, int32_t capacity,
    int total_rows, int n, int k, int group_size, int experts, int max_rows, int qtype);
int32_t quactlize_ppu_list_valid_vecdot_moe_configs_v2(
    quactlize_ppu_config_v2* configs, int32_t capacity,
    int total_rows, int n, int k, int group_size, int experts, int max_rows, int qtype);

// Host-only per-problem validity predicates for the inventories above. They return 1 only when config_name names a
// compiled tactic that the corresponding shipping entry can run for this problem, and 0 otherwise. A null/empty name
// asks about that entry's compiled default. No CUDA/PPU context is required, and the launch entries enforce the same
// exact-type shared-memory/compact-A checks even when a caller neglects to query first.
//
// Both inventories contain tensor-core and CUDA-core families. Call the predicate matching enable_cuda_kernel;
// using a family with the other predicate returns 0. max_rows is the largest expert extent (the only distribution
// property that can affect a compiled kernel); total_rows and experts validate the grouped domain.
//
// "Any shape" for the compiled tensor fallback means every positive M (or every positive total_rows/experts/max_rows
// grouped problem) with N and K positive multiples of 256, the qtype's GGUF group size (16 for Q2/Q3/Q6, 32 for
// Q4/Q5), and any extra format-selected constraint reported by this binary (currently K%512 for paired Q3/Q6 packed
// units). Tile extents do not narrow that domain: M/N tails are predicated. In particular N=32 is outside the current
// resident-artifact ABI, not rejected because a default TileN is wider. Fully-quantized entries are outside the
// default build and return invalid unless PPU_PACKED_SCALE and that qtype's PPU_PACKED_FORMAT were compiled.
int32_t quactlize_ppu_dense_lowbit_config_valid_v1(
    int m, int n, int k, int group_size, int qtype, char const* config_name);
int32_t quactlize_ppu_gemv_lowbit_config_valid_v1(
    int m, int n, int k, int group_size, int qtype, char const* config_name);
int32_t quactlize_ppu_dense_fully_quantized_config_valid_v1(
    int m, int n, int k, int group_size, int qtype, char const* config_name);
int32_t quactlize_ppu_grouped_fully_quantized_config_valid_v1(
    int total_rows, int n, int k, int group_size, int experts, int max_rows,
    int qtype, char const* config_name);
int32_t quactlize_ppu_vecdot_moe_config_valid_v1(
    int total_rows, int n, int k, int group_size, int experts, int max_rows,
    int qtype, char const* config_name);

// Config-selecting host-pointer operator entries. config_name comes from the corresponding dense or grouped
// inventory above. A null/empty name requests that entry's compiled default; an unknown non-empty name reports
// the decline and also runs that default. Each tensor default is instantiated with compile-time assertions that its
// exact shared storage fits ppu001 and that it uses the unrestricted ordinary-A path throughout the admitted domain.
int quactlize_ppu_dense_lowbit_config_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint16_t const* scale, uint16_t const* zero, uint16_t* out,
    int m, int n, int k, int group_size, int qtype, char const* config_name);
int quactlize_ppu_gemv_lowbit_config_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint16_t const* scale, uint16_t const* zero, uint16_t* out,
    int m, int n, int k, int group_size, int qtype, char const* config_name);
int quactlize_ppu_dense_fully_quantized_config_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, int qtype, char const* config_name);
int quactlize_ppu_grouped_fully_quantized_config_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units,
    int const* rows_per_expert, uint16_t* out,
    int total_rows, int n, int k, int experts, int qtype, char const* config_name);
int quactlize_ppu_vecdot_moe_config_v1(
    uint8_t const* blocks, int64_t block_bytes, uint16_t const* x,
    int const* offsets, float* out, int n, int blocks_per_row, int experts,
    int total_rows, int max_rows, int qtype, char const* config_name);

#ifdef __cplusplus
}
#endif
