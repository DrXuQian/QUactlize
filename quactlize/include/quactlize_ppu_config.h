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

// Stores a static config-array address in *configs when configs is non-null and returns its element count.
// No CUDA/PPU context is required. Dense and grouped are separate operators and therefore separate inventories;
// the grouped array also contains its CUDA-core GEMV tactic, discriminated before its meaningless tile fields.
int32_t quactlize_ppu_list_configs(quactlize_ppu_config_v1 const** configs);
int32_t quactlize_ppu_list_grouped_configs(quactlize_ppu_config_v1 const** configs);

// Host-only per-problem validity predicates for the inventories above. They return 1 only when config_name names a
// compiled tactic that the corresponding shipping entry can run for this problem, and 0 otherwise. A null/empty name
// asks about that entry's compiled default. No CUDA/PPU context is required, and the launch entries enforce the same
// exact-type shared-memory/compact-A checks even when a caller neglects to query first.
//
// The grouped inventory contains tensor-core and CUDA-core families. Call the predicate matching
// enable_cuda_kernel; using a family with the other predicate returns 0. max_rows is the largest expert extent (the
// only distribution property that can affect a compiled kernel); total_rows and experts validate the grouped domain.
int32_t quactlize_ppu_dense_lowbit_config_valid_v1(
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
// the decline and also runs that default.
int quactlize_ppu_dense_lowbit_config_v1(
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
