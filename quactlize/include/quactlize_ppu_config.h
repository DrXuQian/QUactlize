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

// Stores the static config-array address in *configs when configs is non-null and returns its element count.
// No CUDA/PPU context is required. The current list contains dense tensor-core configurations; later CUDA-core
// entries use this same record and call without assigning meaning to the tile fields.
int32_t quactlize_ppu_list_configs(quactlize_ppu_config_v1 const** configs);

// Config-selecting host-pointer GEMM entries. config_name is one of the names returned above. A null/empty name
// requests the compiled default; an unknown non-empty name reports the decline and also runs that default.
int quactlize_ppu_dense_lowbit_config_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint16_t const* scale, uint16_t const* zero, uint16_t* out,
    int m, int n, int k, int group_size, int qtype, char const* config_name);
int quactlize_ppu_dense_fully_quantized_config_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, int qtype, char const* config_name);

#ifdef __cplusplus
}
#endif
