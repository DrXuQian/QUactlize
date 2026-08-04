#pragma once
// THE PPU DEVICE BACKEND, REACHED BY dlopen. This is how a PPU kernel becomes a torch op.
//
// THE CONSTRAINT THAT FORCES THIS SHAPE. quactlize's torch extension is host C++ built by setup.py with gcc, and it
// has to stay that way: it is what runs on an ordinary machine to prepare a checkpoint, and it is what makes the
// official gguf package usable as an oracle. PPU device code is built by build.sh with hgcc, which setup.py cannot
// invoke and should not learn to. So the two halves cannot be one translation unit.
//
// They can be one PROCESS. build.sh emits libquactlize_ppu.so with C entry points taking raw pointers, the extension
// dlopens it on first use, and the op forwards. The extension never needs the SDK; the .so never needs torch. That
// is the same decoupling the FA and DeepGemm work used, for the same reason.
//
// WHAT "auto" MEANS, and why it is not silent. A backend that quietly falls back to the CPU when the .so is missing
// produces correct numbers slowly and reports nothing, which is indistinguishable from the device path working. So
// the resolved backend is queryable, the tests assert on it, and asking for "ppu" explicitly when it is unavailable
// is an error rather than a fallback.
#include <cstdint>
#include <string>

namespace torch_ext {
namespace ppu_backend {

// Every entry point the device .so must export, with C linkage. Raw pointers and shapes only -- no torch types, so
// the .so can be built by a toolchain that has never heard of torch.
struct Api {
  // Independent block rows: blocks [rows*bpr,block_bytes], x fp16 [rows*bpr*256], out fp32 [rows].
  int (*vecdot)(uint8_t const* blocks, int64_t block_bytes, uint16_t const* x, float* out,
                int rows, int blocks_per_row, int qtype);
  // Dense GEMV: blocks [rows*bpr,block_bytes], one shared x fp16 [bpr*256], out fp32 [rows]. Optional so an older
  // library/stub still loads, but gguf_vecdot_dense refuses explicitly if the distinct contract is unavailable.
  int (*vecdot_dense)(uint8_t const* blocks, int64_t block_bytes, uint16_t const* x, float* out,
                      int rows, int blocks_per_row, int qtype);
  // blocks [E,n,bpr,block_bytes], gathered x [total_rows,bpr*256], offsets [E+1], out [total_rows,n]
  int (*vecdot_moe)(uint8_t const* blocks, int64_t block_bytes, uint16_t const* x,
                    int const* row_offsets, float* out, int n, int blocks_per_row, int experts,
                    int total_rows, int max_rows, int qtype);
  // blocks [n_blocks, block_bytes] -> out fp16, dst_span elements per block
  int (*dequantize)(uint8_t const* blocks, int64_t block_bytes, uint16_t* out, int n_blocks, int qtype);
  // scale blocks + fp16 headers -> the two planes
  int (*prepass)(uint8_t const* blocks, int64_t block_bytes, uint16_t const* d, uint16_t const* dmin,
                 int n, uint16_t* scale, uint16_t* zero, int groups, int qtype, int zmul);
  // Merged BC packed units [E,K-unit,N,unit-bytes] -> consumer fp16 planes [E,K-group,N]. Optional while an older
  // device library can still serve the established raw prepass; a BC route must check this pointer and refuse.
  int (*prepass_unit)(uint8_t const* units, uint16_t* scale, uint16_t* zero,
                      int n, int k, int experts, int qtype, int zmul);
  // Merged BC artifact -> CUDA-core fp16-activation/fp32-output GEMV. experts==0 is exactly one dense decode row;
  // otherwise offsets is [E+1] and out is [total_rows,n]. Optional while older device libraries remain loadable.
  int (*bc_gemv)(uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units,
                 int const* row_offsets, float* out,
                 int total_rows, int n, int k, int experts, int max_rows, int qtype);
  // Resident SCALE_FIRST artifact -> fp16 GEMV result. experts==0 is dense; otherwise offsets is [E+1].
  int (*gemv_lowbit)(uint16_t const* act, uint8_t const* low, uint8_t const* high,
                     uint16_t const* scale, uint16_t const* zero, uint16_t* out,
                     int total_rows, int n, int k, int group_size, int qtype,
                     int experts, int const* row_offsets, int max_rows);
  // Optional hgcc-only dense half. prepare_dense is host-only but linked beside it so the artifact and launcher
  // cannot silently disagree about the fixed xplane tactic. A plain-nvcc GEMV library legitimately leaves both null.
  int (*prepare_dense)(uint8_t const* low_native, uint8_t const* high_native,
                       uint8_t* low_layout, uint8_t* high_layout, int n, int k, int qtype);
  int (*recover_dense)(uint8_t const* low_layout, uint8_t const* high_layout,
                       uint8_t* low_native, uint8_t* high_native, int n, int k, int qtype);
  // Optional tile-aware successors. Fold is derived inside the library from each plane's width and TileK, exactly
  // as the consumer derives it; callers cannot supply a contradictory arrangement.
  int (*prepare_dense_for_tile)(uint8_t const* low_native, uint8_t const* high_native,
                                uint8_t* low_layout, uint8_t* high_layout,
                                int n, int k, int qtype, int tile_k);
  int (*recover_dense_for_tile)(uint8_t const* low_layout, uint8_t const* high_layout,
                                uint8_t* low_native, uint8_t* high_native,
                                int n, int k, int qtype, int tile_k);
  int (*dense_lowbit)(uint16_t const* act, uint8_t const* low, uint8_t const* high,
                      uint16_t const* scale, uint16_t const* zero, uint16_t* out,
                      int m, int n, int k, int group_size, int qtype);
  // Flag-selected fully-quantized entries. high is null for Q4/Q2 and the resident plane for two-plane formats;
  // units are the format's byte-neutral reordered metadata. rc=34 means PPU_PACKED_SCALE was not built in.
  int (*dense_fully_quantized)(uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units,
                               uint16_t* out, int m, int n, int k, int qtype);
  int (*grouped_fully_quantized)(uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units,
                                 int const* rows_per_expert, uint16_t* out,
                                 int total_rows, int n, int k, int experts, int qtype);
};

// Loads libquactlize_ppu.so once. QUACTLIZE_PPU_LIB overrides the path, which is what lets a stub stand in for the
// real thing in a test -- the seam is checkable without the SDK, the same way box_build_dryrun checks build.sh.
// Returns nullptr when the library is absent or a symbol is missing, with `why` set.
// PER-FORMAT LOADING. PPU_PACKED_FORMAT is a compile-time macro, so one library serves one packed format --
// and a Q4_K_M checkpoint is MIXED, carrying Q4_K tensors beside Q6_K ones, so one handle cannot serve a real
// model. Each format gets its own dlopen, its own cached failure, and its own message. RTLD_LOCAL (already used)
// is what keeps several loaded at once from answering for each other's identically-named symbols.
//
// Path: QUACTLIZE_PPU_LIB_FMT<k>, else QUACTLIZE_PPU_LIB with _fmt<k> spliced before .so, else
// libquactlize_ppu_fmt<k>.so. load() without a format keeps its exact previous behaviour.
Api const* load_format(int fmt, std::string* why = nullptr);

Api const* load(std::string* why = nullptr);

// "ppu" when every symbol resolved, otherwise "cpu". Exposed as an op so a caller can assert on it instead of
// inferring the backend from a timing.
std::string resolved_backend();

}  // namespace ppu_backend
}  // namespace torch_ext
