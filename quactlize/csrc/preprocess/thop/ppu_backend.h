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
  // blocks [rows*blocks_per_row, block_bytes], x fp16 [blocks_per_row*256], out fp32 [rows]
  int (*vecdot)(uint8_t const* blocks, int64_t block_bytes, uint16_t const* x, float* out,
                int rows, int blocks_per_row, int qtype);
  // blocks [n_blocks, block_bytes] -> out fp16, dst_span elements per block
  int (*dequantize)(uint8_t const* blocks, int64_t block_bytes, uint16_t* out, int n_blocks, int qtype);
  // scale blocks + fp16 headers -> the two planes
  int (*prepass)(uint8_t const* blocks, int64_t block_bytes, uint16_t const* d, uint16_t const* dmin,
                 int n, uint16_t* scale, uint16_t* zero, int groups, int qtype, int zmul);
};

// Loads libquactlize_ppu.so once. QUACTLIZE_PPU_LIB overrides the path, which is what lets a stub stand in for the
// real thing in a test -- the seam is checkable without the SDK, the same way box_build_dryrun checks build.sh.
// Returns nullptr when the library is absent or a symbol is missing, with `why` set.
Api const* load(std::string* why = nullptr);

// "ppu" when every symbol resolved, otherwise "cpu". Exposed as an op so a caller can assert on it instead of
// inferring the backend from a timing.
std::string resolved_backend();

}  // namespace ppu_backend
}  // namespace torch_ext
