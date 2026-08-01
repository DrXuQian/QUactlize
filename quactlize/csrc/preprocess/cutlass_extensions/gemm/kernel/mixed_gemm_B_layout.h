// SHIM for TensorRT-LLM's cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.h, which was not copied with the port.
//
// The port needs exactly two things from it, and the .cpp defines everything else itself (LayoutDetails and
// getLayoutDetails are declared locally, so only these remain):
//
//   ColumnMajorTileInterleave<RowsPerTile, ColumnsInterleaved>  -- a layout TAG. It is never instantiated as a real
//       cutlass layout here; it is only a compile-time key for getLayoutDetails, which reads the two integers back
//       out. So a tag with the two parameters is the whole contract.
//   LayoutDetailsB<TypeB, Arch>                                 -- per (element, architecture) preprocessing traits:
//       which layout B is stored in and the interleave granularity.
//
// STATUS OF THE VALUES BELOW: NVIDIA-DERIVED, MEMORY-SAFE, PPU-UNVERIFIED. This header once claimed they were "the
// PPU's, not NVIDIA's". That claim is false and the correction matters more than the constant it corrects.
//
// ColumnsInterleaved = ElementsPerCacheLine / ThreadblockK, with a 128-byte line, is TensorRT-LLM's own expression
// for an SM75/SM80 tensor core, reproduced here exactly. Fixing the earlier hardcoded 256 made the preprocessing
// memory-safe -- it was writing a buffer's length past its destination -- and made it agree with the NVIDIA-derived
// consumer. It did NOT establish that this is what the PPU's AIU wants. A cache line size cannot determine a
// swizzle on its own: lane mapping, load width, the MMA K tile and the consumer's own address arithmetic do.
//
// Note also that the preprocessing COMPOSES two layouts -- this column interleave first, then the PPU-specific
// 256-row AIU interleave in cutlass_preprocessors.cpp. The AIU test compares the two-step result against the
// one-step result, so both sides contain this transform and the test is structurally blind to it being wrong.
//
// What would settle it, in increasing order of what it proves:
//   * the device kernel's B-load offset and lane map, or the vendor's preprocessing contract for that kernel;
//   * an offset-by-offset comparison of this output against the device iterator's expected addresses;
//   * an end-to-end PPU run on basis-vector weights against a CPU dense golden, at shapes where k and n are not
//     multiples of each other or of the tile.
// Until one of those exists, do not describe this as the layout the PPU kernels assume.
//
// Anything using a specialisation that is not listed falls back to the unspecialised template, which has no members
// and therefore fails to compile -- deliberately. A missing (element, arch) pair must be added on purpose.
#pragma once
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

namespace cutlass {
namespace layout {
template <int RowsPerTile, int ColumnsInterleaved>
struct ColumnMajorTileInterleave {
  static constexpr int kRowsPerTile = RowsPerTile;
  static constexpr int kColumnsInterleaved = ColumnsInterleaved;
};
}  // namespace layout

// TensorRT-LLM's cutlass_extensions adds this operator tag to distinguish "B is interleaved and dequantised into A's
// type" from the stock multiply-add. The preprocessor only uses it as a TYPE KEY -- it never issues an instruction
// through it -- so a tag is the whole contract here, exactly as with ColumnMajorTileInterleave above.
namespace arch {
struct OpMultiplyAddDequantizeInterleavedBToA {};
}  // namespace arch

namespace gemm {
namespace kernel {

// Unspecialised on purpose: an (element, arch) pair nobody has decided about must not silently pick a layout.
template <typename TypeB, typename Arch, typename Enable = void>
struct LayoutDetailsB {};

// COLUMNS INTERLEAVED IS DERIVED, NOT WRITTEN DOWN. The interleave exists so that one 128-byte cache line holds a
// whole ThreadblockK-tall column tile: a line holds ElementsPerCacheLine = 128*8/bits elements, a tile is
// ThreadblockK of them, so ColumnsInterleaved = ElementsPerCacheLine / ThreadblockK -- 2 for 8-bit, 4 for 4-bit,
// 8 for 2-bit. The same expression appears in cutlass_preprocessors.cpp's AIU branch as `128 * 8 / bits`.
//
// This was originally written as the literal 256 for every width, which is ElementsPerCacheLine for 4-bit -- the
// numerator mistaken for the quotient. It was not a cosmetic error: interleave_column_major_tensor strides by this
// value, so with 256 it wrote a full buffer's length past the end of its destination on any matrix narrower than 256
// columns. ASAN caught it as a heap-buffer-overflow; before that it surfaced as an intermittent Bus error or SIGSEGV
// in an unrelated test, several tests later. Deriving it means the next element width cannot repeat this.
//
// `Operator` names the MMA operator the preprocessing is FOR, and the caller reads it to decide whether B is stored
// interleaved. Both members are required by every specialisation -- the compile error when one is missing is the
// point, since silently defaulting it would pick an arrangement nobody chose.
namespace detail {
// 128 bytes is the cache line the interleave is sized against.
template <int Bits, int ThreadblockK>
struct InterleaveFor {
  static constexpr int kElementsPerCacheLine = 128 * 8 / Bits;
  static constexpr int value = kElementsPerCacheLine / ThreadblockK;
  static_assert(value * ThreadblockK == kElementsPerCacheLine,
                "ThreadblockK must divide the elements in a cache line, or a column tile straddles two lines");
  static_assert(value >= 1, "a column tile wider than a cache line has no interleave to describe");
};
}  // namespace detail

template <typename Arch>
struct LayoutDetailsB<uint8_t, Arch> {
  static constexpr int ThreadblockK = 64;
  using Layout = layout::ColumnMajorTileInterleave<ThreadblockK, detail::InterleaveFor<8, ThreadblockK>::value>;
  using Operator = arch::OpMultiplyAddDequantizeInterleavedBToA;
  static constexpr int ElementsPerAccess = 16;
};
template <typename Arch>
struct LayoutDetailsB<cutlass::int4b_t, Arch> {
  static constexpr int ThreadblockK = 64;
  using Layout = layout::ColumnMajorTileInterleave<ThreadblockK, detail::InterleaveFor<4, ThreadblockK>::value>;
  using Operator = arch::OpMultiplyAddDequantizeInterleavedBToA;
  static constexpr int ElementsPerAccess = 32;
};
template <typename Arch>
struct LayoutDetailsB<cutlass::uint4b_t, Arch> : LayoutDetailsB<cutlass::int4b_t, Arch> {};
template <typename Arch>
struct LayoutDetailsB<cutlass::uint2b_t, Arch> {
  static constexpr int ThreadblockK = 64;
  using Layout = layout::ColumnMajorTileInterleave<ThreadblockK, detail::InterleaveFor<2, ThreadblockK>::value>;
  using Operator = arch::OpMultiplyAddDequantizeInterleavedBToA;
  static constexpr int ElementsPerAccess = 64;
};

static_assert(LayoutDetailsB<uint8_t, cutlass::arch::Sm80>::Layout::kColumnsInterleaved == 2, "8-bit: 128/64");
static_assert(LayoutDetailsB<cutlass::int4b_t, cutlass::arch::Sm80>::Layout::kColumnsInterleaved == 4, "4-bit: 256/64");
static_assert(LayoutDetailsB<cutlass::uint2b_t, cutlass::arch::Sm80>::Layout::kColumnsInterleaved == 8, "2-bit: 512/64");

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
