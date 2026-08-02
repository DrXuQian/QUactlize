#pragma once
// THE ONLINE SCALE PRE-PASS: GGUF k-quant scale metadata -> the fp16 (scale, zero) planes every existing collective
// already consumes. The quantised WEIGHTS are not touched; only the scale channel is expanded, into a workspace.
//
// WHY THIS IS THE THING THAT MAKES "WE SUPPORT GGUF" TRUE. quactlize/schemes.py's matrix has exactly ONE validated
// cell for k-quants -- FULLY_QUANTIZED x GROUPED x Q4_K, the in-kernel packed path. Every other cell is ABSENT or
// PARTIAL, and the reasons recorded there are the same reason twice:
//
//   * GEMV, all k-quants: "the GEMV kernels read fp16 scale planes; at decode those must be resident, and for a
//     k-quant that is the stored-byte increase the constraint forbids"
//   * GROUPED, Q3/Q5/Q6: "two-plane formats run through the SEPARATE two-plane collective, which has no packed-scale
//     plumbing at all. This is structural, not unfinished"
//
// So today a real GGUF checkpoint runs only if the caller already holds fp16 planes -- which for Q4_K costs +11.1%
// stored bytes, and the whole project forbids that. This pre-pass removes the storage objection instead of the
// residency one: the planes live in a WORKSPACE, which formats.py:176 explicitly permits, and every collective that
// already works for GPTQ starts working for all five k-quants at once. formats.select_path already returns this path
// under fp16_planes="workspace" and calls it "the prefill pre-pass"; this is the kernel that promise refers to.
//
// AND WHAT IT DOES NOT FIX, stated here so nobody reads more into it. It removes the STORAGE cost, not the
// RESIDENCY cost. At prefill the planes are built once and amortised over many tokens. At decode they would have to
// be rebuilt every token, or kept -- which is the forbidden storage again. Decode still needs the native in-kernel
// path, and that is still Q4_K-grouped only.
//
// ---------------------------------------------------------------------------------------------------------------
// THE ZERO POINT HAS TWO TERMS FROM TWO DIFFERENT AXES, and conflating them is the known way to get this wrong.
//
//     zero = -dmin * mn            the FORMAT's affine term, chosen by the scale format (KType)
//          + ZMul * scale          the CONSUMER's centre correction, chosen by the weight width / converter
//
// The second is not optional and not a property of the scale format. The int4 converter emits q-8 where a k-quant
// means q, so the packed in-kernel path carries kPackedZMul = 8 to cancel it -- applied AFTER the (scale, zero) pair
// is split, which is why packing the pre-correction pair back into one word is wrong rather than merely different.
// A prepass that produced only -dmin*mn would be numerically wrong for exactly the same reason, and silently: the
// planes would look plausible and every product would be off by 8*scale*<the converter's shift>.
//
// So ZMul is a template parameter beside the format, and callers must pass the one their converter uses. There is no
// default, deliberately.
#include "cutlass/numeric_types.h"
#include "gguf_scale_layout.hpp"
#include "gguf_scale_decode.hpp"

namespace quactlize {
namespace gguf_prepass {

using cutlass::half_t;
using ::quactlize::gguf::KType;
using ::quactlize::gguf::Traits;
using ::quactlize::gguf::GroupScale;

// ONE SUPERBLOCK-COLUMN'S WORTH OF DECODE, and the ONLY place the arithmetic exists. The device kernel below and the
// host reference both call this, so "the two agree" is not a property that has to be tested -- there is one of them.
// What DOES have to be tested is that this agrees with the format spec, which is what l101 does, against a literal
// transcription rather than against this.
template <KType T, int ZMul>
CUTLASS_HOST_DEVICE GroupScale
group_scale_zero(uint8_t const* block, int g, half_t d, half_t dmin) {
  using Tr = Traits<T>;
  GroupScale out;
  int const sc = ::quactlize::gguf::scale_of<T>(block, g);
  if constexpr (Tr::kHasMin) {
    out = ::quactlize::gguf::make_group_scale<T>(sc, ::quactlize::gguf::min_of<T>(block, g), d, dmin);
  } else {
    // NO MIN CHANNEL AT ALL for Q3_K and Q6_K: the code's own centre is carried by Traits::kScaleBias (32 for Q3_K)
    // or by the sign of the code itself (Q6_K's int8), so there is nothing for a format-level zero to hold.
    out.scale = ::quactlize::gguf::make_group_scale_only<T>(sc, d);
    out.zero  = half_t(0.f);
  }
  // THE CONSUMER'S CORRECTION, LAST. It is added to the SPLIT zero, exactly as the in-kernel decoder does it, so a
  // format with no min still gets it -- the converter's shift does not care whether the format had an affine term.
  if constexpr (ZMul != 0) {
    out.zero = out.zero + half_t(float(ZMul)) * out.scale;
  }
  return out;
}

// THE PLANE GEOMETRY IS THE CALLER'S, NOT OURS. The collectives describe their scale tensor with a stride the host
// builds (NonVoidStrideScale), and inventing a convention here would produce a kernel that decodes correctly and
// writes to the wrong addresses -- the exact failure shape this file's neighbours keep recording. So the layout
// arrives as explicit strides in ELEMENTS and this header makes no assumption about which axis is contiguous.
struct PlaneDesc {
  half_t*  scale;         // [n * scale_stride_n + kg * scale_stride_k]
  half_t*  zero;          // same indexing; may be null when the consumer has no zero channel
  int64_t  stride_n;
  int64_t  stride_k;
};

// The source side: one scale block per (column, superblock), plus that superblock's fp16 header.
struct BlockDesc {
  uint8_t const* blocks;      // [n * block_stride_n + sb * block_stride_sb] bytes, Traits<T>::kBlockBytes each
  half_t  const* d;           // [n * hdr_stride_n + sb * hdr_stride_sb]
  half_t  const* dmin;        // same indexing; may be null for scale-only formats
  int64_t block_stride_n, block_stride_sb;
  int64_t hdr_stride_n,   hdr_stride_sb;
};

// THE HOST REFERENCE. Not a test helper -- it is the definition of what the kernel must produce, and the CI gate
// checks THIS against the format spec. A device kernel that matches it is then correct by construction.
template <KType T, int ZMul>
void prepass_host(BlockDesc const& src, PlaneDesc const& dst, int num_cols, int num_superblocks) {
  constexpr int kG = Traits<T>::kGroups;
  for (int n = 0; n < num_cols; ++n) {
    for (int sb = 0; sb < num_superblocks; ++sb) {
      uint8_t const* blk = src.blocks + n * src.block_stride_n + sb * src.block_stride_sb;
      int64_t  const hi  = n * src.hdr_stride_n + sb * src.hdr_stride_sb;
      half_t const d     = src.d[hi];
      half_t const dmin  = src.dmin ? src.dmin[hi] : half_t(0.f);
      for (int g = 0; g < kG; ++g) {
        GroupScale const sz = group_scale_zero<T, ZMul>(blk, g, d, dmin);
        int64_t const o = n * dst.stride_n + (int64_t(sb) * kG + g) * dst.stride_k;
        dst.scale[o] = sz.scale;
        if (dst.zero) dst.zero[o] = sz.zero;
      }
    }
  }
}

#if defined(__CUDACC__) || defined(__HGGCCC__)
// ONE THREAD PER (column, superblock). That is the granularity the format itself has -- d and dmin are per
// superblock and shared by all its groups -- so a finer split would re-read the header once per group, and a coarser
// one would leave the k axis serial. It is a pure bandwidth kernel: kBlockBytes + 4 in, kGroups * (2 or 4) out.
template <KType T, int ZMul>
__global__ void prepass_kernel(BlockDesc src, PlaneDesc dst, int num_cols, int num_superblocks) {
  constexpr int kG = Traits<T>::kGroups;
  int const tid   = blockIdx.x * blockDim.x + threadIdx.x;
  int const total = num_cols * num_superblocks;
  if (tid >= total) return;
  int const n  = tid / num_superblocks;
  int const sb = tid - n * num_superblocks;

  uint8_t const* blk = src.blocks + n * src.block_stride_n + sb * src.block_stride_sb;
  int64_t  const hi  = n * src.hdr_stride_n + sb * src.hdr_stride_sb;
  half_t const d    = src.d[hi];
  half_t const dmin = src.dmin ? src.dmin[hi] : half_t(0.f);

  CUTLASS_PRAGMA_UNROLL
  for (int g = 0; g < kG; ++g) {
    GroupScale const sz = group_scale_zero<T, ZMul>(blk, g, d, dmin);
    int64_t const o = n * dst.stride_n + (int64_t(sb) * kG + g) * dst.stride_k;
    dst.scale[o] = sz.scale;
    if (dst.zero) dst.zero[o] = sz.zero;
  }
}
#endif

// THE TRAFFIC, as an expression rather than a comment, because the whole case for this path is a byte count and a
// comment cannot be invalidated by a shape change. Read: one scale block plus a 2- or 4-byte header per (column,
// superblock). Written: kGroups fp16 per plane.
template <KType T>
CUTLASS_HOST_DEVICE constexpr int bytes_per_column_superblock(bool with_zero) {
  return Traits<T>::kBlockBytes + (Traits<T>::kHasMin ? 4 : 2)
       + Traits<T>::kGroups * 2 * (with_zero ? 2 : 1);
}

}  // namespace gguf_prepass
}  // namespace quactlize
