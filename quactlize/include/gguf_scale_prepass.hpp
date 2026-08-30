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
#include "gguf_packed_unit.hpp"

namespace gguf_scale {
namespace prepass {

using cutlass::half_t;
using ::gguf_scale::KType;
using ::gguf_scale::Traits;
using ::gguf_scale::GroupScale;

// ONE SUPERBLOCK-COLUMN'S WORTH OF DECODE, and the ONLY place the arithmetic exists. The device kernel below and the
// host reference both call this, so "the two agree" is not a property that has to be tested -- there is one of them.
// What DOES have to be tested is that this agrees with the format spec, which is what l101 does, against a literal
// transcription rather than against this.
template <KType T, int ZMul>
CUTLASS_HOST_DEVICE GroupScale
group_scale_zero(uint8_t const* block, int g, half_t d, half_t dmin) {
  using Tr = Traits<T>;
  GroupScale out;
  int const sc = ::gguf_scale::scale_of<T>(block, g);
  if constexpr (Tr::kHasMin) {
    out = ::gguf_scale::make_group_scale<T>(sc, ::gguf_scale::min_of<T>(block, g), d, dmin);
  } else {
    // NO MIN CHANNEL AT ALL for Q3_K and Q6_K: the code's own centre is carried by Traits::kScaleBias (32 for Q3_K)
    // or by the sign of the code itself (Q6_K's int8), so there is nothing for a format-level zero to hold.
    out.scale = ::gguf_scale::make_group_scale_only<T>(sc, d);
    out.zero  = half_t(0.f);
  }
  // THE CONSUMER'S CORRECTION, LAST. It is added to the SPLIT zero, exactly as the in-kernel decoder does it, so a
  // format with no min still gets it -- the converter's shift does not care whether the format had an affine term.
  // THE HOST AND DEVICE RESULTS DIFFER BY UP TO 1 ULP HERE, and only here. On the host this goes through float and
  // rounds once; on the device half_t arithmetic lowers to native fp16, so 8*scale is rounded before the add.
  // Measured on a 5090: Q3_K and Q6_K (ZMul = 0) are bit-identical, while Q2_K, Q4_K and Q5_K differ by 3.12e-2,
  // which is exactly one fp16 ulp at magnitude 32. Keep this expression explicitly in float on BOTH sides: leaving
  // the operands as half_t makes CUDA lower it to native fp16 (round the product, then the sum), while the host does
  // one float FMA-shaped expression and rounds once. A pre-pass produces a persistent artifact, so having its host
  // and device builders disagree by a bit is a worse contract than saving two scalar conversions per group.
  if constexpr (ZMul != 0) {
    out.zero = half_t(float(out.zero) + float(ZMul) * float(out.scale));
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

// The merged BC artifact's scale source and destination. Units are stored K-major exactly as the packed GEMM copies
// them: [expert, copyable-unit, column, kUnitTotal]. The destination is the existing consumer-ready plane convention
// [expert, superblock*group, column]. Keeping every stride explicit matters for Q3/Q6: one 28/36-byte COPYABLE unit
// contains two independently headed superblocks, and treating kUnitTotal as one scale block would decode only half.
struct UnitPlaneDesc {
  half_t* scale;
  half_t* zero;
  int64_t stride_e;
  int64_t stride_k;
  int64_t stride_n;
};

template <KType T, int ZMul>
void prepass_unit_host(uint8_t const* units, UnitPlaneDesc const& dst,
                       int num_experts, int num_cols, int num_superblocks) {
  using U = packed_unit::Unit<T>;
  int const num_units = num_superblocks / U::kSbPerUnit;
  for (int e = 0; e < num_experts; ++e) {
    for (int sb = 0; sb < num_superblocks; ++sb) {
      int const unit = sb / U::kSbPerUnit;
      int const sb_in_unit = sb % U::kSbPerUnit;
      for (int n = 0; n < num_cols; ++n) {
        uint8_t const* src = units + ((int64_t(e) * num_units + unit) * num_cols + n) * U::kUnitTotal;
        for (int g = 0; g < U::kGroups; ++g) {
          GroupScale const sz = packed_unit::unit_group_sb<T, ZMul>(src, sb_in_unit, g);
          int64_t const o = int64_t(e) * dst.stride_e
                          + (int64_t(sb) * U::kGroups + g) * dst.stride_k
                          + int64_t(n) * dst.stride_n;
          dst.scale[o] = sz.scale;
          if (dst.zero) dst.zero[o] = sz.zero;
        }
      }
    }
  }
}

// THE SAME PRE-PASS, READING THE PACKED UNIT INSTEAD OF THE GGUF BLOCK. This is what makes ONE offline artifact
// serve every route: the packed collective reads the unit in the kernel, this expands it to fp16 planes for the
// collectives that want planes, and the GEMV reads it per group. Without it the reordered checkpoint would need the
// GGUF scale block kept beside it purely so the pre-pass had something to read, which is stored bytes for nothing.
//
// AND THE PLANES NEED NO TRANSFORMATION FOR AN OFFLINE-REORDERED WEIGHT. I claimed the opposite twice, in both
// directions, before measuring; the answer is neither and only becomes visible from the reorder's own code.
//
// The k axis is safe because the row permutation stays inside its own block: measured at max displacement 18 over a
// 32-row permutation for int4 and 6 over 16 rows for int8, 100% of elements inside their own 32-block in both. A
// scale indexed by k//gs at gs=32 therefore cannot see it. (The int4 measurement needs two label passes -- 4-bit
// labels alias across a 32-row permutation, and a single pass reports agreement it has not established.)
//
// The n axis is safe for a different and less obvious reason. mem_cacheline_col_tile_interleave does not permute n
// within the n axis; it FOLDS `interleave` adjacent columns into one, moving them along the row axis --
// write_col = read_col / interleave, vec_write_row = interleave*base + vec_rows_per_tile*(read_col % interleave) +
// ... -- so the output has N/interleave columns and interleave*K rows. The kernel must recover the source n to write
// its output column at all, and having recovered it, it indexes the scale in LOGICAL (n, k//gs) order. The reorder
// changes where bytes sit, not which logical coordinate they are.
//
// Corroboration that this is the design and not a coincidence: nothing in this codebase ever preprocesses a scale
// tensor. preprocess_weights_to_layout takes only the weight, and symmetric_quantize returns its scales unprocessed.
//
// The weight CODES are not read here in any case, so what happens to them is invisible to this path.
template <KType T, int ZMul>
CUTLASS_HOST_DEVICE GroupScale group_scale_zero_from_unit(uint8_t const* unit, int g) {
  return ::gguf_scale::packed_unit::unit_group<T, ZMul>(unit, g);
}

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
// THE OLD ONE-THREAD KERNEL, retained as the bit-exact device reference. It is intentionally not the production name:
// one lane serialises 8 or 16 groups and, more importantly, a warp's stores are separated by a whole scale row. The
// cooperative kernel below is checked against this one in the CUDA golden before either is timed.
template <KType T, int ZMul>
__global__ void prepass_kernel_serial(BlockDesc src, PlaneDesc dst, int num_cols, int num_superblocks) {
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

// FOUR LANES PER (column, superblock), EIGHT SUPERBLOCKS PER WARP. The measured winner's TiledCopy thread layout is
// `(group=4, block=8):(1,4)`: group is lane-fast, and each pass is one 32-element copy per plane. Q4_K/Q5_K use two
// four-group passes; Q2_K/Q3_K/Q6_K use four. Every group is decoded exactly once -- this is not the failed shape
// where every lane replicated the traversal and retained its share.
//
// REVIEW NOTE ON THE REPORTED BANDWIDTH: the 16x figure came out at 2.138 TB/s, ABOVE the 5090's 1.792 TB/s DRAM
// peak, because dirty output can be written back during the untimed L2 flush that follows the timed launch. So that
// number is not a valid bandwidth -- it is evidence the measurement leaks. The SPEEDUP ratios are still sound, since
// both arms use the same method, and an independent harness reproduced the direction and the bit-exactness (1.32x,
// 1.91x, 2.95x at a smaller shape, poisoned outputs fully overwritten).
//
// The proposed `(group=8, block=4):(1,8)` mapping is fully coalesced, but it redoes the header and address work in
// twice as many threads. On a 5090 with cold L2, 4 lanes measured 6.112/14.304/47.104 us at 1x/4x/16x versus
// 8.160/18.400/65.536 us for 8 lanes. Two lanes lost store efficiency (10.208/24.544/81.888 us), so four is the
// measured balance rather than a coalescing assumption.
//
// Columns are kept separate in the warp mapping. This costs at most seven inactive four-lane subgroups when a
// caller's superblock count is not divisible by eight, but it means a copy tile never straddles `stride_n`: arbitrary
// caller-supplied plane strides keep the same contract as the serial reference.
//
// Launch `prepass_thread_count(num_cols, num_superblocks)` threads, with a block size divisible by 32. Stating the
// geometry as code matters: reusing the serial kernel's `total`-thread launch would execute only one quarter of the
// work and can look correct on an output buffer that was not poisoned first.
static constexpr int kPrepassThreadsPerSuperblock = 4;
CUTLASS_HOST_DEVICE constexpr int64_t prepass_thread_count(int num_cols, int num_superblocks) {
  return int64_t(num_cols) * ((num_superblocks + 7) / 8) * 32;
}
CUTLASS_HOST_DEVICE constexpr int prepass_grid_size(int num_cols, int num_superblocks, int threads_per_cta) {
  return int((prepass_thread_count(num_cols, num_superblocks) + threads_per_cta - 1) / threads_per_cta);
}

template <KType T, int ZMul>
__global__ void prepass_kernel(BlockDesc src, PlaneDesc dst, int num_cols, int num_superblocks) {
  constexpr int kG = Traits<T>::kGroups;
  static_assert(kG == 8 || kG == 16, "the warp mapping handles two or four four-group tiles");
  using CopyLayout = cute::Layout<cute::Shape<cute::_4, cute::_8>, cute::Stride<cute::_1, cute::_4>>;
  auto tiled_copy = cute::make_tiled_copy(
      cute::Copy_Atom<cute::UniversalCopy<half_t>, half_t>{}, CopyLayout{}, cute::Layout<cute::Shape<cute::_1>>{});

  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  int const warp = tid >> 5;
  int const lane = threadIdx.x & 31;
  int const warp_groups_per_col = (num_superblocks + 7) / 8;
  int const n = warp / warp_groups_per_col;
  if (n >= num_cols) return;

  // Derive (group-within-pass, superblock-within-warp) from the copy itself. Keeping a second hand-written lane
  // formula beside the TiledCopy would let the arithmetic and the destinations drift independently.
  auto thr_copy = tiled_copy.get_thread_slice(lane);
  auto tile_coord = cute::make_identity_tensor(cute::make_shape(cute::_4{}, cute::_8{}));
  auto thr_coord = thr_copy.partition_D(tile_coord);
  auto const coord = thr_coord(0);
  int const g0 = int(cute::get<0>(coord));
  int const sb0 = (warp - n * warp_groups_per_col) * 8;
  int const sb = sb0 + int(cute::get<1>(coord));
  if (sb >= num_superblocks) return;

  uint8_t const* blk = src.blocks + n * src.block_stride_n + sb * src.block_stride_sb;
  int64_t const hi = n * src.hdr_stride_n + sb * src.hdr_stride_sb;
  half_t const d = src.d[hi];
  half_t const dmin = src.dmin ? src.dmin[hi] : half_t(0.f);

  CUTLASS_PRAGMA_UNROLL
  for (int pass = 0; pass < kG / 4; ++pass) {
    int const g = g0 + pass * 4;
    GroupScale const sz = group_scale_zero<T, ZMul>(blk, g, d, dmin);

    auto tile_layout = cute::make_layout(cute::make_shape(cute::_4{}, cute::_8{}),
                                         cute::make_stride(dst.stride_k, int64_t(kG) * dst.stride_k));
    int64_t const tile_offset = n * dst.stride_n + (int64_t(sb0) * kG + pass * 4) * dst.stride_k;
    auto scale_tile = cute::make_tensor(cute::make_gmem_ptr(dst.scale + tile_offset), tile_layout);
    auto thr_scale = thr_copy.partition_D(scale_tile);
    auto scale_fragment = cute::make_fragment_like(thr_scale);
    scale_fragment(0) = sz.scale;
    cute::copy(tiled_copy, scale_fragment, thr_scale);

    if (dst.zero) {
      auto zero_tile = cute::make_tensor(cute::make_gmem_ptr(dst.zero + tile_offset), tile_layout);
      auto thr_zero = thr_copy.partition_D(zero_tile);
      auto zero_fragment = cute::make_fragment_like(thr_zero);
      zero_fragment(0) = sz.zero;
      cute::copy(tiled_copy, zero_fragment, thr_zero);
    }
  }
}

// FOUR GROUPS x EIGHT COLUMNS per warp for one logical superblock. This is the transpose of the raw prepass's
// ownership because the two sources have different resident orders: packed units are [unit,N,bytes], and the fp16
// destination is [K-group,N]. Lanes therefore stay contiguous along N on both sides. Each pass advances four groups;
// Q4/Q5 take two passes and Q2/Q3/Q6 four. The source record is selected through (unit,sb_in_unit), so the device path
// witnesses the same paired-superblock axis as prepass_unit_host rather than relying on an outer caller to offset it.
template <KType T, int ZMul>
__global__ void prepass_unit_kernel(uint8_t const* units, UnitPlaneDesc dst,
                                    int num_experts, int num_cols, int num_superblocks) {
  using U = packed_unit::Unit<T>;
  static_assert(U::kGroups == 8 || U::kGroups == 16, "packed unit warp mapping expects 8 or 16 groups");
  static_assert(U::kUnitTotal == U::kSbPerUnit * U::kSbBytes,
                "the kernel must address a complete copyable/paired unit");
  int const warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int const lane = threadIdx.x & 31;
  int const col_tiles = (num_cols + 7) / 8;
  int const warps_per_expert = num_superblocks * col_tiles;
  int const e = warp / warps_per_expert;
  if (e >= num_experts) return;
  int const in_expert = warp - e * warps_per_expert;
  int const sb = in_expert / col_tiles;
  int const n = (in_expert - sb * col_tiles) * 8 + lane / 4;
  if (n >= num_cols) return;
  int const g0 = lane & 3;
  int const unit = sb / U::kSbPerUnit;
  int const sb_in_unit = sb % U::kSbPerUnit;
  int const num_units = num_superblocks / U::kSbPerUnit;
  uint8_t const* src = units + ((int64_t(e) * num_units + unit) * num_cols + n) * U::kUnitTotal;

  CUTLASS_PRAGMA_UNROLL
  for (int pass = 0; pass < U::kGroups / 4; ++pass) {
    int const g = g0 + 4 * pass;
    GroupScale const sz = packed_unit::unit_group_sb<T, ZMul>(src, sb_in_unit, g);
    int64_t const o = int64_t(e) * dst.stride_e
                    + (int64_t(sb) * U::kGroups + g) * dst.stride_k
                    + int64_t(n) * dst.stride_n;
    dst.scale[o] = sz.scale;
    if (dst.zero) dst.zero[o] = sz.zero;
  }
}

// ONE THREAD PER (expert, superblock, column), retained as a device-localisation arm for the packed-unit prepass.
// It deliberately shares unit_group_sb and the destination descriptor with the cooperative kernel above; the only
// changed variable is ownership/placement.  This matters on PPU, where a cooperative launch can return success yet
// leave a plane unwritten without implicating the packed-unit bit map.  Do not use a host fallback for this check:
// that would remove both device decoding and device placement at once and could not adjudicate either one.
template <KType T, int ZMul>
__global__ void prepass_unit_kernel_serial(uint8_t const* units, UnitPlaneDesc dst,
                                           int num_experts, int num_cols, int num_superblocks) {
  using U = packed_unit::Unit<T>;
  // This counterfactual is intentionally int32.  PPU's host launch accepts int64 address arithmetic, but using a
  // 64-bit divide in the thread-ownership guard made an unwritten destination indistinguishable from a bad unit
  // decode.  Production probe shapes are far below INT_MAX; keep int64 only in byte/destination offsets below.
  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  int const per_expert = num_superblocks * num_cols;
  int const total = num_experts * per_expert;
  if (tid >= total) return;

  int const e = tid / per_expert;
  int const in_expert = tid - e * per_expert;
  int const sb = in_expert / num_cols;
  int const n = in_expert - sb * num_cols;
  int const num_units = num_superblocks / U::kSbPerUnit;
  int const unit = sb / U::kSbPerUnit;
  int const sb_in_unit = sb % U::kSbPerUnit;
  uint8_t const* src = units +
      ((int64_t(e) * num_units + unit) * num_cols + n) * U::kUnitTotal;

  CUTLASS_PRAGMA_UNROLL
  for (int g = 0; g < U::kGroups; ++g) {
    GroupScale const sz = packed_unit::unit_group_sb<T, ZMul>(src, sb_in_unit, g);
    int64_t const o = int64_t(e) * dst.stride_e
                    + (int64_t(sb) * U::kGroups + g) * dst.stride_k
                    + int64_t(n) * dst.stride_n;
    dst.scale[o] = sz.scale;
    if (dst.zero) dst.zero[o] = sz.zero;
  }
}

template <KType T>
CUTLASS_HOST_DEVICE constexpr int prepass_unit_grid_size(
    int num_experts, int num_cols, int num_superblocks, int threads_per_cta) {
  int64_t const warps = int64_t(num_experts) * num_superblocks * ((num_cols + 7) / 8);
  int64_t const threads = warps * 32;
  return int((threads + threads_per_cta - 1) / threads_per_cta);
}

CUTLASS_HOST_DEVICE constexpr int prepass_unit_serial_grid_size(
    int num_experts, int num_cols, int num_superblocks, int threads_per_cta) {
  int64_t const threads = int64_t(num_experts) * num_superblocks * num_cols;
  return int((threads + threads_per_cta - 1) / threads_per_cta);
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

}  // namespace prepass
}  // namespace gguf_scale
