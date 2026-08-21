/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

namespace cutlass::gemm::collective::detail {

// The mixed-input API is deliberately wider than the resident formats.  A
// placed/interleaved B tensor, for example, has one canonical physical pitch;
// accepting an arbitrary dB and then rebuilding that canonical layout in the
// mainloop is not support for arbitrary dB.  Keep the host-side admission
// rules in one POD so query and launch consume exactly the same predicate.
enum MixedArgumentIssue : uint32_t {
  MixedArgumentOk                 = 0,
  MixedArgumentBadGroupSize       = 1u << 0,
  MixedArgumentStaticGroupDrift   = 1u << 1,
  MixedArgumentScaleTileDrift     = 1u << 2,
  MixedArgumentBadLowFold         = 1u << 3,
  MixedArgumentBadHighFold        = 1u << 4,
  MixedArgumentBadLowInterleave   = 1u << 5,
  MixedArgumentBadHighInterleave  = 1u << 6,
  MixedArgumentBadLowStride       = 1u << 7,
  MixedArgumentBadHighStride      = 1u << 8,
  MixedArgumentMissingHighStride  = 1u << 9,
  MixedArgumentPackedZeroPointer  = 1u << 10,
  MixedArgumentPackedScaleStride  = 1u << 11,
  MixedArgumentPackedGroupTail    = 1u << 12,
  MixedArgumentPackedTileTail     = 1u << 13,
  MixedArgumentPackedUnitTail     = 1u << 14,
  MixedArgumentFractionalLowByte  = 1u << 15,
  MixedArgumentFractionalHighByte = 1u << 16,
};

CUTE_HOST_DEVICE constexpr bool mixed_bit_offset_byte_aligned(
    int64_t elements, int bits) {
  if (bits <= 0) return false;
  int a = bits;
  int b = 8;
  while (b != 0) {
    int const r = a % b;
    a = b;
    b = r;
  }
  return elements % (8 / a) == 0;
}

struct MixedArgumentContract {
  int64_t n = 0;
  int64_t k = 0;
  int64_t l = 0;
  int64_t group_size = 0;
  int64_t tile_k = 0;
  int64_t scale_tile_k = 0;
  int static_group_size = 0;       // 0=runtime, -1=per-column, >0=static

  int low_fold = 1;
  int high_fold = 1;
  int low_bits = 0;
  int high_bits = 0;
  int interleave = 1;
  int packed_tiles_per_unit = 1;
  bool has_scales = false;
  bool two_plane = false;
  bool packed_scale = false;
  bool dB2_valid = false;
  bool ptr_Z_nonnull = false;

  int64_t dB0 = 0;
  int64_t dB1 = 0;
  int64_t dBL = 0;
  int64_t dB20 = 0;
  int64_t dB21 = 0;
  int64_t dB2L = 0;
  int64_t dS0 = 0;
  int64_t dS1 = 0;
  int64_t dSL = 0;
};

CUTE_HOST_DEVICE constexpr uint32_t mixed_argument_issues(
    MixedArgumentContract const& x) {
  uint32_t issues = MixedArgumentOk;
  int64_t scale_k = 0;
  if (x.has_scales) {
    if (x.group_size <= 0) {
      issues |= MixedArgumentBadGroupSize;
    } else {
      scale_k = (x.k + x.group_size - 1) / x.group_size;
      if ((x.static_group_size > 0 && x.group_size != x.static_group_size) ||
          (x.static_group_size == -1 && x.group_size != x.k)) {
        issues |= MixedArgumentStaticGroupDrift;
      }
      int64_t const expected_groups =
          (x.tile_k + x.group_size - 1) / x.group_size;
      if (x.scale_tile_k != expected_groups) {
        issues |= MixedArgumentScaleTileDrift;
      }
    }
  }

  if (x.low_fold <= 0 || x.n <= 0 || x.n % x.low_fold != 0) {
    issues |= MixedArgumentBadLowFold;
  }
  if (x.two_plane &&
      (x.high_fold <= 0 || x.n <= 0 || x.n % x.high_fold != 0)) {
    issues |= MixedArgumentBadHighFold;
  }
  if (x.two_plane && x.low_fold != x.high_fold && !x.dB2_valid) {
    issues |= MixedArgumentMissingHighStride;
  }

  // mixed_subbyte_l_slice converts the selected expert base back to a raw
  // byte pointer for the AIU descriptor.  Reject an outer pitch that lands
  // between bytes instead of relying on raw_pointer_cast's assertion.
  if (x.l > 1 && x.low_bits > 0 &&
      !mixed_bit_offset_byte_aligned(x.dBL, x.low_bits)) {
    issues |= MixedArgumentFractionalLowByte;
  }
  if (x.l > 1 && x.two_plane && x.high_bits > 0) {
    int64_t const high_l = x.dB2_valid ? x.dB2L : x.dBL;
    if (!mixed_bit_offset_byte_aligned(high_l, x.high_bits)) {
      issues |= MixedArgumentFractionalHighByte;
    }
  }

  if (x.interleave > 1) {
    int64_t const low_k = x.k * x.low_fold;
    if (low_k <= 0 || low_k % x.interleave != 0) {
      issues |= MixedArgumentBadLowInterleave;
    }
    if ((x.n * x.k * x.low_bits) % 8 != 0) {
      issues |= MixedArgumentFractionalLowByte;
    }
    int64_t const canonical_l = x.l > 1 ? x.n * x.k : 0;
    if (x.dB0 != low_k || x.dB1 != 1 || x.dBL != canonical_l) {
      issues |= MixedArgumentBadLowStride;
    }
    if (x.two_plane) {
      int64_t const high_k = x.k * x.high_fold;
      if (high_k <= 0 || high_k % x.interleave != 0) {
        issues |= MixedArgumentBadHighInterleave;
      }
      if ((x.n * x.k * x.high_bits) % 8 != 0) {
        issues |= MixedArgumentFractionalHighByte;
      }
      // A false dB2_valid is legal only when both folds coincide, in which
      // case dB is the canonical marker for both planes.
      if (x.dB2_valid &&
          (x.dB20 != high_k || x.dB21 != 1 || x.dB2L != canonical_l)) {
        issues |= MixedArgumentBadHighStride;
      }
    }
  }

  if (x.packed_scale) {
    if (x.ptr_Z_nonnull) {
      issues |= MixedArgumentPackedZeroPointer;
    }
    int64_t const canonical_scale_l = x.l > 1 ? x.n * scale_k : 0;
    if (x.dS0 != 1 || x.dS1 != x.n || x.dSL != canonical_scale_l) {
      issues |= MixedArgumentPackedScaleStride;
    }
    if (x.group_size <= 0 || x.k % x.group_size != 0) {
      issues |= MixedArgumentPackedGroupTail;
    } else if (x.scale_tile_k <= 0 || scale_k % x.scale_tile_k != 0) {
      issues |= MixedArgumentPackedTileTail;
    } else if (x.packed_tiles_per_unit <= 0 ||
               (scale_k / x.scale_tile_k) % x.packed_tiles_per_unit != 0) {
      issues |= MixedArgumentPackedUnitTail;
    }
  }
  return issues;
}

CUTE_HOST_DEVICE constexpr bool mixed_arguments_supported(
    MixedArgumentContract const& x) {
  return mixed_argument_issues(x) == MixedArgumentOk;
}

// Resolve the outer A base from the stride the caller actually supplied.  The
// ragged ABI flattens expert rows and publishes their cumulative row offsets;
// the uniform ABI retains the rank-3 tensor's L pitch.  Multiplying either by
// the logical K extent silently substitutes a compact layout for dA.
template <class Element, class Stride>
CUTE_HOST_DEVICE Element const* mixed_a_expert_base(
    Element const* base, Stride const& dA, int const* group_row_offsets,
    int l_coord) {
  int64_t const element_offset = group_row_offsets
      ? int64_t(group_row_offsets[l_coord]) * int64_t(cute::get<0>(dA))
      : int64_t(l_coord) * int64_t(cute::get<2>(dA));
  return base + element_offset;
}

// Metadata is indexed in logical output columns even when the resident B
// artifact folds several logical N columns into one physical row.  Using
// size<0>(gB) charges the physical TileN/Fold and can admit OOB metadata on an
// N residue.
CUTE_HOST_DEVICE constexpr int64_t mixed_logical_n_residue(
    int64_t N, int logical_tile_n, int n_coord) {
  return N - int64_t(logical_tile_n) * int64_t(n_coord);
}

// Slice the outer L coordinate while the pointer still has sub-byte element
// semantics, then hand the byte-aligned expert base to the AIU mix tensor.
// StrideB is expressed in logical elements. Calling make_gmem_ptr on an
// already-typed int4/int2 pointer selects CuTe's generic Iterator overload;
// raw C++ pointer arithmetic would then count every sub-byte value as a byte.
// The explicit Element overload constructs subbyte_iterator first. The raw
// cast deliberately happens after the L slice because the AIU mix iterator
// consumes a byte-aligned resident-artifact base.
template <class Element, class Pointer, class Shape, class Stride>
CUTE_HOST_DEVICE auto mixed_subbyte_l_slice(
    Pointer base, Shape const& shape, Stride const& stride, int l_coord) {
  auto logical_nkl = cute::make_tensor(
      cute::make_gmem_ptr<Element>(static_cast<void const*>(base)),
      shape, stride);
  auto logical_nk = logical_nkl(cute::_, cute::_, l_coord);
  return cute::make_tensor(
      cute::make_gmem_ptr(cute::raw_pointer_cast(logical_nk.data())),
      logical_nk.layout());
}

// Interleaved AIU descriptors take a raw byte base, but their inner shape and
// strides are still expressed in logical sub-byte codes.  Do not manufacture
// one rank-3 tensor whose inner modes are codes and whose L mode is bytes: it
// happens to work while callers only extract raw_pointer_cast after slicing L,
// but generic tensor indexing would silently interpret all three strides in
// one unit.  Select the expert in bytes first, then build a pure logical inner
// layout beside that base.
template <class Pointer>
CUTE_HOST_DEVICE uint8_t const* mixed_packed_byte_expert_base(
    Pointer base, int64_t bytes_per_expert, int l_coord) {
  return reinterpret_cast<uint8_t const*>(base) +
         int64_t(l_coord) * bytes_per_expert;
}

}  // namespace cutlass::gemm::collective::detail
