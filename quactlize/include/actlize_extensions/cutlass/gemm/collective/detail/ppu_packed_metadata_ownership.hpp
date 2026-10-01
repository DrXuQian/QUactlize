#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"
#include "cute/algorithm/copy.hpp"
#include "cute/tensor.hpp"

namespace cutlass::gemm::collective::detail {

// A packed metadata unit belongs to one logical N column. The copy layout and
// decoder derive ownership from both the N tile and the threads that actually
// exist in the CTA. TileN may exceed the CTA thread count (TM8/WN64 is the
// first shipping candidate with that shape), so one-thread-per-column is not a
// general contract.
template <int TileN, int CtaThreads>
struct PackedMetadataColumnOwnership {
  static_assert(TileN > 0 && CtaThreads > 0,
                "packed metadata ownership requires positive extents");

  static constexpr int owner_threads = TileN < CtaThreads ? TileN : CtaThreads;
  static_assert(TileN % owner_threads == 0,
                "packed metadata columns must divide among the available owners");
  static constexpr int columns_per_thread = TileN / owner_threads;

  CUTE_HOST_DEVICE static constexpr bool owns_physical_thread(int thread_idx) {
    return thread_idx >= 0 && thread_idx < owner_threads;
  }
  CUTE_HOST_DEVICE static constexpr int copy_owner(int thread_idx) {
    return thread_idx % owner_threads;
  }
  CUTE_HOST_DEVICE static constexpr int column(int owner, int subcolumn) {
    // make_tiled_copy lays the thread mode inside the repeated value tile:
    // with 32 owners and two columns, owner t receives t and t+32 (not
    // 2t and 2t+1). Keep this formula beside the copy-shape derivation.
    return owner + subcolumn * owner_threads;
  }
};

// Preserve the established one-column code path exactly. Only a CTA with
// fewer threads than metadata columns needs atom-granular N predicates: its
// thread slice contains several column units and one predicate cannot describe
// an N residue between them.
template <int ColumnsPerThread, class TiledCopy, class SrcTensor,
          class DstTensor, class CoordTensor>
CUTLASS_DEVICE void copy_packed_metadata_if(
    TiledCopy const& tiled_copy, bool one_column_valid, int64_t residue_n,
    SrcTensor const& src, DstTensor dst, CoordTensor const& coords) {
  static_assert(ColumnsPerThread > 0,
                "packed metadata copy must own at least one column");
  if constexpr (ColumnsPerThread == 1) {
    if (one_column_valid) {
      cute::copy(tiled_copy, src, dst);
    }
  } else {
    auto coord_vm = cute::group_modes<1, -1>(coords);
    auto pred = [&] (auto... i) {
      return int64_t(cute::get<0>(coord_vm(0, i...))) < residue_n;
    };
    cute::copy_if(tiled_copy, pred, src, dst);
  }
}

}  // namespace cutlass::gemm::collective::detail
