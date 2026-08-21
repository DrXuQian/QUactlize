/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

// Device harnesses use this as a stale-submodule gate. Version 3 is the materialized CuTe fragment construction;
// keep the marker beside the shared implementation so it cannot describe one collective while another drifts.
#define PPU_SCALE_FRAGMENT_API 3

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/numeric/numeric_types.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/copy.hpp"
#include "cute/algorithm/tensor_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"

namespace cutlass::gemm::collective::detail {

// One construction for the caller-declared fp16 metadata layout. Production
// load_init in every shipping mixed-input collective and the host oracles call
// this exact function.  The stride is part of the ABI: accepting dS in
// Arguments and then rebuilding a compact layout here was a silent parameter
// substitution, not an optimization.  L127 varies dS while holding pointer,
// shape and payload fixed and anchors both address maps to an independent
// int64 formula.
inline constexpr int kStridedMetadataTileApi = 2;

template <class Stride>
CUTE_HOST_DEVICE constexpr Stride lower_metadata_stride(Stride const& dS) {
  return dS;
}

template <class ScaleTileShape, class Element, class Stride>
CUTE_HOST_DEVICE auto make_metadata_tile(
    Element const* base, Stride const& dS, int N, int64_t scale_k, int L,
    int l_coord, int n_coord) {
  auto metadata_nkl = cute::make_tensor(
      cute::make_gmem_ptr(base), cute::make_shape(N, scale_k, L), dS);
  auto metadata_nk = metadata_nkl(cute::_, cute::_, l_coord);
  return cute::local_tile(
      metadata_nk, ScaleTileShape{}, cute::make_coord(n_coord, cute::_));
}

// CuTe neither diagnoses a tiled-copy thread layout larger than the CTA nor wraps get_slice(t) when t exceeds the
// logical layout. Callers wrap the physical thread index: slots > CTA would leave metadata behind, while slots < CTA
// deliberately makes surplus threads repeat the same source/destination transfers. Keep the explicit layout
// parameters in this witness: the negative gate instantiates the old uncapped (16,8)x8 layout and must fail here,
// while the production plan proves that its slots fit, use whole atoms and still cover the complete tile.
template <int TileN, int TileK, int CtaThreads, int ThreadLayoutH, int ValuesPerThread,
          int AtomValues = 8>
struct ScaleCopyCoverage {
  static_assert(TileN > 0 && TileK > 0 && CtaThreads > 0,
                "scale copy needs positive tile and CTA extents");
  static_assert(ThreadLayoutH > 0 && ValuesPerThread > 0 && AtomValues > 0,
                "scale copy needs positive thread/value extents");
  static constexpr int thread_layout_w = TileK;
  static constexpr int thread_slots = ThreadLayoutH * thread_layout_w;
  static constexpr int tile_values = TileN * TileK;
  static constexpr int covered_values = thread_slots * ValuesPerThread;
  static constexpr bool within_cta = thread_slots <= CtaThreads;
  static constexpr bool atom_aligned = ValuesPerThread % AtomValues == 0;
  static constexpr bool full_tile = covered_values == tile_values;
  static constexpr bool value = within_cta && atom_aligned && full_tile;
  static_assert(within_cta,
                "scale copy asks for more thread slots than the CTA has -- the modulo slice would silently truncate it");
  static_assert(atom_aligned,
                "scale copy values per thread must contain complete copy atoms");
  static_assert(full_tile,
                "capped scale copy must cover every value in the metadata tile");
};

// Cap the N-direction thread extent and let each remaining thread issue more fixed-width atoms. ValuesPerThread can
// therefore exceed AtomValues without widening the instruction: CuTe represents the extra values as a rest mode and
// copy() iterates it. The two-plane collective shipped this exact construction first; all three collectives now bind
// this one plan so host tactic generation no longer needs to reject otherwise legal large-WarpN shapes.
template <int TileN, int TileK, int CtaThreads, int AtomValues = 8>
struct ScaleCopyPlan {
  static_assert(TileN > 0 && TileK > 0 && CtaThreads > 0,
                "scale copy needs positive tile and CTA extents");
  static_assert(TileN % AtomValues == 0,
                "scale copy TileN must contain a whole number of copy atoms");
  static_assert(CtaThreads / TileK >= 1,
                "scale copy needs at least one CTA thread per metadata K group");

  static constexpr int thread_layout_h_uncapped = TileN / AtomValues;
  static constexpr int thread_layout_h =
      thread_layout_h_uncapped < CtaThreads / TileK ? thread_layout_h_uncapped : CtaThreads / TileK;
  static_assert(TileN % (thread_layout_h * AtomValues) == 0,
                "scale copy TileN must split into capped threads of complete copy atoms");
  static constexpr int thread_layout_w = TileK;
  static constexpr int values_per_thread = TileN / thread_layout_h;
  static constexpr int thread_slots = thread_layout_h * thread_layout_w;
  static constexpr int tile_n = TileN;
  static constexpr int tile_k = TileK;
  static constexpr int cta_threads = CtaThreads;

  // CuTe's tiled copy is indexed by logical copy slot, while the kernel is
  // launched with every physical CTA thread.  Surplus physical threads are
  // consumers only: issuing the same asynchronous shared-memory write from
  // each of them is neither required for coverage nor part of the copy
  // contract.  Keep the ownership and slot mapping beside the coverage proof
  // so a collective cannot silently reintroduce modulo-replayed publishers.
  CUTE_HOST_DEVICE static constexpr bool owns_physical_thread(int thread) {
    return thread >= 0 && thread < thread_slots;
  }
  CUTE_HOST_DEVICE static constexpr int logical_slot(int thread) {
    return thread % thread_slots;
  }
  using Coverage = ScaleCopyCoverage<TileN, TileK, CtaThreads, thread_layout_h,
                                     values_per_thread, AtomValues>;
  static_assert(Coverage::value,
                "capped scale-copy plan must fit the CTA and cover the complete metadata tile");
};

// One source of truth for the two values formerly spelled as `% Per` and `/ Per` at every COARSE/FINE call site.
template <int Per, int NGroups>
struct ScaleSplit {
  static_assert(Per > 0 && NGroups > 0, "metadata groups must have positive extents");
  using InGroupL = cute::Layout<cute::Shape <cute::Int<Per>, cute::Int<NGroups>>,
                                cute::Stride<cute::_1,       cute::_0>>;
  using GroupL   = cute::Layout<cute::Shape <cute::Int<Per>, cute::Int<NGroups>>,
                                cute::Stride<cute::_0,       cute::Int<1>>>;

  CUTE_HOST_DEVICE static constexpr int in_group(int i) { return int(InGroupL{}(i)); }
  CUTE_HOST_DEVICE static constexpr int group(int i)    { return int(GroupL{}(i)); }
};

// COARSE means one metadata group covers one or more complete B-copy steps. The policy owns both the predicate and
// the divisibility invariant so a collective cannot select COARSE with a different boundary/index calculation.
template <int ScaleGroups, int CopySteps>
struct CoarseScalePolicy {
  static_assert(ScaleGroups > 0 && CopySteps > 0, "metadata and B-copy extents must be positive");
  static constexpr bool active = ScaleGroups <= CopySteps;
  static_assert(!active || CopySteps % ScaleGroups == 0,
                "COARSE scale groups must evenly partition the B copy steps");
  static constexpr int steps_per_group = active ? CopySteps / ScaleGroups : 1;
  using Split = ScaleSplit<steps_per_group, ScaleGroups>;

  CUTE_HOST_DEVICE static constexpr bool starts_group(int copy_step) {
    return Split::in_group(copy_step) == 0;
  }
  CUTE_HOST_DEVICE static constexpr int group(int copy_step) {
    return Split::group(copy_step);
  }
};

// FINE means a B-copy step crosses metadata groups, so metadata is reloaded at MMA-atom boundaries. This is the
// exact complement of COARSE for a particular copy view; MmaAtoms supplies the group-to-atom mapping.
template <int ScaleGroups, int CopySteps, int MmaAtoms>
struct FineScalePolicy {
  using Coarse = CoarseScalePolicy<ScaleGroups, CopySteps>;
  static constexpr bool active = !Coarse::active;
  static_assert(!active || (MmaAtoms >= ScaleGroups && MmaAtoms % ScaleGroups == 0),
                "FINE scale groups must evenly partition the MMA K atoms");
  static constexpr int atoms_per_group = active ? MmaAtoms / ScaleGroups : 1;
  using Split = ScaleSplit<atoms_per_group, ScaleGroups>;

  CUTE_HOST_DEVICE static constexpr bool starts_group(int atom) {
    return Split::in_group(atom) == 0;
  }
  CUTE_HOST_DEVICE static constexpr int group(int atom) {
    return Split::group(atom);
  }
};

// Fragment construction is intentionally the plain CuTe idiom. Callers pass an already rank-2 scale tile; keeping
// the slice outside this helper lets flat and (group,stage) shared-memory views use the same implementation.
template <class ElementScale>
struct ScaleFragment {
  template <class TiledMma, class STensor>
  CUTLASS_DEVICE static auto make(TiledMma const& thr_mma, STensor const& scale_tile) {
    return cute::make_fragment_like<ElementScale>(thr_mma.partition_fragment_B(scale_tile));
  }

  template <class TiledMma, class STensor>
  CUTE_HOST_DEVICE static constexpr auto layout(TiledMma const& thr_mma, STensor const& scale_tile) {
    return cute::make_layout_like(thr_mma.partition_fragment_B(scale_tile).layout());
  }
};

// The ordinary and folded collectives flatten (group,stage); the two-plane collective keeps both coordinates. These
// are storage policies only. Group boundaries and the published register-fragment tuple contract remain common.
struct FlatMetadataAddress {
  template <int ScaleGroups, class Tensor>
  CUTLASS_DEVICE static auto source(Tensor const& tensor, int group, int stage) {
    return tensor(cute::_, cute::_, 0, stage * ScaleGroups + group);
  }
};

struct SplitMetadataAddress {
  template <int ScaleGroups, class Tensor>
  CUTLASS_DEVICE static auto source(Tensor const& tensor, int group, int stage) {
    return tensor(cute::_, cute::_, 0, group, stage);
  }
};

// All three collectives publish (scale source, scale fragment [, zero source, zero fragment]) and retile that as
// (copy atom, scale destination [, zero destination]). Centralizing those positional contracts makes reload one rule.
template <class Address, int ScaleGroups, bool HasZero, class Info, class Views>
CUTLASS_DEVICE void reload_metadata(Info const& info, Views const& views, int group, int stage) {
  auto smem_tiled_copy = cute::get<0>(views);
  auto scale_src       = cute::get<0>(info);
  auto scale_dst       = cute::get<1>(views);
  cute::copy(smem_tiled_copy,
             Address::template source<ScaleGroups>(scale_src, group, stage),
             scale_dst(cute::_, cute::_, 0));
  if constexpr (HasZero) {
    auto zero_src = cute::get<2>(info);
    auto zero_dst = cute::get<2>(views);
    cute::copy(smem_tiled_copy,
               Address::template source<ScaleGroups>(zero_src, group, stage),
               zero_dst(cute::_, cute::_, 0));
  }
}

template <bool HasZero, class BSlice, class Info>
CUTLASS_DEVICE void apply_metadata(BSlice&& b_slice, Info const& info) {
  cute::transform(b_slice, cute::get<1>(info)(cute::_, cute::_, 0), b_slice, cute::multiplies{});
  if constexpr (HasZero) {
    cute::transform(b_slice, cute::get<3>(info)(cute::_, cute::_, 0), b_slice, cute::plus{});
  }
}

// Public policy seam owned by every mixed-input CollectiveMma. The collective still owns shared-memory shape and
// packed-provider details, but it cannot independently redefine fragment construction, group boundaries, reload
// addressing or ordinary scale/zero application.
template <class ElementScale, int ScaleGroups, bool HasZero, class Address>
struct MixedMetadataPolicy {
  static_assert(ScaleGroups > 0, "a mixed metadata policy needs at least one K group");
  static constexpr int scale_groups = ScaleGroups;
  static constexpr bool has_zero = HasZero;
  using AddressPolicy = Address;

  template <int TileN, int CtaThreads>
  using ScaleCopy = ScaleCopyPlan<TileN, ScaleGroups, CtaThreads>;

  template <int CopySteps>
  using Coarse = CoarseScalePolicy<ScaleGroups, CopySteps>;

  template <int CopySteps, int MmaAtoms>
  using Fine = FineScalePolicy<ScaleGroups, CopySteps, MmaAtoms>;

  template <class TiledMma, class STensor>
  CUTLASS_DEVICE static auto make_fragment(TiledMma const& thr_mma, STensor const& scale_tile) {
    return ScaleFragment<ElementScale>::make(thr_mma, scale_tile);
  }

  template <class TiledMma, class STensor>
  CUTE_HOST_DEVICE static constexpr auto fragment_layout(TiledMma const& thr_mma, STensor const& scale_tile) {
    return ScaleFragment<ElementScale>::layout(thr_mma, scale_tile);
  }

  template <class Info, class Views>
  CUTLASS_DEVICE static void reload(Info const& info, Views const& views, int group, int stage) {
    reload_metadata<Address, ScaleGroups, HasZero>(info, views, group, stage);
  }

  template <class BSlice, class Info>
  CUTLASS_DEVICE static void apply(BSlice&& b_slice, Info const& info) {
    apply_metadata<HasZero>(static_cast<BSlice&&>(b_slice), info);
  }
};

} // namespace cutlass::gemm::collective::detail
