/***************************************************************************************************
 * THE A SMEM TILE, SHARED BY EVERY MIXED-INPUT COLLECTIVE.
 *
 * WHY THIS EXISTS. Compact A shrinks the A tile from TileM rows to a capacity, and A is fp16 activations --
 * identical for every weight format, bit width and plane count. Nothing about it depends on B. Yet it landed in
 * exactly one of the three collectives, because each of them stages A itself:
 *
 *     quactlize_mma_mixed_input.hpp        compact + ordinary
 *     ppu_mma_aiu_fold.hpp                 ordinary only, `static constexpr int compact_a_rows = 0;`
 *     ppu_mma_aiu_mixed_input_2plane.hpp   ordinary only, same
 *
 * and all three spelled the ordinary layout with the SAME four tokens. ppu_mixed_policy.hpp declares
 * AiuAProvider / PackedRowAProvider / CompactAProvider<Rows> and selects among them, which reads like an
 * abstraction and is not one: all three are EMPTY TAG STRUCTS, a witness for the descriptor and nothing else. So
 * an A-side feature has to be written three times, and a reader following the AProvider chain would reasonably
 * believe otherwise.
 *
 * WHAT IS HERE IS THE TYPE HALF ONLY, and deliberately: the layouts are a pure function of (TileShape, Stages,
 * the atom) with no reference to B, so sharing them is behaviour-preserving and every existing build exercises it
 * on the first compile. The RUNTIME half -- the residue clamp and the plain cp.async copy that replaces the AIU
 * bulk load when a capacity is set -- still lives in each collective's own load path, because those paths differ
 * (the fold presents B as (N/F, F*K), the two-plane carries a second plane). Porting that is what actually makes
 * compact A reachable for the other two formats; this header is the part that does not need a compile loop.
 *
 * WHY THE PADDING ROWS ALIAS RATHER THAN BEING PREDICATED. The MMA reads TileM logical rows whatever M is. The
 * hierarchical M mode has logical size TileM and physical strides (TileK, 0), so rows [0,Capacity) are distinct
 * and every later row aliases modulo that prefix: cosize is Capacity*TileK*Stages, and all logical padding rows
 * still address valid shared memory. At runtime only min(M-residue, Capacity) rows are copied and the remaining
 * physical rows are zeroed.
 **************************************************************************************************/
#pragma once

#include "cute/layout.hpp"
#include "cute/tensor.hpp"

namespace quactlize::collective::detail {

using namespace cute;

// The A tile's smem layout for a given capacity. Capacity 0 is the ordinary TileM-row tile; positive values are
// the compact prefix. Both spellings live here so a collective names one type and cannot drift from the others.
template <class TileShape, int Stages, class SmemLayoutAtomA, int Capacity>
struct CompactASmem {
  static_assert(Capacity >= 0, "compact A row capacity cannot be negative");
  static constexpr int kTileM = int(shape<0>(TileShape{}));
  static constexpr int kTileK = int(shape<2>(TileShape{}));
  static_assert(Capacity == 0 || Capacity <= kTileM, "compact A row capacity cannot exceed TileM");

  static constexpr int kStorageRows = Capacity > 0 ? Capacity : kTileM;
  static_assert(kTileM % kStorageRows == 0,
                "positive compact A row capacity must divide TileM so padding rows alias exactly");

  using Ordinary = decltype(tile_to_shape(
      SmemLayoutAtomA{}, make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<Stages>{})));

  using Compact = decltype(make_layout(
      make_shape(make_shape(Int<kStorageRows>{}, Int<kTileM / kStorageRows>{}),
                 shape<2>(TileShape{}), Int<Stages>{}),
      make_stride(make_stride(shape<2>(TileShape{}), _0{}),
                  _1{}, Int<kStorageRows * kTileK>{})));

  using Layout = cute::conditional_t<(Capacity > 0), Compact, Ordinary>;

  // A compact twin, NEVER ALLOCATED, used only to shape the mma fragment: partition_fragment_A on the stride-0
  // layout would inherit the zero and allocate fewer registers than the mma reads.
  using FragLayout = Ordinary;
};

}  // namespace quactlize::collective::detail
