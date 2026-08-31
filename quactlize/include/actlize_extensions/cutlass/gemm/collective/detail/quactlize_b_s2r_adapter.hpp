#pragma once

#include <type_traits>

#include "cute/atom/copy_atom.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"

#include "actlize_extensions/cutlass/gemm/collective/detail/quactlize_b_delivery_policy.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/quactlize_q4_n16k64_delivery.hpp"

// The S2R half of B delivery is deliberately a sequence of factories rather
// than one aggregate containing an owning register Tensor and views into it.
// A CuTe owning fragment may be moved when an aggregate is returned by value;
// any pointer-engine Tensor stored next to it would then retain the old
// address.  These adapters instead require the owner to be a named lvalue
// before a copy view or converter destination can be formed.

namespace cutlass::gemm::collective::detail::quactlize_b_s2r {

namespace bd =
    cutlass::gemm::collective::detail::quactlize_b_delivery;
namespace direct =
    cutlass::gemm::collective::detail::quactlize_q4_n16k64_delivery;

// Type-preserving adapter for the shipping TSM-swizzle reader.  Every method
// is the exact expression used by quactlize_mma_mixed_input.hpp today.  The
// adapter is production-adjacent but the collective intentionally has not
// been rewritten to call it yet: changing that call site requires a PPU
// codegen equality closure, not merely a host type proof.
template <class SmemCopyAtom_>
struct LegacyTsmSwizzleReader {
  using S2R = bd::TsmSwizzleS2R;
  using SmemCopyAtom = SmemCopyAtom_;

  template <class SharedTensor>
  CUTE_HOST_DEVICE static constexpr auto
  make_shared_source(SharedTensor const& shared) {
    return cute::make_mix_tensor_like(shared);
  }

  template <class LoadMma>
  CUTE_HOST_DEVICE static constexpr auto
  make_tiled_copy(LoadMma const& load_mma) {
    return cute::make_tiled_copy_B(SmemCopyAtom{}, load_mma);
  }

  template <class LoadMma, class SharedStage, class Thread>
  CUTE_HOST_DEVICE static constexpr auto
  make_register_owner(LoadMma const& load_mma,
                      SharedStage const& shared_stage,
                      Thread const& thread) {
    return load_mma.get_thread_slice(thread).partition_fragment_B(
        shared_stage);
  }

  template <class TiledCopy, class SharedSource, class Thread>
  CUTE_HOST_DEVICE static constexpr auto
  make_source_partition(TiledCopy const& tiled_copy,
                        SharedSource const& shared_source,
                        Thread const& thread) {
    return tiled_copy.get_thread_slice(thread).partition_S(shared_source);
  }

  // Owner is intentionally Owner&, never a forwarding reference.  An
  // unnamed temporary therefore cannot escape with an alias into moved rmem.
  template <class TiledCopy, class Owner, class Thread>
  CUTE_HOST_DEVICE static constexpr auto
  make_copy_view(TiledCopy const& tiled_copy, Owner& owner,
                 Thread const& thread) {
    return tiled_copy.get_thread_slice(thread).retile_D(owner);
  }
};

// Universal uint128 reader for the proposed Q4 N16xK64 plain-shared ABI.
//
// SharedSourceLayout is an atomized view of PhysicalShared::Layout:
//
//   ((local_n16_word, local_k16_row), n16_cohort, k64_block)
//
// The address map remains exactly [K/16][2*N] row-major.  Atomizing the modes
// does not alter resident bytes.  The local-N word is deliberately the
// stride-1 submode: UniversalCopy<uint128_t> assigns its four values to four
// adjacent u32 words.  Reversing these two submodes still covers the stage but
// silently makes one vector gather four K rows instead of one 16-byte row.
template <int TileN, int WarpN, int LogicalK>
struct Q4N16K64UniversalReader {
  static_assert(TileN > 0 && TileN % WarpN == 0,
                "Q4 Universal S2R needs WarpN to tile CTA TileN");
  static_assert(WarpN == 16 || WarpN == 32 || WarpN == 64,
                "Q4 Universal S2R admits WN16/WN32/WN64");
  static_assert(LogicalK > 0 && LogicalK % direct::kLogicalKAtom == 0,
                "Q4 Universal S2R needs whole K64 blocks");

  using S2R = bd::UniversalS2R;
  // Physical owns the complete CTA stage.  Its row pitch is 2*TileN u32,
  // never 2*WarpN.  Conflating these is silent whenever TileN==WarpN and
  // reads the next physical K row from another N cohort for TN>WN.
  using Physical = direct::PhysicalShared<TileN, LogicalK>;
  using Reader = direct::UniversalReader<TileN, LogicalK>;
  using Element = typename Physical::Element;
  using TiledCopy = typename Reader::Copy;

  static constexpr int n_cohorts = WarpN / direct::kNAtom;
  static constexpr int k_blocks = LogicalK / direct::kLogicalKAtom;
  static constexpr int warp_n_tiles = TileN / WarpN;
  static constexpr int reader_threads = direct::kReaderThreads;
  static constexpr int n16_source_words =
      direct::kWordsPerNPerPhysicalRow * direct::kNAtom;
  // TiledMMA's N cohorts are warp-interleaved: for TN64/WN32, warp 0
  // owns N16 cohorts 0/2 and warp 1 owns 1/3.  Consequently a semantic
  // warp-N coordinate advances by one N16 cohort, while successive local
  // cohorts advance over every other cohort.  Treating either distance as
  // one contiguous WN tile is accidentally correct only for WN16/WN64.
  static constexpr int warp_n_base_words = n16_source_words;
  static constexpr int n_cohort_stride_words =
      n16_source_words * warp_n_tiles;
  static constexpr int k_atoms_per_copy =
      direct::kLogicalKAtom / 16;

  using SharedSourceShape = cute::Shape<
      cute::Shape<cute::_32, cute::_4>,
      cute::Int<n_cohorts>, cute::Int<k_blocks>>;
  using SharedSourceStride = cute::Stride<
      cute::Stride<cute::_1, cute::Int<Physical::physical_n_words>>,
      cute::Int<n_cohort_stride_words>,
      cute::Int<4 * Physical::physical_n_words>>;
  using SharedSourceLayout =
      cute::Layout<SharedSourceShape, SharedSourceStride>;

  static_assert(k_atoms_per_copy == 4,
                "one N16xK64 Q4 read feeds four logical K16 MMA atoms");
  static_assert(cute::size(TiledCopy{}) == reader_threads,
                "Q4 Universal S2R is one 32-lane warp-local copy");
  static_assert(cute::size(SharedSourceLayout{}) * warp_n_tiles ==
                    Physical::physical_words,
                "all warp-N sources must exactly cover one physical stage");
  static_assert(cute::cosize_v<SharedSourceLayout> +
                        (warp_n_tiles - 1) * warp_n_base_words ==
                    Physical::physical_words,
                "last warp-N source must end at the full CTA-stage boundary");

  // `warp_n_tile` is the semantic N coordinate in [0,warp_n_tiles), not a
  // physical warp id.  A CTA with warps on M and N must derive this coordinate
  // from the TiledMMA warp layout before calling the reader.
  template <class Pointer, class WarpNTile>
  CUTE_HOST_DEVICE static constexpr auto
  make_shared_source(Pointer const& stage_pointer,
                     WarpNTile const& warp_n_tile) {
    return cute::make_tensor(
        stage_pointer + warp_n_tile * cute::Int<warp_n_base_words>{},
        SharedSourceLayout{});
  }

  CUTE_HOST_DEVICE static constexpr TiledCopy make_tiled_copy() {
    return {};
  }

  // `lane` is warp-local in [0,31], not the CTA thread index.  The Universal
  // TiledCopy has exactly 32 thread slots; callers with multiple warps must
  // select the warp's N tile separately and pass thread_idx % 32 here.
  template <class SharedSource, class Lane>
  CUTE_HOST_DEVICE static constexpr auto
  make_source_partition(SharedSource const& shared_source,
                        Lane const& lane) {
    return TiledCopy{}.get_thread_slice(lane).partition_S(shared_source);
  }

  template <class SourcePartition>
  CUTE_HOST_DEVICE static constexpr auto
  make_copy_source_view(SourcePartition const& source_partition) {
    CUTE_STATIC_ASSERT_V(cute::rank(source_partition) == cute::Int<4>{});
    CUTE_STATIC_ASSERT_V(cute::size<1>(source_partition) == cute::Int<1>{});
    return source_partition(cute::_, 0, cute::_, cute::_);
  }

  template <class SourcePartition>
  CUTE_HOST_DEVICE static constexpr auto
  make_register_owner(SourcePartition const& source_partition) {
    return cute::make_fragment_like<Element>(source_partition);
  }

  // See the class comment: this lvalue-only API is the lifetime boundary.
  template <class Owner, class Lane>
  CUTE_HOST_DEVICE static constexpr auto
  make_copy_view(Owner& owner, Lane const& lane) {
    auto raw = TiledCopy{}.get_thread_slice(lane).retile_D(owner);
    // make_tiled_copy retains a unit tiler-rest mode between the CopyAtom
    // value mode and the source tensor's N/K rest modes.  Normalize that
    // implementation detail here so collective code sees the same semantic
    // (CPY, CPY_N, CPY_K) interface as the shipping reader.
    CUTE_STATIC_ASSERT_V(cute::rank(raw) == cute::Int<4>{});
    CUTE_STATIC_ASSERT_V(cute::size<1>(raw) == cute::Int<1>{});
    return raw(cute::_, 0, cute::_, cute::_);
  }

  template <class CopyView, class KBlock>
  CUTE_HOST_DEVICE static constexpr auto
  make_converter_input(CopyView const& copy_view, KBlock const& k_block) {
    return cute::recast<cutlass::int4b_t>(
        copy_view(cute::_, cute::_, k_block));
  }

  // Converter destination placement is rooted in the logical MMA owner.  It
  // borrows mode-0 shape/order from the physical converter input, but derives
  // every rest-mode stride from the compute fragment.  This is the L231 rule;
  // reusing the source rest stride is a known deterministic corruption.
  template <class LogicalMmaOwner, class ConverterInput, class KBlock,
            int KAtomsPerCopy>
  CUTE_HOST_DEVICE static constexpr auto
  make_converter_destination(LogicalMmaOwner& logical_mma_owner,
                             ConverterInput const& converter_input,
                             KBlock const& k_block,
                             cute::Int<KAtomsPerCopy>) {
    static_assert(KAtomsPerCopy == k_atoms_per_copy,
                  "Q4 N16xK64 converter must advance four K16 MMA atoms");
    auto dst_n_stride = cute::compact_col_major(
        cute::shape<1>(converter_input.layout()),
        cute::stride<1>(logical_mma_owner.layout()));
    auto dst_layout = cute::make_layout(
        cute::shape(converter_input.layout()),
        cute::make_stride(cute::stride<0>(converter_input.layout()),
                          dst_n_stride));
    auto destination = cute::make_tensor(
        logical_mma_owner(cute::_, cute::_,
                          k_block * cute::Int<KAtomsPerCopy>{}).data(),
        dst_layout);
    CUTE_STATIC_ASSERT_V(cute::size(destination) ==
                         cute::size(converter_input));
    return destination;
  }
};

}  // namespace cutlass::gemm::collective::detail::quactlize_b_s2r
