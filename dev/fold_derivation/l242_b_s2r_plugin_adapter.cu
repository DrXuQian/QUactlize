// L242 -- type and lifetime closure for the production-adjacent B S2R
// plugin seam.
//
// The shipping TSM-swizzle reader is compared expression-for-expression with
// LegacyTsmSwizzleReader.  The unselected Q4 plain-shared reader then composes
// the real 32-lane UniversalCopy with the real PPU0010 m8 B fragment for
// WN16/WN32/WN64.  Its copy step is one N16xK64 atom and therefore feeds
// exactly four logical K16 MMA atoms for every WN.

#if defined(L242_PPU_TYPE_PROBE)

#include <type_traits>

#include "fpA_intB_ppu.cuh"
#include "actlize_extensions/cutlass/gemm/collective/detail/quactlize_b_s2r_adapter.hpp"

using namespace cute;
using L242Schedule = ppu_group_schedule::FinegrainedSchedule<32>;
using L242TileShape = Shape<_8, _64, _256>;
using L242ScaleTile = Shape<_64, _8>;
using L242Warp = Shape<_8, _16, _256>;
using L242Types = fpa_intb_ppu::DenseQ4KPack4KernelTypes<
    ppu_mixed_policy::QuantMode::FinegrainedScaleZero,
    L242Schedule, L242TileShape, L242ScaleTile, L242Warp, 2, true>;
using L242Mainloop = typename L242Types::CollectiveMainloop;

using L242ManualSmemCopyOp = PPU0010_TSM_LD_SWZL<
    cutlass::half_t, 64, 64, true, true, 1>;
using L242ManualSmemCopyAtom =
    Copy_Atom<L242ManualSmemCopyOp, cutlass::half_t>;
using L242ManualSmemLayoutAtom =
    Layout<Shape<_64, _64>, Stride<_1, _64>>;
using L242ManualSmemLayout = decltype(tile_to_shape(
    L242ManualSmemLayoutAtom{}, Shape<_64, _64, _2>{}));
static_assert(std::is_same_v<typename L242Mainloop::SmemCopyAtomB,
                             L242ManualSmemCopyAtom>);
static_assert(std::is_same_v<typename L242Mainloop::SmemLayoutB,
                             L242ManualSmemLayout>);

using L242ComputeMma = typename L242Mainloop::TiledMma;
using L242WarpOnM = decltype(
    get<1>(L242ComputeMma{}.get_thr_layout_vmnk().shape()));
using L242WarpOnN = decltype(
    get<2>(L242ComputeMma{}.get_thr_layout_vmnk().shape()));
using L242PermutationN = decltype(
    L242ComputeMma{}.template permutation_mnk<1>());
using L242ShadowPermutationM = Int<L242WarpOnM{} * 16>;
using L242RealLoadMma = TiledMMA<
    MMA_Atom<PPU0010_16x16x16_F32F16F16F32_TN>,
    Layout<Shape<L242WarpOnM, L242WarpOnN, _1>>,
    Tile<L242ShadowPermutationM, L242PermutationN, _16>>;
using L242ManualLoadMma = TiledMMA<
    MMA_Atom<PPU0010_16x16x16_F32F16F16F32_TN>,
    Layout<Shape<_1, _4, _1>>, Tile<_16, _64, _16>>;
static_assert(std::is_same_v<L242RealLoadMma, L242ManualLoadMma>);

using L242LegacyAdapter =
    cutlass::gemm::collective::detail::quactlize_b_s2r::
        LegacyTsmSwizzleReader<typename L242Mainloop::SmemCopyAtomB>;
using L242RealTiledCopy = decltype(make_tiled_copy_B(
    typename L242Mainloop::SmemCopyAtomB{}, L242RealLoadMma{}));
using L242AdapterTiledCopy = decltype(
    L242LegacyAdapter::make_tiled_copy(L242RealLoadMma{}));
static_assert(std::is_same_v<L242RealTiledCopy, L242AdapterTiledCopy>);

using L242PpuReader =
    cutlass::gemm::collective::detail::quactlize_b_s2r::
        Q4N16K64UniversalReader<64, 64, 256>;
static_assert(L242PpuReader::n_cohorts == 4);
static_assert(L242PpuReader::k_blocks == 4);
static_assert(L242PpuReader::k_atoms_per_copy == 4);

int main() { return 0; }

#elif defined(L242_COMPILER_PROBE)

#include <cuda_fp16.h>

__global__ void l242_compiler_probe(half const* x, half* y) {
  int const i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i == 0) y[0] = __hadd(x[0], x[0]);
}

int main() { return 0; }

#else

#include <array>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "cute/atom/mma_traits_ppu0010.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/quactlize_b_s2r_adapter.hpp"

namespace {
using namespace cute;
namespace s2r =
    cutlass::gemm::collective::detail::quactlize_b_s2r;

using WarpOnM = _1;
using WarpOnN = _4;
using PermutationN = _64;
using LegacyLoadMma = TiledMMA<
    MMA_Atom<PPU0010_16x16x16_F32F16F16F32_TN>,
    Layout<Shape<WarpOnM, WarpOnN, _1>>,
    Tile<_16, PermutationN, _16>>;
using LegacySmemCopyOp = PPU0010_TSM_LD_SWZL<
    cutlass::half_t, 64, 64, true, true, 1>;
using LegacySmemCopyAtom = Copy_Atom<LegacySmemCopyOp, cutlass::half_t>;
using LegacySmemLayoutAtom =
    Layout<Shape<_64, _64>, Stride<_1, _64>>;
using LegacySmemLayout = decltype(tile_to_shape(
    LegacySmemLayoutAtom{}, Shape<_64, _64, _2>{}));
using LegacyAdapter = s2r::LegacyTsmSwizzleReader<
    LegacySmemCopyAtom>;

static_assert(std::is_same_v<typename LegacyAdapter::S2R,
    cutlass::gemm::collective::detail::quactlize_b_delivery::
        TsmSwizzleS2R>);

bool prove_legacy_exact_types() {
  int thread = 0;
  auto shared = make_tensor(
      make_smem_ptr((cutlass::half_t*)nullptr),
      LegacySmemLayout{});

  auto old_source = make_mix_tensor_like(shared);
  auto new_source = LegacyAdapter::make_shared_source(shared);
  static_assert(std::is_same_v<decltype(old_source), decltype(new_source)>);

  auto old_copy = make_tiled_copy_B(
      LegacySmemCopyAtom{}, LegacyLoadMma{});
  auto new_copy = LegacyAdapter::make_tiled_copy(LegacyLoadMma{});
  static_assert(std::is_same_v<decltype(old_copy), decltype(new_copy)>);

  auto old_owner = LegacyLoadMma{}.get_thread_slice(thread)
      .partition_fragment_B(shared(_, _, 0));
  auto new_owner = LegacyAdapter::make_register_owner(
      LegacyLoadMma{}, shared(_, _, 0), thread);
  static_assert(std::is_same_v<decltype(old_owner), decltype(new_owner)>);

  auto old_partition = old_copy.get_thread_slice(thread)
      .partition_S(old_source);
  auto new_partition = LegacyAdapter::make_source_partition(
      new_copy, new_source, thread);
  static_assert(std::is_same_v<decltype(old_partition),
                               decltype(new_partition)>);

  auto old_view = old_copy.get_thread_slice(thread).retile_D(old_owner);
  auto new_view = LegacyAdapter::make_copy_view(
      new_copy, new_owner, thread);
  static_assert(std::is_same_v<decltype(old_view), decltype(new_view)>);

  return int(size(old_partition)) == int(size(new_partition)) &&
         int(size(old_owner)) == int(size(new_owner)) &&
         int(size(old_view)) == int(size(new_view));
}

struct DirectMetrics {
  int source_bad = 0;
  int source_words = 0;
  int register_words = 0;
  int converter_values = 0;
  int mma_values = 0;
  int k_atoms_per_copy = 0;
  int legacy_destination_differences = 0;
  int adversarial_destination_same = 0;
  int wrong_warp_pitch_bad = 0;
  int warp_n_coordinate_bad = 0;
};

template <int WN>
DirectMetrics prove_direct_reader() {
  constexpr int TN = 64;
  using Adapter = s2r::Q4N16K64UniversalReader<TN, WN, 64>;
  using Physical = typename Adapter::Physical;
  using WarpOnN = Int<TN / WN>;
  using CtaComputeMma = TiledMMA<
      MMA_Atom<PPU0010_8x16x16_F32F16F16F32_TN>,
      Layout<Shape<_1, WarpOnN, _1>>,
      Tile<_8, Int<(TN / WN) * 16>, _64>>;
  DirectMetrics out{};

  // TiledMMA's warp layout is M-fast, then N.  The future collective must
  // pass this recovered N coordinate to make_shared_source; passing physical
  // warp_idx directly is only correct in the WARP_M==1 special case.
  auto warp_layout = CtaComputeMma{}.get_thr_layout_vmnk();
  for (int warp_n = 0; warp_n < Adapter::warp_n_tiles; ++warp_n) {
    int const physical_thread =
        int(warp_layout(make_coord(0, 0, warp_n, 0)));
    int const physical_warp = physical_thread / 32;
    int const recovered_warp_n = physical_warp % Adapter::warp_n_tiles;
    out.warp_n_coordinate_bad += physical_thread % 32 != 0;
    out.warp_n_coordinate_bad += recovered_warp_n != warp_n;
  }

  auto identity = make_identity_tensor(typename Adapter::SharedSourceShape{});
  typename Adapter::SharedSourceLayout source_layout{};
  std::array<int, Physical::physical_words> hits{};
  std::array<int, Physical::physical_words> wrong_hits{};
  for (int warp_n_tile = 0;
       warp_n_tile < Adapter::warp_n_tiles; ++warp_n_tile) {
    for (int lane = 0; lane < 32; ++lane) {
      auto partition = Adapter::make_source_partition(identity, lane);
      for (int i = 0; i < int(size(partition)); ++i) {
        auto const coordinate = partition(i);
        int const offset =
            warp_n_tile * Adapter::warp_n_base_words +
            int(source_layout(coordinate));
        if (offset < 0 || offset >= Physical::physical_words) {
          ++out.source_bad;
        } else {
          ++hits[std::size_t(offset)];
        }

        // RED: the tempting WN-sized physical row pitch.  Keep the same
        // local coordinate and warp-N base, changing only k-row stride.
        int const local_n_word = int(get<0>(get<0>(coordinate)));
        int const local_k_row = int(get<1>(get<0>(coordinate)));
        int const n_cohort = int(get<1>(coordinate));
        int const k_block = int(get<2>(coordinate));
        int const wrong =
            warp_n_tile * Adapter::warp_n_base_words +
            local_k_row * (2 * WN) + local_n_word +
            n_cohort * Adapter::n_cohort_stride_words +
            k_block * (4 * 2 * WN);
        if (wrong < 0 || wrong >= Physical::physical_words)
          ++out.wrong_warp_pitch_bad;
        else
          ++wrong_hits[std::size_t(wrong)];
        ++out.source_words;
      }
    }
  }
  for (int h : hits) out.source_bad += h != 1;
  for (int h : wrong_hits) out.wrong_warp_pitch_bad += h != 1;

  int thread = 0;
  std::array<std::uint32_t, Physical::physical_words> stage{};
  auto source = Adapter::make_shared_source(
      make_smem_ptr(stage.data()), _0{});
  auto source_partition = Adapter::make_source_partition(source, thread);
  auto register_owner = Adapter::make_register_owner(source_partition);

#if defined(L242_PLANT_RVALUE_OWNER)
  // Must not compile: a view may only be created after the owner is named.
  auto dangling = Adapter::make_copy_view(
      Adapter::make_register_owner(source_partition), thread);
  (void)dangling;
#endif

  auto copy_view = Adapter::make_copy_view(register_owner, thread);
  auto converter_input = Adapter::make_converter_input(copy_view, _0{});

  auto logical = make_tensor(
      make_smem_ptr((cutlass::half_t*)nullptr),
      make_layout(Shape<Int<TN>, _64>{}, Stride<_64, _1>{}));
  auto logical_owner = CtaComputeMma{}.get_thread_slice(thread)
      .partition_fragment_B(logical);

#if defined(L242_PLANT_K_ATOMS_TWO)
  auto destination = Adapter::make_converter_destination(
      logical_owner, converter_input, _0{}, _2{});
#else
  auto destination = Adapter::make_converter_destination(
      logical_owner, converter_input, _0{}, _4{});
#endif

  out.register_words = int(size(register_owner));
  out.converter_values = int(size(converter_input));
  out.mma_values = int(size(logical_owner));
  out.k_atoms_per_copy =
      int(size<2>(logical_owner)) / int(size<2>(copy_view));

  static_assert(int(size<2>(decltype(copy_view.layout()){})) == 1);
  static_assert(int(size<2>(decltype(logical_owner.layout()){})) == 4);
  static_assert(int(size(decltype(converter_input.layout()){})) ==
                int(size(decltype(destination.layout()){})));

  // Independence is about authority, not accidental type inequality.  The
  // direct reader's compact source and compute-owned destination happen to
  // have equal layouts in this geometry.  Perturb the source rest stride and
  // prove that the adapter still derives the identical destination type from
  // the logical MMA owner.
  auto adversarial_n_stride = compact_col_major(
      shape<1>(converter_input.layout()), Int<777>{});
  auto adversarial_layout = make_layout(
      shape(converter_input.layout()),
      make_stride(stride<0>(converter_input.layout()),
                  adversarial_n_stride));
  auto adversarial_input = make_tensor(
      converter_input.data(), adversarial_layout);
  auto adversarial_destination = Adapter::make_converter_destination(
      logical_owner, adversarial_input, _0{}, _4{});
  out.adversarial_destination_same =
      std::is_same_v<decltype(destination.layout()),
                     decltype(adversarial_destination.layout())>;

  // Exact historical RED under an adversarial physical rest stride: borrow
  // all rest strides from the physical input.  The current compact direct
  // source happens to equal the compute destination, so equality today is not
  // proof of ownership.  Perturbing only the source authority makes WN16 an
  // identity control and forces WN32/WN64 red while the candidate above stays
  // unchanged.
  auto legacy_destination = make_tensor(
      logical_owner(_, _, _0{}).data(), adversarial_input.layout());
  for (int i = 0; i < int(size(destination)); ++i)
    out.legacy_destination_differences +=
        int(destination.layout()(i)) !=
        int(legacy_destination.layout()(i));
  return out;
}

#if defined(L242_PLANT_WN8)
using InvalidWN = s2r::Q4N16K64UniversalReader<64, 8, 64>;
static_assert(InvalidWN::n_cohorts > 0);
#endif

}  // namespace

int main() {
  bool const legacy = prove_legacy_exact_types();
  auto const w16 = prove_direct_reader<16>();
  auto const w32 = prove_direct_reader<32>();
  auto const w64 = prove_direct_reader<64>();

  auto report = [] (int wn, DirectMetrics const& m) {
    std::printf(
        "L242 DIRECT wn=%d source=%d register_words=%d converter=%d "
        "mma=%d k_atoms_per_copy=%d source_bad=%d legacy_dst_diff=%d "
        "adversarial_dst_same=%d wrong_warp_pitch_bad=%d "
        "warp_n_coordinate_bad=%d\n",
        wn, m.source_words, m.register_words, m.converter_values,
        m.mma_values, m.k_atoms_per_copy, m.source_bad,
        m.legacy_destination_differences,
        m.adversarial_destination_same, m.wrong_warp_pitch_bad,
        m.warp_n_coordinate_bad);
  };
  std::printf(
      "L242 LEGACY exact_shared_source=1 exact_tiled_copy=1 "
      "exact_owner=1 exact_copy_view=1 result=%s\n",
      legacy ? "PASS" : "FAIL");
  report(16, w16);
  report(32, w32);
  report(64, w64);

  bool const direct =
      w16.source_bad == 0 && w32.source_bad == 0 && w64.source_bad == 0 &&
      w16.k_atoms_per_copy == 4 && w32.k_atoms_per_copy == 4 &&
      w64.k_atoms_per_copy == 4 &&
      w16.converter_values == w16.mma_values &&
      w32.converter_values == w32.mma_values &&
      w64.converter_values == w64.mma_values &&
      w16.adversarial_destination_same == 1 &&
      w32.adversarial_destination_same == 1 &&
      w64.adversarial_destination_same == 1 &&
      w16.legacy_destination_differences == 0 &&
      w32.legacy_destination_differences > 0 &&
      w64.legacy_destination_differences > 0 &&
      w16.wrong_warp_pitch_bad > 0 &&
      w32.wrong_warp_pitch_bad > 0 &&
      w64.wrong_warp_pitch_bad == 0 &&
      w16.warp_n_coordinate_bad == 0 &&
      w32.warp_n_coordinate_bad == 0 &&
      w64.warp_n_coordinate_bad == 0;
  std::printf(
      "L242 B_S2R_PLUGIN %s legacy=%d direct=%d owner_alias=SEPARATE "
      "source_destination=INDEPENDENT lane_scope=WARP_LOCAL reds=4\n",
      legacy && direct ? "PASS" : "FAIL", int(legacy), int(direct));
  return legacy && direct ? 0 : 1;
}

#endif
