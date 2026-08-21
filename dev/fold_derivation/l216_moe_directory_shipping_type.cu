// Force one exact ScaleFirst grouped mixed-input collective through the new
// directory-persistent driver.  L216_QTYPE selects Q2_K..Q6_K; the companion
// script compiles all five in separate compiler invocations so one format's
// vendor-asm stop cannot masquerade as coverage of the remaining formats.
//
// The local CUDA compiler cannot encode PPU instructions.  The runner admits
// only those known vendor-asm diagnostics and rejects every scheduler,
// adapter, format, layout and type error.

#include <type_traits>

#include "moe_grouped_ppu.cuh"
#include "ppu_format_config.hpp"
#include "actlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_fold.hpp"
#include "actlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_mixed_input_2plane.hpp"

#ifndef L216_QTYPE
#error "L216_QTYPE must select one of the five shipping GGUF-K formats"
#endif

using namespace cute;
using Q = moe_grouped_ppu::QuantMode;

template <int QType>
struct L216Format;

template <>
struct L216Format<10> {
  using Low = cutlass::uint2b_t;
  using High = void;
  static constexpr Q Mode = Q::FinegrainedScaleZero;
  static constexpr int GroupSize = 16;
  static constexpr int WarpN = 32;
};

template <>
struct L216Format<11> {
  using Low = cutlass::uint2b_t;
  using High = cutlass::uint1b_t;
  static constexpr Q Mode = Q::FinegrainedScaleOnly;
  static constexpr int GroupSize = 16;
  static constexpr int WarpN = 64;
};

template <>
struct L216Format<12> {
  using Low = cutlass::int4b_t;
  using High = void;
  static constexpr Q Mode = Q::FinegrainedScaleZero;
  static constexpr int GroupSize = 32;
  static constexpr int WarpN = 32;
};

template <>
struct L216Format<13> {
  using Low = cutlass::int4b_t;
  using High = cutlass::uint1b_t;
  static constexpr Q Mode = Q::FinegrainedScaleZero;
  static constexpr int GroupSize = 32;
  static constexpr int WarpN = 64;
};

template <>
struct L216Format<14> {
  using Low = cutlass::int4b_t;
  using High = cutlass::uint2b_t;
  static constexpr Q Mode = Q::FinegrainedScaleOnly;
  static constexpr int GroupSize = 16;
  static constexpr int WarpN = 32;
};

using F = L216Format<L216_QTYPE>;
using Low = typename F::Low;
using High = typename F::High;
constexpr auto kRegistry = ppu_formats::for_qtype(L216_QTYPE);
constexpr int kArtifactTileK = kRegistry.scale_first_tile_k;
constexpr int kTacticTileK = kArtifactTileK;
constexpr int kScaleGroups =
    ppu_group_schedule::scale_groups_v<kTacticTileK, F::GroupSize>;

static_assert(kRegistry.qtype == L216_QTYPE &&
              kRegistry.low_bits == ppu_mixed_policy::element_bits_v<Low> &&
              kRegistry.high_bits == ppu_mixed_policy::element_bits_v<High> &&
              kRegistry.group_size == F::GroupSize,
              "L216 format types must agree with the shipping registry");

using L216Tile = Shape<Int<64>, Int<64>, C<kTacticTileK>>;
using L216ScaleTile = Shape<Int<64>, C<kScaleGroups>>;
using L216Warp = Shape<Int<64>, C<F::WarpN>, C<kTacticTileK>>;
using L216Schedule = ppu_group_schedule::FinegrainedSchedule<F::GroupSize>;
using L216Policy = moe_grouped_ppu::MixedMainloopPolicy<
    F::Mode, L216Schedule, L216Tile, L216ScaleTile, L216Warp, 3, true,
    Low, High, kArtifactTileK>;

static_assert(L216Policy::Descriptor::quant_mode == F::Mode &&
              L216Policy::Descriptor::low_bits == kRegistry.low_bits &&
              L216Policy::Descriptor::high_bits == kRegistry.high_bits &&
              L216Policy::Descriptor::artifact_tile_k ==
                  kRegistry.scale_first_tile_k &&
              !L216Policy::Descriptor::packed_metadata,
              "L216 must instantiate the exact ScaleFirst layout, not a packed surrogate");
static_assert((kRegistry.high_bits == 0 &&
               std::is_same_v<typename L216Policy::Descriptor::BProviderType,
                              ppu_mixed_policy::OrdinaryBProvider>) ||
              (kRegistry.high_bits != 0 &&
               std::is_same_v<typename L216Policy::Descriptor::BProviderType,
                              ppu_mixed_policy::TwoPlaneBProvider<1, 1>>),
              "L216 single/two-plane provider must match the resident layout");

void force_l216(
    cutlass::half_t const* a, Low const* b,
    cutlass::half_t const* scale, cutlass::half_t const* zero,
    High const* b2, cutlass::half_t** d,
    moe_grouped_ppu::DStride* stride_d, int const* rows,
    moe_grouped_ppu::GroupShape* shapes, int const* offsets,
    char* workspace) {
  (void)moe_grouped_ppu::launch<
      F::Mode, L216Schedule,
      L216Tile, L216ScaleTile, L216Warp, 3, true,
      Low, High, false,
      false, false, kArtifactTileK, true>(
          a, b, scale, zero, d, stride_d, rows,
          128, 4096, 4096, 64, F::GroupSize,
          shapes, nullptr, offsets,
          workspace, 1u << 20, nullptr, b2);
}

int main() { return 0; }
