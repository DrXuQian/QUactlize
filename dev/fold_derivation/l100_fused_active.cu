// L100 -- metadata publication is an ordinary schedule identity.
//
// The generic builder keeps two fp16 planes. The canonical dense Q4 K-pack4
// type requests an interleaved half2 publication, which becomes active only
// when the collective actually carries packed ScaleZero metadata. No device is
// needed: every claim is a property of the production CollectiveOp.

#include <cstdio>
#include <type_traits>

#include "fpA_intB_ppu.cuh"

using namespace cute;
using Schedule = ppu_group_schedule::FinegrainedSchedule<32>;
using TileShape = Shape<_16, _128, _256>;
using ScaleTileShape = Shape<_128, _8>;
using WarpShape = Shape<_16, _16, _256>;

using Generic = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cute::tuple<cutlass::int4b_t, cutlass::half_t, cutlass::half_t>,
    cutlass::layout::ColumnMajorInterleaved<256>, 32, float,
    cute::tuple<TileShape, ScaleTileShape>, WarpShape, cute::Int<2>,
    cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs32>::CollectiveOp;

using ProductionTypes = fpa_intb_ppu::DenseQ4KPack4KernelTypes<
    ppu_mixed_policy::QuantMode::FinegrainedScaleZero, Schedule, TileShape,
    ScaleTileShape, WarpShape, 2, true>;
using Production = typename ProductionTypes::CollectiveMainloop;

static_assert(std::is_same_v<typename Generic::MetadataPublication,
                             cutlass::gemm::SeparateHalfPlanes>);
static_assert(!Generic::is_fused_scale_zero,
              "generic/base schedules must retain separate metadata planes");
static_assert(std::is_same_v<typename Production::MetadataPublication,
                             cutlass::gemm::InterleavedHalf2>);
static_assert(Production::is_packed_scale,
              "the pinned production type must carry packed metadata");
static_assert(Production::has_zero_channel,
              "the pinned production type must carry a zero channel");
static_assert(Production::is_fused_scale_zero,
              "canonical dense Q4 K-pack4 must activate interleaved publication");

static constexpr int kScaleElems =
    int(Production::SharedStorage::scale_elements);
static constexpr int kZeroElems =
    int(Production::SharedStorage::zero_elements);
static constexpr int kPlain =
    int(cute::cosize_v<typename Production::SmemLayoutScale>);
static_assert(kZeroElems == 0);
static_assert(kScaleElems == 2 * kPlain);

using W = typename Production::FusedScaleWordLayout;
using H = typename Production::FusedScaleHalfLayout;
static_assert(int(cute::stride<0>(W{})) == 1);
static_assert(int(cute::stride<1>(W{})) == 128);
static_assert(int(cute::stride<0>(H{})) == 2);

int main() {
  std::printf("[l100] typed metadata publication PASS scale=%d zero=%d plain=%d\n",
              kScaleElems, kZeroElems, kPlain);
  return 0;
}
