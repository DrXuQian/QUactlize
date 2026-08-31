// Production-type closure for the first canonical Q4_K K-pack4 reader.
// This gate composes the real builder, ordinary mixed-input collective,
// packed metadata channel, dense epilogue and Split-K wrapper without lowering
// PPU-only instructions on the host.

#include <cstddef>
#include <cstdio>
#include <type_traits>

#include "fpA_intB_ppu.cuh"
#include "dense_splitk_parallel_ppu.cuh"
#include "q4_kpack4_offline.hpp"

using namespace cute;
namespace b_delivery =
    cutlass::gemm::collective::detail::quactlize_b_delivery;
using Schedule = ppu_group_schedule::FinegrainedSchedule<32>;
using TileShape = Shape<_8, _64, _256>;
using ScaleTile = Shape<_64, _8>;
using Warp = Shape<_8, _16, _256>;
using Types = fpa_intb_ppu::DenseQ4KPack4KernelTypes<
    ppu_mixed_policy::QuantMode::FinegrainedScaleZero,
    Schedule, TileShape, ScaleTile, Warp, 2, true>;
using PackedTypes = fpa_intb_ppu::DenseQ4KPack4KernelTypes<
    ppu_mixed_policy::QuantMode::FinegrainedScaleZero,
    Schedule, TileShape, ScaleTile, Warp, 2, true, 1>;
using D32Types = fpa_intb_ppu::DenseQ4KPack4KernelTypes<
    ppu_mixed_policy::QuantMode::FinegrainedScaleZero,
    Schedule, TileShape, ScaleTile, Warp, 2, true, 0, 32>;
using D16Types = fpa_intb_ppu::DenseQ4KPack4KernelTypes<
    ppu_mixed_policy::QuantMode::FinegrainedScaleZero,
    Schedule, TileShape, ScaleTile, Warp, 2, true, 0, 16>;
using PackedD32Types = fpa_intb_ppu::DenseQ4KPack4KernelTypes<
    ppu_mixed_policy::QuantMode::FinegrainedScaleZero,
    Schedule, TileShape, ScaleTile, Warp, 2, true, 1, 32>;
using PackedD16Types = fpa_intb_ppu::DenseQ4KPack4KernelTypes<
    ppu_mixed_policy::QuantMode::FinegrainedScaleZero,
    Schedule, TileShape, ScaleTile, Warp, 2, true, 1, 16>;
using TileShapeN16 = Shape<_8, _16, _256>;
using ScaleTileN16 = Shape<_16, _8>;
using WarpN16 = Shape<_8, _16, _256>;
using D32CapN16Types = fpa_intb_ppu::DenseQ4KPack4KernelTypes<
    ppu_mixed_policy::QuantMode::FinegrainedScaleZero,
    Schedule, TileShapeN16, ScaleTileN16, WarpN16, 2, true, 0, 32>;
using Policy = typename Types::MainloopPolicy;
using PackedPolicy = typename PackedTypes::MainloopPolicy;
using D32Policy = typename D32Types::MainloopPolicy;
using D16Policy = typename D16Types::MainloopPolicy;
using PackedD32Policy = typename PackedD32Types::MainloopPolicy;
using PackedD16Policy = typename PackedD16Types::MainloopPolicy;
using Mainloop = typename Types::CollectiveMainloop;
using PackedMainloop = typename PackedTypes::CollectiveMainloop;
using D32Mainloop = typename D32Types::CollectiveMainloop;
using D16Mainloop = typename D16Types::CollectiveMainloop;
using PackedD32Mainloop = typename PackedD32Types::CollectiveMainloop;
using PackedD16Mainloop = typename PackedD16Types::CollectiveMainloop;
using D32CapN16Mainloop = typename D32CapN16Types::CollectiveMainloop;
using MainloopStorage = typename Mainloop::SharedStorage;
using PackedMainloopStorage = typename PackedMainloop::SharedStorage;
using Builder = typename Policy::CollectiveBuilderType;
using DefaultOperandB = typename Builder::DefaultOperandB;
using BDeliveryBinding = typename DefaultOperandB::BDeliveryBinding;
using BDeliveryShared = typename BDeliveryBinding::Shared;
using Split = dense_splitk_parallel_ppu::KernelTypes<Types, TileShape, Warp>;

static_assert(Policy::ArtifactTileK == 0);
static_assert(Policy::TacticTileK == 256);
static_assert(Policy::Descriptor::q4_kpack4_transpose);
static_assert(Policy::Descriptor::transport_tile_k == 64);
static_assert(Policy::Descriptor::kpack4_scheduled_delivery_n == 0);
static_assert(Policy::Descriptor::kpack4_resolved_delivery_n == 64);
static_assert(D32Policy::Descriptor::kpack4_scheduled_delivery_n == 32);
static_assert(D32Policy::Descriptor::kpack4_resolved_delivery_n == 32);
static_assert(D16Policy::Descriptor::kpack4_scheduled_delivery_n == 16);
static_assert(D16Policy::Descriptor::kpack4_resolved_delivery_n == 16);
static_assert(PackedD32Policy::Descriptor::kpack4_scheduled_delivery_n == 32);
static_assert(PackedD32Policy::Descriptor::kpack4_resolved_delivery_n == 32);
static_assert(PackedD16Policy::Descriptor::kpack4_scheduled_delivery_n == 16);
static_assert(PackedD16Policy::Descriptor::kpack4_resolved_delivery_n == 16);
static_assert(std::is_same_v<typename Policy::Descriptor::BProviderType,
                             ppu_mixed_policy::KPack4TransposedBProvider>);
static_assert(std::is_same_v<typename Policy::Descriptor::WeightLayoutType,
                             ppu_mixed_policy::Q4KPack4TransposeWeightLayout>);
static_assert(Builder::HasQ4KPack4);
static_assert(Builder::ArtifactTileK == 0);
static_assert(std::is_same_v<typename Mainloop::BDeliveryPolicy,
                             b_delivery::ProductionBDelivery>);
static_assert(std::is_same_v<typename DefaultOperandB::BDeliveryTags,
                             b_delivery::ProductionBDelivery>);
static_assert(std::is_same_v<typename BDeliveryBinding::G2S,
                             typename b_delivery::ProductionBDelivery::G2S>);
static_assert(std::is_same_v<typename BDeliveryBinding::S2R,
                             typename b_delivery::ProductionBDelivery::S2R>);
static_assert(std::is_same_v<BDeliveryShared,
                             typename DefaultOperandB::PhysicalSharedContract>);
static_assert(std::is_same_v<typename BDeliveryShared::Layout,
                             typename DefaultOperandB::StageSmemLayout>);
static_assert(std::is_same_v<typename BDeliveryShared::Element,
                             typename Mainloop::BTransportElement>);
static_assert(BDeliveryShared::bytes_per_stage ==
                  q4_kpack4::placed_bytes(
                      int(cute::size<1>(TileShape{})),
                      int(cute::size<2>(TileShape{}))),
              "Q4 K-pack B-delivery bytes must be one compact offline tile");
static_assert(BDeliveryShared::n_atom == q4_kpack4::kTransportN);
static_assert(BDeliveryShared::logical_k_atom == q4_kpack4::kTransportK);
static_assert(BDeliveryShared::alignment_bytes == 32,
              "PPU0010 AIU shared base is a 32-byte contract");
static_assert(offsetof(MainloopStorage, smem_b) %
                      BDeliveryShared::alignment_bytes == 0,
              "ordinary Q4 storage must realize the declared B alignment");
static_assert(offsetof(PackedMainloopStorage, smem_b) %
                      BDeliveryShared::alignment_bytes == 0,
              "packed-A Q4 storage must realize the declared B alignment");
static_assert(DefaultOperandB::CodesPerWord == q4_kpack4::kPack &&
                  DefaultOperandB::PhysicalK * DefaultOperandB::CodesPerWord ==
                      int(cute::size<2>(TileShape{})));
static_assert(Mainloop::kQ4KPack4Transpose);
static_assert(Mainloop::kQ4KPack4ResolvedDeliveryN == 64);
static_assert(D32Mainloop::kQ4KPack4ResolvedDeliveryN == 32);
static_assert(D16Mainloop::kQ4KPack4ResolvedDeliveryN == 16);
static_assert(PackedD32Mainloop::kQ4KPack4ResolvedDeliveryN == 32);
static_assert(PackedD16Mainloop::kQ4KPack4ResolvedDeliveryN == 16);
static_assert(PackedD32Mainloop::kPackedA && PackedD16Mainloop::kPackedA);
static_assert(D32CapN16Mainloop::kQ4KPack4ScheduledDeliveryN == 32);
static_assert(D32CapN16Mainloop::kQ4KPack4ResolvedDeliveryN == 16,
              "a named delivery is a cap, not a minimum tactic N");
static_assert(Mainloop::PhysicalBTileK == 64);
static_assert(cute::cosize_v<typename Mainloop::SmemLayoutB> *
                      int(sizeof(typename Mainloop::BTransportElement)) ==
                  64 * 256 / 2 * 2,
              "K-pack4 shared bytes must equal compact Q4 bytes per stage");
static_assert(cute::cosize_v<typename D32Mainloop::SmemLayoutB> ==
                  cute::cosize_v<typename Mainloop::SmemLayoutB>);
static_assert(cute::cosize_v<typename D16Mainloop::SmemLayoutB> ==
                  cute::cosize_v<typename Mainloop::SmemLayoutB>);
static_assert(cute::cosize_v<typename PackedD32Mainloop::SmemLayoutB> ==
                  cute::cosize_v<typename PackedMainloop::SmemLayoutB>);
static_assert(cute::cosize_v<typename PackedD16Mainloop::SmemLayoutB> ==
                  cute::cosize_v<typename PackedMainloop::SmemLayoutB>);
static_assert(Mainloop::is_packed_scale);
static_assert(std::is_same_v<typename Split::GemmKernel::CollectiveMainloop,
                             Mainloop>);
static_assert(Policy::PackedARows == 0);
static_assert(PackedPolicy::PackedARows == 1);
static_assert(PackedPolicy::Descriptor::q4_kpack4_transpose);
static_assert(std::is_same_v<typename PackedPolicy::Descriptor::AProviderType,
                             ppu_mixed_policy::PackedRowAProvider>);
static_assert(std::is_same_v<typename PackedPolicy::Descriptor::BProviderType,
                             ppu_mixed_policy::KPack4TransposedBProvider>);
static_assert(PackedPolicy::CollectiveBuilderType::HasQ4KPack4);
static_assert(PackedMainloop::kQ4KPack4Transpose);
static_assert(PackedMainloop::kPackedA);
static_assert(PackedMainloop::kAPackRows == 1);

int main() {
  std::printf("L229 Q4_K KPACK4 production-type PASS "
              "layout=0x%08x mapping=0x%016llx physical=NxK/4 "
              "transport=N16xK64 delivery=auto64+32+16 "
              "tactic=8x64x256_w8x16_s2 "
              "providers=standard-aiu+packed-row\n",
              unsigned(q4_kpack4::kLayoutId),
              static_cast<unsigned long long>(q4_kpack4::kMappingId));
}
