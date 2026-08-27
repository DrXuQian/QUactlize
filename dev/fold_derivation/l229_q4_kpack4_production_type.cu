// Production-type closure for the first canonical Q4_K K-pack4 reader.
// This gate composes the real builder, ordinary mixed-input collective,
// packed metadata channel, dense epilogue and Split-K wrapper without lowering
// PPU-only instructions on the host.

#include <cstdio>
#include <type_traits>

#include "fpA_intB_ppu.cuh"
#include "dense_splitk_parallel_ppu.cuh"
#include "q4_kpack4_offline.hpp"

using namespace cute;
using Schedule = ppu_group_schedule::FinegrainedSchedule<32>;
using TileShape = Shape<_8, _64, _256>;
using ScaleTile = Shape<_64, _8>;
using Warp = Shape<_8, _16, _256>;
using Types = fpa_intb_ppu::DenseQ4KPack4KernelTypes<
    ppu_mixed_policy::QuantMode::FinegrainedScaleZero,
    Schedule, TileShape, ScaleTile, Warp, 2, true>;
using Policy = typename Types::MainloopPolicy;
using Mainloop = typename Types::CollectiveMainloop;
using Builder = typename Policy::CollectiveBuilderType;
using Split = dense_splitk_parallel_ppu::KernelTypes<Types, TileShape, Warp>;

static_assert(Policy::ArtifactTileK == 0);
static_assert(Policy::TacticTileK == 256);
static_assert(Policy::Descriptor::q4_kpack4_transpose);
static_assert(Policy::Descriptor::transport_tile_k == 64);
static_assert(std::is_same_v<typename Policy::Descriptor::BProviderType,
                             ppu_mixed_policy::KPack4TransposedBProvider>);
static_assert(std::is_same_v<typename Policy::Descriptor::WeightLayoutType,
                             ppu_mixed_policy::Q4KPack4TransposeWeightLayout>);
static_assert(Builder::HasQ4KPack4);
static_assert(Builder::ArtifactTileK == 0);
static_assert(Mainloop::kQ4KPack4Transpose);
static_assert(Mainloop::PhysicalBTileK == 64);
static_assert(cute::cosize_v<typename Mainloop::SmemLayoutB> *
                      int(sizeof(typename Mainloop::BTransportElement)) ==
                  64 * 256 / 2 * 2,
              "K-pack4 shared bytes must equal compact Q4 bytes per stage");
static_assert(Mainloop::is_packed_scale);
static_assert(std::is_same_v<typename Split::GemmKernel::CollectiveMainloop,
                             Mainloop>);

int main() {
  std::printf("L229 Q4_K KPACK4 production-type PASS "
              "layout=0x%08x mapping=0x%016llx physical=NxK/4 "
              "transport=N16xK64 tactic=8x64x256_w8x16_s2\n",
              unsigned(q4_kpack4::kLayoutId),
              static_cast<unsigned long long>(q4_kpack4::kMappingId));
}
