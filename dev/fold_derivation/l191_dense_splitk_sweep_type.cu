// Compile-time witness for the generated sweep's prepared M1 packed-A path.
#include <cstdio>
#include <type_traits>

#define PPU_B_CHUNK 0
#include "dense_splitk_parallel_ppu.cuh"
#include "ppu_group_schedule.hpp"

using namespace cute;
using Schedule = ppu_group_schedule::FinegrainedSchedule<128>;
using TileShape = Shape<_8, _128, _128>;
using ScaleTile = Shape<_128, _1>;
using Warp = Shape<_8, _32, _128>;
using Shipping = fpa_intb_ppu::DensePackedAKernelTypes<
    1, fpa_intb_ppu::QuantMode::FinegrainedScaleOnly, Schedule,
    TileShape, ScaleTile, Warp, 3, true, cutlass::int4b_t, 64>;
using Prepared = dense_splitk_parallel_ppu::PreparedOnePlaneLauncher<
    Shipping, TileShape, Warp>;
using Split = dense_splitk_parallel_ppu::KernelTypes<Shipping, TileShape, Warp>;

static_assert(Shipping::MainloopPolicy::ArtifactTileK == 64 &&
              Shipping::MainloopPolicy::PackedARows == 1);
static_assert(std::is_same_v<typename Shipping::CollectiveMainloop,
                             typename Split::CollectiveMainloop>);
static_assert(size<0>(typename Shipping::CollectiveMainloop::TiledMma::AtomShape_MNK{}) == 8);

int main() {
  dense_splitk_parallel_ppu::SplitKParallelSpanEvents events;
  std::printf(
      "[l191:type] prepared_bytes=%zu event_null_valid=%d "
      "artifact_tk=%d packed_rows=%d atom_m=%d -> %s\n",
      sizeof(Prepared), int(events.valid()),
      Shipping::MainloopPolicy::ArtifactTileK,
      Shipping::MainloopPolicy::PackedARows,
      int(size<0>(typename Shipping::CollectiveMainloop::TiledMma::AtomShape_MNK{})),
      sizeof(Prepared) > 0 && !events.valid() ? "PASS" : "FAIL");
  return sizeof(Prepared) == 0 || events.valid();
}
