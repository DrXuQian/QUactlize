// L121 -- THE OWNED GROUPED WRAPPER MUST CONSUME ITS POLICY SPECIALIZATION.
//
// Min=2 with Stages=3 is intentional: replacing the three-argument scheduler
// with its default Min=8 spelling still compiles and is ABI-compatible, but
// must trip the wrapper's exact Params identity assertion.  Min=8 would make
// that planted regression invisible.
#include <cstdio>
#include <type_traits>

#include "moe_grouped_streamk_ppu.cuh"
#include "quactlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_fold.hpp"

namespace {

using Operation = moe_grouped_streamk_ppu::Operation<
    moe_grouped_streamk_ppu::QuantMode::FinegrainedScaleZero,
    ppu_group_schedule::FinegrainedSchedule<32>,
    cute::Shape<cute::Int<16>, cute::Int<256>, cute::Int<64>>,
    cute::Shape<cute::Int<256>, cute::Int<2>>,
    cute::Shape<cute::Int<16>, cute::Int<64>, cute::Int<64>>,
    3, true, cutlass::int4b_t, void, 64, 2u>;
using Kernel = typename Operation::Kernel;
using Expected = cutlass::gemm::kernel::detail::
    PersistentTileSchedulerPPUStreamKParamsT<2>;

static_assert(std::is_same_v<typename Kernel::TileSchedulerParams, Expected>);
static_assert(Kernel::TileSchedulerParams::min_iters_per_sk_unit_ == 2);
static_assert(Kernel::MaxThreadsPerBlock == 128);
static_assert(Kernel::TileScheduler::FixupThreadCount ==
              Kernel::MaxThreadsPerBlock);
static_assert(sizeof(Kernel) > 0);

}  // namespace

int main() {
  std::printf("L121 grouped wrapper min=%u stages=%d threads=%u "
              "cohort=%u Params=exact PASS\n",
              Kernel::TileSchedulerParams::min_iters_per_sk_unit_,
              Kernel::DispatchPolicy::Stages, Kernel::MaxThreadsPerBlock,
              Kernel::TileScheduler::FixupThreadCount);
  return 0;
}
