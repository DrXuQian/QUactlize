// L135 -- grouped Marlin consumes every shipping mixed-input mainloop family.
// This is a compile/type oracle, not a device result.
#include <type_traits>

#include "moe_grouped_marlin_ppu.cuh"
#include "actlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_fold.hpp"
#include "actlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_mixed_input_2plane.hpp"

namespace {
namespace gm = moe_grouped_marlin_ppu;
using QM = gm::QuantMode;

template <class B, class B2, int ArtifactTK, int TM, int TN, int TK,
          int WM, int WN, int Stages>
using Op = gm::Operation<
    QM::FinegrainedScaleZero,
    ppu_group_schedule::FinegrainedSchedule<32>,
    cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>,
    cute::Shape<cute::Int<TN>, cute::Int<TK / 32>>,
    cute::Shape<cute::Int<WM>, cute::Int<WN>, cute::Int<TK>>,
    Stages, true, B, B2, ArtifactTK>;

// One-plane F=1 (ordinary converter), one-plane F=2 (folded converter), and
// two-plane Q3 exercise the three shipping collective families.  Scheduler
// code is deliberately identical across them.
using Ordinary = Op<cutlass::int4b_t, void, 64, 16, 256, 64, 16, 64, 3>;
using Folded = Op<cutlass::uint2b_t, void, 64, 16, 256, 64, 16, 64, 3>;
using TwoPlane = Op<cutlass::uint2b_t, cutlass::uint1b_t,
                    64, 16, 256, 64, 16, 64, 3>;

template <class Operation>
constexpr bool common_kernel_contract() {
  using K = typename Operation::Kernel;
  return K::IsGroupedMarlin &&
         cutlass::gemm::kernel::isGroupProblemShape_v<typename K::ProblemShape> &&
         cute::is_base_of_v<cutlass::gemm::KernelAiuMultistageMixedInput,
             typename K::DispatchPolicy::Schedule> &&
         K::TileScheduler::FixupThreadCount == K::MaxThreadsPerBlock &&
         sizeof(K) > 0;
}

static_assert(common_kernel_contract<Ordinary>());
static_assert(common_kernel_contract<Folded>());
static_assert(common_kernel_contract<TwoPlane>());
static_assert(std::is_same_v<typename Ordinary::Kernel::TileScheduler,
                             typename Folded::Kernel::TileScheduler>);
static_assert(std::is_same_v<typename Ordinary::Kernel::TileScheduler,
                             typename TwoPlane::Kernel::TileScheduler>);
static_assert(Ordinary::MainloopPolicy::ArtifactLowFold == 1);
static_assert(Folded::MainloopPolicy::ArtifactLowFold == 2);
static_assert(TwoPlane::MainloopPolicy::HighBits == 1);
static_assert(TwoPlane::MainloopPolicy::ArtifactHighFold == 4);
}  // namespace

int main() { return 0; }
