// L134 -- production-Cfg and generated-device-code binding for Marlin.
//
// Unlike L133, this translation unit starts at the real dense Cfg and obtains
// the scheduler only through Cfg::MarlinGemm::GemmKernel.  The constexpr
// witnesses and runtime PTX probe therefore cannot silently switch to a
// parallel raw-core model.

#define LOWBIT_DENSE_UNIT_BUILD 1
#define DENSE_MARLIN_SWEEP 1
#define BENCH_GS 128
#define BENCH_TSK 64
#define TILE_M 16
#define TILE_N 128
#define WARP_M 16
#define WARP_N 32
#define STAGES 3
#include "test_lowbit_dense_bench.cu"

#include <type_traits>

using L134Cfg = Cfg<128, 16, 128, 128, 16, 32, 3>;
using L134Gemm = typename L134Cfg::MarlinGemm;
using L134Kernel = typename L134Gemm::GemmKernel;
using L134Scheduler = typename L134Kernel::TileScheduler;
using L134ExpectedScheduler =
    cutlass::gemm::kernel::detail::PersistentTileSchedulerPPUMarlin<
        typename L134Kernel::TileShape, typename L134Kernel::ClusterShape,
        L134Kernel::MaxThreadsPerBlock>;

static_assert(std::is_same_v<L134Scheduler, L134ExpectedScheduler>,
              "real dense Cfg must bind the exact production Marlin scheduler");
static_assert(L134Scheduler::FixupThreadCount ==
                  L134Kernel::MaxThreadsPerBlock,
              "real dense Cfg must bind CTA and cooperative cohorts exactly");
static_assert(std::is_trivially_copyable_v<typename L134Scheduler::Params>);
static_assert(!L134Kernel::CollectiveMainloop::SwapAB,
              "this production Cfg must present M/N to the scheduler without an axis swap");

template <class Shape>
CUTLASS_HOST_DEVICE constexpr auto l134_make_params(
    Shape const& raw_problem_shape, unsigned long long cu) {
  auto const scheduler_shape =
      L134Kernel::scheduler_problem_shape(raw_problem_shape);
#if defined(L134_RAW_CORE_PLANT)
  // Deliberate expert-pitch-class regression: reinterpret caller scalar
  // M/N/K as scheduler tile ordinals.  Concrete witnesses below must red.
  return L134Scheduler::make_params_for_tiles(
      (unsigned long long)cute::get<0>(scheduler_shape),
      (unsigned long long)cute::get<1>(scheduler_shape),
      (unsigned long long)cute::get<3>(scheduler_shape),
      (unsigned long long)cute::get<2>(scheduler_shape), cu);
#else
  return L134Scheduler::make_params_for_problem_shape(
      scheduler_shape, cu);
#endif
}

constexpr auto l134_classic =
    l134_make_params(cute::make_shape(16, 2048, 2048, 1), 20);
#if defined(L134_RAW_CORE_PLANT)
static_assert(l134_classic.grid_blocks_ == 20,
              "L134 raw-core substitution may not bypass production raw-shape lowering");
#endif
constexpr auto l134_c1 = L134Scheduler::get_work_for_block(l134_classic, 1);
constexpr auto l134_c2 =
    L134Scheduler::fetch_next_work_for_params(l134_classic, l134_c1);
static_assert(l134_classic.valid_ && l134_classic.grid_blocks_ == 20 &&
              l134_classic.iters_per_block_ == 13);
static_assert(l134_c1.output_tile_idx == 0 && l134_c1.K_idx == 13 &&
              l134_c1.k_tile_count == 3 && l134_c1.slice_count == 2 &&
              l134_c1.slice_idx == 0 && l134_c1.lock_idx == 0 &&
              l134_c1.linear_begin == 13 && l134_c1.linear_next == 16 &&
              l134_c1.linear_end == 26);
static_assert(l134_c2.output_tile_idx == 1 && l134_c2.K_idx == 0 &&
              l134_c2.k_tile_count == 10 && l134_c2.slice_count == 2 &&
              l134_c2.slice_idx == 1 && l134_c2.lock_idx == 1 &&
              l134_c2.N_idx == 1 && l134_c2.M_idx == 0 && l134_c2.L_idx == 0);

constexpr auto l134_decode =
    l134_make_params(cute::make_shape(1, 4096, 4096, 1), 72);
constexpr auto l134_d = L134Scheduler::get_work_for_block(l134_decode, 1);
static_assert(l134_decode.tiles_m_ == 1 && l134_decode.tiles_n_ == 32 &&
              l134_decode.k_tiles_per_output_ == 32 &&
              l134_decode.grid_blocks_ == 72 &&
              l134_decode.iters_per_block_ == 15 &&
              l134_d.output_tile_idx == 0 && l134_d.K_idx == 15 &&
              l134_d.k_tile_count == 15 && l134_d.slice_count == 3 &&
              l134_d.slice_idx == 1 && l134_d.lock_idx == 0 &&
              l134_d.N_idx == 0 && l134_d.M_idx == 0 && l134_d.L_idx == 0);

constexpr auto l134_batched =
    l134_make_params(cute::make_shape(17, 129, 3841, 2), 9);
constexpr auto l134_b1 = L134Scheduler::get_work_for_block(l134_batched, 4);
constexpr auto l134_b2 =
    L134Scheduler::fetch_next_work_for_params(l134_batched, l134_b1);
static_assert(l134_batched.grid_blocks_ == 9 &&
              l134_batched.iters_per_block_ == 28);
static_assert(l134_b1.output_tile_idx == 3 && l134_b1.K_idx == 19 &&
              l134_b1.k_tile_count == 12 && l134_b1.slice_count == 2 &&
              l134_b1.slice_idx == 0 && l134_b1.lock_idx == 3 &&
              l134_b1.N_idx == 1 && l134_b1.M_idx == 1 && l134_b1.L_idx == 0);
static_assert(l134_b2.output_tile_idx == 4 && l134_b2.K_idx == 0 &&
              l134_b2.k_tile_count == 16 && l134_b2.slice_count == 2 &&
              l134_b2.slice_idx == 1 && l134_b2.lock_idx == 4 &&
              l134_b2.N_idx == 0 && l134_b2.M_idx == 0 && l134_b2.L_idx == 1);

constexpr auto l134_c1_output_coord =
    L134Kernel::scheduler_output_tile_coord(l134_c1);
constexpr auto l134_c1_k_coord = L134Kernel::scheduler_k_tile_coord(
    l134_c1, cute::make_shape(cute::Int<4>{}, cute::Int<4>{}));
static_assert(cute::get<0>(l134_c1_output_coord) == 0 &&
              cute::get<1>(l134_c1_output_coord) == 0 &&
              cute::get<2>(l134_c1_output_coord) == 0);
static_assert(cute::crd2idx(
                  l134_c1_k_coord,
                  cute::make_shape(cute::Int<4>{}, cute::Int<4>{})) == 13,
              "K tile ordinal must enter idx2crd without a scalar/code/byte factor");
static_assert(L134Scheduler::get_work_k_tile_start(l134_c1) == 13 &&
              L134Scheduler::get_work_k_tile_count(
                  l134_c1, cute::make_shape(16, 2048, 2048, 1),
                  typename L134Kernel::TileShape{}) == 3);
static_assert(L134Scheduler::output_tile_index(l134_classic, l134_c2) == 1 &&
              L134Scheduler::reduction_workspace_element_offset(l134_c2) == 2048 &&
              L134Scheduler::barrier_lock_index(l134_c2) == 1 &&
              L134Scheduler::requires_fixup(l134_classic, l134_c2) &&
              L134Scheduler::compute_epilogue(l134_c2, l134_classic),
              "q must name one FP32 element tile and the same global lock");

#if defined(L134_WRONG_EXPECTATION)
// The compiler must print the reduced actual value `(13 == 12)` (or the
// equivalent diagnostic) rather than merely failing a hand-written bool.
static_assert(l134_classic.iters_per_block_ == 12,
              "L134 deliberate wrong expectation; actual I must be visible");
#endif

// This constant is generated by the same production scheduler type.  PTX
// inspection checks both its symbol and its literal field vector.
extern "C" __device__ __constant__ unsigned long long
l134_marlin_constexpr_witness[32] = {
  l134_classic.grid_blocks_, l134_classic.iters_per_block_,
  L134Scheduler::output_tile_index(l134_classic, l134_c1),
  L134Scheduler::get_work_k_tile_start(l134_c1), l134_c1.k_tile_count,
  l134_c1.slice_count, l134_c1.slice_idx,
  L134Scheduler::output_tile_index(l134_classic, l134_c1),
  L134Scheduler::reduction_workspace_element_offset(l134_c1),
  (unsigned long long)L134Scheduler::barrier_lock_index(l134_c1),
  L134Scheduler::requires_fixup(l134_classic, l134_c1),
  L134Scheduler::compute_epilogue(l134_c1, l134_classic),
  (unsigned long long)cute::crd2idx(
      l134_c1_k_coord, cute::make_shape(cute::Int<4>{}, cute::Int<4>{})),
  L134Scheduler::output_tile_index(l134_classic, l134_c2),
  L134Scheduler::get_work_k_tile_start(l134_c2), l134_c2.k_tile_count,
  l134_c2.slice_count, l134_c2.slice_idx,
  L134Scheduler::output_tile_index(l134_classic, l134_c2),
  L134Scheduler::reduction_workspace_element_offset(l134_c2),
  (unsigned long long)L134Scheduler::barrier_lock_index(l134_c2),
  L134Scheduler::requires_fixup(l134_classic, l134_c2),
  L134Scheduler::compute_epilogue(l134_c2, l134_classic),
  l134_decode.grid_blocks_, L134Scheduler::get_work_k_tile_start(l134_d),
  l134_d.k_tile_count, l134_d.slice_count, l134_batched.grid_blocks_,
  l134_b1.output_tile_idx, l134_b1.K_idx, l134_b2.output_tile_idx,
  L134Kernel::MaxThreadsPerBlock,
};

// Runtime inputs prevent constant folding. The probe writes every scheduler
// coordinate/progress field and follows the real fetch loop. No PPU opcode is
// involved: this is the integer device seam that can be inspected locally.
extern "C" __global__ void l134_marlin_runtime_probe(
    unsigned long long m, unsigned long long n, unsigned long long k,
    unsigned long long l, unsigned long long cu,
    unsigned long long* out) {
  auto params = l134_make_params(
      cute::make_shape(int(m), int(n), int(k), int(l)), cu);
  L134Scheduler scheduler{params};
  auto work = scheduler.get_work_for_block_index(
      (unsigned long long)blockIdx.x);
  unsigned long long segments = 0;
  while (work.is_valid()) {
    auto const output_coord = L134Kernel::scheduler_output_tile_coord(work);
    auto const k_coord = L134Kernel::scheduler_k_tile_coord(
        work, cute::make_shape(params.k_tiles_per_output_, cute::Int<1>{}));
    unsigned long long slot =
        ((unsigned long long)blockIdx.x * 1024 + segments) * 20;
    out[slot + 0] = (unsigned long long)cute::get<0>(output_coord);
    out[slot + 1] = (unsigned long long)cute::get<1>(output_coord);
    out[slot + 2] = (unsigned long long)cute::get<2>(output_coord);
    out[slot + 3] = L134Scheduler::get_work_k_tile_start(work);
    out[slot + 4] = L134Scheduler::get_work_k_tile_count(
        work, cute::make_shape(m, n, k, l), typename L134Kernel::TileShape{});
    out[slot + 5] = work.slice_count;
    out[slot + 6] = work.slice_idx;
    out[slot + 7] = L134Scheduler::output_tile_index(params, work);
    out[slot + 8] =
        (unsigned long long)L134Scheduler::barrier_lock_index(work);
    out[slot + 9] = work.linear_begin;
    out[slot + 10] = work.linear_next;
    out[slot + 11] = work.linear_end;
    out[slot + 12] =
        L134Scheduler::reduction_workspace_element_offset(work);
    out[slot + 13] = L134Scheduler::requires_fixup(params, work);
    out[slot + 14] = L134Scheduler::compute_epilogue(work, params);
    out[slot + 15] = (unsigned long long)cute::get<0>(k_coord);
    out[slot + 16] = (unsigned long long)cute::get<1>(k_coord);
    ++segments;
    work = scheduler.get_next_work(work);
  }
  unsigned long long summary =
      ((unsigned long long)blockIdx.x * 1024 + 1023) * 20;
  out[summary + 0] = params.grid_blocks_;
  out[summary + 1] = params.iters_per_block_;
  out[summary + 2] = params.active_blocks_;
  out[summary + 3] = segments;
}
