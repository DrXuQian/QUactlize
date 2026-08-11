// Grouped MIXED-INPUT GEMM kernel for actlize v1.0.0 -- the actlize analogue of trtllm's MoeFCGemm<mixed-input Mma>.
//
// This is a NEW GemmUniversal specialization that combines:
//   - the GroupScheduler + GroupProblemShape ragged-M machinery from ppu_aiu_gemm_array_group.hpp, and
//   - the mixed-input collective's scale-aware drive (load_init + operator) from ppu_aiu_gemm_mixed_input.hpp.
// The existing array_group kernel drives the PLAIN (BatchArray) collective with a gA/gB-passed interface and
// has no scales; this one drives the mixed-input collective (which carries scale/zero internally), so W4A16
// grouped GEMM works. enable_if keys on the mixed-input schedule + a GroupProblemShape, so it does not collide
// with the single-GEMM mixed-input specialization (which requires a rank-3/4 ProblemShape).
//
// STEP 2a (this file): DEGENERATE / uniform-M. The mixed-input collective still slices A by l_coord with a
// uniform L-stride (mA_mkl(_,_,l_coord)), so per-expert M must be uniform -- equivalent to step 1 but routed
// through the GroupScheduler. Purpose: prove the scheduler+collective wiring compiles and runs.
// STEP 2b (next): ragged A -- give the mixed-input collective a per-expert A base (ptr_A_array[l_coord] +
// M_e) so token counts can vary. B/scale/zero stay L-strided (uniform per-expert) and are untouched.
//
// UNTESTED on box (private SDK; cannot compile here). Integration points most likely to need a fix are marked
// [Q1]..[Q4] inline.
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/gemm.h"
#include "cute/ppu_util.hpp"
#include "cutlass/utils.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cute/tensor.hpp"

namespace cutlass::gemm::kernel {

#if defined(PPU_METADATA_ADDR_PROBE) && (PPU_METADATA_ADDR_PROBE != 0)
// Test-only address/value trace for the grouped metadata path.  The production
// ABI and codegen do not contain this buffer or branch unless the dedicated
// probe target defines PPU_METADATA_ADDR_PROBE.
//
// Keep the trace split into three observations.  A normal scalar load from an
// explicit int64 GEP establishes the allocation contents; gZ establishes the
// shape-only expert slice; tZgZ plus the copied smem value separate CuTe's
// partition from the PPU cp.async address path.  G5's controlled q==8 fixture
// proves only the zero plane, so this trace deliberately makes no claim about B.
inline constexpr uint32_t kGroupedMetadataProbeMagic = 0x475a4150u; // "GZAP"
inline constexpr uint32_t kGroupedMetadataProbeVersion = 1u;
inline constexpr int kGroupedMetadataProbeExperts = 3;
inline constexpr uint32_t kGroupedMetadataProbeMaxShapeRecords = 1024u;
inline constexpr uint32_t kGroupedMetadataProbeMaxCopyRecords = 4096u;

struct GroupedMetadataShapeRecord {
  int32_t scheduler_expert = -1;
  int32_t local_n = -1;
  int32_t scale_group = -1;
  int32_t thread_idx = -1;
  uint64_t explicit_addr = 0;
  uint64_t gz_addr = 0;
  uint64_t gz_base_addr = 0;
  uint16_t explicit_bits = 0;
  uint16_t gz_bits = 0;
};

struct GroupedMetadataCopyRecord {
  int32_t scheduler_expert = -1;
  int32_t thread_idx = -1;
  int32_t copy_slot = -1;
  int32_t metadata_tile = -1;
  int32_t value_idx = -1;
  uint64_t partition_addr = 0;
  uint64_t destination_addr = 0;
  uint16_t partition_bits = 0;
  uint16_t cp_async_bits = 0;
};

struct GroupedMetadataAddressProbe {
  uint32_t magic = kGroupedMetadataProbeMagic;
  uint32_t version = kGroupedMetadataProbeVersion;
  uint32_t shape_count = 0;
  uint32_t copy_count = 0;
  uint32_t overflow = 0;
  uint32_t configuration_errors = 0;
  uint32_t cta_threads = 0;
  uint32_t thread_slots = 0;
  uint32_t metadata_tiles = 0;
  uint32_t values_per_thread = 0;
  uint32_t expert_ctas[kGroupedMetadataProbeExperts]{};
  GroupedMetadataShapeRecord shape[kGroupedMetadataProbeMaxShapeRecords]{};
  GroupedMetadataCopyRecord copy[kGroupedMetadataProbeMaxCopyRecords]{};
};
#endif

///////////////////////////////////////////////////////////////////////////////

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class GemmUniversal<
  ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_,
  cute::enable_if_t<
    cute::is_base_of_v<KernelAiuMultistageMixedInput, typename CollectiveMainloop_::DispatchPolicy::Schedule>
    && isGroupProblemShape_v<ProblemShape_>>>          // <- mixed-input schedule AND a group problem shape
{
public:
  using ProblemShape = ProblemShape_;
  static_assert(rank(typename ProblemShape::UnderlyingProblemShape{}) == 3
             or rank(typename ProblemShape::UnderlyingProblemShape{}) == 4,
    "UnderlyingProblemShape should be <M,N,K> or <M,N,K,L>");

  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static constexpr bool IsGroupedGemmKernel = isGroupProblemShape_v<ProblemShape>;
  using TileScheduler = typename detail::TileSchedulerSelector<
      GroupScheduler, ArchTag, TileShape, ClusterShape, ProblemShape>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  struct SharedStorage {
    union SharedTensorStorage {
      using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
      using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
      MainloopSharedStorage mainloop;
      EpilogueSharedStorage epilogue;
    } tensors;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static constexpr uint32_t MaxThreadsPerBlock = cute::size(TiledMma{});
  // Reproduce the recorded occupancy arm: PPU_DEFS=PPU_MAXREG=100 TARGET=test_w4a16_diag ./build.sh
  // 100 is load-bearing: the measured default was 106 regs/thread. At 128 threads, 100 asks for 10 blocks and a
  // nominal <=102 regs/thread, while writing 106 would still ask for only 9 blocks and impose no new constraint.
  // PPU_MAXREG: cap registers per thread by asking __launch_bounds__ for more resident blocks. device_kernel.h
  // passes MinBlocksPerMultiprocessor straight into __launch_bounds__, so the compiler must fit
  // 131072 / (blocks * threads) registers. Expressed as a REGISTER target and converted here, because the block
  // count that implies depends on MaxThreadsPerBlock -- 10 blocks means 102 registers at 128 threads and 51 at 256,
  // and hardcoding the block count would silently over-constrain the wider tiles.
  //
  // Whether it pays is a separate question: at S=1 the grid supplies 4 blocks/CU for a 16x64 tile while 106
  // registers already allow 9, so registers are NOT the binding limit there. Measured, not assumed.
#if defined(PPU_MAXREG) && (PPU_MAXREG > 0)
  static constexpr uint32_t MinBlocksPerMultiprocessor =
      (131072u / (uint32_t(PPU_MAXREG) * MaxThreadsPerBlock)) > 0
          ? (131072u / (uint32_t(PPU_MAXREG) * MaxThreadsPerBlock)) : 1u;
#else
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
#endif
  static constexpr uint32_t NumMmaWarpGroups = 1;
  static constexpr int MinWorkspaceAlignment = 16;   // [Q1] array_group takes this from an outer scope; define locally

  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
    int probe = 0;   // debug: 0=normal; 1=ROUTING probe (skip GEMM, write expert+1 to every output element)
    int const* group_M = nullptr;   // device [L] per-expert M_e; O(L) decode of blockIdx.x -> (expert,m_tile)
    int mtiles_uniform = 0;         // >0: 3D grid's per-expert M-tile bound; 0: flat ragged grid, scan group_M
    // Mainloop/epilogue need one representative shape for their uniform N/K strides. Device-only grouped callers
    // deliberately omit host_problem_shapes, so carry that shape independently of the scheduler's host fast path.
    int representative_m = 0;
    int representative_n = 0;
    int representative_k = 0;
    // IN-KERNEL SPLIT-K. >1 slices the K loop across gridDim.z so all S slices' CTAs are resident in ONE launch.
    // Doing it on the host instead -- S launches on a stream -- serialises them and raises no occupancy at all:
    // measured 23.49 / 44.51 / 69.70 / 121.37 us for S = 1,2,4,8, i.e. linear in S.
    //
    // The split is STRIDED (slice z takes k-tiles z, z+S, z+2S ...), which is what cute's SplitkCoordIterator
    // does, so the B pointer never moves and there is NO constraint that a slice fall on the 256-element offline
    // tile boundary -- that constraint belonged to the host-slicing form and is gone.
    //
    // With splitk > 1 the epilogue's ptr_D/stride_D arrays must hold L*splitk entries: slice z of expert e writes
    // plane e + z*L, so the S partials are contiguous for the merge kernel.
    int splitk = 1;
  };
  struct Params {
    GemmUniversalMode mode;
    ProblemShape problem_shape;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    KernelHardwareInfo hw_info;
    TileSchedulerParams scheduler;
    void* workspace;
    int probe = 0;
    int const* group_M = nullptr;
    int mtiles_uniform = 0;
    int representative_n = 0;
    int splitk = 1;
  };

  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    ProblemShape problem_shapes = args.problem_shape;
    int cu_count = args.hw_info.cu_count;
    if (cu_count <= 0)
      cu_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    KernelHardwareInfo hw_info{args.hw_info.device_id, cu_count};

    // Only the GroupScheduler needs workspace. The mixed-input vectorized epilogue needs none (its
    // get_workspace_size is (ProblemShape,Args) and returns 0 for this non-persistent path).
    void* scheduler_workspace = workspace;

    constexpr uint32_t NumEpilogueSubTiles = 1;
    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(
        problem_shapes, TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace, NumEpilogueSubTiles);

    // [Q2] The mixed-input collective + epilogue want ONE (M,N,K,L) shape (K,N uniform across experts; L=groups)
    // to compute scale_k=K/gs, the (N,scale_k,L) scale layout, and the L-strided D. Build a representative
    // rank-4 shape from the group shapes so those L-strided planes span all experts.
    auto host0 = problem_shapes.get_host_problem_shape(0);
    int const rep_m = args.representative_m > 0 ? args.representative_m : int(get<0>(host0));
    int const rep_n = args.representative_n > 0 ? args.representative_n : int(get<1>(host0));
    int const rep_k = args.representative_k > 0 ? args.representative_k : int(get<2>(host0));
    auto rep_mnkl = cute::make_shape(rep_m, rep_n, rep_k, problem_shapes.groups());

    return {
      args.mode,
      problem_shapes,
      CollectiveMainloop::to_underlying_arguments(rep_mnkl, args.mainloop, /*workspace=*/nullptr),
      CollectiveEpilogue::to_underlying_arguments(rep_mnkl, args.epilogue, /*workspace=*/nullptr),
      hw_info, scheduler, workspace, args.probe, args.group_M, args.mtiles_uniform, rep_n,
      args.splitk < 1 ? 1 : args.splitk
    };
  }

  static bool can_implement(Arguments const& args) {
    return args.mode == GemmUniversalMode::kGrouped || args.mode == GemmUniversalMode::kArray;
  }
  static int get_workspace_size(Arguments const& args) {
    // GroupScheduler workspace only (epilogue needs none on this path).
    size_t s = TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
        args.scheduler, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, NumMmaWarpGroups);
    return int(round_nearest(s, MinWorkspaceAlignment));
  }
  static cutlass::Status initialize_workspace(Arguments const&, void* = nullptr, hggcStream_t = nullptr, HostAdapter* = nullptr) {
    return Status::kSuccess;
  }

  // NON-PERSISTENT grid: one block per (m_tile, n_tile, expert), like the standard mixed-input kernel
  // (ppu_aiu_gemm_mixed_input.hpp, ~49% MFU). The persistent GroupScheduler launched only grid=(72,1,1) = #CU
  // blocks and ran tiles serially -> acu measured 2 active warps/CU, 3.1% achieved occupancy, 16% CU throughput.
  // Thousands of blocks let the HW hide per-tile load/epilogue latency across blocks. X uses Mmax over experts;
  // experts with fewer M-tiles early-exit in-kernel. (N uniform across experts.)
  // FLAT non-persistent grid: gridDim.x = SUM_e ceil(M_e/TM) (NOT Mmax*L) -> no per-expert padding, so ragged
  // experts launch ZERO idle blocks. gridDim.y = N-tiles. blockIdx.x is decoded to (expert, local m_tile) in
  // operator() by a per-expert m-tile prefix scan (L is small). (Earlier grid was (ceil(Mmax/TM), N/TN, L),
  // which over-launched idle blocks for small experts.)
  static dim3 get_grid_shape(Params const& params) {
    int const L = params.problem_shape.groups();
    int const TM = cute::size<0>(TileShape{}), TN = cute::size<1>(TileShape{});
    int const S = params.splitk < 1 ? 1 : params.splitk;
    if (params.mtiles_uniform > 0) {
      return dim3(params.mtiles_uniform, int(cute::ceil_div(params.representative_n, TN)), L * S);
    }
    int total_m_tiles = 0, N = 1, mt0 = -1, mt_max = 0; bool uni = true;
    for (int e = 0; e < L; ++e) {
      auto ps = params.problem_shape.get_host_problem_shape(e);
      int const mte = int(cute::ceil_div(int(cute::get<0>(ps)), TM));
      total_m_tiles += mte;  N = int(cute::get<1>(ps));  mt_max = mte > mt_max ? mte : mt_max;
      if (mt0 < 0) mt0 = mte; else if (mte != mt0) uni = false;
    }
    int const Nt = int(cute::ceil_div(N, TN));
    bool const force3d = std::getenv("MOEG_FORCE3D") != nullptr;   // diagnostic: force 3D Mmax grid for ragged
    // UNIFORM (or forced): 3D grid (mt_max, N_tiles, L), blockIdx.z=expert, O(1) decode. small experts' extra
    // m-tiles early-exit (idle). RAGGED default: flat grid (total,N,1), no idle blocks but O(L) decode scan.
    // THE SLICE LIVES ON z. The ragged (flat) path leaves z unused, so it is free there; the uniform path already
    // spends z on the expert, so the two are packed as z = expert*S + slice -- the same packing
    // ppu_aiu_gemm_parallel.hpp uses for (l, slice).
    if (uni || force3d) return dim3(uni ? mt0 : mt_max, Nt, L * S);
    return dim3(total_m_tiles, Nt, S);
  }
  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTLASS_DEVICE typename TileScheduler::WorkTileInfo
  fetch_next_work(typename TileScheduler::WorkTileInfo& work_tile_info, TileScheduler& scheduler) const {
    if (scheduler.continue_current_work(work_tile_info)) return work_tile_info;
    scheduler.advance_to_next_work();
    return scheduler.get_current_work();
  }

  CUTLASS_DEVICE void
  operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    int thread_idx = int(threadIdx.x);
    auto blk_shape = TileShape{};
    int const num_groups = params.problem_shape.groups();   // L extent for the collective's B/S/Z slice

    // FLAT non-persistent: decode blockIdx.x -> (expert, local m_idx) via a per-expert m-tile prefix scan
    // (num_groups is small). Every block maps to a REAL tile of some expert -> no idle blocks (vs the old
    // Mmax-sized (.,.,L) grid). l_coord = expert selects this expert's A/B/scale/zero plane in the collective.
    int const TM = int(size<0>(blk_shape));
    int const S = params.splitk < 1 ? 1 : params.splitk;
    int flat = int(blockIdx.x), n_idx = int(blockIdx.y);
    int expert = -1, m_idx = 0, slice = 0;
    if (params.mtiles_uniform > 0) {
      // 3D grid (M-tile bound, N_tiles, L) -> expert = blockIdx.z (O(1)), m_idx = blockIdx.x. The normal host
      // path selects it for uniform shapes; device-only callers may supply Mmax and let the guard below trim it.
      // z = expert*S + slice
      expert = int(blockIdx.z) / S;  slice = int(blockIdx.z) % S;  m_idx = flat;
    } else {
      // A (O(log L) ragged): binary-search the m-tile prefix [L+1] (written by launch into the scheduler-unused
      // workspace) for the largest e with prefix[e] <= flat. Replaces the O(L) linear scan (the ~5% the flat
      // grid lost at high expert count -- avg ~64 iters at L=128 -> now ~7).
      slice = int(blockIdx.z);        // flat path: z carries the slice alone
      int const* pfx = reinterpret_cast<int const*>(params.workspace);
      int lo = 0, hi = num_groups;
      while (lo + 1 < hi) { int const mid = (lo + hi) >> 1; if (pfx[mid] <= flat) lo = mid; else hi = mid; }
      expert = lo;  m_idx = flat - pfx[lo];
    }
    if (expert < 0 || expert >= num_groups) return;            // guard (grid.x is exactly SUM_e mtiles_e)
    auto pe = append<4>(params.problem_shape.get_problem_shape(expert), Int<1>{});  // ONE struct read for N,K
    int const M = int(get<0>(pe)), N = int(get<1>(pe)), K = int(get<2>(pe));
    if (m_idx * TM >= M) return;                               // idle m-tile (3D Mmax grid: small experts); no-op for flat
    if (n_idx * int(size<1>(blk_shape)) >= N) return;          // N uniform, guard anyway

    auto problem_shape_MNKL = make_shape(M, N, K, num_groups);
    auto blk_coord_mnkl = make_coord(m_idx, n_idx, _, expert);

    CollectiveMainloop collective_mainloop;
    auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, blk_coord_mnkl, params.mainloop);
    Tensor gA = get<0>(load_inputs);
    Tensor gB = get<1>(load_inputs);

#if defined(PPU_METADATA_ADDR_PROBE) && (PPU_METADATA_ADDR_PROBE != 0)
    if (params.probe == 2) {
      // Only the three boundary experts write.  Every other CTA exits before
      // the GEMM, so this mode is an address probe rather than a numerical arm.
      // Spell the three values in the device expression itself.  A namespace-
      // scope constexpr array is ODR-used by indexing here, and NVCC's device
      // pass then requires a separately emitted device definition.  This
      // diagnostic must not acquire a hidden device-global ABI just to select
      // three experts.
      int const probe_expert_slot =
          expert == 127 ? 0 : expert == 128 ? 1 : expert == 129 ? 2 : -1;
      if (probe_expert_slot < 0) return;

      auto* trace = reinterpret_cast<GroupedMetadataAddressProbe*>(params.workspace);
      if (trace == nullptr || trace->magic != kGroupedMetadataProbeMagic ||
          trace->version != kGroupedMetadataProbeVersion) return;

      if (thread_idx == 0) {
        atomicAdd(trace->expert_ctas + probe_expert_slot, 1u);
      }

      // gZ below is a CTA-local tile.  The dedicated harness deliberately
      // makes all 256 experts one row and N exactly one TileN so the address
      // census is one CTA per expert with no ragged-prefix/workspace alias.
      // Refuse to reinterpret another geometry as evidence: an incomplete
      // trace is a probe failure, not a clean metadata path.
      bool const supported_geometry =
          params.mtiles_uniform > 0 && M == 1 && m_idx == 0 &&
          N == int(size<1>(TileShape{})) && n_idx == 0 && S == 1 &&
          params.mainloop.scale_k % int(CollectiveMainloop::Scale_TileK) == 0;
      if (!supported_geometry) {
        if (thread_idx == 0) atomicAdd(&trace->configuration_errors, 1u);
        return;
      }

      // `load_inputs` is the exact tuple the production mainloop consumes.
      // The probe is instantiated only for ScaleZero, so gZ is element 3.
      static_assert(!cute::is_void_v<typename CollectiveMainloop::ElementZero>,
                    "metadata address probe requires a zero plane");
      Tensor gZ = get<3>(load_inputs);
      using Zero = typename CollectiveMainloop::NonVoidElementZero;
      static_assert(sizeof(Zero) == sizeof(uint16_t),
                    "metadata address probe records raw fp16 zero values");

      int64_t const scale_k = params.mainloop.scale_k;
      int64_t const plane_elements = int64_t(N) * scale_k;
      Zero const* explicit_plane = params.mainloop.ptr_Z +
          int64_t(expert) * plane_elements;
      auto const* gz_base = cute::raw_pointer_cast(gZ.data());

      // Shape-only arm.  The logical tensor is (N, ScaleTileK,
      // ceil(scale_k/ScaleTileK)); enumerate every (n,group) independently of
      // the tiled-copy mapping.  If this differs from the int64 GEP, the fault
      // predates partition_S/cp.async.
      for (int64_t linear = thread_idx; linear < plane_elements;
           linear += int64_t(MaxThreadsPerBlock)) {
        int const local_n = int(linear % int64_t(N));
        int const group = int(linear / int64_t(N));
        int const group_in_tile = group % int(CollectiveMainloop::Scale_TileK);
        int const metadata_tile = group / int(CollectiveMainloop::Scale_TileK);
        Zero const* explicit_ptr = explicit_plane + int64_t(group) * int64_t(N) + local_n;
        Zero const* gz_ptr = cute::raw_pointer_cast(&gZ(local_n, group_in_tile, metadata_tile));
        uint32_t const out = atomicAdd(&trace->shape_count, 1u);
        if (out < kGroupedMetadataProbeMaxShapeRecords) {
          auto& rec = trace->shape[out];
          rec.scheduler_expert = expert;
          rec.local_n = local_n;
          rec.scale_group = group;
          rec.thread_idx = thread_idx;
          rec.explicit_addr = reinterpret_cast<uint64_t>(explicit_ptr);
          rec.gz_addr = reinterpret_cast<uint64_t>(gz_ptr);
          rec.gz_base_addr = reinterpret_cast<uint64_t>(gz_base);
          rec.explicit_bits = *reinterpret_cast<uint16_t const*>(explicit_ptr);
          rec.gz_bits = *reinterpret_cast<uint16_t const*>(gz_ptr);
        } else {
          atomicAdd(&trace->overflow, 1u);
        }
      }

      // Production partition and production copy atom, unchanged.  The
      // metadata copy has fewer unique logical thread slots than CTA threads.
      // The shipping ordinary/F=1 collective wraps surplus physical threads
      // onto valid slots (quactlize_mma_mixed_input.hpp); raw CuTe get_slice
      // does NOT wrap.  Reproduce that exact mapping here, or the probe itself
      // creates an out-of-range partition and can manufacture the row
      // divergence it is meant to diagnose.
      constexpr int thread_slots = CollectiveMainloop::ScaleCopyPlan::thread_slots;
      int const copy_slot = thread_idx % thread_slots;
      auto gmem_thr_copy_zero =
          params.mainloop.gmem_tiled_copy_zero.get_slice(copy_slot);
      Tensor tZgZ = gmem_thr_copy_zero.partition_S(gZ);
      Tensor sZ = make_tensor(
          make_smem_ptr(shared_storage.tensors.mainloop.smem_zero.begin()),
          typename CollectiveMainloop::SmemLayoutScale{});
      Tensor tZsZ = gmem_thr_copy_zero.partition_D(sZ);
      int const metadata_tiles = int(size<3>(tZgZ));
      auto first_src = tZgZ(_, _, _, 0);
      auto first_dst = tZsZ(_, _, _, 0);
      constexpr int values_per_thread = CollectiveMainloop::ScaleCopyPlan::values_per_thread;
      static_assert(int(size(first_src)) == values_per_thread,
                    "probe source view must match the shipping metadata copy plan");
      static_assert(int(size(first_dst)) == values_per_thread,
                    "probe destination view must match the shipping metadata copy plan");
      static_assert(thread_slots <= int(MaxThreadsPerBlock),
                    "metadata copy's unique slots must fit the CTA");
      if (probe_expert_slot == 0 && thread_idx == 0) {
        trace->cta_threads = uint32_t(MaxThreadsPerBlock);
        trace->thread_slots = uint32_t(thread_slots);
        trace->metadata_tiles = uint32_t(metadata_tiles);
        trace->values_per_thread = uint32_t(values_per_thread);
      }

      for (int metadata_tile = 0; metadata_tile < metadata_tiles; ++metadata_tile) {
        auto src = tZgZ(_, _, _, metadata_tile);
        auto dst = tZsZ(_, _, _, 0);
        clear(dst);
        __syncthreads();
        copy(params.mainloop.gmem_tiled_copy_zero, src, dst);
        cute::cp_async_fence();
        cute::cp_async_wait<0>();
        __syncthreads();

        if (thread_idx < thread_slots) {
          for (int value_idx = 0; value_idx < values_per_thread; ++value_idx) {
            Zero const* src_ptr = cute::raw_pointer_cast(&src(value_idx));
            Zero const* dst_ptr = cute::raw_pointer_cast(&dst(value_idx));
            uint32_t const out = atomicAdd(&trace->copy_count, 1u);
            if (out < kGroupedMetadataProbeMaxCopyRecords) {
              auto& rec = trace->copy[out];
              rec.scheduler_expert = expert;
              rec.thread_idx = thread_idx;
              rec.copy_slot = copy_slot;
              rec.metadata_tile = metadata_tile;
              rec.value_idx = value_idx;
              rec.partition_addr = reinterpret_cast<uint64_t>(src_ptr);
              rec.destination_addr = reinterpret_cast<uint64_t>(dst_ptr);
              rec.partition_bits = *reinterpret_cast<uint16_t const*>(src_ptr);
              rec.cp_async_bits = *reinterpret_cast<uint16_t const*>(dst_ptr);
            } else {
              atomicAdd(&trace->overflow, 1u);
            }
          }
        }
        __syncthreads();
      }
      return;
    }
#endif

    auto m_max = M - size<0>(gA) * m_idx;
    auto n_max = N - size<0>(gB) * n_idx;
    auto k_res = K - size<1>(gA) * size<2>(gA);
    auto residue_mnk = make_tuple(m_max, n_max, k_res);

    TiledMma tiled_mma;
    Tensor accumulators = make_fragment_like<ElementCompute>(partition_fragment_C(tiled_mma, take<0,2>(blk_shape)));
    clear(accumulators);
    // STRIDED K SPLIT. slice z walks k-tiles z, z+S, z+2S ... so gA/gB are untouched and only the traversal
    // changes. k_tile_count uses ceil, so an indivisible tile count would let the last slice step past the end;
    // the host refuses that case (see moe_grouped_ppu::launch) rather than relying on mainloop predication.
    auto k_tile_iter  = cute::make_splitk_coord_iterator(shape<2>(gA), slice, S);
    int  k_tile_count = (size<2>(gA) + S - 1) / S;

    if (params.probe == 1) {
      // ROUTING PROBE: skip the GEMM, tag every output element with (expert+1). See test_moe_grouped_probe.
      cute::fill(accumulators, ElementCompute(expert + 1));
    } else {
      collective_mainloop(params.mainloop, load_inputs, accumulators, k_tile_iter, k_tile_count, thread_idx, smem_buf);
    }

    // EACH SLICE WRITES ITS OWN D PLANE, e + slice*L, so the S partials are contiguous for the merge. The L
    // EXTENT has to grow with them or the epilogue's own bounds check would reject the shifted coordinate -- the
    // mainloop above already ran with the unshifted expert, which is what selects the right B/scale plane.
    auto problem_shape_epi = make_shape(M, N, K, num_groups * S);
    auto blk_coord_epi     = make_coord(m_idx, n_idx, _, expert + slice * num_groups);
    CollectiveEpilogue epilogue{params.epilogue, shared_storage.tensors.epilogue};
    epilogue(problem_shape_epi, blk_shape, blk_coord_epi, accumulators, tiled_mma, residue_mnk,
             thread_idx, (char*)&shared_storage.tensors.epilogue);
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
