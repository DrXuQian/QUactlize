// STEP 2a launcher: grouped mixed-input GEMM via the GroupScheduler (ppu_aiu_gemm_mixed_input_group.hpp).
// DEGENERATE / uniform-M for now (mainloop args = step-1 batched: single L-strided A/B/S base; A sliced by
// l_coord). Purpose: prove the GroupProblemShape + GroupScheduler + mixed-input collective type stack compiles
// and runs. Ragged A (step 2b) adds group_row_offsets to the collective; nothing here changes structurally.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <type_traits>
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "fold_traits.hpp"
#include "ppu_group_schedule.hpp"
#include "ppu_tactic_space.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "quactlize_actlize.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
#include "ppu_mixed_policy.hpp"
#include "cutlass/epilogue/collective/builders/ppu_builder.inl"
#include "ppu_aiu_gemm_mixed_input_group.hpp"   // the new grouped mixed-input GemmUniversal specialization
#include "actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_group_persistent.hpp"

#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/detail/layout.hpp"

#define PPU_MOEG_STR2(x) #x
#define PPU_MOEG_STR(x) PPU_MOEG_STR2(x)

namespace moe_grouped_ppu {
using namespace cute;
using TacticSpace = ppu_tactics::GroupedSpace;

#if defined(PPU_MOE_PERSISTENT) && (PPU_MOE_PERSISTENT != 0)
inline constexpr bool kPersistentBuild = true;
#else
inline constexpr bool kPersistentBuild = false;
#endif

// Public per-expert output-stride element type for the ptr-array (contiguous) epilogue. RowMajor D, and the
// BATCH stride is static _0 (the epilogue indexes ptr_D[l] per expert, so the batch/L stride is unused) --
// this must match CollectiveEpilogue::StrideD's element exactly (Stride<long,_1,_0>). Callers build a
// DeviceAllocation<DStride> of L entries, one make_cute_packed_stride(DStride{}, {M_e,N,1}) each.
using DStride = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;

using QuantMode = ppu_mixed_policy::QuantMode;
using ppu_mixed_policy::has_zero;
using ppu_mixed_policy::is_finegrained;

template <QuantMode QuantOp, class BaseSchedule, class TileShape, class ScaleTileShape, class WarpShape,
          int Stages, bool AiuInterleaved, class ElementB = cutlass::int4b_t, class PlaneB2 = void,
          int ArtifactTileK = 0, int BChunk = 0>
using MixedMainloopPolicy = ppu_mixed_policy::MainloopPolicy<QuantOp, BaseSchedule, TileShape, ScaleTileShape,
                                                             WarpShape, Stages, AiuInterleaved, ElementB, PlaneB2,
                                                             ArtifactTileK, BChunk>;

using GroupShape = cute::Shape<int,int,int>;                            // per-expert [M,N,K]
using GroupProblemShape = cutlass::gemm::GroupProblemShape<GroupShape>;

// Optional, non-owning event pair used by a benchmark to bracket the selected scheduler path.  The historical
// non-persistent path brackets ONLY Gemm::run(); the directory-persistent path deliberately includes the one-CTA
// directory build immediately followed by Gemm::run(), because that build is part of its end-to-end scheduler cost.
// Ownership stays with the
// caller: a timed batch needs one pair per launch so all per-launch spans can be queried after one final device
// synchronisation rather than serialising the workload launch by launch. `recorded` is a fail-closed handshake --
// can_implement/initialize may reject before either event exists in the stream, and such a pair must never be
// queried or substituted with a host-wall number.
struct KernelSpanEvents {
  hggcEvent_t start{};
  hggcEvent_t stop{};
  bool recorded = false;
};

// LAUNCHES THAT DID NOT HAPPEN MUST BE VISIBLE TO THE CALLER. launch() returns void and reports failure by printf, so a
// harness that times it measures an empty call -- the MoE bench ranked several `init failed` rows as its FASTEST configs at
// 3.17 us, which is 6.6 TB/s against a 2.77 TB/s HBM peak. A counter costs nothing and needs no signature change through the
// twenty-odd callers of filter_and_run.
inline int& moeg_fail_count() { static int c = 0; return c; }

// group_shapes_dev/host: L entries of [M_e,N,K]. A/B/scales single L-strided bases (2a uniform). L=num_experts.
template <QuantMode QuantOp, class BaseSchedule,
          class TileShape, class ScaleTileShape, class WarpShape, int Stages, bool AiuInterleaved,
          class ElementB = cutlass::int4b_t,   // default W4A16; pass cutlass::uint2b_t for W2A16
          class PlaneB2 = void,                // bit-plane concat: 2nd (high) B plane; void = single plane
          bool ExpectPackedScale = false,
          bool QueryOnly = false, bool RequireUniversalFallback = false,
          int ArtifactTileK = 0,
          bool UsePersistent = false,
          class MainloopPolicyOverride = void,
          int BChunk = 0>
bool launch(const cutlass::half_t* A, const ElementB* B, const cutlass::half_t* scales,
            const cutlass::half_t* zeros,
            cutlass::half_t** ptr_D,        // device [L] per-expert output base pointers (contiguous: D+offs[e]*N)
            DStride* stride_D,              // device [L] per-expert output strides ({M_e,N,1} row-major)
            int const* group_M,             // device [L] per-expert M_e (cheap decode of blockIdx.x)
            int m, int n, int k, int L, int group_size,
            GroupShape* group_shapes_dev, GroupShape const* group_shapes_host,
            int const* group_row_offsets,   // ragged: per-expert cumulative A row start; null=uniform
            char* workspace, size_t workspace_bytes, hggcStream_t stream,
            const PlaneB2* B2 = nullptr,     // bit-plane concat: 2nd (high) plane; ignored when PlaneB2 is void
            // SPLIT-K SLICING. `k` is this launch's slice; `k_full` is the whole K the buffers were built
            // for. Every STRIDE has to come from k_full while every PROBLEM SHAPE comes from k: the A row
            // pitch, the B batch pitch and the scale row pitch all belong to the undivided matrix, and only
            // the pointers move. -1 means "not sliced", i.e. k_full == k.
            int k_full = -1,
            // RAGGED PREFIX ALREADY IN THE WORKSPACE. It depends only on the per-expert M values, which K-slicing
            // does not touch, so S slices would each redo the same write -- and it is a BLOCKING hggcMemcpy, which
            // serialises the host and with it any attempt to overlap the slices. The caller writes it once.
            bool prefix_ready = false,
            // IN-KERNEL SPLIT-K: slice the K loop across gridDim.z so all S slices are resident in ONE launch.
            // ptr_D/stride_D must then hold L*splitk entries (slice z of expert e writes plane e + z*L).
            int splitk = 1,
            // DECODE A BROADCAST (retired argument, retained for ABI stability). The original m16 path had 15
            // padding rows per one-row expert; m8 reduces the logical excess to seven but still stages a physical
            // 16-row AIU cube. The experiment mapped those padding rows onto the expert's real row, but the
            // collective now ignores this argument because the coordinate-addressed swizzle made the trick unsafe.
            // Same historical idea as the
            // stride-0 scale broadcast this codebase already uses.
            //
            // ONLY LEGAL WHEN Mmax == 1, and the distinction matters because the FREEDOM is more general than
            // this trick. The don't-care applies to every padding row, i.e. whenever M_e < TileM; but stride 0
            // collapses ALL TileM rows onto row 0, which is only the same thing when there is exactly one real
            // row. At M_e = 3 it would map rows 1 and 2 -- which ARE real -- onto row 0 and compute the wrong
            // answer. Refused above Mmax == 1 rather than silently applied.
            //
            // The general way to spend that freedom is neither this nor a clamp: because a grouped A is GATHERED,
            // expert e owns rows [off_e, off_e + M_e), so an UNPREDICATED read of off_e .. off_e+TileM-1 merely
            // spills into expert e+1's rows and those results are masked anyway. All it needs is the A allocation
            // padded by TileM rows, which buys a uniform fully-vectorised copy for every expert shape. That is a
            // change to the collective's copy, not to a stride, so it is not done here.
            bool /*unused, was a_row_broadcast*/ = false,
            KernelSpanEvents* kernel_span = nullptr,
            uint32_t persistent_grid_ctas_override = 0) {
  if (kernel_span != nullptr) kernel_span->recorded = false;
  using DefaultMainloopPolicy = MixedMainloopPolicy<
      QuantOp, BaseSchedule, TileShape, ScaleTileShape, WarpShape,
      Stages, AiuInterleaved, ElementB, PlaneB2, ArtifactTileK, BChunk>;
  using MainloopPolicy = std::conditional_t<
      std::is_void_v<MainloopPolicyOverride>, DefaultMainloopPolicy,
      MainloopPolicyOverride>;
  using ElementA = typename MainloopPolicy::ElementA;
  using ElementC = cutlass::half_t;  using LayoutC = cutlass::layout::RowMajor;
  using ElementD = cutlass::half_t;  using LayoutD = cutlass::layout::RowMajor;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using ElementAccumulator = float;  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ClusterShape = WarpShape;
  // Ptr-array (grouped) epilogue -> per-expert output pointers ptr_D[l], contiguous by construction (like
  // example 11 / DeepGemm). POINTER layouts (LayoutC*/LayoutD*) signal grouped to the builder. Scalar alpha/beta
  // (array epilogue supports scalar: ThreadEpilogueOp(params.thread, l_coord), collective:221-224).
  using EpilogueSchedule = cutlass::epilogue::EpiloguePtrArraySimtVectorized;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::PPU0010, OperatorClass, TileShape, ClusterShape, EpilogueTileType,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC*, AlignmentC,
      ElementD, LayoutD*, AlignmentD,
      EpilogueSchedule,
      cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>>::CollectiveOp;
  using CollectiveMainloop = typename MainloopPolicy::CollectiveOp;
  static_assert(
      cute::size<0>(typename CollectiveEpilogue::SmemLayout{}) ==
          cute::size<0>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}) *
              cute::size<1>(typename CollectiveMainloop::TiledMma::ThrLayoutVMNK{}),
      "grouped epilogue M layout must match the mainloop's selected MMA instruction and M-warps");
  if constexpr (ExpectPackedScale) {
    static_assert(CollectiveMainloop::is_packed_scale,
                  "fully-quantized grouped requires the shared packed-scale mainloop at this tile shape");
  }

  // Keep the original GemmUniversal specialization as an exact control.  The
  // persistent branch changes only the tile driver; both instantiate these same
  // CollectiveMainloop and CollectiveEpilogue types.
  using NonPersistentKernel =
      cutlass::gemm::kernel::GemmUniversal<GroupProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using PersistentKernel =
      cutlass::gemm::kernel::GroupPersistentMixedInputKernel<
          GroupProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using GemmKernel = std::conditional_t<UsePersistent, PersistentKernel, NonPersistentKernel>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  // Query the exact instantiated type rather than reconstructing its storage from the public tile coordinates.
  // Packed-unit staging and optional A/scale layouts all contribute to this value. The normal launch shares the
  // same guard, so a caller that forgot to query still cannot poison the context with an oversized block.
  static_assert(!RequireUniversalFallback || ppu_tactics::fits_block_smem(
                    GemmKernel::SharedStorageSize),
                "the compiled grouped default must fit one ppu001 block for every admitted shape");
  static_assert(!RequireUniversalFallback || MainloopPolicy::PackedARows == 0,
                "a bounded packed-A provider cannot be the universal grouped fallback");
  if constexpr (!ppu_tactics::fits_block_smem(
                    GemmKernel::SharedStorageSize)) return false;

  using StrideA = typename GemmKernel::StrideA;  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename CollectiveEpilogue::StrideC;  using StrideD = typename CollectiveEpilogue::StrideD;
  using StrideS = typename CollectiveMainloop::StrideScale;

  // Grouped ptr-array epilogue: StrideD/StrideC are POINTER types (per-expert stride arrays from the caller).
  static_assert(std::is_same_v<DStride, cute::remove_pointer_t<StrideD>>,
                "caller DStride must match CollectiveEpilogue::StrideD element type");

  // same fold factor rule as filter_and_run (AIU needs a >=32B contiguous-K run: TK*bits/8 >= 32)
  static constexpr int MOEG_BITS  = MainloopPolicy::LowBits;
  static constexpr int MOEG_RUN_B = cute::size<2>(TileShape{}) * MOEG_BITS / 8;
  static constexpr int MOEG_FOLD  = MainloopPolicy::ArtifactLowFold;
  static_assert(ppu_mixed_policy::kernel_policy_valid_v<TacticSpace, MainloopPolicy>);
  if (k_full <= 0) k_full = k;
  if (splitk < 1) splitk = 1;
  const int scale_k      = (k + group_size - 1) / group_size;        // this slice
  const int scale_k_full = (k_full + group_size - 1) / group_size;   // the whole matrix
  // STRIDES FROM k_full, SHAPES FROM k -- see the k_full parameter. Getting this backwards makes every
  // slice after the first walk gmem with a shrunken row pitch and read the wrong rows entirely.
  StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k_full, L));
  if constexpr (MainloopPolicy::PackedARows > 0) {
    static_assert(MainloopPolicy::PackedARows == 1,
                  "the first typed packed-A provider remains the exact M==1 path");
    if (m != MainloopPolicy::PackedARows) {
      if constexpr (!QueryOnly) ++moeg_fail_count();
      return false;
    }
  }
  // All remaining work constructs strides/arguments or touches the runtime. The checks above are the only
  // M-dependent properties of the compiled type; N/K/operator-domain checks live in the exported query beside the
  // corresponding ABI guards. Thus this path is host-only and does not need valid device pointers or a PPU context.
  if constexpr (QueryOnly) return true;
  // A's GMEM m-stride is NEVER zeroed here, and the parameter that used to do it is gone. It looked like a way to say
  // 'read one row of A' at decode, where the original TileM=16 path made 15/16 of a one-row expert padding --
  // but AiuDesc::init (cute/arch/copy_aiu_base.hpp) takes dim_w, the row PITCH, from exactly that stride, while
  // dim_h, the row EXTENT, comes from the problem's M. So it produced a descriptor claiming TileM rows spaced zero
  // bytes apart, which is malformed, and the kernel returned NaN.
  //
  // Nothing was lost by deleting it. The grouped kernel passes the PER-EXPERT M
  // (ppu_aiu_gemm_mixed_input_group.hpp:243), so dim_h is already 1 at decode, and the instruction is
  // ppu.cp.async.aiu.bulk.tensor...padz... -- the AIU already fetches exactly one row per k-tile and zero-fills the
  // rest of the cube. 'Load one row' is the default. It is why A's whole chain costs 0.15 + 0.26 instructions per
  // mma, 0.6% of the ~66 each mma carries.
  // N-FOLD: a folded weight buffer is physically (N/FoldF) rows x (FoldF*K) codes -- one physical row carries TWO
  // logical N columns -- so its ROW PITCH is FoldF*K, not K. Computing the stride from (n,k) makes the kernel walk
  // gmem with the unfolded pitch and read scrambled bytes no matter how the offline placed them (this is why three
  // structurally different placements all measured ~72% = random).
  StrideB sB = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape(n / MOEG_FOLD, k_full * MOEG_FOLD, L));
  StrideS sS = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(n, scale_k_full, L));
  // C/D strides now come from the caller (per-expert ptr_D + stride_D arrays) -> contiguous output.

  GroupProblemShape ps; ps.num_groups = L; ps.problem_shapes = group_shapes_dev; ps.host_problem_shapes = group_shapes_host;
  cutlass::KernelHardwareInfo hw{};   // cu_count auto-queried in to_underlying_arguments

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGrouped,
    ps,
    { A, sA, B, sB, scales, sS, group_size, zeros, group_row_offsets },
    // EVT ptr-array epilogue Arguments = { fusion_args, ptr_C, dC, ptr_D, dD }. Default fusion_args {} =
    // alpha=1, beta=0 (all ptrs null) -> scale-only, no C. (ptr-array always routes EVT: builder use_evt, 306.)
    { {}, (ElementC const**)nullptr, StrideC{}, ptr_D, stride_D },
    hw
  };
  args.representative_m = m;
  args.representative_n = n;
  args.representative_k = k;
  if constexpr (!std::is_void_v<PlaneB2>) {
    args.mainloop.ptr_B2 = B2;
    // The high-plane stride belongs to the resident artifact and must not be re-derived from TacticTileK. Only reuse
    // dB when the two artifact planes genuinely have the same physical row pitch.
    constexpr int ARTIFACT_HIGH_FOLD = MainloopPolicy::ArtifactHighFold;
    if constexpr (ARTIFACT_HIGH_FOLD != MOEG_FOLD) {
      args.mainloop.dB2 = cutlass::make_cute_packed_stride(
          StrideB{}, cute::make_shape(n / ARTIFACT_HIGH_FOLD, k_full * ARTIFACT_HIGH_FOLD, L));
      args.mainloop.dB2_valid = true;
    }
  }
  constexpr int TMv = int(cute::size<0>(TileShape{}));
  constexpr int TNv = int(cute::size<1>(TileShape{}));
  quactlize::moe_directory::View directory{};
  if constexpr (UsePersistent) {
    if (splitk != 1) {
      std::printf("[moe_grouped persistent] splitk=%d is a separate scheduler axis; first port requires S=1\n",
                  splitk);
      ++moeg_fail_count();
      return false;
    }
    directory = quactlize::moe_directory::make_view(workspace, workspace_bytes, m, L, TMv);
    if (group_M == nullptr || directory.header == nullptr) {
      std::printf("[moe_grouped persistent] directory workspace unavailable: need=%zu have=%zu experts=%d Mmax=%d TM=%d\n",
                  quactlize::moe_directory::workspace_bytes(m, L, TMv), workspace_bytes, L, m, TMv);
      ++moeg_fail_count();
      return false;
    }
    args.directory_header = directory.header;
    args.directory_entries = directory.entries;
    args.logical_work_upper = uint64_t(directory.capacity) * uint64_t(cute::ceil_div(n, TNv));
    args.ctas_per_cu = Gemm::maximum_active_blocks();
    if (args.ctas_per_cu <= 0) {
      std::printf("[moe_grouped persistent] exact kernel occupancy query failed: %d\n", args.ctas_per_cu);
      ++moeg_fail_count();
      return false;
    }
    args.grid_ctas_override = persistent_grid_ctas_override;
    args.splitk = 1;
  } else {
    args.group_M = group_M;
    // THE K-TILE COUNT MUST DIVIDE BY splitk. The kernel's k_tile_count is a ceil, so an indivisible count lets the
    // last slice step past the end of the coordinate space -- refused here rather than left to mainloop predication.
    int const TKv = int(cute::size<2>(TileShape{}));
    int const kt = (k + TKv - 1) / TKv;
    if (splitk > 1 && kt % splitk != 0) {
      std::printf("[moe_grouped] splitk=%d does not divide the k-tile count %d (K=%d, Block_K=%d)\n",
                  splitk, kt, k, TKv);
      ++moeg_fail_count();
      return false;
    }
    args.splitk = splitk;
    // O(1) decode hint: if every expert has the SAME #m-tiles (ceil(M_e/TM)), the kernel uses blockIdx.z (no scan).
    if (group_shapes_host != nullptr) {
      int const mt0 = int(cute::ceil_div(int(cute::get<0>(group_shapes_host[0])), TMv));
      bool uni = true;
      for (int e = 1; e < L; ++e) {
        int const me = int(cute::ceil_div(int(cute::get<0>(group_shapes_host[e])), TMv));
        if (me != mt0) uni = false;
      }
      args.mtiles_uniform = uni ? mt0 : 0;
    } else {
      // A device-only caller cannot read the ragged tile sum without synchronising. Use the caller's M upper
      // bound for a conservative 3D grid; the kernel already rejects tiles beyond each device-resident M_e.
      args.mtiles_uniform = int(cute::ceil_div(m, TMv));
    }
    if (const char* e = std::getenv("MOEG_PROBE")) args.probe = std::atoi(e);
  }

  Gemm gemm;
  auto st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) { std::printf("[moe_grouped] can_implement: %s\n", cutlassGetStatusString(st)); ++moeg_fail_count(); return false; }
  size_t need = gemm.get_workspace_size(args);
  if (need > workspace_bytes) { std::printf("[moe_grouped] workspace %zu > %zu\n", need, workspace_bytes); ++moeg_fail_count(); return false; }
  if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) { std::printf("[moe_grouped] init failed\n"); ++moeg_fail_count(); return false; }
  if constexpr (!UsePersistent) {
    // Ragged O(log L) decode: write the m-tile prefix [L+1] into the workspace (the non-persistent kernel does
    // NOT use the scheduler workspace, so it's free). AFTER initialize (no clobber), BEFORE run. Uniform path
    // (mtiles_uniform>0) uses blockIdx.z and ignores this. Blocking copy -> ordered before run; L+1 ints, tiny.
    if (args.mtiles_uniform == 0 && workspace != nullptr && !prefix_ready) {
      if (group_shapes_host != nullptr) {
        std::vector<int> pfx(L + 1); pfx[0] = 0;
        for (int e = 0; e < L; ++e)
          pfx[e + 1] = pfx[e] + int(cute::ceil_div(int(cute::get<0>(group_shapes_host[e])), TMv));
        hggcMemcpy(workspace, pfx.data(), sizeof(int) * (L + 1), hggcMemcpyHostToDevice);
      }
    }
  }
  // THE MEASURED INTERVAL STARTS HERE. initialize is launcher/setup work and the ragged prefix above is a
  // blocking H2D copy; both dominated the old host timer at decode. The persistent branch has no host prefix and
  // includes its device directory build below. The device span can still include launch scheduling and idle time,
  // so it is an upper bound on a profiler's kernel-only duration, not a synonym for it.
  if (kernel_span != nullptr) {
    hggcError_t const err = hggcEventRecord(kernel_span->start, stream);
    if (err != hggcSuccess) {
      std::printf("[moe_grouped] start-event record failed: %s\n", hggcGetErrorString(err));
      ++moeg_fail_count();
      return false;
    }
  }
  if constexpr (UsePersistent) {
    // Directory construction is part of the end-to-end scheduler cost and is
    // therefore deliberately inside the event interval.  It is ordered before
    // GEMM by the same stream and introduces no host synchronisation.
    if (!quactlize::moe_directory::launch_build<TMv>(
            group_M, group_row_offsets, m, L, directory, stream)) {
      std::printf("[moe_grouped persistent] directory launch arguments rejected\n");
      ++moeg_fail_count();
      return false;
    }
  }
  auto const run_status = gemm.run(stream);
  if (run_status != cutlass::Status::kSuccess) {
    std::printf("[moe_grouped] run failed: %s\n",
                cutlassGetStatusString(run_status));
    ++moeg_fail_count();
    return false;
  }
  if (kernel_span != nullptr) {
    hggcError_t const err = hggcEventRecord(kernel_span->stop, stream);
    if (err != hggcSuccess) {
      std::printf("[moe_grouped] stop-event record failed: %s\n", hggcGetErrorString(err));
      ++moeg_fail_count();
      return false;
    }
    kernel_span->recorded = true;
  }
  return true;
}

template <QuantMode QuantOp, int TM, int TN, int TK, int WM, int WN, int Stages,
          class ElementB = cutlass::int4b_t,   // default W4A16; pass cutlass::uint2b_t for W2A16
          class PlaneB2 = void,                // bit-plane concat: 2nd (high) B plane; void = single plane
          bool ExpectPackedScale = false, int ArtifactTileK = TK,
          int BChunk = 0>
void filter_and_run(const cutlass::half_t* A, const ElementB* B, const cutlass::half_t* scales,
                    const cutlass::half_t* zeros,
                    cutlass::half_t** ptr_D, DStride* stride_D, int const* group_M,
                    int m, int n, int k, int L, int group_size,
                    GroupShape* gsd, GroupShape const* gsh, int const* group_row_offsets,
                    char* ws, size_t ws_bytes, hggcStream_t stream,
                    const PlaneB2* B2 = nullptr, int k_full = -1, bool prefix_ready = false, int splitk = 1,
                    bool /*unused, was a_row_broadcast*/ = false,
                    KernelSpanEvents* kernel_span = nullptr) {
  using TileShape = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;
  using WarpShape = cute::Shape<cute::Int<WM>, cute::Int<WN>, cute::Int<TK>>;
  const bool il = (n % 256 == 0 && k % 256 == 0);
  // Artifact folds come from ArtifactTileK; TK here is TacticTileK and changes only the consumer geometry. A larger
  // tactic keeps the resident physical (N/F, F*K) descriptors instead of re-deriving F and switching providers.
  #define MOEG_CALL(SCH, STK, IL) launch<QuantOp, SCH, TileShape, cute::Shape<cute::Int<TN>, STK>, WarpShape, Stages, IL, ElementB, PlaneB2, ExpectPackedScale, false, false, ArtifactTileK, kPersistentBuild, void, BChunk>( \
      A,B,scales,zeros,ptr_D,stride_D,group_M,m,n,k,L,group_size,gsd,gsh,group_row_offsets,ws,ws_bytes,stream,B2,k_full,prefix_ready,splitk,false,kernel_span)
  // COLLECTIVE CONSTRAINT (SK = Scale_TileK = ceil(TK/gs) = #scale groups per K-tile):
  //   Scale_TileK <= mma_K_atoms (= TK/16), i.e. gs >= 16, so each scale group covers >=1 mma atom. The collective
  //   applies scale per mma-atom (FINE path) when a B copy step (64-K) straddles >1 group, so gs < the copy-step K
  //   (e.g. gs=32) now works -- the old "gs>=64 / SK<=2" limit is gone. gs=32/TK=64 (SK=2) validated vs the
  //   dequant golden; larger SK is structurally supported.
  //   TK=32 DOES compile, contrary to what this comment claimed for a long time. Verified by building the folded
  //   configurations that the delivery bound allows -- i4 (64,128,32) w64x32, i2 (64,128,32) w64x64 and q6 (64,128,32)
  //   w64x64 -- all with zero errors through the front end that DOES fire the collective's static_asserts (an
  //   over-delivering row trips fold_traits.hpp, and WarpM > TileM trips gemm_operands.hpp). The reason the old claim
  //   looked plausible is that B's smem K-extent is FoldF*TK, not TK: at TK=32 int2 folds by 4, giving a 128-element run,
  //   so the >=64 requirement lands on the FOLDED extent and is satisfied. Formats with an int1 plane still cannot reach
  //   TK=32, but the reason is the delivery bound needing WarpN>=128 and that config's accumulator alone wanting 256
  //   registers per thread -- not this.
  #define MOEG_FG(SCH, SK) do { \
      if constexpr ((SK) <= (TK/16)) { if (il) MOEG_CALL(SCH, cute::Int<SK>, true); else MOEG_CALL(SCH, cute::Int<SK>, false); } \
      else std::printf("[moe_grouped] gs=%d + TK=%d -> SK=%d > TK/16=%d UNSUPPORTED (gs<16); use dequant->bf16\n", group_size, TK, (SK), TK/16); \
    } while (0)
  if constexpr (is_finegrained(QuantOp)) {
    if (group_size == 128)     { constexpr int SK=ppu_group_schedule::scale_groups_v<TK,128>; MOEG_FG(ppu_group_schedule::FinegrainedSchedule<128>, SK); }
    else if (group_size == 64) { constexpr int SK=ppu_group_schedule::scale_groups_v<TK,64>;  MOEG_FG(ppu_group_schedule::FinegrainedSchedule<64>,  SK); }
    else if (group_size == 32) { constexpr int SK=ppu_group_schedule::scale_groups_v<TK,32>;  MOEG_FG(ppu_group_schedule::FinegrainedSchedule<32>,  SK); }  // FIXED (per-mma-atom FINE scale)
    else if (group_size == 16) { constexpr int SK=ppu_group_schedule::scale_groups_v<TK,16>;  MOEG_FG(ppu_group_schedule::FinegrainedSchedule<16>,  SK); }  // gs=16 (Q2_K/Q3_K/Q6_K), real grouping via SK=ceil(TK/16)+FINE. Needs TK=128 (SK=8=TK/16 cap).
    else std::printf("[moe_grouped] gs %d unsupported\n", group_size);
  } else {
    if (il) MOEG_CALL(cutlass::gemm::KernelAiuMultistageMixedInputPerCol, cute::_1, true);
    else    MOEG_CALL(cutlass::gemm::KernelAiuMultistageMixedInputPerCol, cute::_1, false);
  }
  #undef MOEG_CALL
}

} // namespace moe_grouped_ppu
