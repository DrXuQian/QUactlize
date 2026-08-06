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

#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/detail/layout.hpp"

#define PPU_MOEG_STR2(x) #x
#define PPU_MOEG_STR(x) PPU_MOEG_STR2(x)

namespace moe_grouped_ppu {
using namespace cute;
using TacticSpace = ppu_tactics::GroupedSpace;

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
          int ArtifactTileK = 0>
using MixedMainloopPolicy = ppu_mixed_policy::MainloopPolicy<QuantOp, BaseSchedule, TileShape, ScaleTileShape,
                                                             WarpShape, Stages, AiuInterleaved, ElementB, PlaneB2,
                                                             ArtifactTileK>;

using GroupShape = cute::Shape<int,int,int>;                            // per-expert [M,N,K]
using GroupProblemShape = cutlass::gemm::GroupProblemShape<GroupShape>;

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
          int ArtifactTileK = 0>
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
            // DECODE A BROADCAST. When every expert has at most ONE row, the A tile's 15 padding rows exist only
            // because TileM >= 16 (every MMA atom is Shape<_16,...>), and their results are discarded by the
            // epilogue's residue mask -- so what they READ is irrelevant. Setting A's m-stride to 0 maps all TM
            // rows of the tile onto the expert's real row: the copy still writes TM*TK elements into shared
            // memory, but reads only TK of them from global, so the other 15 reads hit L1. Same trick as the
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
            bool /*unused, was a_row_broadcast*/ = false) {
  using MainloopPolicy = MixedMainloopPolicy<QuantOp, BaseSchedule, TileShape, ScaleTileShape, WarpShape,
                                              Stages, AiuInterleaved, ElementB, PlaneB2, ArtifactTileK>;
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
  if constexpr (ExpectPackedScale) {
    static_assert(CollectiveMainloop::is_packed_scale,
                  "fully-quantized grouped requires the shared packed-scale mainloop at this tile shape");
  }

  // GroupProblemShape -> hits ppu_aiu_gemm_mixed_input_group.hpp's specialization.
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<GroupProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  // Query the exact instantiated type rather than reconstructing its storage from the public tile coordinates.
  // Packed-unit staging and optional A/scale layouts all contribute to this value. The normal launch shares the
  // same guard, so a caller that forgot to query still cannot poison the context with an oversized block.
  static_assert(!RequireUniversalFallback ||
                    GemmKernel::SharedStorageSize <= ppu_tactics::kBlockSmemBytes,
                "the compiled grouped default must fit one ppu001 block for every admitted shape");
  static_assert(!RequireUniversalFallback || CollectiveMainloop::compact_a_rows == 0,
                "the compiled grouped default must use the unrestricted ordinary-A path");
#if defined(PPU_A_PACK) && (PPU_A_PACK != 0)
  static_assert(!RequireUniversalFallback,
                "PPU_A_PACK is a one-row experiment and cannot be the universal grouped fallback");
#endif
  if constexpr (GemmKernel::SharedStorageSize > ppu_tactics::kBlockSmemBytes) return false;
  // PPU_FORCE_INSTANTIATE: odr-use the kernel's operator() so the WHOLE collective -- mainloop included -- is
  // instantiated by the front end. Without this the local nvcc gate parses the collective but never instantiates it,
  // so every template-DEPENDENT error in the mainloop is invisible: a deliberate undefined symbol there is caught
  // (non-dependent, checked at parse time) while a wrong copy coordinate is not. Two static errors reached the box
  // behind exactly that gap. Costs nothing in a real build (the kernel is instantiated anyway) and is opt-in.
#if defined(PPU_FORCE_INSTANTIATE) && (PPU_FORCE_INSTANTIATE != 0)
  // Taking the address of the __global__ device_kernel<GemmKernel> odr-uses it, which instantiates op(params, smem)
  // and with it the ENTIRE mainloop. (&GemmKernel::operator() does not: it is CUTLASS_DEVICE, and naming it from host
  // code instantiates nothing.) device_kernel is what cutlass::kernel_launch would reference anyway.
  { [[maybe_unused]] void const* _probe = (void const*) cutlass::device_kernel<GemmKernel>; }
#endif

  // MOEG_SMEM=1: report the block's shared-memory budget READ OFF THE TYPES, not recomputed from a formula.
  // Two reasons this exists. (1) Which A path a binary was built with is otherwise invisible in a log -- the perf
  // tag's abcast marker reads an env var, not the macro, so both builds print identically and a timing pair cannot
  // be attributed. (2) "A's smem shrank" is exactly the kind of claim I twice asserted from arithmetic and had the
  // hardware refute; SharedStorageSize and cosize_v<SmemLayoutA> are the objects the compiler actually sized.
  if (std::getenv("MOEG_SMEM")) {
    static bool once = false;
    if (!once) {
      once = true;
      // cosize_v<SmemLayoutA> is the LAYOUT's extent. It equals the allocation only while SharedStorage actually
      // has an A member; a variant that dropped the member but kept the layout printed 'A = 16384 B, 160%' of a
      // 10240 B block.
      constexpr int a_elems = int(cute::cosize_v<typename CollectiveMainloop::SmemLayoutA>);
      constexpr int a_bytes = a_elems * int(sizeof(ElementA));
      std::printf("[moe_grouped] smem/block = %d B  (A = %d B = %d elems, %.0f%%)  A path: %s\n",
                  int(GemmKernel::SharedStorageSize), a_bytes, a_elems,
                  100.0 * a_bytes / double(GemmKernel::SharedStorageSize),
#if defined(PPU_A_PACK) && (PPU_A_PACK != 0)
                  "A in smem, PACKED cubes, cp.async row0 + swzl read"
#else
                  CollectiveMainloop::compact_a_rows > 0
                    ? "A in compact smem rows, plain cp.async + DefaultCopy"
                    : "A in smem via AIU + swzl"
#endif
      );
      std::printf("[moe_grouped]   compact A row capacity = %d\n", CollectiveMainloop::compact_a_rows);
      std::printf("[moe_grouped]   blocks/CU at 256 KB = %d\n", 262144 / int(GemmKernel::SharedStorageSize));
    }
  }

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
  // PPU_A_PACK: A's cubes overlap in smem and only row 0 of each carries data, so rows 1..TileM-1 read a
  // neighbour's bytes. Sound only where those rows are discarded, i.e. Mmax == 1.
#if defined(PPU_A_PACK) && (PPU_A_PACK != 0)
  if (m > 1) {
    if constexpr (!QueryOnly) {
      std::printf("[moe_grouped] PPU_A_PACK requires Mmax <= 1, got %d (packed cubes: only row 0 is real)\n", m);
      ++moeg_fail_count();
    }
    return false;
  }
#endif
  if constexpr (CollectiveMainloop::compact_a_rows > 0) {
    if (m > CollectiveMainloop::compact_a_rows) {
      if constexpr (!QueryOnly) {
        std::printf("[moe_grouped] compact A holds %d rows, got Mmax=%d; select the ordinary A path or a wider compact build\n",
                    CollectiveMainloop::compact_a_rows, m);
        ++moeg_fail_count();
      }
      return false;
    }
  }
#if defined(PPU_A_CPASYNC) && (PPU_A_CPASYNC != 0)
  if constexpr (CollectiveMainloop::compact_a_rows == 0) {
    // Folded and two-plane collectives do not yet implement the compact A reader. Make the inactive macro visible
    // instead of letting a build claim an M-dependent footprint it did not instantiate.
    if constexpr (!QueryOnly) {
      std::printf("[moe_grouped] PPU_A_CPASYNC=%d is unavailable for this selected collective; A remains TileM rows\n",
                  int(PPU_A_CPASYNC));
      ++moeg_fail_count();
    }
    return false;
  }
#endif
  // All remaining work constructs strides/arguments or touches the runtime. The checks above are the only
  // M-dependent properties of the compiled type; N/K/operator-domain checks live in the exported query beside the
  // corresponding ABI guards. Thus this path is host-only and does not need valid device pointers or a PPU context.
  if constexpr (QueryOnly) return true;
  // A's GMEM m-stride is NEVER zeroed here, and the parameter that used to do it is gone. It looked like a way to say
  // 'read one row of A' at decode, where TileM >= 16 against one row per expert makes 15/16 of the tile padding --
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
  args.group_M = group_M;
  // THE K-TILE COUNT MUST DIVIDE BY splitk. The kernel's k_tile_count is a ceil, so an indivisible count lets the
  // last slice step past the end of the coordinate space -- refused here rather than left to mainloop predication.
  {
    int const TKv = int(cute::size<2>(TileShape{}));
    int const kt = (k + TKv - 1) / TKv;
    if (splitk > 1 && kt % splitk != 0) {
      std::printf("[moe_grouped] splitk=%d does not divide the k-tile count %d (K=%d, Block_K=%d)\n",
                  splitk, kt, k, TKv);
      ++moeg_fail_count();
      return false;
    }
  }
  args.splitk = splitk;
  // O(1) decode hint: if every expert has the SAME #m-tiles (ceil(M_e/TM)), the kernel uses blockIdx.z (no scan).
  // MOEG_FORCE3D (diagnostic): force the 3D Mmax grid + blockIdx.z decode even for ragged (small experts idle)
  // to isolate whether the ragged gap is the O(L) scan (jumps -> yes) or load-imbalance (unchanged -> no).
  { int const TMv = int(cute::size<0>(TileShape{}));
    if (group_shapes_host != nullptr) {
      int const mt0 = int(cute::ceil_div(int(cute::get<0>(group_shapes_host[0])), TMv));
      int mt_max = mt0; bool uni = true;
      for (int e = 1; e < L; ++e) {
        int const me = int(cute::ceil_div(int(cute::get<0>(group_shapes_host[e])), TMv));
        if (me > mt_max) mt_max = me;
        if (me != mt0) uni = false;
      }
      bool const force3d = std::getenv("MOEG_FORCE3D") != nullptr;
      args.mtiles_uniform = uni ? mt0 : (force3d ? mt_max : 0);
    } else {
      // A device-only caller cannot read the ragged tile sum without synchronising. Use the caller's M upper
      // bound for a conservative 3D grid; the kernel already rejects tiles beyond each device-resident M_e.
      args.mtiles_uniform = int(cute::ceil_div(m, TMv));
    }
  }
  if (const char* e = std::getenv("MOEG_PROBE")) args.probe = std::atoi(e);   // routing probe (test_moe_grouped_probe)

  Gemm gemm;
  auto st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) { std::printf("[moe_grouped] can_implement: %s\n", cutlassGetStatusString(st)); ++moeg_fail_count(); return false; }
  size_t need = gemm.get_workspace_size(args);
  if (need > workspace_bytes) { std::printf("[moe_grouped] workspace %zu > %zu\n", need, workspace_bytes); ++moeg_fail_count(); return false; }
  if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) { std::printf("[moe_grouped] init failed\n"); ++moeg_fail_count(); return false; }
  // Ragged O(log L) decode: write the m-tile prefix [L+1] into the workspace (the non-persistent kernel does
  // NOT use the scheduler workspace, so it's free). AFTER initialize (no clobber), BEFORE run. Uniform path
  // (mtiles_uniform>0) uses blockIdx.z and ignores this. Blocking copy -> ordered before run; L+1 ints, tiny.
  if (args.mtiles_uniform == 0 && workspace != nullptr && !prefix_ready) {
    int const TMv = int(cute::size<0>(TileShape{}));
    if (group_shapes_host != nullptr) {
      std::vector<int> pfx(L + 1); pfx[0] = 0;
      for (int e = 0; e < L; ++e)
        pfx[e + 1] = pfx[e] + int(cute::ceil_div(int(cute::get<0>(group_shapes_host[e])), TMv));
      hggcMemcpy(workspace, pfx.data(), sizeof(int) * (L + 1), hggcMemcpyHostToDevice);
    }
  }
  gemm.run(stream);
  return true;
}

template <QuantMode QuantOp, int TM, int TN, int TK, int WM, int WN, int Stages,
          class ElementB = cutlass::int4b_t,   // default W4A16; pass cutlass::uint2b_t for W2A16
          class PlaneB2 = void,                // bit-plane concat: 2nd (high) B plane; void = single plane
          bool ExpectPackedScale = false, int ArtifactTileK = TK>
void filter_and_run(const cutlass::half_t* A, const ElementB* B, const cutlass::half_t* scales,
                    const cutlass::half_t* zeros,
                    cutlass::half_t** ptr_D, DStride* stride_D, int const* group_M,
                    int m, int n, int k, int L, int group_size,
                    GroupShape* gsd, GroupShape const* gsh, int const* group_row_offsets,
                    char* ws, size_t ws_bytes, hggcStream_t stream,
                    const PlaneB2* B2 = nullptr, int k_full = -1, bool prefix_ready = false, int splitk = 1,
                    bool /*unused, was a_row_broadcast*/ = false) {
  using TileShape = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;
  using WarpShape = cute::Shape<cute::Int<WM>, cute::Int<WN>, cute::Int<TK>>;
  const bool il = (n % 256 == 0 && k % 256 == 0);
  // Artifact folds come from ArtifactTileK; TK here is TacticTileK and changes only the consumer geometry. A larger
  // tactic keeps the resident physical (N/F, F*K) descriptors instead of re-deriving F and switching providers.
  #define MOEG_CALL(SCH, STK, IL) launch<QuantOp, SCH, TileShape, cute::Shape<cute::Int<TN>, STK>, WarpShape, Stages, IL, ElementB, PlaneB2, ExpectPackedScale, false, false, ArtifactTileK>( \
      A,B,scales,zeros,ptr_D,stride_D,group_M,m,n,k,L,group_size,gsd,gsh,group_row_offsets,ws,ws_bytes,stream,B2,k_full,prefix_ready,splitk,false)
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
    else if (group_size == 16) { constexpr int SK=ppu_group_schedule::scale_groups_v<TK,16>;  MOEG_FG(ppu_group_schedule::FinegrainedSchedule<16>,  SK); }  // gs=16 (Q2_K/Q3_K/Q6_K) reuses the Gs32 tag; real grouping via SK=ceil(TK/16)+FINE. Needs TK=128 (SK=8=TK/16 cap).
    else std::printf("[moe_grouped] gs %d unsupported\n", group_size);
  } else {
    if (il) MOEG_CALL(cutlass::gemm::KernelAiuMultistageMixedInputPerCol, cute::_1, true);
    else    MOEG_CALL(cutlass::gemm::KernelAiuMultistageMixedInputPerCol, cute::_1, false);
  }
  #undef MOEG_CALL
}

} // namespace moe_grouped_ppu
