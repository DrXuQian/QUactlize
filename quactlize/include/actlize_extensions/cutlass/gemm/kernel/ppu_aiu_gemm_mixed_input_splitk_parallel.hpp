/***************************************************************************************************
 * Copyright (c) 2026 Quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Fixed, contiguous split-K kernel for the shipping PPU mixed-input mainloop.
 *
 * This is deliberately a named kernel rather than another GemmUniversal specialization.  The
 * shipping S==1 path remains the existing GemmUniversal<..., SplitKSerialScheduler> type; a caller
 * selects this type only for the parallel partial-producing phase.
 *
 * The kernel owns only decomposition and mainloop delivery:
 *   grid = (ceil(M/TM), ceil(N/TN), split_k_slices), with dense v1 requiring L==1
 *   plane = slice (dense v1 is L==1)
 *   slice = one equal-length contiguous interval of absolute K-tile coordinates
 *
 * Every CTA writes its accumulator directly to a distinct FP32 plane using the
 * production MMA's partition_C ownership.  The default SeparateKernelCompletion
 * policy has no semaphore, peer fixup, D read, or final linear combination; a
 * second kernel performs its ordered reduction.  An explicitly selected
 * completion policy may append a post-store actual-last protocol without
 * changing the mainloop or partial ABI.
 *
 * CollectivePartialEpilogue still supplies the pointer/stride Params ABI:
 *   - ElementD is exactly the mainloop accumulator type (FP32 in shipping kernels),
 *   - Arguments/Params and to_underlying_arguments(problem_shape, args, workspace),
 *   - to_underlying_arguments(problem_shape, args, workspace).
 * The historical shared R2S/S2R collective call remains available only under
 * PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE for hash-bound negative controls.
 * Its D tensor must be compact [M,N,S], with stride-S == M*N.  The kernel deliberately does not
 * allocate that buffer implicitly; the two-launch device wrapper owns its lifetime and size.
 **************************************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/ppu_host_adapter.hpp"

#include "cute/tensor.hpp"

#include "actlize_extensions/cutlass/gemm/kernel/ppu_fixed_splitk_last_arriver.hpp"
#include "actlize_extensions/cutlass/gemm/kernel/ppu_fixed_splitk_partition.hpp"
#include "actlize_extensions/cutlass/gemm/kernel/detail/ppu_splitk_direct_accumulator_store.hpp"

namespace cutlass::gemm::kernel {

template <
    class ProblemShape_,
    class CollectiveMainloop_,
    class CollectivePartialEpilogue_,
    class CompletionPolicy_ = fixed_splitk::SeparateKernelCompletion>
class GemmUniversalMixedInputSplitKParallel {
 public:
  using ProblemShape = ProblemShape_;
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectivePartialEpilogue = CollectivePartialEpilogue_;
  using CompletionPolicy = CompletionPolicy_;
  // GemmUniversalAdapter expects this conventional alias.  It names the partial-store epilogue,
  // not the final output phase, which belongs either to the separate reduction kernel or to the
  // caller-selected completion policy.
  using CollectiveEpilogue = CollectivePartialEpilogue;

  static_assert(cute::rank(ProblemShape{}) == 3 || cute::rank(ProblemShape{}) == 4,
                "ProblemShape must be <M,N,K> or <M,N,K,L>");
  static_assert(
      cute::is_base_of_v<KernelAiuMultistageMixedInput,
                         typename CollectiveMainloop::DispatchPolicy::Schedule>,
      "parallel split-K requires the shipping mixed-input mainloop family");
  static_assert(!detail::Has_SwapAB_v<CollectiveMainloop>,
                "dense fixed Split-K v1 does not admit SwapAB partial ABI");

  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma = typename CollectiveMainloop::TiledMma;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementA = typename CollectiveMainloop::ElementA;
  using StrideA = typename CollectiveMainloop::StrideA;
  using ElementB = typename CollectiveMainloop::ElementB;
  using StrideB = typename CollectiveMainloop::StrideB;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using ElementC = typename CollectivePartialEpilogue::ElementC;
  using StrideC = typename CollectivePartialEpilogue::StrideC;
  using ElementD = typename CollectivePartialEpilogue::ElementD;
  using StrideD = typename CollectivePartialEpilogue::StrideD;
  using PartialEpilogueArguments = typename CollectivePartialEpilogue::Arguments;
  using PartialEpilogueParams = typename CollectivePartialEpilogue::Params;

  static_assert(cute::is_same_v<ElementAccumulator,
                                typename CollectivePartialEpilogue::ElementAccumulator>,
                "mainloop and partial epilogue must agree on the accumulator type");
  static_assert(cute::is_same_v<ElementAccumulator, ElementD> &&
                    cute::is_same_v<ElementD, float>,
                "parallel split-K partials must be stored as compact FP32 values");
  static_assert(cute::rank(StrideD{}) == 3,
                "partial D stride must describe compact [M,N,S] planes");

  struct SharedStorage {
    union SharedTensorStorage {
      typename CollectiveMainloop::SharedStorage mainloop;
      typename CollectivePartialEpilogue::SharedStorage partial_epilogue;
      typename CompletionPolicy::SharedStorage completion;
    } tensors;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static constexpr uint32_t MaxThreadsPerBlock = cute::size(TiledMma{});
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    PartialEpilogueArguments partial_epilogue{};
    int split_k_slices{1};
    KernelHardwareInfo hw_info{};
    typename CompletionPolicy::Arguments completion{};
  };

  struct Params {
    GemmUniversalMode mode;
    ProblemShape problem_shape;
    MainloopParams mainloop;
    PartialEpilogueParams partial_epilogue;
    fixed_splitk::Params partition;
    typename CompletionPolicy::Params completion;
  };

 private:
  template <class Mainloop, class Shape, class MainloopArgs>
  static auto mainloop_can_implement(
      Shape const& shape, MainloopArgs const& args, int)
      -> decltype(Mainloop::can_implement(shape, args), bool()) {
    return Mainloop::can_implement(shape, args);
  }

  template <class Mainloop, class Shape, class MainloopArgs>
  static bool mainloop_can_implement(Shape const&, MainloopArgs const&, ...) {
    return true;
  }

  template <class Epilogue, class Shape, class EpilogueArgs>
  static auto epilogue_can_implement(
      Shape const& shape, EpilogueArgs const& args, int)
      -> decltype(Epilogue::can_implement(shape, args), bool()) {
    return Epilogue::can_implement(shape, args);
  }

  template <class Epilogue, class Shape, class EpilogueArgs>
  static bool epilogue_can_implement(Shape const&, EpilogueArgs const&, ...) {
    return true;
  }

  template <class Shape>
  static auto partial_problem_shape(Shape const& shape, int slices) {
    auto mnkl = cute::append<4>(shape, cute::Int<1>{});
    return cute::make_shape(cute::get<0>(mnkl), cute::get<1>(mnkl),
                            cute::get<2>(mnkl), slices);
  }

  template <class Shape>
  static fixed_splitk::Params partition_params(Shape const& shape, int slices) {
    auto mnkl = cute::append<4>(shape, cute::Int<1>{});
    int64_t const m = int64_t(cute::get<0>(mnkl));
    int64_t const n = int64_t(cute::get<1>(mnkl));
    int64_t const k = int64_t(cute::get<2>(mnkl));
    if (m <= 0 || n <= 0 || k <= 0 || int(cute::get<3>(mnkl)) != 1) {
      return fixed_splitk::Params::invalid();
    }
    uint64_t const m_tiles = uint64_t(cute::ceil_div(
        m, int64_t(cute::size<0>(TileShape{}))));
    uint64_t const n_tiles = uint64_t(cute::ceil_div(
        n, int64_t(cute::size<1>(TileShape{}))));
    uint64_t const k_tiles = uint64_t(cute::ceil_div(
        k, int64_t(cute::size<2>(TileShape{}))));
    if (m_tiles == 0 || n_tiles == 0 || k_tiles == 0 ||
        m_tiles > uint64_t((std::numeric_limits<uint32_t>::max)()) ||
        n_tiles > uint64_t((std::numeric_limits<uint32_t>::max)()) ||
        k_tiles > uint64_t((std::numeric_limits<uint32_t>::max)()) ||
        m_tiles > (std::numeric_limits<uint64_t>::max)() / n_tiles) {
      return fixed_splitk::Params::invalid();
    }
    return fixed_splitk::make_params(
        m_tiles * n_tiles, uint32_t(k_tiles), uint32_t(slices));
  }

  static bool compact_partial_abi(Arguments const& args) {
    auto mnkl = cute::append<4>(args.problem_shape, cute::Int<1>{});
    int64_t const m = int64_t(cute::get<0>(mnkl));
    int64_t const n = int64_t(cute::get<1>(mnkl));
    if (m <= 0 || n <= 0 || m > (std::numeric_limits<int64_t>::max)() / n) {
      return false;
    }
    int64_t const mn = m * n;
    auto const& dC = args.partial_epilogue.dC;
    auto const& dD = args.partial_epilogue.dD;
    bool const compact =
        int64_t(cute::get<0>(dC)) == n && int64_t(cute::get<1>(dC)) == 1 &&
        int64_t(cute::get<2>(dC)) == mn &&
        int64_t(cute::get<0>(dD)) == n && int64_t(cute::get<1>(dD)) == 1 &&
        int64_t(cute::get<2>(dD)) == mn;
    constexpr int kElementsPerStore = 128 / cutlass::sizeof_bits<ElementD>::value;
    static_assert(kElementsPerStore == 4,
                  "v1 partial epilogue assumes one 128-bit FP32 store atom");
    return compact && n % kElementsPerStore == 0 &&
           args.partial_epilogue.ptr_C != nullptr &&
           args.partial_epilogue.ptr_D != nullptr &&
           reinterpret_cast<uintptr_t>(args.partial_epilogue.ptr_C) % 16 == 0 &&
           reinterpret_cast<uintptr_t>(args.partial_epilogue.ptr_D) % 16 == 0;
  }

 public:
  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    auto problem_shape = args.problem_shape;
    int const lowering_slices =
        args.split_k_slices > 0 ? args.split_k_slices : 1;
    auto partial_shape = partial_problem_shape(problem_shape, lowering_slices);
    // Do not normalize an invalid S to one: callers that bypass can_implement must still
    // receive an invalid descriptor and therefore a zero launch grid.
    auto partition = partition_params(problem_shape, args.split_k_slices);
    return {
        args.mode,
        problem_shape,
        CollectiveMainloop::to_underlying_arguments(
            args.problem_shape, args.mainloop, workspace),
        CollectivePartialEpilogue::to_underlying_arguments(
            partial_shape, args.partial_epilogue, workspace),
        partition,
        CompletionPolicy::to_underlying_arguments(
            problem_shape, partition, args.completion)};
  }

  static bool
  can_implement(Arguments const& args) {
    if (args.mode != GemmUniversalMode::kGemm || args.split_k_slices < 1) {
      return false;
    }
    auto mnkl = cute::append<4>(args.problem_shape, cute::Int<1>{});
    if (int(cute::get<3>(mnkl)) != 1) {
      return false;  // dense v1; batched output-plane order is deliberately not inferred
    }
    int const slices = args.split_k_slices;
    auto const partition = partition_params(args.problem_shape, slices);
    if (!partition.is_valid()) {
      return false;  // includes supported S, Kt>=S, and the v1 Kt%S contract
    }
    if (!compact_partial_abi(args)) {
      return false;
    }
    if (!CompletionPolicy::can_implement(
            args.problem_shape, partition, args.completion,
            args.partial_epilogue, TileShape{})) {
      return false;
    }
    // Mixed cp.async mainloops prime Stages-1 tiles.  Reject a decomposition that would
    // manufacture a shorter slice instead of relying on an unproved underfilled pipeline path.
    if (int(partition.k_tiles_per_split) < DispatchPolicy::Stages - 1) {
      return false;
    }
    auto partial_shape = partial_problem_shape(args.problem_shape, slices);
    return mainloop_can_implement<CollectiveMainloop>(
               args.problem_shape, args.mainloop, 0) &&
           epilogue_can_implement<CollectivePartialEpilogue>(
               partial_shape, args.partial_epilogue, 0);
  }

  // The compact partial allocation is an explicit output of this phase, not anonymous CUTLASS
  // workspace.  Only private workspace requested by the caller-supplied epilogue is reported here.
  static size_t
  get_workspace_size(Arguments const& args) {
    int const slices = args.split_k_slices > 0 ? args.split_k_slices : 1;
    auto partial_shape = partial_problem_shape(args.problem_shape, slices);
    return CollectivePartialEpilogue::get_workspace_size(
        partial_shape, args.partial_epilogue);
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr,
                       hggcStream_t stream = nullptr,
                       HostAdapter* host_adapter = nullptr) {
    (void)host_adapter;
    int const slices = args.split_k_slices > 0 ? args.split_k_slices : 1;
    auto partial_shape = partial_problem_shape(args.problem_shape, slices);
    return CollectivePartialEpilogue::initialize_workspace(
        partial_shape, args.partial_epilogue, workspace, stream);
  }

  static dim3
  get_grid_shape(Params const& params) {
    if (!params.partition.is_valid()) {
      return dim3(0, 0, 0);
    }
    return dim3(
        cute::size(cute::ceil_div(cute::shape<0>(params.problem_shape),
                                  cute::shape<0>(TileShape{}))),
        cute::size(cute::ceil_div(cute::shape<1>(params.problem_shape),
                                  cute::shape<1>(TileShape{}))),
        params.partition.splits);
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    CUTE_STATIC_ASSERT(is_static<TileShape>::value);
    static_assert(rank(StrideA{}) == 3,
                  "StrideA must be rank-3 [M,K,L]");
    static_assert(rank(StrideB{}) == 3,
                  "StrideB must be rank-3 [N,K,L]");

    auto problem_shape_mnkl = append<4>(params.problem_shape, Int<1>{});
    auto [M, N, K, L] = problem_shape_mnkl;
    (void)L;  // can_implement has already restricted dense v1 to L==1

    SharedStorage& shared_storage =
        *reinterpret_cast<SharedStorage*>(smem_buf);

    uint32_t const slice = uint32_t(blockIdx.z);
    int const m_coord = int(blockIdx.x);
    int const n_coord = int(blockIdx.y);
    int const thread_idx = int(threadIdx.x);

    auto const blk_shape = TileShape{};
    auto const blk_coord_mnkl = make_coord(m_coord, n_coord, _, Int<0>{});

    CollectiveMainloop collective_mainloop;
    auto load_inputs = collective_mainloop.load_init(
        problem_shape_mnkl, blk_coord_mnkl, params.mainloop);
    static_assert(tuple_size_v<decltype(load_inputs)> >= 2,
                  "mixed-input load_init must return at least A and B tensors");
    Tensor gA = get<0>(load_inputs);
    Tensor gB = get<1>(load_inputs);

    // gA/gB describe physical transfer shapes; output residue follows the logical CTA tile.
    auto const m_max_coord = M - size<0>(blk_shape) * m_coord;
    auto const n_max_coord = N - size<1>(blk_shape) * n_coord;
    auto const k_residue = K - size<1>(gA) * size<2>(gA);
    auto const residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);

    TiledMma tiled_mma;
    Tensor accumulators = partition_fragment_C(
        tiled_mma, take<0, 2>(blk_shape));
    clear(accumulators);

    uint64_t const n_tiles = uint64_t(gridDim.y);
    uint64_t const q = uint64_t(m_coord) * n_tiles + uint64_t(n_coord);
    fixed_splitk::FixedSplitKWork const work =
        fixed_splitk::work_for(params.partition, q, slice);
    if (!fixed_splitk::work_matches_params(params.partition, work)) {
      return;
    }
    auto k_tile_iter = make_coord_iterator(
        idx2crd(work.k_begin, shape<2>(gA)), shape<2>(gA));

    collective_mainloop(params.mainloop, load_inputs, accumulators,
                        k_tile_iter, int(work.k_count), thread_idx, smem_buf);

    // The epilogue constructor offsets ptr_C/ptr_D by partial_plane * stride-L.
    // Passing l=0 below prevents a second plane offset.  Its logical shape carries S so
    // bounds and compact plane strides are checked against the actual allocation.
    // Workspace planes are split-major; logical_work_id is queue identity,
    // never a physical partial offset.
    int const plane = int(work.peer_idx);
    auto const partial_shape = make_shape(M, N, K, int(params.partition.splits));
    auto const partial_coord = make_coord(m_coord, n_coord, _, Int<0>{});
#if defined(PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE) && \
    (PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE != 0)
    // Exact historical negative for the PPU raw-bit closure.  The generic
    // output epilogue redistributes FP32 accumulators through shared memory;
    // this is unnecessary for a same-type internal partial workspace and its
    // cross-thread handoff is the device-confirmed corruption seam.
    CollectivePartialEpilogue partial_epilogue{
        params.partial_epilogue, plane};
    partial_epilogue(partial_shape, blk_shape, partial_coord, accumulators,
                     tiled_mma, residue_mnk, thread_idx,
                     reinterpret_cast<char*>(
                         &shared_storage.tensors.partial_epilogue));
#else
    // The mainloop accumulator is already partitioned by the production MMA's
    // exact C ownership.  Store that FP32 fragment directly to the split-major
    // FP32 workspace: no conversion, no shared redistribution, no added fence
    // or synchronization, and the reducer ABI remains unchanged.
    detail::store_splitk_accumulators_direct(
        params.partial_epilogue, partial_shape, blk_shape, partial_coord,
        accumulators, tiled_mma, residue_mnk, plane, thread_idx);
#endif
    CompletionPolicy::after_partial(
        params.completion, work, thread_idx, TileShape{},
        shared_storage.tensors.completion);
  }
};

}  // namespace cutlass::gemm::kernel
