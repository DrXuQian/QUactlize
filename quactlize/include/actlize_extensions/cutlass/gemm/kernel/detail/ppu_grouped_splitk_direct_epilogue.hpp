// Copyright (c) 2026 Quactlize contributors.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "cutlass/ppu_host_adapter.hpp"
#include "ppu_splitk_direct_accumulator_store.hpp"

namespace cutlass::gemm::kernel::detail {

// Fixed alpha=1/beta=0 FP32 partial publication, not a general fused output
// epilogue. Only the runtime's S>1 floating-point workspace instantiates it.
// The final FP16 output continues through the separate ordered reducer.
template <class LayoutEpilogue>
class GroupedSplitKDirectEpilogue {
 public:
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementC = float;
  using ElementD = float;
  using StrideC = typename LayoutEpilogue::StrideC;
  using StrideD = typename LayoutEpilogue::StrideD;
  using InternalStrideD = cute::remove_pointer_t<StrideD>;
  using SmemLayout = typename LayoutEpilogue::SmemLayout;
  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;
  using ThreadEpilogueOp = typename LayoutEpilogue::ThreadEpilogueOp;
  // Keep the existing resource envelope while isolating the publication
  // change. No shared-memory operation or thread callback executes below.
  using SharedStorage = typename LayoutEpilogue::SharedStorage;
  static_assert(cute::is_same_v<typename LayoutEpilogue::ElementD, float>);
  static_assert(!cute::is_same_v<InternalStrideD, StrideD>);

  // No alpha/beta/fusion arguments can be expressed on this internal edge.
  struct ThreadArguments {};
  struct Arguments {
    ThreadArguments thread{};
    float const** ptr_C = nullptr;
    StrideC dC{};
    float** ptr_D = nullptr;
    StrideD dD{};
  };
  using Params = Arguments;

  template <class Shape>
  static constexpr Params to_underlying_arguments(Shape const&, Arguments const& args, void*) {
    return args;
  }
  template <class Shape>
  static size_t get_workspace_size(Shape const&, Arguments const&, int = 0) { return 0; }
  template <class Shape>
  static Status initialize_workspace(Shape const&, Arguments const&, void*,
      hggcStream_t, HostAdapter* = nullptr) { return Status::kSuccess; }
  template <class Shape>
  CUTLASS_HOST_DEVICE static bool can_implement(Shape const&, Arguments const& args) {
    return args.ptr_C == nullptr && args.ptr_D != nullptr && args.dD != nullptr;
  }

  CUTLASS_HOST_DEVICE
  GroupedSplitKDirectEpilogue(Params const& params, SharedStorage&) : params_(params) {}

  template <class Shape, class Tile, class Coord, class Accumulator, class Mma, class Residue>
  CUTLASS_DEVICE void operator()(Shape shape, Tile tile, Coord coord,
      Accumulator const& accumulators, Mma mma, Residue residue, int thread, char*) {
    using namespace cute;
    // The pointer/stride arrays already include expert + slice*E and that
    // expert's compact row prefix. Never apply the slice or prefix twice.
    int const entry = int(get<3>(coord));
    struct Destination { float* ptr_D; InternalStrideD dD; };
    Destination dst{params_.ptr_D[entry], params_.dD[entry]};
    auto local_shape = make_shape(get<0>(shape),get<1>(shape),get<2>(shape),Int<1>{});
    auto local_coord = make_coord(get<0>(coord),get<1>(coord),_,Int<0>{});
    store_splitk_accumulators_direct(dst,local_shape,tile,local_coord,
        accumulators,mma,residue,0,thread);
  }

 private:
  Params const& params_;
};

} // namespace cutlass::gemm::kernel::detail
