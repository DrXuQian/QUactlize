/***************************************************************************************************
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Actual-last completion policy for the dense fixed Split-K producer.
 **************************************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"

#include "quactlize_extensions/cutlass/gemm/device/ppu_mixed_input_splitk_parallel.hpp"
#include "quactlize_extensions/cutlass/gemm/kernel/ppu_fixed_splitk_completion_protocol.hpp"
#include "quactlize_extensions/cutlass/gemm/kernel/ppu_fixed_splitk_partition.hpp"

namespace cutlass::gemm::kernel::fixed_splitk {

// Default policy: preserve the existing partial-producing kernel exactly and
// let a second kernel consume the workspace.
struct SeparateKernelCompletion {
  struct Arguments {};
  struct Params {};
  struct SharedStorage {};

  template <class ProblemShape, class PartialEpilogueArguments, class TileShape>
  static bool can_implement(
      ProblemShape const&, fixed_splitk::Params const&, Arguments const&,
      PartialEpilogueArguments const&, TileShape const&) {
    return true;
  }

  template <class ProblemShape>
  static Params to_underlying_arguments(
      ProblemShape const&, fixed_splitk::Params const&, Arguments const&) {
    return {};
  }

  template <class TileShape>
  CUTLASS_DEVICE static void after_partial(
      Params const&, FixedSplitKWork const&, int, TileShape const&, SharedStorage&) {}
};

// First fused admission is deliberately narrow: compact M=1 FP16 output and
// full TileN stripes.  The policy is nevertheless a separate kernel template
// axis, so the established SeparateKernel producer remains an executable
// arithmetic oracle and fallback.
template <int ElementsPerAccess = 2>
struct LastArriverM1Fp16Completion {
  static_assert(ElementsPerAccess == 1 || ElementsPerAccess == 2 ||
                    ElementsPerAccess == 4 || ElementsPerAccess == 8);

  struct Arguments {
    float const* partials = nullptr;
    half_t* destination = nullptr;
    int32_t* counters = nullptr;
    int64_t rows = 0;
    int64_t columns = 0;
    int64_t destination_stride = 0;
    uint64_t counter_count = 0;
    int partitions = 0;
  };

  struct Params {
    float const* partials = nullptr;
    half_t* destination = nullptr;
    int32_t* counters = nullptr;
    int64_t columns = 0;
    int64_t partition_stride = 0;
    uint64_t counter_count = 0;
  };

  struct SharedStorage {
    int32_t old_count;
    int32_t published_count;
  };

  static_assert(std::is_standard_layout_v<Params> &&
                    std::is_trivially_copyable_v<Params>);
  static_assert(std::is_trivially_copyable_v<SharedStorage>);
  static_assert(sizeof(Params) == 48 && alignof(Params) == 8 &&
                    offsetof(Params, partials) == 0 &&
                    offsetof(Params, destination) == 8 &&
                    offsetof(Params, counters) == 16 &&
                    offsetof(Params, columns) == 24 &&
                    offsetof(Params, partition_stride) == 32 &&
                    offsetof(Params, counter_count) == 40,
                "fused completion Params ABI drifted");

 private:
  static bool disjoint_ranges(
      void const* lhs, size_t lhs_bytes, void const* rhs, size_t rhs_bytes) {
    uintptr_t const lhs_begin = reinterpret_cast<uintptr_t>(lhs);
    uintptr_t const rhs_begin = reinterpret_cast<uintptr_t>(rhs);
    if (lhs_begin > (std::numeric_limits<uintptr_t>::max)() - lhs_bytes ||
        rhs_begin > (std::numeric_limits<uintptr_t>::max)() - rhs_bytes) {
      return false;
    }
    uintptr_t const lhs_end = lhs_begin + lhs_bytes;
    uintptr_t const rhs_end = rhs_begin + rhs_bytes;
    return lhs_end <= rhs_begin || rhs_end <= lhs_begin;
  }

  CUTLASS_DEVICE static int32_t load_acquire(int32_t* pointer) {
#if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100)
    int32_t value = 0;
    asm volatile(
        "ppu.ld.global.acquire.gpu.b32 %0, [%1];\n"
        : "=r"(value) : "l"(pointer) : "memory");
    return value;
#else
    // Stock CUDA's atomic RMW supplies the acquire side for local compilation.
    // The PPU shipping branch above is bound by the source/codegen gate.
    return atomicAdd(pointer, int32_t(0));
#endif
  }

 public:

  template <class ProblemShape, class PartialEpilogueArguments, class TileShape>
  static bool can_implement(
      ProblemShape const& problem_shape, fixed_splitk::Params const& partition,
      Arguments const& args, PartialEpilogueArguments const& partial,
      TileShape const&) {
    auto mnkl = cute::append<4>(problem_shape, cute::Int<1>{});
    int64_t const rows = int64_t(cute::get<0>(mnkl));
    int64_t const columns = int64_t(cute::get<1>(mnkl));
    constexpr int TileN = int(cute::size<1>(TileShape{}));
    bool const supported_partitions = partition.splits == 2 ||
        partition.splits == 4 || partition.splits == 8;
    if (!partition.is_valid() || !supported_partitions || rows != 1 ||
        columns <= 0 || columns % TileN != 0 ||
        TileN % ElementsPerAccess != 0 || args.rows != rows ||
        args.columns != columns || args.destination_stride != columns ||
        args.partitions != int(partition.splits) ||
        args.counter_count != partition.output_tiles ||
        partition.output_tiles != uint64_t(columns / TileN) ||
        args.partials == nullptr || args.destination == nullptr ||
        args.counters == nullptr ||
        static_cast<void const*>(args.partials) !=
            static_cast<void const*>(partial.ptr_D)) {
      return false;
    }
    if (uint64_t(columns) >
            uint64_t((std::numeric_limits<size_t>::max)() / sizeof(float)) /
                uint64_t(partition.splits)) {
      return false;
    }
    size_t const partial_bytes = size_t(columns) *
        size_t(partition.splits) * sizeof(float);
    size_t const destination_bytes = size_t(columns) * sizeof(half_t);
    size_t const counter_bytes = size_t(args.counter_count) * sizeof(int32_t);
    return reinterpret_cast<uintptr_t>(args.partials) % 128 == 0 &&
        reinterpret_cast<uintptr_t>(args.destination) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(args.counters) % alignof(int32_t) == 0 &&
        disjoint_ranges(
            args.partials, partial_bytes, args.destination, destination_bytes) &&
        disjoint_ranges(
            args.partials, partial_bytes, args.counters, counter_bytes) &&
        disjoint_ranges(
            args.destination, destination_bytes, args.counters, counter_bytes);
  }

  template <class ProblemShape>
  static Params to_underlying_arguments(
      ProblemShape const&, fixed_splitk::Params const&, Arguments const& args) {
    return Params{args.partials, args.destination, args.counters, args.columns,
                  args.rows * args.columns, args.counter_count};
  }

 private:
  template <int Partitions>
  CUTLASS_DEVICE static void reduce_tile(
      Params const& params, FixedSplitKWork const& work, int thread_idx, int tile_n) {
    using FragmentOutput = AlignedArray<half_t, ElementsPerAccess>;
    NumericArrayConverter<half_t, float, ElementsPerAccess,
                          FloatRoundStyle::round_to_nearest>
        convert;
    int64_t const tile_column = int64_t(work.q) * int64_t(tile_n);
    for (int local_column = thread_idx * ElementsPerAccess;
         local_column < tile_n;
         local_column += int(blockDim.x) * ElementsPerAccess) {
      int64_t const column = tile_column + local_column;
      Array<float, ElementsPerAccess> const accumulator =
          cutlass::gemm::device::splitk_parallel::
              reduce_fp32_volatile_fixed_partition_order<
                  ElementsPerAccess, Partitions>(
                  params.partials, params.partition_stride, column);
      Array<half_t, ElementsPerAccess> const converted = convert(accumulator);
      FragmentOutput output;
      CUTLASS_PRAGMA_UNROLL
      for (int lane = 0; lane < ElementsPerAccess; ++lane) {
        output[lane] = converted[lane];
      }
      *reinterpret_cast<FragmentOutput*>(params.destination + column) = output;
    }
  }

 public:
  template <class TileShape>
  CUTLASS_DEVICE static void after_partial(
      Params const& params, FixedSplitKWork const& work, int thread_idx,
      TileShape const&, SharedStorage& shared) {
    // This is the audited actlize PPU last-arriver sequence: every writer
    // completes its partial stores, publishes them before a fetch-old atomic,
    // and only the actual last physical arrival reads the peer planes.
    __syncthreads();
    __threadfence();
    __syncthreads();
    if (thread_idx == 0) {
      shared.old_count = atomicAdd(
          params.counters + work.completion_slot(), int32_t(1));
    }
    __syncthreads();

    uint32_t const old_count = uint32_t(shared.old_count);
    if (!completion_arrival_is_last(old_count, work.peer_count)) {
      return;
    }

    // Fetch the terminal counter with an explicit acquire before consuming
    // other CTAs' weakly-written partial planes.  Volatile below prevents load
    // reuse; it is not being asked to provide memory ordering.
    if (thread_idx == 0) {
      shared.published_count = load_acquire(
          params.counters + work.completion_slot());
    }
    __syncthreads();
    if (uint32_t(shared.published_count) != work.peer_count) {
      return;
    }

    constexpr int TileN = int(cute::size<1>(TileShape{}));
    switch (work.peer_count) {
      case 2: reduce_tile<2>(params, work, thread_idx, TileN); break;
      case 4: reduce_tile<4>(params, work, thread_idx, TileN); break;
      case 8: reduce_tile<8>(params, work, thread_idx, TileN); break;
      default: return;
    }

    // All output stores and peer reads finish before the slot is reusable.
    // A subsequent launch on the same stream cannot overlap this reset.
    __syncthreads();
    if (thread_idx == 0) {
      atomicExch(params.counters + work.completion_slot(), int32_t(0));
    }
  }
};

}  // namespace cutlass::gemm::kernel::fixed_splitk
