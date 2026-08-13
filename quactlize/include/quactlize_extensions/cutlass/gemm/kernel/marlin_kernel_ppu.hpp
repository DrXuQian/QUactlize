/***************************************************************************************************
 * Copyright (c) 2026, quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Standalone Marlin kernel/cooperative for PPU.
 *
 * This is deliberately not a compatibility mode of the mixed-input kernel.
 * MarlinCollectivePPU owns the classic packed format, copy cadence, dequant
 * and MMA loop; MarlinSchedulerPPU owns K-fast stripes and q locks; this file
 * owns Marlin's 4->2->1 CTA reduction and ordered fp16 D-chain.
 **************************************************************************************************/
#pragma once

#include <cstdint>
#include <type_traits>

#include <cuda_fp16.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/utils.h"
#include "quactlize_extensions/cutlass/gemm/kernel/marlin_scheduler_ppu.hpp"

namespace cutlass::gemm::kernel {

template <class ProblemShape_, class MarlinCollective_, class CollectiveEpilogue_>
class MarlinKernelPPU {
 public:
  using ProblemShape = ProblemShape_;
  using CollectiveMainloop = MarlinCollective_;
  using CollectiveEpilogue = CollectiveEpilogue_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma = typename CollectiveMainloop::TiledMma;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementA = typename CollectiveMainloop::ElementA;
  using ElementB = typename CollectiveMainloop::ElementB;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using Accumulator = typename CollectiveMainloop::Accumulator;
  using StrideA = typename CollectiveMainloop::StrideA;
  using StrideB = typename CollectiveMainloop::StrideB;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using ElementC = typename CollectiveEpilogue::ElementC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideC = typename CollectiveEpilogue::StrideC;
  using StrideD = typename CollectiveEpilogue::StrideD;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using TileScheduler = marlin::MarlinSchedulerPPU<TileShape, ClusterShape>;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr uint32_t MaxThreadsPerBlock = CollectiveMainloop::Threads;
  // Match classic's __launch_bounds__(256,2).  The fixed 50,176-byte shared
  // store and this bound are also the admitted BPC ceiling for the first target.
  static constexpr uint32_t MinBlocksPerMultiprocessor = 2;
  static constexpr uint32_t MaxBlocksPerCu = 2;
  static constexpr uint32_t WarpKCohorts = CollectiveMainloop::WarpOnK;
  static constexpr uint32_t OutputThreads =
      MaxThreadsPerBlock / WarpKCohorts;
  static constexpr bool IsDenseMarlin = true;
  static constexpr bool IsStandaloneMarlin = true;

  static_assert(cute::rank(ProblemShape{}) == 4,
                "standalone Marlin requires dense <M,N,K,L>");
  static_assert(MaxThreadsPerBlock == 256 && WarpKCohorts == 4 &&
                    OutputThreads == 64,
                "first standalone Marlin target is 1M x 2N x 4K");
  static_assert(std::is_same_v<ElementAccumulator, float> &&
                    std::is_same_v<ElementCompute, float>,
                "Marlin CTA reduction is FP32");
  static_assert(std::is_same_v<ElementD, cutlass::half_t> &&
                    sizeof(ElementD) == sizeof(__half),
                "Marlin D-chain is fp16");

  // The first 16 KiB is enough for classic's reduction tree; the same store
  // then stages 4,352 B of padded row-major output.  Both alias the mainloop
  // only after its final cp.async wait + CTA barrier.
  struct SharedStorage {
    union {
      typename CollectiveMainloop::SharedStorage mainloop;
      alignas(16) unsigned char cooperative[
          sizeof(typename CollectiveMainloop::SharedStorage)];
    } tensors;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static_assert(SharedStorageSize == 50176,
                "standalone Marlin shared ledger drifted from classic");

  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
  };

 private:
  struct alignas(16) Vector128 {
    uint32_t x, y, z, w;
  };

  static KernelHardwareInfo real_hw_info(Arguments const& args) {
    int cu_count = args.hw_info.cu_count;
    if (cu_count <= 0) {
      cu_count = KernelHardwareInfo::query_device_multiprocessor_count(
          args.hw_info.device_id);
    }
    return KernelHardwareInfo{args.hw_info.device_id, cu_count};
  }

  static bool epilogue_contract(Arguments const& args) {
    auto shape = cute::append<4>(args.problem_shape, cute::_1{});
    int64_t const m = int64_t(cute::get<0>(shape));
    int64_t const n = int64_t(cute::get<1>(shape));
    int64_t const l = int64_t(cute::get<3>(shape));
    // C is intentionally unused: beta=0.  D itself is Marlin's ordered fp16
    // partial chain, so only compact row-major D is admitted.
    return args.epilogue.ptr_D != nullptr &&
           args.epilogue.thread.alpha_ptr == nullptr &&
           args.epilogue.thread.beta_ptr == nullptr &&
           float(args.epilogue.thread.alpha) == 1.0f &&
           float(args.epilogue.thread.beta) == 0.0f &&
           int64_t(cute::get<0>(args.epilogue.dD)) == n &&
           int64_t(cute::get<1>(args.epilogue.dD)) == 1 &&
           int64_t(cute::get<2>(args.epilogue.dD)) == m * n && l == 1;
  }

  CUTLASS_DEVICE static int acc_i(int lane, int value) {
    return lane / 4 + (((value >> 2) & 1) << 3);
  }
  CUTLASS_DEVICE static int acc_j(int lane, int value) {
    return lane % 4 + ((value % 4) << 2);
  }

  CUTLASS_DEVICE static void thread_block_reduce(
      Accumulator& accum, SharedStorage& shared) {
    Vector128* sh = reinterpret_cast<Vector128*>(shared.tensors.cooperative);
    int const tid = int(threadIdx.x);
    constexpr int red_off = 2;
    int const red_idx = tid / int(OutputThreads);
    constexpr int red_sh_stride = int(OutputThreads) * 4 * 2;
    constexpr int red_sh_delta = int(OutputThreads);
    int const red_sh_rd = red_sh_stride * red_idx + tid % int(OutputThreads);

    #pragma unroll
    for (int step = red_off; step > 0; step /= 2) {
      if (step <= red_idx && red_idx < 2 * step) {
        #pragma unroll
        for (int n_block = 0; n_block < 4; ++n_block) {
          #pragma unroll
          for (int half = 0; half < 2; ++half) {
            int const chunk = 2 * n_block + half;
            int const value_base = 4 * half;
            int const write = red_sh_delta * chunk +
                              (red_sh_rd - red_sh_stride * step);
            if (step < red_off) {
              float const* peer = reinterpret_cast<float const*>(
                  &sh[red_sh_delta * chunk + red_sh_rd]);
              float const* prior = reinterpret_cast<float const*>(&sh[write]);
              #pragma unroll
              for (int i = 0; i < 4; ++i) {
                accum.fragments[n_block].value[value_base + i] +=
                    peer[i] + prior[i];
              }
            }
            *reinterpret_cast<float4*>(&sh[write]) = make_float4(
                accum.fragments[n_block].value[value_base + 0],
                accum.fragments[n_block].value[value_base + 1],
                accum.fragments[n_block].value[value_base + 2],
                accum.fragments[n_block].value[value_base + 3]);
          }
        }
      }
      __syncthreads();
    }
    if (red_idx == 0) {
      #pragma unroll
      for (int n_block = 0; n_block < 4; ++n_block) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
          int const chunk = 2 * n_block + half;
          int const value_base = 4 * half;
          float const* peer = reinterpret_cast<float const*>(
              &sh[red_sh_delta * chunk + red_sh_rd]);
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
            accum.fragments[n_block].value[value_base + i] += peer[i];
          }
        }
      }
    }
    __syncthreads();
  }

  CUTLASS_DEVICE static void global_handoff(
      Accumulator& accum, Params const& params,
      typename TileScheduler::WorkTileInfo const& work,
      int problem_m, int problem_n) {
    int const tid = int(threadIdx.x);
    if (tid >= int(OutputThreads)) {
      return;
    }
    __half* d = reinterpret_cast<__half*>(params.epilogue.ptr_D);
    int const warp_n = tid / 32;
    int const lane = tid % 32;
    bool const first = TileScheduler::is_first_peer(work);
    bool const last = TileScheduler::is_final_peer(work);
    #pragma unroll
    for (int n_block = 0; n_block < 4; ++n_block) {
      int const n_base = (int(work.N_idx) * 8 + warp_n * 4 + n_block) * 16;
      #pragma unroll
      for (int value = 0; value < 8; ++value) {
        int const row = acc_i(lane, value);
        int const col = n_base + acc_j(lane, value);
        if (row < problem_m && col < problem_n) {
          int64_t const offset = int64_t(row) * problem_n + col;
          if (!first) {
            accum.fragments[n_block].value[value] += __half2float(d[offset]);
          }
          if (!last) {
            d[offset] = __float2half(accum.fragments[n_block].value[value]);
          }
        }
      }
    }
  }

  CUTLASS_DEVICE static void write_result(
      Accumulator const& accum, Params const& params,
      typename TileScheduler::WorkTileInfo const& work,
      int problem_m, int problem_n, SharedStorage& shared) {
    constexpr int kRowStrideHalf = 136;
    constexpr int kVectorsPerRow = 16;
    int const tid = int(threadIdx.x);
    __half* sh = reinterpret_cast<__half*>(shared.tensors.cooperative);
    __syncthreads();
    if (tid < int(OutputThreads)) {
      int const warp_n = tid / 32;
      int const lane = tid % 32;
      #pragma unroll
      for (int n_block = 0; n_block < 4; ++n_block) {
        int const n_base = warp_n * 64 + n_block * 16;
        #pragma unroll
        for (int value = 0; value < 8; ++value) {
          int const row = acc_i(lane, value);
          int const col = n_base + acc_j(lane, value);
          if (row < problem_m) {
            sh[row * kRowStrideHalf + col] =
                __float2half(accum.fragments[n_block].value[value]);
          }
        }
      }
    }
    __syncthreads();

    int const row = tid / kVectorsPerRow;
    int const vector_col = tid % kVectorsPerRow;
    if (row < problem_m) {
      auto const* src = reinterpret_cast<Vector128 const*>(
          sh + row * kRowStrideHalf) + vector_col;
      auto* dst = reinterpret_cast<Vector128*>(params.epilogue.ptr_D) +
                  int64_t(row) * (problem_n / 8) +
                  int64_t(work.N_idx) * kVectorsPerRow + vector_col;
      *dst = *src;
    }
    __syncthreads();
  }

 public:
  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    KernelHardwareInfo hw = real_hw_info(args);
    return {
        args.mode,
        args.problem_shape,
        CollectiveMainloop::to_underlying_arguments(
            args.problem_shape, args.mainloop, nullptr),
        CollectiveEpilogue::to_underlying_arguments(
            args.problem_shape, args.epilogue, nullptr),
        hw,
        TileScheduler::to_underlying_arguments(
            args.problem_shape, hw, args.scheduler, workspace),
    };
  }

  static bool can_implement(Arguments const& args) {
    KernelHardwareInfo hw = real_hw_info(args);
    return args.mode == GemmUniversalMode::kGemm && hw.cu_count > 0 &&
           args.scheduler.blocks_per_cu <= MaxBlocksPerCu &&
           CollectiveMainloop::can_implement(args.problem_shape, args.mainloop) &&
           epilogue_contract(args) &&
           TileScheduler::can_implement(args.problem_shape, hw, args.scheduler);
  }

  static size_t get_workspace_size(Arguments const& args) {
    return TileScheduler::get_workspace_size(
        args.scheduler, args.problem_shape, real_hw_info(args));
  }

  static Status initialize_workspace(
      Arguments const& args, void* workspace = nullptr,
      hggcStream_t stream = nullptr, HostAdapter* host_adapter = nullptr) {
    if (!can_implement(args)) {
      return Status::kErrorInvalidProblem;
    }
    Status status = TileScheduler::initialize_workspace(
        args.scheduler, workspace, stream, args.problem_shape,
        real_hw_info(args), host_adapter);
    if (status != Status::kSuccess) {
      return status;
    }
    return CollectiveEpilogue::initialize_workspace(
        args.problem_shape, args.epilogue, nullptr, stream, host_adapter);
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.scheduler);
  }
  static dim3 get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem_buf);
    auto shape = cute::append<4>(params.problem_shape, cute::_1{});
    int const problem_m = int(cute::get<0>(shape));
    int const problem_n = int(cute::get<1>(shape));
    int const problem_k = int(cute::get<2>(shape));
    int const tid = int(threadIdx.x);

    TileScheduler scheduler(params.scheduler);
    auto work = scheduler.get_current_work();
    // An inactive physical CTA must not manufacture mainloop address state.
    // This mirrors classic's early return before its pointer initialization
    // and makes init-count a meaningful, locally checkable property.
    if (!work.is_valid()) {
      return;
    }
    auto cta_state = CollectiveMainloop::init_cta_state(
        params.mainloop, problem_m, problem_n, problem_k, tid);
    while (work.is_valid()) {
      Accumulator accum;
      // SegmentState contains every q/K-dependent address and ends its
      // lifetime before the cooperative.  CtaState alone survives reduction.
      {
        auto segment = CollectiveMainloop::rebase_segment(cta_state, work);
        CollectiveMainloop::run_segment(
            cta_state, segment, accum, shared.tensors.mainloop);
      }

      thread_block_reduce(accum, shared);
      TileScheduler::acquire_peer_turn(params.scheduler, work, tid);
      global_handoff(accum, params, work, problem_m, problem_n);
      TileScheduler::release_peer_turn(params.scheduler, work, tid);
      if (TileScheduler::is_final_peer(work)) {
        write_result(accum, params, work, problem_m, problem_n, shared);
      }
      work = scheduler.get_next_work(work);
    }
  }
};

}  // namespace cutlass::gemm::kernel
