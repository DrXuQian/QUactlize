/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief PPU GEMM example with different data types for PPU architecture

    Examples:
      
      Runs the mixed input batched gemm (with batch size 2), converting B to the type of A (mode 0)
      $ ./examples/16_ppu_mixed_dtype_gemm/16_ppu_mixed_dtype_gemm --m=2048 --n=2048 --k=2048 --l=2 --mode=0

      Runs the mixed input gemm, and applies a scaling factor to B before mma (mode 1). Applies a vector of scales to the entire
      matrix (group size is the same as the gemm k dimension).
      $ ./examples/16_ppu_mixed_dtype_gemm/16_ppu_mixed_dtype_gemm --m=4096 --n=5120 --k=8192 --g=8192 --mode=1

      Runs the mixed input gemm, and applies a scaling factor and adds a zero-point to B before mma (mode 2). Uses a group size of 128.
      $ ./examples/16_ppu_mixed_dtype_gemm/16_ppu_mixed_dtype_gemm --m=2048 --n=5120 --k=8192 --g=128 --mode=2
*/

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <utility>

#include "cutlass/cutlass.h"
#include "cutlass/relatively_equal.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include "helper.h"
#include "ppu_group_schedule.hpp"
#include "unfused_weight_dequantize.hpp"

#include "quactlize_actlize.hpp"
#include "quactlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_persistent.hpp"
#include "quactlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_streamk.hpp"
#include "quactlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_marlin.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
#include "ppu_mixed_policy.hpp"
#include "bench_select.hpp"

// Cross-check hook: pull in the grouped mixed-input launcher so we can run it (L=1) on THIS file's own verified
// data via a runtime --xcheck flag. Always included (no compile-time guard): the two-stage PPU host/device
// build made -DMOEG_XCHECK unreliable to propagate to host main(), so we gate at runtime instead.
// ppu_mma_builder.inl is #pragma once, so re-inclusion via moe_grouped_ppu.cuh is safe.
#if !defined(LOWBIT_DENSE_UNIT_BUILD)
#include "moe_grouped_ppu.cuh"
#endif

using namespace cute;

#if defined(DENSE_PERSISTENT_AB) || defined(DENSE_STREAMK_AB) || defined(DENSE_MARLIN_AB)
#define DENSE_SCHEDULER_AB 1
#endif
#if defined(DENSE_SCHEDULER_AB) || defined(DENSE_MARLIN_SWEEP)
// DENSE_SCHEDULER_AB owns the one-row mechanism binaries.  The full Marlin
// sweep is deliberately not one of them (it owns a filtered committed table),
// but it still needs the named-scheduler occupancy/grid/provenance path.
#define DENSE_NAMED_SCHEDULER 1
#endif


// This is just an example, so we use a regular enum so we can compare directly to the command-line int.
enum GemmMode {
  ConvertOnly,
  ScaleOnly,
  ScaleWithZeroPoint
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////
// CHANGED from the stock example: MmaType is half_t, not bfloat16_t. Our W4A16 kernel (marlin_gguf_ppu.cuh)
// is fp16 x int4, so the comparison must be fp16 x int4 too. The example explicitly supports this (it was the
// commented alternative). The dense harness additionally uses the same group-size schedule selector as the
// shipping dense and grouped launchers, so runtime --g selects a matching compile-time scale tile and policy.
using MmaType = cutlass::half_t;
#ifdef BENCH_UINT1
using QuantType = cutlass::uint1b_t;                 // W1A16 perf bench (build: QUANT=uint1 ... ./build.sh)
#ifndef BENCH_TSK
#define BENCH_TSK 256                                 // int1 needs TK%256==0 (AIU 32B min: TK*1/8 % 32 == 0)
#endif
constexpr int TileShapeK = BENCH_TSK;
#elif defined(BENCH_UINT2)
using QuantType = cutlass::uint2b_t;                 // W2A16 perf bench (build: QUANT=uint2 ... ./build.sh)
#ifndef BENCH_TSK
#define BENCH_TSK 128                                 // int2 fp16 B-fragment is 2x int4's (density) -> smaller TK helps
#endif
constexpr int TileShapeK = BENCH_TSK;                // int2 needs TK>=128 (AIU 32B min: TK*2/8 % 32 == 0 -> TK%128==0); sweep 128/256 via TSK=
#else
using QuantType = cutlass::int4b_t;
#ifdef BENCH_TSK
constexpr int TileShapeK = BENCH_TSK;                // apples-to-apples: force int4 to int2's TK (TSK=128) to isolate the tile effect
#else
constexpr int TileShapeK = 128 * 8 / sizeof_bits<MmaType>::value;   // 64 (default: 4bit*64/8 = 32B AIU-legal)
#endif
#endif

// THE ARTIFACT'S TileK, named. TileShapeK is the binary's build-time constant (BENCH_TSK) and, since the
// 2026-08-05 split, that is precisely the ARTIFACT's TileK -- the one that fixes the fold and therefore the bytes
// on disk. Rows carry their own TacticTileK. Spelling it out here because `TileShapeK` now reads like the tactic
// quantity, and two different things under one name is what made a quarantined table read as a dense one.
// The static_assert further down ties this to the generated table's LOWBIT_DENSE_CFG_ARTIFACT_TILEK; it cannot
// be used inside Cfg<> because the .inc is included after that template.
constexpr int kArtifactTileK = TileShapeK;

// A matrix configuration
using         ElementA    = MmaType;                                        // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = QuantType;                                      // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
using         LayoutB_opt = cutlass::layout::ColumnMajorInterleaved<256>;
// using         LayoutB_opt = LayoutB;
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// This example manually swaps and transposes, so keep transpose of input layouts
using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

using ElementZero = MmaType;
using ElementScale = MmaType;
using LayoutScale = cutlass::layout::RowMajor;

// C/D matrix configuration
using         ElementC    = MmaType;                                        // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::RowMajor;                      // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// D matrix configuration
using         ElementD    = ElementC;
using         LayoutD     = LayoutC;
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementCompute      = float;                                          // Element type for epilogue computation
using ArchTag             = cutlass::arch::PPU0010;                         // Tag indicating the minimum CU that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
// TUNING KNOBS. The stock example ships TILE 32x32 / WARP 16x16 / 3 stages -- fine for a correctness demo,
// the default here is the sweep winner 64x64/32x32/s4 (61% MFU). The stock 32x32 tile leaves the MMA pipe
// mostly idle (measured: 25% MFU). Bigger tiles raise reuse and warp count. Overridable at compile time
// (build.sh forwards TILE_M / TILE_N / WARP_M / WARP_N / STAGES from the environment).
#ifndef TILE_M
#define TILE_M 64
#endif
#ifndef TILE_N
#define TILE_N 64
#endif
#ifndef WARP_M
#define WARP_M 32
#endif
#ifndef WARP_N
#define WARP_N 32
#endif
#ifndef STAGES
#define STAGES 4
#endif
using TileShape           = Shape<cute::Int<TILE_M>,cute::Int<TILE_N>,cute::Int<TileShapeK>>;  // Threadblock tile
using WarpShape           = Shape<cute::Int<WARP_M>,cute::Int<WARP_N>,cute::Int<TileShapeK>>;  // Warp tile

using EpilogueSchedule    = cutlass::epilogue::EpilogueSimtVectorized;
using EpilogueTileType    = cutlass::epilogue::collective::EpilogueTileAuto;
using ConvertKernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, WarpShape,
    EpilogueTileType,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule
  >::CollectiveOp;

// ============================================================ MIXED INPUT NO SCALES ============================================================================
// The vendor's no-scale collective supports int4/int8 only. Keep its original int4 control available in the i4
// binary, and use that valid type merely to own the common C/D stride/output aliases in i1/i2 binaries; main rejects
// mode 0 there before instantiating run<GemmConvertOnly>. The per-format tactic path below uses the real ElementB.
using ConvertElementB = cutlass::int4b_t;
constexpr int ConvertAlignmentB = 128 / cutlass::sizeof_bits<ConvertElementB>::value;
using CollectiveMainloopConvertOnly = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ConvertElementB, LayoutB_opt, ConvertAlignmentB,
    ElementAccumulator,
    cute::tuple<TileShape>, WarpShape,
    Int<STAGES>,
    ConvertKernelSchedule
  >::CollectiveOp;

using GemmKernelConvertOnly = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, // Indicates ProblemShape
    CollectiveMainloopConvertOnly,
    CollectiveEpilogue
>;

using GemmConvertOnly = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelConvertOnly>;

// =========================================================== MIXED INPUT WITH SCALES ===========================================================================
// The Scale information must get paired with the operand that will be scaled. In this example, B is scaled so we make a tuple of B's information and the scale information.

// =========================================================== MIXED INPUT WITH SCALES AND ZEROS ==================================================================
// The group size is a runtime option, but it changes the compile-time scale tile and dispatch policy. Instantiate
// the four supported variants once and select them at the host boundary; this preserves the bench's one-binary
// --g contract while making it consume the same schedule ladder as the shipping launchers.
template <int GroupSize>
struct GroupKernels {
  using Schedule = ppu_group_schedule::FinegrainedSchedule<GroupSize>;
  static constexpr int ScaleK = ppu_group_schedule::scale_groups_v<TileShapeK, GroupSize>;
  using ScaleTile = Shape<cute::Int<TILE_N>, cute::Int<ScaleK>>;
  using ScaleOnlyPolicy = ppu_mixed_policy::MainloopPolicy<
      ppu_mixed_policy::QuantMode::FinegrainedScaleOnly, Schedule, TileShape, ScaleTile, WarpShape,
      STAGES, true, ElementB>;
  using ScaleOnlyMainloop = typename ScaleOnlyPolicy::CollectiveOp;
  using ScaleOnlyKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>, ScaleOnlyMainloop, CollectiveEpilogue>;
  using ScaleOnly = cutlass::gemm::device::GemmUniversalAdapter<ScaleOnlyKernel>;

  using ScaleZeroPolicy = ppu_mixed_policy::MainloopPolicy<
      ppu_mixed_policy::QuantMode::FinegrainedScaleZero, Schedule, TileShape, ScaleTile, WarpShape,
      STAGES, true, ElementB>;
  using ScaleZeroMainloop = typename ScaleZeroPolicy::CollectiveOp;
  using ScaleZeroKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>, ScaleZeroMainloop, CollectiveEpilogue>;
  using ScaleZero = cutlass::gemm::device::GemmUniversalAdapter<ScaleZeroKernel>;
};
// =================================================================================================================================================================

// ================================ TACTIC REGISTRY (machete / fpA_intB style) =====================================
// The tile is a compile-time template argument, so autotuning it means compiling a FIXED SET of instantiations
// into ONE binary and dispatching at runtime -- exactly how vLLM Machete (../machete_standalone) and TRT-LLM
// fpA_intB (../fpA_intB_standalone) do it. Cfg<> rebuilds the ScaleOnly (mode-1) type stack with a given
// tile/warp/stages. Generated translation units instantiate the configs behind exported wrappers, and TileCfg
// stores the matching wrapper pointer for runtime selection. This replaces the recompile-per-config sweep.sh.
template <int GroupSize, int TM, int TN, int TK, int WM, int WN, int St>
struct Cfg {
  // TK IS THE ROW'S TacticTileK, not the binary's. Until 2026-08-05 these three used the global TileShapeK,
  // because TileK was a build-time constant that also determined the bytes on disk. It no longer does: the artifact
  // carries its own fold, so a consumer may read the same weights at a different TileK, and the sweep searches
  // it. The ARTIFACT's TileK stays a whole-table constant (kArtifactTileK) and is asserted
  // against the binary below -- one weight file, many readers.
  using CfgTile = Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;
  using CfgScale = Shape<cute::Int<TN>,
      cute::Int<ppu_group_schedule::scale_groups_v<TK, GroupSize>>>;
  using CfgWarp = Shape<cute::Int<WM>, cute::Int<WN>, cute::Int<TK>>;
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, CfgTile, CfgWarp, EpilogueTileType,
      ElementAccumulator, ElementAccumulator, ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD, EpilogueSchedule>::CollectiveOp;
  // THE ARTIFACT'S TileK IS PASSED EXPLICITLY, and this line is the one that makes a multi-TileK table correct
  // rather than merely compilable. MainloopPolicy defaults ArtifactTileK_ to 0, meaning "this tactic also
  // defines the artifact" -- true while TileK was one build-time constant, and silently WRONG once rows carry
  // their own: a TacticTileK=256 row would derive fold_for(bits,256)=1 and read an F=2 artifact as unfolded.
  // The bytes would be there and the pairing would be there, and the numbers would be wrong.
  using Policy = ppu_mixed_policy::MainloopPolicy<
      ppu_mixed_policy::QuantMode::FinegrainedScaleOnly,
      ppu_group_schedule::FinegrainedSchedule<GroupSize>, CfgTile, CfgScale, CfgWarp,
      St, true, ElementB, void, kArtifactTileK>;
  using Main = typename Policy::CollectiveOp;
  static_assert(ppu_mixed_policy::kernel_policy_valid_v<ppu_tactics::DenseSpace, Policy>);
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
#if defined(DENSE_SCHEDULER_AB)
  // A named kernel, not a GemmUniversal specialization: the vendor non-persistent
  // mixed-input kernel remains byte-for-byte the control side of this experiment.
  using PersistentKernel = cutlass::gemm::kernel::PersistentMixedInputKernel<
      Shape<int,int,int,int>, Main, Epi>;
  using PersistentGemm = cutlass::gemm::device::GemmUniversalAdapter<PersistentKernel>;
#endif
#if defined(DENSE_STREAMK_AB) || defined(DENSE_MARLIN_AB)
  // Keep the four-warp Stream-K hard gate in a separate target.  Merely naming
  // this type in 107a's 64/256-thread rows would make their valid persistent
  // controls fail a constraint that belongs only to Stream-K fixup.
  using StreamKKernel = cutlass::gemm::kernel::StreamKMixedInputKernel<
      Shape<int,int,int,int>, Main, Epi>;
  using StreamKGemm = cutlass::gemm::device::GemmUniversalAdapter<StreamKKernel>;
#endif
#if defined(DENSE_MARLIN_AB) || defined(DENSE_MARLIN_SWEEP)
  // Marlin is a third, additive scheduler/cooperative.  It consumes the same
  // mixed-input Main/Epi types as the DP and Stream-K arms; no format or
  // converter type is rebuilt for this scheduler.
  using MarlinKernel = cutlass::gemm::kernel::MarlinMixedInputKernel<
      Shape<int,int,int,int>, Main, Epi>;
  using MarlinGemm = cutlass::gemm::device::GemmUniversalAdapter<MarlinKernel>;
#endif
};

struct Options;
struct Result;
struct TileCfg;
using LowbitDenseWrapper = Result (*)(Options&, TileCfg const&);
struct TileCfg {
  char const* name;
  int tm, tn, tk, wm, wn, st, b_chunk, b_chunk_effective;
  LowbitDenseWrapper wrapper;
};

inline bench_measure::Tactic dense_tactic(TileCfg const& c) {
  return {nullptr, c.tm, c.tn, c.tk, c.wm, c.wn, c.st,
          c.b_chunk, c.b_chunk_effective, false};
}

inline bench_measure::Tactic dense_convert_tactic() {
  return {nullptr, TILE_M, TILE_N, TileShapeK, WARP_M, WARP_N, STAGES, 0, 0, false};
}

#if !defined(LOWBIT_DENSE_UNIT_BUILD)
#if defined(PPU_B_CHUNK)
inline constexpr int kDenseFixedBChunkRequest = PPU_B_CHUNK;
#else
inline constexpr int kDenseFixedBChunkRequest = 0;
#endif

// Fixed (non-generated) scale rows still compile a real Policy. Read its descriptor rather than claiming bc0->0:
// PPU_DEFS=PPU_B_CHUNK=1 reaches this main TU, and the policy may accept or reject that request by format.
template <class Policy>
inline bench_measure::Tactic dense_fixed_tactic() {
  return {nullptr, TILE_M, TILE_N, TileShapeK, WARP_M, WARP_N, STAGES,
          kDenseFixedBChunkRequest, Policy::Descriptor::atom_at_a_time ? 1 : 0, false};
}
#endif

// The generated table. Regenerate with benchmarks/emit_tactic_configs.cpp when the binary's (QUANT, BENCH_TSK)
// changes -- the static_assert below is what turns forgetting into a compile error rather than a sweep over a
// set of tactics this binary cannot select.
// The optional collectives this file INSTANTIATES. quactlize_actlize.hpp carries the base only, so a
// consumer names the specialisation it needs; omitting it makes CollectiveMma incomplete, which the
// compiler reports by naming the exact instantiation.
#include "quactlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_fold.hpp"
#include "quactlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_mixed_input_2plane.hpp"

#if !defined(LOWBIT_DENSE_UNIT_BUILD)
#include "bench_samples.hpp"
#include "bench_floor.cuh"
#if defined(DENSE_MARLIN_SWEEP)
// A second registry over the SAME committed table: CMake emits rows whose
// final Marlin kernel has a warp-aligned CTA cohort in [32,1024].  The named
// kernel independently proves the exact accumulator/cohort binding.  Keep the
// source table's bits/artifact identity, but make both the registry and
// provenance distinct from the ordinary DP sweep.
#if defined(BENCH_UINT1)
#include "lowbit_dense_i1_configs.inc"
#define LOWBIT_DENSE_TABLE_CFG_BITS             LOWBIT_DENSE_I1_CFG_BITS
#define LOWBIT_DENSE_TABLE_CFG_ARTIFACT_TILEK   LOWBIT_DENSE_I1_CFG_ARTIFACT_TILEK
#define LOWBIT_DENSE_MARLIN_SOURCE_ROWS         LOWBIT_DENSE_I1_CFG_ROWS
#define LOWBIT_DENSE_MARLIN_SOURCE_SPACE        LOWBIT_DENSE_I1_CFG_SPACE_FNV1A64
#define LOWBIT_DENSE_MARLIN_SOURCE_EMITTER      LOWBIT_DENSE_I1_CFG_EMITTER_FNV1A64
#define LOWBIT_DENSE_MARLIN_SOURCE_FILE         "lowbit_dense_i1_configs.inc"
#elif defined(BENCH_UINT2)
#include "lowbit_dense_i2_configs.inc"
#define LOWBIT_DENSE_TABLE_CFG_BITS             LOWBIT_DENSE_I2_CFG_BITS
#define LOWBIT_DENSE_TABLE_CFG_ARTIFACT_TILEK   LOWBIT_DENSE_I2_CFG_ARTIFACT_TILEK
#define LOWBIT_DENSE_MARLIN_SOURCE_ROWS         LOWBIT_DENSE_I2_CFG_ROWS
#define LOWBIT_DENSE_MARLIN_SOURCE_SPACE        LOWBIT_DENSE_I2_CFG_SPACE_FNV1A64
#define LOWBIT_DENSE_MARLIN_SOURCE_EMITTER      LOWBIT_DENSE_I2_CFG_EMITTER_FNV1A64
#define LOWBIT_DENSE_MARLIN_SOURCE_FILE         "lowbit_dense_i2_configs.inc"
#else
#include "lowbit_dense_configs.inc"
#define LOWBIT_DENSE_TABLE_CFG_BITS             LOWBIT_DENSE_CFG_BITS
#define LOWBIT_DENSE_TABLE_CFG_ARTIFACT_TILEK   LOWBIT_DENSE_CFG_ARTIFACT_TILEK
#define LOWBIT_DENSE_MARLIN_SOURCE_ROWS         LOWBIT_DENSE_CFG_ROWS
#define LOWBIT_DENSE_MARLIN_SOURCE_SPACE        LOWBIT_DENSE_CFG_SPACE_FNV1A64
#define LOWBIT_DENSE_MARLIN_SOURCE_EMITTER      LOWBIT_DENSE_CFG_EMITTER_FNV1A64
#define LOWBIT_DENSE_MARLIN_SOURCE_FILE         "lowbit_dense_configs.inc"
#endif
#include "lowbit_dense_marlin_sweep_configs.inc"
#define LOWBIT_DENSE_TABLE_FILE                 "scheduler=marlin;source=" LOWBIT_DENSE_MARLIN_SOURCE_FILE
#define LOWBIT_DENSE_TABLE_CFG_ROWS             LOWBIT_DENSE_MARLIN_SWEEP_ROWS
#define LOWBIT_DENSE_TABLE_CFG_SPACE_FNV1A64    LOWBIT_DENSE_MARLIN_SOURCE_SPACE
#define LOWBIT_DENSE_TABLE_CFG_EMITTER_FNV1A64  LOWBIT_DENSE_MARLIN_SOURCE_EMITTER
#define LOWBIT_DENSE_TABLE_CFG_LIST             LOWBIT_DENSE_MARLIN_SWEEP_CONFIGS
#elif defined(DENSE_SCHEDULER_AB)
// Each scheduler A/B target deliberately instantiates exactly one row.  These are mechanism
// experiments, not truncated tactic searches; a separate registry identity prevents a
// one-row result from being mistaken for the winner of the full dense table.
#if defined(DENSE_MARLIN_AB)
#define LOWBIT_DENSE_TABLE_FILE                 "marlin-ab-single-row"
#define LOWBIT_DENSE_TABLE_CFG_SPACE_FNV1A64    "marlin-ab"
#define LOWBIT_DENSE_TABLE_CFG_EMITTER_FNV1A64  "marlin-ab"
#elif defined(DENSE_STREAMK_AB)
#define LOWBIT_DENSE_TABLE_FILE                 "streamk-ab-single-row"
#define LOWBIT_DENSE_TABLE_CFG_SPACE_FNV1A64    "streamk-ab"
#define LOWBIT_DENSE_TABLE_CFG_EMITTER_FNV1A64  "streamk-ab"
#else
#define LOWBIT_DENSE_TABLE_FILE                 "persistent-ab-single-row"
#define LOWBIT_DENSE_TABLE_CFG_SPACE_FNV1A64    "persistent-ab"
#define LOWBIT_DENSE_TABLE_CFG_EMITTER_FNV1A64  "persistent-ab"
#endif
#define LOWBIT_DENSE_TABLE_CFG_BITS             DENSE_AB_BITS
#if defined(DENSE_MARLIN_AB)
#define LOWBIT_DENSE_TABLE_CFG_ARTIFACT_TILEK   DENSE_AB_ARTIFACT_TK
#else
#define LOWBIT_DENSE_TABLE_CFG_ARTIFACT_TILEK   DENSE_AB_TK
#endif
#define LOWBIT_DENSE_TABLE_CFG_ROWS             1
#define LOWBIT_DENSE_TABLE_CFG_LIST(X,A) \
  X(DENSE_AB_TM,DENSE_AB_TN,DENSE_AB_TK,DENSE_AB_WM,DENSE_AB_WN,DENSE_AB_ST,DENSE_AB_BC,A)
#elif defined(BENCH_UINT1)
#include "lowbit_dense_i1_configs.inc"
#define LOWBIT_DENSE_TABLE_FILE                 "lowbit_dense_i1_configs.inc"
#define LOWBIT_DENSE_TABLE_CFG_BITS             LOWBIT_DENSE_I1_CFG_BITS
#define LOWBIT_DENSE_TABLE_CFG_ARTIFACT_TILEK   LOWBIT_DENSE_I1_CFG_ARTIFACT_TILEK
#define LOWBIT_DENSE_TABLE_CFG_ROWS             LOWBIT_DENSE_I1_CFG_ROWS
#define LOWBIT_DENSE_TABLE_CFG_SPACE_FNV1A64    LOWBIT_DENSE_I1_CFG_SPACE_FNV1A64
#define LOWBIT_DENSE_TABLE_CFG_EMITTER_FNV1A64  LOWBIT_DENSE_I1_CFG_EMITTER_FNV1A64
#define LOWBIT_DENSE_TABLE_CFG_LIST             LOWBIT_DENSE_I1_CFG_LIST
#elif defined(BENCH_UINT2)
#include "lowbit_dense_i2_configs.inc"
#define LOWBIT_DENSE_TABLE_FILE                 "lowbit_dense_i2_configs.inc"
#define LOWBIT_DENSE_TABLE_CFG_BITS             LOWBIT_DENSE_I2_CFG_BITS
#define LOWBIT_DENSE_TABLE_CFG_ARTIFACT_TILEK   LOWBIT_DENSE_I2_CFG_ARTIFACT_TILEK
#define LOWBIT_DENSE_TABLE_CFG_ROWS             LOWBIT_DENSE_I2_CFG_ROWS
#define LOWBIT_DENSE_TABLE_CFG_SPACE_FNV1A64    LOWBIT_DENSE_I2_CFG_SPACE_FNV1A64
#define LOWBIT_DENSE_TABLE_CFG_EMITTER_FNV1A64  LOWBIT_DENSE_I2_CFG_EMITTER_FNV1A64
#define LOWBIT_DENSE_TABLE_CFG_LIST             LOWBIT_DENSE_I2_CFG_LIST
#else
#include "lowbit_dense_configs.inc"
#define LOWBIT_DENSE_TABLE_FILE                 "lowbit_dense_configs.inc"
#define LOWBIT_DENSE_TABLE_CFG_BITS             LOWBIT_DENSE_CFG_BITS
#define LOWBIT_DENSE_TABLE_CFG_ARTIFACT_TILEK   LOWBIT_DENSE_CFG_ARTIFACT_TILEK
#define LOWBIT_DENSE_TABLE_CFG_ROWS             LOWBIT_DENSE_CFG_ROWS
#define LOWBIT_DENSE_TABLE_CFG_SPACE_FNV1A64    LOWBIT_DENSE_CFG_SPACE_FNV1A64
#define LOWBIT_DENSE_TABLE_CFG_EMITTER_FNV1A64  LOWBIT_DENSE_CFG_EMITTER_FNV1A64
#define LOWBIT_DENSE_TABLE_CFG_LIST             LOWBIT_DENSE_CFG_LIST
#endif
#include "bench_device.hpp"
static_assert(cutlass::sizeof_bits<QuantType>::value == LOWBIT_DENSE_TABLE_CFG_BITS &&
              TileShapeK == LOWBIT_DENSE_TABLE_CFG_ARTIFACT_TILEK,
              "the selected dense config table was generated for a different (bits, TileK) than this binary. Regenerate "
              LOWBIT_DENSE_TABLE_FILE " with the exact command stamped in its provenance header");
#define LOWBIT_DENSE_COUNT_ROW(TM,TN,TK,WM,WN,ST,BC,_UNUSED) + 1
inline constexpr int kLowbitDenseConfigRows = 0 LOWBIT_DENSE_TABLE_CFG_LIST(LOWBIT_DENSE_COUNT_ROW, );
#undef LOWBIT_DENSE_COUNT_ROW
static_assert(kLowbitDenseConfigRows == LOWBIT_DENSE_TABLE_CFG_ROWS,
              "the selected dense config table's row-count provenance does not match its X-macro list; regenerate it");

inline void print_dense_table_provenance() {
#if defined(DENSE_MARLIN_SWEEP)
  static_assert(LOWBIT_DENSE_MARLIN_SWEEP_SOURCE_ROWS == LOWBIT_DENSE_MARLIN_SOURCE_ROWS,
                "the Marlin filter header and committed source table disagree");
  static_assert(LOWBIT_DENSE_MARLIN_SWEEP_FILTERED_ROWS +
                    LOWBIT_DENSE_MARLIN_SWEEP_ROWS ==
                LOWBIT_DENSE_MARLIN_SWEEP_SOURCE_ROWS,
                "the Marlin source/eligible/filtered census does not close");
  std::printf("[dense-table] scheduler=marlin file=%s rows=%d source_rows=%d "
              "eligible_rows=%d filtered_rows=%d "
              "cohort_capability=warp-aligned-threads-32..1024 "
              "source_space_fnv1a64=%s "
              "source_emitter_fnv1a64=%s\n",
              LOWBIT_DENSE_TABLE_FILE, kLowbitDenseConfigRows,
              LOWBIT_DENSE_MARLIN_SWEEP_SOURCE_ROWS,
              LOWBIT_DENSE_MARLIN_SWEEP_ROWS,
              LOWBIT_DENSE_MARLIN_SWEEP_FILTERED_ROWS,
              LOWBIT_DENSE_TABLE_CFG_SPACE_FNV1A64,
              LOWBIT_DENSE_TABLE_CFG_EMITTER_FNV1A64);
#else
  std::printf("[dense-table] file=%s rows=%d space_fnv1a64=%s emitter_fnv1a64=%s\n",
              LOWBIT_DENSE_TABLE_FILE, kLowbitDenseConfigRows,
              LOWBIT_DENSE_TABLE_CFG_SPACE_FNV1A64, LOWBIT_DENSE_TABLE_CFG_EMITTER_FNV1A64);
#endif
}
#endif
// =================================================================================================================

using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
using StrideC = typename GemmKernelConvertOnly::StrideC;
using StrideD = typename GemmKernelConvertOnly::StrideD;

using StrideC_ref = cutlass::detail::TagToStrideC_t<LayoutC>;
using StrideD_ref = cutlass::detail::TagToStrideC_t<LayoutD>;

//
// Data members
//

/// Initialization
// Scale and Zero share a stride since the layout and shapes must be the same.
using StrideS = cute::Stride<cute::_1, int64_t, int64_t>;
using StrideS_ref = cutlass::detail::TagToStrideB_t<LayoutScale>;

#if defined(LOWBIT_DENSE_UNIT_BUILD)
extern StrideA stride_A;
extern StrideB stride_B;
extern StrideC stride_C;
extern StrideC_ref stride_C_ref;
extern StrideD stride_D;
extern StrideD_ref stride_D_ref;
extern uint64_t seed;
extern StrideS stride_S;
extern StrideS_ref stride_S_ref;
extern cutlass::DeviceAllocation<ElementA> block_A;
extern cutlass::DeviceAllocation<ElementB> block_B;
extern cutlass::HostTensor<QuantType, LayoutB> tensor_B;
extern cutlass::HostTensor<QuantType, LayoutB> block_B_buff;
extern cutlass::DeviceAllocation<ElementA> block_B_dq;
extern cutlass::DeviceAllocation<ElementScale> block_scale;
extern cutlass::DeviceAllocation<ElementZero> block_zero;
extern cutlass::DeviceAllocation<ElementC> block_C;
extern cutlass::DeviceAllocation<typename GemmConvertOnly::EpilogueOutputOp::ElementOutput> block_D;
extern cutlass::DeviceAllocation<typename GemmConvertOnly::EpilogueOutputOp::ElementOutput> block_ref_D;
#else
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideC_ref stride_C_ref;
StrideD stride_D;
StrideD_ref stride_D_ref;
uint64_t seed;
StrideS stride_S;
StrideS_ref stride_S_ref;
cutlass::DeviceAllocation<ElementA> block_A;
cutlass::DeviceAllocation<ElementB> block_B;
cutlass::HostTensor<QuantType, LayoutB> tensor_B;
cutlass::HostTensor<QuantType, LayoutB> block_B_buff;
cutlass::DeviceAllocation<ElementA> block_B_dq;
cutlass::DeviceAllocation<ElementScale> block_scale;
cutlass::DeviceAllocation<ElementZero> block_zero;
cutlass::DeviceAllocation<ElementC> block_C;
cutlass::DeviceAllocation<typename GemmConvertOnly::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename GemmConvertOnly::EpilogueOutputOp::ElementOutput> block_ref_D;
#endif


/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help = false;

  // CHANGED defaults: mode=1 (scale-only == symmetric dense W4A16, what marlin_gguf gs=128 computes; mode=2
  // adds a zero-point, which is the affine path we compare separately), g=128, iterations=100 for a real timing,
  // and the qwen35moe dense shapes at a representative prefill M. --g also supports 16, 32 and 64 at runtime.
  float alpha = 1.0f;
  float beta = 0.0f;
  int iterations = 100;
  int mode = 1;
  int m = 2048, n = 4096, k = 4096;
  int g = 128;
  int l = 1;

  // Tactic controls (machete / fpA_intB style). Empty config => use the tactic cache if given, else default.
  std::string config;            // force one compiled tile, e.g. "64x64:32x32:s4"
  std::string tactic_file;       // load best config for this shape from a cache
  std::string save_tactic_file;  // write the searched winner to a cache
  bool list_configs = false;
  bool search_configs = false;
  bool xcheck = false;           // --xcheck: run the grouped kernel (L=1) on this run's verified data and compare
  bool persistent = false;       // --persistent: 107a A/B target only; select serial persistent work loop
  bool streamk = false;          // --streamk: 107b target only; select deterministic dense Stream-K
  bool marlin = false;           // --marlin: select the independent Marlin stripe scheduler/cooperative
  bool streamk_gate = false;     // --streamk_gate: require the independent CPU-FP32/fixup gate
  bool streamk_split_gate = false; // --streamk_split_gate: require an actually split output tile
  bool streamk_exact_fixture = false; // --streamk_exact_fixture: use the exact-by-construction A0 inputs on every arm

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("l", l);
    cmd.get_cmd_line_argument("g", g);
    cmd.get_cmd_line_argument("mode", mode);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("config", config);
    cmd.get_cmd_line_argument("tactic", tactic_file);
    cmd.get_cmd_line_argument("save_tactic", save_tactic_file);
    list_configs   = cmd.check_cmd_line_flag("list_configs");
    search_configs = cmd.check_cmd_line_flag("search_configs");
    xcheck         = cmd.check_cmd_line_flag("xcheck");
    persistent     = cmd.check_cmd_line_flag("persistent");
    streamk        = cmd.check_cmd_line_flag("streamk");
    marlin         = cmd.check_cmd_line_flag("marlin");
    streamk_gate   = cmd.check_cmd_line_flag("streamk_gate");
    streamk_split_gate = cmd.check_cmd_line_flag("streamk_split_gate");
    streamk_exact_fixture = cmd.check_cmd_line_flag("streamk_exact_fixture");
    if (streamk_gate || streamk_split_gate) streamk = true;
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "16_ppu_mixed_dtype_gemm\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   The number of independent gemm problems with mnk shape\n"
      << "  --g=<int>                   Scale/zero group size (mixed modes support 16, 32, 64 or 128).\n"
      << "  --mode=<int>                The mode to run the gemm. 0 does (A @ B), 1 means A @ (scale * B), 2 means A @ (scale * B + zero-point).\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";
#if defined(DENSE_SCHEDULER_AB)
    out << "  --persistent                Use the dense persistent scheduler A/B arm.\n";
#endif
#if defined(DENSE_STREAMK_AB)
    out << "  --streamk                   Use deterministic dense Stream-K (splits=1).\n"
        << "  --streamk_gate              Also require fixup witness + CPU FP32 golden.\n"
        << "  --streamk_split_gate        Fail closed unless lowered Params contain a real cross-CTA seam.\n"
        << "  --streamk_exact_fixture     Fill the actual A0 arm with sparse integer inputs whose sums are order-independent.\n";
#endif
#if defined(DENSE_MARLIN_AB)
    out << "  --marlin                    Use the independent Marlin CTA-stripe scheduler.\n";
#endif
#if defined(DENSE_MARLIN_SWEEP)
    out << "  scheduler=marlin            Fixed at build time for every compiled table row; "
           "runtime scheduler flags and ordinary dense tactic caches are rejected.\n";
#endif

    out
      << "\n\nExamples:\n\n"
      << "$ " << "16_ppu_mixed_dtype_gemm" << " --m=1024 --n=512 --k=1024 -g 0 --l=10 --alpha=2 --mode=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k * l;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms = 0.0;
  double gflops = 0.0;
  cutlass::Status status = cutlass::Status::kSuccess;
  hggcError_t error = hggcSuccess;
  bool passed = false;
  // A numerical PASS is not evidence about the fixup seam when the scheduler
  // produced no split output tile.  The explicit split-path gate maps that
  // third state to process rc=2 instead of collapsing it into PASS/FAIL.
  bool split_path_exercised = true;
  // A validator that cannot describe an arm has not proved that the arm is
  // numerically wrong.  Keep that third state separate from `passed`: the box
  // driver may continue on to the actual Stream-K subject, while process rc=3
  // prevents NOT CLASSIFIABLE from being consumed as correctness evidence.
  bool verification_classified = true;

};

// Numerical evidence belongs to one invocation of initialize(), not to a log
// file.  Keeping it as a returned POD prevents the exact 64x128x4352 seam
// fixture from being accidentally consumed as evidence for a different A0 arm
// that happens to run later in the same box script.
struct DenseFixtureEvidence {
  bool order_independent = false;
  bool fp16_output_exact = false;
};

struct DenseReplayEvidence {
  bool fixup_closed = false;
  uint64_t split_reference_bitdiff = 0;
  uint64_t non_split_reference_bitdiff = 0;
};

// Published for ci/check_exact_fixture.py.  The A0 fixture below consumes the
// same constants; the checker derives its bounds instead of restating them.
inline constexpr int kExactFixtureNonzerosPerRow = 32;
inline constexpr int kExactFixtureScales[] = {1, 2, 4};
inline constexpr int kExactFixtureZeros[] = {0};

enum class DenseVerifyBucket : uint8_t {
  DataParallel = 0,
  StreamKWhole = 1,
  StreamKSplit = 2,
  Count = 3,
};

struct DenseVerifyPeerRange {
  uint32_t q = 0;
  uint32_t k_begin = 0;
  uint32_t k_count = 0;
  uint32_t unit = 0;
  uint16_t visit = 0;
  uint32_t capture_slot = 0;
};

// Host-only diagnostic map.  It never enters Gemm::Arguments and therefore
// cannot change decomposition or kernel behaviour.  The dedicated 107b binary
// fills one entry per logical output tile from the scheduler's lowered Params;
// verify() then attributes every half output to DP, an unsplit Stream-K tile,
// or a Stream-K tile that crossed the global fixup path.
struct DenseVerifyPartition {
  int tile_m = 0;
  int tile_n = 0;
  int problem_m = 0;
  int problem_n = 0;
  int tiles_m = 0;
  int tiles_n = 0;
  int batches = 0;
  int fixup_threads = 0;
  uint32_t k_tiles_per_output_tile = 0;
  std::vector<DenseVerifyBucket> tile_bucket;
  // One entry per logical output tile.  These make a sparse A0 failure
  // attributable to the scheduler's global q and to the final peer's position
  // inside its persistent work unit, rather than leaving "233 mismatches" as
  // an unlocatable scalar count.
  std::vector<uint32_t> scheduler_q;
  std::vector<uint32_t> final_peer_unit;
  std::vector<uint16_t> final_peer_visit;
  // One entry per local (m,n) output coordinate.  BlockStripedReduce addresses
  // scalar `stripe * threads + lane`; derive that inverse from the actual MMA
  // fragment instead of inventing a lane formula in the diagnostic.
  std::vector<uint16_t> local_lane;
  std::vector<uint16_t> local_stripe;
  // Exact K-ordered peer ranges for split q.  The device capture map is sparse
  // in (q,K_idx) but compact in capture_slot, so A0 needs ~one FP32 tile per
  // real peer instead of q*Kt tiles of debug storage.
  std::vector<uint32_t> split_peer_offsets;
  std::vector<DenseVerifyPeerRange> split_peer_ranges;
  std::vector<int32_t> capture_slot_by_qk;
  uint64_t split_tiles = 0;
  uint64_t peer_excess = 0;
  uint64_t valid_fixup_elements = 0;
  // The basic DP bucket map and the Stream-K replay map have different
  // contracts.  In particular, ordinary DP has no K-peer capture slots.  The
  // c96fe8d gate accidentally required those Stream-K-only fields for every
  // arm and rejected a perfectly ordinary 32x32 logical DP tile grid.
  bool is_streamk = false;
  bool classification_closed = false;
};

#if defined(DENSE_STREAMK_AB)
template <class Gemm>
bool dense_map_accumulator_owners(DenseVerifyPartition& partition) {
  using TiledMma = typename Gemm::GemmKernel::TiledMma;
  using TileShape = typename Gemm::GemmKernel::TileShape;
  using ElementAccumulator = typename Gemm::GemmKernel::ElementAccumulator;
  int const threads = int(Gemm::GemmKernel::MaxThreadsPerBlock);
  int const tile_elements = partition.tile_m * partition.tile_n;
  if (partition.tile_m != int(cute::size<0>(TileShape{})) ||
      partition.tile_n != int(cute::size<1>(TileShape{})) ||
      threads <= 0 || tile_elements <= 0) {
    std::fprintf(stderr,
                 "  [dense verify owners] fail-close: tactic and MMA tile disagree\n");
    return false;
  }

  partition.local_lane.assign(std::size_t(tile_elements), uint16_t(-1));
  partition.local_stripe.assign(std::size_t(tile_elements), uint16_t(-1));
  partition.fixup_threads = threads;
  std::vector<uint8_t> coverage(std::size_t(tile_elements), 0);
  auto identity = cute::make_identity_tensor(cute::take<0, 2>(TileShape{}));
  TiledMma tiled_mma;
  auto accumulators = cute::make_fragment_like<ElementAccumulator>(
      cute::partition_fragment_C(tiled_mma, cute::take<0, 2>(TileShape{})));
  auto physical_to_fragment = cute::right_inverse(accumulators.layout());
  int const stripes = int(cute::size(accumulators));
  if (threads * stripes != tile_elements) {
    std::fprintf(stderr,
                 "  [dense verify owners] fail-close: threads*stripes=%d != tile=%d\n",
                 threads * stripes, tile_elements);
    return false;
  }
  for (int lane = 0; lane < threads; ++lane) {
    auto coordinates = tiled_mma.get_thread_slice(lane).partition_C(identity);
    if (int(cute::size(coordinates)) != stripes) return false;
    for (int stripe = 0; stripe < stripes; ++stripe) {
      auto mn = coordinates(physical_to_fragment(stripe));
      int const m = int(cute::get<0>(mn));
      int const n = int(cute::get<1>(mn));
      if (m < 0 || m >= partition.tile_m || n < 0 || n >= partition.tile_n) {
        return false;
      }
      std::size_t const local = std::size_t(m) * partition.tile_n + n;
      if (coverage[local]++) return false;
      partition.local_lane[local] = uint16_t(lane);
      partition.local_stripe[local] = uint16_t(stripe);
    }
  }
  if (std::find(coverage.begin(), coverage.end(), uint8_t(0)) != coverage.end()) {
    return false;
  }
  std::printf("  [dense verify owners] tile=%dx%d lanes=%d stripes/lane=%d "
              "coverage=exact-once\n",
              partition.tile_m, partition.tile_n, threads, stripes);
  return true;
}

template <class SchedulerParams>
bool dense_classify_streamk_tiles(
    DenseVerifyPartition& partition, SchedulerParams const& sk) {
  uint64_t const tile_count = uint64_t(partition.tiles_m) *
      uint64_t(partition.tiles_n) * uint64_t(partition.batches);
  uint64_t const tiles_per_batch =
      uint64_t(partition.tiles_m) * uint64_t(partition.tiles_n);
  uint64_t const cluster_size = sk.get_cluster_size();
  uint64_t const groups = sk.divmod_sk_groups_.divisor;
  uint64_t const units_per_group = sk.divmod_sk_units_per_group_.divisor;
  uint64_t const k_tiles = sk.divmod_tiles_per_output_tile_.divisor;
  constexpr uint64_t min_k_tiles = SchedulerParams::min_iters_per_sk_unit_;

  auto reject = [](char const* why) {
    std::fprintf(stderr, "  [dense verify partition] fail-close: %s\n", why);
    return false;
  };
  if (partition.tile_bucket.size() != tile_count || tiles_per_batch == 0 ||
      partition.batches <= 0) {
    return reject("invalid logical output-tile map");
  }
  // This bench-local mirror intentionally accepts only the exact geometry that
  // it mirrors below.  General clusters, swizzles, aligned K starts, split-K,
  // and separate reduction must add their own oracle rather than inherit a
  // plausible-looking but wrong DP/SK label.
  if (cluster_size != 1 || sk.divmod_cluster_shape_major_.divisor != 1 ||
      sk.divmod_cluster_shape_minor_.divisor != 1) {
    return reject("only a 1x1 dense scheduler cluster is classified");
  }
  if (sk.log_swizzle_size_ != 0 || sk.ktile_start_alignment_count_ != 1 ||
      sk.divmod_splits_.divisor != 1 || sk.separate_reduction_units_ != 0) {
    return reject("swizzle/aligned-K/split-K/separate-reduction is outside this diagnostic");
  }
  if (groups == 0 || units_per_group == 0 || k_tiles < min_k_tiles ||
      sk.sk_tiles_ == 0 || sk.sk_units_ != groups * units_per_group ||
      groups > sk.sk_tiles_ || sk.sk_tiles_ > tile_count ||
      sk.big_groups_ != sk.sk_tiles_ % groups ||
      sk.divmod_batch_.divisor != tiles_per_batch) {
    return reject("lowered Stream-K group/tile counts are inconsistent");
  }
  uint64_t const expected_major =
      sk.raster_order_ == SchedulerParams::RasterOrder::AlongN
          ? uint64_t(partition.tiles_n) : uint64_t(partition.tiles_m);
  if (sk.divmod_cluster_blk_major_.divisor != expected_major) {
    return reject("raster divisor does not match the logical tile grid");
  }

  // Prove the q <-> (l,m,n) inverse for every output tile before using q as
  // the lock/fixup identity.  With cluster=1 and swizzle=1, AlongN linearizes
  // m-major and AlongM linearizes n-major.
  std::vector<uint8_t> mapped(tile_count, 0);
  auto logical_tile_for_q = [&](uint64_t q, uint64_t& logical) {
    uint64_t const l = q / tiles_per_batch;
    uint64_t const r = q % tiles_per_batch;
    uint64_t m = 0, n = 0;
    if (sk.raster_order_ == SchedulerParams::RasterOrder::AlongN) {
      m = r / uint64_t(partition.tiles_n);
      n = r % uint64_t(partition.tiles_n);
    }
    else {
      n = r / uint64_t(partition.tiles_m);
      m = r % uint64_t(partition.tiles_m);
    }
    if (l >= uint64_t(partition.batches) ||
        m >= uint64_t(partition.tiles_m) || n >= uint64_t(partition.tiles_n)) {
      return false;
    }
    logical = (l * uint64_t(partition.tiles_m) + m) *
        uint64_t(partition.tiles_n) + n;
    return logical < tile_count;
  };
  for (uint64_t q = 0; q < tile_count; ++q) {
    uint64_t logical = 0;
    if (!logical_tile_for_q(q, logical) || mapped[logical]++) {
      return reject("scheduler q does not bijectively cover logical output tiles");
    }
  }
  if (std::find(mapped.begin(), mapped.end(), uint8_t(0)) != mapped.end()) {
    return reject("scheduler q leaves a logical output tile unmapped");
  }

  // Mirror get_current_work_iter_start_possible_update_work_tile_k_remaining()
  // for the accepted cluster=1 case.  Counting interval intersections is
  // equivalent to the device work loop, which consumes those intersections in
  // reverse tile order.  A (q,k) coverage table makes this mirror fail closed
  // if its boundary repair ever drifts from a complete, disjoint partition.
  std::vector<uint16_t> coverage(uint64_t(sk.sk_tiles_) * k_tiles, 0);
  std::vector<uint32_t> peers(sk.sk_tiles_, 0);
  std::vector<uint32_t> final_unit(sk.sk_tiles_, uint32_t(-1));
  std::vector<uint16_t> final_visit(sk.sk_tiles_, uint16_t(-1));
  std::vector<std::vector<DenseVerifyPeerRange>> peer_ranges(sk.sk_tiles_);
  for (uint64_t linear = 0; linear < sk.sk_units_; ++linear) {
    uint64_t const group = linear % groups;
    uint64_t const unit = linear / groups;
    uint64_t const tiles_in_group = sk.sk_tiles_ / groups +
        (group < sk.big_groups_ ? 1u : 0u);
    uint64_t const group_k_tiles = tiles_in_group * k_tiles;
    uint64_t const base = group_k_tiles / units_per_group;
    uint64_t const big_units = group_k_tiles % units_per_group;
    uint64_t start = base * unit + std::min(unit, big_units);
    uint64_t count = base + (unit < big_units ? 1u : 0u);

    uint64_t const start_in_tile = start % k_tiles;
    if (start_in_tile < min_k_tiles) {
      start -= start_in_tile;
      count += start_in_tile;
    }
    else if (start_in_tile > k_tiles - min_k_tiles) {
      uint64_t const adjustment = k_tiles - start_in_tile;
      if (count < adjustment) return reject("start-boundary repair underflowed");
      start += adjustment;
      count -= adjustment;
    }

    uint64_t const initial_end_in_tile = (start + count) % k_tiles;
    if (initial_end_in_tile < min_k_tiles) {
      if (count < initial_end_in_tile) return reject("end-boundary repair underflowed");
      count -= initial_end_in_tile;
    }
    else if (initial_end_in_tile > k_tiles - min_k_tiles) {
      count += k_tiles - initial_end_in_tile;
    }
    if (count == 0 || start + count > group_k_tiles) {
      return reject("repaired work-unit interval is empty or outside its group");
    }

    uint64_t const first_local_tile = start / k_tiles;
    uint64_t const last_local_tile = (start + count - 1) / k_tiles;
    for (uint64_t local = first_local_tile; local <= last_local_tile; ++local) {
      uint64_t const q = local * groups + group;
      if (q >= sk.sk_tiles_) return reject("work-unit interval maps outside SK tiles");
      uint64_t const tile_begin = local * k_tiles;
      uint64_t const begin = std::max(start, tile_begin);
      uint64_t const end = std::min(start + count, tile_begin + k_tiles);
      if (begin >= end) return reject("empty work-unit/output-tile intersection");
      ++peers[q];
      peer_ranges[q].push_back(DenseVerifyPeerRange{
          uint32_t(q), uint32_t(begin - tile_begin), uint32_t(end - begin),
          uint32_t(linear), uint16_t(last_local_tile - local), 0u});
      if (end == tile_begin + k_tiles) {
        if (final_unit[q] != uint32_t(-1)) {
          return reject("output tile has more than one final peer");
        }
        final_unit[q] = uint32_t(linear);
        // Device work walks this unit from its highest local output tile down.
        final_visit[q] = uint16_t(last_local_tile - local);
      }
      for (uint64_t k = begin - tile_begin; k < end - tile_begin; ++k) {
        ++coverage[q * k_tiles + k];
      }
    }
  }

  if (std::find_if(coverage.begin(), coverage.end(),
                   [](uint16_t visits) { return visits != 1; }) != coverage.end()) {
    return reject("(q,k) coverage has a hole or duplicate owner");
  }
  uint64_t split_tiles = 0;
  uint64_t peer_excess = 0;
  uint64_t valid_fixup_elements = 0;
  partition.k_tiles_per_output_tile = uint32_t(k_tiles);
  partition.split_peer_offsets.assign(std::size_t(sk.sk_tiles_) + 1, 0u);
  partition.split_peer_ranges.clear();
  partition.capture_slot_by_qk.assign(
      std::size_t(sk.sk_tiles_) * std::size_t(k_tiles), int32_t(-1));
  partition.scheduler_q.assign(tile_count, uint32_t(-1));
  partition.final_peer_unit.assign(tile_count, uint32_t(-1));
  partition.final_peer_visit.assign(tile_count, uint16_t(-1));
  for (uint64_t q = 0; q < sk.sk_tiles_; ++q) {
    if (peers[q] == 0 || final_unit[q] == uint32_t(-1) ||
        final_visit[q] == uint16_t(-1)) {
      return reject("Stream-K output tile has no unique final owner");
    }
    uint64_t logical = 0;
    if (!logical_tile_for_q(q, logical)) return reject("Stream-K q cannot be decoded");
    bool const split = peers[q] > 1;
    auto& ranges = peer_ranges[q];
    std::sort(ranges.begin(), ranges.end(),
              [](DenseVerifyPeerRange const& a, DenseVerifyPeerRange const& b) {
                return a.k_begin < b.k_begin;
              });
    uint32_t expected_k = 0;
    for (DenseVerifyPeerRange const& range : ranges) {
      if (range.q != q || range.k_count == 0 || range.k_begin != expected_k ||
          uint64_t(range.k_begin) + range.k_count > k_tiles) {
        return reject("K-ordered peer ranges have a hole, overlap, or wrong q");
      }
      expected_k += range.k_count;
    }
    if (ranges.size() != peers[q] || expected_k != k_tiles ||
        ranges.back().unit != final_unit[q] ||
        ranges.back().visit != final_visit[q]) {
      return reject("peer-range chain disagrees with coverage or final owner");
    }
    partition.split_peer_offsets[q] =
        uint32_t(partition.split_peer_ranges.size());
    if (split) {
      for (DenseVerifyPeerRange range : ranges) {
        range.capture_slot = uint32_t(partition.split_peer_ranges.size());
        std::size_t const map_index = std::size_t(q) * k_tiles + range.k_begin;
        if (partition.capture_slot_by_qk[map_index] != -1) {
          return reject("two split peers share one (q,K_idx) capture key");
        }
        partition.capture_slot_by_qk[map_index] = int32_t(range.capture_slot);
        partition.split_peer_ranges.push_back(range);
      }
    }
    partition.split_peer_offsets[q + 1] =
        uint32_t(partition.split_peer_ranges.size());
    partition.tile_bucket[logical] = split
        ? DenseVerifyBucket::StreamKSplit : DenseVerifyBucket::StreamKWhole;
    partition.scheduler_q[logical] = uint32_t(q);
    partition.final_peer_unit[logical] = final_unit[q];
    partition.final_peer_visit[logical] = final_visit[q];
    split_tiles += split ? 1u : 0u;
    peer_excess += peers[q] - 1;
    uint64_t const tile_in_batch = logical % tiles_per_batch;
    uint64_t const m = tile_in_batch / uint64_t(partition.tiles_n);
    uint64_t const n = tile_in_batch % uint64_t(partition.tiles_n);
    int const valid_m = std::min(
        partition.tile_m, partition.problem_m - int(m) * partition.tile_m);
    int const valid_n = std::min(
        partition.tile_n, partition.problem_n - int(n) * partition.tile_n);
    if (valid_m <= 0 || valid_n <= 0) {
      return reject("logical tile has an empty output residue");
    }
    valid_fixup_elements +=
        (peers[q] - 1) * uint64_t(valid_m) * uint64_t(valid_n);
  }
  uint64_t dp_tiles = 0, whole_tiles = 0;
  for (DenseVerifyBucket bucket : partition.tile_bucket) {
    dp_tiles += bucket == DenseVerifyBucket::DataParallel ? 1u : 0u;
    whole_tiles += bucket == DenseVerifyBucket::StreamKWhole ? 1u : 0u;
  }
  // DP tiles do not touch fixup, but retaining their global q makes every
  // mismatch record self-contained and proves there is no hidden q reuse.
  for (uint64_t q = sk.sk_tiles_; q < tile_count; ++q) {
    uint64_t logical = 0;
    if (!logical_tile_for_q(q, logical)) return reject("DP q cannot be decoded");
    partition.scheduler_q[logical] = uint32_t(q);
  }
  if (std::find(partition.scheduler_q.begin(), partition.scheduler_q.end(),
                uint32_t(-1)) != partition.scheduler_q.end()) {
    return reject("logical output tile has no scheduler q");
  }
  if (dp_tiles + whole_tiles + split_tiles != tile_count ||
      whole_tiles + split_tiles != sk.sk_tiles_) {
    return reject("DP/SK tile census does not close");
  }
  if ((split_tiles == 0) != (peer_excess == 0)) {
    return reject("split-tile and peer-excess witnesses disagree");
  }
  if (partition.split_peer_ranges.size() != split_tiles + peer_excess ||
      partition.split_peer_offsets.back() != partition.split_peer_ranges.size()) {
    return reject("compact split-peer capture census does not close");
  }
  partition.split_tiles = split_tiles;
  partition.peer_excess = peer_excess;
  partition.valid_fixup_elements = valid_fixup_elements;
  std::printf("  [dense verify partition] DP=%llu SK-whole=%llu SK-split=%llu "
              "peer_excess=%llu valid_fixup_elements=%llu qk_cells=%llu "
              "coverage=exact-once\n",
              static_cast<unsigned long long>(dp_tiles),
              static_cast<unsigned long long>(whole_tiles),
              static_cast<unsigned long long>(split_tiles),
              static_cast<unsigned long long>(peer_excess),
              static_cast<unsigned long long>(valid_fixup_elements),
              static_cast<unsigned long long>(coverage.size()));
  return true;
}

// Distinct pairs preserve every launch interval while keeping event creation,
// querying, and destruction outside it.  Reusing one PpuTimer and querying it
// after every launch would serialize the batch and make "independent events"
// a false label; one pair around all launches would only return their sum.
class DenseKernelEventBatch {
 public:
  struct Pair { hggcEvent_t start{}, stop{}; };

  explicit DenseKernelEventBatch(int count) : events_(size_t(count)) {
    for (auto& e : events_) {
      CUTLASS_PPU_CHECK(hggcEventCreate(&e.start));
      CUTLASS_PPU_CHECK(hggcEventCreate(&e.stop));
    }
  }
  ~DenseKernelEventBatch() {
    for (auto& e : events_) {
      CUTLASS_PPU_CHECK(hggcEventDestroy(e.start));
      CUTLASS_PPU_CHECK(hggcEventDestroy(e.stop));
    }
  }
  DenseKernelEventBatch(DenseKernelEventBatch const&) = delete;
  DenseKernelEventBatch& operator=(DenseKernelEventBatch const&) = delete;

  Pair& at(int i) { return events_[size_t(i)]; }
  Pair const& at(int i) const { return events_[size_t(i)]; }

 private:
  std::vector<Pair> events_;
};
#endif

// BENCH_GS optionally restricts every generated wrapper to one group-size instantiation. Unset preserves the
// one-binary --g contract. The same preprocessor decision is compiled into the main TU and every unit TU.
#if defined(BENCH_GS)
#define DENSE_GS_ARM(gs) ((gs) == BENCH_GS)
#else
#define DENSE_GS_ARM(gs) 1
#endif

#if !defined(LOWBIT_DENSE_UNIT_BUILD)
// Main names only exported wrappers. Cfg<>::Gemm appears in lowbit_dense_unit.inc, so expanding this table-driven
// registry does not serialize template instantiation in the main translation unit. Every field participates in
#define LOWBIT_DENSE_SYMBOL_I(TM,TN,TK,WM,WN,ST,BC) \
  lowbit_dense_cfg_tm##TM##_tn##TN##_tk##TK##_wm##WM##_wn##WN##_st##ST##_bc##BC
#define LOWBIT_DENSE_SYMBOL(TM,TN,TK,WM,WN,ST,BC) LOWBIT_DENSE_SYMBOL_I(TM,TN,TK,WM,WN,ST,BC)
#define LOWBIT_DENSE_TAG_SYMBOL_I(TM,TN,TK,WM,WN,ST,BC) \
  lowbit_dense_cfg_tm##TM##_tn##TN##_tk##TK##_wm##WM##_wn##WN##_st##ST##_bc##BC##_tag
#define LOWBIT_DENSE_TAG_SYMBOL(TM,TN,TK,WM,WN,ST,BC) LOWBIT_DENSE_TAG_SYMBOL_I(TM,TN,TK,WM,WN,ST,BC)
#define LOWBIT_DENSE_BC_EFF_SYMBOL_I(TM,TN,TK,WM,WN,ST,BC) \
  lowbit_dense_cfg_tm##TM##_tn##TN##_tk##TK##_wm##WM##_wn##WN##_st##ST##_bc##BC##_effective
#define LOWBIT_DENSE_BC_EFF_SYMBOL(TM,TN,TK,WM,WN,ST,BC) LOWBIT_DENSE_BC_EFF_SYMBOL_I(TM,TN,TK,WM,WN,ST,BC)
#define LOWBIT_DENSE_DECLARE(TM,TN,TK,WM,WN,ST,BC,_UNUSED) \
  Result LOWBIT_DENSE_SYMBOL(TM,TN,TK,WM,WN,ST,BC)(Options&, TileCfg const&); \
  char const* LOWBIT_DENSE_TAG_SYMBOL(TM,TN,TK,WM,WN,ST,BC)(); \
  int LOWBIT_DENSE_BC_EFF_SYMBOL(TM,TN,TK,WM,WN,ST,BC)();
LOWBIT_DENSE_TABLE_CFG_LIST(LOWBIT_DENSE_DECLARE, )
#undef LOWBIT_DENSE_DECLARE

inline std::vector<TileCfg> const& supported_configs() {
  static std::vector<TileCfg> const configs = {
#define LOWBIT_DENSE_REGISTRY_ROW(TM,TN,TK,WM,WN,ST,BC,_UNUSED) \
    TileCfg{LOWBIT_DENSE_TAG_SYMBOL(TM,TN,TK,WM,WN,ST,BC)(), TM, TN, TK, WM, WN, ST, BC, \
            LOWBIT_DENSE_BC_EFF_SYMBOL(TM,TN,TK,WM,WN,ST,BC)(), \
            &LOWBIT_DENSE_SYMBOL(TM,TN,TK,WM,WN,ST,BC)},
    LOWBIT_DENSE_TABLE_CFG_LIST(LOWBIT_DENSE_REGISTRY_ROW, )
#undef LOWBIT_DENSE_REGISTRY_ROW
  };
  return configs;
}
#undef LOWBIT_DENSE_TAG_SYMBOL
#undef LOWBIT_DENSE_TAG_SYMBOL_I
#undef LOWBIT_DENSE_BC_EFF_SYMBOL
#undef LOWBIT_DENSE_BC_EFF_SYMBOL_I
#undef LOWBIT_DENSE_SYMBOL
#undef LOWBIT_DENSE_SYMBOL_I
#endif


/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

DenseFixtureEvidence initialize(Options const& options);

#if !defined(LOWBIT_DENSE_UNIT_BUILD)
/// Helper to initialize a block of device data
template <class Element>
bool initialize_tensor(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023) {

  double scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;
  int bits_output = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  }
  else if (bits_input <= 8) {
    scope_max = 2;
    scope_min = -2;
  }
  else if (bits_output == 16) {
    scope_max = 5;
    scope_min = -5;
  }
  else {
    scope_max = 8;
    scope_min = -8;
  }
  cutlass::reference::device::BlockFillRandomUniform(
      block.get(), block.size(), seed, Element(scope_max), Element(scope_min));

  return true;
}

template <typename Element>
bool initialize_quant_tensor(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023) {

  float scope_min = float(cutlass::platform::numeric_limits<Element>::lowest());
  float scope_max = float(cutlass::platform::numeric_limits<Element>::max());

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, Element(scope_max), Element(scope_min));

  return true;
}

template <class Element>
bool initialize_scale(
  cutlass::DeviceAllocation<Element>& block, 
  Options const& options) {
  
  if (options.mode == GemmMode::ConvertOnly) {
    // No scales, so just initialize with 1 so we can use the same kernel to dequantize the data.
    std::vector<Element> stage(block.size(), Element(1.0f));
    block.copy_from_host(stage.data());
  } 
  else {
    float elt_max_f = float(cutlass::platform::numeric_limits<QuantType>::max());
    const float max_dequant_val = 4.f;
    const float min_dequant_val = 0.5f;

    float scope_max(max_dequant_val / elt_max_f);
    float scope_min(min_dequant_val / elt_max_f);

    cutlass::reference::device::BlockFillRandomUniform(
      block.get(), block.size(), seed, Element(scope_max), Element(scope_min));
  }
  return true;
}

template <class Element>
bool initialize_zero(
  cutlass::DeviceAllocation<Element>& block,
  Options const& options) {
  
  if (options.mode == GemmMode::ScaleWithZeroPoint) {
    cutlass::reference::device::BlockFillRandomUniform(
      block.get(), block.size(), seed, Element(2.0f), Element(-2.0f));
  } else {
    // No bias, so just initialize with 1 so we can use the same kernel to dequantize the data.
    std::vector<Element> stage(block.size(), Element(0.0f));
    block.copy_from_host(stage.data());
  }
  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
DenseFixtureEvidence initialize(Options const& options) {

  DenseFixtureEvidence fixture_evidence;

  auto shape_b = cute::make_shape(options.n, options.k, options.l);
  int const scale_k = (options.k + options.g - 1) / options.g;
  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, options.l));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_b);
  // Reverse stride here due to swap and transpose
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(options.n, options.m, options.l));
  stride_C_ref = cutlass::make_cute_packed_stride(StrideC_ref{}, cute::make_shape(options.m, options.n, options.l));
  // Reverse stride here due to swap and transpose
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.n, options.m, options.l));
  stride_D_ref = cutlass::make_cute_packed_stride(StrideD_ref{}, cute::make_shape(options.m, options.n, options.l));

  auto a_coord = cutlass::make_Coord(options.m * options.l, options.k);
  auto b_coord = cutlass::make_Coord(options.k, options.n * options.l);
  auto c_coord = cutlass::make_Coord(options.m * options.l, options.n);

  block_A.reset(a_coord.product());
  block_B.reset(b_coord.product());
  block_B_buff.resize(b_coord);
  tensor_B.resize(b_coord);
  block_B_dq.reset(b_coord.product());
  block_C.reset(c_coord.product());
  block_D.reset(c_coord.product());
  block_ref_D.reset(c_coord.product());

  block_scale.reset(scale_k * options.l * options.n);
  block_zero.reset(scale_k * options.l * options.n);

  initialize_tensor(block_A, seed + 2022);
  initialize_quant_tensor(block_B, seed + 2021);
  initialize_tensor(block_C, seed + 2020);
  initialize_scale(block_scale, options);
  initialize_zero(block_zero, options);

  block_B.copy_to_host(tensor_B.host_data());

#if defined(DENSE_STREAMK_AB)
  if (options.streamk_gate) {
    // A K-asymmetric, nonzero fixture whose scales change every gs128 group.
    // Random input would usually expose a wrong absolute K iterator, but it
    // would not prove that the seam's local and absolute scale groups differ.
    // This pattern makes that distinction part of the gate's construction.
    std::vector<ElementA> host_a(block_A.size());
    std::vector<ElementC> host_c(block_C.size());
    std::vector<ElementScale> host_scale(block_scale.size());
    for (int m = 0; m < options.m; ++m) {
      for (int k = 0; k < options.k; ++k) {
        int const v = ((m + 3 * k) % 5) - 2;
        host_a[size_t(m) * options.k + k] = ElementA(float(v) * 0.25f);
      }
    }
    for (int m = 0; m < options.m; ++m) {
      for (int n = 0; n < options.n; ++n) {
        int const v = 1 + ((m + 2 * n) % 7);
        host_c[size_t(m) * options.n + n] = ElementC(float(v) * 0.125f);
      }
    }
    for (int kg = 0; kg < scale_k; ++kg) {
      for (int n = 0; n < options.n; ++n) {
        int const v = 1 + ((5 * kg + 3 * n) % 7);
        host_scale[size_t(kg) * options.n + n] = ElementScale(float(v) * 0.0625f);
      }
    }
    auto b_view = tensor_B.host_view();
    for (int k = 0; k < options.k; ++k) {
      for (int n = 0; n < options.n; ++n) {
        int const q = ((7 * k + 3 * n) % 15) - 7;
        b_view.at({k, n}) = QuantType(q);
      }
    }
    block_A.copy_from_host(host_a.data());
    block_C.copy_from_host(host_c.data());
    block_scale.copy_from_host(host_scale.data());
    block_B.copy_from_host(tensor_B.host_data());
    // THIS GATE FIXTURE PROVES ITS OWN EXACTNESS, from the arrays it just filled.
    //
    // SCOPE, STATED FIRST BECAUSE I GOT IT WRONG. This block is inside `if (options.streamk_gate)`,
    // so it describes the small 64x128x4352 gate arm and NOTHING ELSE. At the time this mistake was
    // found, A0's three arms passed only --streamk and used the random initialize_* path. I read this
    // verdict off a log, applied it to A0's
    // 233 SK-split mismatches, and concluded they could not be reassociation. That conclusion does not
    // follow. I even noticed the printed 426496 did not match the 401408 I had computed for K=4096,
    // explained it as "the gate arm has K=4352", and then still carried the conclusion across --
    // exactness is a property of the VALUES, not of K, so a smaller K proves nothing about a different
    // fixture. The separate --streamk_exact_fixture branch below now gives A0 its own evidence; this
    // gate's evidence must still never be substituted for it.
    //
    // What it does establish, for this arm: every a is a multiple of 1/4, every dequantised
    // w = q*scale is a multiple of 1/16, so every product is a multiple of 1/64 and every partial sum
    // is an exactly representable FP32 multiple of it. Under exact arithmetic EVERY accumulation order
    // yields the identical value, and cancellation -- the mechanism "large ULP near zero" gets
    // attributed to -- amplifies pre-existing rounding error, of which there is none. So a mismatch
    // IN THIS ARM cannot be reassociation. This arm is currently BIT-EXACT PASS, which is also why it
    // does not indict the seam: its split deliberately lands inside a gs128 group and still agrees.
    //
    // This is computed rather than asserted so it cannot drift: change any generator above and the
    // printed verdict changes with it.  A future fixture that DOES round will say so instead of
    // silently restoring the ambiguity.
    {
      auto granule_exp = [](double v) {                 // smallest e with v an integer multiple of 2^-e
        for (int e = 0; e <= 24; ++e) {
          double const u = std::ldexp(1.0, -e);
          if (std::fabs(v / u - std::nearbyint(v / u)) < 1e-12) return e;
        }
        return 99;
      };
      // a and w have INDEPENDENT granularities; the product's is their SUM, not twice the larger.
      // Using max(g_a,g_w) twice over-counts by 4x here -- conservative, but a printed number that
      // is not the quantity it names is how a wrong bound gets believed later.
      int ga = 0, gw = 0;
      double amax = 0.0, wmax = 0.0;
      for (int m = 0; m < options.m; ++m)
        for (int k = 0; k < options.k; ++k) {
          double const a = double(float(host_a[size_t(m) * options.k + k]));
          ga = std::max(ga, granule_exp(a));
          amax = std::max(amax, std::fabs(a));
        }
      for (int k = 0; k < options.k; ++k)
        for (int n = 0; n < options.n; ++n) {
          double const w = double(int(b_view.at({k, n}))) *
                           double(float(host_scale[size_t(k / options.g) * options.n + n]));
          gw = std::max(gw, granule_exp(w));
          wmax = std::max(wmax, std::fabs(w));
        }
      // Every product is a multiple of 2^-(ga+gw); the worst-case partial sum is K*amax*wmax.
      double const units = double(options.k) * amax * wmax * std::ldexp(1.0, ga + gw);
      bool const exact = ga < 99 && gw < 99 && units <= std::ldexp(1.0, 24);
      fixture_evidence.order_independent = exact;
      std::printf("  [streamk fixture] deterministic K-asymmetric A/B, per-group-distinct scale, nonzero C\n");
      std::printf("  [streamk fixture exactness] fixture=seam-gate shape=%dx%dx%d "
                  "granule=2^-(%d+%d) |a|max=%g |w|max=%g partial-sum"
                  " units=%.0f vs 2^24=%.0f -> %s\n", options.m, options.n, options.k,
                  ga, gw, amax, wmax, units, std::ldexp(1.0, 24),
                  exact ? "ORDER-INDEPENDENT (a mismatch cannot be reassociation)"
                        : "ROUNDS (a mismatch is ambiguous; fix the fixture before reading the verdict)");
    }
  }
  else if (options.streamk_exact_fixture) {
    // The actual A0 correctness fixture, deliberately separate from the tiny
    // --streamk_gate fixture above.  Every row has one +/-1 in each gs128
    // group, so every split peer sees useful work; scales and int4 codes vary
    // by both absolute K and N.  Products and all subset sums are integers.
    std::vector<ElementA> host_a(block_A.size(), ElementA(0));
    std::vector<ElementC> host_c(block_C.size(), ElementC(0));
    std::vector<ElementScale> host_scale(block_scale.size());
    std::vector<ElementZero> host_zero(block_zero.size(), ElementZero(0));
    static_assert(sizeof(kExactFixtureScales) / sizeof(kExactFixtureScales[0]) == 3,
                  "the exact A0 scale cycle is part of its checked construction");
    static_assert(sizeof(kExactFixtureZeros) / sizeof(kExactFixtureZeros[0]) == 1 &&
                      kExactFixtureZeros[0] == 0,
                  "ScaleOnly A0 must not claim a load-bearing zero plane");
    if (options.k / options.g != kExactFixtureNonzerosPerRow) {
      std::fprintf(stderr,
                   "--streamk_exact_fixture requires K/gs=%d (got %d/%d)\n",
                   kExactFixtureNonzerosPerRow, options.k, options.g);
      std::exit(1);
    }
    for (int m = 0; m < options.m; ++m) {
      for (int kg = 0; kg < kExactFixtureNonzerosPerRow; ++kg) {
        int const within_group = (17 * m + 29 * kg) % options.g;
        int const k = kg * options.g + within_group;
        // B's sign band below depends only on K.  Align A's sign with it so
        // every nonzero product is nonnegative: exact cancellation cannot
        // leave a +/-0 raw-bit ambiguity while q still spans all 16 int4 codes.
        int const sign = ((k >> 3) & 1) ? -1 : 1;
        host_a[size_t(m) * options.k + k] = ElementA(float(sign));
      }
    }
    for (int kg = 0; kg < scale_k; ++kg) {
      for (int n = 0; n < options.n; ++n) {
        int const scale = kExactFixtureScales[(5 * kg + 3 * n) % 3];
        host_scale[size_t(kg) * options.n + n] = ElementScale(float(scale));
      }
    }
    auto b_view = tensor_B.host_view();
    for (int k = 0; k < options.k; ++k) {
      for (int n = 0; n < options.n; ++n) {
        int const code = (5 * k + 3 * n) & 7;
        int const q = ((k >> 3) & 1) ? (-8 + code) : code;
        b_view.at({k, n}) = QuantType(q);
      }
    }
    block_A.copy_from_host(host_a.data());
    block_C.copy_from_host(host_c.data());
    block_scale.copy_from_host(host_scale.data());
    block_zero.copy_from_host(host_zero.data());
    block_B.copy_from_host(tensor_B.host_data());

    // Derive the two bounds from the arrays just installed.  The standalone
    // checker independently derives the same bounds from the published
    // constants; neither side can green by copying a printed expected number.
    int max_nonzeros = 0;
    double max_abs_a = 0.0;
    bool integer_a = true;
    for (int m = 0; m < options.m; ++m) {
      int nonzeros = 0;
      for (int k = 0; k < options.k; ++k) {
        double const a = double(float(host_a[size_t(m) * options.k + k]));
        nonzeros += a != 0.0;
        integer_a &= a == std::nearbyint(a);
        max_abs_a = std::max(max_abs_a, std::fabs(a));
      }
      max_nonzeros = std::max(max_nonzeros, nonzeros);
    }
    double max_weight = 0.0;
    bool integer_weights = true;
    for (int k = 0; k < options.k; ++k) {
      for (int n = 0; n < options.n; ++n) {
        double const weight = double(int(b_view.at({k, n}))) *
            double(float(host_scale[size_t(k / options.g) * options.n + n]));
        integer_weights &= weight == std::nearbyint(weight);
        max_weight = std::max(max_weight, std::fabs(weight));
      }
    }
    double const max_output = double(max_nonzeros) * max_abs_a * max_weight;
    fixture_evidence.order_independent = integer_a && integer_weights &&
        max_nonzeros == kExactFixtureNonzerosPerRow &&
        max_output <= std::ldexp(1.0, 24);
    fixture_evidence.fp16_output_exact = fixture_evidence.order_independent &&
        options.alpha == 1.0f && options.beta == 0.0f &&
        max_output <= std::ldexp(1.0, 11);
    std::printf(
        "  [streamk fixture exactness] fixture=a0-exact shape=%dx%dx%d "
        "nonzeros/row=%d integer_A=%d integer_weights=%d max|A|=%g max|w|=%g max|D|=%g "
        "vs fp32=2^24 fp16=2^11 -> %s\n",
        options.m, options.n, options.k, max_nonzeros, int(integer_a),
        int(integer_weights), max_abs_a, max_weight, max_output,
        fixture_evidence.order_independent && fixture_evidence.fp16_output_exact
            ? "ORDER-INDEPENDENT+FP16-EXACT"
            : "ROUNDS/INVALID (do not classify ordinary-reference differences)");
  }
  else {
    std::printf("  [streamk fixture exactness] fixture=random shape=%dx%dx%d -> "
                "ROUNDS/UNKNOWN (ordinary-reference differences are not classifiable)\n",
                options.m, options.n, options.k);
  }
#endif
  
  auto layout_B = make_layout(shape_b, stride_B);

  auto shape_scale_zero = cute::make_shape(options.n, scale_k, options.l);
  stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(options.n, scale_k, options.l));
  stride_S_ref = cutlass::make_cute_packed_stride(StrideS_ref{}, cute::make_shape(options.n, scale_k, options.l));
  auto layout_scale_zero = make_layout(shape_scale_zero);

  dequantize_weight(block_B_dq.get(), block_B.get(), layout_B, block_scale.get(), block_zero.get(), layout_scale_zero, options.g);
  
  int row = options.n;
  int col = options.k;
  int batch = options.l;

  constexpr bool is_rowmajor = std::is_same<LayoutB, cutlass::layout::RowMajor>::value;
  QuantTypeClass quant_type;
  if (sizeof_bits<QuantType>::value == 8) {
    quant_type = QuantTypeClass::INT8_WEIGHT_ONLY;
  } else if (sizeof_bits<QuantType>::value == 4) {
    quant_type = QuantTypeClass::PACKED_INT4_WEIGHT_ONLY;
  } else if (sizeof_bits<QuantType>::value == 2) {
    quant_type = QuantTypeClass::PACKED_INT2_WEIGHT_ONLY;
  } else if (sizeof_bits<QuantType>::value == 1) {
    quant_type = QuantTypeClass::PACKED_INT1_WEIGHT_ONLY;
  } else {
    std::cerr << "unsupported QuantType" << std::endl;
    exit(-1);
  }
  for (int b = 0; b < batch; b++) {
    int64_t batch_offset = b * row * col;
    // preprocess_weights_for_mixed_gemm<is_rowmajor, -1>((int8_t*)(&block_B_buff.host_data()[batch_offset]),
    preprocess_weights_for_mixed_gemm<is_rowmajor, 256>((int8_t*)(&block_B_buff.host_data()[batch_offset]),
        (int8_t*)(&tensor_B.host_data()[batch_offset]),
        {static_cast<size_t>(col), static_cast<size_t>(row)}, quant_type);
  }
  block_B_buff.sync_device();
  return fixture_evidence;
}
#endif

/// Populates a Gemm::Arguments structure from the given commandline options
template <typename Args>
Args args_from_options(Options const& options)
{
// Swap the A and B tensors, as well as problem shapes here.
  if (options.mode == GemmMode::ConvertOnly) {
    return Args {
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.m, options.n, options.k, options.l},
      {block_A.get(), stride_A, block_B_buff.device_data(), stride_B},
      {{options.alpha, options.beta}, block_C.get(), stride_C_ref, block_D.get(), stride_D_ref}
    };
  } 
  else if (options.mode == GemmMode::ScaleOnly) {
    return Args {
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.m, options.n, options.k, options.l},
      {block_A.get(), stride_A, block_B_buff.device_data(), stride_B, block_scale.get(), stride_S, options.g},
      {{options.alpha, options.beta}, block_C.get(), stride_C_ref, block_D.get(), stride_D_ref}
    };
  }
  else if (options.mode == GemmMode::ScaleWithZeroPoint) {
    return Args {
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.m, options.n, options.k, options.l},
      {block_A.get(), stride_A, block_B_buff.device_data(), stride_B, block_scale.get(), stride_S, options.g, block_zero.get()},
      {{options.alpha, options.beta}, block_C.get(), stride_C_ref, block_D.get(), stride_D_ref}
    };
  } else {
    std::cerr << "Invalid mode " << options.mode << ". Must be 0, 1 or 2." << std::endl;
    exit(-1);
  }
}

enum class DenseVerifyState : uint8_t {
  NotClassifiable,
  Classified,
  InvariantFailure,
};

bool verify(const Options &options, DenseFixtureEvidence const& fixture_evidence,
            DenseVerifyPartition const* partition,
            DenseVerifyState* diagnostic_state = nullptr);
bool verify_streamk_cpu_fp32(const Options &options);
#if defined(DENSE_STREAMK_AB)
template <class Gemm>
DenseReplayEvidence verify_streamk_same_order_partial_replay(
    const Options& options, DenseVerifyPartition const& partition,
    std::vector<ElementD> const& normal_output,
    std::vector<ElementD> const& capture_output,
    std::vector<ElementAccumulator> const& captured_partials,
    std::vector<uint32_t> const& capture_slot_visits,
    std::vector<uint32_t> const& capture_slot_k_counts,
    uint32_t capture_errors);
#endif

#if !defined(LOWBIT_DENSE_UNIT_BUILD)
bool verify(const Options &options, DenseFixtureEvidence const& fixture_evidence,
            DenseVerifyPartition const* partition,
            DenseVerifyState* diagnostic_state) {
  if (diagnostic_state) {
    *diagnostic_state = DenseVerifyState::NotClassifiable;
  }
  //
  // Compute reference output
  //

  // In this example, we use the PPU default kernels as a reference (unfused scale)
  // This avoids numerical differences due to different accumulation order.

  // Again, due to numerical differences, we must use fast acc here when the mma type is
  // FP8 as the fused implementation only supports fast acc at the moment.
  constexpr bool IsFP8Input = cute::is_same_v<MmaType, cutlass::float_e4m3_t> || cute::is_same_v<MmaType, cutlass::float_e5m2_t>;
  using FP8Sched = cute::conditional_t<size<0>(TileShape{}) == 64, cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum, cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum>;
  using ScheduleRef = cute::conditional_t<IsFP8Input, FP8Sched, cutlass::gemm::collective::KernelScheduleAuto>;

  using CollectiveMainloopRef = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      MmaType, LayoutA, AlignmentA,
      MmaType, LayoutB, AlignmentA,
      ElementAccumulator,
      TileShape, WarpShape,
      cutlass::gemm::collective::StageCountAuto,
      ScheduleRef
    >::CollectiveOp;

  using CollectiveEpilogueRef = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, cutlass::arch::OpClassTensorOp,
      TileShape, WarpShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using GemmKernelRef = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>, // Indicates ProblemShape
      CollectiveMainloopRef,
      CollectiveEpilogueRef
  >;

  using GemmRef = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelRef>;

  typename GemmRef::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, options.l},
    {block_A.get(), stride_A, block_B_dq.get(), stride_B},
    {{options.alpha, options.beta}, block_C.get(), stride_C_ref, block_ref_D.get(), stride_D_ref}
  };

  // Run the gemm where the scaling is performed outside of the kernel.
  GemmRef gemm_ref;
  size_t workspace_size = GemmRef::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  CUTLASS_CHECK(gemm_ref.can_implement(arguments));
  CUTLASS_CHECK(gemm_ref.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm_ref.run());

  // compare_reference
  ElementD const epsilon(1e-2f);
  ElementD const non_zero_floor(1e-4f);
  bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(block_ref_D.get(), block_D.get(), block_D.size(), epsilon, non_zero_floor);
  bool const device_comparator_passed = passed;

  if (partition != nullptr) {
    std::size_t const expected_tiles = std::size_t(partition->tiles_m) *
        std::size_t(partition->tiles_n) * std::size_t(partition->batches);
    bool const common_map_ok =
        partition->classification_closed &&
        partition->tile_m > 0 && partition->tile_n > 0 &&
        partition->tiles_m == int(cute::ceil_div(options.m, partition->tile_m)) &&
        partition->tiles_n == int(cute::ceil_div(options.n, partition->tile_n)) &&
        partition->batches == options.l &&
        partition->tile_bucket.size() == expected_tiles &&
        partition->scheduler_q.size() == expected_tiles &&
        partition->final_peer_unit.size() == expected_tiles &&
        partition->final_peer_visit.size() == expected_tiles &&
        partition->fixup_threads > 0 &&
        partition->local_lane.size() ==
            std::size_t(partition->tile_m) * partition->tile_n &&
        partition->local_stripe.size() ==
            std::size_t(partition->tile_m) * partition->tile_n;
    // These fields describe K-peer capture and have no meaning for ordinary
    // data-parallel or serial-persistent arms.  Requiring them in the common
    // map check made every DP grid look invalid before its buckets ran.
    bool const replay_map_ok = !partition->is_streamk ||
        (partition->k_tiles_per_output_tile > 0 &&
        partition->split_peer_offsets.size() >= 1 &&
        partition->capture_slot_by_qk.size() ==
            (partition->split_peer_offsets.size() - 1) *
                std::size_t(partition->k_tiles_per_output_tile));
    if (!common_map_ok || !replay_map_ok) {
      std::fprintf(stderr,
                   "  [dense verify buckets] NOT CLASSIFIABLE: tile=%dx%d "
                   "logical_grid=%dx%dx%d entries=%zu common=%d replay=%d streamk=%d\n",
                   partition->tile_m, partition->tile_n, partition->tiles_m,
                   partition->tiles_n, partition->batches,
                   partition->tile_bucket.size(), int(common_map_ok),
                   int(replay_map_ok), int(partition->is_streamk));
      return passed;
    }

    static_assert(cutlass::sizeof_bits<ElementD>::value == 16,
                  "107b half-ULP diagnostics require a 16-bit epilogue output");
    std::vector<ElementD> host_ref(block_ref_D.size());
    std::vector<ElementD> host_got(block_D.size());
    block_ref_D.copy_to_host(host_ref.data());
    block_D.copy_to_host(host_got.data());

    struct BucketStats {
      uint64_t tiles = 0;
      uint64_t outputs = 0;
      uint64_t mismatches = 0;
      uint64_t nonfinite = 0;
      double max_abs = 0.0;
      double max_rel_sym = 0.0;
      uint32_t max_half_ulp = 0;
    };
    constexpr std::size_t bucket_count =
        static_cast<std::size_t>(DenseVerifyBucket::Count);
    std::array<BucketStats, bucket_count> stats{};
    std::vector<uint32_t> mismatch_per_tile(expected_tiles, 0);
    std::vector<uint32_t> bitdiff_per_tile(expected_tiles, 0);
    std::vector<uint32_t> mismatch_per_local(
        std::size_t(partition->tile_m) * partition->tile_n, 0);
    uint64_t mismatch_position_hash = 1469598103934665603ull;
    uint64_t mismatch_value_hash = 1469598103934665603ull;
    uint64_t bitdiff_position_hash = 1469598103934665603ull;
    uint64_t bitdiff_count = 0;
    uint64_t detail_count = 0;
    auto fnv_mix = [](uint64_t& hash, uint64_t value) {
      for (int byte = 0; byte < 8; ++byte) {
        hash ^= uint8_t(value >> (8 * byte));
        hash *= 1099511628211ull;
      }
    };
    for (DenseVerifyBucket bucket : partition->tile_bucket) {
      std::size_t const i = static_cast<std::size_t>(bucket);
      if (i >= stats.size()) {
        std::fprintf(stderr,
                     "  [dense verify buckets] invalid bucket id=%zu in tile map\n", i);
        if (diagnostic_state) {
          *diagnostic_state = DenseVerifyState::InvariantFailure;
        }
        return false;
      }
      ++stats[i].tiles;
    }

    auto ordered_half = [](ElementD value) {
      uint16_t const bits = value.raw();
      // Collapse +/-0 to the same point and make adjacent finite half values
      // adjacent integers on both sides of zero.
      return (bits & 0x8000u)
          ? int32_t(0x8000u - (bits & 0x7fffu))
          : int32_t(0x8000u + bits);
    };
    auto half_ulp = [&](ElementD a, ElementD b) {
      if (!std::isfinite(float(a)) || !std::isfinite(float(b))) {
        return std::numeric_limits<uint32_t>::max();
      }
      int64_t const delta = int64_t(ordered_half(a)) - int64_t(ordered_half(b));
      return uint32_t(delta < 0 ? -delta : delta);
    };

    for (int l = 0; l < options.l; ++l) {
      for (int m = 0; m < options.m; ++m) {
        for (int n = 0; n < options.n; ++n) {
          std::size_t const out =
              (std::size_t(l) * options.m + m) * options.n + n;
          std::size_t const tile =
              (std::size_t(l) * partition->tiles_m + m / partition->tile_m) *
                  partition->tiles_n + n / partition->tile_n;
          std::size_t const bucket = std::size_t(partition->tile_bucket[tile]);
          if (bucket >= stats.size()) {
            std::fprintf(stderr,
                         "  [dense verify buckets] invalid bucket id=%zu at logical tile=%zu\n",
                         bucket, tile);
            if (diagnostic_state) {
              *diagnostic_state = DenseVerifyState::InvariantFailure;
            }
            return false;
          }

          ElementD const want = host_ref[out];
          ElementD const got = host_got[out];
          double const want_f = double(float(want));
          double const got_f = double(float(got));
          bool const finite = std::isfinite(want_f) && std::isfinite(got_f);
          BucketStats& s = stats[bucket];
          ++s.outputs;
          bool const mismatch =
              !cutlass::relatively_equal(want, got, epsilon, non_zero_floor);
          bool const bitdiff = want.raw() != got.raw();
          if (bitdiff) {
            ++bitdiff_count;
            ++bitdiff_per_tile[tile];
            fnv_mix(bitdiff_position_hash, out);
          }
          if (mismatch) {
            ++s.mismatches;
            ++mismatch_per_tile[tile];
            int const local_m = m % partition->tile_m;
            int const local_n = n % partition->tile_n;
            std::size_t const local =
                std::size_t(local_m) * partition->tile_n + local_n;
            ++mismatch_per_local[local];
            fnv_mix(mismatch_position_hash, out);
            fnv_mix(mismatch_value_hash,
                    (uint64_t(want.raw()) << 16) | uint64_t(got.raw()));
            if (detail_count < 16) {
              std::printf(
                  "  [dense verify mismatch] out=%zu tile=%zu q=%u "
                  "global=(%d,%d,%d) local=(%d,%d) lane=%u stripe=%u "
                  "final_unit=%u final_visit=%u want=0x%04x got=0x%04x ulp=%u\n",
                  out, tile, partition->scheduler_q[tile], l, m, n,
                  local_m, local_n, unsigned(partition->local_lane[local]),
                  unsigned(partition->local_stripe[local]),
                  partition->final_peer_unit[tile],
                  unsigned(partition->final_peer_visit[tile]),
                  unsigned(want.raw()), unsigned(got.raw()), half_ulp(want, got));
              ++detail_count;
            }
          }
          if (finite) {
            double const abs_err = std::abs(got_f - want_f);
            // This symmetric denominator is the same scale used by the
            // comparator's non-near-zero branch.  Pass/fail still calls the
            // exact CUTLASS predicate above; max_abs explains its near-zero
            // branch and ULP makes small half regrouping visible.
            double const denom = std::abs(got_f) + std::abs(want_f);
            double const rel_sym = denom == 0.0 ? 0.0 : abs_err / denom;
            s.max_abs = std::max(s.max_abs, abs_err);
            s.max_rel_sym = std::max(s.max_rel_sym, rel_sym);
          }
          else {
            ++s.nonfinite;
          }
          s.max_half_ulp = std::max(s.max_half_ulp, half_ulp(want, got));
        }
      }
    }

    constexpr char const* names[] = {"DP", "SK-whole", "SK-split"};
    uint64_t bucket_mismatches = 0;
    for (std::size_t i = 0; i < stats.size(); ++i) {
      BucketStats const& s = stats[i];
      bucket_mismatches += s.mismatches;
      std::printf("  [dense verify bucket=%s] tiles=%llu outputs=%llu mismatches=%llu "
                  "max_abs=%.9g max_rel_sym=%.9g max_half_ulp=%u nonfinite=%llu\n",
                  names[i], static_cast<unsigned long long>(s.tiles),
                  static_cast<unsigned long long>(s.outputs),
                  static_cast<unsigned long long>(s.mismatches), s.max_abs,
                  s.max_rel_sym, s.max_half_ulp,
                  static_cast<unsigned long long>(s.nonfinite));
    }
    uint64_t mismatch_tiles = 0;
    uint64_t one_mismatch_tiles = 0;
    uint64_t bitdiff_tiles = 0;
    uint32_t max_mismatches_per_tile = 0;
    uint32_t max_bitdiff_per_tile = 0;
    uint64_t mismatch_on_first_final_visit = 0;
    uint64_t mismatch_after_prior_final_visit = 0;
    for (std::size_t tile = 0; tile < expected_tiles; ++tile) {
      uint32_t const count = mismatch_per_tile[tile];
      mismatch_tiles += count != 0;
      one_mismatch_tiles += count == 1;
      max_mismatches_per_tile = std::max(max_mismatches_per_tile, count);
      bitdiff_tiles += bitdiff_per_tile[tile] != 0;
      max_bitdiff_per_tile = std::max(
          max_bitdiff_per_tile, bitdiff_per_tile[tile]);
      if (partition->tile_bucket[tile] == DenseVerifyBucket::StreamKSplit) {
        if (partition->final_peer_visit[tile] == 0) {
          mismatch_on_first_final_visit += count;
        }
        else {
          mismatch_after_prior_final_visit += count;
        }
      }
    }
    auto local_mode = std::max_element(
        mismatch_per_local.begin(), mismatch_per_local.end());
    std::size_t const local_mode_idx =
        local_mode == mismatch_per_local.end()
            ? 0 : std::size_t(local_mode - mismatch_per_local.begin());
    uint32_t const local_mode_count =
        local_mode == mismatch_per_local.end() ? 0 : *local_mode;
    std::printf(
        "  [dense verify fingerprint] comparator_positions=%llu "
        "position_fnv1a=%016llx value_fnv1a=%016llx raw_bitdiff=%llu "
        "raw_position_fnv1a=%016llx raw_bitdiff_tiles=%llu raw_max_per_tile=%u "
        "mismatch_tiles=%llu one_mismatch_tiles=%llu max_per_tile=%u "
        "final_visit0=%llu final_visit_gt0=%llu "
        "local_mode=(%zu,%zu):%u\n",
        static_cast<unsigned long long>(bucket_mismatches),
        static_cast<unsigned long long>(mismatch_position_hash),
        static_cast<unsigned long long>(mismatch_value_hash),
        static_cast<unsigned long long>(bitdiff_count),
        static_cast<unsigned long long>(bitdiff_position_hash),
        static_cast<unsigned long long>(bitdiff_tiles), max_bitdiff_per_tile,
        static_cast<unsigned long long>(mismatch_tiles),
        static_cast<unsigned long long>(one_mismatch_tiles),
        max_mismatches_per_tile,
        static_cast<unsigned long long>(mismatch_on_first_final_visit),
        static_cast<unsigned long long>(mismatch_after_prior_final_visit),
        local_mode_idx / std::size_t(partition->tile_n),
        local_mode_idx % std::size_t(partition->tile_n), local_mode_count);
    bool const raw_reference_passed = bitdiff_count == 0;
    if (fixture_evidence.order_independent) {
      std::printf(
          "  [dense verify interpretation] ORDER-INDEPENDENT fixture: raw_bitdiff=%llu; "
          "any nonzero ordinary-reference difference is a numerical failure.\n",
          static_cast<unsigned long long>(bitdiff_count));
    }
    else {
      std::printf(
          "  [dense verify interpretation] fixture rounds: ordinary-reference ULP is diagnostic only; "
          "same-order replay can validate fixup, not the captured partials.\n");
    }
    if ((bucket_mismatches == 0) != device_comparator_passed) {
      std::fprintf(stderr,
                   "  [dense verify buckets] fail-close: bucket comparator disagrees with device comparator "
                   "(bucket mismatches=%llu device_passed=%d)\n",
                   static_cast<unsigned long long>(bucket_mismatches), int(passed));
      if (diagnostic_state) {
        *diagnostic_state = DenseVerifyState::InvariantFailure;
      }
      return false;
    }
    if (fixture_evidence.order_independent) {
      passed = raw_reference_passed;
    }
  }
  if (diagnostic_state) {
    *diagnostic_state = DenseVerifyState::Classified;
  }
  return passed;
}

// Independent scalar host accumulation for the exact dyadic 107b fixture.
// This is deliberately sequential k=0..K-1; it does NOT reproduce arbitrary
// Stream-K peer grouping or the PPU MMA instruction's internal reduction
// order.  The fixture's powers-of-two make those regroupings exact, while A,
// scale and C still come independently from host data and nonzero beta proves
// that only the final slice executes the epilogue.  A0 carries its own explicit
// fixture identity and exactness evidence instead of extrapolating this arm.
bool verify_streamk_cpu_fp32(const Options &options) {
  if (options.l != 1 || options.beta == 0.0f) {
    std::fprintf(stderr,
                 "[streamk sequential CPU-FP32 fixture] requires l=1 and nonzero beta "
                 "(got l=%d beta=%g)\n",
                 options.l, double(options.beta));
    return false;
  }
  uint64_t const macs = uint64_t(options.m) * uint64_t(options.n) * uint64_t(options.k);
  constexpr uint64_t kMaxGateMacs = 100000000ull;
  if (macs > kMaxGateMacs) {
    std::fprintf(stderr,
                 "[streamk sequential CPU-FP32 fixture] gate is capped at %llu MACs "
                 "(requested %llu); "
                 "use the documented 64x128x4352 shape\n",
                 static_cast<unsigned long long>(kMaxGateMacs),
                 static_cast<unsigned long long>(macs));
    return false;
  }

  std::vector<ElementA> host_a(block_A.size());
  std::vector<ElementScale> host_scale(block_scale.size());
  std::vector<ElementC> host_c(block_C.size());
  std::vector<ElementD> host_d(block_D.size());
  block_A.copy_to_host(host_a.data());
  block_scale.copy_to_host(host_scale.data());
  block_C.copy_to_host(host_c.data());
  block_D.copy_to_host(host_d.data());

  int bad = 0;
  int bitdiff = 0;
  double max_abs = 0.0;
  double max_rel = 0.0;
  for (int m = 0; m < options.m; ++m) {
    for (int n = 0; n < options.n; ++n) {
      float accum = 0.0f;
      for (int k = 0; k < options.k; ++k) {
        // The source HostTensor has extent [K,N] and column-major layout, so
        // at({k,n}) is the same packed source element the device preprocesses.
        int const q = int(tensor_B.host_view().at({k, n}));
        float const scale = float(host_scale[size_t(k / options.g) * options.n + n]);
        accum += float(host_a[size_t(m) * options.k + k]) * (float(q) * scale);
      }
      size_t const out = size_t(m) * options.n + n;
      float const golden = options.alpha * accum + options.beta * float(host_c[out]);
      float const got = float(host_d[out]);
      float const rounded_golden = float(ElementD(golden));
      double const abs_err = std::abs(double(got) - double(golden));
      double const rel_err = abs_err / (std::abs(double(golden)) + 1.0e-3);
      max_abs = std::max(max_abs, abs_err);
      max_rel = std::max(max_rel, rel_err);
      if (got != rounded_golden) {
        ++bad;
        ++bitdiff;
      }
    }
  }
  size_t const outputs = size_t(options.m) * options.n;
  std::printf("  [streamk sequential CPU-FP32 fixture] order=k-ascending "
              "dyadic=1 outputs=%zu bad=%d bitdiff=%d max_abs=%.6g "
              "max_rel=%.6g %s\n",
              outputs, bad, bitdiff, max_abs, max_rel,
              bad == 0 ? "BIT-EXACT" : "MISMATCH");
  return bad == 0;
}
#endif

#if defined(DENSE_STREAMK_AB)
// The broad reference above deliberately uses a whole-K GEMM.  A Stream-K
// split changes FP32 parenthesization before the half epilogue, so disagreement
// with that reference is not by itself a kernel defect.  This gate captures
// each real pre-fixup FP32 peer fragment from the production mainloop, folds
// those fragments on the host in increasing K_idx order (the deterministic
// lock-chain order), and compares the resulting half output bit-for-bit.
//
// Capturing production partials is intentional: a scalar host dot product does
// not know the PPU m16n16k16 instruction's internal reduction order.  The old
// CPU fixture below/above is sequential-k and uses exact dyadic inputs; it is
// an excellent absolute-K seam gate, but it is not an A0 same-order replay.
template <class Gemm>
DenseReplayEvidence verify_streamk_same_order_partial_replay(
    const Options& options, DenseVerifyPartition const& partition,
    std::vector<ElementD> const& normal_output,
    std::vector<ElementD> const& capture_output,
    std::vector<ElementAccumulator> const& captured_partials,
    std::vector<uint32_t> const& capture_slot_visits,
    std::vector<uint32_t> const& capture_slot_k_counts,
    uint32_t capture_errors) {
  auto fail = [](char const* why) {
    std::fprintf(stderr, "  [streamk same-order replay] UNVERIFIED: %s\n", why);
    return DenseReplayEvidence{};
  };
  if (options.l != 1) return fail("first replay supports l=1 only");
  std::size_t const outputs =
      std::size_t(options.m) * options.n * options.l;
  std::size_t const tile_elements =
      std::size_t(partition.tile_m) * partition.tile_n;
  std::size_t const slots = partition.split_peer_ranges.size();
  if (partition.split_tiles == 0 || partition.peer_excess == 0 ||
      slots != partition.split_tiles + partition.peer_excess) {
    return fail("no nonempty split-peer census");
  }
  if (normal_output.size() != outputs || capture_output.size() != outputs ||
      captured_partials.size() != slots * tile_elements ||
      capture_slot_visits.size() != slots ||
      capture_slot_k_counts.size() != slots ||
      partition.fixup_threads <= 0 ||
      partition.split_peer_offsets.size() < 2) {
    return fail("capture/output extent does not match the scheduler census");
  }
  if (capture_errors != 0) {
    return fail("device capture rejected a q/K_idx/slot mapping");
  }
  uint64_t bad_slot_visits = 0;
  for (uint32_t visits : capture_slot_visits) bad_slot_visits += visits != 1;
  if (bad_slot_visits != 0) {
    return fail("a compact peer slot was captured zero or multiple times");
  }
  uint64_t bad_k_counts = 0;
  for (std::size_t peer = 0; peer < slots; ++peer) {
    bad_k_counts += capture_slot_k_counts[peer] !=
        partition.split_peer_ranges[peer].k_count;
  }
  if (bad_k_counts != 0) {
    return fail("device peer K counts disagree with the host scheduler census");
  }

  uint64_t capture_holes = 0;
  for (ElementAccumulator value : captured_partials) {
    capture_holes += !std::isfinite(float(value));
  }
  if (capture_holes != 0) {
    return fail("poison remained in a captured FP32 peer fragment");
  }

  uint64_t capture_vs_normal_bitdiff = 0;
  for (std::size_t out = 0; out < outputs; ++out) {
    capture_vs_normal_bitdiff +=
        capture_output[out].raw() != normal_output[out].raw();
  }
  if (capture_vs_normal_bitdiff != 0) {
    return fail("the diagnostic capture changed the production output");
  }

  std::vector<ElementD> host_ref(block_ref_D.size());
  std::vector<ElementC> host_c(block_C.size());
  block_ref_D.copy_to_host(host_ref.data());
  block_C.copy_to_host(host_c.data());
  if (host_ref.size() != outputs || host_c.size() != outputs) {
    return fail("whole-K reference/C extent drifted");
  }

  ElementD const epsilon(1e-2f);
  ElementD const non_zero_floor(1e-4f);
  uint64_t split_outputs = 0;
  uint64_t device_replay_bitdiff = 0;
  uint64_t non_split_reference_mismatches = 0;
  uint64_t non_split_reference_bitdiff = 0;
  uint64_t device_reference_bitdiff = 0;
  uint64_t replay_reference_bitdiff = 0;
  uint64_t device_reference_position_hash = 1469598103934665603ull;
  uint64_t replay_reference_position_hash = 1469598103934665603ull;
  uint64_t device_reference_value_hash = 1469598103934665603ull;
  uint64_t replay_reference_value_hash = 1469598103934665603ull;
  auto fnv_mix = [](uint64_t& hash, uint64_t value) {
    for (int byte = 0; byte < 8; ++byte) {
      hash ^= uint8_t(value >> (8 * byte));
      hash *= 1099511628211ull;
    }
  };

  // Pin the host replay to the exact linear-combination operation selected by
  // this shipping collective.  The PPU EVT clears C when beta is zero, then
  // evaluates beta*C + alpha*acc through the same CUTLASS multiply/multiply-add
  // functors and destination converter used below.  If the collective grows a
  // different epilogue, this gate must learn that operation instead of silently
  // retaining a hand-written arithmetic expression.
  using ExpectedFusionOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementAccumulator, ElementC, ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;
  static_assert(cute::is_same_v<
                    typename Gemm::CollectiveEpilogue::ThreadEpilogueOp,
                    ExpectedFusionOp>,
                "same-order replay is pinned to the shipping linear-combination EVT");
  using ReplayEpilogue = cutlass::epilogue::thread::LinearCombination<
      ElementD, 1, ElementAccumulator, ElementAccumulator,
      cutlass::epilogue::thread::ScaleType::Default,
      cutlass::FloatRoundStyle::round_to_nearest, ElementC>;
  typename ReplayEpilogue::Params replay_params{
      ElementAccumulator(options.alpha), ElementAccumulator(options.beta)};
  ReplayEpilogue replay_epilogue(replay_params);

  for (int m = 0; m < options.m; ++m) {
    for (int n = 0; n < options.n; ++n) {
      std::size_t const out = std::size_t(m) * options.n + n;
      std::size_t const logical_tile =
          std::size_t(m / partition.tile_m) * partition.tiles_n +
          std::size_t(n / partition.tile_n);
      ElementD const got = normal_output[out];
      ElementD const ref = host_ref[out];
      if (partition.tile_bucket[logical_tile] !=
          DenseVerifyBucket::StreamKSplit) {
        non_split_reference_mismatches +=
            !cutlass::relatively_equal(ref, got, epsilon, non_zero_floor);
        non_split_reference_bitdiff += got.raw() != ref.raw();
        continue;
      }

      ++split_outputs;
      uint32_t const q = partition.scheduler_q[logical_tile];
      if (std::size_t(q + 1) >= partition.split_peer_offsets.size()) {
        return fail("split output decoded outside peer-offset table");
      }
      uint32_t const first = partition.split_peer_offsets[q];
      uint32_t const last = partition.split_peer_offsets[q + 1];
      if (last <= first + 1 || last > partition.split_peer_ranges.size()) {
        return fail("split output does not have at least two captured peers");
      }
      int const local_m = m % partition.tile_m;
      int const local_n = n % partition.tile_n;
      std::size_t const local =
          std::size_t(local_m) * partition.tile_n + local_n;
      std::size_t const physical =
          std::size_t(partition.local_stripe[local]) *
              std::size_t(partition.fixup_threads) +
          partition.local_lane[local];
      if (physical >= tile_elements) {
        return fail("logical accumulator coordinate maps outside capture tile");
      }

      auto captured = [&](uint32_t peer) -> float {
        DenseVerifyPeerRange const& range = partition.split_peer_ranges[peer];
        return float(captured_partials[
            std::size_t(range.capture_slot) * tile_elements + physical]);
      };
      // Mirror the live fixup operation-for-operation.  Peer 0 stores the
      // workspace value, middle peers atomic-add into it in K_idx order, and
      // the final peer executes load_add as (final_partial + workspace).
      // For finite FP32 values a single left fold happens to produce the same
      // rounded additions (the last operation only swaps its two operands),
      // but spelling out the vendor sequence keeps this oracle auditable and
      // avoids depending on that equivalence.
      volatile float workspace_replay = captured(first);
      for (uint32_t peer = first + 1; peer + 1 < last; ++peer) {
        workspace_replay = float(workspace_replay) + captured(peer);
      }
      float const replay = captured(last - 1) + float(workspace_replay);
      // EVT's beta==0 specialization clears its source fragment rather than
      // multiplying the live C value by zero (which matters for NaN and signed
      // zero).  Keep that detail in the oracle as well.
      ElementC const replay_source =
          options.beta == 0.0f ? ElementC(0) : host_c[out];
      ElementD const replay_half = replay_epilogue(replay, replay_source);
      device_replay_bitdiff += replay_half.raw() != got.raw();

      bool const device_ref_diff = got.raw() != ref.raw();
      bool const replay_ref_diff = replay_half.raw() != ref.raw();
      device_reference_bitdiff += device_ref_diff;
      replay_reference_bitdiff += replay_ref_diff;
      if (device_ref_diff) {
        fnv_mix(device_reference_position_hash, out);
        fnv_mix(device_reference_value_hash,
                (uint64_t(ref.raw()) << 16) | uint64_t(got.raw()));
      }
      if (replay_ref_diff) {
        fnv_mix(replay_reference_position_hash, out);
        fnv_mix(replay_reference_value_hash,
                (uint64_t(ref.raw()) << 16) | uint64_t(replay_half.raw()));
      }
    }
  }

  bool const triangle_closed =
      device_reference_bitdiff == replay_reference_bitdiff &&
      device_reference_position_hash == replay_reference_position_hash &&
      device_reference_value_hash == replay_reference_value_hash;
  // FIXUP-CLOSED is deliberately independent of the whole-K reference,
  // including non-split tiles.  It answers one question only: did production
  // fixup reproduce the captured peer partials?  Reference agreement is
  // consumed separately with this invocation's fixture exactness.
  bool const fixup_closed = split_outputs > 0 && device_replay_bitdiff == 0 &&
      triangle_closed;
  std::printf(
      "  [streamk same-order replay] split_tiles=%llu peers=%zu "
      "split_outputs=%llu capture_scalars=%zu capture_holes=%llu "
      "bad_slot_visits=%llu bad_k_counts=%llu "
      "capture_vs_normal_bitdiff=%llu device_replay_bitdiff=%llu "
      "non_split_reference_mismatches=%llu non_split_reference_bitdiff=%llu "
      "reference_raw_bitdiff=%llu "
      "triangle=%s %s\n",
      static_cast<unsigned long long>(partition.split_tiles), slots,
      static_cast<unsigned long long>(split_outputs), captured_partials.size(),
      static_cast<unsigned long long>(capture_holes),
      static_cast<unsigned long long>(bad_slot_visits),
      static_cast<unsigned long long>(bad_k_counts),
      static_cast<unsigned long long>(capture_vs_normal_bitdiff),
      static_cast<unsigned long long>(device_replay_bitdiff),
      static_cast<unsigned long long>(non_split_reference_mismatches),
      static_cast<unsigned long long>(non_split_reference_bitdiff),
      static_cast<unsigned long long>(device_reference_bitdiff),
      triangle_closed ? "CLOSED" : "OPEN",
      fixup_closed ? "FIXUP-CLOSED" : "FIXUP-FAIL");
  if (fixup_closed) {
    std::printf("  [streamk replay meaning] FIXUP-CLOSED: production fixup matches "
                "captured partials; partial correctness is not established.\n");
  }
  return DenseReplayEvidence{
      fixup_closed, device_reference_bitdiff, non_split_reference_bitdiff};
}
#endif

/// Execute a given example GEMM computation. Returns the Result (does not exit on failure, so the tactic
/// search can skip a config that does not verify and move on). The structured tactic drives both the display
/// tag and the traffic model; neither is recovered by parsing the other.
template <class T, class = void>
struct dense_has_persistent_ctas : std::false_type {};

template <class T>
struct dense_has_persistent_ctas<
    T, std::void_t<decltype(std::declval<T&>().ctas_per_cu)>> : std::true_type {};

template <class Gemm, class = void>
struct dense_is_streamk_gemm : std::false_type {};

template <class Gemm>
struct dense_is_streamk_gemm<
    Gemm, std::void_t<decltype(std::declval<typename Gemm::Arguments&>().diagnostics)>>
    : std::true_type {};

template <class Gemm, class = void>
struct dense_is_marlin_gemm : std::false_type {};

template <class Gemm>
struct dense_is_marlin_gemm<
    Gemm, std::void_t<decltype(Gemm::GemmKernel::IsDenseMarlin)>>
    : std::bool_constant<Gemm::GemmKernel::IsDenseMarlin> {};

struct DenseNoStreamKDiagnostics { uint32_t witness[3]{}; };

template <class Gemm, bool = dense_is_streamk_gemm<Gemm>::value>
struct dense_streamk_diagnostic_type { using type = DenseNoStreamKDiagnostics; };

template <class Gemm>
struct dense_streamk_diagnostic_type<Gemm, true> {
  using type = typename Gemm::GemmKernel::DiagnosticState;
};

template <typename Gemm>
Result run(Options &options, bench_measure::Tactic tactic = dense_convert_tactic(),
           char const* scheduler_kind = "non-persistent")
{
  DenseFixtureEvidence const fixture_evidence = initialize(options);

  // Instantiate kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options<typename Gemm::Arguments>(options);

#if defined(DENSE_STREAMK_AB)
  DenseVerifyPartition verify_partition;
  verify_partition.is_streamk = dense_is_streamk_gemm<Gemm>::value;
  verify_partition.tile_m = tactic.tm;
  verify_partition.tile_n = tactic.tn;
  verify_partition.problem_m = options.m;
  verify_partition.problem_n = options.n;
  verify_partition.tiles_m = int(cute::ceil_div(options.m, tactic.tm));
  verify_partition.tiles_n = int(cute::ceil_div(options.n, tactic.tn));
  verify_partition.batches = options.l;
  verify_partition.tile_bucket.assign(
      std::size_t(verify_partition.tiles_m) *
          std::size_t(verify_partition.tiles_n) *
          std::size_t(verify_partition.batches),
      DenseVerifyBucket::DataParallel);
  std::size_t const verify_tiles = verify_partition.tile_bucket.size();
  verify_partition.scheduler_q.resize(verify_tiles);
  verify_partition.final_peer_unit.assign(verify_tiles, uint32_t(-1));
  verify_partition.final_peer_visit.assign(verify_tiles, uint16_t(-1));
  for (std::size_t q = 0; q < verify_tiles; ++q) {
    verify_partition.scheduler_q[q] = uint32_t(q);
  }
  bool const owner_map_closed =
      dense_map_accumulator_owners<Gemm>(verify_partition);
  // An ordinary DP arm needs no K-peer capture plan: its all-DP map is already
  // complete once the MMA owner map closes.  Stream-K fills the remaining
  // peer fields from its accepted lowered Params below.
  verify_partition.classification_closed =
      owner_map_closed && !verify_partition.is_streamk;
#endif

#if defined(DENSE_NAMED_SCHEDULER)
  // Query the FINAL kernel, not a tile-model estimate.  This accounts for registers, dynamic
  // shared memory, threads, and any work-loop register pressure introduced by persistence.
  int current_device = 0;
  CUTLASS_PPU_CHECK(hggcGetDevice(&current_device));
  int const cu_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(current_device);
  int const ctas_per_cu = Gemm::maximum_active_blocks();
  Result occupancy_failure;
  if (cu_count <= 0 || ctas_per_cu <= 0) {
    std::fprintf(stderr,
                 "[dense scheduler=%s] invalid runtime occupancy: cu=%d cta_per_cu=%d\n",
                 scheduler_kind, cu_count, ctas_per_cu);
    occupancy_failure.passed = false;
    return occupancy_failure;
  }
  arguments.hw_info = cutlass::KernelHardwareInfo{current_device, cu_count};
  if constexpr (dense_has_persistent_ctas<decltype(arguments)>::value) {
    arguments.ctas_per_cu = ctas_per_cu;
  }
#if defined(DENSE_STREAMK_AB) || defined(DENSE_MARLIN_AB)
  using StreamKDiagnosticState =
      typename dense_streamk_diagnostic_type<Gemm>::type;
  cutlass::DeviceAllocation<StreamKDiagnosticState> streamk_diagnostics;
  if constexpr (dense_is_streamk_gemm<Gemm>::value) {
    using SchedulerParams = typename Gemm::GemmKernel::TileSchedulerParams;
    arguments.scheduler.splits = 1;
    arguments.scheduler.max_swizzle_size = 1;
    arguments.scheduler.reduction_mode = SchedulerParams::ReductionMode::Deterministic;
    arguments.scheduler.decomposition_mode = SchedulerParams::DecompositionMode::StreamK;
    if (options.streamk_gate) {
      streamk_diagnostics.reset(1);
      StreamKDiagnosticState host_diagnostics{};
      streamk_diagnostics.copy_from_host(&host_diagnostics);
      arguments.diagnostics = streamk_diagnostics.get();
    }
  }
#endif
#endif

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Keep these names available in every instantiation.  The reporting chain
  // below is one syntactic if-constexpr ladder so the Stream-K contract can
  // isolate its own arm even when another named scheduler is compiled in a
  // different target.  Non-Marlin instantiations discard the arm.
  uint64_t marlin_peer_excess = 0;
  uint64_t marlin_valid_fixup_elements = 0;

#if defined(DENSE_NAMED_SCHEDULER)
  dim3 const physical_grid = Gemm::get_grid_shape(arguments, workspace.get());
  uint64_t const logical_ctas =
      uint64_t(cute::ceil_div(options.m, tactic.tm)) *
      uint64_t(cute::ceil_div(options.n, tactic.tn)) * uint64_t(options.l);
  uint64_t const physical_ctas =
      uint64_t(physical_grid.x) * uint64_t(physical_grid.y) * uint64_t(physical_grid.z);
  int const warps_per_cta =
      (int(Gemm::GemmKernel::MaxThreadsPerBlock) + 31) / 32;
  size_t const mainloop_bytes = sizeof(typename Gemm::CollectiveMainloop::SharedStorage);
  size_t const epilogue_bytes = sizeof(typename Gemm::CollectiveEpilogue::SharedStorage);
  size_t const union_bytes = Gemm::GemmKernel::SharedStorageSize;
  size_t const overlap_sum_bytes = mainloop_bytes + epilogue_bytes;
  constexpr size_t kSharedPerCu = 256u << 10;
  size_t const union_shared_ctas = union_bytes ? kSharedPerCu / union_bytes : 0;
  size_t const sum_shared_ctas = overlap_sum_bytes ? kSharedPerCu / overlap_sum_bytes : 0;
#if defined(DENSE_STREAMK_AB)
  if constexpr (dense_is_streamk_gemm<Gemm>::value) {
    using SchedulerParams = typename Gemm::GemmKernel::TileSchedulerParams;
    auto const params = Gemm::GemmKernel::to_underlying_arguments(arguments, workspace.get());
    auto const& sk = params.scheduler;
    uint64_t const workers = uint64_t(cu_count) * uint64_t(ctas_per_cu);
    uint64_t const dp_units = sk.units_per_problem_ >= sk.sk_units_
        ? sk.units_per_problem_ - sk.sk_units_ : 0;
    char const* actual = sk.divmod_splits_.divisor > 1 ? "SplitK" :
                         (sk.sk_tiles_ > 0 && sk.sk_units_ > 0 ? "StreamK" : "DataParallel");
    std::printf(
        "  [dense streamk decomposition] actual=%s real_cu=%d ctas_per_cu=%d "
        "workers=%llu scheduler_workers=%d sk_tiles=%u sk_units=%u dp_units=%llu "
        "units=%llu splits=%d separate=%u workspace=%zu\n",
        actual, cu_count, ctas_per_cu, static_cast<unsigned long long>(workers),
        params.scheduler_hw_info.cu_count, sk.sk_tiles_, sk.sk_units_,
        static_cast<unsigned long long>(dp_units),
        static_cast<unsigned long long>(sk.units_per_problem_),
        sk.divmod_splits_.divisor, sk.separate_reduction_units_, workspace_size);
    bool const valid_decomposition =
        sk.sk_tiles_ > 0 && sk.sk_units_ > 0 &&
        sk.divmod_splits_.divisor == 1 && sk.separate_reduction_units_ == 0 &&
        params.scheduler_hw_info.cu_count == int(workers) &&
        sk.reduction_mode_ == SchedulerParams::ReductionMode::Deterministic;
    if (!valid_decomposition) {
      std::fprintf(stderr,
                   "dense Stream-K fail-close: requested deterministic StreamK but actual decomposition/worker contract differs\n");
      Result decomposition_failure;
      decomposition_failure.passed = false;
      return decomposition_failure;
    }
  }
#endif
  if constexpr (dense_is_marlin_gemm<Gemm>::value) {
    auto const params = Gemm::GemmKernel::to_underlying_arguments(
        arguments, workspace.get());
    auto const& ms = params.scheduler;
    uint64_t handoffs = 0;
    uint64_t const tiles_n = uint64_t(cute::ceil_div(options.n, tactic.tn));
    uint64_t const tiles_m = uint64_t(cute::ceil_div(options.m, tactic.tm));
    if (ms.valid_ && ms.iters_per_block_ > 0) {
      for (uint64_t q = 0; q < ms.output_tiles_; ++q) {
        uint64_t const q_begin = q * ms.k_tiles_per_output_;
        uint64_t const q_end = q_begin + ms.k_tiles_per_output_;
        uint64_t const first = q_begin / ms.iters_per_block_;
        uint64_t const last = (q_end - 1) / ms.iters_per_block_;
        uint64_t const peer_excess = last - first;
        handoffs += peer_excess;
        uint64_t const n_idx = q % tiles_n;
        uint64_t const q_m = q / tiles_n;
        uint64_t const m_idx = q_m % tiles_m;
        uint64_t const m_begin = m_idx * uint64_t(tactic.tm);
        uint64_t const n_begin = n_idx * uint64_t(tactic.tn);
        uint64_t const valid_m = std::min<uint64_t>(
            uint64_t(tactic.tm), uint64_t(options.m) - m_begin);
        uint64_t const valid_n = std::min<uint64_t>(
            uint64_t(tactic.tn), uint64_t(options.n) - n_begin);
        marlin_valid_fixup_elements += valid_m * valid_n * peer_excess;
      }
    }
    marlin_peer_excess = handoffs;
    uint64_t const expected_grid = logical_ctas >= uint64_t(cu_count)
        ? logical_ctas : uint64_t(cu_count);
    uint64_t const expected_kt = uint64_t(cute::ceil_div(options.k, tactic.tk));
    bool const valid_decomposition = ms.valid_ &&
        ms.output_tiles_ == logical_ctas &&
        ms.k_tiles_per_output_ == expected_kt &&
        ms.grid_blocks_ == expected_grid;
    std::printf(
        "  [dense marlin decomposition] real_cu=%d occupancy_api=%d Q=%llu "
        "Kt=%llu G=%llu I=%llu active=%llu handoffs=%llu workspace=%zu\n",
        cu_count, ctas_per_cu,
        static_cast<unsigned long long>(ms.output_tiles_),
        static_cast<unsigned long long>(ms.k_tiles_per_output_),
        static_cast<unsigned long long>(ms.grid_blocks_),
        static_cast<unsigned long long>(ms.iters_per_block_),
        static_cast<unsigned long long>(ms.active_blocks_),
        static_cast<unsigned long long>(handoffs), workspace_size);
    if (!valid_decomposition) {
      std::fprintf(stderr,
                   "dense Marlin fail-close: lowered Q/Kt/G differs from the scheduler-owned launch contract\n");
      Result decomposition_failure;
      decomposition_failure.passed = false;
      return decomposition_failure;
    }
  }
  std::printf(
      "  [dense scheduler=%s] logical_cta=%llu cu=%d occupancy_api=%d "
      "grid=(%u,%u,%u) physical_cta=%llu block_threads=%u warps/cta=%d "
      "resident_warps/cu=%d\n",
      scheduler_kind, static_cast<unsigned long long>(logical_ctas), cu_count,
      ctas_per_cu, physical_grid.x, physical_grid.y, physical_grid.z,
      static_cast<unsigned long long>(physical_ctas),
      unsigned(Gemm::GemmKernel::MaxThreadsPerBlock), warps_per_cta,
      ctas_per_cu * warps_per_cta);
  std::printf(
      "  [dense smem scheduler=%s] main=%zu epi=%zu union=%zu "
      "overlap-sum-counterfactual=%zu shared-only-cta/cu=%zu->%zu\n",
      scheduler_kind, mainloop_bytes, epilogue_bytes, union_bytes,
      overlap_sum_bytes, union_shared_ctas, sum_shared_ctas);
  if constexpr (dense_is_streamk_gemm<Gemm>::value) {
    uint64_t const expected = uint64_t(cu_count) * uint64_t(ctas_per_cu);
    if (physical_ctas != expected) {
      std::fprintf(stderr,
                   "Stream-K grid mismatch: got %llu CTA, expected full worker grid %d*%d=%llu\n",
                   static_cast<unsigned long long>(physical_ctas), cu_count, ctas_per_cu,
                   static_cast<unsigned long long>(expected));
      Result grid_failure;
      grid_failure.passed = false;
      return grid_failure;
    }
  } else if constexpr (dense_is_marlin_gemm<Gemm>::value) {
    uint64_t const expected = logical_ctas >= uint64_t(cu_count)
        ? logical_ctas : uint64_t(cu_count);
    if (physical_ctas != expected || physical_grid.y != 1 || physical_grid.z != 1) {
      std::fprintf(stderr,
                   "Marlin grid mismatch: got (%u,%u,%u)=%llu CTA, expected max(Q=%llu,CU=%d)=%llu (occupancy must not multiply it)\n",
                   physical_grid.x, physical_grid.y, physical_grid.z,
                   static_cast<unsigned long long>(physical_ctas),
                   static_cast<unsigned long long>(logical_ctas), cu_count,
                   static_cast<unsigned long long>(expected));
      Result grid_failure;
      grid_failure.passed = false;
      return grid_failure;
    }
  } else if constexpr (dense_has_persistent_ctas<decltype(arguments)>::value) {
    uint64_t const expected = std::min(
        logical_ctas, uint64_t(cu_count) * uint64_t(ctas_per_cu));
    if (physical_ctas != expected) {
      std::fprintf(stderr,
                   "persistent grid mismatch: got %llu CTA, expected min(%llu,%d*%d)=%llu\n",
                   static_cast<unsigned long long>(physical_ctas),
                   static_cast<unsigned long long>(logical_ctas), cu_count,
                   ctas_per_cu, static_cast<unsigned long long>(expected));
      Result grid_failure;
      grid_failure.passed = false;
      return grid_failure;
    }
  } else if (physical_ctas != logical_ctas) {
    std::fprintf(stderr,
                 "non-persistent grid mismatch: got %llu CTA for %llu logical tiles\n",
                 static_cast<unsigned long long>(physical_ctas),
                 static_cast<unsigned long long>(logical_ctas));
    Result grid_failure;
    grid_failure.passed = false;
    return grid_failure;
  }
#endif

  Result result;
  // Check if the problem size is supported or not. Do not hard-CHECK (which aborts): during a search an
  // unsupported tile must be skippable, so return a failed Result instead of killing the process.
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) { result.passed = false; return result; }
  if (gemm.initialize(arguments, workspace.get()) != cutlass::Status::kSuccess) { result.passed = false; return result; }

#if defined(DENSE_STREAMK_AB)
  if constexpr (dense_is_streamk_gemm<Gemm>::value) {
    static_assert(!Gemm::CollectiveMainloop::SwapAB,
                  "107b DP/SK diagnostic maps the unswapped dense output-tile axes");
    // Lower only after can_implement()/initialize() accepted the arm.  A Params
    // object from a rejected request is not decomposition evidence.
    auto const diagnostic_params =
        Gemm::GemmKernel::to_underlying_arguments(arguments, workspace.get());
    verify_partition.classification_closed =
        owner_map_closed && dense_classify_streamk_tiles(
            verify_partition, diagnostic_params.scheduler);
    // Every Stream-K arm that claims numerical evidence must actually contain
    // a peer seam.  This used to guard only --streamk_split_gate, so fixed A0
    // could print an ordinary Failed disposition while the same-order replay
    // silently ran on the tiny, already-bit-exact seam fixture instead.  A
    // machine whose worker count divides A0 then made that failure even worse:
    // zero split tiles looked like an exercised numerical arm.  The lowered
    // scheduler, not the command-line spelling, decides whether this run has a
    // subject to replay.
    if (options.streamk && verify_partition.classification_closed) {
      uint64_t const workers =
          uint64_t(cu_count) * uint64_t(ctas_per_cu);
      uint64_t const remainder = logical_ctas % workers;
      if (verify_partition.split_tiles == 0 ||
          verify_partition.peer_excess == 0) {
        char const* reason = remainder == 0
            ? "complete-worker-waves"
            : "lowered-scheduler-produced-no-peer-seam";
        std::printf(
            "  [dense streamk split gate] NOT EXERCISED real_cu=%d "
            "ctas_per_cu=%d workers=%llu logical_cta=%llu "
            "logical_cta%%workers=%llu%%%llu=%llu SK-split=%llu "
            "peer_excess=%llu reason=%s\n",
            cu_count, ctas_per_cu,
            static_cast<unsigned long long>(workers),
            static_cast<unsigned long long>(logical_ctas),
            static_cast<unsigned long long>(logical_ctas),
            static_cast<unsigned long long>(workers),
            static_cast<unsigned long long>(remainder),
            static_cast<unsigned long long>(verify_partition.split_tiles),
            static_cast<unsigned long long>(verify_partition.peer_excess), reason);
        std::cout << "  Disposition: NOT EXERCISED" << std::endl;
        result.passed = false;
        result.split_path_exercised = false;
        return result;
      }
      std::printf(
          "  [dense streamk split gate] EXERCISED real_cu=%d ctas_per_cu=%d "
          "workers=%llu logical_cta=%llu logical_cta%%workers=%llu%%%llu=%llu "
          "SK-split=%llu peer_excess=%llu\n",
          cu_count, ctas_per_cu,
          static_cast<unsigned long long>(workers),
          static_cast<unsigned long long>(logical_ctas),
          static_cast<unsigned long long>(logical_ctas),
          static_cast<unsigned long long>(workers),
          static_cast<unsigned long long>(remainder),
          static_cast<unsigned long long>(verify_partition.split_tiles),
          static_cast<unsigned long long>(verify_partition.peer_excess));
    }
  }
#endif

  // Correctness / Warmup iteration
#if defined(DENSE_STREAMK_AB)
  // Create the whole pool before warmup so first-use event initialization
  // cannot perturb timed launch 1. Pair zero belongs only to warmup; every
  // timed launch below owns a distinct remaining pair.
  DenseKernelEventBatch streamk_events(std::max(options.iterations, 0) + 1);
#endif
#if defined(DENSE_NAMED_SCHEDULER)
  // Make a missed scheduler tile deterministic: 0xffff is a half NaN, so an unwritten D
  // element cannot accidentally compare equal to the reference.  This is outside all timing.
  CUTLASS_PPU_CHECK(hggcMemset(block_D.get(), 0xff,
                              block_D.size() * sizeof(typename Gemm::ElementD)));
#endif
#if defined(DENSE_STREAMK_AB)
  CUTLASS_PPU_CHECK(hggcEventRecord(streamk_events.at(0).start, nullptr));
#endif
  CUTLASS_CHECK(gemm.run());
#if defined(DENSE_STREAMK_AB)
  CUTLASS_PPU_CHECK(hggcEventRecord(streamk_events.at(0).stop, nullptr));
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
#endif

  // Check if output from kernel and reference kernel are equal or not
  DenseReplayEvidence replay_evidence;
#if defined(DENSE_STREAMK_AB)
  DenseVerifyState ordinary_diagnostic_state =
      DenseVerifyState::NotClassifiable;
  result.passed = verify(
      options, fixture_evidence, &verify_partition, &ordinary_diagnostic_state);
  bool const ordinary_reference_passed = result.passed;
  bool replay_attempted = false;
  result.verification_classified =
      ordinary_diagnostic_state != DenseVerifyState::NotClassifiable;
#else
  result.passed = verify(options, fixture_evidence, nullptr);
#endif
#if defined(DENSE_STREAMK_AB)
  if constexpr (dense_is_streamk_gemm<Gemm>::value) {
    // This is the fixup oracle for the actual Stream-K arm, including
    // fixed/adaptive A0.  It says whether production fixup matches the captured
    // partials.  Fixture exactness, carried separately, decides whether a
    // whole-K reference difference is a numerical failure or unclassifiable.
    if (options.streamk) {
      if (ordinary_diagnostic_state == DenseVerifyState::NotClassifiable) {
        std::fprintf(stderr,
                     "[streamk same-order replay] NOT CLASSIFIABLE: "
                     "the scheduler tile/capture map did not close\n");
        result.verification_classified = false;
      }
      else if (ordinary_diagnostic_state ==
               DenseVerifyState::InvariantFailure) {
        std::fprintf(stderr,
                     "[streamk same-order replay] fail-close: the claimed "
                     "bucket map violated its own invariant\n");
        result.passed = false;
      }
      else {
        std::size_t const tile_elements =
            std::size_t(verify_partition.tile_m) * verify_partition.tile_n;
        std::size_t const capture_slots =
            verify_partition.split_peer_ranges.size();
        std::size_t const capture_scalars = capture_slots * tile_elements;
        if (capture_slots == 0 || capture_scalars == 0 ||
            verify_partition.capture_slot_by_qk.empty()) {
          std::fprintf(stderr,
                       "[streamk same-order replay] empty compact capture plan\n");
          result.passed = false;
        }
        else {
          std::vector<ElementD> normal_output(block_D.size());
          block_D.copy_to_host(normal_output.data());

          cutlass::DeviceAllocation<ElementAccumulator> pre_fixup_capture(
              capture_scalars);
          cutlass::DeviceAllocation<int32_t> pre_fixup_slot_map(
              verify_partition.capture_slot_by_qk.size());
          cutlass::DeviceAllocation<uint32_t> pre_fixup_slot_visits(
              capture_slots);
          cutlass::DeviceAllocation<uint32_t> pre_fixup_slot_k_counts(
              capture_slots);
          pre_fixup_slot_map.copy_from_host(
              verify_partition.capture_slot_by_qk.data());
          CUTLASS_PPU_CHECK(hggcMemset(
              pre_fixup_capture.get(), 0xff,
              capture_scalars * sizeof(ElementAccumulator)));
          CUTLASS_PPU_CHECK(hggcMemset(
              pre_fixup_slot_visits.get(), 0,
              capture_slots * sizeof(uint32_t)));
          CUTLASS_PPU_CHECK(hggcMemset(
              pre_fixup_slot_k_counts.get(), 0xff,
              capture_slots * sizeof(uint32_t)));
          streamk_diagnostics.reset(1);
          StreamKDiagnosticState host_diagnostics{};
          host_diagnostics.pre_fixup_capture_magic =
              Gemm::GemmKernel::PreFixupCaptureMagic;
          host_diagnostics.pre_fixup_capture = pre_fixup_capture.get();
          host_diagnostics.pre_fixup_capture_slot_map =
              pre_fixup_slot_map.get();
          host_diagnostics.pre_fixup_capture_slot_visits =
              pre_fixup_slot_visits.get();
          host_diagnostics.pre_fixup_capture_slot_k_counts =
              pre_fixup_slot_k_counts.get();
          host_diagnostics.pre_fixup_capture_k_stride =
              verify_partition.k_tiles_per_output_tile;
          host_diagnostics.pre_fixup_capture_slot_capacity = capture_slots;
          host_diagnostics.pre_fixup_capture_map_capacity =
              verify_partition.capture_slot_by_qk.size();
          streamk_diagnostics.copy_from_host(&host_diagnostics);
          arguments.diagnostics = streamk_diagnostics.get();

          // A second, correctness-only launch captures the exact production
          // mainloop partials.  Reset locks first and require its final D to be
          // bit-identical to the uninstrumented launch, so instrumentation
          // cannot silently "fix" the phenomenon it is meant to classify.
          CUTLASS_PPU_CHECK(hggcMemset(
              block_D.get(), 0xff, block_D.size() * sizeof(ElementD)));
          if (gemm.initialize(arguments, workspace.get()) !=
                  cutlass::Status::kSuccess ||
              Gemm::GemmKernel::initialize_workspace(
                  arguments, workspace.get(), /*stream=*/nullptr) !=
                  cutlass::Status::kSuccess ||
              gemm.run() != cutlass::Status::kSuccess) {
            std::fprintf(stderr,
                         "[streamk same-order replay] capture launch failed\n");
            result.passed = false;
          }
          else {
            CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
            std::vector<ElementD> capture_output(block_D.size());
            std::vector<ElementAccumulator> host_capture(capture_scalars);
            std::vector<uint32_t> host_slot_visits(capture_slots);
            std::vector<uint32_t> host_slot_k_counts(capture_slots);
            StreamKDiagnosticState captured_diagnostics{};
            block_D.copy_to_host(capture_output.data());
            pre_fixup_capture.copy_to_host(host_capture.data());
            pre_fixup_slot_visits.copy_to_host(host_slot_visits.data());
            pre_fixup_slot_k_counts.copy_to_host(host_slot_k_counts.data());
            streamk_diagnostics.copy_to_host(&captured_diagnostics);
            replay_attempted = true;
            replay_evidence = verify_streamk_same_order_partial_replay<Gemm>(
                options, verify_partition, normal_output, capture_output,
                host_capture, host_slot_visits, host_slot_k_counts,
                captured_diagnostics.pre_fixup_capture_error_count);
          }
        }
      }
    }
    if (options.streamk && result.verification_classified) {
      uint64_t const reference_raw_bitdiff =
          replay_evidence.split_reference_bitdiff +
          replay_evidence.non_split_reference_bitdiff;
      if (!replay_attempted || !replay_evidence.fixup_closed) {
        result.passed = false;
      }
      else if (fixture_evidence.order_independent) {
        // Exact arithmetic removes reassociation from the hypothesis space.
        // The replay still localises a failure: closed fixup plus a reference
        // difference means the captured mainloop partials are already wrong.
        result.passed = ordinary_reference_passed &&
            reference_raw_bitdiff == 0;
      }
      else if (reference_raw_bitdiff == 0) {
        result.passed = true;
      }
      else {
        // Random A0 rounds.  A closed replay proves only that fixup consumed
        // the captured partials correctly; it cannot classify the partials
        // against a differently-parenthesised whole-K reference.
        result.passed = false;
        result.verification_classified = false;
      }
    }
    if (options.streamk_gate) {
      StreamKDiagnosticState host_diagnostics{};
      streamk_diagnostics.copy_to_host(&host_diagnostics);
      uint32_t const* witness = host_diagnostics.witness;
      std::printf("  [streamk witness] fixup_work=%u epilogue_work=%u "
                  "separate_reduction_work=%u\n",
                  witness[0], witness[1], witness[2]);
      bool const witness_ok = witness[0] == 8 && witness[1] == 1 && witness[2] == 0;
      if (!witness_ok) {
        std::fprintf(stderr,
                     "[streamk witness] expected exact 107b seam decomposition 8/1/0\n");
      }
      bool const cpu_ok = verify_streamk_cpu_fp32(options);
      result.passed = result.passed && witness_ok && cpu_ok;
    }
    // The replay capture (and the exact-fixture witness) points `arguments` at
    // a correctness-only allocation whose lifetime ends before run() returns.
    // It must never leak into the timing loop.  Reinitialising after clearing it
    // also proves the timed launch is the ordinary, uninstrumented kernel whose
    // output was compared by capture_vs_normal_bitdiff above.
    if (options.iterations > 0) {
      arguments.diagnostics = nullptr;
      if (gemm.initialize(arguments, workspace.get()) != cutlass::Status::kSuccess) {
        result.passed = false;
      }
    }
  }
#endif

  if (!result.verification_classified) {
    std::cout << "  Disposition: NOT CLASSIFIABLE "
              << "(fixture rounds; fixup replay closed but partial correctness "
                 "is not established)" << std::endl;
  }
  else if (options.streamk) {
    if (result.passed) {
      std::cout << "  Disposition: Passed (whole-K reference bit-exact; "
                   "fixup replay closed)" << std::endl;
    }
    else if (fixture_evidence.order_independent && replay_evidence.fixup_closed) {
      std::cout << "  Disposition: Failed (ORDER-INDEPENDENT fixture differs "
                   "from whole-K reference; fixup replay closed, so the "
                   "discrepancy is upstream of fixup)" << std::endl;
    }
    else {
      std::cout << "  Disposition: Failed (StreamK capture/fixup replay did not close)"
                << std::endl;
    }
  }
  else {
    std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed")
              << std::endl;
  }

  if (!result.passed) {
#if defined(DENSE_STREAMK_AB)
    if (options.streamk_gate) return result;
#endif
    // A failed numerical arm remains failed.  When timing was explicitly
    // requested, retain its timing only as a diagnostic alongside that failure;
    // do not guess whether the reference or the kernel is responsible.
    if (options.iterations <= 0) return result;
    std::cout << "  (verify failed; timing was requested, so this failed arm is timed only for diagnosis)\n";
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
#if defined(DENSE_STREAMK_AB)
    // One distinct event pair per launch for every arm in this binary.  A
    // Stream-K launch leaves its turnstile lock at the completed K count, so
    // reset scheduler workspace on the same stream before (and outside) the
    // start event.  The old 107a binary retains its original aggregate-event
    // protocol; no historical row silently changes meaning.
    std::vector<double> launch_ms;
    launch_ms.reserve(options.iterations);
    for (int iter = 0; iter < options.iterations; ++iter) {
      if constexpr (dense_is_streamk_gemm<Gemm>::value) {
        CUTLASS_CHECK(Gemm::GemmKernel::initialize_workspace(
            arguments, workspace.get(), /*stream=*/nullptr));
      }
      auto& events = streamk_events.at(iter + 1);
      CUTLASS_PPU_CHECK(hggcEventRecord(events.start, nullptr));
      CUTLASS_CHECK(gemm.run());
      CUTLASS_PPU_CHECK(hggcEventRecord(events.stop, nullptr));
    }
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    for (int iter = 0; iter < options.iterations; ++iter) {
      auto const& events = streamk_events.at(iter + 1);
      float elapsed_ms = 0.0f;
      CUTLASS_PPU_CHECK(hggcEventElapsedTime(&elapsed_ms, events.start, events.stop));
      if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f) {
        std::fprintf(stderr,
                     "dense kernel-span event %d/%d is invalid (%g ms); no timing fallback\n",
                     iter + 1, options.iterations, double(elapsed_ms));
        result.passed = false;
        return result;
      }
      launch_ms.push_back(double(elapsed_ms));
    }
    double mean_ms = 0.0;
    for (double ms : launch_ms) mean_ms += ms;
    mean_ms /= double(launch_ms.size());
    std::sort(launch_ms.begin(), launch_ms.end());
    double const median_ms = launch_ms.size() & 1
        ? launch_ms[launch_ms.size() / 2]
        : 0.5 * (launch_ms[launch_ms.size() / 2 - 1] + launch_ms[launch_ms.size() / 2]);
    double const spread_pct = mean_ms > 0.0
        ? 100.0 * (launch_ms.back() - launch_ms.front()) / mean_ms : 0.0;
    result.avg_runtime_ms = median_ms;
    std::printf("  [dense kernel-span-upper] n=%zu median=%.3f us mean=%.3f us "
                "min=%.3f us max=%.3f us spread=(max-min)/mean=%.2f%% "
                "distinct-event-pairs=%zu warmup-event-pairs=1 includes-launch-idle=1 "
                "lock-reset-before-start=%d\n",
                launch_ms.size(), median_ms * 1.0e3, mean_ms * 1.0e3,
                launch_ms.front() * 1.0e3, launch_ms.back() * 1.0e3,
                spread_pct, launch_ms.size(), dense_is_streamk_gemm<Gemm>::value ? 1 : 0);
#else
    PpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
#endif

    // Compute runtime-normalized throughput.  The 107b target uses the median
    // of independent kernel spans; the existing targets retain their mean.
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
#if defined(DENSE_STREAMK_AB)
    std::cout << "  Median runtime: " << result.avg_runtime_ms << " ms" << std::endl;
#else
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
#endif
    std::cout << "  GFLOPS: " << result.gflops << std::endl;

    // Report BOTH traffic ends -- neither alone tells you what is binding:
    //   distinct = the ALGORITHMIC minimum traffic (A + B + D each touched once) vs HBM peak. This is what a
    //              perfect kernel would move from HBM; at prefill M it is tiny, which is why MFU is the headline.
    //   tile     = the TILE-LEVEL traffic REQUESTED: A once per N-tile, B once per M-tile. It is GB/s plus a
    //              REUSE FACTOR over distinct, NOT a percentage of HBM.
    //
    //   WHY NOT AGAINST HBM, corrected 2026-08-05 after a box run printed "195.2% HBM". Tile traffic is what the
    //   kernel ASKS FOR; L2 serves the re-reads, so measuring it against the DRAM peak is a category error and any
    //   value above 100% is proof of that rather than of a bandwidth problem. The figure is still worth printing --
    //   a large reuse factor is what a narrow TileN costs -- but whether it binds depends on L2, which this bench
    //   does not measure.
    //
    //   THE OLD COMMENT HERE DREW A CAUSAL CONCLUSION FROM IT, and that is the reason for the rewrite: it said
    //   int1 32x128 at 2295 GB/s "= 83% of HBM, i.e. it is BANDWIDTH-bound on the A re-reads even though cmp is
    //   only 48%". 83% of tile-level traffic does not establish DRAM-bound either; the same L2 objection applies,
    //   and that claim should be treated as unproven until something measures actual DRAM traffic (acu can).
    const double us = result.avg_runtime_ms * 1e3;
    // NOTE: the w%d tag (w4/w2/w1) is essential -- the cfg label alone is identical across quant builds and the
    // numbers are easy to mix up between them (int4/int2/int1 all have a "64x64:32x32:s3").
    // distinct IS the only figure with a legitimate HBM denominator: it counts every byte once, which is what DRAM
    // must actually deliver. tile carries a reuse factor instead, and a marker when it exceeds the DRAM peak --
    // which says the re-reads are cache-served, not that the kernel is bandwidth-bound.
    char tag[bench_measure::kTagBytes];
    bench_measure::format_tag(tag, sizeof tag, tactic);
#if defined(DENSE_STREAMK_AB)
    if constexpr (dense_is_streamk_gemm<Gemm>::value) {
      const double Mm = options.m, Nn = options.n, Kk = options.k;
      const double bq = double(sizeof_bits<QuantType>::value) / 8.0;
      const double n_tiles = tactic.tn > 0 ? std::ceil(Nn / tactic.tn) : 1.0;
      const double m_tiles = tactic.tm > 0 ? std::ceil(Mm / tactic.tm) : 1.0;
      bench_measure::TiledGemmTrafficInput const traffic_input{
          Mm * Kk * 2.0, Nn * Kk * bq, 0.0, Mm * Nn * 2.0,
          1.0, n_tiles, m_tiles};
      double const logical_fixup_bytes =
          2.0 * sizeof(float) * double(verify_partition.valid_fixup_elements);
      double const modeled_output_bytes =
          traffic_input.output_bytes + logical_fixup_bytes;
      const auto traffic = bench_measure::make_traffic_with_output_bytes(
          traffic_input, modeled_output_bytes);
      const auto metrics = bench_measure::measure(
          us, 2.0 * Mm * Nn * Kk * double(options.l), traffic);
      char core[256];
      bench_measure::format_metrics(core, sizeof core, metrics);
      std::printf(
          "  [CUTLASS w%d gs=%d cfg=%s scheduler=%s] M=%d %7.2f us | "
          "%s | StreamK-C valid_elements=%llu peer_excess=%llu "
          "logical_RW=%.0f MODEL-ONLY/not-a-DRAM-counter\n",
          int(sizeof_bits<QuantType>::value), options.g, tag, scheduler_kind,
          options.m, us, core,
          static_cast<unsigned long long>(verify_partition.valid_fixup_elements),
          static_cast<unsigned long long>(verify_partition.peer_excess),
          logical_fixup_bytes);
    } else if constexpr (dense_is_marlin_gemm<Gemm>::value) {
#else
    if constexpr (dense_is_marlin_gemm<Gemm>::value) {
#endif
      const double Mm = options.m, Nn = options.n, Kk = options.k;
      const double bq = double(sizeof_bits<QuantType>::value) / 8.0;
      const double n_tiles = tactic.tn > 0 ? std::ceil(Nn / tactic.tn) : 1.0;
      const double m_tiles = tactic.tm > 0 ? std::ceil(Mm / tactic.tm) : 1.0;
      bench_measure::TiledGemmTrafficInput const traffic_input{
          Mm * Kk * 2.0, Nn * Kk * bq, 0.0, Mm * Nn * 2.0,
          1.0, n_tiles, m_tiles};
      // Each extra peer contributes one predicated FP32 store/load pair.
      // This is the logical B2-lite model, not a DRAM counter: cache-line
      // amplification remains a box-side measurement.
      double const logical_fixup_bytes =
          2.0 * sizeof(float) * double(marlin_valid_fixup_elements);
      const auto traffic = bench_measure::make_traffic_with_output_bytes(
          traffic_input, traffic_input.output_bytes + logical_fixup_bytes);
      const auto metrics = bench_measure::measure(
          us, 2.0 * Mm * Nn * Kk * double(options.l), traffic);
      char core[256];
      bench_measure::format_metrics(core, sizeof core, metrics);
      std::printf(
          "  [CUTLASS w%d gs=%d cfg=%s scheduler=%s] M=%d %7.2f us | "
          "%s | Marlin-C valid_elements=%llu peer_excess=%llu "
          "logical_RW=%.0f MODEL-ONLY/not-a-DRAM-counter\n",
          int(sizeof_bits<QuantType>::value), options.g, tag, scheduler_kind,
          options.m, us, core,
          static_cast<unsigned long long>(marlin_valid_fixup_elements),
          static_cast<unsigned long long>(marlin_peer_excess),
          logical_fixup_bytes);
    } else
    {
      const double Mm = options.m, Nn = options.n, Kk = options.k;
      const double bq = double(sizeof_bits<QuantType>::value) / 8.0;
      const double n_tiles = tactic.tn > 0 ? std::ceil(Nn / tactic.tn) : 1.0;
      const double m_tiles = tactic.tm > 0 ? std::ceil(Mm / tactic.tm) : 1.0;
      // Preserve the established dense model. Stream-K uses the same base terms
      // above, but supplies its per-q valid-residue C term explicitly rather
      // than fabricating one uniform split count.
      const auto traffic = bench_measure::make_traffic({
          Mm * Kk * 2.0, Nn * Kk * bq, 0.0, Mm * Nn * 2.0,
          1.0, n_tiles, m_tiles});
      const auto metrics = bench_measure::measure(
          us, 2.0 * Mm * Nn * Kk * double(options.l), traffic);
      char core[256];
      bench_measure::format_metrics(core, sizeof core, metrics);
#if defined(DENSE_NAMED_SCHEDULER)
      std::printf("  [CUTLASS w%d gs=%d cfg=%s scheduler=%s] M=%d %7.2f us | %s\n",
                  int(sizeof_bits<QuantType>::value), options.g, tag, scheduler_kind,
                  options.m, us, core);
#else
      std::printf("  [CUTLASS w%d gs=%d cfg=%s] M=%d %7.2f us | %s\n",
                  int(sizeof_bits<QuantType>::value), options.g, tag, options.m, us, core);
#endif
    }
  }

  return result;
}

#if !defined(LOWBIT_DENSE_UNIT_BUILD)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Tactic dispatch + shape-keyed cache (exact-match text, like machete's cutlass55_tactics.cache).

// BENCH_GS: BUILD ONE GROUP SIZE INSTEAD OF FOUR. Each generated wrapper switches on a RUNTIME --g, so every arm
// is instantiated whether or not it can be selected. Restricting it makes each wrapper instantiate one kernel
// instead of four.
//
// HOW MUCH BUILD TIME THAT SAVES IS NOT ESTABLISHED, and the first version of this comment claimed 4x. Measured
// on the local nvcc FRONT END: 194s for all four, 136s for BENCH_GS=32 -- 1.43x, because much of a front-end pass
// is fixed cost in the headers. But the front end stops before codegen, and per-kernel code generation is exactly
// where a 4x instantiation count would show, so that measurement neither confirms nor refutes the saving on the
// box; it only bounds the front-end part. Nobody has timed hgcc with and without it. The instantiation count IS
// four times smaller -- that is read off the dispatch, not timed -- so this is strictly less work, by an unknown
// amount.
//
// It is a build-time restriction and NOT a default: leaving it unset keeps the one-binary --g contract that
// everything else in this repo assumes. An unsupported --g still reports itself at run time rather than
// mis-selecting, so a binary built for one group size cannot silently answer for another:
//   BENCH_GS=32 ./build.sh

bool supported_group_size(int group_size) {
#if defined(BENCH_GS)
  return group_size == BENCH_GS;
#else
  return group_size == 16 || group_size == 32 || group_size == 64 || group_size == 128;
#endif
}

Result run_scale_only(Options& options) {
  switch (options.g) {
#if DENSE_GS_ARM(16)
    case 16:  return run<typename GroupKernels<16>::ScaleOnly>(
        options, dense_fixed_tactic<typename GroupKernels<16>::ScaleOnlyPolicy>());
#endif
#if DENSE_GS_ARM(32)
    case 32:  return run<typename GroupKernels<32>::ScaleOnly>(
        options, dense_fixed_tactic<typename GroupKernels<32>::ScaleOnlyPolicy>());
#endif
#if DENSE_GS_ARM(64)
    case 64:  return run<typename GroupKernels<64>::ScaleOnly>(
        options, dense_fixed_tactic<typename GroupKernels<64>::ScaleOnlyPolicy>());
#endif
#if DENSE_GS_ARM(128)
    case 128: return run<typename GroupKernels<128>::ScaleOnly>(
        options, dense_fixed_tactic<typename GroupKernels<128>::ScaleOnlyPolicy>());
#endif
    default:  std::fprintf(stderr, "unsupported dense group size %d (supported: 16, 32, 64, 128)\n", options.g);
              return {};
  }
}

Result run_scale_zero(Options& options) {
  switch (options.g) {
#if DENSE_GS_ARM(16)
    case 16:  return run<typename GroupKernels<16>::ScaleZero>(
        options, dense_fixed_tactic<typename GroupKernels<16>::ScaleZeroPolicy>());
#endif
#if DENSE_GS_ARM(32)
    case 32:  return run<typename GroupKernels<32>::ScaleZero>(
        options, dense_fixed_tactic<typename GroupKernels<32>::ScaleZeroPolicy>());
#endif
#if DENSE_GS_ARM(64)
    case 64:  return run<typename GroupKernels<64>::ScaleZero>(
        options, dense_fixed_tactic<typename GroupKernels<64>::ScaleZeroPolicy>());
#endif
#if DENSE_GS_ARM(128)
    case 128: return run<typename GroupKernels<128>::ScaleZero>(
        options, dense_fixed_tactic<typename GroupKernels<128>::ScaleZeroPolicy>());
#endif
    default:  std::fprintf(stderr, "unsupported dense group size %d (supported: 16, 32, 64, 128)\n", options.g);
              return {};
  }
}

// THE SAMPLE'S IDENTITY FIELDS, in one place. They are built twice per candidate -- once for the attempt record
// written before the launch, once for the sample written after -- and two spellings of "which run is this" that
// can disagree is the defect the attempt record exists to avoid in the first place.
inline char const* dense_schema() {
  return cutlass::sizeof_bits<QuantType>::value == 4 ? "i4"
       : cutlass::sizeof_bits<QuantType>::value == 2 ? "i2" : "i1";
}

inline char const* dense_fixture(Options const& o) {
  // Static, because Sample holds a `char const*` and does not own its strings; the shape does not change within
  // a process, so one buffer is correct and a per-call one would dangle.
  static char fx[96];
  std::snprintf(fx, sizeof fx, "dense-m%d-n%d-k%d-gs%d", o.m, o.n, o.k, o.g);
  return fx;
}

inline char const* dense_sample_family() {
#if defined(DENSE_MARLIN_SWEEP)
  return "cutlass_w4a16_marlin";
#else
  return "cutlass_w4a16";
#endif
}

inline char const* dense_distribution() {
#if defined(DENSE_MARLIN_SWEEP)
  return "dense-marlin-v1";
#else
  return "dense-v1";
#endif
}

Result run_config(Options& options, TileCfg const& cfg) {
  return cfg.wrapper(options, cfg);
}

TileCfg find_config(std::string const& name) {
  for (auto const& c : supported_configs()) {
    if (name == c.name) return c;
    // Compatibility input only. Before dense and MoE shared one formatter, dense persisted this compact form in
    // --tactic caches and accepted it through --config. New output uses the canonical tag, but old measurements
    // remain reproducible rather than becoming unreadable overnight.
    char legacy[bench_measure::kTagBytes];
    std::snprintf(legacy, sizeof legacy, "%dx%dx%d:%dx%d:s%d:bc%d->%d",
                  c.tm, c.tn, c.tk, c.wm, c.wn, c.st, c.b_chunk, c.b_chunk_effective);
    if (name == legacy) return c;
  }
  std::fprintf(stderr, "unknown config '%s'; use --list_configs\n", name.c_str());
  std::exit(1);
}

std::string tactic_key(Options const& o) {  // shape identity for the cache
  return std::to_string(o.m) + "," + std::to_string(o.n) + "," + std::to_string(o.k) + "," + std::to_string(o.g);
}

// Cache line: "m,n,k,g|config=<name>,tflops=<x>". Returns the config name for this shape, or "" if absent.
std::string load_tactic(std::string const& path, Options const& o) {
  std::ifstream f(path); if (!f) return "";
  std::string const prefix = tactic_key(o) + "|config=";
  std::string line;
  while (std::getline(f, line)) {
    auto p = line.find(prefix);
    if (p == 0) { auto s = line.substr(prefix.size()); return s.substr(0, s.find(',')); }
  }
  return "";
}

void save_tactic(std::string const& path, Options const& o, std::string const& name, double tflops) {
  std::vector<std::string> keep; std::ifstream in(path); std::string line;
  std::string const key = tactic_key(o);
  while (std::getline(in, line)) if (line.rfind(key + "|", 0) != 0 && !line.empty()) keep.push_back(line);
  in.close();
  std::ofstream out(path);
  if (!out) { std::fprintf(stderr, "cannot write tactic cache %s\n", path.c_str()); return; }
  for (auto const& l : keep) out << l << "\n";
  out << key << "|config=" << name << ",tflops=" << tflops << "\n";
}

// ================================ GROUPED KERNEL CROSS-CHECK (L=1) ================================
// The decisive root-cause test. Reuses this file's initialize()-produced data VERBATIM:
//   block_A         -- the same A the verified mixed kernel uses,
//   block_B_buff    -- B already shuffled by the TRUSTED preprocess_weights_for_mixed_gemm (interleave-256),
//   block_scale     -- the same scales, same stride convention,
//   block_ref_D     -- the VERIFIED reference (dequantize_weight -> fp16 GemmRef), filled by verify().
// It runs the grouped kernel as a single expert (group_shape = {m,n,k}, no ragged offset) and compares its D
// against block_ref_D. NO hand-rolled dequant/packing/orientation here -- that is exactly the thing under
// suspicion in test_moe_grouped_verify. Outcome interpretation:
//   MATCH    -> grouped kernel + strides are correct; the earlier MISMATCH was my hand-rolled verify setup.
//   MISMATCH -> the grouped no-swap stride convention (esp. StrideB from the interleaved LayoutB) is wrong;
//               compare vs block_D (the mixed kernel's own output) to localize.
template <class QT = QuantType>
void xcheck_grouped(Options const& options) {
  if constexpr (sizeof_bits<QT>::value < 2) {
    // W1A16: the grouped cross-check is hard-wired to TK=128, but int1 requires Block_K % 256 == 0 (AIU 32B min),
    // so filter_and_run<...,uint1b_t> at TK=128 is ill-formed (BlockContSize%32 static_assert). int1's grouped
    // path is validated by test_w1a16_diag (L=1) and test_w1a16_grouped instead. Discard this branch for int1.
    std::printf("\n[xcheck grouped] SKIPPED for W1A16 (grouped cross-check fixed at TK=128, invalid for int1)\n");
    (void)options;
  } else {
  using GS = moe_grouped_ppu::GroupShape;
  int const L = 1, m = options.m, n = options.n, k = options.k, g = options.g;

  std::vector<GS> host_shapes(L, cute::make_shape(m, n, k));
  cutlass::DeviceAllocation<GS> dev_shapes(L); dev_shapes.copy_from_host(host_shapes.data());
  cutlass::DeviceAllocation<cutlass::half_t> Dgrp((size_t)m * n);
  size_t const ws_bytes = (size_t)cutlass::ceil_div(m, 16) * cutlass::ceil_div(n, 64) * L * 64;
  cutlass::DeviceAllocation<char> ws(ws_bytes);
  // ptr-array (contiguous) output; L=1 so plane 0 is the whole [m][n].
  using DStride = moe_grouped_ppu::DStride;
  std::vector<cutlass::half_t*> pdh{Dgrp.get()};
  std::vector<DStride> sdh{cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(m, n, 1))};
  std::vector<int> gmh{m};
  cutlass::DeviceAllocation<cutlass::half_t*> pd(L); pd.copy_from_host(pdh.data());
  cutlass::DeviceAllocation<DStride> sd(L); sd.copy_from_host(sdh.data());
  cutlass::DeviceAllocation<int>     gm(L); gm.copy_from_host(gmh.data());
  std::vector<ElementD> hr((size_t)m * n); block_ref_D.copy_to_host(hr.data());   // dequant golden

  // Fixed TK=128 (matches the perf-sweep configs in test_moe_grouped_ppu, so this gate validates the SAME shape).
  // The dense binary and grouped launch now select the same group-size tag and ceil(TK/gs) scale tile, so both
  // comparisons are valid for every compiled group size (16/32/64/128).
  moe_grouped_ppu::filter_and_run<moe_grouped_ppu::QuantMode::FinegrainedScaleOnly, 64, 64, 128, 32, 32, 3>(
      block_A.get(), block_B_buff.device_data(), block_scale.get(), /*zeros=*/nullptr,
      pd.get(), sd.get(), gm.get(),
      m, n, k, L, g, dev_shapes.get(), host_shapes.data(), /*group_row_offsets=*/nullptr,
      ws.get(), ws_bytes, /*stream=*/nullptr);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());

  // (A) grp vs block_ref_D -- the fp16 dequant->GemmRef reference (algorithm-independent golden). This is the
  //     TRUTH and is always reported.
  // (B) grp vs block_D -- the dense mixed kernel's own output, now on the same per-gs schedule ladder.
  std::vector<ElementD> hk((size_t)m * n), hg((size_t)m * n);   // hr (dequant golden) already copied above
  block_D.copy_to_host(hk.data()); Dgrp.copy_to_host(hg.data());
  double maxrelR = 0, maxrelK = 0; int badR = 0, badK = 0;
  for (size_t i = 0; i < (size_t)m * n; ++i) {
    double r = float(hr[i]), gt = float(hg[i]);
    double relR = std::abs(gt - r) / (std::abs(r) + 1e-3);
    if (relR > maxrelR) maxrelR = relR;  if (relR > 5e-2) ++badR;
    double kk = float(hk[i]); double relK = std::abs(gt - kk) / (std::abs(kk) + 1e-3);
    if (relK > maxrelK) maxrelK = relK;  if (relK > 5e-2) ++badK;
  }
  std::printf("\n[xcheck grouped L=1] m=%d n=%d k=%d g=%d\n", m, n, k, g);
  std::printf("  (A) grp vs ref_D (dequant golden): max_rel=%.3e bad=%d/%zu -> %s\n",
              maxrelR, badR, (size_t)m * n, badR == 0 ? "MATCH" : "MISMATCH");
  std::printf("  (B) grp vs block_D (dense kernel): max_rel=%.3e bad=%d/%zu -> %s\n",
              maxrelK, badK, (size_t)m * n, badK == 0 ? "MATCH" : "MISMATCH");
  std::printf("  ref_D[0..5]  ="); for (int i = 0; i < 6; ++i) std::printf(" %8.3f", float(hr[i]));
  std::printf("\n  grp_D[0..5]  ="); for (int i = 0; i < 6; ++i) std::printf(" %8.3f", float(hg[i]));
  std::printf("\n  blkD[0..5]  ="); for (int i = 0; i < 6; ++i) std::printf(" %8.3f", float(hk[i]));
  std::printf("\n");
  }   // end if constexpr (sizeof_bits<QT> >= 2)
}

int main(int argc, char const **args) {
  // FIRST, before any allocation: a device switch after the context exists is not a switch.
  bench_device::bind_from_env();
  // WITNESS THE A PROVIDER, IN main(), because nothing else does -- and because the first version of
  // this sat inside `if (options.search_configs)`, so the --config= path that every A/B actually uses printed
  // nothing at all. A witness on the branch nobody takes is the defect it was written against.
  //
  // PPU_A_PACK is a binary-wide #if in ppu_mixed_policy.hpp that selects PackedRowAProvider and outranks
  // everything downstream. The grouped launcher prints its A path; this bench builds its own Gemm and printed
  // nothing, which is how BACKTEST's D7 became a number nobody could attribute.
  std::printf("  [A path] %s\n",
#if defined(PPU_A_PACK) && (PPU_A_PACK != 0)
              "PACKED cubes (PPU_A_PACK), cp.async row0 + swzl read -- ONE-ROW EXPERIMENT, valid at M<=1 only"
#else
              "ordinary AIU + swzl"
#endif
  );
  hggcDeviceProp props;
  int current_device_id;
  CUTLASS_PPU_CHECK(hggcGetDevice(&current_device_id));
  CUTLASS_PPU_CHECK(hggcGetDeviceProperties(&props, current_device_id));
  hggcError_t error = hggcGetDeviceProperties(&props, 0);

  // should run on PPU 1.0
  if (props.major != 8 || props.minor != 0) {
    std::cerr << " This example should only be run on PPU 1.0!!! " << std::endl;
    return 0;
  }

  //
  // Parse options
  //
  Options options;
  options.parse(argc, args);
  print_dense_table_provenance();

#if defined(DENSE_MARLIN_SWEEP)
  // This binary's registry and every generated wrapper are compile-time
  // Marlin.  Runtime scheduler flags would only create a false identity, and
  // the ordinary dense cache key has no scheduler field, so both cache load
  // and save must be rejected rather than aliasing DP samples/tactics.
  if (options.persistent || options.streamk || options.marlin ||
      options.streamk_gate || options.streamk_split_gate ||
      options.streamk_exact_fixture) {
    std::fprintf(stderr,
                 "test_lowbit_dense_marlin_sweep fixes scheduler=marlin at build time; "
                 "runtime scheduler flags are unsupported\n");
    return 1;
  }
  if (options.xcheck) {
    std::fprintf(stderr,
                 "test_lowbit_dense_marlin_sweep cannot use --xcheck because it bypasses the Marlin table\n");
    return 1;
  }
  if (options.mode != GemmMode::ScaleOnly) {
    std::fprintf(stderr,
                 "test_lowbit_dense_marlin_sweep covers ScaleOnly (mode=1) only\n");
    return 1;
  }
  if (!options.tactic_file.empty() || !options.save_tactic_file.empty()) {
    std::fprintf(stderr,
                 "ordinary dense tactic cache load/save is rejected for scheduler=marlin; "
                 "use --config or --search_configs without --tactic/--save_tactic\n");
    return 1;
  }
#elif !defined(DENSE_SCHEDULER_AB)
  if (options.persistent || options.streamk || options.marlin || options.streamk_gate ||
      options.streamk_split_gate || options.streamk_exact_fixture) {
    std::fprintf(stderr,
                 "scheduler A/B flags are available only in the dedicated dense scheduler targets; "
                 "the ordinary dense sweep is unchanged\n");
    return 1;
  }
#else
  int const scheduler_arms = int(options.persistent) + int(options.streamk) +
                             int(options.marlin);
  if (scheduler_arms > 1) {
    std::fprintf(stderr,
                 "--persistent, --streamk, and --marlin are mutually exclusive A/B arms\n");
    return 1;
  }
  if ((options.persistent || options.streamk || options.marlin ||
       options.streamk_exact_fixture) &&
      options.mode != GemmMode::ScaleOnly) {
    std::fprintf(stderr, "dense scheduler A/B currently covers ScaleOnly (mode=1) only\n");
    return 1;
  }
#if !defined(DENSE_STREAMK_AB)
  if (options.streamk || options.streamk_gate || options.streamk_split_gate ||
      options.streamk_exact_fixture) {
    std::fprintf(stderr, "--streamk is available only in test_lowbit_dense_streamk_ab\n");
    return 1;
  }
#else
  if (options.streamk_gate &&
      (options.m != 64 || options.n != 128 || options.k != 4352 || options.l != 1 ||
       options.g != 128 || std::abs(options.alpha - 0.75f) > 1.0e-7f ||
       std::abs(options.beta - 0.5f) > 1.0e-7f)) {
    std::fprintf(stderr,
                 "--streamk_gate is the exact dyadic seam fixture: "
                 "--m=64 --n=128 --k=4352 --l=1 --g=128 --alpha=.75 --beta=.5\n");
    return 1;
  }
  if (options.streamk_gate && options.streamk_split_gate) {
    std::fprintf(stderr,
                 "--streamk_gate and --streamk_split_gate are independent fixtures; "
                 "run them separately\n");
    return 1;
  }
  if (options.streamk_gate && options.streamk_exact_fixture) {
    std::fprintf(stderr,
                 "--streamk_gate and --streamk_exact_fixture name different fixtures; "
                 "run them separately\n");
    return 1;
  }
  if (options.streamk_exact_fixture &&
      (
#if defined(DENSE_MARLIN_AB)
       (options.m != 1 && options.m != 2048) ||
#else
       options.m != 2048 ||
#endif
       options.k != 4096 || options.l != 1 ||
       options.g != 128 || options.alpha != 1.0f || options.beta != 0.0f)) {
    std::fprintf(stderr,
#if defined(DENSE_MARLIN_AB)
                 "--streamk_exact_fixture requires --m=1-or-2048 --k=4096 --l=1 "
#else
                 "--streamk_exact_fixture requires --m=2048 --k=4096 --l=1 "
#endif
                 "--g=128 --alpha=1 --beta=0 (N may adapt to runtime workers)\n");
    return 1;
  }
  if (options.streamk_split_gate && options.iterations != 0) {
    std::fprintf(stderr,
                 "--streamk_split_gate is correctness-only and requires --iterations=0\n");
    return 1;
  }
  if (options.streamk_split_gate &&
      (options.l != 1 || options.alpha != 1.0f || options.beta != 0.0f)) {
    std::fprintf(stderr,
                 "--streamk_split_gate first replay requires --l=1 --alpha=1 --beta=0\n");
    return 1;
  }
#endif
#if !defined(DENSE_MARLIN_AB)
  if (options.marlin) {
    std::fprintf(stderr,
                 "--marlin is available only in test_lowbit_dense_marlin_ab\n");
    return 1;
  }
#endif
#endif

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  // --list_configs: enumerate the compiled tactics and exit.
  if (options.list_configs) {
    std::printf("compiled CUTLASS W%dA16 tile configs scheduler=%s "
                "(group sizes 16, 32, 64, 128):\n",
                int(cutlass::sizeof_bits<QuantType>::value),
#if defined(DENSE_MARLIN_SWEEP)
                "marlin"
#else
                "default"
#endif
    );
    for (auto const& c : supported_configs())
      std::printf("  %-22s  tile %dx%dx%d  warp %dx%d  stages %d\n",
                  c.name, c.tm, c.tn, c.tk, c.wm, c.wn, c.st);
    return 0;
  }

  if (options.mode != GemmMode::ConvertOnly && !supported_group_size(options.g)) {
    std::fprintf(stderr, "unsupported dense group size %d (supported: 16, 32, 64, 128)\n", options.g);
    return 1;
  }

  // The tactic path is ScaleOnly (mode 1). Modes 0/2 keep the original fixed-config run.
  if (options.mode != GemmMode::ScaleOnly) {
    if (options.mode == GemmMode::ConvertOnly) {
#if defined(BENCH_UINT1) || defined(BENCH_UINT2)
      std::fprintf(stderr, "mode 0 (no-scale convert) is available only in the int4 dense binary; "
                           "this W%d binary searches scale-only tactics\n",
                   int(cutlass::sizeof_bits<QuantType>::value));
      return 1;
#else
      std::cout << "PPU1.0 no-scale mode.\n"; run<GemmConvertOnly>(options);
#endif
    }
    else                                       { std::cout << "PPU1.0 scale+zero mode.\n"; run_scale_zero(options); }
    return 0;
  }
  std::cout << (options.g == options.k ? "PPU1.0 per-column scale mode.\n" : "PPU1.0 group scale mode.\n");

  // Root-cause cross-check (runtime --xcheck): run the VERIFIED mixed kernel once (fills block_ref_D + the
  // shuffled block_B_buff), then run the grouped kernel L=1 on that exact data and compare. Bypasses tactics.
  if (options.xcheck) {
    std::cout << "[xcheck] stock (non-grouped) mixed kernel -> reference, then grouped L=1 on the SAME data\n";
    Options o = options; o.iterations = 0;   // correctness only, skip timing
    run_scale_only(o);                        // fills block_A / block_B_buff / block_scale / block_ref_D
    xcheck_grouped(o);
    return 0;
  }

  // --search_configs: time every compiled config (in-process, no recompile), keep the best that PASSED,
  // print a table, optionally persist to a tactic cache, then run the winner once.
  if (options.search_configs) {
    // THE WHOLE LIST, REPEATED -- and the winner is a median with a band, not one timing. `if (tf > best_tf)`
    // over a single measurement was ordering candidates inside the recorded 13% cross-run spread by whichever
    // happened to run in a good moment. Repeating the LIST rather than each candidate is what keeps clock and
    // thermal drift from landing on one of them; the same reasoning and the same procedure as the MoE bench,
    // which is why both call bench_select.hpp rather than each carrying a copy.
    // DEFAULT 1, not 5. The user asked for one pass and five stayed the default. One repeat is legal and is
    // explicitly NOT a ranking -- the verdict lines below say so rather than printing something that reads like one.
    const int reps = bench_measure::read_reps();
    char build[160];
#if defined(DENSE_MARLIN_SWEEP)
    std::snprintf(build, sizeof build, "bits=%d TSK=%d gs=%d scheduler=%s",
                  int(cutlass::sizeof_bits<QuantType>::value), TileShapeK, options.g,
                  "marlin");
#else
    std::snprintf(build, sizeof build, "bits=%d TSK=%d gs=%d",
                  int(cutlass::sizeof_bits<QuantType>::value), TileShapeK, options.g);
#endif
    bench_floor::banner();
    bench_samples::run_header(dense_sample_family(), build, reps);

    std::printf("%-18s %-10s %s\n", "CONFIG", "TFLOP/s", "status");
    Best best; best.tag[0] = '\0'; best.us = 1e18;
    for (int rep = 0; rep < reps; ++rep) {
      if (reps > 1) std::printf("\n  --- pass %d/%d ---\n", rep + 1, reps);
      for (auto const& c : supported_configs()) {
        // NAME THE CANDIDATE BEFORE LAUNCHING IT. A device assert takes the whole process, and every other
        // report of which config ran happens AFTER run_config() returns -- so without this, the row that killed
        // a sweep is not named anywhere and "reproduce it by name" has nothing to work from. Both channels get
        // it: the sample file (an `a` record with no matching `s` is where it stopped) and stdout, flushed
        // because stdout to a file is block-buffered and would lose its tail with the process.
        bench_samples::Sample _a{};
        _a.fixture = dense_fixture(options); _a.dist = dense_distribution(); _a.schema = dense_schema();
        _a.n = options.n; _a.k = options.k; _a.gs = options.g;
        _a.experts = 0; _a.rows = options.m; _a.mmax = options.m;
        _a.tm = c.tm; _a.tn = c.tn; _a.tk = c.tk; _a.wm = c.wm; _a.wn = c.wn; _a.st = c.st;
        _a.bc = c.b_chunk; _a.bc_eff = c.b_chunk_effective;
        _a.pass = rep;
        bench_samples::attempt(_a);
        std::printf("%-18s ", c.name);
        std::fflush(stdout);

        Result r = run_config(options, c);
        if (!r.passed) {
          // RECORDED, NOT JUST PRINTED. "tried and rejected" is evidence for pruning and is distinguishable
          // from a crash only if it lands in the file; without it, unfinished() reports this as a dead run.
          bench_samples::excluded(_a, "bench reported not-passed (unsupported for this shape, or failed)");
          if (!rep) std::printf("%-10s %s\n", "-", "skipped (unsupported/failed)"); else std::printf("\n");
          continue;
        }
        const double tf = r.gflops / 1e3;
        // SECONDS, NOT TFLOP/s, IS WHAT GETS COMPARED. The selection works on a time, so converting once here
        // keeps one definition of "better" rather than a maximised rate in one bench and a minimised time in
        // the other -- two orderings that agree until a shape makes the FLOP count differ between candidates.
        // The timer is already the one measured quantity. Reconstructing it from TFLOP/s previously omitted L
        // even though Options::gflops includes L, shrinking every batched sample and launch-floor comparison by L.
        const double us = r.avg_runtime_ms > 0.0 ? r.avg_runtime_ms * 1e3 : 1e18;
        std::printf("%-10.1f ok\n", tf);
        upd(best, dense_tactic(c), us);
        // THE SAME RECORD THE ATTEMPT CARRIED, plus the measurement. Rebuilding the identity fields here was
        // how an attempt and its sample could disagree about what ran; now `us` is the only field that differs.
        if (bench_samples::enabled()) { bench_samples::Sample s = _a; s.us = us; bench_samples::emit(s); }
      }
    }
    bench_samples::flush();
    if (best.tag[0] == '\0') { std::fprintf(stderr, "no config passed\n"); return 1; }
    const int ties = settle(best);
    const double best_tf = 2.0 * double(options.m) * options.n * options.k * options.l /
                           (best.us * 1e-6) / 1e12;
    // WRITE A TACTIC ONLY FROM A RESOLVED SWEEP. The comment below says an unresolved one is "a wrong answer
    // that never gets revisited", and for one commit the line above it saved unconditionally anyway -- the
    // property was documented and not implemented, which is worse than neither, because the comment is what a
    // reader trusts. Declining is loud: a tactic that is absent gets regenerated, a tactic that is wrong does
    // not.
    if (!options.save_tactic_file.empty()) {
      if (reps < 2) {
        std::printf("  [tactic] NOT saved: one pass cannot rank. Re-run with BENCH_REPS>=2.\n");
      } else if (ties != 0) {
        std::printf("  [tactic] NOT saved: %d candidate(s) tie the leader, so there is no winner to record.\n"
                    "           Expand the tied stratum, or accept a wider table -- but do not cache this.\n", ties);
      } else {
        save_tactic(options.save_tactic_file, options, best.tag, best_tf);
      }
    }
    // NAMED A WINNER ONLY WHEN NOTHING TIES IT. With ties this prints the leader and says it is unresolved,
    // because a tactic cache written from an unresolved sweep is a wrong answer that never gets revisited.
    if (reps < 2)
      std::printf("\n==== LOWEST: %s at %.1f TFLOP/s (ONE pass -- NOT a ranking) ====\n", best.tag, best_tf);
    else if (ties == 0)
      std::printf("\n==== WINNER: %s at %.1f TFLOP/s (separated) ====\n", best.tag, best_tf);
    else
      std::printf("\n==== UNRESOLVED: %s leads at %.1f TFLOP/s, %d candidate(s) tie it ====\n",
                  best.tag, best_tf, ties);
    if (bench_floor::launch_bound(best.us))
      std::printf("     [LAUNCH-BOUND] the leader is within 3x the empty-launch floor (%.2f us): this shape is\n"
                  "     too small for the loop to be measuring the kernel rather than the launch rate.\n",
                  bench_floor::us());
    Result const winner_result = run_config(options, find_config(best.tag));
#if defined(DENSE_MARLIN_SWEEP)
    // A scheduler-specific sweep is evidence only if the selected Marlin arm
    // itself passes; do not inherit the ordinary bench's historical rc=0
    // convention for a failed final launch.
    if (!winner_result.passed) return 1;
    return 0;
#else
    (void)winner_result;
    return 0;
#endif
  }

  // Single run. Config precedence: --config > --tactic cache > compiled default (first supported).
  std::string name = options.config;
  if (name.empty() && !options.tactic_file.empty()) {
    name = load_tactic(options.tactic_file, options);
    if (!name.empty()) std::printf("[tactic] %s -> %s\n", tactic_key(options).c_str(), name.c_str());
  }
  if (name.empty()) name = supported_configs().front().name;
  Result const final_result = run_config(options, find_config(name));
#if defined(DENSE_STREAMK_AB)
  // The 107b target is a mechanism/numerical gate, not a sweep that may skip
  // an unsupported candidate.  Propagate every decomposition, witness,
  // golden, event, or correctness failure to the operator and automation.
  if (!final_result.split_path_exercised) return 2;
  if (!final_result.verification_classified) return 3;
  return final_result.passed ? 0 : 1;
#elif defined(DENSE_MARLIN_SWEEP)
  if (!final_result.passed) return 1;
  return 0;
#else
  (void)final_result;
  return 0;
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////
#endif
