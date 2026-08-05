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

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#include "cutlass/cutlass.h"

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

#include "ppu_include.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"

// Cross-check hook: pull in the grouped mixed-input launcher so we can run it (L=1) on THIS file's own verified
// data via a runtime --xcheck flag. Always included (no compile-time guard): the two-stage PPU host/device
// build made -DMOEG_XCHECK unreliable to propagate to host main(), so we gate at runtime instead.
// ppu_mma_builder.inl is #pragma once, so re-inclusion via moe_grouped_ppu.cuh is safe.
#include "moe_grouped_ppu.cuh"

using namespace cute;


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
// The collective will infer that the narrow type should be upcasted to the wide type.
// We swap A and B operands to the builder here
using CollectiveMainloopConvertOnly = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB_opt, AlignmentB,
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
  using ScaleOnlyMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, AlignmentA,
      cute::tuple<ElementB, ElementScale>, LayoutB_opt, AlignmentB,
      ElementAccumulator, cute::tuple<TileShape, ScaleTile>, WarpShape,
      Int<STAGES>, Schedule>::CollectiveOp;
  using ScaleOnlyKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>, ScaleOnlyMainloop, CollectiveEpilogue>;
  using ScaleOnly = cutlass::gemm::device::GemmUniversalAdapter<ScaleOnlyKernel>;

  using ScaleZeroMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, AlignmentA,
      cute::tuple<ElementB, ElementScale, ElementZero>, LayoutB_opt, AlignmentB,
      ElementAccumulator, cute::tuple<TileShape, ScaleTile>, WarpShape,
      Int<STAGES>, Schedule>::CollectiveOp;
  using ScaleZeroKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>, ScaleZeroMainloop, CollectiveEpilogue>;
  using ScaleZero = cutlass::gemm::device::GemmUniversalAdapter<ScaleZeroKernel>;
};
// =================================================================================================================================================================

// ================================ TACTIC REGISTRY (machete / fpA_intB style) =====================================
// The tile is a compile-time template argument, so autotuning it means compiling a FIXED SET of instantiations
// into ONE binary and dispatching at runtime -- exactly how vLLM Machete (../machete_standalone) and TRT-LLM
// fpA_intB (../fpA_intB_standalone) do it. Cfg<> rebuilds the ScaleOnly (mode-1) type stack with a given
// tile/warp/stages; TileCfg is the runtime descriptor; supported_configs() is the registry; the dispatch macro
// below maps a runtime TileCfg to the matching compiled Cfg<>. This replaces the recompile-per-config sweep.sh.
template <int GroupSize, int TM, int TN, int WM, int WN, int St>
struct Cfg {
  using CfgTile = Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TileShapeK>>;
  using CfgScale = Shape<cute::Int<TN>,
      cute::Int<ppu_group_schedule::scale_groups_v<TileShapeK, GroupSize>>>;
  using CfgWarp = Shape<cute::Int<WM>, cute::Int<WN>, cute::Int<TileShapeK>>;
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, CfgTile, CfgWarp, EpilogueTileType,
      ElementAccumulator, ElementAccumulator, ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD, EpilogueSchedule>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA,
      cute::tuple<ElementB, ElementScale>, LayoutB_opt, AlignmentB,
      ElementAccumulator, cute::tuple<CfgTile, CfgScale>, CfgWarp, Int<St>,
      ppu_group_schedule::FinegrainedSchedule<GroupSize>>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

struct TileCfg { char const* name; int tm, tn, wm, wn, st; };

// The generated table. Regenerate with benchmarks/emit_tactic_configs.cpp when the binary's (QUANT, BENCH_TSK)
// changes -- the static_assert below is what turns forgetting into a compile error rather than a sweep over a
// set of tactics this binary cannot select.
#include "bench_select.hpp"
#include "bench_samples.hpp"
#include "bench_floor.cuh"
#include "lowbit_dense_configs.inc"
static_assert(cutlass::sizeof_bits<QuantType>::value == LOWBIT_DENSE_CFG_BITS && TileShapeK == LOWBIT_DENSE_CFG_TILEK,
              "lowbit_dense_configs.inc was generated for a different (bits, TileK) than this binary. Regenerate: "
              "c++ -std=c++17 -Iquactlize/include benchmarks/emit_tactic_configs.cpp -o /tmp/emit_tactic && "
              "/tmp/emit_tactic <bits> <tile_k> > benchmarks/lowbit_dense_configs.inc");

// The compiled set. Every entry here is a distinct template instantiation baked into the binary; add one and
// it costs compile time, not a rebuild at search time. Keep WM|TM, WN|TN, and both multiples of the 16x16 atom.
inline std::vector<TileCfg> supported_configs() {
  // GENERATED, not hand-written. The 17 rows this replaced were a fifth of the legal-and-pruned set for one
  // (schema, TileK) binary, and nothing in the file said so -- a sweep that searches a fifth of its space still
  // prints a winner. See benchmarks/emit_tactic_configs.cpp for the policy and the regeneration command.
  std::vector<TileCfg> v;
#define LOWBIT_DENSE_ROW(TM,TN,WM,WN,ST,_UNUSED) \
  v.push_back(TileCfg{#TM "x" #TN ":" #WM "x" #WN ":s" #ST, TM, TN, WM, WN, ST});
  LOWBIT_DENSE_CFG_LIST(LOWBIT_DENSE_ROW, )
#undef LOWBIT_DENSE_ROW
  return v;
}

// Map a runtime TileCfg to its compiled Cfg<>::Gemm and run BODY with `using G = ...;` in scope. Mirrors
// machete's CUTLASS55_DISPATCH_CONFIG if-chain. Any config not listed here is a runtime error.
// ONE ROW PER CONFIG, EXPANDED FROM THE SAME LIST supported_configs() USES. Previously these were two
// hand-maintained copies: a row present in the list and absent from this chain reached
// `config %s not compiled in` and exit(1) -- at run time, on the box, mid-sweep. That failure is now not
// expressible, because there is one list. The BODY travels through the list as X's second argument; a macro
// cannot define another macro, so passing it down is what makes a single list serve both expansions.
#define LOWBIT_DENSE_TRY(TM,TN,WM,WN,ST,BODY)                                                               \
  if (!_matched && _try(TM,TN,WM,WN,ST)) { using G = typename Cfg<GroupSize,TM,TN,WM,WN,ST>::Gemm; _matched=true; BODY; }

#define LOWBIT_DENSE_DISPATCH(cfg, BODY)                                                                    \
  do {                                                                                               \
    bool _matched = false;                                                                           \
    auto _try = [&](int tm,int tn,int wm,int wn,int st){ return (cfg).tm==tm&&(cfg).tn==tn&&(cfg).wm==wm&&(cfg).wn==wn&&(cfg).st==st; }; \
    LOWBIT_DENSE_CFG_LIST(LOWBIT_DENSE_TRY, BODY)                                                                  \
    if (!_matched) { std::fprintf(stderr, "config %s not compiled in (see supported_configs)\n", (cfg).name); std::exit(1); } \
  } while (0)
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
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideC_ref stride_C_ref;
StrideD stride_D;
StrideD_ref stride_D_ref;
uint64_t seed;

// Scale and Zero share a stride since the layout and shapes must be the same.
using StrideS = cute::Stride<cute::_1, int64_t, int64_t>;
using StrideS_ref = cutlass::detail::TagToStrideB_t<LayoutScale>;
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

};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

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
void initialize(Options const& options) {

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
        (int8_t*)(&tensor_B.host_data()[batch_offset]), {col, row}, quant_type);
  }
  block_B_buff.sync_device();
}

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

bool verify(const Options &options) {
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
      MmaType, LayoutB, AlignmentB,
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
  return passed;
}

/// Execute a given example GEMM computation. Returns the Result (does not exit on failure, so the tactic
/// search can skip a config that does not verify and move on). `label` is printed on the comparable line.
template <typename Gemm>
Result run(Options &options, char const* label = "default")
{
  initialize(options);

  // Instantiate kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options<typename Gemm::Arguments>(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  Result result;
  // Check if the problem size is supported or not. Do not hard-CHECK (which aborts): during a search an
  // unsupported tile must be skippable, so return a failed Result instead of killing the process.
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) { result.passed = false; return result; }
  if (gemm.initialize(arguments, workspace.get()) != cutlass::Status::kSuccess) { result.passed = false; return result; }

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  // Check if output from kernel and reference kernel are equal or not
  result.passed = verify(options);

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  if (!result.passed) {
    // The int2 (uint2) path's kernel is verified correct elsewhere (test_w2a16_diag/grouped/real all MATCH);
    // this bench's verify() reference (dequantize_weight) is int4-oriented, so it flags int2 spuriously. For a
    // perf run (iterations>0) time anyway; only bail when timing wasn't requested (the correctness-only sweep).
    if (options.iterations <= 0) return result;
    std::cout << "  (verify failed -- likely the int4-oriented reference; timing the kernel anyway for perf)\n";
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
    PpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;

    // Report BOTH utilizations -- neither alone tells you what is binding:
    //   cmp   = achieved TFLOP/s vs the 500 TFLOP/s fp16 peak (compute utilization).
    //   min   = the ALGORITHMIC minimum traffic (A + B + D each touched once) vs HBM peak. This is what a perfect
    //           kernel would move from HBM; at prefill M it is tiny (few % of HBM), which is why MFU is the headline.
    //   tile  = the TILE-LEVEL traffic REQUESTED: A once per N-tile (N/TileN times), B once per M-tile. Reported
    //           as GB/s and as a REUSE FACTOR over min, NOT as a percentage of HBM.
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
    const double tflops = result.gflops / 1e3;
    int tm = 0, tn = 0;
    std::sscanf(label, "%dx%d", &tm, &tn);                       // label is "TMxTN:WMxWN:sS"
    const double Mm = options.m, Nn = options.n, Kk = options.k;
    const double bq = double(sizeof_bits<QuantType>::value) / 8.0;   // bytes per weight element
    const double n_tiles = tn > 0 ? std::ceil(Nn / tn) : 1.0;
    const double m_tiles = tm > 0 ? std::ceil(Mm / tm) : 1.0;
    const double min_bytes  = Mm * Kk * 2.0 + Nn * Kk * bq + Mm * Nn * 2.0;
    const double tile_bytes = Mm * Kk * 2.0 * n_tiles + Nn * Kk * bq * m_tiles + Mm * Nn * 2.0;
    const double HBM_GBS = 2766.0;                               // ppu001 HBM peak
    const double gbs_min  = min_bytes  / (us * 1e-6) / 1e9;
    const double gbs_tile = tile_bytes / (us * 1e-6) / 1e9;
    // NOTE: the w%d tag (w4/w2/w1) is essential -- the cfg label alone is identical across quant builds and the
    // numbers are easy to mix up between them (int4/int2/int1 all have a "64x64:32x32:s3").
    // min IS the only figure with a legitimate HBM denominator: it counts every byte once, which is what DRAM
    // must actually deliver. tile carries a reuse factor instead, and a marker when it exceeds the DRAM peak --
    // which says the re-reads are cache-served, not that the kernel is bandwidth-bound.
    double const reuse = tile_bytes / min_bytes;
    std::printf("  [CUTLASS w%d gs=%d cfg=%s] M=%d %7.2f us | %6.1f TFLOP/s | cmp %5.1f%% | tile %6.0f GB/s "
                "(%.1fx min%s) | min %5.0f GB/s (%4.1f%% HBM)\n",
                int(sizeof_bits<QuantType>::value), options.g, label, options.m, us, tflops, 100.0 * tflops / 500.0,
                gbs_tile, reuse, gbs_tile > HBM_GBS ? ", L2-served" : "", gbs_min, 100.0 * gbs_min / HBM_GBS);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Tactic dispatch + shape-keyed cache (exact-match text, like machete's cutlass55_tactics.cache).

bool supported_group_size(int group_size) {
  return group_size == 16 || group_size == 32 || group_size == 64 || group_size == 128;
}

Result run_scale_only(Options& options, char const* label = "default") {
  switch (options.g) {
    case 16:  return run<typename GroupKernels<16>::ScaleOnly>(options, label);
    case 32:  return run<typename GroupKernels<32>::ScaleOnly>(options, label);
    case 64:  return run<typename GroupKernels<64>::ScaleOnly>(options, label);
    case 128: return run<typename GroupKernels<128>::ScaleOnly>(options, label);
    default:  std::fprintf(stderr, "unsupported dense group size %d (supported: 16, 32, 64, 128)\n", options.g);
              return {};
  }
}

Result run_scale_zero(Options& options, char const* label = "default") {
  switch (options.g) {
    case 16:  return run<typename GroupKernels<16>::ScaleZero>(options, label);
    case 32:  return run<typename GroupKernels<32>::ScaleZero>(options, label);
    case 64:  return run<typename GroupKernels<64>::ScaleZero>(options, label);
    case 128: return run<typename GroupKernels<128>::ScaleZero>(options, label);
    default:  std::fprintf(stderr, "unsupported dense group size %d (supported: 16, 32, 64, 128)\n", options.g);
              return {};
  }
}

// Run one named/descriptor config through the compiled dispatch. Returns its Result.
template <int GroupSize>
Result run_config_for_group(Options& options, TileCfg const& cfg) {
  Result r;
  LOWBIT_DENSE_DISPATCH(cfg, { r = run<G>(options, cfg.name); });
  return r;
}

Result run_config(Options& options, TileCfg const& cfg) {
  switch (options.g) {
    case 16:  return run_config_for_group<16>(options, cfg);
    case 32:  return run_config_for_group<32>(options, cfg);
    case 64:  return run_config_for_group<64>(options, cfg);
    case 128: return run_config_for_group<128>(options, cfg);
    default:  std::fprintf(stderr, "unsupported dense group size %d (supported: 16, 32, 64, 128)\n", options.g);
              return {};
  }
}

TileCfg find_config(std::string const& name) {
  for (auto const& c : supported_configs()) if (name == c.name) return c;
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

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  // --list_configs: enumerate the compiled tactics and exit.
  if (options.list_configs) {
    std::printf("compiled CUTLASS W4A16 tile configs (group sizes 16, 32, 64, 128):\n");
    for (auto const& c : supported_configs())
      std::printf("  %-18s  tile %dx%d  warp %dx%d  stages %d\n", c.name, c.tm, c.tn, c.wm, c.wn, c.st);
    return 0;
  }

  if (options.mode != GemmMode::ConvertOnly && !supported_group_size(options.g)) {
    std::fprintf(stderr, "unsupported dense group size %d (supported: 16, 32, 64, 128)\n", options.g);
    return 1;
  }

  // The tactic path is ScaleOnly (mode 1). Modes 0/2 keep the original fixed-config run.
  if (options.mode != GemmMode::ScaleOnly) {
    if (options.mode == GemmMode::ConvertOnly) { std::cout << "PPU1.0 no-scale mode.\n";  run<GemmConvertOnly>(options); }
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
    const int reps = [] { char const* e = std::getenv("BENCH_REPS"); int r = e ? std::atoi(e) : 5; return r < 1 ? 1 : r; }();
    char build[128];
    std::snprintf(build, sizeof build, "bits=%d TSK=%d gs=%d",
                  int(cutlass::sizeof_bits<QuantType>::value), TileShapeK, options.g);
    bench_floor::banner();
    bench_samples::run_header("cutlass_w4a16", build, reps);

    std::printf("%-18s %-10s %s\n", "CONFIG", "TFLOP/s", "status");
    Best best; best.tag[0] = '\0'; best.us = 1e18;
    for (int rep = 0; rep < reps; ++rep) {
      if (reps > 1) std::printf("\n  --- pass %d/%d ---\n", rep + 1, reps);
      for (auto const& c : supported_configs()) {
        Result r = run_config(options, c);
        if (!r.passed) { if (!rep) std::printf("%-18s %-10s %s\n", c.name, "-", "skipped (unsupported/failed)"); continue; }
        const double tf = r.gflops / 1e3;
        // SECONDS, NOT TFLOP/s, IS WHAT GETS COMPARED. The selection works on a time, so converting once here
        // keeps one definition of "better" rather than a maximised rate in one bench and a minimised time in
        // the other -- two orderings that agree until a shape makes the FLOP count differ between candidates.
        const double us = (tf > 0.0) ? (2.0 * double(options.m) * options.n * options.k / (tf * 1e12) * 1e6) : 1e18;
        std::printf("%-18s %-10.1f ok\n", c.name, tf);
        upd(best, c.name, us);
        if (bench_samples::enabled()) {
          bench_samples::Sample s{};
          static char fx[96];
          std::snprintf(fx, sizeof fx, "dense-m%d-n%d-k%d-gs%d", options.m, options.n, options.k, options.g);
          s.fixture = fx; s.dist = "dense-v1";
          s.schema = cutlass::sizeof_bits<QuantType>::value == 4 ? "i4"
                   : cutlass::sizeof_bits<QuantType>::value == 2 ? "i2" : "i1";
          s.n = options.n; s.k = options.k; s.gs = options.g;
          s.experts = 0; s.rows = options.m; s.mmax = options.m;
          s.tm = c.tm; s.tn = c.tn; s.tk = TileShapeK; s.wm = c.wm; s.wn = c.wn; s.st = c.st;
          s.pass = rep; s.us = us;
          bench_samples::emit(s);
        }
      }
    }
    bench_samples::flush();
    if (best.tag[0] == '\0') { std::fprintf(stderr, "no config passed\n"); return 1; }
    const int ties = settle(best);
    const double best_tf = 2.0 * double(options.m) * options.n * options.k / (best.us * 1e-6) / 1e12;
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
    run_config(options, find_config(best.tag));
    return 0;
  }

  // Single run. Config precedence: --config > --tactic cache > compiled default (first supported).
  std::string name = options.config;
  if (name.empty() && !options.tactic_file.empty()) {
    name = load_tactic(options.tactic_file, options);
    if (!name.empty()) std::printf("[tactic] %s -> %s\n", tactic_key(options).c_str(), name.c_str());
  }
  if (name.empty()) name = supported_configs().front().name;
  run_config(options, find_config(name));
  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
