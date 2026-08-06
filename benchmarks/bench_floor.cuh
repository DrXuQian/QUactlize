// WHAT DOES AN EMPTY LAUNCH COST? -- because a timing at that cost is not a measurement of the kernel.
//
// THE PROBLEM THIS MAKES VISIBLE. Both benches time N back-to-back launches with ONE synchronise at the end:
//
//     f(); sync;                       // warm
//     t0; for (i < iters) f(); sync; t1;
//     return (t1 - t0) / iters;
//
// That is the right shape -- the launches queue, so the CPU-side cost overlaps GPU execution and the result is
// the kernel's throughput. It is right ONLY while the kernel is the bottleneck. When the kernel is shorter than
// the launch rate the loop becomes launch-bound and the number reported is the launch rate, wearing the kernel's
// name. Nothing in the output distinguishes the two cases, and the decode shapes -- eight experts of one row --
// are exactly where a kernel gets short enough for it to matter.
//
// A SECOND CASE, not currently live but one launch away. If a tactic ever issues MORE than one kernel per call
// (split-K adds a reduction; a prepass adds a pass), the gap between them lands inside the same wall clock and
// cannot be separated from either kernel's time. The swept configs are single-launch today -- dense at
// test_lowbit_dense_bench.cu:773, grouped at moe_grouped_ppu.cuh:362, with split-K excluded from the sweep -- so
// this is a floor report, not a decomposition. If split-K returns, this comment is where to start.
//
// So: measure the floor with the SAME loop shape, print it, and put it in the sample stream so a row near it can
// be flagged rather than believed.
#pragma once
#include <chrono>
#include "cutlass/util/device_memory.h"

// INCLUDE WHAT IT USES. This called hggcDeviceSynchronize while including nothing that declares it -- it
// arrived transitively from actlize's device_memory.h, so the file compiled only under one include ORDER and on
// one platform. The split is the one gemv_common.hpp already uses; the intrinsic spellings match, only the
// header and the runtime prefix differ.
#if defined(__HGGCCC__)
#include <hggc_runtime.h>
#define BENCH_FLOOR_SYNC() hggcDeviceSynchronize()
#define BENCH_FLOOR_OK     hggcSuccess
#else
#include <cuda_runtime.h>
#define BENCH_FLOOR_SYNC() cudaDeviceSynchronize()
#define BENCH_FLOOR_OK     cudaSuccess
#endif

namespace bench_floor {
// INTERNAL LINKAGE, ALL OF IT. A __global__ defined in a header has EXTERNAL linkage, so every translation
// unit that includes this one defines it and the link fails with `multiple definition`. The MoE sweep has 180
// generated units, all of which include the bench header transitively -- so the first version of this file
// linked nowhere.
//
// AND THE LOCAL SYNTAX GATE CANNOT SEE IT: it compiles ONE translation unit. A multi-TU collision is invisible
// to every check this repo had, which is why tests/test_bench_floor_links.py exists -- it compiles two TUs and
// LINKS them, which is the smallest thing that can fail this way.
namespace {

// THE PROBE MUST NOT BE OPTIMISED AWAY. An empty __global__ can be elided or fused by the driver, and a floor of
// zero would make every row look kernel-bound. One store to a caller-owned word, whose address the compiler
// cannot prove dead, is enough to keep the launch real while costing nothing measurable.
__global__ void bench_floor_nop(int* sink) { if (threadIdx.x == 0 && blockIdx.x == 0) *sink = 1; }

// Same loop shape as time_it: one warm launch, then `iters` queued launches and a single sync. Returns
// microseconds per launch. Anything the sweep measures at or near this is a launch-rate reading.
// hggc* AND cutlass::DeviceAllocation, NOT cudaMalloc. The box compiles this header, and the first version
// used the NVIDIA runtime directly -- caught by the repo's ppu_portability gate, which is exactly the class of
// error that gate exists for: it compiles here and nowhere else, and "it built on my machine" is not evidence
// about the target. DeviceAllocation is the allocation the rest of the benches use and is portable to both.
double probe(int iters = 200) {
  cutlass::DeviceAllocation<int> sink(1);
  if (sink.get() == nullptr) return 0.0;                  // 0 == "not measured", and it is reported as such
  bench_floor_nop<<<1, 32>>>(sink.get());
  if (BENCH_FLOOR_SYNC() != BENCH_FLOOR_OK) return 0.0;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) bench_floor_nop<<<1, 32>>>(sink.get());
  if (BENCH_FLOOR_SYNC() != BENCH_FLOOR_OK) return 0.0;
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

// Cached: the floor is a property of the machine and the process, not of a candidate.
double us() { static const double f = probe(); return f; }

// HOW CLOSE IS TOO CLOSE. Within 3x the floor, a difference between two candidates is partly a difference in how
// well each one hides the launch, which is not the quantity being tuned. The multiple is a judgement and is
// stated as one; what matters is that the reader is told, rather than the row being presented as a clean
// measurement of the kernel.
bool launch_bound(double row_us) { const double f = us(); return f > 0.0 && row_us < 3.0 * f; }

void banner() {
  const double f = us();
  if (f <= 0.0) {
    std::printf("             launch floor: NOT MEASURED (device allocation failed) -- short rows cannot be flagged\n");
    return;
  }
  std::printf("             launch floor: %.2f us per empty launch. A row under %.2f us is within 3x of it and\n"
              "             is reported as LAUNCH-BOUND: at that scale the number is the launch rate as much as\n"
              "             the kernel, and two candidates differ partly by how well each hides it.\n", f, 3.0 * f);
}

}  // anonymous -- see the note at the top of the namespace
}  // namespace bench_floor
