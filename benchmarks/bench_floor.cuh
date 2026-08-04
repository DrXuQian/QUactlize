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
// bench_cutlass_w4a16.cu:773, grouped at moe_grouped_ppu.cuh:362, with split-K excluded from the sweep -- so
// this is a floor report, not a decomposition. If split-K returns, this comment is where to start.
//
// So: measure the floor with the SAME loop shape, print it, and put it in the sample stream so a row near it can
// be flagged rather than believed.
#pragma once
#include <chrono>

namespace bench_floor {

// THE PROBE MUST NOT BE OPTIMISED AWAY. An empty __global__ can be elided or fused by the driver, and a floor of
// zero would make every row look kernel-bound. One store to a caller-owned word, whose address the compiler
// cannot prove dead, is enough to keep the launch real while costing nothing measurable.
__global__ void bench_floor_nop(int* sink) { if (threadIdx.x == 0 && blockIdx.x == 0) *sink = 1; }

// Same loop shape as time_it: one warm launch, then `iters` queued launches and a single sync. Returns
// microseconds per launch. Anything the sweep measures at or near this is a launch-rate reading.
inline double probe(int iters = 200) {
  int* sink = nullptr;
  if (cudaMalloc(&sink, sizeof(int)) != cudaSuccess) return 0.0;   // 0 == "not measured", reported as such
  bench_floor_nop<<<1, 32>>>(sink);
  cudaDeviceSynchronize();
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) bench_floor_nop<<<1, 32>>>(sink);
  cudaDeviceSynchronize();
  auto t1 = std::chrono::high_resolution_clock::now();
  cudaFree(sink);
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

// Cached: the floor is a property of the machine and the process, not of a candidate.
inline double us() { static const double f = probe(); return f; }

// HOW CLOSE IS TOO CLOSE. Within 3x the floor, a difference between two candidates is partly a difference in how
// well each one hides the launch, which is not the quantity being tuned. The multiple is a judgement and is
// stated as one; what matters is that the reader is told, rather than the row being presented as a clean
// measurement of the kernel.
inline bool launch_bound(double row_us) { const double f = us(); return f > 0.0 && row_us < 3.0 * f; }

inline void banner() {
  const double f = us();
  if (f <= 0.0) {
    std::printf("             launch floor: NOT MEASURED (cudaMalloc failed) -- short rows cannot be flagged\n");
    return;
  }
  std::printf("             launch floor: %.2f us per empty launch. A row under %.2f us is within 3x of it and\n"
              "             is reported as LAUNCH-BOUND: at that scale the number is the launch rate as much as\n"
              "             the kernel, and two candidates differ partly by how well each hides it.\n", f, 3.0 * f);
}

}  // namespace bench_floor
