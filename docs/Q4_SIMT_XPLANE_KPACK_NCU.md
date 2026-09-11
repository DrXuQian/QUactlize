# Q4 Xplane / K-pack NCU comparison, 2026-09-11

The current K-pack N2 reader is much more instruction/ALU-heavy, while the
historical Xplane reader drives substantially more DRAM bandwidth on the
same cold-capacity-pressure workload. This is NVIDIA development evidence,
not a PPU admission or a production-format change.

## Device and profiling boundary

RTX 5090 still rejects hardware counters with `ERR_NVGPUCTRPERM`. Both arms
were therefore profiled on the same RTX 5070 / WSL (48 SMs, 48 MiB L2), using
NCU 2025.1.1. Do not mix these durations with the prior 5090 timing table.
Recipes are the independently selected **5090** winners; this run does not
claim to retune or establish the 5070 optimum.

The original Xplane and N2 DSOs, same Q4 GGUF fixtures, F16 A, F32 output,
and identical packed metadata are reused from the matched A/B. No kernel
body is edited. All eight profile arms pass the independent numeric oracle.
The standalone runner is also checked against both arms on the 5090.

Each process prepares and validates its output, then runs five warm calls or
five complete weight rotations **before** `cudaProfilerStart`. NCU uses
`--profile-from-start off --replay-mode application --cache-control none
--clock-control none`. Every replay pass re-creates that cache history.
Rotations use 24 copies for K2048 or 12 copies for K4096: 108 MiB, 2.25x the
actual 5070 L2. This is not an assertion that every cache line is cold.

The profiled region contains one Xplane kernel, or the K-pack producer and
its separate S8 reducer. Allocation, packing, copies and warmup are outside
that region. Separate unprofiled graph timing is preserved in the receipt.
NCU replay durations must not replace those end-to-end timing samples.

## Main comparison: M1, N4096, K4096

Metrics below are for the **producer only**. DRAM GB/s and percentages are
actual NCU byte-counter metrics, not logical weight bytes divided by time.

| Metric | Xplane, rotating | K-pack, rotating | Xplane, warm | K-pack, warm |
| --- | ---: | ---: | ---: | ---: |
| SM throughput (% sustained peak) | 23.41 | 76.38 | 46.11 | 79.64 |
| FMA pipeline active (% sustained peak elapsed) | 20.43 | 20.97 | 40.26 | 22.32 |
| ALU-heavy pipeline active (% sustained peak elapsed) | 13.76 | 76.38 | 27.18 | 79.64 |
| DRAM bandwidth (GB/s) | 529.09 | 290.65 | 4.44 | 4.49 |
| DRAM bandwidth (% sustained peak) | 80.14 | 43.98 | 0.67 | 0.68 |
| L2 throughput (% sustained peak) | 15.65 | 8.12 | 31.05 | 8.26 |
| Executed warp instructions | 2,019,328 | 11,663,360 | 2,019,328 | 12,302,336 |
| Achieved occupancy (%) | 56.36 | 57.87 | 53.71 | 60.03 |
| Registers/thread | 63 | 64 | 63 | 64 |
| DRAM bytes (MiB) | 9.010 | 9.225 | 0.039 | 0.141 |
| NCU replay duration (us, diagnostic only) | 17.856 | 33.280 | 9.216 | 32.832 |

The extra K-pack reducer is 1.888 us in the rotating capture and 1.760 us
in the warm capture, with 23,168 executed warp instructions. It is not
included in the producer rows above and is not the dominant difference.

For M1/N4096/K2048 the direction is the same: rotating Xplane reaches
453.28 GB/s / 68.80%, versus K-pack 254.63 GB/s / 38.53%. Their producer
instruction counts are 1,097,728 and 6,166,528. Full counters for both shapes
and both cache modes are retained in the linked JSON.

## Interpretation and limits

1. **Higher SM throughput does not mean more useful GEMV arithmetic.** In
   these K-pack captures its SM throughput equals the ALU-heavy submetric;
   the FMA activity is only about 21-22%. NCU throughput metrics report the
   maximum of their constituent counters. FMA pipeline activity is also not
   a measurement of useful FLOPs/MFU; its pipeline can execute integer
   multiply/add operations as well. See the [NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
2. Cold-capacity-pressure Xplane is predominantly memory-limited: DRAM is
   at 80.14%, and long-scoreboard waiting accounts for roughly 67.5% of its
   reported per-issued-instruction warp latency. K-pack has about **5.78x**
   the warp instruction count, ALU-heavy activity at 76.38%, and math-pipe
   throttling accounts for about 23.6% of the corresponding warp latency.
   This supports prioritizing instruction and load-topology analysis rather
   than assuming that the SMs are simply idle or that prefetch alone closes
   the gap. It does not prove which individual expression is responsible.
3. NCU also reports **1,972,224 excessive global sectors out of 2,893,824
   (68%)** for the K-pack K4096 producer. This is the source-counter
   coalescing diagnostic, **not** a claim that 68% of DRAM bytes are wasted:
   actual DRAM traffic is only 9.225 versus 9.010 MiB. Cached and repeated
   requests still require instructions and memory-system work. The exact
   load sites need source/SASS attribution before a fix.
4. Xplane is not perfect: its shared-memory diagnostic reports 71%
   excessive wavefronts. Despite that, it is much faster in this comparison.
   One counter or occupancy number alone is not a performance verdict.
5. Arithmetic still differs: historical Xplane uses partial half2
   accumulation; current N2 uses FP32 dot accumulation. This is not yet a
   layout-only or precision-matched causal A/B, and not all extra K-pack
   instructions can be called redundant.

Some derived warm-cache L2 hit ratios exceed 100% (100.05% for one producer,
101.68% for a reducer), a replay/counter-consistency anomaly. These raw values
are retained and explicitly marked unusable for conclusions, not clamped.
Warm DRAM activity is assessed using the actual byte counters instead.
Clocks were not forced and the Windows display remained active.

## Evidence and reproduction

- [Named metrics with units and separate timing lines](measurements/q4_xplane_kpack_ncu_20260911.json).
- [Raw CSV counters, rule diagnostics and run manifest](measurements/q4_xplane_kpack_ncu_20260911.logs.tgz).
- Full `.ncu-rep` files: `E:/q4-layout-ncu-20260911-r1/results/` on the
  profiling host and `/root/autodl-tmp/q4-layout-ncu-20260911-r1/results/`
  on the development host. Their hashes are recorded in the manifest.

Compile only the CUDA-only profiler runner; reuse the source-bound DSOs and
binary fixtures from the matched A/B:

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 -I. -Iquactlize/include \
  dev/gemv_cuda/profile_xplane.cu -ldl -o /work/profile_xplane
python3 dev/gemv_cuda/run_xplane_ncu.py \
  --runner /work/profile_xplane \
  --xplane-library /work/libq4_xplane.so \
  --kpack-library /work/libkpack_gemv_n2_v2.so \
  --fixtures /work/fixtures \
  --recipes docs/measurements/q4_xplane_kpack_5090_20260911.json \
  --output /work/q4-layout-ncu-results
python3 dev/gemv_cuda/read_xplane_ncu.py /work/q4-layout-ncu-results \
  --output /work/q4-layout-ncu-summary.json
```

The runner requires a fresh result directory and rejects DSOs that differ
from the measured A/B receipt. It runs eight profiles sequentially, emits
progress, and preserves successful profiles if another arm fails.
