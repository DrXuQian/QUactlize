# Q4 reader followup: four of six cold shapes closed

The uploaded remaining-four experiment is complete. N4096/K2048 and
N4096/K4096 now meet the user's updated **reference +5%** allowance.
Together with the preceding two large shapes, that leaves only N512/K2048
and N1024/K5120. These are measured Q4_K dense M=1 results, not admission
of grouped GEMV, other M/qtypes, or a production selector.

Source `dc7403a`; archive `q4-reader-followup-ppu.zKgNNU.results.tgz`, SHA256
`0bd831a5a7b803af80f058c65973a2cc5fd2fe37efdc6cacf0dee9b9205c2d30`.
The [review receipt](measurements/q4_reader_followup_20260912/review.json)
contains source/runtime/device authority, selected geometries, confirmation
medians and round deltas, and imported counters for all 20 ACU reports.

## Current board

Times are microseconds, medians of six alternating round medians, fifteen
samples per round. Negative deltas mean faster. The last two rows are
historical selections from their own contemporaneous cohort, not timings
remeasured in this upload or one combined six-shape measurement epoch.

| N x K | Selected K-pack | K-pack us | Ref us | vs ref | Updated gate |
|---|---|---:|---:|---:|---|
| 512 x 2048 | meta-h0 / N8-W16 | 2.923906 | 2.450312 | +19.33% | OPEN |
| 1024 x 5120 | v4 / C4-W20-P4 | 5.707308 | 5.291538 | +7.86% | OPEN |
| 4096 x 2048 | v7 / C4-W8-P4 | 6.603750 | 6.538125 | +1.00% | CLOSED |
| 4096 x 4096 | v6 / C4-W16-P8 | 10.544375 | 10.589375 | -0.42% | CLOSED |
| 5120 x 8192 | v7 / C4-W8-P4 | 22.932572 | 22.575428 | +1.58% | CLOSED, prior cohort |
| 8192 x 5120 | v6 / C4-W10-P8 | 20.748000 | 22.420000 | -7.46% | CLOSED, prior cohort |

The newly closed shapes meet the 5% line in every paired round. The two
small shapes miss it in every paired round: +19.09% to +19.53%, and +6.31%
to +8.82%. Their remaining reductions to reach the line, relative to the
selected K-pack times, are about **0.351 us (12.01%)** and **0.151 us (2.65%)**.
These are arithmetic targets, not predicted optimization gains.

## Validation

- 314 complete timing cells, 132 raw child logs, 831 source hashes and
  20 reports verified against receipts; no failed cell was used for timing.
- All candidates pass independent original-GGUF dot, zero-code/zero-A,
  guard and replay checks. Exact FP32 outputs agree within each of the
  28 matched geometries. Max conditioned error across all arms is
  `5.953598e-5`, and across reader candidates `5.080658e-5`.
- All arms accumulate/output FP32. Raw reference and Xplane reconstruct
  individual weights in FP16; affine K-pack uses FP32 group-affine arithmetic.
  Same-geometry exact bits do not imply identical cross-family rounding.
- PPU-ZW810, ordinal 0, PCI `0000:08:00.0`; 64 MiB L2 is independently
  corroborated by the runtime attribute. The ABI property's `sm=1` is not
  the known physical 72-CU count. Runtime libraries match the build receipt.
- Rotating weight rings exceed 2.25 times L2; complete ring traversals are
  timed after first-use/graph upload. A is resident. There were no concurrent
  context warnings, but absence of a warning is not proof of an idle device.
- Measured campaign wall time is **462.63 seconds**, excluding fixtures.
  There was no box compilation or JIT.

## Small-shape bottleneck evidence

All ACU reports were imported locally (`--page raw --csv`), with 653 metrics
per report and exact kernel/geometry checks. ACU uses forced-cold replay;
its durations are not substituted for the rotating event medians above.
Bytes below are decimal MB, occupancy is measured active-warps percentage.

| Shape / arm | Grid x threads | Registers | Occupancy | DRAM MB | L1/L2 MB | Vector loads |
|---|---:|---:|---:|---:|---:|---:|
| 512 x 2048 / old cooperative | 64 x 512 | 34 | 20.99% | 0.6011 | 4.5322 | 5,120 |
| 512 x 2048 / H32 cooperative | 64 x 512 | 32 | 20.34% | 0.6011 | 4.5322 | 3,072 |
| 512 x 2048 / ref | 64 x 512 | 84 | 21.77% | 0.5999 | 0.8622 | 2,560 |
| 1024 x 5120 / old C4-W10-P2 | 128 x 320 | 56 | 26.29% | 2.9732 | 22.1664 | 35,840 |
| 1024 x 5120 / winner H32 C4-W20-P4 | 64 x 640 | 58 | 25.19% | 2.9722 | 11.8488 | 20,480 |
| 1024 x 5120 / A + H32 C4-W20-P4 | 64 x 640 | 60 | 25.75% | 2.9720 | 11.4851 | 16,640 |
| 1024 x 5120 / ref | 128 x 512 | 84 | 43.11% | 2.9674 | 3.7069 | 12,800 |

DRAM reads are similar, but selected K-pack L1/L2 traffic is 5.26x and 3.20x
reference. The smallest case already matches ref's grid and block size;
simply increasing CTAs is not established as the solution. H32 removes
40% of its vector loads without a timing win. The medium case improves
with a wider per-lane column group, yet adding A reuse reduces loads further
without improving latency. These observations motivate testing the issue
dependency chain alongside ownership, not chasing one counter alone.

At aligned bases, the smallest reader has 16-byte B requests per lane at
1024-byte lane spacing: 25% unique utilization of a 64-byte footprint,
12.5% of a 128-byte footprint. The medium winner has four adjacent lanes
forming 32 contiguous bytes: 50% and 25%, respectively. These are source
request-footprint models, not directly measured DRAM efficiencies.

The [next experiment](Q4_SMALL_LATENCY_PPU.md) preserves both actual winners
and changes only the two remaining shapes. Offline bytes, shipping selection
and llama.cpp stay unchanged pending device results.
