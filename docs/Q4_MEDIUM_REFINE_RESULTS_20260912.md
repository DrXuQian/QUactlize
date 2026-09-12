# Q4 cold M1: six-shape reference margin closed

N1024/K5120 now meets the user's **raw-GGUF reference +5%** criterion:
selected K-pack is **5.324615 us**, reference **5.290000 us**, a **+0.65%**
gap. Each of the six paired rounds is within 5%; the worst is +1.44%.
Together with the five previously reviewed shapes, the declared Q4 dense
M=1 cold-weight scope is **6/6 closed**. This is a bounded best-measured
selection, not proof of global optimality or model-level speedup.

Source `36d5e4b`; archive `q4-medium-refine-ppu.q3vvwX.results.tgz`, SHA256
`db548ce2620c040aa62b8aab6e5f506b91851428216f65d1417d54613925a605`.
The [review receipt](measurements/q4_medium_refine_20260912/review.json)
records source/image/runtime/device authority, all screen medians, six
round medians, log/report hashes, address models and imported ACU counters.
Production dispatch, shipping libraries, offline bytes and llama.cpp are
unchanged.

## Current confirmation cohort

Microseconds, median of six alternating round medians, fifteen event
samples per round. These are resident full-call rotating-weight timings,
not profiler durations. All arms use FP32 accumulation/output.

| Arm | Confirmed us | vs raw reference |
|---|---:|---:|
| Xplane control | 6.836731 | +29.24% |
| Raw-GGUF FP32 reference | 5.290000 | 0.00% |
| Immutable K-pack P2/W10 anchor | 5.673654 | +7.25% |
| Immutable K-pack P4/W20 anchor | 5.658654 | +6.97% |
| **P4/W20, original fold, unsigned indices** | **5.324615** | **+0.65%** |
| P4/W20, explicit fold, unsigned indices | 5.363077 | +1.38% |
| P4/W20, explicit fold, signed indices | 5.405192 | +2.18% |

Selected key: `p4-w20-r0-u1`, recipe `(4,20,0,1)` in the isolated
`q4-medium-refine-v1` package. It is 6.15% faster than the preceding P2
selection, 5.90% faster than the same-geometry P4 anchor, and 22.12% faster
than this cohort's Xplane. These controls were remeasured, not imported
from a different time window.

Paired reference gaps, in round order: -0.015%, +1.437%, -0.036%, +0.537%,
+1.104%, +0.889%. No threshold rounding was needed to admit this result.
The nearby confirmed variants also pass; their ordering is not a claim
that this particular winner is universally better.

## Change and address contract

The winning source keeps the original CTA fold (`R0`). `U1` makes known
nonnegative thread/work indices unsigned. It does **not** change the
canonical format, lane ownership, FP32 addition order, dequant formula or
number of kernel launches. Geometry remains 64 CTAs x640 threads,
TileN16, 160 K workers and one K pass. This is S1, with an intra-CTA fold;
there is no inter-CTA Split-K or separate reducer.

In the emitted PPU ISA, the fold's static shared-load sites drop 16 to 10
and conditional branches 18 to 2. The same simplification appears with
the separately tested explicit constant-round fold. These are static
code sites, **not** sixteen versus ten dynamic loads by every thread.
One CTA barrier, `lop3`/half2 fast code construction, and FP32 FMA remain.
Unsigned addressing also changes load/address lowering outside the fold,
so this experiment does not assign the entire gain to fold branches alone.

Observed A/B/metadata bases are all aligned modulo 128. Each B request is
8 bytes/lane, with four adjacent lanes forming a contiguous 32-byte run:
100%/50%/25% unique footprint utilization at 32/64/128-byte granularity.
Per representative instruction, A has fourfold lane duplication and
metadata eightfold duplication. All these byte addresses and widths are
unchanged relative to the same-geometry P4 parent. They are source request
models, not measured DRAM traffic. No shared-A staging or AIU was added.

The selected reader uses FP32 group-affine arithmetic. Controls reconstruct
individual weights in FP16 before their FP32 accumulation. Same-geometry
reader variants match exactly; cross-family bit identity is not required
or claimed.

## ACU cross-check

All five actual ACU reports were imported locally, 653 metrics per report.
Exact symbols and launch geometry match the selected reader, both immutable
anchors, Xplane and raw reference. The three shortlisted candidates share
one geometry, so profiling one geometry plus the four anchors correctly
produces five reports, not six.

Forced-cold replay metrics below use decimal MB. Event results above remain
the performance authority.

| Counter | P4 signed anchor | P4 unsigned winner | Raw reference |
|---|---:|---:|---:|
| Registers/thread | 58 | 58 | 84 |
| Shared bytes/CTA | 1,280 | 1,280 | 10,368 |
| DRAM bytes read, MB | 2.9719 | 2.9706 | 2.9674 |
| L1/L2 traffic, MB | 11.8220 | 11.9291 | 3.7069 |
| Vector load instructions | 20,480 | 20,480 | 12,800 |
| Shared-load transactions | 1,280 | 1,280 | 167,936 |
| Shared-load bank conflicts | 0 | 0 | 245,760 |
| Executed scalar branches | 3,328 | 2,688 | 10,752 |
| Executed instructions (`pu__inst_executed.sum`) | 1,008,064 | 975,232 | 1,643,520 |
| Achieved occupancy | 25.72% | 27.21% | 43.18% |

This supports simpler control/address execution, not a reduction in B
bytes or improved coalescing. P4's dynamic shared loads remain identical
despite fewer static load sites. K-pack still has materially more L1/L2
traffic than raw reference; parity does not mean that overhead disappeared.

The instrumented replay durations were 5.781 us for the P4 anchor and
5.916 us for the winner: this individual replay does **not** reproduce the
event gain. Do not combine these regimes or claim the counters prove a
universal speedup. The admitted result is the six-round rotating event
comparison. Vector-memory dependency/busy per-issue ratios also move in
opposite directions; neither is a wall-time percentage.

## Verification and scope

- Reparsed **68 timing cells** and five profile cells from **40 hashed child
  logs**. All 22 candidate screen cells, shortlist, samples, medians,
  comparison/TSV values, device identity and complete profile set agree.
- **839 source hashes**, candidate DSO, full parent-package verification,
  native ISA receipt and runtime libraries match. The fixture digest matches
  the preceding experiment. The compact review records the authority-file
  hash and canonical source-map hash; the full map remains in the archive.
- All candidates pass independent original-GGUF dot, zero-code/zero-A,
  guards and replay checks. FP32 fingerprints agree across R/U variants at
  all nine geometries; eligible parents match the actual immutable image.
  Maximum conditioned error is `2.7578722e-5` across all arms and
  `8.4589082e-9` among new reader candidates.
- PPU-ZW810, PCI `0000:08:00.0`, ordinal 0. The 64 MiB L2 override is
  corroborated by the runtime attribute. The properties field `sm=1` is
  not the physical 72-CU count. No warnings occur; exclusive device use is
  not independently proven by the logs.
- A resident activation with 52 rotating weight copies of 2,949,120 bytes
  exceeds 2.25x L2. Each graph traverses 104 complete calls. Setup, JIT and
  graph first-use are excluded. This is not an all-input-cold model run.
- Campaign time: **144.66 seconds (2.41 minutes)**, excluding fixture setup.

## Combined six-shape closure ledger

Each row uses its own contemporaneous controls. This combines reviewed
cohorts; it is **not** a new simultaneous six-shape measurement.

| N x K | Selected key | K-pack us | Raw ref us | vs ref |
|---|---|---:|---:|---:|
| 512 x 2048 | meta-h0-l1-a1-w16 | 2.524766 | 2.437812 | +3.57% |
| 1024 x 5120 | p4-w20-r0-u1 | 5.324615 | 5.290000 | +0.65% |
| 4096 x 2048 | v7-c4-w8-p4 | 6.603750 | 6.538125 | +1.00% |
| 4096 x 4096 | v6-c4-w16-p8 | 10.544375 | 10.589375 | -0.42% |
| 5120 x 8192 | v7-c4-w8-p4 | 22.932572 | 22.575428 | +1.58% |
| 8192 x 5120 | v6-c4-w10-p8 | 20.748000 | 22.420000 | -7.46% |

Earlier evidence: [small latency](Q4_SMALL_LATENCY_RESULTS_20260912.md),
[reader followup](Q4_READER_FOLLOWUP_RESULTS_20260912.md),
[reader reuse](Q4_READER_REUSE_RESULTS_20260912.md). Repeated key strings
must retain their associated package/body identity during integration.

Stop further tuning of this closed dense M1 scope. Next, extend and admit
**dense M=2..7** and **indexed/batched M1 MoE with real routing**, measuring
the full call including any adapters/reduction. Then integrate the admitted
selection through Quactlize dispatch/JIT and validate llama.cpp performance
and actual route execution. Other qtypes, model no-slowdown and prefill
remain separate tasks; this result does not admit them.
