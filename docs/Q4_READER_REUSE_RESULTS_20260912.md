# Q4 reader reuse: PPU results, 2026-09-12

The two-shape experiment is complete. **N8192/K5120 beats raw-GGUF
reference by 7.46%; N5120/K8192 improves 7.91% over the previous K-pack
winner but remains 1.58% slower than reference.** The zero-regression
target is met on one of two shapes, not the full six-shape campaign.
Production dispatch, offline bytes and llama.cpp routing are unchanged.

Source: `4ed9c31cf2d6887a4749179e7cf053b930bc1a03`.
Archive: `q4-reader-reuse-ppu.WbTrXt.results.tgz`, SHA-256
`00ad12ddc9457d70244d1f9729ce13597bcf6f1fab9339e2c9652254b93ec83f`.
The [review receipt](measurements/q4_reader_reuse_20260912/review.json)
contains all configuration medians, round distributions, identities and
imported counters for all 24 ACU reports.

## Evidence checked

- Q4_K, dense M=1, canonical K-pack4, rotating weights only. Seven copies
  of 23,592,960 bytes exceed 2.25 times the verified 64 MiB L2. Each timed
  graph covers 35 complete calls; first-use/upload effects are excluded.
- Two shapes, two geometries per shape, eight reader variants per geometry:
  32 reader/config contexts. Six alternating rounds with 15 event samples,
  including three contemporaneous controls: **228 timing cells**.
- 24 separately profiled calls, 72 raw child logs, 829 source hashes and
  every image/log/report hash verified. All receipts, address models,
  medians and verdicts were recomputed. Fixture hashes match the preceding
  config sweep; runtime libraries match the build receipt.
- All 32 reader/config contexts passed the immutable same-C/W/P **raw FP32
  output-bit comparison**, independent GGUF dot, zero-code/zero-A, guards
  and replay checks. The matched hash is consistent across all variants
  and rounds within each of the four geometries. Maximum reader conditioned
  error is 1.268e-8; maximum across readers and controls is 3.442e-5.
- PPU-ZW810, ordinal 0, PCI `0000:08:00.0`. The known physical device has
  72 CUs; the ABI probe's `sm=1` is not used as the physical count. No
  concurrent-context warning occurs in the logs, but this is not an
  independent guarantee of an otherwise idle machine.
- Campaign wall time: **338.63 seconds (5.64 minutes)**, excluding fixture
  creation. No box compilation or JIT was performed.

Both controls accumulate/output FP32 but reconstruct individual weights
in FP16. K-pack retains its FP32 group-affine order. Cross-layout raw-bit
identity is not claimed; the exact-bit checks are against the unchanged
K-pack kernel at the identical geometry.

## Rotating event results

Times are medians of six round medians in microseconds. Negative deltas
mean faster. These are not ACU instrumented durations.

| N x K | Previous K-pack | New winner | New us | Ref us | vs previous | vs ref |
|---|---:|---|---:|---:|---:|---:|
| 5120 x 8192 | 24.9023 | v7 / C4-W8-P4 | 22.9326 | 22.5754 | -7.91% | +1.58% |
| 8192 x 5120 | 22.7474 | v6 / C4-W10-P8 | 20.7480 | 22.4200 | -8.79% | -7.46% |

The first shape is essentially tied with Xplane (22.9337 us), but still
fails the stricter reference line in every paired round: +1.30% to +1.95%.
The second beats reference in every paired round: -7.92% to -6.81%; its
Xplane gain is 7.21%. Modelled weight bandwidth is 1028.8 and 1137.1 GB/s,
respectively; these are weight-bytes/event-time, not measured DRAM rates.

## Which switches helped

`A` means cooperative activation loads followed by register shuffle; `U`
means cooperative packed-unit loads; `H32` means 32-bit metadata extraction.
The following compares variants at each winning geometry, including the
unchanged v0 clone. The full four-geometry table is in the JSON receipt.

| Variant | Switches | N5120/K8192 C4-W8-P4, us | N8192/K5120 C4-W10-P8, us |
|---|---|---:|---:|
| v0 | unchanged | 24.9211 | 22.9063 |
| v1 | A | 25.2640 | 27.7240 |
| v2 | U | 25.5046 | 23.2491 |
| v3 | A + U | 24.4880 | 23.2554 |
| v4 | H32 | 24.0286 | 21.9589 |
| v5 | A + H32 | 23.5571 | 25.6377 |
| v6 | U + H32 | 25.5823 | 20.7480 |
| v7 | A + U + H32 | 22.9326 | 20.7874 |

These switches interact; they are not independently positive optimizations.
At N8192/K5120, adding A alone costs 21.03% at C4-W10-P8. H32 plus U wins
there, while that same switch pair loses 6.68% at the old C8-W10-P4 geometry.
Turning all switches on is not a valid universal selector.

v6 and v7 at C4-W10-P8 are only 0.0394 us apart (0.19%). Preserve both as
close candidates; the medians do not establish a robust unique winner.
The alternative C8-W10-P4 v3 also beats ref (21.1789 us), so the observed
gain is not contingent on one unusually fast sample or one geometry alone.

## ACU: fewer loads, not proportionately fewer bytes

All 24 reports were imported locally through the SDK's `acu --import
--page raw --csv`. Each contains one kernel and 653 metrics; actual symbols,
template parameters and grids/blocks match the timing arm. ACU is forced-cold
replay, separate from the rotating event regime. MB below are decimal.

| N x K / arm | L1/L2 MB | Vector loads | Registers/thread | Achieved occupancy |
|---|---:|---:|---:|---:|
| 5120 x 8192 previous | 92.350 | 163,840 | 58 | 54.17% |
| 5120 x 8192 v7 C4-W8-P4 | 88.438 | 102,400 | 74 | 54.18% |
| 5120 x 8192 ref | 24.984 | 92,160 | 100 | 54.29% |
| 8192 x 5120 previous | 47.508 | 163,840 | 56 | 53.84% |
| 8192 x 5120 v6 C4-W10-P8 | 46.433 | 66,560 | 94 | 54.02% |
| 8192 x 5120 ref | 26.347 | 87,040 | 96 | 41.74% |

DRAM reads remain approximately 23.64–23.66 MB for these arms. The first
winner removes 37.5% of vector loads, but only 4.24% of L1/L2 bytes. The
second removes 59.38% of vector loads, but only 2.26% of L1/L2 bytes.
This supports reduced instruction/request overhead, not a claim that the
speedup came from reading dramatically fewer weights from DRAM.

For N8192/K5120, P4 to P8 changes each lane's B load from 8 to 16 bytes;
C8 to C4 keeps TileN=32, grid=256 and threads=320 unchanged. K workers
increase from 40 to 80 and passes fall from four to two. Under the aligned
64-byte footprint model, both geometries fill sectors: this improvement
is not a new claim of better B sector utilization. Source/ISA loop expansion
gives the following vector-load counts, whose totals agree with ACU:

| Operand | Previous C8-W10-P4 | Winner v6 C4-W10-P8 |
|---|---:|---:|
| B | 81,920 | 40,960 |
| A | 40,960 | 20,480 |
| Units | 40,960 | 5,120 |
| Total | 163,840 | 66,560 |

The per-operand split is derived, not separately measured by the aggregate
ACU counter. B bytes and offline layout are unchanged; P8 reuses A for more
columns without enabling the A-shuffle switch. U shares unit loads between
K groups. The winning kernel still uses fast code unpack and FP32 affine
arithmetic, not a precision downgrade.

For N5120/K8192, the remaining winner still uses 8-byte B loads and
32 contiguous B bytes per lane group (50% of an aligned 64-byte footprint,
25% of a 128-byte footprint). Its L1/L2 traffic remains 3.54x reference.
Yet widening C to eight with W16 reduces traffic to 46.106 MB and is slower
at 24.3366 us. Therefore neither transaction bytes nor occupancy alone
defines the best next config. Load width, work distribution and dependencies
must be considered together.

The first winner's vector-load dependency/issue ratio is 1.581 versus
reference's 0.356; the second's is 1.587 versus reference's 0.274 even though
the second kernel is faster. Such ratios are not wall-time percentages or
standalone acceptance criteria. Registers rise without reducing measured
occupancy here. K-pack's shared bank-conflict counter remains zero; ref's
shared-A path has conflicts and still performs well. No single counter
proves a causal latency breakdown.

## Next scope

1. Retain v7 C4-W8-P4 and v6/v7 C4-W10-P8 as measured per-shape anchors.
   Do not reopen their completed numerical gate or replace them globally.
2. Prioritize N5120/K8192: test a bounded P8/16-byte B reader neighborhood
   with the proven U/H32 factors and an optional A factor, rather than
   re-running 393 unchanged configurations. Preserve the prior winner,
   check 32/64/128-byte footprints and actual native load widths, and
   account for registers, CTA count and whole-warp tail activity.
3. Extend the reader/config search to the other four shapes, keeping their
   existing per-shape implementations as controls. M1 dense gains do not
   yet admit M2–7, indexed/batched MoE, other qtypes or llama.cpp deployment.
4. No rerun of this unchanged two-shape experiment is required. Future
   comparisons still need contemporaneous baseline/ref timings; historical
   times select candidates, not substitute for current measurements.
