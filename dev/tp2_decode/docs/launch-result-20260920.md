# Returned launch/chain repair and low-bandwidth follow-up

The repaired public launcher and MoE-chain admission now execute on the box.
All requested component, TP2/cache, short numerical, benchmark and trace stages
complete. This is **not** another material model decode speedup: warmed TPOT
is 11.232953 ms, versus 11.239113 ms in the preceding run. The new Q5 component
winner is not yet in the measured model package. Neither Q8 focus candidate
beats its retained incumbent.

## Authority

| Item | Identity |
|---|---|
| Uploaded archive | `kpack-tp2.yj9GRn.results.tgz` |
| Archive SHA256 | `94bee15978e1a5ff702f2e550cd494906e233059062d20126178a03ec7ab6eeb` |
| Runner source | `0aeffd7e977054bea05a369bc5c1a23e7380b588` |
| Model artifact commit | `46727e96de92c6a7da15aa12f066a31f1bed8673` |
| Model manifest SHA256 | `3a1608d2702447b7a038905bb7285d3fa6626371387bec3bde2c8eee2c7c20af` |
| Execution library SHA256 | `f7a5a5c768b5a94a4c8c613f55e83bd592018b781a9df00a9134cb61698ba213` |
| Caller source | `1f7da3bd94b1fc6b9520e156456c89221fddd07f` |
| NCP source | `9bfb44383588cbf4eed98e3d51de90e8f82d1779` |
| Box result root | `/sim/eec/shared/junfu.qx/kpack-newbox.QYl8F9/kpack-tp2.yj9GRn` |

The archive has 232 unique safe relative members, with no links or special
files. Model measurements use physical devices **6,7**. Component measurements
use device 6, reported as ordinal 0, PCI `0001:AA:00.0`; runtime-reported L2 is
67,108,864 bytes. Device idleness remains an operator precondition, not an
independently collected utilization audit.

Local review recomputed all nine confirmation summaries from their 810 raw
samples and imported **all six actual ACU reports**, not just their supplied
CSV files. Report hashes, numerical receipts, device/fixture identities,
producer names and launch geometry match. Imported timing/resource/traffic
metrics reproduce the supplied exports. No new device experiment was run here.

## Correctness and execution coverage

- 17 selected points, 28 public-entry checks: PASS. Tokens 1 through 8 are
  covered; the 11 F16 v1/v2 pairs have identical M1 snapshot hashes.
- Finish snapshot gate: all 128 combinations of token count, F16/BF16 compute,
  router mode and live-storage reuse pass four replays. Maximum observed error
  is `8.69691379e-08`. This is a prepare/snapshot/finish test, not GEMM timing.
- Cold and new-process hot cache each pass 72 TP2 cases, six chain cases and
  six segmented cases, with three replays. Synthetic M32 probes intentionally
  retain some legacy coverage; they are not model selection misses.
- Three focus points: all 22 candidate arms plus three frozen incumbents pass
  numerical controls for tokens 1 through 8; all six ACU arms pass. The
  measurements rank **M1 only**, not token counts 2 through 8.
- Both measured model processes report fully selected plans, seven modules,
  zero legacy fallbacks, 96 paired plans and 96 MoE-chain plans (both ranks).

Short likelihood checks are finite and have matching reference/self/native
coverage within each token-batch group. They do not constitute full task
accuracy admission; the result still says `PENDING_REVIEW`.

| Token batch | Context / scored tokens | Mean KLD | Maximum KLD | Same top probability | PPL ratio |
|---|---|---:|---:|---:|---:|
| 1 | 256 / 254 | 0.002293 | 0.321870 | 99.606% | 0.995060 |
| 2048 | 2048 / 2046 | 0.001621 | 0.135317 | 99.316% | 1.001439 |

The batch-1 mean KLD is below the preceding 0.002444, while maximum KLD is
above its 0.296673. Do not describe these short checks as bit-equivalence or
conclude that BF16 intermediate-rounding differences have disappeared.

## Warmed model performance

Qwen3.5-122B-A10B Q4_K_M, TP2, request batch 1, PP2048/TG128. Matched ABBA has
four measured samples per arm; each process's first warmup is excluded.
These measurements have no profiler attached.

| Metric | Reference | Repaired K-pack | Change |
|---|---:|---:|---:|
| Prefill us/token | 259.069092 | 203.288574 | -21.5311% |
| Decode ms/token | 12.618078 | 11.232953 | -10.9773% |

Decode measured samples, in process/repeat order (us/token):

- Reference: `12551.6328125, 12696.3671875, 12675.4609375, 12560.6953125`.
- K-pack: `11216.953125, 11228.765625, 11264.5234375, 11237.140625`.

The old K-pack result was 11.239113 ms/token, only 0.055% above this result.
Those are different runs/devices, not a controlled repair-only A/B. Report
decode as effectively unchanged, not a new causal speedup. The within-run
reference comparison remains valid; do not substitute profiled durations or
sum both ranks to estimate TPOT.

## Cold complete-call component decisions

The active-weight rings exceed 2.25 times reported L2. Counts are active
weights, not all allocated experts. Useful-weight MBU uses assumed peak
2700 GB/s and includes producer/reducer time where applicable; it excludes
activation/output bytes and is **not** an ACU hardware-utilization counter.

| Point / local N x K | Incumbent us | Best new candidate us | Candidate change | Incumbent / candidate MBU | Decision |
|---|---:|---:|---:|---:|---|
| Q8 K/V, 256 x 3072, F16 | 5.090939 | 5.376132 | +5.602% | 6.079% / 5.756% | Retain incumbent |
| Q8 shared down, 3072 x 1024, F16 | 6.143913 | 6.360435 | +3.524% | 20.148% / 19.463% | Retain incumbent |
| Q5 MoE down, 3072 x 512, BF16, top8 | 14.080000 | 12.835556 | -8.838% | 22.756% / 24.962% | Candidate for M1 integration |

Q8 K/V uses 835,584 useful weight bytes and 181 copies; shared down uses
3,342,336 bytes and 46 copies. Q5 uses 8,650,752 bytes across eight active
experts per call, 18 copies, and an E256 address space. Alternating routed
sets initialize/use 16 distinct experts; charging every call for E256 is wrong.

Both Q8 finalists retain their incumbent geometry and add fixed N/K. They do
not improve performance. Keep the original implementation, not a fixed-shape
candidate simply because it is newer or theoretically simpler.

Q5 incumbent: variant3/C4/W2/P8/S1/changes3, dynamic N/K. Winner:
variant3/C4/W2/**P4**/S1/changes3 with **fixed N/K**, candidate arm `3`.
Its six medians are 12.836667, 12.842222, 12.845556, 12.834445, 12.832222 and
12.826667 us; it wins all six paired rounds. The dynamic P4 arm instead takes
15.691111 us (+11.443%). Thus changing only P in a fallback bucket would be
the wrong implementation and a regression. Numerical portability to M2..8
does not establish their performance ranking.

No production selector, runtime pin or model binary was changed by this review.
The Q5 model trace still uses P8/dynamic N/K; the new 12.84 us result is not
already included in the 11.233 ms TPOT. A simple 48-layer arithmetic estimate
of this component delta is only about 0.060 ms/token per rank, before timing
scope/cache/overlap differences. Do not double that for two concurrent ranks
or present it as an end-to-end promise.

## ACU diagnosis

ACU uses forced-cache-control kernel replay and is a separate experiment from
the rotating complete-call ranking. In particular its small Q8 fixed-shape
single-launch timings look faster, but the six-round rotating comparisons
are slower. The production decision follows the matched unprofiled objective,
not whichever profiling replay duration looks best.

| Arm | Grid / block | Registers | Shared bytes | Theoretical occupancy | Active-warp counter | DRAM read bytes | L1/L2 total bytes |
|---|---|---:|---:|---:|---:|---:|---:|
| Q8 K/V incumbent | 16 / 256 | 74 | 512 | 75% | 2.737% | 857984 | 3545600 |
| Q8 K/V fixed | 16 / 256 | 78 | 512 | 75% | 2.733% | 856192 | 3545088 |
| Q8 shared down incumbent | 96 / 128 | 78 | 512 | 75% | 8.234% | 3355520 | 7010432 |
| Q8 shared down fixed | 96 / 128 | 78 | 512 | 75% | 8.216% | 3353856 | 7010304 |
| Q5 down incumbent | 768 / 64 | 114 | 256 | 50% | 32.803% | 8691712 | 17837056 |
| Q5 down P4-fixed | 1536 / 64 | 58 | 128 | 100% | 58.469% | 8777216 | 33949568 |

The active-warp metric is
`cu__warps_active.avg.pct_of_peak_sustained_active`, distinct from the
launch occupancy limit. All six shared-memory bank-conflict counters are zero.
`derived__l2_bytes_pipe_l1_total` sums global load/store/atomic traffic between
L1 and L2; it is not unique DRAM reads. The installed SDK section definition
was inspected before interpreting it.

Q8 useful-weight bytes and DRAM reads are close. Small K/V has only 16 CTAs
for 72 compute units, with unchanged geometry and almost identical traffic
after fixing N/K. This supports investigating launch/parallelism and
dependency overhead, not claiming that huge DRAM over-reading is the cause.
The current narrow-tile alternatives still lose; lower register use alone
is not a ranking rule.

Q5 P4-fixed halves register/shared usage and doubles available CTAs. However,
L1/L2 traffic grows about 1.90 times and vector-load instruction count grows
from 30,720 to 61,440. The vector-memory-pipe-busy issue ratio rises from
0.105562 to 0.946839, while the dependency ratio falls from 0.877612 to
0.669828. These are issue-normalized counters, not percentages of wall time.
The result is a concurrency/register tradeoff, **not** evidence of improved
memory coalescing. All three useful-weight MBU goals remain open.

## Actual model trace and remaining uncertainty

Both trace arms pass the second-request PP2048/TG16 proof. Input, request and
response hashes agree. The uploaded statistics count 67,490 native kernels
and 76,402 reference kernels across prefill and both ranks.

Q8 now executes `measured_decode_kernel<8,...>`, fixing the public-v1 bypass.
All 48 layers appear in chain plan logs. Snapshot prepare executes 1,316 times
(47 layers x 14 steady replays x 2 ranks), and the original no-snapshot
prepare executes 28 times (last layer x 14 x 2). Chain SwiGLU and weighted
finish each execute 1,344 times. Q4 gate/up and Q5 down measured producers
each execute 1,440 times. Their captured averages, 18.841 and 16.475 us,
are almost unchanged from the preceding trace's 18.836 and 16.497 us.

Relative to the old complete capture, independent topk calls fall from 1,508
to 192. Independent gated activation, multiply and eight-source sum each
lose 1,316 calls; snapshot prepare, chain SwiGLU and weighted finish replace
them. Net launch reduction is **1,316**, not the sum of removed helpers.
The new helper averages are respectively 7.332, 5.193 and 2.591 us. Fusion
does not make their work free; complete-chain cost still matters.

The 192 residual topk calls are consistent with two prefill batches across
48 layers and both ranks, but aggregate counts alone do not locate them on
the timeline. Likewise gather/scatter names in the whole capture must not be
declared decode regressions. The uploaded archive contains Asys statistics,
not this run's `.asysrep` or SQLite timeline. The two reports already under
`/root` belong to the preceding run and must not be relabeled as this run.

To resolve current per-token critical-path/collective waiting and residual
helper locations, upload the already generated files under the box result
root's `results/trace/qwen35-122b-q4km/{native,reference}/proof.asysrep`.
This needs no new box execution. Until then, no exact explanation of the
flat TPOT is claimed from mixed prefill/decode or two-rank duration sums.

## Next work

1. Integrate the Q5 P4-fixed M1 winner with exact implementation identity;
   keep the old donor for unmeasured dynamic/table-miss cases and M2..8.
   Verify the compiled body, typed entry and full-call scope before model A/B.
2. Retain both Q8 winners. Design another bounded small-shape experiment from
   the CTA/traffic evidence, explicitly comparing complete call cost and the
   old winners; avoid repeating the losing fixed-only change.
3. Use the new timeline to separate useful compute, chain helpers and TP2
   collective wait. Optimize chain SwiGLU/prepare as measured components,
   not by promising savings equal to removed launch counts.
4. Preserve validated prior cells, cold/hot cache coverage, first-warmup
   exclusion and the physical 6,7 assignment. No unrestricted resweep is
   justified, and full model accuracy remains a separate admission item.
