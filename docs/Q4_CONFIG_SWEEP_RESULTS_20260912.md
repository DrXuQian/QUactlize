# Q4 cold config sweep: measured results, 2026-09-12

**The experiment completed; the zero-regression-to-reference target did not.**
All 393 declared configs passed their numerical screen. Confirmation and
profiling are complete, but 0/6 selected K-pack results are no slower than
raw-GGUF reference. Five of six are within 5% of Xplane. No production policy,
offline format or llama.cpp route is changed by this report.

Source `d510faf4005ec6a3b93bf5df82e69e45f9acad40`; upload
`q4-config-sweep-ppu.BdEY54.results.tgz`, SHA-256
`dbfac3b191d4944d78a1e0777457c347fe573a6c7e550dce9831923e0c02e741`.
The [review receipt](measurements/q4_config_sweep_20260912/review.json)
retains source/image/fixture/runtime identity, all confirmation medians and
the imported metrics for all 24 ACU reports.

## Verified scope

- Q4_K, dense M=1, six N/K shapes, canonical K-pack4; one kernel per call,
  no inter-CTA Split-K or external reducer. Xplane/reference accumulate FP32.
- 393 screen cells, 216 confirmation cells, 24 profiles, 174 hashed child
  logs. Recomputed config inventories, shortlists, medians, address patterns
  and final verdicts match. All log/report/build/source hashes match.
- Independent GGUF-dot, zero-code, zero-A, guards and deterministic replay
  gates executed in the bound runner. Maximum condition-normalized error is
  2.503e-8 in the new screen and 5.954e-5 across confirmation/control arms.
  This does not extend numeric admission to other M/qtypes or whole models.
- Six alternating confirmation rounds, fifteen graph-event samples each.
  PPU-ZW810 at PCI `0000:08:00.0`; verified L2=64 MiB, weight rings >=2.25 L2.
  The ABI's `sm=1` field is not physical CU count (known device: 72 CUs).
- Actual measurement campaign wall time: **603.35 s**, fixture creation
  excluded. No compile/JIT on box. No concurrent-context warning appears in
  the profiler logs; this alone is not proof the machine was otherwise idle.

K-pack group-affine and the FP16-reconstructed reference have different
weight rounding orders; both dot accumulators/output are FP32. Numeric
acceptance is independent of that difference, not cross-arm bit equality.

## Cold event times

Times are medians of six round medians, in microseconds. Negative delta
means K-pack is faster. The previous winner is remeasured, not imported.

| N x K | Selected K-pack | Previous K-pack | Selected us | Ref us | vs ref | vs Xplane |
|---|---|---:|---:|---:|---:|---:|
| 512 x 2048 | previous cooperative reader | 2.9401 | 2.9401 | 2.4466 | +20.17% | -14.08% |
| 1024 x 5120 | C4/W10/P2 | 6.9248 | 5.9242 | 5.2992 | +11.79% | -13.17% |
| 4096 x 2048 | C8/W16/P4 | 7.1038 | 6.9741 | 6.5334 | +6.74% | +0.50% |
| 4096 x 4096 | C8/W16/P4 | 11.7469 | 11.3063 | 10.5975 | +6.69% | +0.67% |
| 5120 x 8192 | C4/W8/P4 | 24.9537 | 24.9343 | 22.5451 | +10.60% | +8.66% |
| 8192 x 5120 | C8/W10/P4 | 23.0023 | 22.7549 | 22.3646 | +1.75% | +1.74% |

N1024/K5120 improves 14.45% against the previous winner; N4096/K4096
improves 3.75%. N5120/K8192 selects the same C4/W8/P4 body/config as before:
its -0.078% difference is **not** an implementation improvement. At
N8192/K5120 all six paired rounds still trail ref by 1.22–1.95%, so the
old 5% allowance cannot be used to call it a new-target pass.

The largest confirmation round-median span is 2.78%, on C4/W10/P2 at
N1024/K5120. Its paired ref regression remains 9.95–13.52%, not a reversal
hidden by noisy medians. No shape passes the stricter ref line in any round.

## What ACU establishes

ACU forced-cold replay is a separate regime from rotating event timing.
Its instrumented duration is not substituted into the table above. The
following uses selected K-pack, not a losing config or an old profile.
MB here means decimal MB; occupancy is achieved active-cycle occupancy.

| N x K | L1/L2 MB: K-pack / ref | Vector loads: K-pack / ref | Occupancy: K-pack / ref |
|---|---:|---:|---:|
| 512 x 2048 | 4.532 / 0.862 | 5120 / 2560 | 20.51% / 21.82% |
| 1024 x 5120 | 22.141 / 3.707 | 35840 / 12800 | 26.30% / 43.12% |
| 4096 x 2048 | 9.693 / 5.069 | 32768 / 18432 | 39.91% / 41.58% |
| 4096 x 4096 | 19.484 / 10.097 | 65536 / 36864 | 41.69% / 41.61% |
| 5120 x 8192 | 92.305 / 24.952 | 163840 / 92160 | 54.17% / 54.36% |
| 8192 x 5120 | 47.475 / 26.367 | 163840 / 87040 | 53.95% / 41.81% |

The N5120/K8192 pair is especially diagnostic: both have 320 CTAs x256
threads and nearly equal DRAM reads (23.657 vs23.639 MB). K-pack uses only
512 B of static shared memory. Its measured occupancy is effectively equal
to ref, yet L1/L2 traffic is **3.70x** and vector-load count **1.78x**. Vector
load-dependency per-issue ratio is 1.123 vs0.377. A generic “too little
occupancy/shared-memory bound” explanation is not supported for this pair.

Total instruction count is actually lower for selected K-pack than ref on
all six shapes; for N5120/K8192 it is 8.20M vs9.32M. Thus “more total
instructions” is also insufficient. Memory requests and dependency chains
are more actionable than maximizing occupancy or minimizing one aggregate
counter. Per-issue stall ratios are not wall-time percentages.

Different regimes still matter:

- N1024/K5120 gains speed by doubling CTA count from64 to128 and raising
  achieved occupancy from13.75% to26.30%, despite *increasing* L1/L2 traffic
  from11.78 to22.14 MB. This supports an underfilled-small-grid tradeoff,
  not a universal instruction to minimize on-chip bytes first.
- N4096/K2048 and N4096/K4096 roughly halve prior K-pack's L1/L2 traffic
  with C8/W16/P4, but retain almost the same occupancy and gain only1.83%
  and3.75%. They still make about1.78x ref's vector loads.
- N8192/K5120 keeps TileN32 but changes W8 toW10. K workers become40 and
  160 K groups fit four full passes, instead of five at W8. Occupancy rises
  43.74% to53.95%, with similar on-chip bytes; event improvement is1.08%.

## Pattern and fast-dequant interpretation

All observed new-config plane bases are represented in the uploaded
per-row alignment models. Under the aligned64 B model, C4/P4 requests32
contiguous B bytes per K group; C8/P4 requests64. Metadata and A have their
own footprints and duplication, so a fully coalesced B request does not
prove the whole reader is efficient. The source footprint is not an exact
transaction simulator: compiler load formation and device coalescing/cache
behavior still need the native ISA/counters.

The new configs all use the existing unsigned `lop3 + half2` code extraction;
ISA/code evidence is in the uploaded `isa-stats.json`. Code unpack is fast,
but metadata uses separate packed bit extraction and FP32 group-affine
work. There is no evidence here that the canonical offline format must be
changed, or that Xplane must become a runtime fallback.

## Next bounded experiment

Retain these confirmed per-shape winners. Do not rerun393 unchanged configs
as a substitute for improving the reader, and do not claim this bounded
space was a proof of global optimality.

1. Start with N5120/K8192 and the nearby N8192/K5120 control. Isolate
   cooperative **contiguous A loads plus register redistribution** from
   current duplicate/strided loads. For C4, four lanes can load disjoint
   16 B portions of one64 B A group in a single source load step; all still
   need the same full A group, so the added shuffle/register cost must be
   measured. Merely making one lane load and broadcasting does not by itself
   guarantee fewer transactions. Preserve dot order and all offline bytes.
2. Independently test reuse of packed-unit headers across K groups and
   simplify metadata bit extraction. Reduce repeated decode/load work without
   introducing a scale workspace or silently altering rounding.
3. Treat the N512/K2048 cooperative baseline separately: it already uses an
   A register-transpose reader, so do not propose the same change as new.
   Isolate metadata/load formation and reduction overhead against raw ref.
4. Confirm any changed reader with the previous winner and both controls,
   then extend to the six shapes. Admission of dense M1 does not yet admit
   dense M2–7, indexed/batched GEMV or llama.cpp end-to-end performance.

The counters support these as experiments; they do not establish in advance
that a particular shuffle, staging or scheduling change is profitable.
