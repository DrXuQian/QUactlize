# JIT native gate: sparse grid and fixture ordering

## Current result: both corrected runs pass

Archive: `kpack-jit-retry.5gSImm.results.tgz`.
SHA256: `d4d4a4b80191c0ef80eddf026c08c8db2d6de27e005074b52fd4256aaf224c5a`.
Source: `72b6db864fc335ad15f48165943d2cf42c243177`.
Package: `prebuilt/ppu0010/kpack-jit-v2`, manifest SHA256
`d9a2b7f6c689a287eccc0e8b1a62d2aa3cc5a186c6f1eff0b1929ecd04d85266`.
Device: ordinal 0, PCI `0000:08:00.0`, one visible device.

| Check | First process | Restarted process |
| --- | --- | --- |
| Native contexts | 28/28 PASS | 28/28 PASS |
| Dense FQ / SF | 10 / 10 PASS | 10 / 10 PASS |
| Grouped FQ / SF | 4 / 4 PASS | 4 / 4 PASS |
| Eager checks / graph replays | 28 / 84 PASS | 28 / 84 PASS |
| Independent scale+zero prepass proofs | 14/14 PASS | 14/14 PASS |
| JIT disk-cache resolutions | 28/28 hits, zero compiles | 28/28 hits, zero compiles |

Both JSONs match the published manifest. Every selected parent/build key is
unchanged between processes and matches the original 23 compiled parents.
All 168 recorded replay-profile errors are finite and below the `5e-3`
condition-scaled dot-error limit; the maximum is `2.448390734577721e-4`.
All 840 recorded full-call samples are finite and positive. This confirms
the declared native gate, not additional shapes or a full-model release.

Q4 SF-grouped now prepares and runs with grid=256. Q6 SF-dense M128 passes
eager and replay checks in both processes with the **same device binary** as
the failing run. Together with the local delayed-queue negative, this closes
the operational regression through host-grid and fixture-order corrections,
without changing the collective. The old log did not retain enough raw
output to reconstruct the precise historical NaN interleaving.

### Bounded full-call timing observations

Both processes produce the same 14/14 FQ/SF choices when checked by the V2
exporter. The times below are medians of per-profile medians in microseconds;
each range is the two process medians, not a confidence interval. SF includes
its **per-call GPU metadata expansion**. This is the native gate scope, not
llama adapter/model latency or an ACU-only kernel duration.

| Context | FQ us | SF full-call us | Both runs favor |
| --- | --- | --- | --- |
| All five tested dense M1 contexts | 14.60--20.64 | 25.52--34.80 | FQ |
| Dense M128, Q2--Q6, N1024/K5120 | Per-qtype comparison | SF is 14.35--32.60% faster | SF |
| Q4 grouped, tokens=1, M8/N512/K2048/E256 | 18.32--18.44 | 75.08--75.24 | FQ |
| Q5 grouped, tokens=1, M8/N2048/K512/E256 | 18.48--18.64 | 71.56--71.80 | FQ |
| Q4 grouped, tokens=128, M1024/N512/K2048/E256 | 121.04--121.60 | 192.32--192.40 | FQ |
| Q5 grouped, tokens=128, M1024/N2048/K512/E256 | 254.56--255.96 | 218.72--219.12 | SF |

These observations do not change production routing in this review.
Automatic decode stays FQ; no all-prefill SF promotion is justified by this
small workload set. Standalone prepass samples include early/cold effects
and must not be added again to the already-combined SF samples.

Cache resolution is not free: the per-resolution median was 0.4515 s in the
first process and 0.3945 s after restart, with tails of 5.039 s and 3.430 s.
This gate creates a fresh dispatcher for each context and reports helper
wall time; it is not a per-token cost. Keep startup/TTFT and warm-cache
resolution overhead tracked for the model gate, rather than calling it zero.

The native retry is complete. Next: the updated llama adapter with selected
JIT/cached modules, graph execution, real routing and full-model latency.
Same-parent prebuilt/JIT A/B, full model startup and removal of the six
intake/fallback libraries remain separate tasks. No repeat of these 23 cold
compilations is required while the cache/source/SDK identities remain valid.

## Original failing result (2026-09-10)

Archive: `kpack-jit-box.maMVzW.results.tgz`.
SHA256: `c384fcadaf2d20e2f4a3732bac3c3388d2ad9142e6ec621a19dca1292b734422`.
Source: `2927e59d98280c2a048cbeef69e7d2da5c0d04aa`.
Device: visible ordinal 0, PCI `0000:08:00.0`.

The first process completed 26/28 contexts. All 14 FQ contexts passed; 12/14
SF contexts passed. These are bounded native/operator observations, not a
full-model or all-shape release. The command stopped before the restart run.

| Failure | Selected parent | Observation |
| --- | --- | --- |
| Q4 SF grouped, tokens=1, M=8, N=512, K=2048, E=256, max_rows=1 | `sfg_q12_tm16_tn16_tk256_wm16_wn16_s2_ap0_dn16` | Module prepare returned `QK_UNSUPPORTED=1` |
| Q6 SF dense, M=128, N=1024, K=5120 | `sf_q14_a0_tm16_tn32_tk256_wm16_wn16_s3_bc0_ap0_dn32` | Output was nonfinite; the old log did not retain the bad count, coordinate or eager/replay phase |

Independent packed-unit scale and zero checks passed immediately before both
failures. The same Q6 DSO passed M=1. Its key was
`3f848fd908e6f7ff1a90f01bfbc69f540b971b0704c47d25be84a19f4f43fba3`.
Do not infer that the M=128 arithmetic or the later metadata values were
correct from the earlier prepass check.

There were 23 cold parent compilations: 42.281--59.309 seconds each, median
49.342 seconds, sum 1140.981 seconds. Five subsequent cache resolutions took
0.341--0.593 seconds each. Those numbers are compiler/helper wall time on
this box, not kernel timings. Retain `/workspace/kpack-jit-cache` for retry.

## Confirmed host recipe defect

The old recipe used `E * ceil(max_rows/TM)` to calculate grouped work. For
the sparse Q4 row that is 256 M blocks, although there are only eight rows.
The kernel's compact directory uses its tighter, total-row-aware bound:

```
active_bound = min(M, E)
m_blocks = min(E * ceil(max_rows/TM), floor((M + active_bound*(TM-1))/TM))
work = m_blocks * ceil(N/TN) * split
```

Here `m_blocks=8`, `work=256`. With occupancy >=4, the selected capacity
policy formerly requested `72*4=288` CTAs. The grouped kernel explicitly
rejects `grid_ctas_override > logical_work_upper` during `can_implement`.
Resource query checked residency but not that directory inequality.

The dispatcher now uses the same bound, including the Split-K axis, before
applying the existing capacity/balanced policy. This row requests 256 CTAs.
The parent, AP, delivery N, split and scheduler kind do not change. Dense
recipes and the measured ordinary Q4/S4 and Q5/S1 decode choices are unchanged.
CPU tests compare recipe grids to the **actual shipping directory helper**
across all supported TM values, splits and sparse/ragged cases. The old
Q4 recipe is red at occupancies 4/6/12; the corrected recipe is green.

## Confirmed harness ordering defect (retry closure above)

The gate used `SDK.fill`, a default-stream device memset, while its prepass
and GEMM ran on a nonblocking stream. Device memset may return before it
finishes. There was no dependency preventing a delayed NaN fill from
overwriting freshly decoded scale/zero, or an output fill from overwriting
GEMM output. This is the same fixture-order hazard previously recorded in
`4ec9688`; the new native gate had not adopted `Resources.fill`.

The gate now queues all poison fills on the consumer stream and drains
pageable fixture uploads before consumption. These setup operations are
outside captured/timed GEMM calls. Each SF call still enqueues its actual
prepass, including every graph replay; there is no cross-call scale cache.

The CPU test executes the gate's actual poison helper under a legal delayed
default-stream interleaving. Legacy ordering produces NaNs; same-stream
ordering produces the expected values. A missing prepass remains red. This
proves the test defect, **not** a reconstruction of the original PPU NaN
interleaving. The corrected PPU row subsequently passed both processes above.

The next gate checks eager output before capture and poisons again before
each replay. If it fails, the JSON includes selected parent/recipe, phase,
profile, first coordinate/raw bits, nonfinite/poison counts and output/golden
hashes. Route export rejects receipts lacking ordered-fixture and separate
eager/replay evidence; incomplete or failed runs cannot produce a policy.

## Delivery and completed retry procedure

Use `prebuilt/ppu0010/kpack-jit-v2`. Only the small host dispatcher was
rebuilt. The module ABI, compiler, source contract, all device kernel bodies
and execution DSO are unchanged. Existing valid JIT entries require **zero
device recompilations** under the same source/SDK/host-tool identity. The old
package and uploaded evidence remain intact.

No changes were made to collective loads/conversions/MMA, barriers, reducer,
offline bytes, adapter D2H or production stream synchronization. Only the
grouped host grid formula and test setup/evidence changed. Parent instruction
counts are unchanged because those binaries are reused. Kernel and full-model
performance still require the corresponding device measurements.

The retry ran the native gate in two fresh processes with the same JIT cache
and returned `KPACK_NATIVE_GATE PASS contexts=28/28` twice. The second process
checked restart/cache reuse; an old cached file alone would not be that proof.
Keep the old failure archive and the compiled cache for reproducibility.
