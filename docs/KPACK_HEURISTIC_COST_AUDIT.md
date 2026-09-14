# Heuristic cost evidence audit, 2026-09-14

Small M does **not** admit full-dequant + BF16. For the existing decode
coverage this means dense M=1..8 and grouped tokens=1..8, not grouped total
rows=1..8. This audit does not invent a full-dequant admission threshold for
the remaining intermediate range or change the production policy.

## Estimated, measured, and missing are different

| Evidence | What its latency means | Action |
|---|---|---|
| Early `TC_SPLITK_S*_MODELED_E2E` / SF `MODELED_E2E_REDUCER_80PCT_NO_LAUNCH` | Measured producer plus `(4*S*M*N+2*M*N)/(0.8*2766*1000)` us; reducer launch cost assumed zero | Never promote this sum as a measured complete-call cost. Keep old winners as candidate references, not a latency oracle. |
| September overnight campaign | Measured full-output FQ/SF calls; FQ Split-K includes actual reduction | Retain the measurements in their original wrapper/cache scope. Do not add another reducer. |
| Historical runtime table and recent calibration | 2,982 historical contexts plus 90 recent overlays; exact measured tactics or explicitly labelled same-family predictions | The selector transfers tactics, not bandwidth-estimated microseconds. A predicted point still needs validation. |
| Q4 decode policy | 372 complete F32-endpoint comparisons, including real TC reduction and adapter costs | Retain small-M SIMT/TC ranking. No full-dequant candidate. |
| Old resident SF GEMM sweep | Measured GEMM with scale/zero already available | Missing per-call expansion is not the same as an estimated GEMM. Add separately measured SF expansion once before comparing routes. |
| Native per-call SF gate | Its timed SF call already includes the prepass | Do not add the separate diagnostic prepass sample again. Its 14 paired contexts are not global coverage. |
| Latest selected prefill run | 88/88 measured FQ/SF calls, Q4/Q5, tokens 2048/4096; all selected S1; SF expansion excluded | Combine with the matching already measured SF expansion. These are deployed heuristic choices, not a new exhaustive optimum. |
| Latest full-dequant + BF16 board | Both components individually measured; 44 sums | `sum_estimate_us` labels composition, not a theoretical-bandwidth dequant estimate. External adapters and the producer-consumer cache interaction are not measured by the sum. |
| Latest standalone reducer run | 48/48 measured large-output S2/S4/S8 reducers with rotating synthetic partials | Diagnostic costs, not small-M costs and not attribution of a complete producer-consumer call. |
| MFU / `distinct_MBU_model_pct` / useful-byte GB/s | FLOP or logical-byte count divided by measured time and a peak denominator | Modelled utilization is not modelled latency or an ACU-measured DRAM byte count. |

Source checks:

- Early cost formula: `tools/analyze_fq_q4k_decode_real_shapes.py::reducer_us`,
  `benchmarks/fq_q4k_decode_real_shapes_policy.json`, and
  `tools/analyze_scalefirst_q4k_real_shapes.py::reducer_model`.
- Replayed 13,585 raw logs from `kpack-overnight-c0c1361-results.tgz`:
  87,269 measured FQ S2/S4/S8 records, all `scope=FULL_OUTPUT`; no
  `MODELED_E2E` or `modeled_reducer_us` log records. These counts include
  repeated phases, not 87,269 distinct workloads. The import rejects measured
  producer-only cells in `tools/run_kpack_tuner.py::parse_cells`.
- `docs/measurements/q4_decode_policy_20260913.json.gz` binds the 96 dense
  and 276 grouped comparisons to
  `DECODE_ONLY_F32_ENDPOINTS_TC_CASTS_ADAPTERS_REAL_REDUCER_INCLUDED`.
- `kpack-prefill-gemm.KlD17K.results.tgz` SHA256
  `639e327e73a70e7f608916ca790f416ad0b999c357ddbfa48a3c0cb4987119f6`:
  88/88 PASS, 1,175 s. Five dense and six grouped weight families, Q4/Q5.
- `kpack-prefill-reducer.4rNkhE.results.tgz` SHA256
  `e20bcd21f7ded677062d154dab301047ea6f4776d632ff65ffb34481a46dcfc7`:
  48/48 PASS, 1,234 s. Dense M2048/4096 and grouped total rows16384/32768;
  no M1 fast-dense case. The result and authority JSON/file hashes match.

The old `plan_fq_splitk_reducer_lookup.py` is **not** the current supplement:
it still assumes grouped Split-K is structurally unavailable and requests
1,035 dense reducer points. Neither assumption should silently determine a
new campaign.

## Remaining work, without another config Cartesian product

| Priority | Missing evidence / implementation | Minimal supplement |
|---|---|---|
| 1 | Small-M reducer attribution | Received: 138/138 numerical PASS, 137 stable timings; recheck only `dense-m1-n4096-s4`. Complete TC calls still must not receive a second reducer charge. |
| 2 | Intermediate/large-M cross-route crossover | Start with dense M128/512/1024, then challenge only differing/close boundaries. Measure current selected FQ/SF and BF16 separately. Reuse matching weight-only dequant costs. Retain applicable historical winning configs as a bounded challenge set. |
| 3 | Q2/Q3/Q6 real-shape dequant and matched cross-route costs | Existing small numerical smokes do not price real weights. Add these formats' real geometries; do not copy Q4/Q5 timings. BF16 provider measurements may be shared only with compatible geometry/provider/precision and an explicit evidence contract. |
| 4 | Remaining weight families | Historical dense coverage has 12 N/K families; the current large-M component board has five. Add the missing in-scope families, rather than extrapolating full-dequant winners to them. Grouped board covers six families, including doubled-N gate/up. |
| 5 | Sparse MoE full-dequant | First implement/verify selection of the used noncontiguous experts on GPU. Current dequant ABI expands every expert in a supplied contiguous slice. E8 slice measurements do not establish selected IDs from E256, and an all-E256 cost must not stand in for active-only expansion. Then measure matching routing profiles/active counts and BF16 MoE. Small-token decode remains excluded. |
| 6 | Production admission and endpoints | Wire the measured dequant implementation and full-BF16 provider before admitting that path; validate direct heuristic choices and caller adapters, excluding JIT/first use. Plain reducer time must not replace a fused reduce/scatter time. |

The new component timings affect cross-route selection where coverage exists.
They do not retroactively invalidate every measured small-M tactic. Q4's
optimized SIMT ranking is not a Q2/Q3/Q5/Q6 or Q8 SIMT admission either.
No result here establishes a global best over unmeasured configurations.

The executable remaining supplement is now documented in
[Remaining heuristic component measurements](KPACK_COST_SUPPLEMENT.md):
458 finite workload points, 134 deduplicated selected/historical parent
images, separate dequant/provider/GEMM measurements, resumable by component.
No small-M full-dequant admission and no automatic production change.

## Ready box supplement: small-M reducers

```bash
cd /sim/eec/shared/junfu.qx/quactlize &&
git pull --ff-only &&
CUDA_VISIBLE_DEVICES=0 \
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
bash tools/run_kpack_decode_reducer_ppu_box.sh
```

This reuses `prebuilt/ppu0010/kpack-prefill-measure-v1/libprefill_reducer.so`:
122 dense keys and 16 grouped keys, deduplicated across format, K, router and
tactic. It tests the real M1 fast dense, generic M>1 dense and compact grouped
implementations. Small M never runs full-dequant. The 88 prefill and 48
large-reducer measurements are not repeated.

The timing uses two reused partial buffers, 64 calls per captured graph,
three rounds of five event samples, after first use. Partial/output working
sets are bounded below half of reported L2, but L2 hit rate is **not** claimed
measured. This is intentionally distinct from the previous cold rotating
large-partial experiment: small partials just produced by a GEMM need not
come from cold HBM. Neither isolated experiment proves that producer's exact
cache state. No producer, scatter, dequant, allocation or H2D is timed.
Complete output guards, fixed-order FP32-to-FP16 oracle, and a changed partial
inside the captured graph are checked. The negative must fail the old oracle.

This run has completed: `kpack-decode-reducer.aVNayM.results.tgz`, SHA256
`86a1cfc28416636551552782f7435cd3fd6217822a564048041ce2e86bfcd1d7`.
All 138 numerical tests pass. 137 points have round-median spread within
5%; `dense-m1-n4096-s4` has round medians 1.7294/1.7519/2.3944 us and needs
one recheck. Do not repeat all 138. M1 fast dense costs are about 1.73--1.93 us;
generic dense M2--8 costs 2.99--7.14 us, and compact grouped costs 1.74--1.87 us.
Successful cases survive a later failure and `RESUME_RUN` resumes only with
identical sources, package, SDK and device. Send its printed `*.results.tgz`.
