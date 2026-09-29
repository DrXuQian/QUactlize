# Restore a representative GPTQ baseline

The measured results below belong to the previous3.6 dependency. See
[PPU CUTLASS4.3.0 migration](ppu17-cutlass430.md) for current build status.

The previous synthetic common-group64 GGUF comparison was rejected by the
user. Preserve its raw evidence but do not use it for acceptance or selectors.
The new comparison keeps the already-supported GPTQ symmetric INT4 contract:
group128, desc_act=false, original FP16 scales, FP32 accumulation, FP16 A/C.
Only native packing may differ. No GGUF metadata or requantization is added.

## Independent Machete / Stream-K acceptance (2026-09-26)

Machete is now an **independent** performance requirement. A complete Marlin
winner, or the Machete reader with our Marlin scheduler, cannot satisfy it.
`--family machete-streamk` checks the compiled config descriptors and excludes
those alternatives before screening and finalist selection. The provider
reuses the pinned upstream MMA-prepacked reader/collective, TMA/cooperative
WGMMA, and PPU CUTLASS3.6's native StreamKScheduler. This is not just renaming
a scheduling flag, nor a claim to have rewritten the upstream implementation.

On idle H800 PCIe, the nine reference geometries and eleven pure Stream-K
candidate configs passed all six screen points (120 configuration cells).
Two finalists per arm plus required RAW-equality controls were reconfirmed
in three alternating rounds. Both selected arms have matched1755/1593MHz
SM/memory clocks; weights rotate through a ring larger than2.25x the queried
50MiB L2. Report the median of three per-config medians, not the fastest round.

| M | N | K | Reference Machete us | Integrated Machete us | Delta |
|---:|---:|---:|---:|---:|---:|
|1|512|2048|11.603|11.552|-0.44%|
|8|512|2048|11.237|11.440|+1.81%|
|128|512|2048|12.462|12.445|-0.13%|
|1|4096|4096|16.034|16.282|+1.54%|
|8|4096|4096|15.839|16.178|+2.14%|
|128|4096|4096|24.049|23.428|-2.58%|

All six points, including the worst individual paired round (+3.480%), are
within the previously accepted5% tolerance. **Strict no-slower at every
point/round is NOT achieved.** Full prepared calls include any required
Stream-K reset and FP32 reduction. Packing, allocation, JIT and first use are
excluded on both sides. Original GPTQ group128 codes/scales and FP32
accumulation/reduction remain fixed. These are synthetic dense GPTQ fixtures,
not real-model or grouped-GPTQ measurements, and not native PPU1.7 admission.

The first confirmation had two large-M point attempts with candidate1740MHz
versus reference1755MHz. Preserve these as clock-unmatched evidence. A bounded
last-point retry supplied the first matched attempts for those two rounds;
the already-matched third round and other15 points were kept. The assembly
records source receipts/hashes and selects by clocks, never by latency.

The CPU planner audit compiles `tests/ppu17/streamk_plan.cu` against each
unchanged CUTLASS tree. At54 same-geometry points, grids, raster order,
split factors, work counts and workspace sizes agree.26 Stream-K cases have
a different stored `big_units` field: regular work assignment recomputes its
per-group count, while separate-reduction peer lookup uses the stored value.
All54 audited plans use no separate-reduction units. This is scoped planner
evidence, not a proof of all scheduler paths or grounds to transplant modern
SM100-related code into the PPU dependency. Cached work assignment differs
between versions but is not established as the cause of the small timing gap.

Evidence under `/workspace/quactlize-ppu17-20260926/gptq-ab`:

- Immutable binaries: `build-aligned-stage-v1` (unchanged for this gate).
- Screen: `results/cohort-machete-only-v1`.
- Original/retry: `results/confirm-machete-only-v1`,
  `results/confirm-machete-last-retry-v1`.
- Admitted assembly: `results/confirm-machete-clock-matched-v1`.
- Independent gate/table: `results/machete-only-report-v1`.
- Planner audit: `scheduler-plan-v1`, including the unnormalized differences.

No production defaults, canonical GGUF bytes, PPU1.0/1.5 or dependency sources
changed. The table establishes this bounded H800 implementation parity only.

## Existing implementations to reuse

| Existing entry | Mechanism | PPU1.7 integration status |
|---|---|---|
| `benchmarks/test_lowbit_dense_bench.cu` / `StreamKKernel` | Mixed-input mainloop with Stream-K decomposition and FP32 fixup | GPTQ now uses CUTLASS3.6's SM90 RS/TMA cooperative collective plus native StreamKScheduler; it does not import the legacy CTA driver |
| `MarlinKernel` / `MarlinMixedInputKernel` | Generic collective, K-fast stripes, reverse peers, FP32 partial workspace, per-output locks | SM90 cooperative adapter implemented; two consumer groups, Cluster1, one CTA/SM; tested with conventional and MMA-prepacked readers |
| `StandaloneMarlinCfg` / `MarlinCollectivePPU` / `MarlinKernelPPU` | Separate native packing, K-warp topology, cp.async mainloop and C-chain cooperation | More than a scheduler. Do not rename a policy and claim this complete stack was ported |
| `moe_grouped_marlin_ppu.cuh` / `GroupMarlinMixedInputKernel` | Ragged expert output prefix with grouped Marlin scheduling | Existing isolated mechanism path; SM90 metadata/async boundary still needs admission |

The pure integer `MarlinStripeSchedulerCore` is already in actlize's
`ppu_tile_scheduler_marlin_core.hpp`. Reuse its work/peer algebra. Its device
wrapper, CU query, CTA barriers and accumulator ownership are architecture
seams, not interchangeable Hopper types. The source dependency and both
include graphs must remain isolated; no PPU1.0/1.5 kernel is modified.

The former q17 bring-up control serialized raw loads, B dequantization,
shared-memory stores, WGMMA and waits. It was useful for initial correctness,
but not a valid proxy for the library's mature baseline. Scheduler selection
alone cannot recover the missing producer/consumer pipeline. Establish the
complete existing-equivalent path before a performance optimization sweep.

## Complete implementation reuse (2026-09-26, aligned-v1)

The optional `gptq_marlin.cuh` now integrates the **whole** pinned Hopper
Marlin device implementation, not only its work scheduler: cp.async pipeline,
vector dequantization, MMA/K-warp topology, persistent stripes, FP32 reduction,
output and final lock reset. Its host packer reuses Quactlize's existing
classic address functions, verified independently against upstream Python.
Do not conflate this with porting the old PPU001 m8n16 register maps to SM90.
The provider deliberately reuses upstream code rather than claiming a new
algorithm. The optional Machete provider retains all nine upstream default
geometries on PPU CUTLASS3.6, plus the existing Marlin-scheduler controls.

All six compiled Marlin instruction bodies exactly match the separately built
reference. All18 same-config Marlin comparisons (three configs across six
points) have identical output bits, pass the CPU dot and guards, and survive
repeated graph execution. Packing/scales match upstream on N128 and N512
fixtures. PPU1.0/1.5, dependencies, canonical GGUF bytes, public ABI and product
routing are unchanged. Native PPU1.7 compilation/performance remains UNVERIFIED.

The aligned screen admits180 configurations. The first attempt's final point
failed the clock gate; only that point was retried. `aligned-complete-screen`
assembles five original admitted points plus the separately admitted retry,
with matching binary/runner identities and preserved failed-attempt provenance.
Do not call the original `cohort-aligned-v1` a complete run.

| M | N | K | Best integrated provider | Integrated us | Faster reference us | Gap |
|---:|---:|---:|---|---:|---:|---:|
|1|512|2048|Machete/Stream-K|11.810|11.511|+2.60%|
|8|512|2048|Machete/Stream-K|11.759|11.468|+2.54%|
|128|512|2048|Prepacked/cooperative Marlin|10.285|12.491|-17.66%|
|1|4096|4096|Complete Marlin|11.553|11.545|+0.07%|
|8|4096|4096|Complete Marlin|11.521|11.512|+0.08%|
|128|4096|4096|Prepacked/cooperative Marlin|20.790|23.994|-13.35%|

These are SCREEN medians, not a strict no-slower verdict. Complete Marlin
removes the prior20–24% large-decode deficit. The small-weight deficit remains
open. `confirm-aligned-v1` stopped at the last point of round1 on the clock
gate; this is not three-round confirmation. No numerical failure is reported.
The next bounded challenge retains every winner and varies only pipeline
depth4/8 for prepacked128x16x64, requiring same-scheduler RAW equality.

### Three-round decode confirmation

`build-aligned-stage-v1` adds four stage4/8 alternatives. All12 preceding
device bodies remain instruction-identical. Do not install a blanket shallow
pipeline: the large-M128 versions are slower and keep their existing winner.
`cohort-aligned-stages-v1` admitted five complete points and stopped on the
sixth point's old config7 clock gate. This is not a complete204-config screen.

The four admitted decode points were then explicitly selected for three
alternating-order confirmation rounds (`confirm-aligned-decode-v1`). All12
fresh-process point-runs passed, including full CPU dots, guards, graphs and
transitive same-math RAW controls. The reported config minimizes the median
of its three round medians; no fastest-round selection is used. Units:us.

| M | N | K | Provider/config | Integrated | Faster reference | Delta | Paired round delta range |
|---:|---:|---:|---|---:|---:|---:|---:|
|1|512|2048|Prepacked Stream-K s8 /16|11.484|11.690|-1.77%|-2.15% to -0.85%|
|8|512|2048|Prepacked Stream-K s8 /16|11.384|11.284|+0.89%|-4.79% to +3.53%|
|1|4096|4096|Complete Marlin /2|11.521|11.532|-0.10%|-0.30% to +0.57%|
|8|4096|4096|Complete Marlin /2|11.285|11.264|+0.19%|-1.07% to +0.44%|

This closes the large implementation deficit, not a strict every-point
no-slower target: only the first point has a win in all three paired rounds;
the other three cross zero. Prefill has screen evidence above but is NOT
covered by this decode confirmation. Product selection remains unchanged.
The final host-only follow-ups improve recursive control selection, retain
other points after a fresh-process failure, check invalid uint4 input and
compare encoded SASS/control words. They do not change the measured C++ DSOs;
the immutable manifests retain their original build-time script hashes.

## Historical scheduler/reader evidence (2026-09-26)

- Three comparison DSOs compile with CUDA12.8 SM90a. Quactlize uses pinned
  PPU CUTLASS3.6; references use pinned vLLM and its own CUTLASS. Original
  dependency/device files are unchanged.
- CPU tests preserve original codes/scale bits, independently invert our
  signed-int4 pack, compare Marlin's pack/scales with its upstream CPU code,
  and reject incompatible group/zero/precision/act-order inputs and old logs.
- Six-point screens completed:108 baseline configurations,120 with Marlin
  scheduling,144 with explicit stage4/8 controls. All CPU dots/guards passed;
  stage variants also preserve their same-tile/scheduler output bits. The
  scheduler or stage count alone does not close the large decode gap.
- The optional `gptq_prepacked.cuh` provider reuses pinned Machete packing,
  conversion and collective on **PPU CUTLASS3.6**, not reference CUTLASS4.4.
  It combines that reader with native Stream-K or the ported Marlin scheduler.
  Independent host checks cover8192 coordinates,256 aligned16B vectors,2738
  block offsets, and8160 wrong coordinates for a stale-layout negative.
- Generated128x16x64 code changes48 LDS.U8 sites to3 LDS.128 sites, with12
  WGMMA sites unchanged. These are static instruction facts, not dynamic
  memory transaction counts or percentages of time saved.
- Its six-point/108-config H800 screen passed. Small-weight points approach
  the faster reference, and the large prefill point is a screen win. Large
  decode remains substantially slower than standalone Marlin. Three-round
  reconfirmation stopped on a changing-clock gate, not a numeric mismatch;
  therefore no final parity/production admission is claimed.
- All timing uses complete prepared calls including required reset/reduction,
  rotating weights >2.25x queried L2, original GPTQ semantics, FP32 accumulation
  and idle guards. Foreign jobs are never stopped; clock-rejected samples are
  retained. The first valid clock-stable attempt is used, at most three per
  configuration; no fastest-attempt selection is allowed.

### Latest complete reader screen

H800 PCIe, FP16, group128. Microseconds. These are screening medians, not
confirmed optimality or a native PPU1.7 result (`cohort-prepacked-v3`).

| M | N | K | Quactlize prepacked | Machete best | Marlin best | Versus faster reference |
|---:|---:|---:|---:|---:|---:|---:|
|1|512|2048|11.914|11.762|12.089|+1.30%|
|8|512|2048|12.021|11.622|11.955|+3.43%|
|128|512|2048|12.497|12.403|17.225|+0.76%|
|1|4096|4096|15.778|15.992|11.502|+37.18%|
|8|4096|4096|16.277|15.588|11.205|+45.26%|
|128|4096|4096|22.576|23.941|27.588|-5.70%|

The old conventional baseline was19.078/18.895us for large M1/8 and27.161us
for large M128, on the same GPTQ contract. Do not compare these to the
superseded common-group64 GGUF experiment.

### Self-reset prepared-call follow-up

The wrapper's redundant lock clear is now an explicit candidate, with the
original per-call reset retained as its control. All six native device bodies
are byte-for-byte instruction-identical between build-prepacked-v1/v2; only
the host prepared-call behavior changes. The kernel still performs every
FP32 fixup and resets its final-peer locks. First-use preparation is excluded
equally with native Marlin. Workspaces are reused serially on the same stream.

Five points in `cohort-self-reset-v1` passed CPU dots, guards, repeated graphs
and exact equality against the per-call-reset controls. Screening medians:

| M | N | K | Quactlize best | Faster reference | Gap |
|---:|---:|---:|---:|---:|---:|
|1|512|2048|11.885|11.787|+0.83%|
|8|512|2048|11.991|11.714|+2.36%|
|128|512|2048|10.165|12.403|-18.04%|
|1|4096|4096|13.422|11.182|+20.03%|
|8|4096|4096|13.924|11.195|+24.38%|

The final M128/N4096/K4096 point failed all three clock-stability attempts.
The cohort is INCOMPLETE; do not manufacture its missing value, discard the
clock guard, or claim complete parity. No further candidates were launched
after this bounded checkpoint. All failed/contaminated attempts are preserved.

### Remaining work after implementation alignment

1. Finish same-clock full-cohort confirmation of the self-reset prepared
   path, retaining the controls. Native Stream-K still requires its reset.
2. Close the remaining small-weight prepacked gap and confirm same-device
   finalist timings. Full Marlin reuse is now implemented for the stated dense
   GPTQ scope; grouped/dtype/shape expansion is still separate work.
3. Reconfirm the faster-reference target per shape, then expand grouped,
   dtypes and shapes. No slow case may be hidden by the aggregate mean.
4. Separately port the mature GGUF reader/metadata/pipeline mechanisms while
   preserving its group16/32 and canonical planes. GPTQ scores do not close
   this work. Native PPU1.7 compiler/device admission remains unverified.

Evidence: `/workspace/quactlize-ppu17-20260926/gptq-ab`, remote counterpart
`/workspace/ppu17-h800-20260926/gptq-ab`. Reproduction and contract are in
[the GPTQ harness](../tests/ppu17/reference/README.md).
