# Restore a representative GPTQ baseline

The previous synthetic common-group64 GGUF comparison was rejected by the
user. Preserve its raw evidence but do not use it for acceptance or selectors.
The new comparison keeps the already-supported GPTQ symmetric INT4 contract:
group128, desc_act=false, original FP16 scales, FP32 accumulation, FP16 A/C.
Only native packing may differ. No GGUF metadata or requantization is added.

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

## Current evidence (2026-09-26)

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

### Remaining work

1. Finish same-clock full-cohort confirmation of the self-reset prepared
   path, retaining the controls. Native Stream-K still requires its reset.
2. Complete small-M Marlin mainloop reuse/adaptation, not just its scheduler:
   CTA/K-warp topology, asynchronous staging, vector decode and FP32 fixup.
   The original PPU001 register/MMA ownership is not a Hopper mapping.
3. Reconfirm the faster-reference target per shape, then expand grouped,
   dtypes and shapes. No slow case may be hidden by the aggregate mean.
4. Separately port the mature GGUF reader/metadata/pipeline mechanisms while
   preserving its group16/32 and canonical planes. GPTQ scores do not close
   this work. Native PPU1.7 compiler/device admission remains unverified.

Evidence: `/workspace/quactlize-ppu17-20260926/gptq-ab`, remote counterpart
`/workspace/ppu17-h800-20260926/gptq-ab`. Reproduction and contract are in
[the GPTQ harness](../tests/ppu17/reference/README.md).
