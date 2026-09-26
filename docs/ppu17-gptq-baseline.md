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
| `MarlinKernel` / `MarlinMixedInputKernel` | Generic collective, K-fast stripes, reverse peers, FP32 partial workspace, per-output locks | Existing algorithm; Hopper warpgroup/fixup adapter still pending |
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
- One idle-guarded H800 M1/N512/K2048 smoke with one config per arm passes
  complete CPU-dot and output-guard checks. It is NOT a best-config or speed
  acceptance result; no speedup is computed against the different old task.
- The six-point/108-config GPTQ screen did not start: another GPU process
  appeared at admission. No contaminated result is kept as a measurement.
- Marlin scheduler adaptation, full candidate numerics and the measured
  fastest baseline remain incomplete. Native PPU1.7 execution is unverified.

Evidence: `/workspace/quactlize-ppu17-20260926/gptq-ab`, remote counterpart
`/workspace/ppu17-h800-20260926/gptq-ab`. Reproduction and contract are in
[the GPTQ harness](../tests/ppu17/reference/README.md).
