# All-format SIMT reader experiment

Canonical Q2_K/Q3_K/Q4_K/Q5_K/Q6_K and Q8_0, dense M1--8 and indexed
MoE token1--8. `execution/simt.h` is additive; no existing policy or layout
is replaced. GPU IDs are consumed by the kernel. There is no standalone
activation gather or output scatter. S>1 includes the real F32 reducer.

## Candidates and controls

`spec.py` imports the exact production compile inventory from
`quactlize/execution/simt_codegen.py`:

- 4/8 N-sharing lanes, 2/4/8 output values per lane, 2/4/8 warps;
- independent A and packed-unit cooperation;
- S1/2/4/8; split is a runtime argument, not another compiled shape axis;
- C*P>32 is rejected. Q8 packed-unit cooperation is rejected because its
  workers have different K32 scale entries, not a common K-quant unit.

The full build has660 producer bodies, including both input types, and1320
runtime configurations across six formats. The smoke profile has88 bodies.
These are bounded inventories, not proof of a global optimum.

`tools/build_kpack_execution.py` includes these candidates in the execution
library alongside the existing optimized Q4 reader. The C entrypoints are
`quactlize_kpack_simt_query_v1` and `quactlize_kpack_simt_run_v1`.
Mixed MoE chains use `qks_moe_endpoint_v3.reuse_config` with the unchanged
run/router/finish lifecycle. These interfaces require explicit recipes;
they do not choose an unmeasured winner or modify the production heuristic.

The old scalar/pair implementation is rebuilt unchanged as a separate control.
It is **not** the optimized Q4 baseline. `q4_controls.py` builds the current
measured Q4 closure, preserving unusual previous winners such as W10/W20.
The existing372-workload Q4 registry is reused for2232 format/workload pairs;
typed TC winners must remain in the eventual PPU route comparison. Do not
promote a SIMT-only screen to a SIMT/TC or model-speed conclusion.

Storage is F16 or F32 A, with F32 output/accumulation. F32 A is rounded to
F16 in registers, preserving the existing SIMT input arithmetic boundary.
This does **not** solve the separate model activation-range issue above65504.
The new reader uses F32 group-affine dequantization; the old scalar/pair
readers reconstruct individual weights in F16. Compare both to independent
official GGUF dots, not to each other's raw bits. Existing BF16 TC endpoints
are unaffected.

## Build and validate

Use a new output directory; generated sources and manifests bind the exact
candidate list. Compilation needs no GPU.

```bash
python dev/gemv_simt/build.py --platform cuda --sdk /usr/local/cuda-12.8 \
    --output /data/simt-cuda --jobs 6
python dev/gemv_simt/run.py --sdk /usr/local/cuda-12.8 \
    --bundle /data/simt-cuda --qtype 13 --phase numeric \
    --output /data/simt-results/q13-numeric.json
```

For PPU, use `--platform ppu --sdk /path/to/PPU_SDK`. CUDA compatibility
headers are confined to `dev/gemv_cuda/compat`; they are not PPU inputs.
Run each qtype in a fresh process. The numerical gate checks every compiled
configuration on M1--8, both storage types, dense/shared-A/per-slot-A indexing,
compact rows with empty experts, output/workspace guards, mutable graph
inputs, and a zero-A negative. Successful cells are journaled if a later
configuration fails. Fixture upload is ordered on the nonblocking consumer
stream and completed outside timing.

```bash
python dev/gemv_simt/run.py --sdk /usr/local/cuda-12.8 \
    --bundle /data/simt-cuda --qtype 13 --phase perf \
    --mode indexed --tokens 1 --channels 1 --n 2048 --k 512 \
    --cache rotating --output /data/simt-results/q13-indexed.json
```

Rotating weights occupy at least2.25 times the queried L2 capacity, with
whole traversals and at least32 calls per graph. If PPU reports no L2, supply
verified `--l2-bytes`; do not guess from an unrelated device. First graph
upload/replay is excluded. Every candidate is correctness-checked before
timing. Screen all compiled recipes, confirm the top two per implementation
in alternating order, retain samples and recompute medians.

`access.py` records warp byte addresses for low/high/A/metadata and
32/64/128-byte footprint models using actual buffer-base alignment. It
explicitly exposes repeated packed-word requests across scale groups.
These models are not hardware bandwidth counters. Inspect native load widths,
register/local-memory use, half2 decoding and F32 FMAs; collect NCU on5070
and ACU on PPU before attributing performance differences.

Status: candidate and full production execution-library compilation pass.
The reviewed 5070 run passes 68,640 numerical checks and 30 performance
contexts; see `docs/SIMT_ALL_FORMATS_20260915.md` for exact scope and controls.
PPU device admission, full mixed-chain numerical gates and typed-TC comparison
remain separate. No automatic production recipe is admitted by this directory.
