# Q4 indexed MoE S1 gate

This ports the tuned dense Q4 SIMT readers to the existing indexed/compact
call contract. It does **not** change the offline format, production
heuristic, llama.cpp selection or the existing fused MoE chain.

## What the kernel does

One launch reads GPU expert IDs, chooses an activation row, reads that
expert's canonical low/metadata slices, computes the dot and writes FP32
output directly in token/slot order. No standalone IDs helper, gather,
directory build, scatter or inter-CTA reducer is needed for this S1 path.
There is one output row per CTA group, not a cross-row B-reuse algorithm.

| Mode | Expert | Activation | Output row |
|---|---|---|---|
| Dense | 0 | row | row |
| Compact grouped | device offsets lookup | compact row | compact row |
| Indexed gate/up | `ids[token,slot]` | token (`channels=1`) | `token*topk+slot` |
| Indexed down | `ids[token,slot]` | token/slot (`channels=topk`) | `token*topk+slot` |

Strides are in elements, with padded activation/IDs/output strides tested.
The actual model-facing F32 activation is rounded to F16 in registers. Dot
accumulation/output are FP32. META reconstructs each weight in F16; the
MEDIUM/REUSE readers perform FP32 group-affine arithmetic. Those arithmetic
contracts are intentionally not labelled identical.

Implementation: `quactlize/execution/q4_s1_api.h`, `q4_s1_kernel.cuh`,
`q4_s1_readers.cuh`, `q4_s1_activation.cuh` and `q4_s1_validation.hpp`.
The recipe API is additive; old `qkg_config_v1` calls retain their meaning.
The row bodies/helpers have a reproducible mechanical transplant test against
the frozen dense sources. Production headers do not include development code.

## Exact box scope

- Q4 only, E=256, topk=8; tokens **1 through 8**, both activation conventions.
- N/K: 512/2048, 512/3072, 2048/512, 3072/512 plus merged gate/up
  1024/2048 and 1024/3072.
- 108 cases: 96 dispersed-routing cases plus 12 token8 clustered cases.
- 16 explicit recipes per shape; 192 compiled specializations including
  both input types. No Cartesian TC build or JIT on the box.
- Independent official-GGUF dot; exact F32-rounded/F16, dense/indexed and
  compact/indexed comparisons; empty experts; padding guards; zero-A,
  zero-code and invalid-ID controls; graph replay after modifying GPU IDs.
- For N512/K2048 META recipes, exact comparison with the **old immutable
  dense DSO**, not merely the new implementation agreeing with itself.
- Five screening samples per recipe; top two receive six alternating rounds
  of fifteen samples. Graph upload/first launch/setup are excluded.
- Rotating weights: count only the eight active experts when sizing the ring,
  not all 256 allocated slots. At least 2.25 times verified L2, complete ring
  traversals. Approximately 5 GB of device scratch/weights for this gate.
- Default ACU: 12 profiled calls (token1 gate/up and clustered token8 down,
  two per shape) in six reports. ACU forced-cold times are separate from
  rotating event measurements. It is not an end-to-end model benchmark.

For every recipe, the receipt includes lane address/32-64-128-byte footprint
models and a separately recomputed F32 A model. Build receipts include native
load/shared/shuffle/fast-dequant/FP32 instruction counts per specialization.
These are source/static-ISA facts, not measured traffic or utilization.

## Run

From the quactlize checkout on an otherwise idle card:

```bash
git pull --ff-only origin develop && \
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
CUDA_VISIBLE_DEVICES=0 bash tools/run_q4_moe_s1_ppu_box.sh
```

The script fetches only the required LFS payloads, compiles nothing, prints
per-case/recipe progress, and preserves the caller's Docker shell on failure.
Send the printed `q4-moe-s1-ppu.*.results.tgz` archive back. `ACU=0` disables
profiling; it does not change event samples or correctness checks.

To resume, set `RESUME_RUN` to the previous run directory. Completed cases
are checked against the source/package/device identity and exact sample
denominators; failed shapes restart in a fresh process. Existing reports
require a successful hashed receipt to be reused. Raw failures are retained.

## Remaining work after this gate

The result is a bounded SIMT recipe winner, not yet the best GEMV/GEMM route.
Next combine admitted SIMT with all previously measured TC winners, compare
complete calls (including necessary reductions/adapters), then update the
heuristic. Never label an unmeasured TC candidate slower.

The llama.cpp chain currently rejects direct GEMV plans. Integrating this
reader must preserve gate/up fusion, shared routing and intermediate layout;
simply choosing isolated GEMVs could regress model latency by disabling that
fusion. Q5/Q6 down matrices in Q4_K_M models require their own admission.
Neither that integration nor other-format parity is certified by this gate.
