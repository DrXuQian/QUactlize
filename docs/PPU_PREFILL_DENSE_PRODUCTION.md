# Measured dense prefill helpers

This is a helper integration, not a new GEMM route table or a model-level
performance admission. `quactlize/dense/runtime.cpp` calls the same compiled
policy and typed runtime used by the diagnostic. Public caller ABI, offline
K-pack bytes, NCP wrapper and llama.cpp code are unchanged.

## Admitted scope

PPU-ZW810, 72 CUs, contiguous dense F32 endpoints, M=2048. Other devices,
M, strides and unlisted N/K retain their previous helpers. MoE and decode
are unchanged. Full-dequant's existing production route remains BF16 with
FP32 accumulation, even in the default F16 context. TC F16 stays F16.

| Q | N | K | Typed dequant | F16 IO | BF16 IO |
|---|---:|---:|---:|---:|---:|
| Q4_K | 25600 | 5120 | 21 | 3 | 3 |
| Q4_K | 8192 | 5120 | 21 | 3 | 3 |
| Q4_K | 5120 | 8192 | 20 | 1 | 1 |
| Q4_K | 5120 | 25600 | 8 | 1 | 1 |
| Q4_K | 1024 | 5120 | 20 | 2 | 2 |
| Q6_K | 5120 | 25600 | 21 | 1 | 2 |
| Q6_K | 1024 | 5120 | 20 | 2 | 2 |

The single source is `quactlize/prefill/dense_helpers.hpp`, exposed through
`quactlize_kpack_dense_helper_policy_v1`. The query checks device/shape and
does no GPU work. The build receipt binds its hash to the private prefill
runtime; ncp still builds/installs the ordinary public package.

DQ20/21 = cooperative metadata + separate low/high shared-bank maps,
N64/K128 or N64/K256. DQ8 retains the prior N32/K128 pair reader.
IO1 = vec8; IO2 = two warp-contiguous vec4 chunks; IO3 = four chunks.
Do not extrapolate IO3 to small outputs: R4 measured a 10–12.5% regression.
Q4 5120x25600 retains DQ8 because the new full-call result was inconsistent.

Evidence: `review-1792644908006383291.tgz`, SHA256
`4f80fee84ea87413da6a11808aac7ce7cca83f6c039749400605b0ff1c59820a`,
source `afe3ee611cedd3d5b8e985bbea28588fedec2690`. Seven points / 112 arms /
40 ACU captures passed. Raw-bit IO/dequant, changed addresses, two streams,
redzones and independent GGUF dot checks passed. These are measured helper
choices; no whole-model latency is inferred from their component gains.

## Re-sweep against native

Use `tools/run_prefill_model_asys_box.sh` as documented in
`PPU_PREFILL_ASYS_SWEEP.md`. It remeasures all retained FQ/SF configs with
the production IO helpers, plus old model/current helper controls, old Full
BF16 configs, and the production typed helper combination for both Full
F16 and Full BF16. It does not cross every helper variant with every GEMM.

The typed full arms use production `prepare/run` entries. Every SF call
expands scales online; TC cases, including the old-helper controls, use
production enqueue order SF -> input -> GEMM/reducer -> output. Old-helper
controls are labelled `legacy-*`. Older diagnostic times with input before
SF are not imported into this run or relabelled as its control times.

Native must actually call cuBLAS with F16 A/B/C and FP32 accumulation;
the name `build-ci-32F-cublas` is not sufficient. Compare `same_dtype_*`
summary fields first. A faster BF16 arm is not relabelled as F16 parity.
Full F16 is a new explicit sweep candidate, not a silent production dtype
change; model numerics are still required before such a route is admitted.

## Next instruction-pressure experiment (not shipped)

In R4 N64/K256 native ISA, Q4 has 52 vector shifts / 39 integer MADs;
Q6 has 118 / 61. These are static instruction-body counts, not dynamic
instruction totals or fractions of latency. Shifts serve code/scale bit
extraction as well as address/swizzle calculations. Integer MADs include
address strides. MOVs include register setup/zero initialization, not HBM
transfers. The source still computes per-iteration producer/consumer indices.

Candidate: hoist common plane bases and strength-reduce repeated index
arithmetic, then inspect emitted code. Keep mandatory bit extraction,
separate low/high bank layouts, canonical bytes and exact affine rounding.
Investigate initialization/MOV provenance before removing it; do not disable
the compiler's `ppu-fix-uninit` correctness workaround to improve a count.
Confirm raw bits, occupancy and same-frequency cycles before promotion.
