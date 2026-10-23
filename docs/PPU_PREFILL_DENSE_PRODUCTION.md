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

## Retired experiments

The high-plane unique-read R5 candidate was not admitted: paired cycle medians
changed by -1.21%, +0.08%, -0.67%, +6.16% for Q6 large/small and Q5 large/small.
Two samples per arm do not establish the small improvements; Q5 small regressed.
The R1–R5 runners and unselected typed helper bodies have been removed from
develop. Source and historical commands remain at commit
`8a004d42a91a23664e69762509d34c3b76bbf8e8`; do not run them against a new library.

The retained production helpers are typed IDs 0/8/20/21: direct, pair N32/K128 and cooperative
N64/K128,K256. Their IDs, descriptor ABI and the selection table above are
unchanged. IO0..3 and the separate general SF/full-dequant config namespace
are retained. All removed typed IDs fail validation rather than falling back.
Independent GGUF, ownership/tail and changed-address/stream tests now live
together in `tests/test_prefill_dense.py`.

## One-command remaining-helper closeout

`tools/run_dense_helpers_box.sh` covers the 33 returned model families at
M128/256/512/1024/2048, plus two labelled Q2 format-extension families. These
are not inferred DeepSeek TP4 shards. The inventory and prior Full-F16 helper
recipes (not historical times) live in `dev/dense_helpers/cohort.json`.

Explicit candidates22/23 now cover Q2/Q3/Q8 paired-plane N64 K128/K256.
Q4/Q5/Q6 retain0/8/20/21, including each old full-call winner. Both F16 and
BF16 are checked/timed independently. Q2 preserves its d/dmin and16 groups;
Q3 retains its odd14-byte-superblock header; no artifact or rounding changes.

The runner builds one small component probe, partitions whole families onto
the requested GPUs, and measures each dequant N/K/storage once per card and
each IO M/width/direction/storage once per card. Single-kernel rotating graphs
are allowed here, never around cuBLAS. Each scalar/direct baseline remains.
Two combinations shortlisted from isolated components proceed automatically
to complete ordinary calls under Asys, along with prior and production-helper
controls and native F16/FP32. Forward and reverse pools plus bracketing native
captures expose order drift. First complete warmup/primer rings are excluded.
No fresh TC sweep; no automatic production selector, dtype or bucket changes.

```bash
PPU_SDK=/usr/local/PPU_SDK \
LLAMA_CI_DIR=/sim/eec/shared/junfu.qx/llama.cpp.ncp \
LLAMA_CI_BUILD_DIR=/sim/eec/shared/junfu.qx/llama.cpp.ncp/build-ci-32F-cublas \
GPUS=1,2,3,4,5,6,7 L2_BYTES=67108864 IDLE_CONFIRMED=1 \
bash tools/run_dense_helpers_box.sh
```

Use only available idle GPUs. A resume supplies `RESUME_RUN` printed by the
command and the same GPU list/SDK/caller build. Successful component cells
and successful Asys batches are verified and reused. A bad component blocks
only its card's finalist stage; other cards continue, all errors stay visible.
The top-level `review-*.tgz` contains numeric proofs, stage/sample times,
source/device receipts and short error logs. Weights, DSOs, SQLite and GUI
reports stay on Box. The tables distinguish isolated sums from full-call
Asys timings; neither is a whole-model TPOT result.
