# MoE decoding configuration sweep

`moe-config-models` extends the existing Asys runner. It is **not** the old
SIMT-only `moe-single-row-models` cohort; old runs keep their frozen sources.

## Scope

- Actual GGUF headers: Qwen3.5-35B-A3B TP1, Qwen3.5-122B-A10B TP2,
  DeepSeek-V4-Flash TP4. An explicit model subset is supported; the default
  requires all three. Use the caller's CPU split function, not N/K division.
- All actual supported expert qtypes and local N/K/E/top-k/channel families,
  M=1..8. Gate/up and down keep their different activation indexing.
- M1 spread; M2..8 spread/cluster/repeat2/repeat4/synthetic-real routing.
  These are synthetic weights and IDs, not captured model traffic.
- F32 endpoints, direct BF16 compute, FP32 accumulation. Canonical K-pack
  bytes, production kernels and selection are unchanged.
  SIMT retains its existing FP32 group-affine dequantization; TC reconstructs
  BF16 weights. These legal operation orders have separate numerical checks,
  not an assertion of bitwise equality between implementations.
- SIMT, Q4 specialized readers, ordinary and persistent TC. Runtime Split-K
  and resource-resolved persistent grids remain explicit search axes.

Blind rules exclude the target N/K family, discard historical timings and
rank donor geometry. One-axis neighbors and common SIMT rules generate the
pool. Historical recipes are then checked for coverage; a missing champion
fails planning rather than being appended. The current public C++ selection
is always an independent control, including its existing fallback path.

Grid modes and blocks/CU expand only after querying the exact compiled BF16
endpoint on the assigned device. Equivalent actual launch grids are timed
once; their aliases and unavailable-resource proofs are retained. A resource
query error or unavailable current control is not a candidate win.

## Run on idle ZW810 cards

```bash
cd /sim/eec/shared/junfu.qx/quactlize
git pull --ff-only origin develop
unset RUN RESUME_RUN PLAN_ONLY
BASELINE=quactlize COHORT=moe-config-models \
MOE_MODELS=qwen35-35b-a3b,qwen35-122b-a10b \
LLAMA_CI_DIR=/sim/eec/shared/junfu.qx/llama.cpp.ncp \
GPUS=4 JOBS=8 PPU_SDK=/usr/local/PPU_SDK \
L2_BYTES=67108864 IDLE_CONFIRMED=1 \
bash tools/run_smallm_regressions_asys_box.sh
```

Set `GPUS` to the otherwise idle physical cards available for this run.
Each point's screen, control and finalists stay on the same physical card.
No llama binary is built or run. `MODEL_ROOT` defaults to
`/sim/eec/shared/AI_workspace/llm-models/bench_model_zoo`;
`QWEN35_MODEL`, `QWEN122_MODEL`, `DEEPSEEK_MODEL` override exact first shards.
`PLAN_ONLY=1` performs header/CPU selection checks without a GPU or SDK.

`MOE_MODELS` maps to `--moe-models`. The current example explicitly defers
DeepSeek: the inspected `UD-Q4_K_XL` checkpoint contains MXFP4 (qtype 39)
expert weights, not Q4_K. MXFP4 is not a supported K-pack MoE format. The file
name does not authorize interpreting its tensors as Q4_K. Omitting a model
is never automatic: `MOE_MODEL_SCOPE`, the sealed plan and the final summary
record the requested/deferred sets, and COMPLETE refers only to that request.
Every requested model/family must still cover M1..8; an entirely filtered
model fails with its qtypes and exclusion reasons. The two-Qwen selection
does not resolve DeepSeek or establish native/model no-regression.

The runner archives the committed sources and initialized, pinned local
actlize. No HTTPS nested submodule fetch is attempted. All latency records
carry source commit, source/DSO hashes, SDK/device and Asys trace identities.

## Timing and result interpretation

1. Compile shared modules once, before timing.
2. Fresh Asys screen: 4 alternating rounds x 3 samples, at most 12 live arms
   per batch. Numerical/negative checks precede timing. Identical helper
   components are measured once across screen batches, not per config.
3. Confirm the current control, each compute family's best and candidates
   within 5% by mean or a full-round mean. Confirmation is 6 x 5 alternating
   samples. More than 32 competitive finalists is explicit incomplete work,
   not an arbitrary top-K truncation; the completed screen remains reusable.
4. Every timed graph contains one kernel. Sum actual prepare/main/reducer/
   finish components once. Do not add host launch estimates or reuse old
   latencies. Rotating active weight reads exceed 2.25 x verified L2.

`MOE_MODEL_COVERAGE` prints actual shape/type counts; local synthetic tests
do not establish those counts. `MOE_SWEEP_EXEC` prints resource-expanded arm
counts and `MOE_SWEEP_SCREEN` prints measured/finalist counts and wall time.
Estimate the remaining run from completed points, including build/fixture/
profiler overhead, not from microsecond kernel times.

For resume keep the original run, source/cache/results and card assignment:

```bash
BASELINE=quactlize COHORT=moe-config-models \
RESUME_RUN=/workspace/kpack-smallm-asys.REPLACE_WITH_RUN \
PPU_SDK=/usr/local/PPU_SDK IDLE_CONFIRMED=1 \
bash tools/run_smallm_regressions_asys_box.sh
```

Resume inherits the frozen model subset if `MOE_MODELS` is unset; supplying a
different subset is rejected. A failure before `plan.json` (such as an
unsupported checkpoint) needs a new run after changing scope, not resume.

Passed points and valid screens are verified and not remeasured. Upload only
`results/review-lite-*.tar.xz`; raw `.asysrep`, SQLite and binaries stay on Box.
Do not merge samples across changed source/device identities.

This selects the best **measured bounded configuration**, not a proven global
optimum. It does not update production tables. Large M, BF16-storage entries,
gate/up/SwiGLU fusion (backup), native MUL_MAT_ID comparison and whole-model
TPOT remain separate gates. Q8 uses resident scales; any K-quant SF control
uses its explicitly declared standalone helper, not a hidden full-dequant.
