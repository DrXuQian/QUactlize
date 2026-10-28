# MoE configuration sweeps

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

## Reviewed Qwen selection-table refresh

The five-card Qwen run at source `9c4fe3b6f3a342e2073661b9427bfc173d1bc511`
completed 144 points: four local weight families, M1..8, with five synthetic
router profiles for M2..8. Its review archive SHA-256 is
`55e8bd5bdc175a1c1396bcf43ac65f2f1e89ff6c4c9ba569583ea7a3d8b38022`.
The reviewed evidence supersedes the earlier partial Q5-only table; it does
not add a runtime policy layer. The existing generated C++ catalog remains
the sole matched-query lookup. No runtime ID readback or tuning is added.

For each of the 32 geometry/M keys, intersect the fully confirmed public
recipes across all its router profiles, retaining the current control.
Require mean producer + helper/reducer cost to be no slower than that control
on every profile. Among eligible recipes minimize worst regret against each
profile's measured best, then total mean time; exact ties retain the control.
Do not discard a candidate merely because its variance is higher, substitute
minimum samples, or promote a screen-only result.

Nine exact recipes change relative to the measured source:

| Model-local family | Changed M | Selected configuration |
|---|---|---|
| 35B TP1 Q4 N512/K2048, channels1 | 2 | reader2/V7/C4/W8/P8 |
| 35B TP1 Q4 N512/K2048, channels1 | 6, 7 | reader2/V7/C8/W4/P4 |
| 122B TP2 Q4 N512/K3072, channels1 | 5 | reader2/V6/C4/W4/P8 |
| 122B TP2 Q4 N512/K3072, channels1 | 7 | reader2/V7/C8/W4/P4 |
| 35B TP1 Q5 N2048/K512, channels8 | 1 | V3/C4/W2/P8/S1 |
| 35B TP1 Q5 N2048/K512, channels8 | 3, 5 | V3/C8/W4/P4/S1 |
| 122B TP2 Q5 N3072/K512, channels8 | 2 | V3/C8/W4/P4/S1 |

All use E256/top8, F32 endpoints, BF16-rounded activations and unchanged
FP32 accumulation. Q4 is its
existing specialized SIMT reader; all selected calls are S1. The Q5 M1 change
is only 0.28% in this run, not a substantial new optimization. The other
23 keys retain the contemporaneously confirmed control and refresh provenance.

The five per-profile TC wins remain in the evidence. They do not establish
a better default across every router. Seven static rows still have over 5%
regret against a router-specific winner; their `router_sensitive` diagnostic
is retained. This is no regression against the tested current Quactlize
control, not global optimality, native llama.cpp parity or model TPOT.

All existing bucket donors, dense/F16 choices, offline bytes and device
arithmetic are unchanged. The matched v3 API remains F32-storage-only.
The separate Q4 BF16-storage API already borrows F32 selection geometry; it
can inherit these proposals but is **not** admitted as a fresh BF16-storage
performance result. Fused gate/up shapes and large M are not covered here.

Reimport the original archive with the measured Python 3.12 arithmetic,
then regenerate the existing catalog:

```bash
python3.12 tools/fit_moe_decode.py --review /path/to/review-lite-1793162603239819679.tar.xz --check
python3 tools/generate_decode_selection.py --check
python3 -m pytest -q tests/test_moe_decode_policy.py
```

Building the execution package regenerates Q4's selected module closure,
including the prior explicit tuples. Refresh the complete package/manifest;
do not combine a new dispatcher with a stale execution library. Normal host
table verification needs neither the original archive nor a GPU.

## Next closure: selected/native AB, then large-M execution pilot

These entrypoints require the revision containing this section. They are not
present in the earlier `9c4fe3b6` checkout. Commit/deliver the complete source
change before using the committed-archive wrapper; no old result is relabelled
with a new source tag. The two commands below use the same cards: run them
sequentially, or allocate disjoint idle GPUs.

### Small M: actual selected route versus native

```bash
cd /sim/eec/shared/junfu.qx/quactlize
unset RUN RESUME_RUN PLAN_ONLY QTYPES N K
BASELINE=native-ab COHORT=moe-native-models \
MOE_MODELS=qwen35-35b-a3b,qwen35-122b-a10b \
LLAMA_CI_DIR=/sim/eec/shared/junfu.qx/llama.cpp.ncp \
LLAMA_CI_BUILD_DIR=/sim/eec/shared/junfu.qx/llama.cpp.ncp/build-ci-32F-cublas \
GPUS=3,4,5,6,7 JOBS=8 PPU_SDK=/usr/local/PPU_SDK \
L2_BYTES=67108864 IDLE_CONFIRMED=1 \
bash tools/run_smallm_regressions_asys_box.sh
```

The inspected Qwen headers yield 144 points/32 shape-M keys, 14 selected
SIMT module specializations and no selected TC at this revision. This is not
a new configuration sweep: each point runs native/candidate/candidate/native
in four fresh Asys processes, with 6x5 independent component samples per pass.
All four passes stay on the same physical PCI device. Native dispatch and its
activation quantization, if selected, remain unchanged and explicitly labelled.
The test adapter builds against the specified existing llama backend; it does
not rebuild llama or enable the production K-pack caller.

Candidate and native must consume identical original GGUF bytes, F32 A and
expert IDs. Both use input/router repeat1 for timing, and byte fingerprints
are checked in addition to independent numerical controls. Indexed output
coverage is `tokens * top-k * N`, not the dense `M * N` denominator. Compare
the arithmetic mean of the complete sum of separately measured kernels;
do not use minimum samples or drop noisy points. Valid regressions survive
resume; they are not retried until a favourable sample appears.

`comparison.csv` contains model-local q/M/N/K/E/top-k/channels/router,
producer/helper/reducer costs, total delta and source commit. A completed
comparison with regressions returns 2; an infrastructure/coverage failure
returns 1. Only PASS establishes no regression for this measured component
scope, not model TPOT. Upload `results/review-lite-*.tar.xz`; retain all raw
traces and the frozen source on Box.

### Large M: finite execution gate before another full sweep

```bash
cd /sim/eec/shared/junfu.qx/quactlize
pilot_root=$(mktemp -d /workspace/kpack-moe-prefill.XXXXXXXX)
test -n "${pilot_root}" && test -d "${pilot_root}"
python3 -u tools/run_moe_prefill_pilot.py \
    --output "${pilot_root:?}/results" --sdk /usr/local/PPU_SDK \
    --llama-source /sim/eec/shared/junfu.qx/llama.cpp.ncp \
    --llama-build /sim/eec/shared/junfu.qx/llama.cpp.ncp/build-ci-32F-cublas \
    --moe-models qwen35-35b-a3b,qwen35-122b-a10b \
    --devices 3,4,5,6,7 --jobs 8 --l2-bytes 67108864 --confirm-idle
```

Local uncommitted/staged/untracked files do not block this direct runner.
It records HEAD as the revision anchor, labels execution `WORKTREE`, and binds
the actual source/harness hashes separately. Retain those files unchanged
through resume. Archive wrappers still execute committed HEAD, not local edits.
`--resume --output <same results>` with the same devices/model subset keeps
successful cases and retries only failed/missing cases. `--plan-only` stops
after header/CPU TP/rule/capability checks, without needing a PPU or llama DSO.
Use the same model-path overrides as the small-M runner, expressed as Python
options (`--qwen35-model`, `--qwen122-model`, `--deepseek-model`).

The full audit covers M16/32/64/128/256/512/1024/2048/4096/5120 and five
synthetic router profiles: 200 actual Qwen-local points. Blind donor rules
retain historical-recipe recall without importing old timings or appending
missing champions. Its 20,810 legal seed arms and 99 TC modules are an audit
inventory, **not automatically launched**. Resource expansion and measured
pilot wall times must be reviewed before deciding the full-sweep budget.

The bounded pilot selects one ordinary S1 recipe per legal FQ/SF/Full route
solely for mechanism coverage: 8 points, 20 arms, 5 TC modules for these
headers. It covers both channel layouts and M16/128/5120 boundaries; this
does not establish the optimal config or all-M correctness/performance.
It calls the linked native GPU routing helper, poisons CPU-prepared maps,
validates GPU counts/permutation/source indices, and has the actual TC/provider
consume those GPU maps. Active-expert dequant, SF expansion, GPU preparation,
input, GEMM and finish are individually profiled by Asys. SF uses its own
metadata cold ring. Full expands only selected experts, retaining the E-axis
allocation/strides. JIT, provider preparation and graph upload are untimed.
Kernel capture reuses the native adapter's Runtime/Driver node reader; Driver
function handles are never cast to Runtime host stubs. Non-kernel nodes remain
an explicit failure, not an excuse to omit a component from the measured sum.

The shipping Full-BF16 grouped layout accepts 1024..32768 **routed rows**.
For top-k8 that means token M128..4096. M16/32/64 and M5120 are explicitly
unavailable for that route; this is checked by compiling the real C++ layout
validator, not by assuming token M equals GEMM M. Do not widen an ABI guard
or silently fall back to FQ to make the report green. TC/SF remain separate
candidate routes. Expanding Full's range needs its own production/provider
capability and numerical validation.

The pilot is not native large-M AB or production admission. The per-call
error-reset memory API cost, full screen/confirmation, native large-M adapter
and model AB remain pending even if it passes. Provider graph incompatibility
is a device failure with its original log, never permission to omit nodes or
substitute event times. `review-lite-*.tar.xz` includes compact correctness,
component times, failure logs and provenance; raw reports stay on Box.

## Large-M continuation after the passed pilot

The uploaded `kpack-moe-prefill.0blqj2En` pilot passed all 8 points / 20 arms.
That admits the GPU-map, BF16 general TC and provider capture mechanisms for
the tested recipes, not an optimal configuration or native large-M parity.
`tools/run_moe_prefill_sweep.py` now performs the bounded screen/confirmation.
Use the checkout containing this entrypoint; it is not in the earlier pilot
revision `84839327`. No production policy or kernel is changed by this runner.

The two requested Qwen checkpoints give four local weight families:

| Model placement | qtype | local N/K | activation channels |
|---|---|---|---|
| 35B-A3B TP1 | Q4_K | 512/2048 | 1 |
| 35B-A3B TP1 | Q5_K | 2048/512 | 8 |
| 122B-A10B TP2 | Q4_K | 512/3072 | 1 |
| 122B-A10B TP2 | Q5_K | 3072/512 | 8 |

Each has E256/top8, ten token-M values (16,32,64,128,256,512,1024,2048,4096,5120)
and five synthetic router profiles: **200 points**. TP here means actual local
expert shapes, not a multi-GPU collective/model timing. Headers and the real
CPU split function are checked again on Box. DeepSeek MXFP4 and fused
gate/up/SwiGLU are still explicitly deferred.

The blind shortlist uses donor strata, delivery peers and a narrow-row tile
floor: large total token M is not large M per expert. It excludes the target
N/K family's observations and does not read historical times. Historical
recipes are revealed only to audit recall; a gap stops the plan, rather than
appending that recipe. The current public selector is a separate control.
For the inspected headers this reduces 20,810 audit seeds / 99 TC modules to
5,790 seeds / 53 modules. These are **not** final launch counts. Real resource
queries add a logarithmic persistent-grid ladder and remove duplicate launches;
Split-K proposals with two resident waves already available without splitting
are deferred. Exact current controls are never pruned. Historical **actual
launches**, not merely parent tuples, must remain in the rule pool.

Screening is 4 rounds x 3 samples; confirmation is 6 x 5. Every graph times
one kernel. At most four TC configurations share a fixture/process; arms
alternate order, each retains its own prepared handles/allocations through
the final numerical gate. Full is a singleton batch. Shared GPU route/input/
finish and expansion timings are reused only within the same source, physical
card, phase, semantic workload and cold/resident protocol. SF and Full helpers
are factored separately: selecting a new helper does not require re-timing
every producer. The original control keeps its original helper choice.

Finalists include the original, the best of each route/scheduler stratum,
and candidates within 5% by mean or a full-round mean. More than 32 competitive
finalists is explicit incomplete work, not silent truncation. Mean complete
component sums determine selection; noisy results are retained and labelled,
not vetoed or replaced by their minimum samples.

Before launching workers, the runner tests one ordinary Full screen batch.
Besides the already-passed kernel capture, this checks separately profiled
GPU duration of the public Full API's four-byte error-reset memset. A fixed
primer is excluded, and every measured reset must have successful API and GPU
activity correlation. Missing activity fails preflight; it is not estimated
from host API time or assigned zero cost. The successful batch is reused.
This new reset-accounting seam still needs its first Box execution.

```bash
cd /sim/eec/shared/junfu.qx/quactlize
moe_sweep_root=$(mktemp -d /workspace/kpack-moe-prefill-sweep.XXXXXXXX)
test -n "${moe_sweep_root}" && test -d "${moe_sweep_root}"
python3 -u tools/run_moe_prefill_sweep.py \
    --output "${moe_sweep_root:?}/results" --sdk /usr/local/PPU_SDK \
    --llama-source /sim/eec/shared/junfu.qx/llama.cpp.ncp \
    --llama-build /sim/eec/shared/junfu.qx/llama.cpp.ncp/build-ci-32F-cublas \
    --moe-models qwen35-35b-a3b,qwen35-122b-a10b \
    --devices 3,4,5,6,7 --jobs 8 --l2-bytes 67108864 \
    --budget-hours 8 --confirm-idle
```

Choose only idle physical cards. Compilation is shared once; entire points
(including their controls and finalists) stay on one card. Assignment balances
estimated validation/output volume rather than equal point counts. CU clock
lock requests 1700 MHz; observed throttling is advisory, not a numeric failure.
The eight-hour budget is a **soft per-worker time slice**, not a promise that
all 200 points finish. It stops between batches, never kills an active profile.
Use batch wall times, including independent correctness and Asys overhead,
to judge duration; raw GPU microseconds are not a completion estimate.

`MOE_PREFILL_BATCH_START/DONE` shows point, phase, completed arms and batch
seconds. `MOE_PREFILL_POINT_DONE` shows control/winner means and point count.
Logs are under `results/workers/*/gpu-*.log`. An infrastructure failure is
saved; independent points continue in fresh processes. An incomplete point
does not receive a partial winner. Rerun the **same checkout and directory**:

```bash
python3 -u tools/run_moe_prefill_sweep.py \
    --resume --output /workspace/kpack-moe-prefill-sweep.REPLACE/results \
    --sdk /usr/local/PPU_SDK \
    --llama-source /sim/eec/shared/junfu.qx/llama.cpp.ncp \
    --llama-build /sim/eec/shared/junfu.qx/llama.cpp.ncp/build-ci-32F-cublas \
    --devices 3,4,5,6,7 --jobs 8 --l2-bytes 67108864 \
    --budget-hours 8 --confirm-idle
```

No dirty-checkout gate is reinstated. The direct runner records the HEAD tag
and actual source/harness/DSO/SDK/device hashes, and rejects a changed source
or binary during resume. Do not pull or rebuild dependencies in an active
run. Successful point and batch receipts retain raw evidence hashes; retries
reuse them, not old historical timings or a different card's measurements.

Only upload the printed `results/review-lite-*.tar.xz`: it contains precision,
raw duration samples, recipe/helper bindings, provenance, status and errors.
Raw `.asysrep`, `.sqlite`, compiled modules and caches stay on Box. To repack:

```bash
python3 tools/pack_simt_split_review.py /workspace/kpack-moe-prefill-sweep.REPLACE/results
```

Full remains unavailable outside routed rows1024..32768 (top8 token M128..4096).
All-M native comparison and public runtime/model validation remain separate
gates after this fresh configuration sweep; COMPLETE here does not claim
native no-regression, fused-chain performance or model TPOT.
