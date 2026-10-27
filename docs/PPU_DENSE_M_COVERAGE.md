# Dense M coverage experiment

## Targeted prefill/native gate: Qwen35 and Qwen32 M5120

`tools/run_dense_model_native_gate.py` closes a different coverage gap from
the config sweeps below. It reads the actual GGUF headers and tests one
current production exact/bucket recipe against native at each requested point:

| Model | M | Shapes in the previously uploaded headers |
| --- | --- | --- |
| Qwen3.5-35B-A3B Q4_K_M | 512,1024,2048,4096,5120 | 5 Q8 dense families, 25 points |
| Qwen3-32B Q4_K_M | 5120 | 7 Q4/Q6 dense families, 7 points |

The Box re-derives these counts from its files. The current inventory totals
32 points, 128 Asys captures (native/Qz/Qz/native), and three TC modules. This
is **not** a config sweep. Shared experts' dense weights are included; routed
expert MUL_MAT_ID, floating/native-only matrices and non-layer/output tensors
are explicitly listed in `plan.json` exclusions. The output head may execute
at M1 in a prefill graph and is not silently charged as M5120 here.

Operands are F16, endpoints F32, accumulation FP32. Use the existing
`build-ci-32F-cublas` reference. The narrow test adapter must match the linked
backend on changed inputs; no caller rebuild, source-mtime gate or installed
Quactlize library replacement occurs. This measures the frozen repository's
selected production stage implementations, **not** proof that a model loaded
the same installed DSO or that its default native GEMM accumulates in FP32.
Do not relabel the result as parity with default F16 accumulation.

All component durations come from Asys GPU activities, with ordinary calls,
no cuBLAS CUDA Graph, first-ring/warmup exclusion, and same-card paired arms.
Input conversion, dequant/scale expansion, main, every reducer and output
conversion are individually recorded and summed. Compare arithmetic means,
not minima; one slower point remains REGRESSION even if the weighted model
sum improves. Missing/failed points are INCOMPLETE, not zero-time passes.

On an otherwise idle card (or replace `--gpus 7` with idle ordinals):

```bash
(
  set -euo pipefail
  cd /sim/eec/shared/junfu.qx/quactlize
  git pull --ff-only origin develop
  python3 tools/run_dense_model_native_gate.py \
    --llama-source /sim/eec/shared/junfu.qx/llama.cpp.ncp \
    --llama-build /sim/eec/shared/junfu.qx/llama.cpp.ncp/build-ci-32F-cublas \
    --sdk /usr/local/PPU_SDK --gpus 7 --jobs 8 \
    --l2-bytes 67108864 --confirm-idle
)
```

Defaults use the exact two `bench_model_zoo` GGUF files already named in the
model A/B commands. Override `--qwen35-model` and `--qwen32-model` with exact
files if needed. `--plan-only` needs neither SDK nor llama build; it still
checks the real C++ selector. Resume with the same flags plus the printed
`--output ... --resume`. Completed paired measurements, including regressions,
are retained; no resampling until a pass. GPU assignment, source, models,
precision, build and SDK must remain unchanged.

This archive-based entry does not reject a dirty checkout. Workers still
execute the committed HEAD and pinned actlize archives; uncommitted kernel
edits are not included. Commit/tree, payload and resume identity checks remain
in place, so ignored checkout edits are never relabelled as measured HEAD code.

Upload only the printed `review-*.json.xz`. It includes the complete plan,
raw component samples, numerical proofs, commit/DSO/device identities and
errors, without binaries, weights or Asys blobs. On Box `comparison.csv`,
`components.csv`, `coverage.csv` and `model-summary.csv` expose the per-point,
per-kernel and model/tensor mappings. Model sums assume one call per layer
weight and are explicitly **not** whole-network prefill predictions; compare
actual model graph counts and timings separately. Return codes: 0 PASS,
2 REGRESSION, 1 INCOMPLETE. No selectors or kernels are modified by the gate.

This records a bounded experiment and its subsequent exact-only admission. The inventory
is the 45 `(qtype,N,K)` families in `policies/dense_prefill_v1.json`, including
its recorded TP-local dense shapes. It does not exercise tensor-parallel
communication or add MoE measurements.

Current small-M admission at `28180f7f` covers all 360 keys exactly: 181 TC,
176 generic SIMT and 3 specialized Q4 SIMT. The six gaps mentioned in the
historical sections below are now closed. This is still not a fresh native
no-regression certificate; use the Asys gate linked below.

## Intermediate-M bucket observations

Run `plan_dense_m_span.py --m 16 32 64 128 256`, then the existing
`run_prefill_cohort_multi.py --finalist-plan ... --no-native --confirm-idle`.
This is 225 points from the same 45 real families; not a dtype Cartesian
product. M16/32/64 retain legal incumbent and nearby controls, light FQ tile
strata, SF geometry, persistent and legal Split-K proposals. M128/256 also
include Full-F16. The existing SIMT M<=8 and Full M>=128 contracts are not
extended. Unsupported donor recipes are explicitly listed, not called losses.
Default large-M plans are unchanged. At most 48 configs per point and 16 per
capture; the current five-M plan has 5897 cases and 59 reusable TC modules
before resource filtering. These are bounded candidates, not global optima.

Both forward/reverse captures remain; every helper/main/reducer is separately
measured by Asys. Frozen commit/tree and actlize identity accompany the review.
This collects bucket-training observations; no table or bucket is updated
until the returned evidence is reviewed. No native timing is claimed.

## Scope and controls

Large-M closure (2026-10-26): all 405 M-span points are complete and admitted
as exact mean-ranked selections: 259 Full-F16 / 146 SF / 0 FQ. This replaces
the old 225 dense prefill exact rows, retaining them as historical evidence,
and extends exact selection to M8192. The single +0.551429% Full-F16 versus
old Full-BF16 control is user-accepted. No large-M rerun is needed for that
exception. Buckets, small-M and MoE are unchanged; native/model A/B was not
part of this cohort. Details: `DENSE_PREFILL_ROUTING.md`. The commands and
counts below describe the frozen experiments, not a request to rerun them.

The returned 243-point small-M gap run is now recorded in
`policies/kpack_dense_decode_v1.json`. Its 237 stable keys are generated into
the production exact catalog: 160 replacements and 77 measured retains.
Six unstable points keep their original paths. The existing 1852 exact rows
and 637 bucket entries are unchanged; the total exact catalog is now 2089.
Within this document's 45 dense families, 354/360 M1-8 keys are exact, with
six pending. The old counts below describe the frozen experiment, not work
that must be rerun after this admission.

Q8 admission also generates 36 additional exact body bindings. A geometry
tuple without its measured body is not an equivalent selection. Regenerate
and verify with `tools/fit_dense_m_selection.py --archive <review-lite.tar.xz>
--check`, `tools/generate_decode_selection.py --check`, and
`tools/generate_q8_refresh.py --check`. The library must be rebuilt to use the
new header and bindings; an old installed DSO is not updated by changing JSON.
This is component evidence against the current Quactlize control, not a
fresh native comparison or a model TPOT result. No large-M/MoE or table-miss
policy is changed by this exact-only admission.

| Run | Scope | Planned work |
| --- | --- | --- |
| `dense-m-gaps` | M1–8, missing exact keys | 243 points, 5064 arms, at most 27/point; 29 TC and 28 vector modules |
| `plan_dense_m_span.py` + `--no-native` | M128,256,512,1024,2048,3072,4096,5120,8192 | 405 points, 9412 configs; 55 TC modules; 1632 Asys captures before resource filtering |

The small-M denominator is 360 keys: 117 have existing exact selections and
are retained, not relabelled as newly measured. A bucket prediction does not
count as an exact observation. The new run compares the complete current
host-selected control against bounded SIMT/Q4/TC candidates, including
Split-K and its reducer. It does not use Full dequant and is **not a fresh
native llama.cpp comparison**. Historical recipes propose controls/candidates;
their old microsecond measurements are not imported into this run.

Three Q4 M1 controls still use legacy FQ with flat input/output casts. Those
unchanged shipping casts are compiled with the control and individually
timed. They are not replaced by typed F32 endpoints or improved helpers.
An SDK-backed host test compares the offline query with the real provider's
selection for all 360 keys; this is not device numerical validation.

The large-M run retains the old selector, current whole-call selection where
available, nearby-M winners and nearest per-route champions as required
controls. Separate geometric rules propose ordinary/persistent SF and
underfilled Split-K, including large tiles and distinct pipeline geometries.
At most two K/pipeline alternatives per M/N tile are proposed; this is a
bounded search, not a proof of a globally optimal configuration. Q3 gets two
large SF tile proposals absent from its old candidate inventory. Full-F16
helpers are included, with a functional baseline. FQ is a **control only**,
not a new prefill fallback rule.

Persistent grids are resolved from the actual compiled occupancy and device
CU count. An unsupported optional candidate remains a resource rejection in
the compact review, not a measured loser. Missing/unavailable required
controls fail rather than silently changing the comparison.

The recommended M-span run uses `--no-native`: compare Quactlize configurations
with the current/old selector without building or loading a llama adapter.
No llama source, build directory or model file is required. Same-N/K/format
dequant comparisons need not be repeated for each M. This run still records
each current helper/main/reducer activity; it does not substitute historical
component times. Native time/delta remain null and native no-regression is
not claimed. The base plan retains optional native inventory; the execution
plan records its explicit omission and the reduced capture count.

If explicitly enabled (omit `--no-native`), native large-M uses the existing
llama carrier: F32 storage, F16 operands and
verified FP32 cuBLAS accumulation. Candidate F16 routes use the same precision
contract. An old Full-BF16 control stays separately labelled. Each run checks
numerics before timing; first warmup/JIT/weight-ring passes are excluded.
Asys provides individual helper, main and reducer durations and their sum;
there is no cuBLAS CUDA Graph capture. Forward/reverse captures are both kept,
with at most 16 configs in one profiler process so an interrupted late config
does not discard a whole shape's preceding batches.

## Box commands

Use idle ZW810 cards. The sweep commands below use cards 5,6,7,8. Do not run both tasks on
the same cards concurrently. Both request 1700 MHz; subsequent TDP-related
frequency deviations are recorded, not automatically marked as failures.
Quactlize-only mode has no llama build dependency. Optional native comparison
still verifies the existing FP32-accumulator build's bindings/precision.

Small-M gaps:

```bash
(
    set -e
    cd /sim/eec/shared/junfu.qx/quactlize
    git pull --ff-only origin develop
    unset RUN RESUME_RUN CLOCK_RETRY_RUN PLAN_ONLY BUILD_ONLY
    PPU_SDK=/usr/local/PPU_SDK \
    GPUS=5,6,7,8 PHASE=dense-m-gaps JOBS=32 \
    L2_BYTES=67108864 IDLE_CONFIRMED=1 \
    bash tools/run_decode_refresh_box.sh
)
```

Large-M SF/Full config selection, after the cards are free:

```bash
(
    set -e
    cd /sim/eec/shared/junfu.qx/quactlize
    git pull --ff-only origin develop
    M_RUN=$(mktemp -d /workspace/kpack-dense-m.XXXXXXXX)
    test -d "${M_RUN:?}"
    python3 tools/plan_dense_m_span.py --output "${M_RUN:?}/plan"
    echo "M_RUN=${M_RUN}"
    PPU_SDK=/usr/local/PPU_SDK \
    L2_BYTES=67108864 \
    python3 tools/run_prefill_cohort_multi.py \
        --finalist-plan "${M_RUN:?}/plan/plan.json" \
        --output "${M_RUN:?}/run" --gpus 5,6,7,8 \
        --jobs 8 --configs-per-batch 16 --no-native --confirm-idle
)
```

The planners print exact counts before device timing. One family and all its
controls stay on the same physical card. Existing workers share a TC cache;
unchanged modules are reused. Runtime estimates use observed wall time rather
than kernel microseconds; no fixed completion duration is promised.

Small-M prints its exact `RESUME_RUN=...` command. For large M retain the
printed `M_RUN` directory and repeat the same coordinator command with
`--resume`; keep the same plan, GPU list, jobs, batch limit and `--no-native`
mode. An old run that failed building the native adapter needs a new output
directory: its frozen source revision will otherwise repeat that failure.
The old `run/cohort.json` can be used as the new `--finalist-plan`; do not
delete its failed logs. Do not update
or delete the frozen worker sources/caches. Completed valid batches are not
rerun. Upload only the printed small-M `review-lite-*.tar.xz` and large-M
`run/review-*.json.xz`; raw Asys/SQLite/binaries remain on Box.

## Historical event-based selected small-M comparison

**For current acceptance use `BASELINE=native-ab` in
[PPU_SMALLM_NATIVE_ASYS.md](PPU_SMALLM_NATIVE_ASYS.md), not this older entry.**
The old event/median/stability policy below is retained as historical context;
it is not interchangeable with the current Asys mean-based acceptance rule.

`run_dense_smallm_ab_box.sh` is a different task from `dense-m-gaps`:
45 existing dense `(q,N,K)` families crossed with **every M1..8 = 360 points**,
including the six unresolved exact keys' current fallback. It queries the
current C++ selector and measures **one selected recipe per point**, not another
candidate sweep or a comparison against old historical microseconds.

Use idle device ordinals **0,1,2,3** (zero based), separate from the large-M
workers. All four devices are probed before shared compilation/timing. The
wrapper freezes source and actlize locally; it never fetches or rebuilds llama.

```bash
(
    set -euo pipefail
    cd /sim/eec/shared/junfu.qx/quactlize
    git pull --ff-only origin develop
    unset RESUME_RUN PLAN_ONLY
    GPUS=0,1,2,3 JOBS=8 IDLE_CONFIRMED=1 \
    PPU_SDK=/usr/local/PPU_SDK L2_BYTES=67108864 \
    LLAMA_CI_DIR=/sim/eec/shared/junfu.qx/llama.cpp.ncp \
    LLAMA_CI_BUILD_DIR=/sim/eec/shared/junfu.qx/llama.cpp.ncp/build-ci \
    bash tools/run_dense_smallm_ab_box.sh
)
```

Each whole family's measurement order is native / Quactlize / Quactlize /
native. Every pass has six rounds of five samples per stage. Raw GGUF hashes,
F32 activation fixture (repeat 1), physical card and SDK are matched. Producer
addresses rotate over at least 2.25 L2; helpers/reducers are independently
measured, and their actual costs are included in the sum. No estimated launch
penalty. Native cuBLAS, if selected, uses ordinary provider events, never graph
capture. Native activation quantization remains present and is charged; the
report labels it separately from Quactlize F16-compute/F32-accumulate semantics.

`run/comparison.csv` gives native/current sums and deltas. Point records contain
the selected recipe, component names/configurations, all samples, numerical
and negative proofs, and source/device authority. `COMPLETE` means measurement
coverage, **not** no-regression: check `component_no_regression`, `REGRESSION`
and `INCONCLUSIVE`. There is no allowed slowdown percentage; 5% is only the
stability check, not a regression allowance. Actual clock deviations after the
1700MHz request are advisory. Nothing is silently rerouted or promoted.

Upload the printed `run/review-lite-*.tar.xz`, not the entire run directory.
The compact archive excludes fixture weights, binaries and JIT objects. Once
`run/run.json` exists, resume with the printed `SMALLM_RUN` as `RESUME_RUN` and
the same card list/SDK/caller build. Valid completed comparisons, including
regressions, are not retimed; an incomplete point gets all four passes in a
fresh attempt, preserving its earlier raw files. Do not resume by editing the
frozen source or copying results to another GPU. Setup failures before the
run receipt require a new run after fixing their cause.

This entry is host-tested; device correctness/performance admission remains
pending the Box run. Large-M remains `--no-native`; this experiment adds no
MoE, TP communication or model TPOT claims.

## What this does not close

- M9–127: the current native carrier intentionally requires a verified
  Full-F16/cuBLAS path. Native MMQ/Q8-activation paths need their own labelled
  precision contract; they cannot silently stand in for that comparator.
- Fresh native small-M regression comparisons for the admitted final
  selected calls (the 1700-MHz winner import itself is complete).
- Bucket holdouts and unsampled M/N/K shapes, additional MoE families, and
  model integration/TPOT. Training-point minima are not holdout evidence.
- Production M5120 routing is unchanged by these scripts. The old FQ miss
  path is measured as a control; its replacement requires these results and
  a separately tested selector update. No unmeasured all-M bucket is installed.

Local checks include host query/provider agreement, all six shard plans,
resource/receipt parsing and actual HGCC object compilation of the two legacy
Q4 cast controls and the two new Q3 SF geometries. Compile-only and host
checks do not prove device correctness, occupancy or performance.
