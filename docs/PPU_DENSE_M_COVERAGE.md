# Dense M coverage experiment

This is a bounded experiment, not a production selector update. The inventory
is the 45 `(qtype,N,K)` families in `policies/dense_prefill_v1.json`, including
its recorded TP-local dense shapes. It does not exercise tensor-parallel
communication or add MoE measurements.

## Scope and controls

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

Use idle ZW810 cards. The commands below use cards 5,6,7,8. Do not run both tasks on
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

## What this does not close

- M9–127: the current native carrier intentionally requires a verified
  Full-F16/cuBLAS path. Native MMQ/Q8-activation paths need their own labelled
  precision contract; they cannot silently stand in for that comparator.
- Import and adjudication of the prior 1700-MHz small-M winners, then fresh
  native small-M regression comparisons for the final selected calls.
- Bucket holdouts and unsampled M/N/K shapes, additional MoE families, and
  model integration/TPOT. Training-point minima are not holdout evidence.
- Production M5120 routing is unchanged by these scripts. The old FQ miss
  path is measured as a control; its replacement requires these results and
  a separately tested selector update. No unmeasured all-M bucket is installed.

Local checks include host query/provider agreement, all six shard plans,
resource/receipt parsing and actual HGCC object compilation of the two legacy
Q4 cast controls and the two new Q3 SF geometries. Compile-only and host
checks do not prove device correctness, occupancy or performance.
