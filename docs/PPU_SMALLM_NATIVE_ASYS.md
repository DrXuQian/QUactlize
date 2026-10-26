# Small-M native baseline

The default cohort is Q5_K/Q6_K, N1024 K5120, each M1–8 (16 points).
This is a native llama.cpp baseline, not a configuration sweep, a paired
no-regression certificate, or whole-model TPOT. No caller source or build is
changed. A small test adapter records the exact native dispatcher and links
the kernels from the supplied existing build. No source-mtime gate.

```bash
(
    set -e
    cd /sim/eec/shared/junfu.qx/quactlize
    git pull --ff-only origin develop
    unset RUN RESUME_RUN PLAN_ONLY COHORT
    BASELINE=native QTYPES=13,14 N=1024 K=5120 GPUS=0,1,2,3 \
    LLAMA_CI_DIR=/sim/eec/shared/junfu.qx/llama.cpp.ncp \
    LLAMA_CI_BUILD_DIR=/sim/eec/shared/junfu.qx/llama.cpp.ncp/build-ci-32F-cublas \
    PPU_SDK=/usr/local/PPU_SDK L2_BYTES=67108864 IDLE_CONFIRMED=1 \
    bash tools/run_smallm_regressions_asys_box.sh
)
```

Only run this when those cards are idle. Requests1700MHz; actual frequency is
diagnostic, not a failure gate. A committed source archive is frozen locally;
no nested submodule/network fetch. Only the two test adapter/carrier binaries
are built; no Quactlize compute kernels are compiled or executed.

Each producer rotates through at least2.25x verified L2 of raw weights. Helpers
use resident scratch. Each timer contains one kernel type, at least16calls
and complete ring traversals. Graph creation/upload, first use, independent
oracle, D2H and JIT are outside the profiler range. Six alternating stage-order
rounds, five samples each; timing is each Asys GPU record's end-start, not a
CUDA-event interval. Do not reject adjacent timestamp overlap or replace raw
durations with clipped/union intervals. Positive durations, exact symbols,
launch geometry and complete sample coverage are still checked.

The native activation path is not changed: `quantize_q8_1` is counted as its
own helper when selected, followed by MMVQ. This is **not** identical arithmetic
to Quactlize F16 activations / FP32 affine accumulation. The build name
`32F-cublas` does not prove that a small-M dispatch uses cuBLAS or any particular
accumulator. Actual kernels/launches and the native Q8_1 oracle are recorded.
A cuBLAS/provider selection is rejected by this bounded kernel-only baseline;
providers are never captured into its isolated graphs.

Upload only the printed `results/review-lite-*.tar.xz`. It includes numerical
checks, raw per-kernel durations, component/total means, failures and
source/DSO/SDK/device hashes. Full `.asysrep` and `.sqlite` stay on Box under
`results/gpu-*/attempts/*/`. Resume with the printed `BASELINE=native RESUME_RUN=...`
and the same cohort/build/source/device options. Completed PASS points are
retained and failures run in fresh processes. Do not rebuild llama midway.

## Existing Q6 M4 / Q5 M2 offline reparse

This needs the old run's SQLite on Box, no GPU or SDK. It preserves old FAIL
and PASS receipts, writes a separate `results/reparse-*/` review, and packages
that review alongside original evidence. Measurement commit stays the old
one; the new parser's hashes are recorded separately.

```bash
python3 tools/run_smallm_regressions_asys.py \
    --output /workspace/kpack-smallm-asys.rG2gJdVV/results --reparse-only
```

Use `/workspace/kpack-smallm-asys.OttZwNL1/results` for the older Q5 run.
Only timestamp-adjacency failures are eligible for this migration; other
failures stay unadmitted. Do not use GPU resume to change an old parser/source.
