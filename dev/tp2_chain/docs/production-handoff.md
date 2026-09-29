# Measured paired recipes: production-entry and model gate

Selection is generated from `policies/kpack_gate_up_selection_v2.json` by
`tools/generate_decode_selection.py`. `recipes.inc` is its runtime view, not
a second hand-maintained heuristic. The request distinguishes local logical
N/K, precision, indexed/dense operation, token count, row order and down
consumer. A recipe identifies the complete body as well as launch geometry.

The caller forwards/caches a recipe; it does not implement the shape policy.
The explicit launcher validates that recipe and never silently replaces its
reader. Existing v1 APIs retain their geometry and historical reader behavior.
A miss keeps the existing admitted unfused chain. Paired-N4 and canonical
bytes are unchanged; allocation/repack stay outside graph capture and decode.

## What changes

| Request | Selection | Evidence / gate |
| --- | --- | --- |
| Q4 N512/K3072, M1, E256/top8, BF16, original rows, Q5 SIMT down | paired16 C4/W4/P4, Changes3 | Complete chain -8.11%; new public entry/model gate pending |
| Q4 N512/K2048 | Existing paired winner | No body change; same-device regression control |
| Existing Q8 shared expert | Existing tile16/generic winner | No body change; model trace checks exact recipe |
| TP2 M2..8 outside admitted rows | Existing unfused chain | Candidate capability tested explicitly, not automatically selected |
| Ordinary GEMV/GEMM | Existing exact and bucket choices | 1,852 exact / 637 buckets byte-identical |

Dense F32 storage/F16 compute and MoE BF16 compute are retained. The new
body's native instructions/resources exactly match the measured candidate;
that is a local code-generation proof, not a substitute for device numerics.

## Run

Use `dev/tp2-paired-chain`. The runner fetches its pinned runtime from the
Quactlize fork and its pinned caller from the llama.cpp fork. It does not
switch the shared caller checkout. Caller compilation uses `.aoneci` on box;
the large sweep/TC bundle is not rebuilt. Only new LFS objects transfer.

```bash
(
    set -e
    cd /sim/eec/shared/junfu.qx/quactlize
    git fetch origin dev/tp2-paired-chain
    GIT_LFS_SKIP_SMUDGE=1 git switch dev/tp2-paired-chain
    GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only origin dev/tp2-paired-chain
    git submodule update --init third_party/actlize

    PPU_SDK=/sim/eec/shared/junfu.qx/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
    RESULT_ROOT=/sim/eec/shared/junfu.qx/kpack-newbox.QYl8F9 \
    QUACTLIZE_PPU_BUNDLE=/sim/eec/shared/junfu.qx/kpack-newbox.QYl8F9/kpack-q4-smallk.5DyMwt/update/bundle \
    LLAMA_CI_DIR=/sim/eec/shared/junfu.qx/llama.cpp \
    NCP_LIB_DIR=/sim/eec/shared/junfu.qx/ncp_flash_lib \
    PAIRED_DEVICES=6,7 JOBS=192 \
    bash tools/run_paired_selection_box.sh
)
```

Keep devices 6/7 idle. The SDK path is the native root, not `CUDA_SDK`.
The compatibility six-library bundle above is an existing box artifact;
override its path if moved. No large old bundle is implicitly downloaded.
Model paths are resolved before timing. Local GSM8K Parquet is used only as
a deterministic likelihood corpus, not an answer-accuracy benchmark.

The default makes an isolated pinned caller build. To reuse a configured
caller build, first update the caller to the exact commit in
`tools/kpack_paired_selection_artifact.json`, then supply `LLAMA_CI_BUILD_DIR`.
`NCP_CI_DIR` can independently reuse a matching NCP checkout/build. The joint
builder validates source/compiler/configuration before resuming objects.

## Coverage and output

1. Two independent component jobs on devices 6/7. Each checks both full chains
   with M1..8, routing/ID changes, guards and BF16-range inputs. M1 timing uses
   six alternating rounds of 15 samples, rotating weights beyond verified
   64 MiB L2. First graph launch is excluded. No isolated-sum timing proxy.
2. Qwen3.5-122B TP2 on 6/7 and Qwen3.5-35B on 6, sequentially. Each runs
   reference/native likelihood checks, a warmed ABBA PP2048/TG128 benchmark
   (one warmup per process; two recorded rows), then private warmed Asys.
3. Asys must show the exact selected paired recipe, not merely any grouped
   GEMM. First JIT, graph upload, model loading and cache writing are excluded
   from inference timing. Component speedup is not a guaranteed model TPOT.

`PAIRED_PHASES=components` runs only step 1. `VERIFY_ONLY=1` checks the package
and input paths without a GPU benchmark. Every stage prints its log location
and elapsed time. No full-run duration is inferred from kernel microseconds.

Resume with `PAIRED_SELECTION_RUN=<printed run directory>` and the same
settings. Completed stages are reused only under matching identities. Failed
component points resume independently; failed model stages get fresh logs,
and an independent model continues. Do not change code/binaries in place
between attempts; start a fresh run after an identity change.

Return the printed `*.results.tgz`: receipts, correctness, samples, summary
and trace proofs are included. Model files, caller builds and `.asysrep`/SQLite
are excluded. Full Asys reports remain under
`<run>/results/*-trace-*/<model>/{reference,native}/proof.asysrep`.
No final performance/device admission is claimed until the returned results
are reviewed. `device_admission=PENDING_RESULT_REVIEW` is intentional.
