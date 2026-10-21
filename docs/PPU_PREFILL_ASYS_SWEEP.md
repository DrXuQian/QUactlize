# Qwen3-32B PP2048 config screen

`tools/run_prefill_model_asys_box.sh` changes no kernel, selector, or llama
source/binary. It reads the real GGUF header and covers the seven quantized
dense weight families observed at M2048 (445 calls per prefill). Token lookup,
the output head, and last-layer FFN M1 operations are explicitly excluded.

The finite pool contains all rule proposals, historical controls, current
source-selector configs and the configs from the reviewed model trace. There
are 54 unique TC compile templates and 752 resource queries before actual
Split-K/grid deduplication. This is not the Cartesian product of every legal
tile/warp/pipeline. It does not promise the global optimum or model admission.
Both FQ and SF remain measurable candidates, not mandatory fallbacks. Full
BF16 dequant plus cuBLAS is also measured with the legal expansion configs.
SF metadata expansion stays at its model-control config (or Q4=4/Q6=0), so
this is a GEMM-config screen, not a Cartesian helper-times-producer search.

## Run

Use an idle ZW810; an explicit 1700MHz CU clock lock is requested. Actual
clocks are recorded but throttling does not fail the run or rescale results.

```bash
cd /sim/eec/shared/junfu.qx/quactlize
git pull --ff-only origin develop
PREVIOUS_RUN=/workspace/kpack-prefill-point.T71HvBXC \
GPU=1 IDLE_CONFIRMED=1 JOBS=8 \
PPU_SDK=/usr/local/PPU_SDK \
LLAMA_CI_DIR=/sim/eec/shared/junfu.qx/llama.cpp.ncp \
LLAMA_CI_BUILD_DIR=/sim/eec/shared/junfu.qx/llama.cpp.ncp/build-ci-32F-cublas \
bash tools/run_prefill_model_asys_box.sh
```

`PREVIOUS_RUN` reuses the successful point's unchanged native base adapter,
helper and dequant libraries after source/SDK/image checks. New tiny host
profiling wrappers and missing TC templates are built once. If that run no
longer exists, omit the variable; the diagnostic builds its own dependencies,
never llama.cpp. `MODEL` can override the default Qwen3-32B GGUF path, but the
header must still match this exact reviewed cohort. `TC_CACHE` optionally
reuses a compiler content-addressed cache. `PLAN_ONLY=1` performs no GPU work.

## Measurement and failure handling

Each config passes independent GGUF/F64-reference, changed-input, zero-input,
omitted-producer and buffer-guard checks. The native arm also compares the
extracted sequence to the linked backend and verifies the actual GemmEx call
uses F16 A/B/C with `CUBLAS_COMPUTE_32F` (68). The build directory name alone
is not precision evidence. Full-BF16 has different rounding and provider
settings; those differences remain explicit in its receipt.

Asys uses profiler-API capture ranges, merging a small batch of configs into
one report. CPU fixture construction, correctness, compilation and warmup are
outside each range. Ordinary C++ calls rotate weights; no Python between
stages, CUDA Graph, or per-stage events. The first captured ring is discarded;
three further rings are retained. Startup validates native and a two-config
SF/Full capture before continuing a family's larger batches.

The parser reads GPU kernel/memset activity `end-start`, retains all provider
subkernels, and reports input, expansion, main, reducer, output and reset
separately. It also records GPU activity sum, first-to-last span, overlap and
gaps. Unknown provider subkernels are retained, never dropped. Missing calls,
unexpected activity, changed post-warmup launch inventory, or numeric failure
stops admission. There is no fallback to CPU/API/event times.

Native rotating calls own per-weight-copy scratch; Quactlize reuses converted
A/C. This is declared, not claimed to reproduce all model cache residency.
The minimum GPU-sum config is a **screen winner**, not a production update:
paired finalist confirmation and model trace comparison follow separately.

## Resume and upload

Use the printed run directory on the **same physical GPU and SDK**:

```bash
GPU=1 IDLE_CONFIRMED=1 \
RESUME_RUN=/workspace/qwen32-prefill-asys.EXACT_RUN \
bash tools/run_prefill_model_asys_box.sh
```

The original frozen source and payload/cache paths are reused. Valid completed
batches are verified and skipped; an interrupted batch is retried into a new
attempt, preserving its old evidence. No 24-hour timeout or automatic process
kill is installed. Do not change `CONFIGS_PER_BATCH` (default16) when resuming.

Upload only the printed `results/review-*.tgz`. It contains correctness,
per-kernel samples/configs and bounded error logs, not weights, binaries or
reports. GUI reports remain at
`results/batches/*/attempt-*/trace.asysrep`; adjacent SQLite allows host-only
parser repair without rerunning the GPU.
