# Dense conversion and compact reducer reuse

IO4 adapts the MIT-licensed contiguous body from llama.cpp's
`ggml/src/ggml-cuda/convert-ppu.cuh` at
`d30de5bbe45fabb23bcc7a6832d84f0046c16f23`. It does not depend on llama headers
or binaries. IO0..3, the measured helper tables, GEMM and dequant implementations
are unchanged. The first experiment left IO4/compact selection disabled. The
reviewed exact-only admission below now enables the confirmed combinations.

## Bounded followup complete; next is public-library/model validation

Review `d3eec044e46f3beff68dc1fc5ce6371faaed7232aa73e673d91fb81f7e1f4128`
(source `2be024c`, ZW810/SDK2.1.1,1700MHz request) passed all15points,
49candidate arms,128captures,24960complete-call samples,1296IO and288reducer
checks. Maximum independent relative-L1 error was0.00011438. Archive SHA256:
`cca53222d242a568e10fd922cd3aca086f5fee490602d381376773909f0b8467`.
No additional geometry sweep is planned.

The existing 225-point catalog now replaces13exact choices and retains2.
All4missing incumbent comparisons are resolved; historical pending records stay
in the evidence alongside the explicit resolution. All147bucket predictions
and their original training costs remain unchanged. A recipe changes only if
both reversed captures beat the current production control.

| Scope | Admission |
|---|---|
| Six S2 outputs with M*N=524288 | compact8; reducer3.68–4.10us ->2.26–2.28us; complete call3.86–6.98% faster |
| Q8 M512/N512/K4096,S4 | compact4; complete call22.00 ->21.110us |
| Q8 N4096/K2048,M128/256/512 | Complete incumbent comparisons now admit the measured SF winners |
| Q8 M2048,(N,K)=(4096,2048),(4096,3072),(8192,1024) | Admit dequant-before-input, retaining same kernels and FP32 accumulation |
| Q8 M128/N256/K3072 and M2048/N6144/K3072 | Retain current Full recipes |

At the15points,12selected calls beat native;3still trail native by0.95–1.48%.
This is not a strict zero-regression or TPOT claim. Ordering can improve one
helper while slowing another: e.g. M2048/N6144/K3072 gets worse overall, so the
order is NOT a global default. Full uses one internal validation/enqueue entry,
not five public stage calls; the public v1 ABI/default order is unchanged.
The20selected compact rows are13width2,1width4 and6width8.

### Final Box round: fresh public package, accuracy, warmed model A/B

Use an idle ZW810 card. Existing caller binaries are reused without changing
their installed libraries. The reference below is the previously validated
FP32-accumulator build; an environment request alone does not prove a caller
honors it. Trace and loaded-image receipts remain necessary.

```bash
(
  set -euo pipefail
  cd /sim/eec/shared/junfu.qx/quactlize
  git pull --ff-only origin develop
  export PPU_SDK=/usr/local/PPU_SDK CUDA_VISIBLE_DEVICES=1
  export PATH="${PPU_SDK}/bin:${PATH}"
  export LD_LIBRARY_PATH="${PPU_SDK}/lib:${PPU_SDK}/targets/x86_64-linux/lib:${PPU_SDK}/CUDA_SDK/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  ppu-smi --id=1 --lock-ppu-clocks=1700,1700 --error-on-warning
  DENSE_RUN=$(mktemp -d /workspace/kpack-dense-final.XXXXXXXX)
  test -d "${DENSE_RUN:?}" && test ! -L "${DENSE_RUN}"
  echo "RUN=${DENSE_RUN}"
  export QUACTLIZE_PPU_CACHE_DIR="${DENSE_RUN}/jit-cache"
  bash build.sh --library --sdk "${PPU_SDK}" \
    --output "${DENSE_RUN}/package" --jobs 8 2>&1 | tee "${DENSE_RUN}/build.log"
  bash run_tests.sh --suite accuracy --sdk "${PPU_SDK}" \
    --package "${DENSE_RUN}/package" --output "${DENSE_RUN}/accuracy"
  python3 tools/run_dense_model_ab.py \
    --llama-dir /sim/eec/shared/junfu.qx/llama.cpp.ncp \
    --reference-build /sim/eec/shared/junfu.qx/llama.cpp.ncp/build-ci-32F-cublas \
    --quactlize-build /sim/eec/shared/junfu.qx/llama.cpp.ncp/build-ci-quactlize-dense-gemm \
    --quactlize-package "${DENSE_RUN}/package" --jit-cache "${DENSE_RUN}/jit-cache" \
    --cublas-accumulator f32 --sdk "${PPU_SDK}" --device 1 --idle-confirmed \
    --pp 2048 --tg 128 --repeats 3 --trace-repeats 2 \
    --output-root "${DENSE_RUN}"
)
```

This runs all3existing single-GPU models: Qwen3-32B,Qwen3.5-35B-A3B,Qwen3.8-27B.
MoE remains native in this caller MR. Accuracy additionally executes the15
followup shapes through public prepare/run on2streams with changing addresses,
finite-output checks, independent FP64 full-K spot dots and full packed-byte
checks. cuBLAS accuracy calls remain ordinary calls. Only ZW810 is required to
select these exact recipes; other devices retain their own fallback domain.

The benchmark discards the first complete PP/TG pass of every process, including
JIT. Unprofiled model latency and profiled diagnostics are separate. Actual
clock deviations are advisory; the successful lock request is not a guarantee
against throttling. Upload the printed `dense-model-ab.*.review.tgz` and the
accuracy summary/logs. The review excludes binaries/JIT/models and raw `.asysrep`;
Asys GUI files remain under `dense-model-ab.*/MODEL/trace/{reference,quactlize}/`.
Do not rerun the historical component commands below.

### BF16 control: existing ppu-dev versus dense-enabled build

For further iteration, keep the optimized caller in build-ci. Rebuild/replace
Quactlize only (with all model/test processes stopped), not llama.cpp, when
its public ABI is unchanged. The current caller install layout requires
bin/libquactlize_ppu.so together with the complete bin/quactlize-runtime/
(private libraries, JIT closure and policy data). bin/quactlize/ holds the
offline gguf_kpack.py test reference; updating that directory alone does not
update inference. Keep it consistent when running the loader tests. Preserve
build-ci-ppu-dev and build-ci-32F-cublas as independent reference builds; the
latter is the explicit FP32-accumulator comparator. Single-operator experiments
stay in the Quactlize repository. Do not replace payloads during a comparison.

To isolate build/integration overhead without any quantized Quactlize compute,
use the original BF16 model. This reuses both installed builds and libraries;
no compile or model conversion. Select an idle ZW810 card before running.

```bash
(
  set -euo pipefail
  cd /sim/eec/shared/junfu.qx/quactlize
  git pull --ff-only origin develop
  GPU=0
  LLAMA_DIR=/sim/eec/shared/junfu.qx/llama.cpp.ncp
  [[ "$GPU" =~ ^[0-9]+$ ]] || exit 1
  ppu-smi --id="$GPU" --lock-ppu-clocks=1700,1700 --error-on-warning
  python3 tools/run_dense_model_ab.py \
    --llama-dir "$LLAMA_DIR" \
    --reference-build "$LLAMA_DIR/build-ci-ppu-dev" \
    --quactlize-build "$LLAMA_DIR/build-ci" \
    --bf16-model /sim/eec/shared/AI_workspace/llm-models/bench_model_zoo/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-BF16-00001-of-00002.gguf \
    --sdk /usr/local/PPU_SDK --device "$GPU" --idle-confirmed \
    --pp 2048 --tg 1 --warmup-passes 2 --repeats 3 --trace-repeats 3 \
    --phases perf,trace --output-root /workspace
)
```

Each process executes five complete2048-token passes. All five are retained
in the logs/JSON and raw trace; the first two are excluded from reported
medians by the fixed protocol. This is not an assertion that the remaining
passes are stable: inspect their individual times and graph/API activity.
TG1 is only the benchmark's minimal continuation, not decode/TPOT evidence.
The reference uses GGML_QUACTLIZE_DENSE=0; build-ci uses1, but neither should
prepare a quantized plan or execute Quactlize kernels for this BF16 inventory.
The entry validates all shards and rejects mixed quantized inputs. It retains
normal native execution and does not force a different cuBLAS accumulator.

Asys files: RUN/bf16-control/trace/{reference,quactlize}/trace.asysrep.
The printed review archive includes SQLite/logs/JSON, not model weights or raw
Asys blobs. Unprofiled performance and profiled diagnostics are separate.

## Initial admission and bounded followup plan (completed)

Review `bd5a4529cbdb2a30b2fd11f918179e81d9e4f382318262cf9da12bfb24495f49`
passed 65 points, 304 candidate arms, 738 captures, 1296 IO checks and 228
reducer checks. The updated catalog has 225 exact points. Of this cohort,
61 points are imported:53 select a new helper combination and8 retain a
remeasured old route. Four existing exact incumbents were absent from the
experiment and remain unchanged, with a pending comparison recorded in the
catalog. No cross-run microsecond comparison substitutes for that control.

Choose the minimum complete call among candidates beating the remeasured old
best in both capture orders. Keep all measured costs, including rejected arms.
All147 old bucket predictions/training remain unchanged; these new helper
choices are exact-only. This is not an unseen-shape or model-TPOT guarantee.
The reserved reducer suffix is reconstructed by both lazy JIT and prewarm;
the actual C++ selector is tested through this source-generation boundary.

Compact2 loses on six larger-output S2 points and is NOT enabled globally.
The next experiment covers15points/49candidatearms, seven TC modules and
128native/candidate confirmations (not another GEMM geometry sweep):

- Six reducer regressions plus two small-output controls: generic vs compact2/4/8,
  with the same GEMM, partials, casts, fixed-S sum and F16 rounding.
- Four Full residuals: unchanged vs dequant-before-input order using the same
  public stage APIs. Allocation, error reset/checked output and precision stay
  unchanged. This diagnoses cache-order effects; it is not a production option.
- Four missing incumbent comparisons, one overlapping a Full residual.

On idle GPUs1,2,3,4,5,7:

```bash
cd /sim/eec/shared/junfu.qx/quactlize
git pull --ff-only origin develop
python3 tools/run_q8_tp_refine.py \
  --phase followup \
  --review /workspace/kpack-q8-tp-refine.mtzrzgb9/timing/review-1792820382106902152.json.xz \
  --llama-source /sim/eec/shared/junfu.qx/llama.cpp.ncp \
  --llama-build /sim/eec/shared/junfu.qx/llama.cpp.ncp/build-ci-32F-cublas \
  --sdk /usr/local/PPU_SDK --gpus 1,2,3,4,5,7 \
  --jobs 8 --configs-per-batch 16 --run --confirm-idle
```

The existing runner requests1700MHz, excludes warmup and collects each helper,
GEMM and reducer from ordinary-call Asys GPU activities. No cuBLAS CUDA Graph
or Python/API timing. The independent reducer gate now tests every proposed
width, including tail/stride/alignment fallback, before full-call timing.
Upload only the printed compact `review-*.tgz`; raw Asys reports stay on Box.
No llama rebuild or model loading is needed for this component experiment.
If new candidates lose, retain the already admitted configuration.

## Contract and local evidence

Use typed restricted pointers, 32-bit eight-element chunk indices and one
chunk/thread, 256 threads/CTA. Host eligibility requires 16-byte aligned
endpoints, contiguous rows, count divisible by eight, and count/8 <= INT_MAX/2
so both `c` and `2*c+1` fit. Otherwise IO4 retains the old IO1 path. No added
shared memory, transpose, clipping, or BF16-through-FP16 conversion.

Output with a non-null producer-error flag uses a checked specialization;
nonzero status still writes canonical FP32 NaNs. A null flag uses the exact
unchecked body; input ignores the flag as before. Buffers remain nonoverlapping
and validated by the existing entry. Streams/addresses are per-call.

Local SDK2.1.1/PPU1.0 native-code comparison, not GPU timing:

| Body | Old IO1 static sites | IO4 static sites | Extracted native control |
|---|---:|---:|---:|
| F32 -> F16 | 92 | 52 | 52; instruction bytes identical |
| F16 -> F32, flag absent | 166 (generic flag branch) | 63 | 63; instruction bytes identical |
| F16 -> F32, flag present | 166 | 75 | Native has no corresponding flag contract |

These are complete static bodies including padding, not dynamically executed
instruction counts or predicted latency. All four unchecked F16/BF16 bodies
match their extracted native controls byte-for-byte; all 24 pre-existing
conversion bodies retain their instruction bytes. All use 32 registers/thread,
zero shared memory and zero local stack with this compiler. Box compiler/device
correctness and timing are separate gates.

The previous same-run Full-F16 example M2048/N6144/K3072 had input/output
casts 32.20025/21.610us versus native24.58025/18.410us. Its dequant was already
faster (31.79025 vs37.75025us), and main GEMM effectively equal. This motivates
cast reuse; it does not certify IO4's speed. SF's custom Split-K reducer is a
separate optimization. Full's provider GEMM and internal reducer are already
cuBLAS and should not be replaced by a copied undocumented reducer entry.

The in-library reuse candidate for SF is:
`PpuMixedInputSplitKParallelCompactReduction` already flattens contiguous
`[S,M,N]` to `[S,M*N]`, uses aligned vector access and compile-timeS2/4/8,
and preserves increasing-S FP32 addition followed by F16 rounding. Grouped
runtime uses it; `dense_splitk_parallel_ppu.cuh` instead instantiates
`PpuMixedInputSplitKParallelM1FastReduction` without `FlattenContiguousRows`,
so M>1 falls back to the generic reducer. Reusing the compact dispatcher is
an isolated experiment, not a new reducer algorithm or a proved speedup.
Do not switch it globally: CTA grain changes, contiguous/aligned/size checks
must still pass, and producer/output layouts must agree. A default-zero
`PreparedMultiformatLauncher` template argument selects a compact width only
when explicitly nonzero (2/4/8 for their admitted exact rows). Generated
native F16 dense parents carry `dense_reducer=N` and an `_rcN` suffix, with their
own source/build identity. Existing parents and bound/dynamic call ABI are
unchanged. Typed decode and grouped modules reject this option.
The option changes only the reducer dispatcher: ordinary/Split-K GEMM, partial
layout, workspace size and increasing-S sum are unchanged. Native-code audits
compare every GPU body/resource against the retained pre-change module, rather
than inferring that from identical source templates.
Local validation for this combined candidate: 773 product-host tests passed;
all seven cohort TC modules and the actual reducer diagnostic carrier compiled
with PPU SDK2.1.1. Every GPU body/resource in those TC modules matches the
retained pre-change module. No local PPU runtime was available; this is not
a device numerical or performance pass.

## Initial 65-point Box check (completed)

Use the latest completed Q8 TP refinement compact review, not the earlier
pre-refinement report. `--phase helpers` keeps the best measured SF and Full-F16
recipe for every point, adds IO4 to each, and remeasures native. The 22 points
whose SF control uses Split-K also get reducer-only and IO4+reducer comparisons.
For the current cohort this is 65 points, four or six candidate arms/point,
304 candidate arms, seven shared TC modules and two opposite-order confirmations
(738 timed arms including native). `--phase io` remains available for casts alone.
No GEMM geometry or DQ sweep and no
historical microseconds inserted into new comparisons. Full sums retain input,
expansion, GEMM, any reducer, output and reset; Asys also records each component.

Each worker first runs `tools/check_dense_io.py` against its actual built
library. The gate checks IO4/IO1 raw equality, all16-bit input words,
rounding boundaries, overflow, infinities, NaNs, strides/tails, vector-unaligned
addresses, redzones/padding, source preservation and producer-error propagation.
Independent IEEE checks compare every non-NaN bit and NaN classification.
Scalar equality is additionally required except for the evidenced, existing
BF16-to-F32 NaN-encoding difference described below. Even in that case IO4 must
match IO1 in every bit, including NaN payload/sign/quiet bits; producer-error
output must retain exactly0x7fc00000. This gate is not timed.

Workers with Split-K candidates also run `tools/check_dense_reducer.py` against
the compiled existing-body diagnostic carrier. It deduplicates [M,N,S] across
K/families, checks actual cohort sizes plus S8, cancellation, F16 ties/overflow,
tails/strides/alignment, repeated execution, guards and invalid spans. Its
independent oracle adds partials in increasing-S FP32 order, rounds once to F16,
and requires raw-bit equality with both reducer variants. Asys must then show
`PpuMixedInputSplitKParallelM1FastReductionKernel<2,S>`, block32, grid=M*N/64 and
zero shared memory in each compact candidate; a generic fallback is a failed
experiment, not an accepted candidate timing. Complete-call numerical checks
remain mandatory as well. Standalone numerical gates are never timed.

After updating to the commit containing this candidate, on idle GPUs:

```bash
cd /sim/eec/shared/junfu.qx/quactlize
python3 tools/run_q8_tp_refine.py \
  --phase helpers \
  --review /workspace/kpack-q8-tp-refine.lj0co_ma/timing/review-1792806745232929096.json.xz \
  --llama-source /sim/eec/shared/junfu.qx/llama.cpp.ncp \
  --llama-build /sim/eec/shared/junfu.qx/llama.cpp.ncp/build-ci-32F-cublas \
  --sdk /usr/local/PPU_SDK --gpus 1,2,3,4,5,7 \
  --jobs 8 --configs-per-batch 16 \
  --run --confirm-idle
```

The existing runner requests1700MHz; actual-frequency variation is advisory,
not a failure. Generated source/cache/raw traces stay on Box. Upload the printed
compact `review-*.tgz`; it also includes IO/reducer gate results/logs. Resume uses
the same arguments plus `--resume --output <printed run directory>`.

The coordinator takes explicit `--jobs` (1..64, default8) and
`--configs-per-batch` (1..32, default16), records both, and overrides ambient
`JOBS`/`CONFIGS_PER_BATCH` for its workers. A parent-shell build setting such as
`JOBS=192` must not leak into this bounded experiment. Worker logs also print
the effective values; invalid CLI input is rejected before launching any workers.
The first uploaded helper attempt (revision731c2db, all six workers rc2) failed
at argument validation, before compilation/numerics/timing; the old diagnostic
did not identify which inherited value was invalid. Keep that attempt intact
and start a fresh run with the fixed launcher, not `--resume` (no completed
device work exists there to reuse).

`check_dense_io.py` saves a failure JSON, including the first failing cell,
input/expected/actual hex bits, fingerprints, per-oracle differences, and all
three comparisons (IO1-vs-IO0, IO4-vs-IO0, IO4-vs-IO1). It completes only the
three variants of that cell before stopping; runtime/guard errors stop at once.

The 2026-10-24 follow-up to the six-worker4e069e1 failure identifies the seam:
BF16-to-F32,1x65536,flag absent. Scalar vs IO1 and scalar vs IO4 both have128
NaN-encoding differences; IO4 vs IO1 has zero raw differences and identical
whole-output hashes. All independent non-NaN bits and NaN classifications pass.
For example input0x7f81 produces scalar0x7fffffff but IO1/IO4 produce0x7f810000.
Local SDK2.1.1 native lowering explains the pair asymmetry: low BF16 uses
`v.cnvt.f32.bf16`, high BF16 uses `v.and.b32 ... 0xffff0000`, whereas scalar
uses conversion for every element. This is an existing scalar/vector NaN
representation difference, not an IO4 finite-value regression. Local ISA is
not a disassembly of the uploaded Box binary.

The revised contract records only that BF16 widening baseline difference in
`baseline_observations`, with all three raw comparisons. It does not permit a
candidate-only NaN change, scalar differences for F16/BF16 narrowing, NaN-class
changes, finite/Inf/signed-zero/subnormal differences or altered error poison.
No kernel, precision, tolerance or production routing is changed. The original
FAIL receipt stays immutable:147checks were executed (F16's108complete, BF16
widening39, narrowing not reached), so it cannot certify the full216-check gate.
Re-run the complete gate on the existing library before starting performance.
Evidence: diagnostic SHA256
`cecf03edd4eff9bfdf3beb1b2969a05a55fce7d1c2c3bd96dc274cdf2de645d9`,
library SHA256
`d08e81a45126c0be52bff6beb6df0c76642a6344c844f9acaa0337b7a38dc9c0`.

No production selection is promoted by this command. Compare complete calls
and both orderings before choosing IO4 or compact reduction for a measured scope.
Do not use `--profile-acu` for the IO/reducer phase: that flag belongs to the
independent dequant phase. The Asys reports and per-kernel durations are retained
for selecting a bounded follow-up profile if required.

Metadata erratum for earlier dequant reports: the old `NO_FLUSH` label meant
no software cold-ring loop in the profile range, but ACU was invoked with
`--cache-control all`. The new label makes both facts explicit. Never use those
profile replay durations as rotating-event or complete-call timings.
