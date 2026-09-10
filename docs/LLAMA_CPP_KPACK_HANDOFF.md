# llama.cpp K-pack integration handoff

This file is the single integration handoff for consuming Quactlize K-pack
artifacts from llama.cpp. Update it whenever the sidecar schema, public C ABI,
binary bundle, or loader contract changes.

## Current box candidate: parallel MoE preparation and measured SIMT intake

Use `prebuilt/ppu0010/kpack-fusion-v2/dispatch` for the native dispatcher and
execution library. The GPU packing library stays at
`prebuilt/ppu0010/kpack-fusion-v1/libquactlize_ppu_pack.so`: offline bytes and
cache schema have not changed. Both new libraries total about 1.8 MiB and
are stored with Git LFS. There are no precompiled GEMM modules in this package.
The client change is private llama.cpp `feat/kpack-gpu-cache` commit
`83e8efdfe` (automatic recipe lookup, Q8 allowance, and trace symbols).

The new preparation removes serial duplicate validation and repeated integer
division in directory construction, distributes expert/split descriptors
across CTAs/warps, and computes routing once per row/expert CTA rather than
once per K/256 gather chunk. All current IDs are still consumed on every
replay; no CPU router readback, scale cache, or ready flag was introduced.

RTX5090 ABBA prepare-only measurements: separate gate/up, 256 experts, top8,
one token **9.207 -> 4.614 us (-49.9%)**; four tokens **26.861 -> 4.803 us
(-82.1%)**. Router-inclusive cases improve as well. Ten SIMT-stage contexts
pass their changing-input replay/descriptor/guard checks. See
[raw timing samples](measurements/moe_prepare_5090_20260910.json).
These results do not establish PPU or whole-model speedup.

`auto` now checks exact measured GEMV recipes instead of reserving them for
forced diagnostics. Q8_0 has a direct K-pack2 SIMT reader: F32 source is
rounded to FP16 in registers, weights use the original FP16 d, accumulation
and output are F32. No A quantization, gather/scatter, or metadata prepass.
It is **not** enabled unconditionally: a missing recipe retains the selected
W8A16 TC path. NVIDIA numerical checks cover 15 contexts x 8 configurations;
PPU numerical/performance admission remains pending. N32 SSM matrices remain
outside the current K-pack intake; this change does not claim full Q8 coverage.

The box runner defaults to the v2 native package and `RUN_Q8_SIMT=1`. After
the existing gates it compares eight SIMT candidates on six real-sized
dense shapes at M1/M4, alternating order over three rounds. A recipe is
exported only if the whole SIMT F32 call beats the selected TC FP16 core,
without credit for TC's adapter overhead. This conservative policy is then
loaded for the same run's warmed model benchmark and Asys trace. First-use
JIT and the first complete PP/TG pass remain excluded. No Cartesian sweep.

The MoE runtime header changes its JIT source contract. Selected missing
parents must compile again under the new cache key; old keys are not reused
or deleted. The old large sweep bundles do not need rebuilding.

```bash
CUDA_VISIBLE_DEVICES=0 JOBS=192 MODEL_NAMES=qwen35-35b-q4km \
  RUN_MODEL_TRACE=1 TRACE_PROMPT=2048 TRACE_GENERATE=16 \
  bash tools/run_kpack_fusion_box.sh
```

For device gates only, set `RUN_MODEL_BENCH=0 RUN_MODEL_TRACE=0`. Results are
packed automatically; Asys report/SQLite stay on the box. GEMV policy and
per-configuration raw timings are included in the result archive.

Remaining model issues: the exact tensor override omits synthesized paired
gate/up names; the 35B output head may retain legacy FQ; 32B prefill has no
admitted SF comparison policy. Do not equate native plan creation with
execution/fusion coverage or a measured global optimum.

## Previous box candidate: Q8 and MoE fusion v1

Q8_0 W8A16, paired gate/up GPU packing, and the small indexed MoE fusion pass
the bounded `XYUgHJ` device gates. See [the fusion checklist](MOE_FUSION_IMPLEMENTATION.md)
and [the uploaded run review](KPACK_FUSION_XYUGHJ_REVIEW.md).
Use `prebuilt/ppu0010/kpack-fusion-v1` and `tools/run_kpack_fusion_box.sh`;
the historical bundles below do not contain these changes. Full-model
accuracy, fusion coverage and performance admission remain pending. Q8's canonical map is
`0x51384b5032540001` (biased int8 K-pack2 plus original resident FP16 d; no
high/zero plane). Gate/up pairing retains each qtype's existing map and uses
N doubled with gate then up within every expert, matching llama.cpp's
**conversion-time** `--fuse-gate-up-exps` ordering.

The initial fusion gate exposed a selector omission: merged Q4 N1024/K2048/E256
had no exact historical family, while its N512 source did. The selector now
allows a single N/2-family transfer for otherwise-uncovered grouped requests,
with identical qtype/K/E/M and whole output tiles. Exact families win first;
there is no recursive N extrapolation. Choices are marked `QKS_PREDICTED`,
not measured winners. Resources/stride/recipe use the actual doubled N. The
selector DSO is rebuilt, but GEMM JIT source contract, GPU producer and
execution helper are unchanged; existing selected-parent cache remains valid.

The additive `quactlize_kpack_dispatch_bind_llama_indexed_v1` binds a prepared
grouped handle to llama's F32 token/slot inputs and outputs before capture.
For <=32 routed rows it performs fused route/gather/metadata/directory, the
same selected producer, then FP16-preserving reduce/scatter. Older modules
or larger contexts explicitly miss this binding and retain their original
preparation.

The optional `quactlize_kpack_dispatch_moe_create_v1` now composes selected
gate/up/down handles with **disjoint retained scratch**, followed by
`moe_run_v1`/`moe_destroy_v1`. NULL up describes an already-merged gate/up
weight. The library shares preparation, preserves projection FP16 rounding,
fuses SwiGLU into down's expert-ordered input and fuses down reduction/scatter.
`moe_run_router_v1` additionally folds a matching 256-expert top-k into the
same preparation launch. There is no persistent scale-ready/router-ready
flag; all current device inputs are consumed on every replay. Large,
non-SwiGLU or concurrent-stream graphs explicitly retain the original path.

The new small package has zero compiled GEMM modules; selected-parent JIT is
required. The execution DSO is unchanged, while the selector and GPU producer
are rebuilt. 147 host/Python tests, loader/cache CTests and seven new 5090
SIMT-stage contexts pass. This is NOT a model/PPU performance admission.
Online two-source merging is implemented behind `QUACTLIZE_KPACK_PAIR_WEIGHTS=1`.
Sources need equal qtype/N/K/E and no per-projection bias/scales; separate-weight
LoRA is outside this opt-in contract. It reuses the conversion-time merged
tensor name and graph. Runtime cache v2 records both real source spans, even
if nonadjacent; v1 caches and offline v3 bundles remain readable. No CPU raw
hashing or inference-thread D2H wait is added.

Run `bash tools/run_kpack_fusion_box.sh --help`. Default model is the uploaded
35B Q4_K_M entry, full PP/TG axes and NPL=1; `MODEL_NAMES=all` selects the full
list (tensor-parallel K-pack remains explicitly NOT_TESTED). It executes
reference plus separate cold/hot K-pack processes and excludes the first
whole pass per PP. `RUN_MODEL_TRACE=1` adds a warmed Asys proof. Return the
one `kpack-fusion.*.results.tgz`; large raw Asys files stay on the box.

The box model root is `/sim/eec/shared/AI_workspace/llm-models`, not
`bench_model_zoo` or `/sim/ollama_HLLM_compare`. The five catalog entries are
retained. Before device gates, the runner resolves selected model directories
to one GGUF file or a complete split set (including nested shard directories),
prints each path and saves `results/model-plan.json`. Benchmark and trace use
that same plan. It does not hash/read weight payloads or substitute HF/GPTQ
weights. Multiple GGUF families require an explicit file `path` in `MODEL_PLAN`;
`MODEL_ROOT` replaces the single root without adding fallback directories.
Direct GGUF files take precedence over nested directories; hidden cache files
are excluded. File symlinks retain their public shard names instead of using
the backing blob's filename. The 35B base entry explicitly names
`Qwen3.5-35B-A3B-BF16-00001-of-00002.gguf` and requires its second shard.
The 32B base entry explicitly names `Qwen3-32b.gguf`; a colocated
`Qwen3-32B-eagle3.gguf` is not a replacement or fallback.
The 32B Q4_K_M directory is exactly `Qwen3-32B-Q4_K_M_GGUF`
(underscore before `GGUF`), with no alternative-spelling fallback.
Paths can be checked separately, without SDK/GPU work:

```bash
python3 tools/resolve_kpack_batched_models.py --model qwen35-35b-q4km
```

For the narrowed comparison use `MODEL_PLAN=$PWD/tools/kpack_batched_int4_2048.json`
and `MODEL_NAMES=all`: only 35B-A3B Q4_K_M and 32B Q4_K_M, PP=2048, TG=128,
NPL=1, token batch/ubatch=2048. Each of reference/K-pack-cold/K-pack-hot runs
one excluded warmup plus one measured pass: 768 generated tokens per model,
versus 107,520 in the original matrix. Progress prints both us/token and total
milliseconds. Existing results and JIT cache stay valid; no GEMM rebuild is
needed for this protocol change.

The uploaded focused run `XYUgHJ` exposed missing plan logs: default verbosity
3 suppresses GGML_LOG_INFO receipts although JSON timing and JIT helper output
remain visible. The benchmark now uses `--verbosity 4` in both arms, without
per-kernel DEBUG. It keeps route checks strict and saves `.selection.json`
even on a missing-plan/Q8-contract failure. This is a script-only repair;
neither Quactlize libraries nor llama C++ nor cached GEMM parents change.
All six model processes exited successfully in the old run, but its four
K-pack arms remain unadmitted and no Asys capture was reached.

Set `RUN_MODEL_TRACE=1 TRACE_PROMPT=2048 TRACE_GENERATE=16` for the subsequent
35B Q4_K_M Asys capture. It now uses `quactlize_native.py --proof-only`: one
warmup and one captured request, with no duplicate server ABBA benchmark.
The trace uses the same header-derived eligible weight names as the benchmark,
including supported dense Q8 weights, instead of the old MoE/output-only regex.
Both repositories must be updated; the wrapper checks the script option
before device work. The private llama `feat/kpack-gpu-cache` change starts at
`558e5a06b`. Partial kernel coverage stays labelled partial, not a
performance or correctness admission. The report remains at
`results/model-proof/proof.asysrep`. The 122B TP and BF16 models are outside
this focused plan.

Last updated: 2026-09-10. Verified offline bundles retain schema v3; the local
runtime cache now has a separate hash-free contract, described below. The
published `2826cf1` loader-safe runtime bundle has passed strict binary
inspection, its selected-config oracle, and all 26 host ABI cases in a fresh
LFS checkout. Its PPU device gate is still **PENDING**. Host/ELF admission is
not device admission and does not authorize deployment by itself.

## Current routing: FQ decode; per-call SF and JIT native gate passed

Latest native JIT retry: `5gSImm` at `72b6db8` passes **28/28 in both
processes**, including Q4 SF-grouped preparation and Q6 SF-dense M128. Use
`prebuilt/ppu0010/kpack-jit-v2` for the next model integration gate. All 56
module resolutions hit the original 23-parent cache, with unchanged device
binaries and execution DSO. Eager, graph and metadata checks pass; this does
not establish the updated llama model's latency, accuracy or startup cost.
No recompilation of those cached parents or llama.cpp is required solely for
this repair. Keep the six intake/fallback libraries. Details:
[JIT gate review](KPACK_JIT_GATE_REVIEW.md).

Weight prefetch is **parked** (T15). The supplied `fBihrU` Q4-to-Q5 summary
shows hints reducing the instrumented pair from 41.24 to 40.88 us (0.87%);
39.40 us for primed weights excludes the preload cost. These are not model
speedups. Preserve the [experiment and deferred checks](KPACK_PREFETCH_EXPERIMENT.md),
but do not add its helper or a prefetch dependency to the production path.
Resume the small-JIT-package model integration gate instead.

The model runner now defaults to `kpack-jit-v2`, uses the existing
`JIT_CACHE=/workspace/kpack-jit-cache`, and prewarms only deduplicated parents
for the consumer's weight-name scope. Cache receipts and payload hashes are
bound to the actual selected trace symbols; the test no longer assumes every
module resides inside `bundle/modules`. This update changes scripts/evidence,
not production kernel binaries or device cache keys. The parked GEMV gate is
not required to resume a complete native gate.

Performance convention: one excluded first request per process and prompt
shape, then steady ABBA requests. Asys uses the same server process, starts
collection only after a completed warmup request, and captures the identical
second request. Cold/JIT/first-use costs stay separate. Local host contracts
pass; the new Asys session and full-model path still need PPU validation.
The known Q6 output-head fallback remains `INCOMPLETE_NATIVE_COVERAGE`; its
timings and partial trace are saved rather than hidden by the missing dense
activity. Q8_0 remains ordinary llama.cpp, not newly admitted K-pack.

The [complete delivery backlog](KPACK_EXECUTION_FOLLOWUP.md#complete-delivery-backlog)
is the current task authority, including production JIT, model-load
preparation, cache lifecycle, small-library packaging, Q8/Q6 coverage,
provider work and main admission. SIMT GEMV is provisionally accepted by
the user for this milestone; further optimization is parked, not a blocker.

ScaleFirst is required to expand scale/zero on the GPU for every SF call.
The updated adapter removes `ggml_quactlize_prepare_scales` and
`art.scale_ready`, using per-stream scratch and a prepass node before each
SF GEMM/replay. The native gate and V2 route exporter include every expansion.
The SDK build, CPU contracts and native PPU gate pass; the updated llama
adapter/full-model gate remains pending.
The K-pack weight disk cache is independent and remains valid.

Following the matched PPU comparison below, automatic single-token decode
uses selected FQ for both dense and grouped K-pack weights (Q2_K through
Q6_K). It does not consult the GEMV policy, even if an older policy file
contains a matching recipe. The existing selected parent, compact schedule
and Split-K settings are retained. This is a llama.cpp adapter change;
the 216-module native package, kernel DSOs and offline format are unchanged.
Incrementally rebuild llama.cpp and restart the process to apply it.

The earlier FQ decode switch is consumer commit `135d7edcf` on the private
`feat/kpack-gpu-cache` branch; the per-call SF/JIT update follows it.
Current consumer: `cba8d4a0620adce5baa4c6534f968a55a87432c3` (implementation
`229fe8660`, followed by JIT-cache evidence and warmed Asys request capture).
The latest local validation passes 80 Quactlize host/orchestration tests and
11 llama evidence tests; the small package's source contract and both DSO
hashes are unchanged. This does not claim the new model capture ran on PPU.
The complete local PPU backend
and adapter/buffer executables link successfully. The Quactlize local gate
passes 173 related pytest cases after the host-grid/gate-order repair;
llama's five parser cases and delayed-D2H buffer positive/three planted
negatives also pass. Native device arithmetic and graph replay pass in
`5gSImm`; full-model/adapter device admission is still separate.
The changed production and adapter-test translation units compile with the
local PPU SDK; 49 host policy/parser tests pass. The new `auto` device test
is included in the box runner, but has not yet been run on PPU after this
change. No new model speedup is claimed.

The old deployed prefill measurements used a resident FQ/SF policy; they
cannot admit the new per-call behavior. The updated adapter rejects that V1
table; produce V2 from new native-gate results. Explicit `sf`
and `gemv` route overrides remain diagnostic options. Automatic FQ decode
is already separate; setting global `fq` would also override prefill.
Requests already falling through an empty GEMV policy to FQ do not acquire
a different kernel from the decode-only change.

Q8_0 is a separate pending integration, not a missing int8 collective: the
repository already has a controlled ScaleFirst/I8 Q8 path and a sweep. Its
historical A32/F1 Xplane fixture, however, is not the production K-pack
arrangement and no checkpoint GPU producer/public native route is connected.
Reuse that SF implementation when adding Q8 production support; do not alias
qtype 8 to a K-quant format. See the
[local TODOs and required PPU gates](KPACK_EXECUTION_FOLLOWUP.md#local-work-and-required-ppu-gates-2026-09-09).

The reviewed `kpack-decode.XZM60u` experiment passed 260/260 cells. A follow-up
[GPU compact/persistent package](KPACK_GPU_COMPACT.md) now implements the
missing device-only compact schedule and persistent S1/S2/S4/S8 inside parent
modules, without changing the external grouped ABI. Sixteen new modules compile;
the uploaded `kpack-compact.yAQMNP` gate passes 204/204 cells. Same-parent Q4
compact S2 improves 7.23%, and Q5 compact S1 improves 45.15%. It reuses the
existing GPU directory and includes its build cost plus Split-K reduction in
timing. No CPU router is introduced. The deployed native bundle and llama.cpp
selection have not been switched to these experimental modules yet.
Standalone profiling now defaults to ACU; the experiment guide provides a
five-report **TM8/WM8** old/new capture, including Q4 compact S2 and S4.
Use `--case q4-up --arm compact --split 4` on its box runner to capture only
the added S4 arm. TM16 is retained only as a historical
control for this single-token profiling task; the profiled modules already
passed the uploaded gate and need no rebuild. This does not claim a new
llama.cpp model trace or change the production selector.

An [equal-weight dense/grouped comparison](KPACK_DENSE_GROUPED_AB.md) reuses
three published modules: dense Q4 N4096/K2048 versus eight active N512/K2048
experts with a shared activation. Five FQ timing arms and three ACU captures
separate the historical AP1/TN128/S8 configuration from matched AP0/TN64
S2/S4 controls. It is a diagnostic, not a new llama.cpp dispatch policy;
ScaleFirst is excluded and PPU reproduction results are pending.

An additive [decode experiment](KPACK_DECODE_SWEEP.md) now supplies ordinary
grouped Split-K and an explicit SIMT word-pair/FMA reader. The experimental
package is not a replacement for `kpack-native-v1` and must not be selected
by llama.cpp yet. Its one-command box gate checks 260 cells, including FP32
partials and mutable device routing. Production heuristic and libraries stay
unchanged until the results are reviewed.

Latest update: `kpack-native-model.O0ki3q.results.tgz` has passing 28 native
contexts, 14 GEMV contexts and adapter tests. Model prefill improves, but
decode remains about 31% slower; the selected-native dense trace admission
is still incomplete because the Q6 output head uses the labelled legacy FQ
fallback. No new production optimization or DSO is published in this update.
The Q4 grouped decode geometry was measured previously, including TM8; the
18.44-us historical winner is the same parent selected now. Its low effective
bandwidth remains a kernel performance debt, independently of adapter cost.
[Single-op coverage audit and replay](KPACK_GROUPED_DECODE_REVIEW.md) provides
a no-recompile same-parent device-only/host-compact diagnostic. Do not use the
host-compact control as a dynamic-router production replacement.

The GSM8K pilot is now reviewed. Archive `llama-kpack-gsm8k.EEZKMu.results.tgz`
(`cb8b7906...`) has 128/128 matched requests, both arms 122 correct, and no
paired correctness flips. It also shows an observed decode regression:
8.0932 -> 10.4625 ms/token (+29.27%), including +29.17% median on the 43
identical-output pairs. This is ordinary llama.cpp GPU versus current FQ
K-pack GEMM, not a standalone Xplane comparison. No new kernel trace was
collected. The full raw-result review, caveats and delivery checklist are in
[KPACK_EXECUTION_FOLLOWUP.md](KPACK_EXECUTION_FOLLOWUP.md).

Native C++ selection is now wired to both dense `MUL_MAT` and grouped
`MUL_MAT_ID` on `feat/kpack-gpu-cache`. `QUACTLIZE_KPACK_EXECUTION` opts into
the additive `prebuilt/ppu0010/kpack-native-v1` package: 216 selected PPU
parents, one SDK-free host dispatcher and the five-format GEMV/prepass DSO.
The old six-library bundle remains required for intake and labelled policy
misses. It is not rebuilt or silently replaced. No new device admission yet.

Handles and complete recipes (including Split-K and persistent grid) are
prepared and cached before graph capture. Grouped bounds stay on the GPU;
ScaleFirst expands metadata on every call into per-stream scratch. The old
per-weight expanded-value cache and separate scale-ready event were removed.
Compute
has no explicit wait on D2H/cache publication; first-use allocator
synchronization remains a timing caveat.
`QUACTLIZE_KPACK_GEMV_POLICY` is an exact measured TSV generated
by the offline gate, used only for explicitly forced GEMV diagnostics;
an unmeasured initial recipe is never substituted.

The default behavior without `QUACTLIZE_KPACK_EXECUTION` is unchanged. With it,
automatic single-token decode uses selected FQ. Prefill reads the paired
FQ/SF measurements in `QUACTLIZE_KPACK_PREFILL_POLICY`; the V2 exporter
compares per-call expansion plus GEMM with a 2% margin. Old resident-only
V1 tables are rejected. Missing entries or resource
declines retain selected FQ. Unknown families retain the
old canonical K-pack FQ path with an explicit log. Grouped bound-based choices
and route-level SF decisions are not globally optimal measurements.

Next box entry (no Quactlize compilation; llama.cpp must rebuild):

```bash
git pull --ff-only
git lfs pull --include='prebuilt/ppu0010/kpack-native-v1/**,prebuilt/ppu0010/kpack-execution-v1/**' --exclude=''
bash tools/run_kpack_native_box.sh --help
# Set LLAMA_DIR, BUILD_DIR, MODEL, CACHE_DIR and the existing SDK/legacy bundle/pack library.
CUDA_VISIBLE_DEVICES=0 JOBS=192 bash tools/run_kpack_native_box.sh
```

Run from Quactlize develop and update the private llama branch first. The
script stops before model measurement on a numeric/binding failure, preserving
the Docker shell. Return its printed `kpack-native-model.*.results.tgz`.
The workflow includes 28 selected numeric contexts, 14 model-shaped GEMV
contexts, adapter graph tests, real-model ABBA and a separate short device
trace. Exact counts, scopes and remaining caveats are in the follow-up document.

The partial `kpack-native-model.q40qV3` run stopped on a Q3_K SF metadata
comparison. The native gate's oracle has been corrected: the historical
timing fixture encodes some Q3/Q6 zeros as `-0`, unlike canonical `unit_group`.
It now decodes the actual packed units independently, retaining strict bit
equality and reporting first-difference coordinates/bits. This is a test-only
change; no kernel, selected recipe, offline format or DSO changes. See the
follow-up's metadata-oracle section for local proof and pending box closure.

For a later loader-stub failure, update the private llama branch to
`908248271` or later as well:
production bundle/pack overrides must not leak into stub negative tests.
`RESUME_RUN=/workspace/kpack-native-model.<previous>` on the native box entry
reuses complete, identity-checked 28-context native and 14-context GEMV gates.
It creates a fresh result directory, regenerates policies, rebuilds/tests the
adapter, then runs new model ABBA/trace measurements. Old results and all
Quactlize DSOs are preserved; incomplete or changed gates are not reusable.

The old model override covered grouped experts only. The current performance
runner uses `^(blk\.[0-9]+\.ffn_[a-z0-9_]+_exps\.weight|output\.weight)$`,
adding only the root Q6 dense head to the expert weights. Use private llama
commit `5837b4d86` or later: the prior unanchored `output\.weight` also matched
`blk.3.attn_output.weight` under the loader's `regex_search`. An explicit
buffer override bypasses automatic buffer admission, so this selected Q8_0
for K-pack and aborted model loading; the client's connection reset was a
consequence. Python and C++ regex regression checks cover 367 tensor names.
No kernel, format or DSO changed. Completed identity-checked micro gates can
still be resumed; model performance and device trace must run again.
Q8_0 dense weights remain ordinary GPU; they are outside the K-quant ABI.
Each model log includes `[quactlize-plan]` (parent/build/split/grid or measured
GEMV recipe). `[quactlize-prepass]` GPU event intervals are read at teardown.
Neither host plan receipts nor library loading alone are device execution proof.

SF timing must include metadata expansion for every SF invocation followed
by GEMM. Expanding all 120 MoE weights writes 3.75 GiB of scale/zero and has
5.625 GiB ideal aggregate read/write traffic; it does not require all planes
to stay resident if temporary scratch is reused. Do not amortize expansion
across requests. Retain packed units and the existing weight disk cache.

## Completed box entry: single-request GSM8K generated answers

llama.cpp `a39917a66acd742ae02b2034549fded8ce9506bb` is pushed to
`DrXuQian/llama.cpp`, `feat/kpack-gpu-cache`; remote SHA verified.
Run `bash tests/run-quactlize-gsm8k.sh` with the existing SDK, model,
runtime bundle, pack library, cache and configured PPU build, plus the local
GSM8K **test** file in `GSM8K_FILE`. No new Quactlize library is required.
Only llama-server and its changed dependencies are incrementally built;
the UI build/download is disabled. Main and develop are unchanged.

This evaluates actual answers, not the earlier question/answer-text PPL.
Default: 128 sampled questions without replacement, seed 20260908, greedy,
thinking off, context 4096 and at most 1024 generated tokens per answer.
The ordinary GPU server runs first, then a cached K-pack server, with one
model load per arm, one server slot and one request at a time. Business
batch is **1**; the default 128-token batch is only a prefill chunk.
Gold reasoning/answers never enter the model prompt. The runner compares
the exact token-array/request hashes across arms and validates effective
sampling, prompt counts, no KV-prompt reuse and the same model/template.
Final numeric answers must follow `####`; truncations and failed parses
remain in the denominator, with separate counts and paired changes.

This suite does not run a profiler. It checks GPU/K-pack buffer placement
and complete disk-cache hits, while retaining the earlier numerical
gate's independent device execution evidence. Its own records explicitly
say `kernel_execution=NOT_COLLECTED`; neither route placement nor generated
text is mislabeled as a new activity trace. Response timings are diagnostic,
not a controlled throughput comparison when answer lengths differ.

Local validation: 16 answer-protocol/lifecycle tests and 32 existing numerical
tests pass. Two real subprocesses/local fake HTTP servers exercise the actual
client, including the full shell wrapper, injected second-answer HTTP failure,
preserved partial answers, process shutdown and result archiving. These are
not model or device oracles. The existing local PPU llama-server target was
incrementally built, and four actual-binary CLI-only argument sets pass with
CPU buffer names. The box pilot is reviewed above; wider accuracy remains
separate from this 128-question result.

Upload the printed `llama-kpack-gsm8k.*.results.tgz`: raw generated responses,
selected IDs/prompts, token receipts, `summary.json`, logs and binary receipts
are included; model/cache contents are not. Every answer prints progress and
an observed ETA for the current arm only. `GSM8K_CASES`, `GSM8K_MAX_TOKENS`
and `GSM8K_TOKEN_BATCH` are documented in `tests/quactlize-gsm8k.md`.
The 128-question default is a pilot, not full benchmark/accuracy admission.

## First model numerical gate with device execution evidence

### QuHHHY partial result and performance-log fix

Archive `llama-kpack-numerical.QuHHHY.results.tgz`, SHA-256
`acba469b94ff2effd2712a32266fac4ef3dc1f2914ad5ca9edcbd584ad9f5005`,
records source `d45f8fec2`. The token-batch-128 short proof passes (480
grouped GEMMs, all 120 tensor routes). The three extended numerical phases
complete with model rc=0 and matching raw metrics/JSON. Context 1024 x eight
chunks scores 4,088 positions; token SHA-256 is
`64219112ff53b2147c1ed520fbb111b332c0b8da971ca05a984ff68d8f40a6e4`.
Ordinary/K-pack paired PPL is 1.739846 / 1.741794, ratio 1.001120, mean KLD
0.002658, max KLD 0.588895 and top-token agreement 98.875% (46/4,088 changes).
Ordinary self-replay has 100% top-token agreement and max KLD 0.000004.
These results are preserved; they are not a generated-answer accuracy score.

The first performance process also exits 0 and completes all eight chunks,
but its log has zero model timer records. `--verbosity 3` hid library INFO:
`common_log_default_callback` maps that severity to TRACE threshold 4,
whereas application INFO uses 3. Thus application PPL remains visible but
library timers/placement do not. No timer can be recovered from that log,
and process wall must not replace a model timer. Token-batch-1 did not start.

Fix `d25b88a18b9e1a449c54b3bfa06d9ec1202e2bbc` is pushed to the same fork
feature branch. Performance now uses threshold 4, with DEBUG (5) and the
profiler still disabled. The actual `common/log.cpp` callback is compiled
and tested locally: old threshold 3 drops timers, 4 retains them without
DEBUG, 5 admits DEBUG. The corrected host orchestration fixture reproduces
the old stop; all 32 tests pass after the fix, plus four actual-binary CLI
checks. No kernel, DSO or computation change.

`--performance-only` is a scoped fresh rerun of ABBA timings, not automatic
resume. Selecting `EVAL_BATCHES=128` repeats only the four missing prefill
timings. `EVAL_BATCHES=1 --extended` can run the as-yet-unstarted single-token
arm. Preserve this archive alongside later results; no rerun of completed
token-batch-128 numerical phases is needed.

Terminology: both token batches use one sequence (`n_seq=1`), hence business
request batch size is always one. `-b 128 -ub 128` is a token submission chunk
for single-request prefill, not 128 concurrent requests or total sequence
length 128. `-b 1 -ub 1` exercises the single-token path. Those numerical runs
used GSM8K as fixed likelihood text only. The later 128-question generated-
answer pilot is reviewed above; it is separate evidence, not a PPL score.

### Next box entry: extended coverage and separate model timings

llama.cpp `d45f8fec232930c47632ecbe48ae201c7dee99e1` is pushed to
`DrXuQian/llama.cpp`, `feat/kpack-gpu-cache`; remote SHA verified. Use the
same environment/build/cache with `bash tests/run-quactlize-numerical.sh --extended`.
Only test scripts/parsers/documentation changed. No GEMM/pack DSO, buffer
ABI, loader or compute dispatcher change; develop/main are untouched.

Each batch/ubatch (128 and 1) runs one short cached device proof, then
ordinary GPU reference-save/self-replay and cached K-pack/reference
comparison at context 1024, eight chunks: 8,192 input tokens and 4,088 scored
positions. Only the short proof is traced; large comparisons explicitly
report `kernel_execution=NOT_COLLECTED`. No repeat of GPU packing or cache
publication, and no CPU reference. GSM8K rendering uses the first 256 records
to provide enough text, not generated-answer evaluation of those questions.

Separate performance processes run reference/cache/cache/reference for
each batch: normal warmup, library INFO logging (threshold 4 after the fix), no profiler, no saved probabilities
or KL comparison. The summary uses model-evaluation timers, not load time or
process wall. It retains both samples per arm and spread; this is an initial
model-throughput comparison, not an isolated GEMM or 5%-confidence claim.
The existing `llama-bench` does not expose this branch's cache option, so this
entry reuses `llama-perplexity` without expanding benchmark/production code.

Local checks: 31 tests pass, including synthetic full shell orchestration
of both modes; 12 phase/batch argument sets pass the actual PPU executable's
host-only `--help` parsing with CPU buffer names. The ten `CigOEn` raw logs
reparse with unchanged routes/metrics. These are not new GPU numerical runs.
The partial `QuHHHY` result and remaining work are recorded above.
At least 8 GiB free is required in `RESULT_ROOT`; two reference probability
files use about 3.8 GiB with this model. Upload the printed `.results.tgz`
(including `performance-summary.json`); large files stay on the box.

### Latest result: CigOEn, execution and cache replay verified

Uploaded `llama-kpack-numerical.CigOEn.results.tgz`, SHA-256
`7511cc22ad4c4fe044439418b07068072e0451f80a6365b6a2d5d453411d8460`,
records llama.cpp source `35b74114f806a4581daff9b916390f7b0ee8996d`.
All ten processes exit 0; raw application metrics/routes, phase JSON and
summary TSV agree. No reported CUDA/PPU launch or decode errors. The five
inventory DSO hashes match the supplied `2826cf1` bundle manifest; the GPU
pack DSO is unchanged (`611ec98c...`). FA/MOE/GDN are OFF, Quactlize and
CUDA graphs are ON. This still uses the transitional FQ format DSOs,
not the new C++ heuristic/JIT dispatcher.

| Mode | Ordinary GPU PPL | Uncached K-pack PPL | Cached/reference PPL increase | Mean KLD | Max KLD | Top-token agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Batch/ubatch 128 | 1.4282 | 1.4325 | 0.2995% | 0.003950 | 0.268156 | 98.819% |
| Batch/ubatch 1 | 1.4239 | 1.4244 | 0.0370% | 0.008010 | 0.503132 | 98.425% |

The PPL increases use the paired `cache-reference` ratio, not rounded PPL
division. Each mode scores the same 254 positions in two 256-token chunks;
all four probability-file receipts have identical token SHA-256
`ecbff63226d90a7498d0ac8a9d09f851aac5331c9450a907eeb4c63bf8fa5bdc`.
The model has 120 routed grouped tensors (80 Q4_K, 40 Q5_K, E256). Each
K-pack phase reports exactly 480 grouped GEMMs at batch 128 (320 fmt0,
160 fmt1), or 61,440 at batch 1 (40,960 fmt0, 20,480 fmt1). Post-start
route records cover all 120 tensors. Ordinary GPU phases have nonempty
activity and zero matching grouped GEMMs. Submitted activity records match
the exact inventory names and positive durations; raw SQLite/asysrep and
probability payloads remain on the box and were not replayed locally.

All four cached processes report 120 uploads, zero resident misses and no
GPU repack. Cached-versus-uncached top-token agreement is 100% in both modes;
max KLD is 0.000004 / 0.000043, versus ordinary GPU self-replay maxima
0.000004 / 0.000046. No additional cache-induced numeric loss is observed
on this sample. This does not assert bitwise equality of compressed logits.

Review outcome: execution coverage and model cache replay pass for this
sample. Cross-route differences are above self-replay noise: 3/254 and
4/254 top-token changes, with KLD tails shown above. Do not label them a
proven kernel bug or proven harmless rounding from these results alone.
Broader model accuracy remains pending; this is neither GSM8K answer
accuracy nor a full-corpus/long-context or Q2/Q3/Q6 model qualification.
Next: expand numerical coverage and collect separate unprofiled performance
measurements. Traced runtimes are not performance admission. No production
kernel, DSO, packing format or runner change was made during this review.

### Historical runner bring-up

The operator has returned the `F56TMR` recheck using `35b74114f`:
batch 128 / `reference-save` has `route_verdict=PASS`, PPL 1.4282,
12,594 GPU kernel calls and zero matching Quactlize grouped GEMMs. The
saved probability payload passes its header/size check: context 256,
two chunks, vocabulary 248,320, 254 scored tokens, token SHA-256
`ecbff63226d90a7498d0ac8a9d09f851aac5331c9450a907eeb4c63bf8fa5bdc`.
These are operator-pasted checker results; the underlying archive has not
been reviewed locally. This admits only the ordinary GPU reference arm,
not K-pack arithmetic. K-pack/cached comparisons and batch 1 were pending
until the later `CigOEn` run above.
The current runner creates a fresh result directory and has no resume mode;
The complete `CigOEn` run repeated this reference arm. No Quactlize DSO rebuild.

Checker follow-up `35b74114f806a4581daff9b916390f7b0ee8996d` fixes a
save-phase log mismatch exposed by `llama-kpack-numerical.F56TMR`:
the producer prints `perplexity: calculating perplexity over ...`, while
the checker omitted the second `perplexity`. Correct executions would be
rejected at the start-record check. The fixed checker uses phase-specific
messages and reports observed headers when the requested shape is absent.
Source-derived save-message tests reproduce four old failures (ordinary GPU
and K-pack, batches 1/128); all 22 tests pass after the fix. Wrong chunk,
context, batch, sequence count and message-kind cases still reject.
The preserved reference-save log/SQLite and probability payload were
rechecked without a model rerun or DSO rebuild, as reported above.
This is not evidence of a kernel fix or K-pack accuracy admission.

CLI follow-up `c7fcaea17210c68a073aa923f04f5fe06f621343` fixes the next
box-side pre-evaluation error: `--color` is completion-specific and was
incorrectly used in perplexity arguments. Both numerical entries now use
`--log-colors off`. The exact phase argv is parsed with `--help` before
profiling/loading the model. Actual local PPU executable rejects old
`--color` and accepts the replacement; six batch/mode parser checks pass
(CPU buffer name for host-only parsing, no model or kernel execution).
All 18 evidence/corpus tests still pass. No Quactlize DSO or profiler-scope
change; PPL/KLD/device-execution results were pending at that point. Asight here is
kernel-activity tracing for execution evidence, not ACU full metrics or a
requirement of perplexity arithmetic itself. Traced times are not performance.

The first box attempt stopped at corpus download (`stage=corpus`, line 84,
rc=4), before inventory/build/evaluation. This is not a numerical/kernel
failure. llama.cpp `ced89fc7560bc6075160357b0a5e0fe78a128a5b` adds
`GSM8K_FILE` for a local JSONL/JSON/Parquet file, exclusive with `EVAL_FILE`.
It renders the first 32 question/answer records as a fixed corpus, bypasses
network download, and saves source/text hash receipts. Parquet needs existing
`pyarrow`; the runner installs nothing. The original file is unchanged.
All 18 local checks pass, including JSONL sampling, schema negatives and
Parquet. The box uses the existing `gsm8k/main/test-00000-of-000001.parquet`.
This is likelihood
comparison, not GSM8K generated-answer accuracy; device trace requirements
and the two-chunk evaluation budget are unchanged.

llama.cpp `50079485e57ea7894c5aa76ce74c3836a5250375` adds
`tests/run-quactlize-numerical.sh` on `feat/kpack-gpu-cache`. It reuses the
existing configured PPU build, published format libraries and complete
`Lwpxya` cache. Only the perplexity tool needs an incremental build; there is
no Quactlize kernel/DSO or cache-format change.

The initial scope is two 256-token chunks / 254 scored tokens at batch/ubatch
128 and 1. Five fresh processes per mode compare ordinary GPU self-replay,
uncached K-pack, cached K-pack against uncached K-pack, and cached K-pack
against ordinary GPU. The standard WikiText-2 test corpus is downloaded if
`EVAL_FILE` is unset. This is not a CPU reference or full-corpus admission.

All numerical processes collect Asight kernel-activity traces. Executed
grouped GEMM symbols must match the actual delivered format-library device
inventory, with positive duration and enough calls for the workload. Model
warmup is disabled; post-evaluation-start route records must cover all cached
grouped tensors at the requested batch. Ordinary GPU baseline must have
nonempty GPU activity and zero Quactlize grouped GEMMs. Loading a DSO or
running a pack/gather/metadata kernel does not pass this gate.

Local checks: 15 evidence-parser tests pass, actual five-library inventory
contains 55 unique grouped GEMM entries, and the existing PPU perplexity
target builds/links. The subsequent `CigOEn` result is reviewed above.
Upload the printed `llama-kpack-numerical.*.results.tgz`;
large traces/SQLite/log-probability payloads stay on the box. Trace timing is
not a performance benchmark. PPL/KLD admission remains review-based; the
reference self-replay measures serialization/replay noise.

## Latest PPU result: Lwpxya, source prefetch and H2D pipeline pass

Archive `llama-kpack-smoke.Lwpxya.results.tgz` reports source
`ced6f241e28c344fef895fa95d6931855c73c4cb` and
`KPACK_MODEL_CACHE_SMOKE PASS`. Library receipts pass. Raw model/time logs
agree with the timing summary; all three processes exit 0 with no reported
CUDA/PPU launch errors and 62 graph reuses each. Same model and PPU-ZW810
PCI `0000:08:00.0` as before.

| Phase | Reported load (s) | Process wall (s) | Peak host RSS (GiB) | Decode (tokens/s) |
| --- | ---: | ---: | ---: | ---: |
| baseline, GPU pack without cache | 4.35924 | 8.38 | 21.17 | 92.05 |
| cold, create cache | 4.38746 | 22.24 | 21.19 | 91.98 |
| hit, upload cache | 3.90882 | 8.54 | 20.53 | 91.94 |

Hit confirms `source_mmap=on-demand`, one two-slot/16 MiB H2D pipeline,
120 uploads, zero misses and zero GPU packs. Its host peak falls from the
previous run's 39.14 GiB to 20.53 GiB (18.61 GiB / 47.56% less). Reported
load is 10.33% below the same-run GPU-pack baseline; total process wall is
still 0.16 s higher. Old hit load/wall were 5.32570/9.98 s, but baseline also
changed between runs, so cross-run timing changes are not isolated speedups.
Hit filesystem-input counter and major faults are both zero: this is a
new-process warm-filesystem-cache result, not a cold-disk benchmark.

Actual-buffer upload preflight passes all ten expected cases (five formats,
E1/E257), 20 eager reads and 60 graph replays with immediate source reuse.
The separate readiness preflight passes 18 eager checks, 18 replays, six
transitions and two missing-wait negatives. These establish byte-transfer
and synchronization coverage, not K-quant model arithmetic accuracy.

The cache remains `llama.kpack-cache` v1 / arrangement v2 with no hashes:
80 Q4_K + 40 Q5_K grouped E256 tensors, 18.125 GiB, 613 skipped records.
Region sizes sum to storage size. Cold publication takes 15.26 s (12.45 s
copy/write and 2.81 s flush); context teardown at 6.097940 s precedes
publication at 19.834908 s by 13.737 s. First-write process-exit latency
therefore remains. Neither physical overlap nor the individual contribution
of each optimization is isolated by this combined single-sample run.

**Both loader optimizations pass this model/transport smoke.** Model PPL/KLD
is still `NOT_RUN reason=EVAL_FILE_NOT_SET`; provide a corpus for the existing
numerical entry. Quactlize libraries/compute route are unchanged; this does
not admit the new C++ dispatcher, broad routes or main.

Archive SHA-256:
`e89a779275566a3e0e7711d662fc6f0f09602a2efe3d9e98b131ab48acdfaa7e`.

## Bounded cache H2D pipeline (2026-09-08, device smoke passed)

The user approved the second optimization. llama.cpp commit
`ced6f241e28c344fef895fa95d6931855c73c4cb` includes the preceding source
prefetch fix and changes cached-plane installation to two reusable 8 MiB
pinned slots per K-pack buffer (16 MiB for the tested single-buffer model).
CPU staging can overlap the other slot's H2D transfer. Chunks can cross
low/high/unit boundaries; an absent high plane is skipped. A slot waits only
before reuse, and the final queued chunks are not drained per tensor.

`set_planes` still consumes all borrowed CPU inputs before return. Remaining
DMA reads buffer-owned pinned memory, not the mmap or caller's temporary
allocation. Immutable per-tensor ready events still order eager and captured
consumers; no change to their flag handling. Teardown drains the upload stream
before freeing slots. GPU packing on a cache miss and the background D2H/file
writer are unchanged. There is no new dependency from compute onto D2H or disk.
This is a bounded loading-thread pipeline, not an unbounded background upload
queue or a claim that all host waits are eliminated.

Pushed only to `DrXuQian/llama.cpp`, branch `feat/kpack-gpu-cache`; the remote
SHA was verified. No upstream push or PR.

Host loader/buffer/cache tests pass five repetitions each. Delayed queues
exercise five formats, E1/E257, discontiguous planes, absent/present high planes,
partial chunks, immediate source overwrite, reuse across tensors, a pending
tail, and teardown with pending DMA. Injected synchronous upload and early
slot reuse both fail; the existing blocking-D2H negative still fails. PPU SDK
compile/link passes for completion, perplexity, readiness and the new upload
device test. These local checks do not establish real transfer speed or model
numerical accuracy.

Use `tests/run-quactlize-cache-smoke.sh` after updating the fork branch. It
incrementally rebuilds llama.cpp/libggml-cuda and the small test, reusing the
unchanged Quactlize compute/pack libraries. A new preflight runs actual buffer
uploads with synthetic bytes: five formats, ten cases, 20 eager reads and
60 graph replays; no CPU quantization reference or GEMM oracle. Then the script
runs independent baseline/cold/hit model processes, asserting both mapping
modes and the pinned pipeline, and reports load/wall/RSS/cache results. The
existing optional `EVAL_FILE` PPL/KLD path is unchanged. The subsequent
combined PPU result is recorded above; isolated A/B timing is not available.

## Cache-hit source prefetch (2026-09-08, combined device smoke passed)

llama.cpp commit `bf2877307f1bcb0d9b264ccf350a2329d2fe03a8` moves the
existing cache metadata probe before `init_mappings`. A valid, nonempty
cache disables whole-source GGUF prefetch; the original file remains mapped
and uncached/CPU/descriptor-declined tensors still load on demand. Missing
or invalid cache keeps the previous prefetch behavior. Metadata-only
`no_alloc` probes do not open/create a cache. `--mlock` semantics are unchanged
and may still fault source ranges when explicitly requested.

Pushed only to `DrXuQian/llama.cpp`, branch `feat/kpack-gpu-cache`; remote
SHA verified. No upstream push or PR.

This commit implements only the first optimization. Its H2D uploads, source
lifetime waits, ready events, D2H writer, plane layout and Quactlize DSOs are
unchanged. The subsequently approved H2D implementation is described above;
it cannot remove upload bytes and has no isolated PPU speedup measurement yet.

Host cache/loader suites pass five repetitions, including partial caches,
CPU tensors, descriptor mismatch, truncation and changed source/inventory.
PPU completion/perplexity compile-link and the disabled-backend host build
pass. A Linux check using the actual `llama_mmap` implementation maps a
32 MiB file: prefetch RSS is 32768 KiB, on-demand RSS before access is 0 KiB,
and subsequent endpoint reads agree in five repetitions. This proves the
mapping mechanism, not model memory or latency savings on PPU.

The existing box smoke explicitly selects mmap and requires
`source_mmap=prefetch` on cold and `source_mmap=on-demand` on hit. Its timing
summary now includes maximum host RSS and filesystem-input counters. Run
the same script after the incremental llama.cpp rebuild; do not rebuild or
replace the Quactlize bundle. Earlier `lx3ae2` timings below predate this fix.
Disk-cache creation is tied to the first model load, not first prefill.
Subsequent processes still upload weights; same-process resident reuse does
neither repacking nor uploading again.

## Prior PPU rerun: cache latency improved, numerical comparison not run

`llama-kpack-smoke.lx3ae2.results.tgz`, source
`2fdd7bc4c25af90f603dc5fb35fe5ee77083c6b8`, reports
`KPACK_MODEL_CACHE_SMOKE PASS`. Raw logs agree with the timing summary;
all three processes exit 0, each reuses 62 graphs, and readiness passes
18 eager checks / 18 replays / six transitions / two missing-wait negatives.
Same model and PPU-ZW810 PCI `0000:08:00.0` as the prior run.

| Phase | Previous wall (s) | New wall (s) | New load (s) | New decode (tokens/s) |
| --- | ---: | ---: | ---: | ---: |
| baseline, GPU pack without cache | 9.64 | 9.58 | 4.92031 | 92.05 |
| cold, GPU pack and create cache | 89.27 | 22.04 | 4.35985 | 91.72 |
| hit, upload cache | 43.41 | 9.98 | 5.32570 | 91.66 |

Cold/hit wall time fell 75.31%/77.01% relative to the earlier run. Cold
user CPU fell 69.78 to 4.15 s; hit user CPU fell 69.94 to 3.89 s. The old
34.303-second hit verification interval is gone: allocation-to-cache-ready
is now 7.743 ms. The hit's subsequent ready-to-uploads-complete interval is
3.901168 s, including other tensor load work, not an isolated H2D duration.

The new hash-free manifest has no checksum fields, uses
`llama.kpack-cache` v1 / arrangement v2, and lists the same 120 grouped E256
tensors (80 Q4_K + 40 Q5_K), totaling 18.125 GiB. Cold uses 16 MiB pinned
staging, takes 11.17 s for copying/writing the planes and 3.90 s for flush/
publication, 15.07 s total background work. Context teardown at 6.081444 s
precedes publication at 19.633112 s by 13.551668 s; persistence still leaves
a process-exit tail, but generation does not wait for publication. Physical
copy/write/compute overlap is not established by these aggregate timestamps.

Hit uploads all 120 tensors with zero misses and zero GPU packs. However,
9.98 s is not faster than the same-run no-cache baseline's 9.58 s; caching
has not shown a startup win over the fast GPU producer. Maximum host RSS is
39.14 GiB on hit versus 21.17 GiB baseline; this is not device or pinned-memory
usage and remains a follow-up measurement, not a diagnosed leak. These are
single samples with ordered/warm filesystem state, not a controlled throughput
benchmark. Prompt timing covers only 11 tokens; decode covers 63 runs.

**Numerical comparison is explicitly NOT_RUN: EVAL_FILE_NOT_SET.** Neither
this operational PASS nor stable decode timing establishes PPL/KLD/logit
accuracy. Next: provide an evaluation corpus and run the existing numerical
entry. The compute route still uses the old FQ libraries; no new C++ heuristic/
JIT binding or broad-format/main admission is implied.

Archive SHA-256:
`2cfa240fc20c808ab42fb45b54f755056b7a0e96f150de1bac8d584f4e6b19e6`.
The uploaded archive contains logs/metadata only, not weight payloads.

## Cache latency change (2026-09-08, fork branch pushed)

Published commit: `35d625eb97503e814663c3d0b55f927d249a4ac7` on
`DrXuQian/llama.cpp`, branch `feat/kpack-gpu-cache`; the remote branch SHA
was verified after push. No upstream push or PR. Existing Quactlize libraries
remain unchanged; incrementally rebuild the llama.cpp targets on the box.

Follow-up runner failure on this revision: `check-cache line=151 rc=1` is the
timing-summary grep, not a model/cache gate. Completion emits
`common_perf_print:`, while the original filter only accepted
`llama_perf_context_print:` or cache messages. Baseline has no cache messages,
so grep returns 1 under errexit. The two-prefix filter is corrected and
published as `2fdd7bc4c25af90f603dc5fb35fe5ee77083c6b8` on the same fork
branch, with remote SHA verified. Regression reproduces the old failure and
executes the corrected full timing-summary block on the three existing logs,
plus the legacy prefix control; all pass. Only two script lines changed.
Reaching line 151 implies the preceding model/cache checks completed; it does
not provide new latency values, and the optional PPL/KLD stage has not run.
The exit trap already archives the raw logs and cache manifest; retain/upload
that archive. The user subsequently chose a fresh rerun; reuse the existing
build and libraries and let the script allocate a new results/cache directory.

The user explicitly requested removal of runtime content verification and a
pipelined writer. llama.cpp `feat/kpack-gpu-cache` now implements:

- Runtime hits use metadata/size/layout/bounds checks and direct uploads,
  with no whole-file or per-plane content hashing. Existing v3 bundles are
  readable on this unchecked path as well.
- Background writes use `llama.kpack-cache` v1 metadata and the unchanged
  K-pack planes. No checksums are emitted and no original GGUF payload is
  reread. The cache records source device/inode/mtime/ctime plus size and
  tensor identity. Copying/modifying that source can invalidate a local cache.
- Two 8 MiB pinned slots per device are allocated before inference. The
  writer prefetches the next D2H range before consuming the current slot,
  allowing copy to overlap CPU writes. At most two ranges are outstanding;
  prefetched ranges are drained on errors before their storage is released.
- No compute submission waits for D2H or disk. Teardown still joins the
  writer before freeing weights. fsync and no-replace publication remain.
  Progress reports bytes/tensors, total background seconds and flush seconds.

**Payload corruption is deliberately not detected by runtime cache loading.**
This is trusted local storage, not a replacement for the verified v3 offline
interchange format. The explicit v3 verifier remains available and rejects
corrupted payloads; it refuses to label a hash-free cache as verified.

Three host suites pass five repetitions each, including two submissions
before the first D2H completion, partial chunks, Q4/Q5 low/high/unit boundaries,
cross-tensor slot reuse, unchanged cached bytes, bounded staging, copy/async
completion/partial-write failures, cleanup, and no D2H on cache hit. Existing
background cancellation tests still pass. PPU SDK compile/link also passes for
`llama-completion`, `llama-perplexity` and the existing readiness target; the
box script passes shell syntax/help checks. These are host/mock ordering tests,
not a physical PPU overlap or model accuracy proof. GEMM and producer libraries
and the C ABI/plane layouts are unchanged. The subsequent PPU latency result
is recorded above; physical overlap profiling remains separate.

The box smoke entry additionally writes `timing-summary.log`. With `EVAL_FILE`
(a text corpus of at least 512 tokens), it uses the existing `llama-perplexity`
PPL/KLD tools for uncached K-pack vs cached K-pack and ordinary GPU weights vs
K-pack. `EVAL_BATCH=128` and `EVAL_BATCH=1` select separate prefill and
teacher-forced decode runs. Results go to `numerical-summary.log`; saved log
probabilities stay on the box. This is a compressed-probability comparison,
not a bitwise logits oracle or automatic numerical admission. Without a
corpus the script explicitly reports numerical comparison NOT_RUN. No CPU
model/packing reference or full kernel sweep is added. The old FQ DSOs still
own inference; selected-module C++ heuristic/JIT binding remains separate.

## Latest PPU model/cache result (2026-09-08, before latency change)

`llama-kpack-smoke.PmwxP9.results.tgz` passes the readiness preflight and all
three model/cache phases on source `5e632dc04`. This closes the scheduler
and ready-event blockers for this workload, not the broad final-runtime gate
above. The old FQ DSOs remain the compute route; the new C++ heuristic/JIT
binding has not been enabled by this test.

| Phase | Process wall (s) | Reported load (s) | Decode (tokens/s) |
| --- | ---: | ---: | ---: |
| baseline, no sidecar | 9.64 | 5.23037 | 90.64 |
| cold, create sidecar | 89.27 | 4.64655 | 91.58 |
| hit, reuse sidecar | 43.41 | 38.87221 | 92.38 |

All processes exit 0 and reuse 62 CUDA graphs. Baseline/cold each enqueue
120 GPU packs; hit uploads all 120 cached tensors, has zero resident misses,
and enqueues no GPU pack. Actual cached coverage is **80 Q4_K + 40 Q5_K
grouped expert tensors**, all E256, not all five formats or dense K-pack.
The cache is 19,461,570,560 bytes (18.125 GiB); the source is 22,016,023,168
bytes. The uploaded manifest is metadata only; no weights were uploaded.

The two long silent intervals are now identified from the log timestamps:
cold reaches context teardown at 6.326153 s and publishes at 86.887108 s
(80.560955 s later); hit prints allocation at 1.589105 s, verifies the cache
at 35.892575 s (34.303470 s later), and finishes uploads at 39.024970 s.
Cold's interval includes the remaining background snapshot/hash/write/fsync
work and teardown, not a measured 80-second D2H transfer. Hit performs
synchronous full-source/full-storage and per-record hash verification before
uploading. Per-component hash, copy and I/O times were not logged separately.

**Integration smoke PASS; cache startup/publication latency remains debt.**
This is one run per phase, with verbose logging and ordered file-cache state;
it is not a numerical/logit oracle or a controlled performance comparison.
Do not infer a decoding speedup or full shipping admission from it. The
subsequent user-approved change drops runtime payload checks and pipelines
the writer; its new latency must be measured. No GEMM DSO rebuild is needed.

Archive SHA-256:
`a4848fd25ebfc9858b1f7daa10c7a7f80d5dce2e678c04d0c3405cac87f2ef79`.
Detailed timeline and scope are in the integration audit. Earlier pending
statements below record the historical blocker sequence.

## Local loader audit and GPU conversion

### Scheduler buffer admission fix (2026-09-08)

The first model smoke run on `9f86a1340` aborts in `sched_reserve()` at
`ggml-backend.cpp:898`, assigning the preallocated `CUDA0_KPACK` weight leaf
whose op is `NONE`. `supports_op(NONE)` already succeeds; the CUDA device's
`supports_buft()` omitted the K-pack buffer type. The local fix adds that
type to the existing same-device predicate. It does not broaden permitted
operators, change artifact bytes, or touch any Quactlize DSO.

This is now reproduced on a separately built RTX 5090/CUDA 12.8 backend:
the old source aborts at the exact same scheduler line (rc=134); the fixed
source passes 21 reserve/allocation cases across Q2-Q6, dense/grouped,
tokens 1/16 and Q4 with 256 experts. Wrong-device and same-name impostor
buffer types, CPU consumption, DUP/VIEW and non-weight-position operands
remain rejected. The test uses the actual backend callback, K-pack buffer
factory/allocation and GGML scheduler. Only the external library inventory
is stubbed: neither pack nor GEMM is run, so this is not a PPU numeric or
performance gate. Four suites (scheduler, loader, buffer, sidecar) each pass
10 repetitions on that machine; the PPU build also compiles/links locally.

Regression: llama.cpp `tests/test-quactlize-scheduler.cpp`. Local evidence:
`/root/autodl-tmp/llama-kpack-scheduler-20260908.3z6rSg/`.
The fix and regression are published as `89e5b254dd39356bb9fc19ff0f8e46d19ec7f295`
on `DrXuQian/llama.cpp` branch `feat/kpack-gpu-cache`. Rebuild only llama.cpp's
affected backend/executable targets and resume the model smoke test. Keep
the existing producer and GEMM libraries. Full PPU/model validation remains
pending; do not treat the NVIDIA scheduler regression as model admission.
`tests/run-quactlize-cache-smoke.sh` now owns the box entry: pass `MODEL`,
`PPU_SDK`, `QUACTLIZE_PPU_BUNDLE`, `QUACTLIZE_PPU_PACK_LIBRARY`, and the
existing `BUILD_DIR=build-kpack-9f86a1340`. It verifies the unchanged library
hashes, incrementally builds `llama-completion`, runs baseline/cold/hit,
and archives only logs plus the cache manifest. Its syntax, missing-input
failure and refusal to be sourced were checked locally. Run it in a guarded
child bash so failure cannot exit the interactive Docker shell.

### Captured readiness dependency fix (2026-09-08)

PPU smoke `llama-kpack-smoke.TXBaN9` passed scheduler reserve, then aborted
in baseline (rc=134). The preceding log identifies `ggml_quactlize_wait_ready`
at `quactlize-buft.cuh:51`: `cudaStreamWaitEvent(stream, art.ready, 0)` reports
`dependency created on uncaptured work in another stream`, immediately after
CUDA graph warmup. Pack/upload records the immutable event outside inference
capture. Waiting with the default flag tries to import uncaptured work as a
graph dependency. This is not a GEMM numerical error, a D2H wait, or the
debugger's separate Python `math` import problem. Cold/hit have not run.

The helper now uses `cudaEventWaitExternal`, as defined by the
[CUDA stream API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html).
The PPU SDK declares the matching flag and wait-event node type. The graph
gets an explicit external-event wait node; eager execution still waits on
the GPU. No CPU event synchronization/query, graph disabling, D2H wait or
format change is introduced. Each ready event is recorded once and retained
with the weights for all graph replays.

`tests/test-quactlize-ready.cu` executes the actual helper with CUDA streams
and synthetic device bytes, without any pack/GEMM test double. The old
helper reproduces the exact error on RTX 5090 even after ready completes.
The candidate captures/replays with ready complete, pack deliberately held,
and D2H deliberately held. All nine replays read the expected bytes; capture
and submission return while pack remains pending, and compute completes
while D2H is still pending. Omitting the wait reads 4096 wrong bytes as an
expected negative. PPU compilation also passes; PPU execution/model/cache
admission remains pending. The fix and regression are published as
`6e3e4a7dcb38cb9957204d2162dad4d2a9a40298` on the same fork branch.
Final regression: readiness, scheduler, loader, delayed-buffer and sidecar
suites each pass ten consecutive NVIDIA runs (50 suite executions, including
90 readiness graph replays); three existing host suites also pass locally.
Evidence: `/root/autodl-tmp/llama-kpack-ready-20260908.7lXOMl/`.

### PPU eager-warmup flag rejection (2026-09-08)

Model artifact `llama-kpack-smoke.o0ghMa` still aborts on `6e3e4a7dc`, now
during initial eager warmup: flags=1 returns illegal state. This is a
separate state restriction, not evidence that PPU lacks external graph waits.
Local SDK 2.1.1 `hgapiStreamWaitEvent` explicitly rejects a nonzero flag on
a non-capturing stream with error 401 and diagnostic
`Illegal external flags for non-capturing stream`. Its capture-state query
reads the same stream field. The earlier regression exercised only capture
and missed this eager path.

The fix queries `cudaStreamIsCapturing` and uses flags=0 outside
capture, external wait inside capture. It queries capture state, not GPU
completion; no CPU completion wait, D2H dependency, graph disabling or GEMM
change is added. The real CUDA regression now covers eager-to-capture and
capture-to-eager, completed/pending pack and held D2H. Each run validates
18 eager checks and 18 graph replays. A test-only flag assertion enforces
PPU's stricter contract on NVIDIA before forwarding to the real runtime;
it rejects the old helper, but is not PPU execution emulation.
Five suites pass ten repetitions on NVIDIA; the modified PPU backend/test
compile and link, and three existing host suites pass locally.

The fix is published as `5e632dc04509ae5923168b0898e34a3a328ba6de` on the
same fork branch. The smoke runner now builds and runs
`test-quactlize-ready` before model loading, with a 60-second timeout. It
still reuses the existing build directory and unchanged Quactlize DSOs.
PPU readiness and this model/cache smoke are now passed by `PmwxP9`; broader
format/route numerical and final-runtime admission remain separate.
Evidence: `/root/autodl-tmp/llama-kpack-ready-state-20260908.Z23Xg7/`.

### Published integration

The llama.cpp integration is published to `DrXuQian/llama.cpp` on `feat/kpack-gpu-cache` at
`5e632dc04509ae5923168b0898e34a3a328ba6de`, based on develop `26955be8a`.
GPU intake/cache wiring was introduced by `9f86a1340`, scheduler admission
by `89e5b254d`, captured readiness by `6e3e4a7dc`, and capture-state flag
selection by `5e632dc04`.
The remote branch SHA was verified after push; no PR was created and develop is unchanged. It preserves Kpack buffer
ownership, per-format loading and dense/MoE boundary transforms.
It still calls the old FQ DSOs. This branch now connects the sidecar
reader/writer through `--kpack-cache DIR` and calls the separate
GPU producer during buffer intake, with no CPU conversion or inverse. A
missing/incompatible producer declines intake before adopting K-pack. The new grouped module ABI also
requires host rows, whereas llama.cpp currently keeps routing entirely on
device; do not add a per-token synchronization merely to reuse that ABI.

A separate five-format GPU producer is now implemented in
`quactlize/packing/`. It writes the same canonical low/high/units bytes
directly to caller-owned device spans, asynchronously, without changing the
GEMM kernel identity or recompiling its cached parents. Local compilation and
host byte proofs pass. The uploaded producer gate now passes **15/15 PPU
byte-exact cases**, including output guards, repeated calls and invalid
overlap/descriptor rejection. Reported payload hash matches the delivered
DSO. See the reviewed timing and scope in the audit below.
The llama.cpp buffer wiring now calls it; `PmwxP9` passes the bounded PPU
model/cache smoke above, with unresolved cache latency. The asynchronous sidecar queue/model loader now
have local host tests; the complete-identity C++ dispatcher remains pending.
See [the audit, local fixes and bounded copy/persistence plan](LLAMA_CPP_KPACK_INTEGRATION_AUDIT.md).

Precompiled producer: `prebuilt/ppu0010/kpack-pack-v1/libquactlize_ppu_pack.so`
(Git LFS, 126,504 bytes), with adjacent `manifest.json`. The box command in
the audit runs 15 byte-exact cases and separated transfer/pack timings with
no compilation. Archive `556e61f5...` completes this gate: 50.431 s wall time,
49.813 s CPU reference construction. At N=1024,K=5120, repeated pack medians
are 55.600-121.960 us; first-call intervals are 196.920-430.560 us. These are
Python/event-harness intervals, not a C++ model-load benchmark. Event-ordered
pinned backcopy passes; overlap with GEMM, disk writeback and llama.cpp are
still unvalidated. Do not repeat this completed conversion gate.

Local intake uses a reusable raw scratch buffer and whole-expert batches
(64 MiB target, at least one complete expert). It records `upload_done`
before the final pack and waits for that input event solely to release the
borrowed CPU source safely. The artifact has a distinct `ready` event after
the final pack. Dense and grouped consumers enqueue a GPU-side wait on
`ready`; neither waits for D2H or a disk writer.

`ggml_quactlize_copy_range_async` replaces the unused global synchronous
sink proc. It requires caller-owned pinned storage and a same-device completion event,
queues ready-wait/D2H/event-record on a separate nonblocking stream, and
returns without a CPU wait, allocation, inverse or file I/O. The writer must
own pinned storage through completion; device-buffer teardown drains pending
copies. There is no copy-to-compute dependency. Whole-tensor copies, when
needed by tests, use the same range API; there is no separate whole-tensor
backcopy path. Its declarations live in SDK-free `quactlize-sidecar.h`.

`src/llama-kpack-cache.{h,cpp}` is owned by the model and destroyed before its
tensor contexts/device buffers. Capture during loading queues metadata only;
one 8 MiB pinned slot and one completion event are allocated per participating
device before inference starts. After successful model loading, one worker
sorts jobs by GGUF index and streams packed device bytes to disk. It does not
borrow the loader's raw pointer, retain raw tensor data or run CPU conversion.
Compute depends only on packed-ready. Normal model teardown joins the writer
before releasing weights; cancellation drains the in-flight read and removes
only that writer's owned staging directory.

Source verification remains a separate **publication** stage required by
schema v3: one sequential GGUF read fills whole-file and tensor SHA-256s
together, with a 1 MiB buffer and before/after file-identity checks. The bound
file must also match the loader's already-open descriptor. Thus the
copy stage needs no original GGUF bytes; final cache publication still needs
the unchanged source file. This source read/hash and disk I/O run off the
compute submission path. Cache hits still verify source and stored hashes at
load time; they are not a hash-free fast path.

First integration scope: a single named GGUF file, including dense and grouped
resident tensors. Multi-file GGUF, FILE*/custom-data loading and nonresident
weights do not use this cache path; unsupported inputs retain ordinary loading.
Only the active backend's admitted resident tensors are captured, with other
source tensors explicitly listed as skipped. This is not a complete offline
export of every mathematically packable tensor. An existing invalid cache is
ignored for loading, never overwritten. Valid hits upload the verified planes
without GPU repacking. The public `llama_model_params` gained
`kpack_cache_path`: rebuild llama.cpp callers with the matching header.

The local PPU backend incrementally compiles; host loader, delayed-buffer
and sidecar tests pass. The buffer test keeps D2H pending while second intake
and the actual compute-ready helper proceed, covers source-buffer release,
all five formats and 1000-expert chunking, and rejects an injected synchronous
copy. The sidecar suite additionally tests delayed worker completion, bounded
chunks, sorting, duplicate capture, copy rejection, replaced source, cancellation,
direct cache-hit upload and descriptor mismatch. Streamed output matches the
original writer and passes the Python schema-v3 validator. This is host queue
simulation, not PPU overlap evidence. The llama.cpp source branch is published;
no new backend bundle is published.

Final local regression: loader, delayed-buffer and sidecar/model-cache suites
each pass 20 consecutive runs (60 successful suite executions). The injected
foreground-copy wait is rejected. Python interoperability also passes separately.

Local `llama` and PPU `llama-cli` compile/link with the target SDK; a CPU-only
`llama` build also passes. Final PPU executable linking needs both the SDK's
CUDA compatibility and native runtime library directories in the build
environment. This workstation cannot start that PPU runtime because its glibc
lacks `GLIBC_2.38`; model execution belongs on the supported Ubuntu 24.04 box.

The next box check uses `llama-completion` for a finite noninteractive run.
Build only that llama.cpp target, with the PPU SDK native and CUDA-compatible
runtime directories in `LD_LIBRARY_PATH`; do not rebuild the GEMM bundle.
For the existing Qwen3.5-35B-A3B Q4_K_M checkpoint, explicitly select
`-ot 'ffn_.*_exps=CUDA0_KPACK'`. The default CUDA buffer precedes extra buffers;
`--kpack-cache` alone does not select K-pack. Run the same prompt without a
cache, with a fresh cache path, then with that published cache. Require
`so-quactlize-kpack` route logs, a nonempty cache manifest, a positive cache
upload count with zero resident misses, and no `GPU pack queued` in the hit
run. Keep the generated text and timings for review. These are grouped
loader/cache smoke checks, not a numerical oracle, overlap proof, or admission
of the new C++ heuristic/JIT route. No full CPU conversion/reference is needed.
Normal process teardown waits for background publication; that wall time is
not kernel-launch latency. Return logs only, not the GGUF or cache weights.
Run the box command in a new `bash` process with a guarded caller, not an
unguarded subshell in a possibly `set -e` Docker shell. Install failure/exit
reporting before prerequisite checks and print every missing input. Check
the feature-branch commit before its files; reuse the per-commit llama.cpp
build directory after a failed preflight. Only result logs enter the archive.

The adjacent prebuilt manifest is a development build/verification receipt,
not a required llama.cpp runtime input. Keep it in development/CI archives;
model-sidecar format metadata and JIT cache compatibility identity have
separate runtime purposes.

After this completed gate, normal loading and subsequent performance gates
must not construct a full CPU reference or run a per-tensor CPU inverse
round-trip. Keep `reference/gguf_kpack.py` for independent format regression
and external integration work, not the loader or timing path. Preserve
descriptor/size checks, runtime error handling and sidecar/cache hashes;
these are not full CPU reference computation. CPU hash/disk work belongs to
the bounded persistence queue. GPU producer changes still need a separate
correctness regression, not an implicit oracle in every model load.

Final packaging follows the JIT direction: a small selector/module-loader
binding, the small conversion DSO, and individually cached compute modules.
Build the small libraries locally; do not require a new monolithic six-DSO
all-config build. The measured module cache stays valid unless its own
ABI/body/build contract changes. A missing parent still needs a one-time
target-SDK compilation. The C++ binding and model-level gate remain pending.

## Deployment target: heuristic plus cached JIT

Latest local delivery: [KPACK_JIT.md](KPACK_JIT.md). The additive
`quactlize_kpack_dispatch_enable_jit_v1` entry accepts copied absolute
Python/helper/SDK/cache paths before query. A selected module miss resolves
exactly that parent; prepared run performs no JIT/CPU search. The new
`kpack-jit-v1` package is about 1.1 MiB (dispatcher plus the unchanged tested
execution DSO), not a replacement for the remaining six intake libraries.
The existing 216-parent package remains intact.

The llama branch also removes per-weight `scale/zero/scale_ready` storage.
SF now expands on the compute stream on each invocation/replay using reusable
scratch. No wait on the background model-cache D2H queue is added. The
adapter test poisons metadata and changes its source each replay. The policy
exporter now emits `KPACK_PREFILL_POLICY_V2_PER_CALL` using measured
prepass+GEMM calls, and rejects V1 resident-only data. Both FQ and SF use the
same existing selected-parent owner; no manual default config is introduced.

Do not claim a new PPU/model result yet. The changed code has CPU contract
tests and SDK compilation; PPU JIT/replay/full-call gates are next. Automatic
decode stays FQ, Q8 remains on llama's own route, and Q6 output-head unknown
native coverage still uses the labelled existing FQ fallback. No original
GGUF disk-cache format or conversion bytes changed.

Current local implementation (2026-09-09): C++ selected-parent JIT and the
llama startup hook are connected, with source-bound cache receipts and a
new small `prebuilt/ppu0010/kpack-jit-v1` candidate. It is opt-in, not yet
PPU-admitted. [Commands and measured compilation cost](KPACK_JIT.md).
T01-T04's bounded local implementation/compilation is complete; device
replay/numerics, cold-JIT latency and removal of the six-library dependency
remain separate work. The dated gate descriptions below retain
their original scopes; their earlier "binding pending" notes are not a
statement that the present prebuilt binding is missing.

Decision updated after the real-shape review on 2026-09-07: default inference
uses a deterministic heuristic to select **one complete tactic**, then loads
its admitted prebuilt/cached module or compiles that missing parent. Do not
call the multi-candidate `Tuner.warmup` during normal model loading/inference.
The existing tuner remains an opt-in development tool, not the default policy.
This matches the inspected DeepGEMM-for-sail W4A16 grouped callers: `space=()`
selects one generated configuration and bypasses the multi-candidate benchmark.

The single-choice host implementation is now available as
`policies/kpack_zw810_heuristic_v1.{hpp,json}` plus
`runtime/dispatch.py::prepare_selected`. It is **not** the old blind Top-1
ranker renamed. Recent exact measurement overrides historical tactics;
otherwise a shared nearest-profile formula proposes one eligible tactic in
the same qtype/route/N/K/expert-count family. Predictions are opt-in and
explicitly unvalidated; unknown families decline. There is no rule tree or
online timing. Keep format facts in traits, legality in the inventory, and
grouped grid calculation tied to actual expert tiles.

Post-hoc calibration corrects nine historical choices that exceed the new
same-run pool's 5% band: all 90 recent choices are now within both 5% limits,
worst median regret 4.153523%, using 55 of the previously compiled 173 parents.
The full data still has 2,987 exact contexts / 247 parents; this does not prove
global compression to 55 or shrink the existing DSOs. See
[the selector contract and query](KPACK_HEURISTIC_V1.md).

The new query binds `kernel_source`/`sdk_digest` to the resident-module hash
scope (`record.identity.kernel/sdk`). Do not pass the older sweep hash scope.
Check the full compiler receipt (`kModuleContract`) and complete parent tuple
at module binding, then call its query/can_implement. Nonpersistent recipe
grid is zero; persistent grids use actual rows. Python `prepare_selected`
performs these guards and never profiles, JITs, or falls back internally.
The 90-context Python selected-dispatch device gate has passed; C++ module
binding and llama.cpp loader admission are still pending.
Unsupported requests need an admitted K-pack fallback or explicit decline;
only residual decisions need tests, not another full sweep.

The selected-dispatch device gate is now **PASS and locally replayed**:
`tools/run_kpack_selected_gate.py`, [box command and return bundle](KPACK_HEURISTIC_V1.md#run-the-selected-dispatch-gate-on-box).
It executes the 90 calibrated real-shape inputs with one selected tactic each,
using the existing 55-module cache. Default behavior cannot compile or tune.
It checks live selection/binding, the module's actual grid, official-GGUF
numerics, prepared-handle replay, direct same-kernel raw equality and a
zero-low negative. Three five-repeat validation timings do not modify choices.
Completed cases and clean per-weight exit receipts resume independently;
one failed weight does not stop the remaining weights. Uploaded archive
`174c58a6...` matches `05de781`: 90/90 cases, 270 positive checks, 90 finite
zero-low negatives, 90 replay/direct raw matches and 25 clean exits.
Maximum condition-scaled error is `2.76839e-4` against the unchanged `5e-3`
bound. All 55 parents came from cache; compilation and online tuning are zero.
Harness wall time is 301.958 seconds, not per-request JIT time. See
[the reviewed result](KPACK_HEURISTIC_V1.md#reviewed-selected-dispatch-device-result)
and [receipt](KPACK_SELECTED_GATE_RESULT.json). This is not the llama.cpp
buffer/loader gate or an any-M promise. Thirteen validation sample sets have
more than 5% spread; there is no same-run alternative-tactic comparison or
new global performance bound.

The earlier 90-context tuning experiment is complete: all numeric/cache checks pass;
89 selected medians are within 5% of that run's bounded pool and one Q6 SF
grouped choice missed the faster fourth candidate after exhausting its soft
budget. This does not certify a universal heuristic or all unseen shapes.
See [the reviewed results](KPACK_WARMUP.md#reviewed-real-shape-result).
Neither the six-library bundle nor llama.cpp wiring has changed. Compiled
module coverage can be reused; C++ deployment binding remains pending. Use
the SDK-free C++ selector and reusable module handles, not the Python query on
the hot path. Its measured setup overhead is not a C++ latency measurement.

## Frozen host runtime policy (2026-09-07)

`policies/kpack_zw810_runtime_v1.hpp` (namespace
`quactlize_kpack_runtime_v1`) and its matching JSON are the **historical frozen
selection policy**, now consumed by the deterministic selector above.
The runtime is an exact table lookup with binding/shape checks and source-owned
grid resolution; no scorer, learned coefficients or profiling runs on this path.

It covers 2,982 measured exact inputs across Q2–Q6 and FQ/SF dense/grouped.
2,889 entries satisfy both 5% limits in all recorded epochs. The other 93 are
explicit measured exceptions, not numerical failures and not certified 5%
choices. Latest-epoch evidence meets both limits for 2,968 entries. Unknown
M/N/K or grouped row vectors return `FallbackRequired`; the consumer retains
responsibility for an admitted K-pack fallback. Actual grouped row vectors,
device/CUs, mapping and kernel/SDK identity are required.

The last planned 212-request box challenge is complete: all 636 raw logs replay,
no numerical/launch failures, 307.515 seconds. SDK-free Python/C++ parity covers
11,928 queries. The policy references 247 parents / 509 runtime recipes; those
are identities, not newly linked function pointers. **The six DSOs remain
unchanged.** Bind full AP/delivery/S/scheduler/grid identity and perform final
device admission before enabling it in llama.cpp. Do not reuse old config v3/v4
names as if they encoded these fields. See [the v1 contract](KPACK_RUNTIME_V1.md).

## Opt-in warmup module path (2026-09-07)

An initialization-time alternative is implemented in `quactlize/runtime/`:
bounded measured-family candidates, CPU-only JIT image cache, explicit timing
cache and device-pointer C modules for all five formats and four FQ/SF
dense/grouped routes. This is **not enabled in the six-library bundle or in
llama.cpp**. See [K-pack warmup](KPACK_WARMUP.md) for the small box gate and
application sequence.

The small gate is now **PASS**: uploaded archive `30a86507...` replays against
the `e47613f` source contract, with 50 contexts, 190 measured candidates, 250
positive checks, 50 detected zero-low negatives and ten changed-router cache
replays. Warmup medians are 1.265–1.838 ms by route, maximum 13.490 ms;
compile-plus-harness wall time is 254.751 seconds. The ten safe exclusions
are SF-dense TM8 at M=9. This is `N=256,K=512` small-context evidence, not
real-shape performance admission or permission to replace the old six DSOs.
See [the exact receipt](KPACK_WARMUP_GATE_RESULT.json).

`quactlize/runtime/abi.h` carries the full compiled parent and algorithm/S/grid
identity. Prepared handles launch existing collectives; no GGUF conversion,
hidden compilation or profiling occurs in `run`. Grouped queries need actual
expert rows and recompute the grid. A reused workload bucket is a performance
hint, not an any-M promise. The caller still owns allocation, stream, SF
metadata preparation and an admitted miss path. This interface is separate
from the old `config_name` exports; do not reinterpret their names.

Next: bind the admitted selected-dispatch module path in C++, close loader
any-M/miss handling, then validate llama.cpp cached execution and any explicit
initialization-time compilation. Offline sidecar bytes and
canonical arrangement exports remain unchanged. The completed small gate
does not need repeating; no new full Cartesian sweep is requested.

The completed gate is `tools/run_kpack_warmup_real.py`, 90 real-shape
contexts / 173 distinct parents / at most fifteen candidates per context, across
all five formats and four routes. It compares the budgeted choice with the
historical incumbent and bounded-pool best **in the same run**, and reports
warmup cost plus changed-M/router cache behavior. Eighty-five incumbents have
exact historical evidence; five M=3072 controls are labelled transfers. This
gate has passed its numerical checks on box; selection has the explicit
performance exceptions above, and does not admit deployment by itself. See
[the command and scope](KPACK_WARMUP.md#next-gate-bounded-real-shape-selection).
No `.so` ABI, sidecar format, mainloop or six-library bundle changes are part
of this step. The next deployment work uses the heuristic direction above,
not automatic online tuning or precompilation of the entire candidate union.

## Earlier host policy experiments

[K-pack measured policy](KPACK_POLICY.md) supplies an SDK-free C++17 selector
and matching JSON for all five formats and FQ/SF dense/grouped. It serves 2,717
observed requests with a maximum 4.983% training round regret. The compact
selection merges near-equal choices: 788 rule leaves / 219 parents instead of
999 / 311, with a mean median-time increase of 0.7012% against the original
selected winners. Current files are `policies/kpack_zw810_compact.{json,hpp}`.
The 315-request merge/boundary run is now raw-replayed: all three rounds
completed without numerical or launch failures. New-M predictions meet both
5% limits in 85/110 cases; grouped proposals exposed 76 stale fixed-grid
identities. This does not admit universal interpolation or a new selector.
See [validation and hybrid-heuristic review](KPACK_HEURISTIC_REVIEW.md).
The original model still has forty-five
requests explicitly blocked; unmeasured M/router queries are proposals,
not admission. The selector is under `policies/`, not wired into the six DSOs.

The subsequent host tactic prototype is now implemented and locally calibrated:
`policies/kpack_zw810_tactics.json` plus `tools/kpack_tactic_model.py`. It has
2,770 exact-context measured cache entries and a shortlist path with at most
five parents / three runtime proposals per parent. Grouped queries require
the **actual expert row vector**, and grid policies resolve its exact CTA
count. Do not pass total/max-row aggregates as an equivalent router.
The blind Top-1 model was not performance-admitted; the 212 targeted device
requests have completed and fed the frozen table above. The model returns required device/source/SDK/
mapping bindings, not a launchable old config string. This is not a new C ABI,
any-M support promise or replacement six-library bundle. See the current
[box command and query entry](KPACK_POLICY.md#tactic-shortlist-prototype-and-next-box-run).

**No new `.so` is delivered by this policy fit.** The bundle below still has
its previous selector/inventory. Its config v3/v4 names cannot fully represent
AP, delivery-N, scheduler/grid and Split-K. Do not translate a returned benchmark
symbol into an old `config_name`, or use this partial policy to replace the
existing any-M buffer admission promise. Full-identity runtime binding,
selected-parent builds and device replay remain pending. Sidecar bytes,
canonical mapping IDs and the existing C exports are unchanged.

## Runtime libraries

The published PPU0010 six-library bundle was built from clean source commit
`2826cf12451e02ca4590f7a44682b57d2098bfb9`. Its immutable artifact commit is
`d5bf726dddc8c685a4eb766e7ec6cc303427501b`, whose only parent is that source
commit. The durable Git authority is:

```text
origin/artifacts/ppu0010/2826cf1-runtime6-46fc3096e1a1
prebuilt/ppu0010/2826cf1/runtime6-46fc3096e1a1/bundle
```

The bundle manifest SHA-256 is
`46fc3096e1a14b712ad5d7a50de096d2a973ad5826aa3ffe6a6764d1fc12180d`.
Its manifest uses the separate `quactlize.ppu-runtime-bundle` schema v1; never
parse it as the persistent sidecar schema v3. No workstation cache path is an
authority.

Use two independent worktrees: a detached artifact worktree owns only bundle
hydration, while a clean develop worktree owns the verifier/oracle/gate
runners. First create the artifact worktree and hydrate its six LFS objects:

```bash
SOURCE_COMMIT=2826cf12451e02ca4590f7a44682b57d2098bfb9
ARTIFACT_COMMIT=d5bf726dddc8c685a4eb766e7ec6cc303427501b
ARTIFACT_BRANCH=artifacts/ppu0010/2826cf1-runtime6-46fc3096e1a1
ARTIFACT_REL=prebuilt/ppu0010/2826cf1/runtime6-46fc3096e1a1
ARTIFACT_WORKTREE=/workspace/quactlize-runtime-artifact-2826cf1-46fc3096e1a1
RUNNER_COMMIT=1d579bc941828ce4b1788d2970f4b454dc3a81f8
RUNNER_WORKTREE=/workspace/quactlize-runner-1d579bc

git fetch origin \
  "refs/heads/${ARTIFACT_BRANCH}:refs/remotes/origin/${ARTIFACT_BRANCH}"
git fetch origin \
  refs/heads/develop:refs/remotes/origin/develop
test "$(git rev-parse "origin/${ARTIFACT_BRANCH}")" = "${ARTIFACT_COMMIT}"
test "$(git rev-list --parents -n 1 "${ARTIFACT_COMMIT}")" = \
  "${ARTIFACT_COMMIT} ${SOURCE_COMMIT}"
git worktree add --detach "${ARTIFACT_WORKTREE}" "${ARTIFACT_COMMIT}"
git -C "${ARTIFACT_WORKTREE}" lfs pull \
  --include="${ARTIFACT_REL}/bundle/*.so" \
  --exclude=""
git worktree add --detach "${RUNNER_WORKTREE}" "${RUNNER_COMMIT}"
git -C "${RUNNER_WORKTREE}" submodule update --init --recursive

BUNDLE="${ARTIFACT_WORKTREE}/${ARTIFACT_REL}/bundle"
test "$(sha256sum "${BUNDLE}/manifest.json" | awk '{print $1}')" = \
  46fc3096e1a14b712ad5d7a50de096d2a973ad5826aa3ffe6a6764d1fc12180d
```

The artifact carries the strict verifier used for admission. It is pinned by
both content and Git identity:

```text
SHA-256  8b552c33a8b9b34e184f6410720ae7150d562b57f47c8e604719b82cb324ec47
Git blob 22516983f4659e712fc04a0278e9e63bd1ef3b14
```

Run it from the artifact worktree, then run the source-pinned selected-config
oracle from the clean develop checkout:

```bash
ARTIFACT_REL=prebuilt/ppu0010/2826cf1/runtime6-46fc3096e1a1
ARTIFACT_WORKTREE=/workspace/quactlize-runtime-artifact-2826cf1-46fc3096e1a1
RUNNER_WORKTREE=/workspace/quactlize-runner-1d579bc
BUNDLE="${ARTIFACT_WORKTREE}/${ARTIFACT_REL}/bundle"
cd "${RUNNER_WORKTREE}"
source "${PPU_SDK:?set PPU_SDK to the admitted SDK root}/envsetup.sh"

test "$(sha256sum "${ARTIFACT_WORKTREE}/${ARTIFACT_REL}/verify_bundle.py" | awk '{print $1}')" = \
  8b552c33a8b9b34e184f6410720ae7150d562b57f47c8e604719b82cb324ec47
test "$(git -C "${ARTIFACT_WORKTREE}" hash-object "${ARTIFACT_REL}/verify_bundle.py")" = \
  22516983f4659e712fc04a0278e9e63bd1ef3b14
python3 "${ARTIFACT_WORKTREE}/${ARTIFACT_REL}/verify_bundle.py" "${BUNDLE}" \
  --ppu-sdk "${PPU_SDK:?set PPU_SDK to the admitted SDK root}"

test "$(sha256sum tools/verify_kquant_selected_config.py | awk '{print $1}')" = \
  5bb69a089074d9c89541ebca7d6106688b7b4c3c56da5dcda387afd4b881ec44
test "$(git hash-object tools/verify_kquant_selected_config.py)" = \
  8774480440d67d31dca41175ba0667701989d381
python3 tools/verify_kquant_selected_config.py "${BUNDLE}"
```

Do not use `--manifest-only`: it omits exported-symbol and embedded-image
inspection. Setting `QUACTLIZE_PPU_BUNDLE` only selects a directory and does
not perform this verification automatically.

The published libraries and the pinned headers include these host-only
selected-config exports:

```text
quactlize_ppu_dense_fully_quantized_selected_config_for_arrangement_v2
quactlize_ppu_grouped_fully_quantized_selected_config_for_arrangement_v2
```

Format selection is mandatory:

| GGML qtype | format | packed format | library |
|---:|---|---:|---|
| 10 | Q2_K, low2 Pack8 | FMT2 | `libquactlize_ppu_fmt2.so` |
| 11 | Q3_K, low2 Pack8 + high1 Pack16 | FMT3 | `libquactlize_ppu_fmt3.so` |
| 12 | Q4_K, low4 K-pack4 | FMT0 | `libquactlize_ppu_fmt0.so` |
| 13 | Q5_K, low4 Pack4 + high1 Pack16 | FMT1 | `libquactlize_ppu_fmt1.so` |
| 14 | Q6_K, low4 Pack4 + high2 Pack8 | FMT4 | `libquactlize_ppu_fmt4.so` |

`libquactlize_ppu.so` is the default Q4 ScaleFirst library. It is not a
fully-quantized FMT library. After `dlopen`, call
`quactlize_ppu_build_packed_format_v1()` and require the exact FMT value before
using any arrangement-aware entry.

The minimal llama.cpp integration documented here uses the qtype-selected FMT
fully-quantized path for every M, including Q4 prefill. The default ScaleFirst
library is therefore not required by this first integration. Quactlize's tuned
Q4 dispatcher uses ScaleFirst at M>=64. Develop commit `d27fee0` added a public
asynchronous packed-units-to-FP16 metadata prepass, but that commit is newer
than this exact A01 bundle. Do not resolve it from these DSOs; A07 must rebuild
and re-admit all six libraries before llama.cpp enables that optional route.

The product host floor is Ubuntu 24.04 with the bundle's admitted PPU SDK
2.1.1-a5c56e runtime. Load the SDK wrapper first with
`RTLD_NOW | RTLD_GLOBAL`, then load every Quactlize DSO with
`RTLD_NOW | RTLD_LOCAL`:

```text
${PPU_SDK}/lib/libhggc_wrapper.so -> RTLD_NOW | RTLD_GLOBAL
absolute path to selected FMT DSO -> RTLD_NOW | RTLD_LOCAL
```

Resolve every Quactlize function with `dlsym` on that format library's returned
handle. The six libraries intentionally export the same C symbol names, so a
process-global lookup can let the first loaded format answer for all later
formats. The default library reports build identity `-1`; the five
format-selected libraries report FMT0 through FMT4. A private host-loader shim
or a developer-only loader environment variable is not part of the deployment
contract.

Before converting or uploading a tensor, obtain the descriptor from the
selected library rather than reconstructing it in llama.cpp:

```c
quactlize_ppu_placed_arrangement_v2 arrangement;
int rc = quactlize_ppu_canonical_arrangement_v2(ggml_qtype, &arrangement);
```

This host-only query succeeds only when `ggml_qtype` is the qtype owned by the
loaded FMT library. The default/ScaleFirst library and every other qtype fail;
on every non-null failure the output structure is all zero bytes. Treat every
nonzero result as a route decline, never as permission to retain a descriptor
returned by another library.

The published bundle exports the arrangement-v2 dense/grouped device APIs, the
canonical descriptor query, complete host producer/inverse, packed-units ABI,
selected-config ABI, and ScaleFirst validity seam required by the strict
verifier. In a fresh LFS clone, the host-only ABI suite passed 26 cases over all
five formats, including dense/grouped exact round trips and fail-closed
negatives. The frozen strict verifier and selected-config oracle also passed.
PPU numerical execution remains `PENDING`; no performance claim is attached to
this bundle until its device gate passes.

PPU runtime setup uses the SDK's public environment script. The exact install
root is deployment-owned:

```bash
source "${PPU_SDK:?set PPU_SDK to the admitted SDK root}/envsetup.sh"
```

## Public headers

The runtime-bundle directory intentionally contains only its manifest and six
shared libraries. Pin the public headers from the exact source commit recorded
by that runtime manifest; do not duplicate the structs or function signatures
in llama.cpp:

```text
quactlize/include/quactlize_ppu_config.h
quactlize/include/quactlize_ppu_device.h
quactlize/include/quactlize_ppu_packed.h
```

They are included in both Python package manifests at source commit
`2826cf12451e02ca4590f7a44682b57d2098bfb9`. Pin those exact three headers to
the six published libraries. Do not substitute headers from another source
revision, even when a structure name or export name appears unchanged.
`quactlize_ppu_config.h` at this commit contains config v3/v4 and both
selected-config declarations.

`quactlize_ppu_placed_arrangement_v2` is 40 bytes on the target ABI:

```c
typedef struct {
    int32_t  version;             // 2
    int32_t  layout;
    int32_t  bits;
    int32_t  high_bits;
    int32_t  artifact_tile_k;     // 0 for canonical K-pack
    int32_t  transport_tile_k;
    int32_t  group_size;
    int32_t  reserved;            // 0
    uint64_t mapping_id;
} quactlize_ppu_placed_arrangement_v2;
```

Canonical mapping IDs:

```text
Q4 K-pack4:       layout=1 mapping_id=0x51344b5034540001
Q2/Q3/Q5/Q6:     layout=2 mapping_id=0x514b504b54000001
```

## Fully-quantized device execution

### Selected-config observability

The published bundle exposes the exact tactic chosen by the same host policy
used by workspace queries and launches. These calls are host-only: they create
no PPU context and enqueue no work.

```c
quactlize_ppu_dense_fully_quantized_selected_config_for_arrangement_v2
quactlize_ppu_grouped_fully_quantized_selected_config_for_arrangement_v2
```

For dense K-pack Q2/Q3/Q5/Q6, a null or empty requested name selects an exact
measured `(qtype,M,N,K)` point when present, then falls back to the established
compiled M-dependent default. Q4 layout 1 has its own shape policy over
`(M,N,K)` and returns its fixed Split-K value explicitly in the v4 record. A
known nonempty name is an explicit override. An unknown nonempty name fails and
clears the output record. Grouped currently exposes only
explicit/compiled-default selection because its measured sweep depends on the
full per-expert row distribution, which is not represented by the public
aggregate arguments.

The strict verifier and selected-config oracle for this bundle are the two
host-only gates shown in the runtime acquisition section. The dense result is
`quactlize_ppu_config_v4`, including `split_k_slices`; the grouped result is
`quactlize_ppu_config_v3`. Success is exactly `1`. Failure is `0` and clears
the complete output record. Do not bind the dense export to the older v3
record or infer split-K from the config name.
On success, `name` points to storage owned by the loaded DSO. Keep that DSO
loaded while using the pointer, or copy the string into loader-owned storage.

Device admission uses the already-hydrated artifact worktree and a separate,
clean develop checkout. The runner must be tracked and have SHA-256
`09110ba66b0d455ed91d44c3b2c0c648c84c923dacd38acbdf0b060412bf8297`.
It is pinned at commit `1d579bc941828ce4b1788d2970f4b454dc3a81f8`
and Git blob `56264dcc327ae5e20d2c5cd49e3f1592e92b929d`.
Its source-authority checks require the runtime implementation and submodule
commits to remain identical to bundle source `2826cf1`; documentation-only
develop commits are allowed. On an Ubuntu 24.04 PPU box:

```bash
RUNNER_WORKTREE=/workspace/quactlize-runner-1d579bc
ARTIFACT_REL=prebuilt/ppu0010/2826cf1/runtime6-46fc3096e1a1
ARTIFACT_WORKTREE=/workspace/quactlize-runtime-artifact-2826cf1-46fc3096e1a1
BUNDLE="${ARTIFACT_WORKTREE}/${ARTIFACT_REL}/bundle"
cd "${RUNNER_WORKTREE}"

source "${PPU_SDK:?set PPU_SDK to the admitted SDK root}/envsetup.sh"
python3 -c 'import importlib.metadata as m; assert m.version("gguf") == "0.19.0"'
test "$(git rev-parse HEAD)" = 1d579bc941828ce4b1788d2970f4b454dc3a81f8
test "$(sha256sum tools/run_prebuilt_ppu_box_gate.py | awk '{print $1}')" = \
  09110ba66b0d455ed91d44c3b2c0c648c84c923dacd38acbdf0b060412bf8297
test "$(git hash-object tools/run_prebuilt_ppu_box_gate.py)" = \
  56264dcc327ae5e20d2c5cd49e3f1592e92b929d
CUDA_VISIBLE_DEVICES=0 \
  python3 tools/run_prebuilt_ppu_box_gate.py "${BUNDLE}" \
    --ppu-sdk "${PPU_SDK:?set PPU_SDK to the admitted SDK root}" \
    --q4-correctness-repeats 8192 \
    --output /workspace/quactlize-a01-device-result-2826cf1
```

`CUDA_VISIBLE_DEVICES` must be one numeric ordinal and the runtime must expose
exactly one device. This gate compiles and links nothing: it loads the SDK
wrapper globally, the six prebuilt Quactlize DSOs locally, and executes the
public ctypes ABI. It covers all five formats at dense measured shape
`M=1,N=1024,K=5120` and grouped empty-expert shape
`rows=[2,0,3,1],N=256,K=512`, with official `gguf==0.19.0` plus independent
NumPy FP64 oracles and planted faults. It writes a new evidence directory and
refuses to overwrite one. Device status remains `PENDING` until this exact gate
reports `PASS` for the published files.

Dense fully-quantized decode/prefill:

```c
quactlize_ppu_dense_fully_quantized_workspace_bytes_for_arrangement_v2
quactlize_ppu_dense_fully_quantized_dev_for_arrangement_v2
```

Grouped/MoE:

```c
quactlize_ppu_grouped_fully_quantized_workspace_bytes_for_arrangement_v2
quactlize_ppu_grouped_fully_quantized_dev_for_arrangement_v2
```

Contracts:

- `act`, `low`, `high`, `units`, `offsets`, `out`, and workspace are device
  pointers.
- Activation and output are FP16. llama.cpp must convert F32 to FP16 before the
  call and FP16 back to F32 afterward on the same stream.
- The workspace query returns `-1` when the qtype, shape, or arrangement is not
  admitted.
- A successful launch only enqueues work on the supplied stream.
- Pass `config_name == nullptr` unless the name came from the matching
  arrangement-aware inventory and passed that route's validity predicate. An
  unknown nonempty name is a hard decline, not a request for fallback.
- Keep activations, weights, outputs, offsets, and workspace alive until the
  caller's stream has completed the launch.
- Q2/Q4 require `high == nullptr`; Q3/Q5/Q6 require a real high-plane pointer.
  Although a Q2/Q4 schema-v3 manifest records the empty high span with its
  canonical offset and shape `[0]`, do not turn that offset into a non-null
  API pointer.
- Grouped `offsets` is a nondecreasing device `int32_t[experts+1]` with
  `offsets[0] == 0`, `offsets[experts] == total_rows`, and every expert extent
  at most `max_rows`. Require positive `experts`, `total_rows`, and `max_rows`,
  with `max_rows <= total_rows`.
- A K-pack buffer may never fall back to a raw GGUF reader after route decline.

All resident artifacts, workspace queries, and launches require positive N and
K multiples of 256; Q3_K/Q6_K additionally require `K % 512 == 0`. Dense also
requires positive M. Do not mirror the remaining tactic-specific shape policy
in llama.cpp: the workspace query and config-valid predicate are the admission
authority.

The old raw vecdot device seam is
`quactlize_ppu_vecdot_dense_dev_v1(..., void * stream)`. Do not pass device
pointers to the host-pointer `quactlize_ppu_vecdot_dense` symbol.

## Persistent sidecar schema v3

This sidecar schema is independent of the runtime-library bundle schema. Its
identity fields are:

```text
schema = "quactlize.kquant-kpack.bundle"
schema_version = 3
arrangement_version = 2
```

The top-level object contains exactly these keys:

```text
schema, schema_version, arrangement_version, model, selection,
source, storage, tensors, skipped
```

`model` records the nonempty source name supplied to the packer. Cache
authority comes from `source`, not from that path string:

```json
"source": {
  "format": "gguf",
  "size_bytes": 10485760,
  "sha256": "..."
}
```

Production selection is also explicit:

```json
"selection": {
  "layout_policy": "production-kpack-only",
  "packable_total": 42,
  "packed": 42,
  "skipped": 7
}
```

`tensors` contains every packable Q2_K/Q3_K/Q4_K/Q5_K/Q6_K dense or grouped
weight. `skipped` is the exact inventory of non-packable tensors, with records
containing only nonempty string `name`, `type_name`, and `reason` values. A
valid product sidecar has a nonempty `tensors` list, every selection count is a
nonnegative integer, `packable_total == packed == len(tensors)`, and
`selection.skipped == len(skipped)`. It never silently omits a packable weight.
Packed and skipped names are each unique and the two sets are disjoint.

The packer writes a sidecar directory, not a rewritten GGUF:

```text
SIDE_CAR/
  manifest.json
  weights.bin
```

`weights.bin` is headerless. Every packed tensor owns one 128-byte-aligned
resident region. The region contains canonical `low`, `high`, and `units`
spans, each at a 128-byte-aligned relative offset. For all admitted formats and
geometries the region is byte-neutral:

```text
region.size_bytes == original GGUF tensor nbytes
```

Top-level storage authority:

```json
"storage": {
  "file": "weights.bin",
  "size_bytes": 9437184,
  "alignment_bytes": 128,
  "sha256": "..."
}
```

Each packed-tensor record contains:

```json
{
  "name": "blk.0.attn_q.weight",
  "ggml_type": 12,
  "type_name": "Q4_K",
  "route_class": "dense",
  "layout_name": "q4-kpack4",
  "plane_packs": { "low": 4, "high": 0 },
  "rank": 2,
  "n": 4096,
  "k": 4096,
  "experts": null,
  "arrangement_version": 2,
  "arrangement": {
    "layout": 1,
    "bits": 4,
    "high_bits": 0,
    "artifact_tile_k": 0,
    "transport_tile_k": 64,
    "group_size": 32,
    "reserved": 0,
    "mapping_id": 5851384623708504065
  },
  "source_tensor": {
    "index": 123,
    "data_offset": 456789,
    "size_bytes": 9437184,
    "sha256": "...",
    "binding_sha256": "..."
  },
  "region": { "offset_bytes": 0, "size_bytes": 9437184 },
  "spans": {
    "low":   { "offset_bytes": 0, "size_bytes": 8388608, "shape": [1,4096,2048], "sha256": "..." },
    "high":  { "offset_bytes": 8388608, "size_bytes": 0, "shape": [0], "sha256": "..." },
    "units": { "offset_bytes": 8388608, "size_bytes": 1048576, "shape": [16,4096,16], "sha256": "..." }
  }
}
```

This is a concrete Q4 dense shape example. Other qtypes and shapes have
different canonical descriptor fields and span extents; compute their expected
values independently as specified below.

The JSON `arrangement` deliberately omits `version`. Set the C structure's
`version` from the tensor record's `arrangement_version`; both the top-level and
tensor-level arrangement versions must equal 2. Copy the remaining eight fields
exactly. Parse `mapping_id`, offsets, extents, and byte counts as lossless
integers, never through an IEEE-754 double.

Read candidate values from the manifest, then independently compute the
expected layout name, plane packs, and span shapes from qtype, route, N, K, and
experts. Query the selected FMT library for the expected complete arrangement
and require exact equality before upload. Production sidecars admit only Q4
layout 1 and Q2/Q3/Q5/Q6 layout 2; reject Xplane layout 0 and experimental
direct layout 3 even though the public enum still reserves those values.

Let E=1 for dense and E=`experts` for grouped. The canonical span shapes are:

```text
low:   [E, N, K*low_bits/8]
high:  [0] when high_bits==0, otherwise [E, N, K*high_bits/8]
units: [K/(256*S), N, U] for dense
       [E, K/(256*S), N, U] for grouped
```

The `(S,U)` packed-unit pairs are Q2 `(1,20)`, Q3 `(2,28)`, Q4 `(1,16)`,
Q5 `(1,16)`, and Q6 `(2,36)`. For every span, require `size_bytes` to equal the
exact product of its canonical shape (zero for `[0]`).

Dense records require `(route_class, rank, experts) == ("dense", 2, null)` and
source GGUF shape `[K,N]` with a `MUL_MAT` consumer. Grouped records require
`("grouped", 3, E>0)`, source shape `[K,N,E]`, and a `MUL_MAT_ID` consumer.
Embeddings, SSM weights, unknown roles, and every `skipped` record remain on
their original route.

`source_tensor.binding_sha256` is SHA-256 of compact UTF-8 JSON for this array:

```text
[name, ggml_type, rank, n, k, experts,
 source_index, source_data_offset, source_size_bytes, source_sha256]
```

The byte encoding is exactly:

```python
json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
```

There is no Unicode normalization and no trailing newline. Consequently a
non-ASCII tensor name is represented by JSON `\uXXXX` escapes (including a
surrogate pair when required), not by its literal UTF-8 bytes. Dense `experts`
is JSON `null`.

`source_tensor.data_offset` is an absolute byte offset from the start of the
source GGUF file. In GGUF terms its value is:

```text
tensor_data_start = align_up(end_of_metadata_and_tensor_info,
                             general.alignment or 32)
source_tensor.data_offset = tensor_data_start + tensor_info.offset
```

Do not store or compare the tensor-info-relative `tensor_info.offset` by itself.
In addition to checking the binding digest, llama.cpp must compare name,
ordinal, this absolute data offset, raw size, qtype and shape with the GGUF
inventory it already parsed.

Cache reuse requires all of the following:

1. Source GGUF size and whole-file SHA-256 match `manifest.source`.
2. Tensor inventory and each source byte-range digest match `source_tensor`.
3. Manifest has no duplicate keys or unlisted fields/files.
4. `weights.bin` whole-file and per-span hashes match.
5. Regions/spans are ordered, non-overlapping, canonically aligned and
   byte-neutral.
6. The complete arrangement-v2 is canonical for the qtype.

Regions start at offset zero in manifest order and continuously cover the
entire storage file. Within each region, low/high/units appear in that order at
`align_up(previous_end, 128)`; all alignment padding is zero and there is no
unlisted gap or tail. Source tensor indices are strictly increasing, and their
source byte ranges are ordered and disjoint.

The sidecar root must be a real, nonsymlink directory containing exactly
`manifest.json` and `weights.bin`. Open both children with `O_NOFOLLOW`,
validate and consume each from one file descriptor, and reject row-split
loading. Layer split is allowed because a complete tensor remains on one
device.

Schema v3 is deliberately a regular, single-file, little-endian GGUF contract:
`manifest.source` binds one source file, every `source_tensor.data_offset` is
absolute within that same file, and `storage.file` names one `weights.bin`. It
cannot represent a split/sharded GGUF source because there is no source-file
index per tensor. A future multi-file schema must add per-file authorities and
bind every tensor to one of them; concatenating shard-relative offsets or
treating them as offsets in the first file is invalid. Runtime layer split does
not relax this source-format limitation.

## Offline production and reuse

Create the persistent sidecar once:

```bash
QUACTLIZE_PPU_BUNDLE=/path/to/six-library/bundle \
  quactlize-pack-gguf MODEL.gguf MODEL.gguf.kpack
```

Subsequent loads validate and upload the saved regions; they do not repeat
placement. Do not overwrite or relabel the source GGUF.

The manifest-pinned `2826cf1` source declares and implements the loader-facing
complete host producer/inverse:

```c
quactlize_ppu_prepare_fully_quantized_for_arrangement_v2
quactlize_ppu_recover_fully_quantized_for_arrangement_v2
```

These host-only entries take official GGUF blocks and separate low/high/units
pointers. Allocation of the units channel is governed by
`quactlize_ppu_units_bytes`; the standalone producers
`quactlize_ppu_prepare_units` and `quactlize_ppu_prepare_units_grouped` remain
part of the verified public bundle contract. The complete producer requires the
matching FMT library and complete canonical arrangement, and its inverse
supplies a byte-exact admission check. The legacy
`quactlize_ppu_prepare_fully_quantized_v1` produces Xplane and must not populate
the K-pack cache.

For in-process first-load conversion, recovery and exact comparison with the
source GGUF tensor are mandatory before publication. This is a one-time cache
admission check; later loads use the manifest and span hashes instead of
repacking or repeating the inverse.

The complete producer shares the resident geometry above and requires positive
`experts` (use 1 for dense). Every nonempty input and output byte range must be
pairwise disjoint; in-place and partial aliasing are rejected before any write.
A format-selected library accepts only its own qtype. Any decline must leave the
tensor on its non-K-pack route; never label partial, Xplane, or unchanged bytes
as K-pack.

Use checked arithmetic for every allocation. Let E=1 for dense and
E=`experts` for grouped: low owns `E*N*K*bits/8` bytes; high is null exactly
when `high_bits==0`, otherwise it owns `E*N*K*high_bits/8` bytes; and units owns
`E*quactlize_ppu_units_bytes(N,K,qtype)` bytes after requiring that query to
return a nonnegative value.

## Integration order

1. Pin all six DSOs to artifact commit
   `d5bf726dddc8c685a4eb766e7ec6cc303427501b` and all three public headers to
   source commit `2826cf12451e02ca4590f7a44682b57d2098bfb9`. Require the strict verifier,
   selected-config oracle, and prebuilt single-device numeric gate to pass on
   those exact files before deployment.
2. On Ubuntu 24.04, load the admitted SDK wrapper globally first. Load each
   qtype-selected FMT DSO locally, resolve symbols from its own handle, and
   require its exact build identity.
3. Parse and structurally/source-validate the schema-v3 sidecar against the
   source GGUF inventory. Query the selected FMT DSO for the canonical
   arrangement and finish semantic tensor validation by exact comparison.
4. Upload one complete tensor region and retain its span offsets plus complete
   arrangement-v2 in tensor metadata.
5. Query the selected-config ABI with a null requested name and retain the
   complete returned record. Dense uses v4 and its explicit
   `split_k_slices`; grouped uses v3. Pass either null for automatic selection
   or an exact previously validated returned name to workspace/launch APIs.
6. Route dense and grouped operations exclusively through the arrangement-v2
   device APIs. A declined K-pack route must not fall back to a raw GGUF or
   Xplane reader for the same resident bytes.
7. For optional first-load conversion, query the canonical descriptor and
   allocate the three spans with the checked formulas above. Q2_K/Q4_K must
   pass a null high pointer to both producer and inverse. Call the complete v2
   producer, require the inverse to reproduce the exact source tensor bytes,
   and only then admit the result.
8. Persist the admitted result atomically as the sidecar above; later loads
   validate and reuse it without repacking.

## Grouped post-operation candidate (2026-09-09)

`prebuilt/ppu0010/kpack-grouped-postops-v1` is a diagnostic same-parent A/B
package, not a replacement for the deployed native bundle. It adds direct
FP32 partial stores and a compact fixed-S reducer for grouped S>1. Offline
arrangements, public C APIs, S1 and llama.cpp wiring are unchanged. The
[bounded PPU gate](KPACK_GROUPED_POSTOPS.md#reviewed-ppu-closure) now passes
nine jobs / 384 cells across two preserved runs. The two directory-poison
failures pass their 96-cell replay with corrected fixture stream ordering,
without changing any packaged DSO or introducing an inference wait.

For the Q4 N512/K2048/E256/top8 model shape, compact S4 is fastest in the
measured candidate set: 14.589 µs versus 17.584 µs for the same-parent S4
baseline (-17.03%). Q5 N2048/K512/E256/top8 still favors compact S1 at
14.341 µs. Timing includes GPU metadata/directory/producer/reducer, but excludes
llama.cpp adapter casts/gather/scatter. The separate Q4 S4 ACU replay measures
the reducer at 5.770→2.161 µs; do not mix replay and warm full-call durations.

The native bundle now adds only the two measured candidate modules and
selects them for grouped FQ total M8/E256/max_rows1: Q4 N512/K2048 TM8/S4,
Q5 N2048/K512 TM8/S1. The host selector returns `QKS_MEASURED_GROUPED=5`;
the selection record's ABI size is unchanged. All 214 previous module DSOs,
the execution DSO, offline format and other shape choices remain unchanged.
The package contains 216 parent modules and 35,866,304 DSO bytes. Only the host
dispatcher was rebuilt; the two new modules reuse byte-identical PPU-tested
postops payloads. This update changes no ABI layout, so it requires no
llama.cpp source rebuild; restart the process to load the new package.

Next: run full-adapter, equal-work model/reference checks. Earlier native-gate
results cannot be resumed as though the changed manifest had already passed.
SIMT GEMV recipe admission and performance remain open; this postops gate
does not close them. Q8_0 also remains on llama.cpp's ordinary path. It needs
a separate format/reader/admission implementation, not a K-quant qtype alias.

The same-RTX-5090 SIMT comparison now includes historical DMMV without A
quantization and current MMVQ with Q8_1 quantization. The experimental FP32
group-affine/vector-load reader reaches Q5/DMMV parity within ~1%, but Q4
remains 35–49% slower; current MMVQ is faster on both anchors. All five-format
CUDA regression cells pass, but that experiment is not PPU-admitted and is
not in this native package. See [exact scopes and reproduction](../dev/gemv_cuda/README.md#matched-llamacpp-comparison).

## PPU GEMV/FQ/SF comparison gate (2026-09-09)

`prebuilt/ppu0010/kpack-gemv-affine-v1` adds only a 1.02 MiB diagnostic DSO,
locally hgcc-compiled. It is not installed into llama.cpp and does not replace
native-v1. The existing FQ/SF modules and selected splits are reused.
`tools/run_kpack_gemv_fq_sf_box.sh` compares five decode shapes (three small
anchors, two larger-weight M1 controls), then produces separate ACU reports.
FQ/SF core timing is distinguished from the common F32 endpoint pipeline;
the SF GPU prepass is separately checked/timed and included in an explicit
recompute scenario. GPU routing arrays are ready inputs to all arms; this is
not a full-model timing claim. PPU numeric/performance admission remains
pending. [Box command, scopes and selective resume](KPACK_GEMV_FQ_SF.md).

The initial `25YktY` box run stopped before any fixture or kernel: its runner
looked up the pair-reader API in native-v1's scalar-only execution DSO. The
runner now takes the already-published pair execution DSO from
`kpack-decode-sweep-v1`, with actual ELF-export and 240 host-query coverage.
No runtime6/native GEMM module, production selection, or llama.cpp adapter
is changed by this repair. Repeat the five-case gate in a fresh directory;
there are no valid timing samples to resume from that failed run.

### Reviewed k5Tp2m result (2026-09-09)

The repaired PPU gate passes all five Q4/Q5 cases: 240 screened SIMT recipes,
25 confirmed endpoint arms and 25 hash-verified ACU captures. This closes
this bounded operator test, not all-format or model performance admission.
See [times, scope and counter evidence](KPACK_GEMV_FQ_SF.md#reviewed-ppu-results-k5tp2m-2026-09-09).

With common F32 endpoints, affine GEMV/FQ takes 17.385/18.025 us on Q4-up
and 21.299/17.748 us on Q5-down. The larger dense Q4 controls favor FQ by
2.99x and 3.48x. Production selections are unchanged. The first small Q4
lead is only 3.55%, not a global GEMV promotion or proof of beating llama's
MMVQ on PPU; this run contains no llama reference arm.

The all-256-expert SF prepass costs ~55 us warm and emits 32 MiB of scale/zero
planes. Under the requested per-call execution contract this cost must be
charged on every SF call; it cannot be amortized by keeping expanded values.
Automatic decode therefore stays FQ. First-event initialization and steady-state
kernel times are separate. The current Q5 grouped SF fallback is still
rectangular, so it is not a measured optimum against compact FQ.

No library, native selection, llama adapter or cache synchronization contract
is changed by this result review. Remaining work targets SIMT instruction/
on-chip traffic, grouped scheduling, and a matched same-PPU llama reference.

## Model Asys rerun after FQ decode switch

Use an idle PPU. These paths come from the last uploaded successful model
setup; edit MODEL/CACHE/BUILD if they were moved. This incrementally rebuilds
llama.cpp and runs its adapter tests, not the Quactlize module sweep.
The command below is the historical cached-SF capture setup. Its V1 prefill
policy is intentionally rejected by the new adapter. Generate a V2 policy
with the updated native gate, or omit the policy and retain FQ; do not rename
the old table. See the current JIT/per-call-SF handoff below.
One request contains prompt evaluation followed by 64 generated tokens;
`-b 128` is the prompt microbatch limit, not 128 concurrent requests.
`default-set` records GPU API, kernel and memory activity together; graphs
remain enabled. Profiled times are not unprofiled latency admission or ACU
bandwidth counters. Q8_0 tensors intentionally stay on the llama route.
The subshell preserves the calling Docker shell on failure.

```bash
(
  set -Ee -o pipefail
  QZ=/sim/eec/shared/junfu.qx/quactlize
  LLAMA=/sim/eec/shared/junfu.qx/llama.cpp
  BUILD="$LLAMA/build-kpack-9f86a1340"
  MODEL=/sim/eec/shared/AI_workspace/llm-models/Qwen3.5-35B-A3B-Q4_K_M-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf
  CACHE=/workspace/llama-kpack-smoke.Lwpxya/cache

  export PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK
  source "$PPU_SDK/envsetup.sh"
  set -Ee -o pipefail
  export CUDA_VISIBLE_DEVICES=0 LC_ALL=C
  export LD_LIBRARY_PATH="$PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib:$PPU_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export QUACTLIZE_PPU_BUNDLE=/workspace/quactlize-runtime-artifact-2826cf1-46fc3096e1a1/prebuilt/ppu0010/2826cf1/runtime6-46fc3096e1a1/bundle
  export QUACTLIZE_PPU_PACK_LIBRARY="$QZ/prebuilt/ppu0010/kpack-pack-v1/libquactlize_ppu_pack.so"
  export QUACTLIZE_KPACK_EXECUTION="$QZ/prebuilt/ppu0010/kpack-native-v1"
  export QUACTLIZE_KPACK_PREFILL_POLICY=/workspace/kpack-native-model.O0ki3q/results/prefill-policy.tsv
  export QUACTLIZE_KPACK_ROUTE=auto
  unset GGML_CUDA_DISABLE_GRAPHS QUACTLIZE_KPACK_GEMV_POLICY

  RUN=$(mktemp -d /workspace/kpack-fq-asys.XXXXXX)
  test -d "$RUN"
  trap 'printf "\ntrace=%s\nrunner_rc=%s\n" "$RUN" "$?"' EXIT

  for f in "$MODEL" "$CACHE/manifest.json" "$BUILD/CMakeCache.txt" \
           "$QUACTLIZE_PPU_BUNDLE/manifest.json" "$QUACTLIZE_KPACK_PREFILL_POLICY"; do
    test -s "$f" || { printf 'MISSING %s\n' "$f"; false; }
  done

  test "$(git -C "$LLAMA" branch --show-current)" = feat/kpack-gpu-cache
  test "$(git -C "$QZ" branch --show-current)" = develop
  git -C "$LLAMA" pull --ff-only
  git -C "$LLAMA" merge-base --is-ancestor 135d7edcf HEAD
  git -C "$QZ" pull --ff-only
  git -C "$QZ" lfs pull --include='prebuilt/ppu0010/kpack-native-v1/**,prebuilt/ppu0010/kpack-pack-v1/**' --exclude=''
  python3 "$QZ/tools/verify_kpack_dispatch.py" "$QUACTLIZE_KPACK_EXECUTION" | tee "$RUN/bundle.log"
  git -C "$LLAMA" rev-parse HEAD | tee "$RUN/llama-source.txt"

  cmake --build "$BUILD" --target llama-completion test-quactlize-execution -j 192 \
    2>&1 | tee "$RUN/build.log"
  ctest --test-dir "$BUILD" --output-on-failure \
    -R '^test-quactlize-execution-(auto|fq|sf|gemv)$' \
    2>&1 | tee "$RUN/adapter-tests.log"

  PROMPT=
  for i in {1..16}; do PROMPT+='Explain matrix multiplication in one paragraph. '; done

  "$PPU_SDK/asight/bin/asys" profile \
    --trace hggc --hggc-trace-set default-set \
    --sample none --kill none --show-output true \
    --output "$RUN/kpack.asysrep" \
    "$BUILD/bin/llama-completion" \
    -m "$MODEL" --mmap -ngl 99 --split-mode none --fit off \
    --no-conversation --no-warmup --log-colors off --verbosity 4 \
    -c 1024 -b 128 -ub 128 -t 16 -tb 32 \
    -ot '^(blk\.[0-9]+\.ffn_[a-z0-9_]+_exps\.weight|output\.weight)$=CUDA0_KPACK' \
    --kpack-cache "$CACHE" -n 64 --ignore-eos --temp 0 \
    -p "$PROMPT" </dev/null 2>&1 | tee "$RUN/kpack.log"

  test -s "$RUN/kpack.asysrep"
  grep -aE '\[quactlize-plan\]|native policy miss|native execution package' \
    "$RUN/kpack.log" | sort -u > "$RUN/selected.txt"
  head -n 30 "$RUN/selected.txt"
  printf '\nOPEN %s/kpack.asysrep\n' "$RUN"
)
```

Open the printed `kpack.asysrep` in Asight. Return `selected.txt`,
`kpack.log`, `adapter-tests.log` and the trace when sharing the complete
timeline. A selected recipe in the log is not by itself proof of execution;
cross-check its parent with the actual compute activities, including the
compact producer and reducer where Split-K is selected.
