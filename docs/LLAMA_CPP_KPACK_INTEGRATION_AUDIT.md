# K-pack loader integration audit

Initial audit: 2026-09-08 against `/root/llama.cpp` develop `26955be8a` and
Quactlize develop `05de781`. The implementation below is now pushed to
`DrXuQian/llama.cpp`, branch `feat/kpack-gpu-cache`, at
`9f86a1340e12f68a6adee2320ef7e6dbb25ac06a` (remote SHA verified), using the GPU
producer from Quactlize `7fbe892`. No PR, compute-route switch or main admission
has occurred; llama.cpp develop remains `26955be8a`.

## Generated-answer follow-up: pilot reviewed, execution follow-up open

`EEZKMu` (`cb8b7906...`) completes 128 paired questions with both arms
122 correct and no correctness flips. Raw responses/timers and receipts
replay exactly. Observed K-pack decode is 10.4625 versus 8.0932 ms/token;
the 43 identical-output pairs retain a median +29.17% gap. The original
sweep omitted canonical GEMV, and the current loader routes small M to FQ
GEMM. No whole-gap causal claim is made from model timers.

Track native dispatch, device-only grouped metadata, GEMV candidates and
SF metadata lifetime/cost in [KPACK_EXECUTION_FOLLOWUP.md](KPACK_EXECUTION_FOLLOWUP.md).
In particular, all-expert SF expansion adds 3.75 GiB for this model. First-use
prepass+GEMM and resident GEMM need separate evidence. These changes are not
yet enabled in the model path; the following describes the completed runner.

llama.cpp `a39917a66acd742ae02b2034549fded8ce9506bb` is pushed to the same
fork feature branch. `tests/run-quactlize-gsm8k.sh` tests actual GSM8K answers:
128 seeded local test questions, question-only prompts, matched token arrays
and greedy generation, thinking off, context 4096 and a 1024-token output cap.
One model load per arm, one server slot and sequential requests keep business
batch at one; prefill token chunks default to 128. Both truncations and failed
final-answer extraction stay in the denominator. Matched correctness changes
and full responses are retained. Runtime failure preserves partial results and
does not claim a partial-sample pass.

Only test scripts/docs changed; local file loading is shared with the existing
PPL helper. No GEMM/pack DSO, ABI, loader or dispatcher change. The suite
checks placement and full cache hits but deliberately does not collect a new
device trace or claim controlled performance from unequal output lengths.
Local 16 new + 32 existing tests pass, including fake-HTTP subprocess/full-shell
and partial-failure/archive checks. Incremental local server build and four
host-only CLI checks pass; the generated-answer pilot is now reviewed above.
See the handoff for the box entry and evidence boundaries.

## Numerical validation: CigOEn reviewed, broad accuracy pending

Latest partial `QuHHHY` (`acba469b...`, source `d45f8fec2`): token-batch-128
short device proof and all three 8x1024 numerical phases complete; raw logs
match JSON. Paired PPL 1.739846 / 1.741794 (+0.1120%), mean/max KLD
0.002658 / 0.588895, top-token agreement 98.875%, 4,088 scored positions.
The first performance model exits 0 but its timers were filtered by
`--verbosity 3`. Real callback code maps library INFO to threshold 4.
`d25b88a18` fixes the level without DEBUG/profiling; 32 tests pass including
native callback execution and old-failure reproduction. Completed numerical
results remain valid; repeat only missing performance and token-batch-1 work.
No performance value is inferred from process wall. See the handoff for
the new `--performance-only` entry and exact archive/commit identifiers.
Both token batches are one business request (`n_seq=1`). This is GSM8K-text
likelihood comparison, not generated-answer GSM8K accuracy; that gate remains
pending and must not be reported as the measured top-token agreement.

Follow-up `d45f8fec232930c47632ecbe48ae201c7dee99e1` is pushed on the same
fork feature branch. `run-quactlize-numerical.sh --extended` reuses the
runtime/cache and adds a fixed 8x1024-token comparison at batch 128 and 1,
with short cached device proofs and separate unprofiled ABBA model timings.
No pack/cache publication replay or CPU reference; only test infrastructure
changed. The 31 local tests include synthetic shell phase/argv coverage;
12 actual-binary CLI-only checks and the ten prior raw-log replays pass.
The later partial results are recorded above. See the handoff and runner document
for evidence boundaries, timing scope and storage requirements.

Archive `llama-kpack-numerical.CigOEn.results.tgz`, SHA-256
`7511cc22ad4c4fe044439418b07068072e0451f80a6365b6a2d5d453411d8460`,
source `35b74114f806a4581daff9b916390f7b0ee8996d`: all ten phases finish
with rc=0 and consistent raw logs/JSON/TSV. All four token receipts agree
(same 254 scored positions per mode). Five DSO inventory hashes match the
bundle manifest. The six K-pack processes cover all 120 grouped tensors;
each reports 480 (batch 128) or 61,440 (batch 1) actual GEMM calls, split
2:1 between Q4_K/fmt0 and Q5_K/fmt1. Ordinary GPU controls report none.
Activity record names/durations agree with the inventory; original SQLite
and probability files were retained on the box, not uploaded or rerun here.

| Comparison | Batch 128 | Batch 1 |
| --- | ---: | ---: |
| Ordinary / uncached K-pack PPL | 1.4282 / 1.4325 | 1.4239 / 1.4244 |
| Cached/reference PPL increase | 0.2995% | 0.0370% |
| Cached/reference mean KLD | 0.003950 | 0.008010 |
| Cached/reference max KLD | 0.268156 | 0.503132 |
| Cached/reference top-token agreement | 98.819% | 98.425% |
| Cached/uncached max KLD | 0.000004 | 0.000043 |
| Ordinary self-replay max KLD | 0.000004 | 0.000046 |
| Cached/uncached top-token agreement | 100% | 100% |

Execution and cache replay are accepted on this short integration sample.
Four cached phases have 120 uploads, zero misses and no repack. Cross-route
differences are real relative to self-replay noise (3/254 and 4/254 changed
top tokens); their cause and broad acceptability are not established.
Keep broader model precision pending, rather than treating low average PPL
change as a full accuracy bound. No new kernel/loader/DSO implementation is
needed to review this result; no production source was changed here.
Remaining: larger numerical sample and unprofiled performance, not another
repeat of the same cache-transport test. The C++ dispatcher remains separate.

### Historical bring-up

Operator-returned `F56TMR` reanalysis at `35b74114f` passes batch 128
`reference-save`: PPL 1.4282, 12,594 GPU kernel calls, zero matching
Quactlize grouped GEMMs. The probability file has the expected context 256,
two chunks, vocabulary 248,320 and 254 scored tokens; token SHA-256 is
`ecbff63226d90a7498d0ac8a9d09f851aac5331c9450a907eeb4c63bf8fa5bdc`.
Only the pasted checker outputs are available locally, not the raw archive.
The ordinary GPU reference is the intended negative route control; empty
matching-kernel and routed-tensor lists are expected. K-pack and batch 1
were untested in `F56TMR`; see the later complete run above.
The complete runner starts a fresh directory
and repeats the reference; no resume or Quactlize DSO rebuild is implied.

`F56TMR` stopped in the reference-save checker. `35b74114f` corrects its
startup matcher: the actual C++ message contains `calculating perplexity`,
not just `calculating`. All 22 host checks now pass, including four
source-derived save-mode positives and strict coverage negatives. Existing
log/SQLite/log-probability files have now been rechecked without GPU work,
with the operator result above. K-pack numerical admission remains pending.
No kernel, profiler scope, or library modification is involved.

Next box failure was CLI-only (`invalid argument: --color`). Follow-up
`c7fcaea17` uses supported `--log-colors off` in both perplexity entries and
parses each exact argv with `--help` before the profiler/model process.
Actual local binary rejects the old flag and accepts the replacement;
six host parser combinations and the 18 existing evidence/corpus tests pass.
No arithmetic/DSO or trace-scope modification; no new device admission.

Box attempt at `50079485e` stopped during corpus download (line 84, rc=4);
no numerical process had started. Follow-up `ced89fc75` accepts existing
GSM8K JSONL/JSON/Parquet through `GSM8K_FILE`, with a fixed first-32-record
question/answer rendering and no download. Eighteen host tests pass.
Device evidence and PPL/KLD remain pending; this does not add generated-answer
scoring or change the GEMM/trace path.

`50079485e57ea7894c5aa76ce74c3836a5250375` adds a tests-only runner and evidence
parser. It compares ordinary GPU, GPU-packed/no-cache and disk-cached K-pack
using the existing perplexity tool at batch 128 and 1. No kernel, loader,
readiness or cache lifetime change is included. Existing FQ DSOs still own
compute; this does not wire the new C++ heuristic/JIT dispatcher.

Unlike the smoke's host route log, this gate requires Asight device-kernel
activity, matched to exact symbols from the format-library inventory.
Positive durations, workload-sized GEMM counts and all grouped tensor route
records are required; the ordinary GPU arm must have no matching Quactlize
GEMMs. Pack/metadata/gather activity and warmup-only evidence are rejected.
15 local parser tests pass; the actual DSO inventory contains 55 unique
grouped GEMMs, and local PPU perplexity compile/link succeeds. Device trace
and model numerical results were pending at runner bring-up. Two chunks / 254 scored tokens
per mode establish only a first integration sample, not broad accuracy.
See the single handoff and `tests/quactlize-numerical.md` in llama.cpp.

## Latest result: Lwpxya on ced6f241e

Uploaded archive SHA-256
`e89a779275566a3e0e7711d662fc6f0f09602a2efe3d9e98b131ab48acdfaa7e`;
source `ced6f241e28c344fef895fa95d6931855c73c4cb`. Smoke PASS, library
receipts OK; raw logs agree with summaries. Baseline/cold/hit each exit 0,
reuse 62 graphs and contain no CUDA/PPU launch errors. Same Qwen3.5-35B-A3B
Q4_K_M model and PPU-ZW810 PCI `0000:08:00.0`.

| Metric | baseline | cold | hit |
| --- | ---: | ---: | ---: |
| Reported load, ms | 4359.24 | 4387.46 | 3908.82 |
| Process wall, s | 8.38 | 22.24 | 8.54 |
| User CPU, s | 4.02 | 4.18 | 3.83 |
| System CPU, s | 2.25 | 15.72 | 2.62 |
| Peak host RSS, GiB | 21.17 | 21.19 | 20.53 |
| Prompt eval, ms / 11 tokens | 685.30 | 689.09 | 687.53 |
| Decode, tokens/s / 63 runs | 92.05 | 91.98 | 91.94 |
| GPU pack records | 120 | 120 | 0 |

Hit metadata ready at 0.588820 s, on-demand mapping logged at 0.588851 s,
buffer allocation logged at 0.653114 s, pinned H2D pipeline at 1.006078 s,
120 uploads queued at 4.063920 s. The 3.475100-second metadata-to-upload
interval includes other loader work and does not certify final DMA completion.
Consumers wait on their ready event; no per-tensor host completion wait was
restored. Compared with `lx3ae2`, peak host RSS falls 18.6142 GiB (47.56%).
Reported load is 10.33% below the same-run GPU-pack baseline, while total
wall remains 0.16 s higher. Hit has zero major faults/filesystem-input counts,
so this tests warm filesystem state; baseline timing also changed across
runs. There is no isolated attribution of speedup to either change.

Upload preflight covers exactly Q2-Q6 x E1/E257: ten cases, 20 eager reads,
60 graph replays and source overwrite. Readiness covers 18 eager / 18 replay /
six transitions / two expected-red missing waits. All pass. Cache metadata
lists unchanged 120 grouped E256 tensors (Q4_K 80, Q5_K 40), region sum equals
19,461,570,560-byte storage, schema v1 and arrangement v2, no SHA fields.

Cold writer starts at 4.578160 s, context teardown at 6.097940 s, copy/write
finishes at 17.026576 s and publication at 19.834908 s. Logged total 15.26 s
= 12.45 s copy/write + 2.81 s flush; a 13.737-second teardown-to-publication
tail remains. Inference proceeds before publication; exact physical overlap
is not measured. Numerical comparison remains explicitly NOT_RUN because
EVAL_FILE is unset. These results close the two targeted loader changes for
this workload, not model PPL/KLD, all-format model coverage or final runtime
admission. Existing Quactlize DSOs still own compute.

## Bounded H2D follow-up (2026-09-08)

User-approved commit `ced6f241e28c344fef895fa95d6931855c73c4cb` pipelines
cache uploads through two buffer-owned 8 MiB pinned slots. Borrowed host bytes
are consumed before return; only staged bytes remain in flight. Reuse waits
on that slot's H2D event, not the tensor's entire upload or any D2H/disk work.
Ready events retain the existing eager/capture contract and teardown drains
before freeing pinned storage. GPU conversion, persistent-cache file format
and the Quactlize DSOs are unchanged. Pinned bound is per K-pack buffer, not
a shared process-global pool; the tested model uses one such buffer.

Three host suites each pass five repeats, covering five formats, small/large
expert tensors, plane and partial-chunk boundaries, source overwrite,
cross-tensor reuse, async tails and pending teardown. Both new injected
upload-wait/early-reuse faults return rc=1 at a test invariant; the existing
blocking-D2H fault remains red. PPU build/link succeeds; the later Lwpxya
device result is recorded above. The box script first exercises actual cached
uploads and eager/graph consumers (ten byte-pattern cases, no CPU quantization
or GEMM reference), then runs the model cache comparison. The prefetch and
H2D changes are measured together; any improvement cannot be assigned to one
alone without a separate A/B. Model PPL/KLD still requires `EVAL_FILE`.

## Source prefetch follow-up (2026-09-08)

Commit `bf2877307f1bcb0d9b264ccf350a2329d2fe03a8` brings cache metadata
inspection before original GGUF mmap setup. Valid cache means on-demand
source mapping, not removal of the source; partial-cache and backend-declined
weights keep the existing raw loader fallback. Invalid/missing cache retains
prefetch. `no_alloc` does not construct the cache, and mlock behavior is not
changed. There is no H2D, event, writer, kernel or DSO change.

Cache/loader host tests pass five repeats each. PPU completion/perplexity
compile-link and disabled-backend host build pass. Actual Linux `llama_mmap`
control, five repeats: 32 MiB file, prefetch RSS 32768 KiB vs on-demand 0 KiB
before access, later endpoint reads correct. This local mapping check does
not establish model latency/RSS gains. The box smoke now asserts the source
mapping mode and includes host RSS/input counters in its timing summary.
Lwpxya above supplies the combined PPU result; `lx3ae2` below is the old behavior.
The second proposal was not included in this commit; its later implementation
is recorded above and remains performance-unmeasured. Upload bytes remain
unavoidable when a new process loads a disk cache.

## Prior rerun: lx3ae2, post-pipeline change

Archive `llama-kpack-smoke.lx3ae2.results.tgz`, SHA-256
`2cfa240fc20c808ab42fb45b54f755056b7a0e96f150de1bac8d584f4e6b19e6`,
matches source `2fdd7bc4c25af90f603dc5fb35fe5ee77083c6b8`. Verdict PASS;
raw phase logs and timing summary agree. All phases exit 0; no CUDA/PPU
launch/assert errors are present. Library receipts pass, and the readiness
gate passes 18 eager checks / 18 graph replays / six transitions. Each model
run reports 62 graph reuses. The same PPU-ZW810 PCI `0000:08:00.0` is selected.

| Metric | baseline | cold | hit |
| --- | ---: | ---: | ---: |
| Process wall, s | 9.58 | 22.04 | 9.98 |
| User CPU, s | 4.56 | 4.15 | 3.89 |
| System CPU, s | 2.36 | 14.67 | 3.98 |
| Reported load, ms | 4920.31 | 4359.85 | 5325.70 |
| Prompt eval, ms / 11 tokens | 704.41 | 681.49 | 677.28 |
| Decode eval, ms / 63 runs | 684.41 | 686.91 | 687.29 |
| Decode, tokens/s | 92.05 | 91.72 | 91.66 |
| GPU pack log records | 120 | 120 | 0 |
| Maximum host RSS, GiB | 21.17 | 21.19 | 39.14 |

Manifest: `llama.kpack-cache` v1, arrangement v2, no SHA fields; storage
19,461,570,560 bytes with 128-byte region alignment. Same coverage as before:
120 grouped E256 tensors, 80 Q4_K + 40 Q5_K, with 613 skipped records. The
manifest region sum equals storage size. Hit logs all 120 uploads and zero
resident misses. No weight payload was uploaded, so this review cannot
independently verify the cached bytes or model arithmetic.

Cold writer starts at 4.564955 s; context teardown at 6.081444 s; planes
finish copying/writing at 15.731103 s; publication at 19.633112 s. Logged
background time is 15.07 s = 11.17 s copy/write + 3.90 s flush/publication.
Pipeline aggregate copy/write rate is about 1,662 MiB/s, not an isolated PCIe
or disk bandwidth metric. Inference finishes before publication, but teardown
retains a 13.552-second context-to-publication tail. Cold total wall is 22.04 s.

Hit allocation at 1.588867 s, metadata ready at 1.596610 s (7.743 ms),
uploads complete at 5.497778 s (3.901168 s later), process wall 9.98 s.
The previous 34.303-second verification interval is eliminated by the
user-approved removal of payload hashes. Cold/hit wall times are 75.31%/
77.01% lower than the previous artifact's 89.27/43.41 s. Both hashing and
pipeline scheduling changed; these logs cannot separate their contributions.

Residuals: hit is 0.40 s slower than the same-run no-cache baseline; there
is no demonstrated cache startup advantage over GPU repacking yet. Host RSS
on hit is higher, and remaining copy/flush/teardown time is explicit debt.
Single samples, verbose logging and warm/ordered filesystem state prevent a
formal performance-regression or physical-overlap verdict. GPU producer and
GEMM libraries are unchanged. No new dispatcher or main admission is implied.

`numerical-summary.log` says
`KPACK_NUMERICAL_COMPARISON NOT_RUN reason=EVAL_FILE_NOT_SET`. PPL/KLD/logit
accuracy remains untested by this model run. Next input needed is an evaluation
corpus for the existing numerical script; no kernel rebuild is indicated.

## Published latency work after the successful box result

Per the user's explicit request, runtime payload checks are removed rather
than optimized. `llama.kpack-cache` v1 stores the unchanged plane layout with
source stat/tensor metadata and no content digests. Runtime loads also accept
old v3 bundles without scanning their source/storage payloads. The explicit
verified v3 offline API is unchanged. This deliberately gives up runtime
corrupt-payload detection; size/layout/bounds and local source identity checks
remain. No model/kernel/Quactlize ABI or library change is involved.

The model-owned writer now has two 8 MiB pinned slots per device. It queues
the next range before waiting for the current range, overlapping D2H with
CPU writes; errors drain any prefetch before releasing its destination.
Pinning happens before inference, completion waits stay on the writer, and
teardown joins before device weights are freed. Atomic fsync/publication and
refusal to overwrite existing output remain. Added progress/total/flush
timings make the remaining disk tail visible.

Host loader/buffer/sidecar suites pass five repeats each, including Q4 partial
chunks and Q5 high-plane/cross-tensor slot reuse. Copy rejection, completion
failure and EFBIG after a partial disk write leave no published/partial cache
or outstanding DMA; the existing cancellation test still holds resources
until completion. Runtime loading intentionally accepts a flipped payload,
while the explicit v3 verifier rejects it. These tests use mocked backend
events/bytes, not PPU numerical execution or physical overlap measurement.
PPU SDK compile/link passes for llama-completion, llama-perplexity and the
readiness target; shell syntax/help checks pass. The changes are published as
`35d625eb97503e814663c3d0b55f927d249a4ac7` on the fork's
`feat/kpack-gpu-cache` branch, with remote SHA verified. New box/model latency and
numerical comparisons remain pending. See the handoff for the optional
`EVAL_FILE` PPL/KLD workflow and its limits.

Follow-up `35d625eb9` box report: line 151 fails while extracting the timing
summary because completion uses `common_perf_print:` rather than the filter's
`llama_perf_context_print:`. The local script now accepts both and passes
shell syntax plus replay of the full timing-summary block on the earlier
three-phase logs. It is pushed as `2fdd7bc4c25af90f603dc5fb35fe5ee77083c6b8`
to the same fork branch; remote SHA verified. The new run's model/cache checks precede this line and remain
usable; PPL/KLD follows it and has not started. No model or library rebuild
is needed to recover the summary from its already-produced results archive.
The user chose to rerun after this script-only fix; existing compiled targets
and Quactlize libraries remain reusable.

## Prior box result: model/cache smoke before the latency change

Uploaded `llama-kpack-smoke.PmwxP9.results.tgz`, SHA-256
`a4848fd25ebfc9858b1f7daa10c7a7f80d5dce2e678c04d0c3405cac87f2ef79`,
reports source `5e632dc04509ae5923168b0898e34a3a328ba6de` and
`KPACK_MODEL_CACHE_SMOKE PASS`. Baseline, cold and hit each exit 0.
The unchanged six-library bundle's five FQ DSO receipts and separate pack
DSO receipt are OK. Readiness preflight passes 18 eager checks, 18 graph
replays, six mode transitions and two missing-wait negatives on PPU. This
closes the two ready-wait failures for this tested runtime/workload.

Model: Qwen3.5-35B-A3B-Q4_K_M, PPU-ZW810 at PCI `0000:08:00.0`.
The buffer override selects 120 MoE expert tensors: 80 Q4_K and 40 Q5_K,
all E256. Manifest regions sum to 19,461,570,560 bytes (18.125 GiB).
Source size is 22,016,023,168 bytes; 613 nonresident/nonselected tensors
are recorded as skipped. All 120 cached tensors are uploaded on hit with
`resident_misses=0` and no `GPU pack queued` records. Baseline and cold each
have 120 such pack records. Each model phase logs 480 K-pack route records
across the same 120 tensors (T2 warmup, T11 prompt and T1 decode); these are
host route messages, not a count of all device launches under replay.

| Metric | baseline | cold | hit |
| --- | ---: | ---: | ---: |
| Process wall, s | 9.64 | 89.27 | 43.41 |
| User CPU, s | 4.57 | 69.78 | 69.94 |
| System CPU, s | 2.63 | 15.96 | 4.76 |
| Reported load, ms | 5230.37 | 4646.55 | 38872.21 |
| Prompt eval, ms / 11 tokens | 698.65 | 642.91 | 682.20 |
| Decode eval, ms / 63 runs | 695.08 | 687.96 | 681.99 |
| Decode, tokens/s | 90.64 | 91.58 | 92.38 |
| Graphs reused | 62 | 62 | 62 |

Cold timeline: background writer starts at 4.844392 s; model execution
reaches context teardown at 6.326153 s; cache publication is at 86.887108 s;
process wall is 89.27 s. Writer-start-to-publication is 82.042716 s and the
visible post-context interval is 80.560955 s. The writer uses one 8 MiB
pinned slot and serializes per-chunk D2H completion, span/whole hashing and
write, then fsyncs and computes full-source/tensor hashes before publishing.
This run demonstrates that inference proceeds before cache publication;
it does not isolate copy/hash/I/O duration or prove physical copy/compute overlap.

Hit timeline: K-pack allocation log 1.589105 s; cache verified 35.892575 s;
uploads complete 39.024970 s; context teardown 40.568951 s; process wall
43.41 s. The allocation-to-verification interval is 34.303470 s, before
GPU weight upload. Code performs `verify_source(...,4)` followed by
`verify_storage(...,4)`, with full-file and per-record hashing. Verification
is not a background inference task. Verified-to-uploaded is 3.132395 s and
includes other tensor load work, so do not label it pure K-pack H2D time.
The small file-input counters indicate little physical disk input in this
run; repeated hashing/page traversal still consumes CPU on warm data.

This is one phase sample each with verbose logging and sequentially warmed
filesystem state. Neither the near-equal decoding rates nor faster cold
load establish a performance win. The test is not an independent model
numerical/logit oracle, all-format/dense-route gate, or final C++ heuristic
dispatcher admission. The cache's long startup/publication times remain
open technical debt. The subsequent explicit user request authorizes removing
runtime content checks and overlapping copy/write. The local implementation
above needs a new box measurement. No GEMM DSO or plane-layout change is
justified by this result. Historical statements below predate this artifact.

## First model smoke blocker: scheduler admission

The `9f86a1340` model run aborts before GEMM: `ggml-backend.cpp:898` cannot
assign a `NONE` leaf in `CUDA0_KPACK`. The backend already accepts the
`NONE` operator, but `ggml_backend_cuda_device_supports_buft` omits the
K-pack type. The local fix adds it alongside ordinary/split device buffers
under the same `buft->device == dev` check. Operator restrictions and all
device compute/format code are unchanged.

An isolated NVIDIA build reproduces the exact abort with the old callback
(rc=134) and passes with the fix. `test-quactlize-scheduler` uses real CUDA
allocations, the production backend/registry and GGML reserve/allocation:
21 cases cover five formats, dense/grouped, tokens 1/16, and Q4 E256.
Wrong-device and same-name fake buffer types and unsupported operators are
rejected. External DSOs are inventory stubs, and no pack/GEMM call occurs.
This regression cannot establish PPU arithmetic, model output or overlap.
The scheduler, loader, buffer and sidecar tests each passed 10 consecutive
runs on RTX 5090/CUDA 12.8. The modified PPU backend and regression target
also compile/link locally; three existing host suites pass locally.

Evidence: `/root/autodl-tmp/llama-kpack-scheduler-20260908.3z6rSg/`.
The remote test build is isolated under
`/root/autodl-tmp/llama-kpack-scheduler.FIczus/`; the machine's existing
llama.cpp checkout and models were not modified. Its older CMake 3.22
required a test-local header-only NVTX3 target; no such compatibility change
was added to production. The fix and regression are published as
`89e5b254dd39356bb9fc19ff0f8e46d19ec7f295` on the same fork branch, with
`tests/run-quactlize-cache-smoke.sh` as the model smoke entry. Script syntax,
missing-input and sourced-script rejection checks pass locally. Only
llama.cpp needs an incremental rebuild; Quactlize libraries stay valid.
The full PPU model/cache smoke remains pending.

Follow-up PPU artifact `llama-kpack-smoke.TXBaN9`: baseline rc=134 after
entering grouped execution, immediately after CUDA graph warmup. The supplied
error now identifies the default-flag `cudaStreamWaitEvent` in
`ggml_quactlize_wait_ready`: capture rejects a dependency on the event that
pack/upload recorded outside capture. This is not a GEMM mismatch or a wait
for D2H. Cold/hit remain untested.

The fix uses `cudaEventWaitExternal` in the shared dense/grouped
readiness helper, explicitly creating a wait node for the external event.
The event is recorded once and retained by the immutable weights. No CPU
wait/query or graph disabling is added; backcopy code is unchanged. CUDA and
the PPU SDK both declare this flag. A real CUDA test reproduces the old
failure and validates nine replays with completed ready, stalled pack and
stalled D2H. A missing-wait negative reads 4096 wrong bytes; submission
returns while pack is pending, and compute finishes with D2H still pending.
This is synchronization coverage with synthetic bytes, not a PPU pack/GEMM
or model oracle. PPU compilation passes; model/device verification is still
pending. Fix and regression are published as
`6e3e4a7dcb38cb9957204d2162dad4d2a9a40298` on the same fork branch.
Five suites each pass ten NVIDIA repetitions (50 executions, including
90 readiness graph replays). Evidence is in
`/root/autodl-tmp/llama-kpack-ready-20260908.7lXOMl/`.

Follow-up `llama-kpack-smoke.o0ghMa` on `6e3e4a7dc` fails earlier, in
eager warmup: the unconditional external flag returns illegal state. The
local SDK driver contains the exact non-capture/flags!=0 check, error 401,
and diagnostic `Illegal external flags for non-capturing stream`.
The fix is now state-dependent: query the stream's capture state, use 0
outside capture and the external flag inside. This does not query or wait
for GPU completion. No backcopy or GEMM change is needed.

The expanded real CUDA test covers eager/capture transitions in both
directions, pending pack and held D2H. PPU's stricter flag contract is also
asserted at the call seam so the regression detects the previous helper on
NVIDIA. Five suites pass ten repetitions; PPU compilation and three host
suites pass. These changes are published as
`5e632dc04509ae5923168b0898e34a3a328ba6de` on the same fork branch. The box runner now
executes the small readiness test before loading the model, with a 60-second
timeout. Actual PPU/model admission is still pending. Evidence:
`/root/autodl-tmp/llama-kpack-ready-state-20260908.Z23Xg7/`.

## Local implementation following the producer gate

The findings below describe the initial `26955be8a` snapshot. The isolated
integration commit replaces CPU `convert_verified` in `set_tensor` with the
separate GPU producer. The converter is loaded by `QUACTLIZE_PPU_PACK_LIBRARY`
or from `QUACTLIZE_PPU_BUNDLE`, and its complete size contract is cross-checked
against the consumer before intake. Missing/incompatible producers decline;
there is no automatic CPU conversion fallback. The compute route remains the
existing FQ DSO binding, not the new heuristic/JIT C++ dispatcher.

The final device allocation still contains contiguous `[low][high][units]`.
Whole experts are batched through reusable raw scratch (64 MiB target, or a
whole expert when larger). Scratch grows geometrically and earlier allocations
are retained until teardown to avoid `cudaFree` synchronizing unrelated work
during intake. This is a transient per-buffer cost, not an in-place raw/packed
alias. A large single expert still needs full-expert scratch.

Input and output readiness are distinct: `upload_done` is recorded before
the last pack, and `set_tensor` waits on it so its borrowed host input can be
released. `ready` is recorded after the last pack; dense and grouped consumers
enqueue `cudaStreamWaitEvent` on that event. This is a GPU dependency, not a
CPU wait for the copy stream. Validated sidecar uploads wait only for their
input upload before releasing their borrowed host planes.

The unused global synchronous sink is replaced by backend proc
`ggml_quactlize_copy_range_async`. The caller supplies a host-registered
destination, a bounded resident offset/size and an event on the same device.
The backend queues ready-wait, D2H and completion-record on its independent
nonblocking copy stream, then returns. It does not allocate, synchronize the
CPU, call an inverse or write a file. Compute never waits on this completion
event; the model-owned background writer owns that wait and the pinned destination.
Weights are immutable after publication. Intake, snapshot submission and
teardown must be serialized by their owner; teardown drains copies before
freeing the device source. This is not a globally thread-safe model queue.

Local validation: five affected backend translation units compile with the
PPU SDK. Existing loader and sidecar tests pass, including Python sidecar
interoperability. Ten new loader cases cover five formats, absent/incomplete
GPU libraries, size/query contradictions and forwarded launch errors. A host
delayed-queue test executes the actual buffer and readiness helper: a blocked
D2H does not prevent a second intake or compute readiness, released source
bytes are not reread, and a 1000-expert case exercises batching. Deliberately
draining D2H synchronously fails that test. These checks validate software
ordering, not device throughput/overlap or model results. There is no request
to repeat the completed 15-case producer gate.

The local model-loader integration now exposes `--kpack-cache DIR` and
`llama_model_params.kpack_cache_path`. It verifies existing caches before
direct plane upload; a miss queues metadata during loading and starts a
single writer only after loading succeeds. The worker calls the new bounded
`copy_range_async` / worker-only completion procs. One 8 MiB pinned slot and
event per device are allocated in the loading phase and kept through writer
completion. Failed copies/cancelled jobs never publish a manifest. The model
destroys its cache owner before freeing tensor metadata and device buffers;
normal teardown waits for persistence, not just the current D2H.

Streaming output matches the old writer byte-for-byte and passes the Python
validator. Host integration tests cover a delayed copy while resident weights
remain readable, cache-hit upload without repacking, descriptor mismatch,
duplicate capture, sorting, copy failure, changed source, and cancellation.
The source authority phase reads GGUF once at publication using bounded host
storage; D2H jobs themselves retain no raw pointer or full source tensor.
There is no CPU reference or inverse conversion in normal loading.

Scope is one named GGUF with the active backend's resident dense/grouped
tensors. Other source tensors are listed as skipped; multi-file, FILE*/custom
data loading are not connected to persistence yet. Invalid existing caches
are never overwritten. The source file must remain unchanged until publication.
Changing the public model-params struct requires rebuilding callers.

Still pending: device-only grouped and C++ heuristic bindings, multi-file cache
coverage, and actual PPU/model validation. PPU `llama-cli` and CPU-only `llama`
compile/link locally. The workstation's older glibc prevents starting the PPU
runtime (`GLIBC_2.38` missing), so local host tests are not a device gate.

## What can be reused

| Component in llama.cpp | Finding | Disposition |
|---|---|---|
| `ggml/src/ggml-cuda/quactlize-lib.cu` | Per-format `RTLD_LOCAL` loading, canonical-arrangement query, registry checks and any-M capability queries already exist. Host loader tests pass. | Reuse; these currently describe the old format DSOs, not the new selected modules. |
| `quactlize-buft.cu` | Owns one byte-neutral `[low][high][units]` device allocation; empty high planes use NULL. Non-Kpack readers/fusions are excluded. | Reuse allocation, artifact registry and validated sidecar upload seams. |
| `mul-mat-quactlize.cu` | Dense F32/FP16 boundary casts and row-major flattening are implemented. Execution is FQ with old `config_name=NULL`. | Reuse boundary handling; new heuristic/SF prefill is not wired. |
| `mmid-quactlize.cu` | Reuses `mm_ids_helper`, device bounds, gather and scatter without routing D2H. | Preserve this device-only property. |
| `src/llama-kpack-sidecar.{h,cpp}` | Schema-v3 parser, byte hashes, writer, no-replace publication and negative tests exist. Cross-check against `quactlize.pack_gguf.load_kpack_bundle` passes. | Reuse format handling after completing loader integration. |

## Gaps that a symbol-level replacement would miss

1. **Conversion is still host-side.** `qz_buffer_set_tensor` allocates three
   host planes plus a full recovered-GGUF buffer, calls
   `ggml_quactlize_convert_verified`, then synchronously uploads the planes.
   Conversion parallelism is across experts, so dense `experts=1` has no such
   parallelism. The metadata/code packer also constructs intermediate native
   code planes. This is not a GPU conversion path.
2. **Sidecar loading is not connected to model loading.** At this snapshot,
   the reader/writer and backend proc-address seams have implementations and
   tests, but `llama-model-loader.cpp`, `llama-model.cpp`, the public model
   parameters and common CLI have no callers. Existing sidecars therefore do
   not automatically bypass `set_tensor` conversion. The header's statement
   that the model loader wires them describes intended integration, not a
   current execution path.
3. **The new grouped ABI needs host rows.** `runtime/module.cuh` validates
   `rows_host`, derives exact group shapes, prefix sums, total tiles and the
   uniform-tile property from them. llama.cpp has only device `bounds` on its
   current hot path. An upper bound such as `n_tokens` cannot be passed as the
   measured per-expert row vector. A direct replacement would either require
   a synchronizing router readback or be incorrect. Close a device-directory
   grouped binding before claiming graph-capturable heuristic integration.
4. **Any-M admission cannot come from the exact table alone.** The old
   `*_any_m_valid` queries are capability promises for the old DSO execution
   path. They do not admit a new module selector that can decline unknown
   M/router/families after the original GGUF representation has been dropped.
   A numerical K-pack miss path must be explicitly admitted before enabling
   the new loader; do not silently use an arbitrary config string.
5. **Operand and device checks need a focused audit.** The early Kpack return
   in `ggml_backend_cuda_device_supports_op` bypasses the ordinary source
   device checks and does not check the F32/contiguous assumptions asserted
   by both execution wrappers. The capability query currently receives the
   weight only. Cover strided activations, mixed devices and unsupported
   sources before expanding its admission.
6. **The current sink is synchronous and borrowed.** `g_sink` and its context
   are global; the callback consumes CPU plane/source pointers that expire
   after `set_tensor`. It is neither a per-model asynchronous job owner nor
   safe to enqueue by merely retaining those pointers. The writer is ordered
   and not thread-safe. Background integration needs explicit ownership and
   cancellation, not just `cudaMemcpyAsync` substituted into the old call.

## Small fixes made during review

- The sidecar test target and its stub dependencies now remain inside
  `if(GGML_NCP_QUACTLIZE)`. Configuration with the option disabled succeeds
  and contains neither Kpack test target.
- `llama_kpack_sidecar_writer::begin` takes ownership of its staging path
  only after `mkdir` succeeds and rejects reentry on an active writer. Before
  the fix, refusal of an existing `.partial.<pid>` directory still left that
  path in the object, so destruction deleted another writer's files. A second
  `begin` could also abandon the first staging file. Both regressions were
  observed locally before the fix and pass afterward.
- The host loader and sidecar tests pass, including the optional Python
  bundle-reader interoperability check. These are not device inference tests.

## GPU producer: bounded device gate passed

`quactlize/packing/` provides a separate PPU conversion leaf:

- `api.h`: size query and asynchronous device-pointer producer.
- `word_pack.hpp`: one writer per final b16 code word; source code fields and
  metadata use the existing CuTe-owned traits. Q5 high-plane mapping and
  paired Q3/Q6 units reuse their canonical maps. There is no floating-point
  dequantization/requantization or atomic nibble scatter.
- `sizes.cpp`, `ppu_pack.cu`: exact canonical-descriptor/shape/range guards,
  low/high/metadata launches, immediate error propagation, no allocation,
  transfer or synchronization inside the producer.

All five formats share this converter; dense is `experts=1`. Input and outputs
are deliberately out of place. The caller uploads raw GGUF to temporary device
storage and writes the planes directly into the final weight allocation.
It must not overwrite raw bytes in place while other threads still read them.
Grouped weights can be batched along the independent expert axis. This first
entry does not support arbitrary N/K subranges of a single expert; a dense
tensor needs raw scratch for that tensor. Add an explicit globally-strided
subrange contract if that exceeds the agreed temporary-memory budget.

Host proof executes the same word/metadata ownership against independent
Python canonical bytes for Q2-Q6, dense and three experts, two K sizes, guard
bytes and invalid-descriptor/overflow cases. The PPU library compiles locally
in 9.355 seconds and is 126,504 bytes. Local artifact:

```text
/root/autodl-tmp/kpack-pack-20260908-v1/libquactlize_ppu_pack.so
```

The same payload and its build receipt are provided under
`prebuilt/ppu0010/kpack-pack-v1/`; the shared library is stored with Git LFS.
Its SHA-256 is `611ec98c4315748e4504082aa3071ae9713ee78958cd09b6baaee770b28a3184`.
Compilation alone is not device evidence; the uploaded gate is reviewed below.
This producer has not replaced the old six DSOs. The GEMM
kernel identity remains exactly
`2a9791f23a252fbb1fc5d6692f7bc7e3238d8cd7ff318ac427513e3d68e2eeb4`;
the measured 55-module cache does not need rebuilding for this converter.

### Reviewed box result (2026-09-08)

Archive `kpack-pack-gate.XSRuVd.results.tgz`, SHA-256
`556e61f5654f3e6b0c113b6e3210586b17654d95b1f47ed6705635cfcf641745`,
contains exactly 15 case JSON files, the summary and the console log. Local
review checked the exact case union, file/summary/log equality, size formulas,
all 75 finite positive timing samples and their recomputed medians. The
reported library hash matches the delivered ELF and build receipt; all
574 local source hashes in that receipt still match. This is a review of
reported box execution, not a local PPU rerun. The uploaded archive does not
contain the ELF or a separate runner-source/SDK receipt.

Device receipt: ordinal 0, PCI `0000:08:00.0`, `CUDA_VISIBLE_DEVICES=0`.
All five formats pass N=256,K=512 with one and three experts, plus
N=1024,K=5120 with one expert: **15/15 cases**. Every case has an initial
exact-byte comparison and a second full comparison after five more pack
calls, intact guard bytes, and rejected overlap/mapping negatives. That is
90 pack calls, 15 initial and 15 post-repeat byte checks, 15 overlap rejects
and 15 descriptor rejects. No intermediate repeated output was separately
downloaded. Event-ordered D2H into a separate pinned buffer also passes.

Timing for the five N=1024,K=5120 single-expert tensors, in microseconds:

| Format | H2D (one sample) | First pack interval | Pack median (five repeats) | D2H (one sample) |
|---|---:|---:|---:|---:|
| Q2_K | 173.480 | 224.640 | 77.200 | 108.280 |
| Q3_K | 100.920 | 306.360 | 110.040 | 95.560 |
| Q4_K | 317.040 | 386.200 | 55.600 | 436.520 |
| Q5_K | 380.480 | 430.560 | 121.960 | 213.480 |
| Q6_K | 136.480 | 196.920 | 97.560 | 133.640 |

The event intervals include any idle gap while Python/ctypes submits the
subsequent work; they do not isolate pure kernel instruction time. The first
interval is the first pack of that case, not an independently cold process or
JIT compilation measurement. Model loading converts a tensor once, so the
repeat medians alone cannot be its latency estimate. H2D and D2H have only one
sample per case; no transfer-bandwidth regression or universal throughput
claim is supported. Whole-harness wall time is 50.430512 s, of which
49.813468 s (98.78%) is CPU reference construction, not production conversion.
This CPU reference is not the old optimized C++ producer, so it is not an
apples-to-apples CPU/GPU speedup benchmark.

Following this result, full CPU reference generation and per-tensor CPU
inverse round-trips are excluded from the normal loader and subsequent
performance gates. Do not carry the old `convert_verified` inverse into the
GPU intake path. Retain the independent Python reference as an explicit
format-regression tool; do not delete the colleague-facing reference API.
Descriptor/size checks, launch/completion errors and source/sidecar hashes
remain required. Hashing and disk persistence are not reference generation
and can run in the bounded background queue. A changed producer needs its
own correctness regression outside performance measurement.

The producer can proceed to loader integration on this byte contract.
`llama_cpp_validated=false` and `overlap_speedup=NOT_MEASURED` remain explicit:
there was no GEMM concurrent with backcopy, disk writer, model checkpoint or
large single-expert memory-budget test. Those are integration work, not a
reason to rerun these same 15 cases. Keep the immutable build manifest as a
development receipt; this review does not rewrite its compile-time
`device_validated=false` into a historical claim that the build ran a device.

## Copy and persistence ordering

```text
raw GGUF -> bounded H2D scratch -> GPU pack -> packed-ready event
                                              |               |
                                     compute stream       copy stream
                                     wait(ready)          wait(ready)
                                     read-only GEMM       D2H pinned slot
                                                          copy-done event
                                                                  |
                                                     CPU hash/write worker
                                                                  |
                                                    ordered manifest commit
```

The inference dependency ends at packed-ready, not at D2H or disk completion.
Concurrent GEMM and backcopy may read the same immutable packed weights.
Actual overlap/speedup depends on the PPU engines and memory contention and
must be measured. Do not claim it from stream names alone.

The model owns a bounded pinned pool and a single ordered writer. Jobs retain
only metadata, the final device tensor and the copy-slot callback; no original
GGUF pointer is needed for D2H. Recycle a slot only after its copy and write
finish. Publication hashes the unchanged source file separately (one bounded
sequential read, no raw mapping kept alive). Cancellation/teardown joins or
drains before freeing device buffers. Publish only after every recorded plane
and required source/storage checks complete; no incomplete sidecar is a hit.

## Completed gate: reproduction command and scope

The first box gate is `tools/run_kpack_pack_gate.py`: exact output bytes,
repeated calls, output guards, overlap/descriptor negatives and an event-
ordered D2H into pinned host memory. It times H2D, pack and D2H separately.
It does not benchmark overlap with a GEMM or implement the background writer.
This gate is now complete; the command is retained for reproduction, not a
request for another box run. It uses the prebuilt converter without compiling:

```bash
(
  set -eo pipefail
  SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK
  source "$SDK/envsetup.sh"
  export LD_LIBRARY_PATH="$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export CUDA_VISIBLE_DEVICES=0
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  git lfs pull --include='prebuilt/ppu0010/kpack-pack-v1/libquactlize_ppu_pack.so' --exclude=''
  RUN=$(mktemp -d /workspace/kpack-pack-gate.XXXXXX)
  set +e
  python3 tools/run_kpack_pack_gate.py --sdk "$SDK" \
    --bundle prebuilt/ppu0010/kpack-pack-v1 \
    --output "$RUN/results" --real-anchor 2>&1 | tee "$RUN/console.log"
  rc=${PIPESTATUS[0]}
  set -e
  tar -czf "$RUN.results.tgz" -C "$RUN" .
  printf '\nrunner_rc=%s\nresults=%s\n' "$rc" "$RUN.results.tgz"
)
```

Update the develop checkout and its actlize submodule before running. The
gate checks the payload, its source receipt, SDK runtime libraries and the
one visible device's PCI identity. It does not require matching compiler or
inspector executables for execution. The optional real anchor adds all five formats at
N=1024,K=5120; CPU reference construction is reported separately from device
timing. Return the small `results` directory and build manifest/logs, not an
entire model. There are 15 cases, not a config sweep. Reference construction
dominates the reviewed 50.431-second run; it is not producer time. If
rebuilding the converter is needed, use
`tools/build_kpack_pack.py --sdk SDK --output FRESH_DIR` locally; the earlier
9.355-second build is for this small producer, not the GEMM modules.

## Final JIT deployment packaging

The current direction is a small selector/module-loader binding plus the
independent packing DSO and a cache of separate compute modules. Compile
small control/conversion libraries locally. Keep the existing admitted
kernel cache; a missing parent is compiled once with the target SDK, not
chosen by online profiling. There is no requirement to rebuild all six
legacy DSOs or to emit every config into a new monolithic library. Packaging
and C++ binding still need implementation; the old six-library bundle remains
unchanged as a transitional artifact. Future changes to a compute module's
ABI or body require rebuilding the affected modules, not merely relinking
the control library.

Next: validate the locally connected GPU intake, cache hit/miss and owned
background persistence on PPU; close C++ complete-identity/any-M and the
device-only grouped binding, then run checkpoint-level dense/MoE prefill and
decode with writeback disabled/enabled. Do not restart the Cartesian sweep.
