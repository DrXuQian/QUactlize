# K-pack TP2

## Implementation and admission

The caller branch is `dev/quactlize-tp2-v0.3.0` in the owner's llama.cpp fork.
The orchestration branch is `dev/kpack-tp2` in Quactlize. The runtime package
remains the existing `87b996559f` artifact; no GEMM/GEMV images or public
Quactlize ABI changed.

| Boundary | Implementation | Verification |
| --- | --- | --- |
| Weight placement | An explicit CUDA K-pack override becomes a Meta buffer containing one K-pack buffer per physical device | Host tests pass |
| Shards | Meta splits raw GGUF in N or K before each device packs its local tensor | 30 byte-exact host cases, six formats |
| Paired gate/up | Upload the two sources into the local shard's gate/up segments; pack after complete coverage | Host segmented transport and delayed upload pass |
| Compute | Existing per-device execution contexts query with local N/K; existing Meta communication and all-reduce remain unchanged | Q8 M1 local compute and PCCL sums pass on two devices after the LOCAL loader fix; six-format/model admission pending |
| Cache | Runtime cache v3 binds local planes to logical rank, split axis, segment widths/repeats and all GGUF source files | Six-format host cold/hot and negative tests pass; two-device admission pending |

Q2_K, Q3_K, Q4_K, Q5_K, Q6_K and Q8_0 keep their existing canonical layouts.
K boundaries must preserve whole GGUF superblocks and the reader's alignment.
Each local shard is checked before upload. Unsupported local geometry fails
explicitly; it must never be consumed by a raw-GGUF reader after packing.
Dense/shared compute stays F16; routed expert compute uses BF16, with F32
llama endpoints. Selection uses the existing exact/bucket policy on each local
shape. This is not a claim that unmeasured TP shapes have been tuned.

The Meta backend's split callback, graph redistribution, and all-reduce are
reused. Single-device loading and ordinary TP without K-pack overrides retain
their original paths. No CPU tensor rearrangement or CPU expert routing was
added.

## Persistent TP cache

Use `--kpack-cache DIR`. First load splits raw GGUF and packs on each GPU.
One background writer streams the final low/high/units planes into weights.bin,
using two 8 MiB pinned slots per device. Publication follows completion. There
is no D2H wait on the inference submission path; teardown drains the writer
before freeing resident weights.

After process exit, the next load uploads saved local planes directly to each
device. It does not repack or split packed bytes. All shards of a tensor must
match before any upload. A changed TP count, axis, segment map, paired-source
order, source file identity or arrangement causes a cache miss. Logical rank,
not physical GPU ID, determines the bytes.

The three-file 122B model is supported, including gate/up sources in different
GGUF files. Source checks use file identity and tensor metadata, not content
hashing. Single-device v1/v2 caches remain readable but cannot be used as TP
shards. Existing invalid or differently partitioned caches are never
overwritten; choose a new directory to publish another partition.

## Box entry

Use `tools/run_kpack_tp2_box.sh`. It builds the caller through its existing
`.aoneci/scripts/build.sh`, with 192 jobs by default. It reuses the pinned
Quactlize runtime; it does not rebuild the large runtime bundle.

Required inputs are `PPU_SDK`, `LLAMA_CI_DIR`, `NCP_LIB_DIR` and
`QUACTLIZE_PPU_BUNDLE` (the six-library compatibility bundle, with the Q4
short-K update below for TP shards).
The SDK precheck uses `PPU_SDK/include/hggc_runtime_api.h` for the native API
and `PPU_SDK/CUDA_SDK/include/cuda_runtime_api.h` for the compatibility API.
These are different include roots in the official 2.1.1 SDK; the runner now
prints the exact missing input instead of a combined, ambiguous test failure.
`KPACK_BUNDLE` optionally points at the pinned small native runtime already
downloaded locally. Otherwise only that pinned artifact is fetched with LFS.
`NCP_CI_DIR` optionally reuses a matching NCP checkout/build.
`LLAMA_CI_BUILD_DIR` optionally reuses a build of the exact same
`LLAMA_CI_DIR` source path and compiler. Omit it for a fresh caller build;
do not point it at a build whose source was a different isolated checkout.

The default model plan binds the exact three-shard 122B Q4_K_M model under
`/sim/eec/shared/AI_workspace/llm-models`. `MODEL_ROOT` changes that parent;
`MODEL_PLAN` supplies a different TP2 plan. Devices default to `0,1`, tensor
split to `1,1`, PP2048/TG128, NPL1. The runner prints progress and preserves
the invoking Docker shell even on failure.
Set `CACHE_ROOT` to the persistent parent directory. The default is
`RESULT_ROOT/kpack-tp2-model-cache`; the runner appends the model name.
Do not create the model subdirectory yourself. Reserve space for approximately
one additional packed model copy; mirrored weights add duplication.

The sequence is:

1. Runtime identity, caller build and host regressions, including the same
   TP graph allocation/split code on CPU-backed Meta devices.
2. `test-quactlize-scheduler --tp2-cache-write DIR`, then a new process using
   `--tp2-cache-read DIR`: 72 two-device dense/grouped cases per load
   across six formats, M1/8/32, N/K splits and three input/router replays.
   Six additional merged gate/up-SwiGLU-down chains test Q4/Q5 and Q8.
   An independent GGUF-to-F32 dot oracle checks outputs; a nonlinear consumer
   after K-split GEMM requires the existing all-reduce before squaring.
3. Reference/self/native model logits at token batch1 and2048, with numerical
   metrics and per-device selected-shape receipts. Likelihood metrics remain
   subject to accuracy review; finiteness alone is not an accuracy claim.
4. ABBA model performance, excluding the first complete PP/TG pass in every
   process. JIT and first-use latency do not count as steady-state TPOT.
5. Separate reference/native Asys capture, after one same-process warmup.

Each measured K-pack process must contain GPU-pack or cache-upload and selected-compute
receipts for both device ordinals, with matching local qtype/N/K/expert counts
and no legacy fallback. Metadata receipts alone are not kernel execution
evidence; the device gate and trace provide the additional checks.
Hot model loads must report zero resident cache misses and only CACHE producers
on both devices. The first numerical native process publishes the cache;
subsequent processes reuse it. The first full PP/TG pass remains excluded from
performance timing.

Upload `kpack-tp2.*.results.tgz`. Full reports remain under
`results/trace/qwen35-122b-q4km/{reference,native}/proof.asysrep` and are
excluded from that archive. Raw logits and model weights are also excluded.

### Q4 grouped numerical follow-up, 2026-09-19

Upload `kpack-tp2.exvMN0.results.tgz` has SHA-256
`7f4a71f60947d2f0d4e0715b0724c40c022f321cc9302f38513ef049680101ae`.
It uses orchestration `31cbdf9` and caller `ee28055`. The Q4 compatibility
update now admits the small shards. The cold gate passes 42 matrix cases:
all Q8/Q2/Q3 cases and Q4 dense M1/8/32 with both K/N splits. Those Q4 dense
cases use the legal compatibility fallback, not an admitted native optimum.

The next case fails: Q4, E4/top2, tokens1, global N512/K1024, local N512/K512,
K split, replay0. Relative L2 error is `0.1823166` on the squared, combined
output. Both devices select generic `simt-reuse`, variant0, columns4, warps4,
values4, S1, policy11, BF16, channels2. This is NOT the old invalid S4 request,
nor evidence that AllReduce itself failed. E4/top2 is outside the measured
E256/top8 small-M table and uses the initial generic SIMT proposal. Q2/Q3 have
passed the same generic geometry; the Q4 packing/compute boundary still needs
separate observation. The fixture's 61 input values are exactly representable
in both F16 and BF16, with maximum magnitude 0.234375.

Use `TP2_MODE=q4-local` with `tools/run_kpack_tp2_box.sh`. It incrementally
builds the changed scheduler test from caller `f9a0dbe2f` through AONECI and
runs three fresh processes:

1. Raw Q4 local `MUL_MAT_ID` on each rank, checked before the unchanged PCCL sum.
2. K-pack Q4 with the exact failing generic BF16 recipe and identical raw/input/
   router/golden hashes. Both local outputs are checked. A local failure stops
   before AllReduce; the runner still collects the next independent process.
3. The original one-case Meta graph with K split, AllReduce and square, without
   the preceding 42 cases or cache writer.

The tool rejects changed geometry, selection or fixture identities. Verdicts
separate local packing/compute failure, a Meta-only failure, and a failure not
reproduced in fresh processes. None claims a root-cause fix or device admission.
The mode loads no model, runs no performance sweep, changes no production
kernel or tolerance, and reuses the same runtime, JIT cache and Q4 overlay.
Upload its `kpack-tp2.*.results.tgz`; the small logs include each rank's local
errors, values, oracle hashes and collective results.

### Q4 short-K admission repair

After the communication fix, `BPvql1` stops before launching the Q4 local
shard: `q=12 N=512 K=512 E=1`, `local shard is unsupported`. This is an
independent admission error in the old `2826cf1` compatibility library. The
default decode selector chooses `kpack4:8x32x256:8x16:s3:S4`: K512 has only
two K tiles, so it cannot form four nonempty partitions. K1024 also cannot
satisfy that parent's minimum two tiles per split. Ragged partitions have the
same problem. The existing S1 parent accepts these geometries; grouped
admission is unaffected.

The actual delivered fmt0 host queries reproduce the rejection and admit
explicit S1. Local inspection used registration-only stubs with all device
operations disabled: it establishes the library's host decision, not GPU
correctness. The patch's C++ regression checks 1,620 M/N/K combinations,
preserves every previously legal default, and does not silently substitute
explicit invalid config names.

Only `libquactlize_ppu_fmt0.so` needs rebuilding. The isolated source is
`025c7e4d4b92330287099675a26bf2879b814f01`, branch
`fix/q4-smallk-admission`, directly based on the published `2826cf1` source.
Only its selector and host regression differ. Keep the current native runtime,
JIT checkout/cache and caller/NCP build unchanged; changing the current kernel
header tree would unnecessarily invalidate its JIT source identity.

On the box, source the official SDK environment, then run:

```bash
python3 tools/build_kpack_tp2_q4.py build \
  --base "$ORIGINAL_SIX_LIBRARY_BUNDLE" --sdk "$PPU_SDK" \
  --output "$NEW_UPDATE_DIRECTORY" --jobs 192
```

The output directory must not exist. The tool creates a detached source tree,
builds only fmt0, prints elapsed time every 30 seconds, runs before/after
host queries, and copies the other five libraries byte-for-byte into
`NEW_UPDATE_DIRECTORY/bundle`. It never overwrites the original bundle.
Failed builds and logs are preserved. This is not a full runtime rebuild or
a new llama binary delivery; single-library parallelism may not occupy all
192 cores.

The output uses `quactlize.ppu-compatibility-overlay.v1`, explicitly recording
both source commits, the original manifest and every file hash. It must not be
represented as six libraries compiled from one source. The TP2 runner verifies
this overlay and archives its manifest. Set `QUACTLIZE_PPU_BUNDLE` to the new
`bundle` directory and rerun `TP2_MODE=model`, reusing the `BPvql1` caller/NCP
build. Two-device cold/hot arithmetic and 122B numerical/performance admission
remain pending until that run passes.

### Cold fixture crash recovery

Caller `9942e30f5` fixes the first `--tp2-cache-write` fixture. A GGUF tensor
copied from a resident Meta tensor retains its buffer, and replacing `data`
alone does not change the GGUF writer's backend-read path. Meta indexes local
shards by the original tensor pointer; the copied tensor has no entry and
caused a null dereference before any GEMM launch.

The fixture now detaches buffer/view metadata and serializes the original host
GGUF bytes. The original and fixed scheduler helper were exercised with real
CPU-backed Meta buffers: exit 139 before the fix, exit 0 with byte-exact source
contents afterwards. A host regression also rejects retaining the buffer.
This is a test-fixture fix, not device correctness admission.

After updating both TP branches, obtain the existing build path from the failed
run's `results/caller-ci-build.json` (`build`) and pass it as
`LLAMA_CI_BUILD_DIR`. The runner incrementally builds the changed test program
and uses a new run directory for cold/hot fixtures. Do not reuse the failed
`device-cache` as a cold-write destination or delete the old evidence. The
Quactlize runtime, NCP kernels and model-cache schema are unchanged.

### K-split arithmetic gate correction

Caller `c2dbaf78d` corrects the subsequent first-cell Q8 failure. The test used
`ggml_backend_alloc_ctx_tensors` for both inputs and computed nodes. Its static
split callback labels unnamed results MIRRORED, so Meta omitted the all-reduce
between local K-split products and the squared output. This is a graph setup
error; it can be reproduced without a PPU, K-pack reader or cache writer.

Inputs now retain their explicit static split, while `ggml_gallocr` allocates
compute nodes and lets Meta infer their split states. The test asserts that a
K-split product is PARTIAL and its nonlinear consumer is MIRRORED. N-split
products and gate/up-SwiGLU intermediates also have explicit split checks.

`test-quactlize-tp-graph` compiles the same fixture source without CUDA and
passes 72 six-format matrix cases and three F32 gate/up-down graph cases. The
retained legacy negative has 88-91% relative error against the full result but
0.54-1.52% against rank0's partial squared result across three input replays.
CPU quantized GEMM requantizes A, so the host chain test uses F32 to isolate
graph semantics; it is not a substitute for the GPU Q8 and Q4/Q5 chain gate.
All device tolerances and the runtime bundle are unchanged. Use the failed
run's caller build for the incremental rebuild as above.

## Communication failure review and small diagnostic

The uploaded `kpack-tp2.JFVDyR` run used caller `c2dbaf78d` and orchestration
`23d2e95f4`. All five host gates pass, and the first two Q8 shard cache artifacts
publish successfully. The first K-split cell reaches the existing PCCL
all-reduce of 512 F32 values (2048 bytes), then aborts in `drv_extension.cc:379`
with `invalid device function` at `ncclGroupEnd`. Neither completed local
arithmetic nor device all-reduce correctness is established by those launch
receipts. The earlier successful 122B reference remains valid; its log does not
identify the loaded PCCL extension or exercise every small collective size.

Set `TP2_MODE=communication` on the existing box entry for a bounded diagnostic.
It reuses the current `.aoneci` build and runtime package without loading a model,
creating model caches, or capturing Asys. Ten fresh processes run:

- Known F32 buffers of 512, 3072 and 32768 elements through the caller's unchanged
  communication entry. Values and sums are exactly representable; require zero error.
- Raw-GGUF Q8 and K-pack Q8 on the same local shape, K512/N512/M1 on both devices.
  Each local result is synchronized and compared to an independent GGUF dot before
  communication. The collective is then checked against the full dot. Three input
  replays retain the existing 2% arithmetic tolerance.
- Repeat the 512-element buffer and K-pack cases with `PCCL_ENABLE_EXT_KERNEL=0`
  in those child processes only. This is an extension-path control, not a proposed
  production default and not permission to ignore a failed collective.
- Load only the SDK wrapper before the first 512-element buffer collective,
  once LOCAL and once GLOBAL. A final K-pack process explicitly promotes the
  wrapper to GLOBAL to restore the previous loader behavior in the same binary.

The binary prints actual loaded communication/runtime library paths from
`/proc/self/maps`, including libraries loaded with `dlopen`. Every failed child
is retained and the remaining cases continue; each has a 180-second timeout.
Read `results/communication/summary.json` and the ten adjacent logs. Runner
success means `DIAGNOSTIC_COMPLETE`, not TP2 admission. Production communication
and kernel dispatch are unchanged. Upload the resulting small results archive.

### Wrapper-scope candidate

`kpack-tp2.1InYnX` completed the original seven cases. All copy and raw Q8
cases pass. Both K-pack cases produce exact local outputs on both devices,
synchronize successfully, then fail at their first collective. Both use the
same SDK 2.1.1 PCCL and 13.0 runtime as the successful controls.

The detailed log order matters: raw Q8 completes its first collective before
CUDA graph preparation loads the execution libraries, wrapper and
`/usr/local/PPU_SDK/targets/x86_64-linux/lib/libhggcrt.12.0.so`. Later raw
collectives still pass. K-pack loads them before its first collective and
explicitly preloads the wrapper GLOBAL. Thus coexistence of runtime versions
alone is not a sufficient explanation. The disabled process confirms its
environment has `PCCL_ENABLE_EXT_KERNEL=0`, but still fails at the same source
line; the filename does not establish which optional plugin path was active.

The candidate changes only the caller wrapper preload from GLOBAL to LOCAL.
The five delivered format DSOs, the pack producer and execution DSO explicitly
name `libhggc_wrapper.so` in DT_NEEDED. They do not need the wrapper promoted
into unrelated libraries' global symbol lookup. No SDK files, ABI, kernel,
collective implementation, numeric threshold or default PCCL option changes.

A host ELF regression exercises the production preload function with real
DSOs: the LOCAL candidate preserves a communication library's own dependency;
the GLOBAL negative interposes the wrapper's same-named function; priming the
communication call before GLOBAL loading masks that failure. An explicit
DT_NEEDED consumer still resolves its wrapper entry in the LOCAL case. This
proves the host binding mechanism, not PPU correctness.

The box diagnostic also records RTLD_DEFAULT symbol owners. It reports
`GLOBAL_WRAPPER_CAUSAL_CANDIDATE_PASS` only if the seven ordinary cases and
LOCAL-only control pass, both GLOBAL controls reproduce the historical
collective failure after clean local checks, and symbol/path receipts match.
Otherwise it reports the unmet condition and retains all logs. Full TP2
cold/hot, model and performance admission remain pending after this small gate.

### PPU wrapper-scope result, 2026-09-18

`kpack-tp2.BPvql1.results.tgz` confirms
`GLOBAL_WRAPPER_CAUSAL_CANDIDATE_PASS` using caller `ee28055efd`,
orchestration `e4f1b79` and NCP `9bfb443835`. The caller worktree was clean;
both caller and NCP were freshly built through `.aoneci`. All five host gates
pass. Archive SHA256:
`266e2c9f7976df0ff2045363bf7640f3721e3b9b8a106d2e811f47647b194e02`.

| Device control | Result |
| --- | --- |
| Original seven controls with the LOCAL loader | All pass, both ranks and three input replays |
| Copy plus explicit LOCAL wrapper preload | Pass |
| Copy plus explicit GLOBAL wrapper preload | Original first-collective `invalid device function`, after exact local results |
| K-pack plus explicit GLOBAL wrapper preload | Same first-collective failure, after exact local results |

Both ordinary K-pack processes have zero local and collective error on every
rank/replay. GLOBAL controls expose `hggcLaunchKernel`, `hggcGetFuncBySymbol`
and `__hggcRegisterFatBinary` from the wrapper in RTLD_DEFAULT; LOCAL controls
do not. The copy-only GLOBAL failure does not require a K-pack GEMM. This
establishes the wrapper's global visibility as the trigger for this communication
failure. It does not identify a specific internal PCCL binding or prove that
runtime-version coexistence alone is the cause: successful K-pack processes
still map both runtime versions.

Retain the production LOCAL change; do not alter PCCL algorithms, disable
collectives, patch the SDK or rebuild the Quactlize kernel bundle. Continue with
`TP2_MODE=model`, reusing this successful run's `caller-ci-build.json` fields
`build` and `ncp_directory` as `LLAMA_CI_BUILD_DIR` and `NCP_CI_DIR` respectively.
The six-format device cold/hot gate, 122B numerical checks, ABBA performance and
Asys capture are still required. `runner_rc=0` here means the diagnostic completed,
not full TP2 admission; the two GLOBAL failures are expected negative controls.

### Q4 local failure and CUDA control, 2026-09-19

`kpack-tp2.Zu8e1e.results.tgz` (SHA256
`8b8840ec384016c32e0636de5400648e07334f836cae92e33336c0059cf968ca`)
isolates the current numerical failure before communication. The local shape
is Q4_K N512/K512/E4, tokens1/top2/channels2, obtained by splitting global
K1024. The native choice is generic SIMT v0/C4/W4/P4/S1, BF16 compute with
F32 endpoints, policy11. This is not the previously rejected TC S4 choice.

| Fresh process | Result |
| --- | --- |
| Raw Q4 local computation followed by PCCL | Both ranks and all three changing-input replays pass |
| K-pack local computation | Rank0 relative error0.126439014; rank1 error0.0813996521; collective not launched |
| Original Meta case without preceding tests/cache writer | Reproduces error0.1823166 |

Among the first eight reported local outputs, only columns0 and4 are wrong;
columns1/2/3/5/6/7 agree to F32 roundoff on both ranks. This implicates a
four-column ownership/arithmetic seam but does not prove which seam is wrong.
Do not change PCCL or relax the numeric threshold to address this failure.

The actual llama host quantizer reproduces the uploaded raw-shard FNV hashes
`df1668dc7ea3b843` and `a29e1d9b60fed398`, input hash`34b4409416754541`,
and replay0 golden hashes`3b28bc5ab3c46aac` and `0947ec09749102d0`.
Host execution of the production word/metadata packer, decoded independently,
recovers all2,097,152 local weight values bit-exactly. Fixture activations are
exactly representable in both F16 and BF16; the earlier activation-overflow
failure does not explain this sample.

An isolated CUDA12.8/sm_120 replay on RTX5070 includes the unchanged production
SIMT header and GPU packer through the development CUDA adapter. It checks both
rank shards, three changing inputs, host versus GPU packing, and F16 versus
BF16 compute: all24 output checks pass, as do all12 low/unit byte comparisons.
Maximum absolute dot error is8.94069672e-8; maximum relative L2 error is
1.00766429e-7. The four core SIMT/helper source hashes match the uploaded PPU
execution receipt. No tactic, canonical layout or production library changes.

CUDA evidence is retained at `/tmp/q4-tp2-local.nUBON1/cuda-results.tgz`,
SHA256`c0191a3ae1872dd2e7c14382d3424fcdcf64493432781b8039fb633b9796b23a`.
The remote source/fixtures/logs are in `/home/qianxu/q4-tp2-input-nUBON1`.
This is a source-level cross-platform control, not the PPU binary or llama
integration. It does not prove a compiler defect or certify PPU correctness.
The next PPU boundary is actual packed bytes versus host packing, then the
same standalone SIMT row versus caller execution and its intermediate values.
Keep TP2 numerical/model admission pending.

### One-device PPU replay

Use `tools/run_kpack_tp2_simt_box.sh` next. It needs only the selected SDK,
one visible PPU and the existing small execution/pack package. No llama, NCP,
six-library compatibility bundle, model, JIT, collective or performance sweep
is involved. Both frozen rank shards run on the same device in fresh processes.
The synthetic 1.2 MiB fixture is versioned in ordinary git, not a new library
or an LFS bundle; see [fixture construction](../dev/gemv_simt/tp2_q4_fixture.md).

```bash
PREVIOUS_RUN=/path/to/kpack-tp2.Zu8e1e \
PPU_SDK=/path/to/PPU_SDK CUDA_VISIBLE_DEVICES=0 JOBS=192 \
RESULT_ROOT=/path/to/results \
bash tools/run_kpack_tp2_simt_box.sh
```

Alternatively set `KPACK_BUNDLE` to the small runtime directory that contains
`manifest.json`, `libquactlize_ppu_execution.so` and `pack/`. The tool verifies
both delivered libraries and the unchanged six SIMT/helper sources against
that manifest, uses the same compiler flags, and records SDK binary differences.

Only four translation units are compiled, with up to four simultaneous jobs;
requesting 192 jobs does not create unnecessary work. The shipped caller is
host-only and loads the original libraries. The fresh caller contains the
unchanged production SIMT specialization, the real GPU packer and a scalar
canonical-reader control. These images never coexist in one process.

There are 12 processes, 72 positive output cells and 12 wrong-expert negatives:
two ranks x three inputs x two callers, with host/GPU packed planes and
F16/BF16 compute in each; only the fresh caller also tests the scalar control.
Each process verifies packed bytes and prelaunch hashes. A failed process is
retained and the remaining processes continue. Incorrect outputs are dumped
with first-error coordinates and a column-mod16 histogram. No timing is valid
in this diagnostic.

Upload the printed `q4-tp2-simt.*.results.tgz`. The final verdict distinguishes
GPU packing, scalar canonical compute, fresh SIMT and shipped-only failures;
`STANDALONE_NOT_REPRODUCED` means caller/environment investigation is still
needed, not that the original failure is fixed. A complete numeric failure
can have runner_rc=0; missing coverage returns nonzero. TP2 admission remains
pending until the actual repair and full two-device gate pass.

Local checks: all four PPU/host objects compile with SDK2.1.1; the host-only
caller contains no device image. Final executable linkage on the local
glibc2.35 host is unsupported by this SDK's glibc2.38/libstdc++ requirements;
the box runner links normally on Ubuntu24.04, without suppressing unresolved
symbols. Do not interpret object compilation as a passed PPU execution gate.
Nine host regressions pass, including missing/duplicate/nonfinite record
rejection and continuation after an individual process failure. The exact
fresh diagnostic body was also compiled and run on RTX5070: all48 output
cells pass, all12 pack comparisons are byte-exact, and all6 wrong-expert
negatives turn red. The largest scalar-control absolute error is1.78813934e-7.
These additional controls are retained at
`/home/qianxu/q4-tp2-replay-control.XVE1xB/results`; they still do not run the
shipped PPU image or establish a PPU root cause.

## Remaining device admission

- Run the six-format two-device cold/hot gate and 122B numerical, timing and
  trace checks. The admitted Q8 communication control does not cover them.
- Review local-shape policy receipts and the TPOT comparison before performance
  claims; the previous 122B upload was an ordinary TP2 baseline only.
- Check cold publication and a fresh process's hot reload on the real 122B model.
