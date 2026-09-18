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
| Compute | Existing per-device execution contexts query with local N/K; existing Meta communication and all-reduce remain unchanged | PPU compilation passes; device execution pending |
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
`QUACTLIZE_PPU_BUNDLE` (the existing six-library compatibility bundle).
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

1. Runtime identity, caller build and host regressions.
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

## Remaining admission

- Run the two-ZW810 numerical, model, timing and trace checks. Local host tests
  and successful PPU compilation are not device admission.
- Review local-shape policy receipts and the TPOT comparison before performance
  claims; the previous 122B upload was an ordinary TP2 baseline only.
- Check cold publication and a fresh process's hot reload on the real 122B model.
