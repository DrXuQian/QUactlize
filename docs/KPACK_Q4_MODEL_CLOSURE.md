# Q4 model integration and performance closure

Current scope: the private `dev/quactlize-v0.3.0` branch, canonical K-pack,
request batch 1, prompt 2048. Q4_K_M model files can contain Q5_K, Q6_K and
Q8_0 tensors; their actual paths must be covered, not just qtype 12.

## NCP runtime link repair, 2026-09-15

The joint build in `kpack-q4-model.N5J9uY` reached the NCP executable link
step. `libncp_moe.so` referenced `hggcGetDeviceProperties_v2` but did not
link its owner, `libhggc_wrapper.so`. The SDK exports that symbol; neither
a missing SDK installation nor a kernel failure explains this linker error.

Private llama `174fcb11b` adds a target-local dependency through
`.aoneci/cmake/ncp-ppu-runtime.cmake`. It searches the selected compiler's
SDK `targets/x86_64-linux/lib`, then `lib`, without changing the existing
runtime link order or any API name. FA link dependencies are unchanged.
Local checks reproduce the missing-symbol failure, then link and run both
consumers after repair while retaining all object timestamps. A link-only
probe against the real 2.1.1 SDK also passes with `--no-undefined`; its
`DT_NEEDED` includes the wrapper. This is not a PPU numerical result.

To reuse this failed build, set the **NCP checkout** path, not its `build`
subdirectory:

```bash
NCP_CI_DIR=/workspace/kpack-q4-model.N5J9uY/ci/ncp_flash_lib \
NCP_LIB_DIR=/sim/eec/shared/junfu.qx/ncp_flash_lib \
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
JOBS=192 CUDA_VISIBLE_DEVICES=0 bash tools/run_kpack_q4_model_box.sh
```

The runner reconfigures that NCP build in place with the link repair. It
checks the revision, submodule revisions, source/compiler paths and build
profile before reuse; it does not clone, reset or clean that NCP checkout.
Unchanged FA/MoE objects remain reusable. The caller build, which had not
started in the failed run, still goes into a new output directory. The
receipt records `ncp_build_mode=REUSE_BUILD` and the reused directory.
Do not run two builds against this NCP directory at the same time.
Unset `LLAMA_CI_DIR` to fetch the new pinned caller, or update your local
`dev/quactlize-v0.3.0` branch before using the local override. No new LFS
payloads or Quactlize kernel rebuilds are needed for this repair.

## Caller and runtime package boundary, 2026-09-15

No llama executables or llama/ggml dependency libraries are published by
Quactlize. `publish_kpack_dispatch.py` only copies Quactlize runtime, policy,
producer and gate payloads; the old `--llama-build`/`--llama-source` options
are removed. The focused `kpack-model-runtime-v1` package has 12 LFS
payloads, about9.3MiB, with unchanged kernel hashes. The old combined
package remains historical and is not fetched by the current runner.
Artifact commit: `58ab5fb`; its producer, execution, prefill, dispatcher,
seven parents and mixed-stage binary match the previous package bytewise.

Update source with `GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only` so unrelated
historical experiments are not downloaded. The runner then pulls only
its pinned runtime package and builds the caller on the box.

## Build entry correction, 2026-09-15

Use the private llama branch's `.aoneci/scripts/build.sh` for the next
caller build. Preserve its joint NCP FA/MoE build and ON hooks; the Quactlize
branch also explicitly enables `GGML_NCP_QUACTLIZE=ON` and passes `PPU_NVCC`
to the llama CMake invocation. Do not substitute a separate CMake recipe.

The previous `f2a2f99` artifact used a direct CMake invocation with NCP
FA/MoE OFF. It remains a historical functional candidate, not the requested
CI performance baseline. A new caller/NCP build must be recorded
before reporting performance under the CI configuration. The Quactlize
mixed-chain and prefill DSOs do not need a new kernel sweep for this change.

The supplied `NCP_LIB_DIR` is `/sim/eec/shared/junfu.qx/ncp_flash_lib` on
the box; it is not mounted locally. The model runner now clones committed
NCP/submodule objects into a fresh run directory and calls the exact
`.aoneci/scripts/build.sh` there with `JOBS=192`. Original checkouts, local
patches and old build products are untouched. NCP revision is pinned to
`9bfb44383588cbf4eed98e3d51de90e8f82d1779`; the supplied checkout and its
initialized submodules must contain the pinned commits.

The model phases use the new `ci/llama/build-ci/bin`, not the historical
packaged caller. `results/caller-ci-build.json` records source IDs, CMake
flags, executable/library hashes and the DeepGEMM JIT headers. Quactlize
DSOs and gate binaries still come from the unchanged LFS package. A joint
build has not run locally; its receipt is produced only after box success.
An empty/comment-only `.gitmodules` means no submodules; Git's no-match
status1 is not a failed checkout. Invalid config still fails before build.

## Current checklist, 2026-09-14

| Step | State | Completion evidence | Location |
| --- | --- | --- | --- |
| Release and patch base | Complete | `dev/v0.3.0` at `e73e2136b`; only `dev/quactlize-v0.3.0` advances | Local/private fork |
| Measured component selection | Complete within documented coverage | Small-M SIMT/TC policy and measured prefill FQ/SF/full-BF16 policy; no double-counted reducer | Existing PPU results |
| Library functional gates | Complete for declared denominators | Decode 208 dense endpoints and four real TC chains; prefill nine composed contexts | Reviewed PPU results |
| Decode fusion wiring | Implemented; mixed-path PPU gate pending | Additive mixed-chain entry retains measured SIMT/TC choices and direct F32 dense endpoints | Local ABI/compile checks; 80 stage and 24 real-chain box contexts |
| Model deployment entry | Implemented; publication receipt pins the payload | New dispatcher/execution/prefill package, v0.3.0 caller, headers/JIT source and paired packer | `tools/run_kpack_q4_model_box.sh` |
| Whole-model accuracy and route proof | Pending | Same model/input against native; actual decode/prefill kernels match selection, no unnoticed legacy fallback or standalone decode dtype casts | PPU box |
| Whole-model performance | Pending | Unprofiled native/K-pack A/B, first JIT/warmup excluded, separate PP/TG latency and throughput | PPU box |
| Final optimization and freeze | Conditional on model A/B | Fix measured bottlenecks, rerun affected checks, retain exact source/binary/policy identities | Local plus PPU box |

The last three device phases can share one staged runner. A failed functional
phase cannot admit a performance result. Passing the first model run does not
promise the no-slower-than-native target: remaining regressions require a
targeted trace-guided follow-up, not another full Cartesian sweep.

## Verified prefill return

`kpack-prefill-composition.A4NxtK.results.tgz` passes all nine contexts.
The [compact receipt](measurements/kpack_prefill_composition_20260914.json)
binds the returned JSON to the published package manifest. Five dense
Q2--Q6 full-BF16/cuBLAS contexts and four Q3 full-BF16/DeepGEMM contexts
pass, including 27 graph replays and 40 SF expansion config checks with zero
bit differences. Maximum absolute-product-normalized dot error is
0.0038757212, below 0.005. All code-plane negatives and output/workspace
guards pass. Each grouped context switches to 16 active experts and leaves
the other 240 expanded-weight slices untouched.

The run took 113.83 seconds including fixtures and checks. That is gate wall
time, not kernel latency. It contains no llama whole-model performance result.
The archive does not supply a runner checkout SHA; none is inferred from its
name or upload time.

## Mixed decode bridge

The caller already implements direct F32 dense TC endpoints, measured Q4
SIMT selection and direct F32 SIMT calls. At private llama commit `03142a8b7`, `prepare_moe()` rejected
any projection for which `Plan::direct` is true. The v1 chain interface takes
only indexed-bound TC handles. A direct SIMT call is not such a handle and
returns output in caller order, whereas the TC chain uses compact expert
order and FP16 completed values (or FP32 split partials).

The additive `dispatch_moe_create_v2` now accepts either an existing TC
handle or a selected SIMT call per projection. It owns a copy of the recipe
and arrangement, not the caller's tensors. Scratch is retained per chain.
No measured policy, GEMV arithmetic, offline format or TC collective changes.

| Projection | Input | Completed result | Chain adapter |
| --- | --- | --- | --- |
| SIMT gate/up | Original F32, caller order | F32, caller order | Shared prepare skips their gather; SwiGLU reads by the shared row map |
| TC gate/up | Compact FP16 | FP16, or FP32 split partials | Existing prepare and ordered reduction semantics |
| SIMT down | SwiGLU writes F32 in caller order | Final caller F32 | No standalone finish/scatter |
| TC down | SwiGLU writes compact FP16 | FP16 or FP32 partials | Existing fused finish/reducer to caller F32 |

The all-TC v1 path remains supported. The mixed stage gate covers every legal
SIMT mask, merged/separate gate/up, tokens1/2/4/8, and router on/off (80
contexts, seven changing replays). The 24 real chains use actual automatic
choices at tokens1/4/8, Q4 gate/up and Q4/Q5 down, independent GGUF dot
oracles, guards and three changing graph replays. Old all-TC PASS results
do not admit this new composition; its device result is still required.

The complete v0.3.0 caller build also catches the old compatibility route's
new `mm_ids_helper` argument. It passes `write_inverse=false`, preserving
the forward map consumed by that route's gather.

The old `run_kpack_fusion_box.sh` still defaults to the early
`kpack-fusion-v3/dispatch` package and the historical llama build directory.
Do not use that default as proof of this new integration. The final model
runner binds the new package and branch explicitly, keeps Asys separate
from reported timing, and exclude the first full PP/TG warmup pass.

Model inputs are the existing focused int4 plan:
`tools/kpack_batched_int4_2048.json` (Qwen3.5-35B-A3B-Q4_K_M MoE and
Qwen3-32B-Q4_K_M dense; NPL=1, PP=2048, TG=128). User files remain unchanged.

## Historical prebuilt package

Local publication: source `d93b118`, private llama `9b2fa0bf6`, artifact
`f2a2f99`. The complete PPU build (not just three translation units) passes;
the packaged server reports commit `9b2fa0bf6` under the local Ubuntu24
runtime loader. No PPU device is present locally. Host checks pass: 302
decode/indexed/policy/JIT tests, 51 model/MoE/prefill tests, 25 llama evidence
tests and four loader/cache CTest cases. The 25 ELF payloads are LFS objects;
all soname links and the manifest dependency closure are tracked. Package
size is about258MiB, mostly llama's ordinary CUDA backend, not a full
Quactlize sweep closure.

## Next box run

Run the following from the updated development checkout:

```bash
NCP_LIB_DIR=/sim/eec/shared/junfu.qx/ncp_flash_lib \
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
JOBS=192 CUDA_VISIBLE_DEVICES=0 bash tools/run_kpack_q4_model_box.sh
```

To build your local llama worktree instead of fetching the pinned caller,
set `LLAMA_CI_DIR` as well:

```bash
LLAMA_CI_DIR=/sim/eec/shared/junfu.qx/llama.cpp \
NCP_LIB_DIR=/sim/eec/shared/junfu.qx/ncp_flash_lib \
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
JOBS=192 CUDA_VISIBLE_DEVICES=0 bash tools/run_kpack_q4_model_box.sh
```

The override builds that working tree directly, including tracked local
edits and untracked inputs. It does not fetch, checkout, reset or clone
llama, and it does not require the default caller commit. Its `.aoneci`
script must support `LLAMA_BUILD_DIR` (private llama commit `bc4585dbc`);
update that build script first if the precheck reports the feature missing.
Output goes to `$RUN/ci/llama-build`, leaving the source's existing
`build-ci` untouched. NCP builds in its isolated pinned checkout unless
`NCP_CI_DIR` explicitly selects a matching existing build as described above.
The result receipt says `LOCAL_WORKTREE` and records the actual source
HEAD, working-tree status, tracked diff hash and build-script hash. Do not
edit the caller sources while the build is running. Unset `LLAMA_CI_DIR`
to retain the default isolated, pinned-source flow.

`tools/kpack_q4_model_artifact.json` separately pins the runtime-only LFS
package and the CI-enabled private llama source. This command builds NCP
FA/MoE and llama via `.aoneci`, then runs the gates and model phases. The
package supplies the paired producer, small runtime, seven gate parents
and mixed stage executable; it contains no llama binaries. No
full sweep or large Quactlize rebuild is required. Missing model-specific
parents are JIT-compiled outside capture during first use. The independent
CI source/build directory remains on disk for inspecting or packaging a
successful build.

The runner first requires both mixed gates, then collects GPU-native
reference/self/K-pack likelihood comparisons on local GSM8K text at token
batches1 and2048. These are logits/KLD tests, not a new GSM8K answer score.
It next performs unprofiled ABBA timings with two measured passes per
process after one excluded full PP/TG pass. Separate Asys requests use
the same prompt tokens in both arms and capture only the second request.
Dense-only models require dense trace evidence, not a nonexistent MoE call.
Any actual legacy compute fallback is rejected as a selected-performance
result. cuBLAS provider attribution can remain explicitly partial.

Return the printed `kpack-q4-model.*.results.tgz`. Large logits and complete
Asys/SQLite files remain on the box; summaries, kernel timings, source and
payload receipts are in the archive. Numerical quality and the no-slowdown
target remain review decisions, not conclusions from `runner_rc=0` alone.
