# New-machine startup and trace recovery

Source: `kpack-q4-model.C99gWo.results.tgz`, SHA256
`0706eca48781cf906c38814c2737172cb7508ffb7e726ce551559aff92a46ca4`.
The four ABBA processes passed. The final failure was reference Asys session
creation, before the profiled server started. Do not discard the benchmark.

Caller: `d906245bca9ea1397a3df71207af9157438b3369` on the owner's
`dev/quactlize-gate-up-v0.3.0`; cache fix is `99f3c24b7` in its history.
Source branch: `dev/gemv-model-tuning`. Runtime pin remains `87b996559f`.

| Independent issue | Evidence | Change | Remaining box check |
| --- | --- | --- | --- |
| Cold selected-parent JIT | Nine distinct misses, 450.902 seconds in total | Precompile these exact nine parents in one bounded pool, including the three typed decode modules; normal cache keys and compiler locks | Prewarm wall time, then no corresponding model JIT misses |
| Cache publication | Background copy finished; `RENAME_NOREPLACE` returned `EINVAL` | For runtime caches only, exclusive directory creation, hard-link weights, publish manifest last; never replace another cache | First process publishes on `/sim`; second process reports `ready`, no background repack/write |
| Asys service startup | `session ... create timeout`, reference server never starts | No-model session probe, one retry for session creation, unique-session cleanup; trace-only continuation | Service and actual GPU capture on the new box |
| DeepGEMM compiler discovery | `hgcc compiler not found` during native TP2 startup | Explicit `DG_JIT_HGCC_COMPILER`; preserve it through profiler launch | Compiler version and native TP2 JIT |

The prewarm list is not a selector or a new tactic table. It is the exact
module closure returned by the successful Qwen3-32B run, checked against the
runtime manifest, source contract, token workload and GGUF weight geometry.
Unknown models/workloads retain on-demand JIT. No GPU work or autotuning is
performed during this prewarm. The runtime, compute choices and K-pack format
are unchanged. No new LFS objects or llama binaries are published.

Local checks: 39 JIT/tool tests, 171 model/harness tests, 38 caller harness
tests and the real host `test-kpack-sidecar` pass. Sidecar tests inject
unsupported rename flags, collisions and link errors using child-process
syscall filters. Successful publication keeps the weight inode, so it does
not copy the payload. The uploaded ABBA receipts and all nine recorded parent
tuples also pass a direct recheck. These checks are not a replacement for
the new machine's filesystem/profiler validation.

## Immediate native TP2 compiler fix

After sourcing the SDK and before launching the native benchmark:

```bash
export PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK
source "$PPU_SDK/envsetup.sh"
export DG_JIT_HGCC_COMPILER="$PPU_SDK/bin/hgcc"
test -x "$DG_JIT_HGCC_COMPILER"
"$DG_JIT_HGCC_COMPILER" --version
```

DeepGEMM's `HGCCCompiler` reads this override before querying the compiler.
Changing it does not require rebuilding llama or Quactlize. A previously
unseen TP2 shape can still require its normal first JIT. This is a native
TP2 baseline, not K-pack TP2 admission: K-pack does not yet implement the
Meta backend's sharded tensor intake. Do not silently substitute layer split.

## Retry only the failed Asys phase

Do this before rebuilding the old caller directory. It uses the original
caller binaries, but the new Python profiler harness. Binary, runtime,
component-gate and complete ABBA receipts are rechecked; changed binaries or
another GPU ordinal are refused. Output is a fresh sibling directory.

```bash
(
    set -e
    cd /sim/eec/shared/junfu.qx/quactlize
    GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only origin dev/gemv-model-tuning
    GIT_LFS_SKIP_SMUDGE=1 git -C /sim/eec/shared/junfu.qx/llama.cpp pull --ff-only \
        https://github.com/DrXuQian/llama.cpp.git dev/quactlize-gate-up-v0.3.0

    PREVIOUS_RUN=/sim/eec/shared/junfu.qx/kpack-newbox.QYl8F9/kpack-q4-model.C99gWo \
    LLAMA_CI_DIR=/sim/eec/shared/junfu.qx/llama.cpp \
    PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
    QUACTLIZE_PPU_BUNDLE=/sim/eec/shared/junfu.qx/kpack-newbox.QYl8F9/legacy/prebuilt/ppu0010/2826cf1/runtime6-46fc3096e1a1/bundle \
    CACHE_DIR=/sim/eec/shared/junfu.qx/kpack-newbox.QYl8F9/cache \
    JIT_CACHE=/sim/eec/shared/junfu.qx/kpack-newbox.QYl8F9/jit \
    TRACE_ONLY=1 CUDA_VISIBLE_DEVICES=0 \
    bash tools/resume_kpack_q4_model_box.sh
)
```

This does not recompile or repeat ABBA/numerical tests. The two model processes
still load and warm up before capture. If the session probe fails twice,
inspect `trace/<model>/<arm>/asys-preflight/*/launch.log` and `environment.log`;
no model was loaded by that failed probe. Do not kill other users' sessions.

## Validate cache publication with the repaired caller

After the trace retry, use the normal runner with a fresh cache root. This
does rebuild the caller through `.aoneci/scripts/build.sh`; it does not
rebuild Quactlize's runtime package. Existing valid JIT objects are reused.
The recorded nine-parent prewarm runs automatically before model evaluation.
This command creates a separate caller build, leaving the original binaries
intact. If a matching local caller build is already available,
`LLAMA_CI_BUILD_DIR` enables incremental compilation; do not point it at a
build in use by the concurrent 122B run. Keep the measured GPU idle.

```bash
(
    set -e
    cd /sim/eec/shared/junfu.qx/quactlize
    QZ_CACHE=$(mktemp -d /sim/eec/shared/junfu.qx/kpack-newbox.QYl8F9/cache-fixed.XXXXXX)
    test -d "$QZ_CACHE"
    RESULT_ROOT=/sim/eec/shared/junfu.qx/kpack-newbox.QYl8F9 \
    MODEL_PHASES=perf MODEL_NAMES=qwen3-32b-q4km MODEL_COMPUTE=fp16 MODEL_GATE_UP=0 \
    LLAMA_CI_DIR=/sim/eec/shared/junfu.qx/llama.cpp \
    NCP_LIB_DIR=/sim/eec/shared/junfu.qx/ncp_flash_lib \
    NCP_CI_DIR=/sim/eec/shared/junfu.qx/kpack-newbox.QYl8F9/kpack-q4-model.tnCy8u/ci/ncp_flash_lib \
    PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
    QUACTLIZE_PPU_BUNDLE=/sim/eec/shared/junfu.qx/kpack-newbox.QYl8F9/legacy/prebuilt/ppu0010/2826cf1/runtime6-46fc3096e1a1/bundle \
    CACHE_DIR="$QZ_CACHE" JIT_CACHE=/sim/eec/shared/junfu.qx/kpack-newbox.QYl8F9/jit \
    CUDA_VISIBLE_DEVICES=0 JOBS=192 JIT_JOBS=9 \
    bash tools/run_kpack_q4_model_box.sh
)
```

Required cache evidence: `1-kpack.log` reports publication, `2-kpack.log`
reports `[kpack-cache] ready` and does not start a background write. Keep
that cache for later runs. An existing target is never overwritten.
If the filesystem also lacks hard links, publication remains a visible
failure; it does not claim a hit or copy another 18 GB behind the scenes.

## Valid steady-state result from the uploaded run

Qwen3-32B-Q4_K_M, one request, PP2048/TG128, first complete pass excluded:

| Metric | Native | K-pack | Change |
| --- | ---: | ---: | ---: |
| Prefill total | 571.367 ms | 557.238 ms | -2.473% |
| TPOT | 33.3503 ms | 21.3962 ms | -35.844% |

Those are benchmark intervals, not cold process startup. The Asys and cache
fixes do not establish an additional steady-state speedup. Fresh model
accuracy was not measured in this perf-only run.
