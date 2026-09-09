# Single-parent JIT and per-call ScaleFirst

The new candidate package is `prebuilt/ppu0010/kpack-jit-v1`. Its dispatcher
and execution libraries total approximately 1.1 MiB; it contains **no GEMM
modules**. The existing `kpack-native-v1` package is unchanged. This is not
yet a replacement for llama.cpp's six intake/admission/fallback libraries.

## Execution contract

1. The existing C++ heuristic selects one complete tactic: parent, provider,
   delivery N, split, scheduler and grid policy. JIT does not select a tactic.
2. A packaged parent is used if present. With explicit JIT opt-in, an absent
   selected parent is compiled or found in a content-addressed disk cache.
   An unknown family remains a policy miss, not a fabricated config.
3. Query and handle preparation happen before graph capture. The prepared
   run path has no compiler, subprocess, filesystem lookup or online timing.
4. A new shape may require preparation and one new parent. Prewarm expected
   shapes to avoid this first-use cost during serving. Prewarm is compilation,
   not device numerical/performance admission.

`quactlize_kpack_dispatch_enable_jit_v1` is additive to the v1 C ABI. It
copies explicit Python/helper/SDK/cache paths, before any module is loaded.
The compiler child is launched with an argument vector, not a shell/fork.
Only the selected tuple reaches it; a failed compile/load is an error and
does not silently choose a different kernel.

Cache identity covers headers/ABI, generator, flags, SDK libraries/tools,
host compiler, full parent tuple and generated source. The dispatcher also
embeds the expected source contract, rejecting a helper from another source
revision before compilation/loading. SDK changes produce different module
keys; matching source does not by itself admit a different SDK on device.
Per-key file locks and atomic publication serialize competing compilers.
Corrupt receipts/payloads and cache-path escapes fail closed. An interrupted
build without a receipt can be retried. A corrupt *published* entry is not
silently overwritten: use a fresh cache and preserve the broken entry for
inspection. Modules have no builder-specific SDK RPATH; source the target
SDK environment. Live handles retain their module after dispatcher close.

The helper uses Python's standard library. Importing the compile tools no
longer imports torch or initializes a device. Source headers and the SDK
are needed for a missing parent, but not for prepared execution.

## Prewarm before inference

Run from the Quactlize checkout with the PPU SDK environment loaded. These
commands do CPU compilation only; they do not run a sweep or touch weights.

```bash
python tools/kpack_jit.py plan --model /path/model.gguf \
  --tokens 1 128 512 --allow-misses --output /path/new-model-plan
python tools/kpack_jit.py prewarm --sdk "$PPU_SDK" \
  --plan /path/new-model-plan/plan.json --cache /path/kpack-module-cache --jobs 8
```

Model mode reads only the existing GGUF header/role authority. It currently
supports unsplit, non-TP models; repeat `--request Q ROUTE M N K E MAX_ROWS`
instead for explicit partitioned requests. Routes are 0 FQ-dense, 1 SF-dense,
2 FQ-grouped and 3 SF-grouped. Grouped M is tokens times top-k; max-rows is
the token bound, not an invented router. `--include-sf` additionally prewarms
SF; it does not choose SF. Q8, embedding and non-matmul tensors are omitted
with reasons. Inspect `plan.json`: `--allow-misses` records uncovered families
but does **not** make the plan complete. In particular, the Q6 output-head
family still needs its separate bounded performance gate.

For the updated llama.cpp feature branch, add these settings to the existing
working K-pack model command, retaining its intake bundle/pack-library/cache:

```bash
export QUACTLIZE_KPACK_EXECUTION=/path/quactlize/prebuilt/ppu0010/kpack-jit-v1
export QUACTLIZE_KPACK_JIT_PYTHON=/absolute/path/to/python3
export QUACTLIZE_KPACK_JIT_HELPER=/path/quactlize/tools/kpack_jit.py
export QUACTLIZE_KPACK_JIT_CACHE=/path/kpack-module-cache
# PPU_SDK must be the SDK root, with bin/hgcc and lib/.
```

This opt-in requires matching dispatcher/helper sources. Do not set these
variables with the old prebuilt dispatcher, which has no JIT entry point.
Automatic decode stays FQ; SIMT remains a parked, explicit diagnostic.

## ScaleFirst correction

The llama adapter now queues `sf_prepare -> GEMM` for **every SF call** on
the compute stream, including captured graph replays. Scale and zero use
per-stream reusable scratch, not per-weight cached values or a `scale_ready`
event. Allocation/plan preparation remains outside capture. There is no D2H
wait or CPU synchronization added to the launch path; K-pack disk persistence
still uses its independent background copy path.

`run_kpack_native_gate.py` poisons both metadata planes before replays and
times prepass plus selected GEMM/directory together. The exporter accepts
only these per-call receipts and emits `KPACK_PREFILL_POLICY_V2_PER_CALL`.
Old resident-only V1 timing tables must not be reused or hand-renamed. Missing
prefill policy retains FQ. The native model runner no longer automatically
repeats the parked SIMT sweep (`RUN_GEMV_GATE=1` explicitly requests it).

## Local evidence and next device boundary

Host tests exercise the actual dispatcher against stub DSOs: exact selected
tuple/recipe, JIT opt-in, single resolution, handle lifetime, helper errors,
identity mismatch, source-contract mismatch, malformed/oversized receipts,
cache relocation/concurrency/corruption and no compilation during run.
They also check model-header deduplication and reject resident-only SF data.

SDK compilation covers Q2-Q6 x dense/grouped x FQ/SF: 20 selected parents.
The initial 8-job cold compile took 318.7 s; all 20 hits took 0.35 s. An
additional cold Q4 grouped compile took 85.2 s and its hit 0.17 s. These are
local compiler wall times under concurrent work, not inference timings or a
promise for another CPU. Cold JIT latency remains a real limitation; use
prewarm/prebuilt caches, not synchronous multi-candidate tuning.

PPU validation remains required: cold/cached/restarted JIT execution,
equivalence to prebuilt parents, per-call SF numerical and graph checks, then
full-model latency. Local compile success is not device admission. The
updated llama adapter tag test poisons metadata and changes its source on
every eager/replayed call; it tests adapters, not GEMM arithmetic.

After waking, a standalone native gate (no model load, no full sweep) is:

```bash
python tools/run_kpack_native_gate.py --sdk "$PPU_SDK" \
  --bundle prebuilt/ppu0010/kpack-jit-v1 --jit-cache /workspace/kpack-jit-cache \
  --samples 5 --output /workspace/kpack-jit-native-new
```

Use one visible idle PPU and a fresh output directory. Uncached selected
parents compile on the CPU before their first launch; there is no fixed
short-duration promise for an empty cache. Return the output directory's
JSON summaries and console log. Do not drop the old intake libraries or
merge this candidate to product main until the relevant gates pass.
