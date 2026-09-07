# K-pack initialization-time tuning

Opt-in development runtime. The small and real-shape numerical gates below
passed on 2026-09-07. The published six libraries and llama.cpp are unchanged;
deployment admission remains pending.

**Deployment direction:** deterministic heuristic -> one complete tactic ->
prebuilt/image-cache hit or on-demand JIT -> execution. Multi-candidate timing
and independent CPU oracles stay in development tools, not default model
startup. The measured exact table is calibration evidence/a known-point
reference; the new general heuristic is not implemented or admitted yet.

The runtime follows the separation in DeepGEMM-for-sail: select a small
configuration set, compile/cache its images, optionally time candidates during
initialization, then launch a cached choice. The local source inspected is
`f89eae1`, particularly `deep_gemm/jit_kernels/tuner.py` and
`deep_gemm/jit/{compiler,runtime}.py`. Several grouped callers pass `space=()`;
that compiles one chosen configuration and does not search. We do not copy
its 20-repeat/L2-flush/8192-square-GEMM prelude into lightweight warmup.

## Runtime contract

| Stage | Implementation | Work performed |
|---|---|---|
| Candidate proposal | `runtime/candidates.py` | At most five measured parent configurations, three runtime recipes each; no Cartesian sweep |
| Compile-only warmup | `runtime/compiler.py` | CPU-only HGCC compilation; content-addressed, locked, atomic `.so` cache |
| Explicit tuning | `runtime/tuning.py` | Correctness callback, two warmups, three timing samples, up to five repeats per sample; fewer repeats for long kernels |
| Cached selection | `Tuner.select` | Cache lookup plus actual-request admissibility; no compilation, profiling or hidden stream synchronization |
| Prepared execution | `runtime/abi.h` | Resident pointers, explicit algorithm/S/grid; includes required grouped directory build or dense reducer |

All paths use the existing production collectives and canonical K-pack bytes:
Q2–Q6, fully-quantized/ScaleFirst, dense/grouped. No offline reshuffle,
converter, mainloop or Xplane fallback is introduced. Q4 grouped uses the
measured **separate-half** metadata publication in both FQ and SF; its dense
factory retains its own default. `tests/kpack_warmup_type_parity.cu` checks
kernel type equality to the measured harnesses, with no benchmark dependency
in the runtime module itself.

Supported requests have positive M, N divisible by 256 and K divisible by
256 for Q2/Q4/Q5 or 512 for Q3/Q6. Individual parents retain their transport,
pipeline, resource and AP/M domain checks. A/outputs are contiguous FP16;
weights are converter-native K-pack planes. FQ consumes packed units; SF
consumes already prepared FP16 scale/zero planes. Repacking, SF prepass and
allocation are outside prepared kernel timing.

The default tuning budget is **100 ms, soft, excluding compilation**. It is
checked between candidates: a running kernel cannot safely be interrupted to
meet a wall-clock deadline. First-use compilation can take minutes and belongs
in a separate startup/precompile phase. Tuning is forbidden during graph
capture. Keep the incumbent if it is within 5% of the best measured candidate;
this is noise tolerance, **not a global-optimality or 5%-accuracy guarantee**.
The real-shape experiment exceeded this soft budget on 47/90 contexts. It is
not an admitted 100-ms startup guarantee and is not the default deployment path.

Numerical or runtime failure aborts that tuning context. Only a pre-launch
unsupported tactic is skipped. There is no default chosen because a library
happens to contain it: an unhandled miss returns `FALLBACK_REQUIRED`. Callers
may supply an explicitly admitted fallback.

## Grouped cache and changing routers

A grouped bucket includes weight shape, format/mapping, expert count,
`ceil(total_rows / experts)` band, max-row band and active-expert band. The
actual row vector remains part of the measurement identity. Reusing the
bucket with different rows is reported as `BUCKET_HINT`, not an exact timing
claim. Every query validates the actual vector and recomputes its CTA count
and capacity/balanced grid. Empty experts and ragged rows are supported;
total rows must be nonzero.

Timing caches bind device/CUs, SDK, kernel code, candidate catalog and runtime
selection code. Compiled caches separately bind compiler, flags, source and
payload hash. Stale or corrupt caches are rejected. Provide the complete
candidate catalog to `NativeBackend` so lazy loading another subset does not
unnecessarily invalidate the timing cache.

Each backend owns one workspace for **sequential** prepared executions on its
stream. Use separate backends/workspaces for concurrent inference. Finish
work before destroying a prepared handle. Grouped preparation copies its
shape/pointer/stride tables; reprepare when the actual rows or pointers change.
The caller supplies matching host/device rows and offsets; there is no hidden
D2H read or verification in the hot path.

## Application sequence

```python
from quactlize.runtime import Request, Tuner, TuningCache
from quactlize.runtime.candidates import MeasuredCandidates
from quactlize.runtime.compiler import Compiler
from quactlize.runtime.native import NativeBackend

request = Request("fq-grouped", 12, 512, 3072, 6, (2, 0, 3, 1))
seeds = MeasuredCandidates("policies/kpack_zw810_runtime_v1.json")
tactics = seeds.shortlist(request)
# Startup/precompile: CPU work only. Preserve records for later loading.
compiler = Compiler(sdk_path, image_cache_path, jobs=8)
records = compiler.compile_only(seeds.parent_union(tactics))

# Device warmup: caller owns valid resident buffers and an oracle callback.
backend = NativeBackend(sdk_path, records, resident_buffers, check_output,
                        stream=stream_pointer, catalog=seeds.parents)
cache = TuningCache(backend.identity, timing_cache_path)
tuner = Tuner(cache)
choice = tuner.warmup(request, tactics, backend)
if choice["status"] == "FALLBACK_REQUIRED":
    raise RuntimeError("request needs an explicitly admitted fallback")

# Inference: selection does not retime. Prepared runs can be repeated.
choice = tuner.select(request, backend)
handle = backend.prepare(request, choice["tactic"])
try:
    backend.run(handle)
finally:
    backend.synchronize()
    backend.close(handle)
    backend.release()
```

The oracle may compare against an admitted reference kernel during startup.
The box gate instead uses independent official GGUF dequantization. No Python,
PyTorch or JIT dependency is required by the **C module ABI**; a C++ caller can
load already compiled modules and manage selection/cache itself. The Python
driver is the initial orchestration implementation, not yet a llama.cpp plugin.

The family-neighbor proposal here is deliberately simple. Prior validation of
the experimental learned shortlist does **not** certify this new proposal or
unknown shapes. Exact measured policy entries are seeds; local timing makes
the final warmup choice.

## Small PPU gate

### Reviewed box result (2026-09-07)

Archive `kpack-warmup-results.EpuIw8.tgz` closes this small module gate:
50 parent modules, 50 exact contexts across 20 format/route pairs, 190 measured
candidate-contexts, 250 positive checks and 50 detected zero-low negatives.
All ten changed-router bucket replays pass. The ten rejected tactics are the
expected SF-dense TM8/M=9 domain exclusions, not numerical or launch failures.
Maximum condition-scaled error is `1.734967e-4` against the unchanged `5e-3`
bound; the smallest planted error is `9.335464e-2`.

| Route | Contexts | Median warmup ms | Maximum warmup ms |
|---|---:|---:|---:|
| FQ dense | 10 | 1.581 | 13.490 |
| SF dense | 10 | 1.265 | 1.469 |
| FQ grouped | 15 | 1.838 | 2.427 |
| SF grouped | 15 | 1.530 | 2.024 |

Total wall time was 254.751 seconds, including 190.906 seconds of compilation
with 16 jobs. The 50 warmup calls together used 89.476 ms. The remaining
63.845 seconds outside compilation also includes oracle/fixture construction,
module loading and other harness work; it is not all tuning time.

Source/kernel, compile flags, generator and runtime-cache identities replay
against `e47613f`. The box SDK/GCC differs from the local compile workstation;
each result is bound to the box's own SDK and module receipts. The upload
contains no ELF payloads, so their hashes were checked by the device runner,
not independently rehashed from this archive. Exact hashes and counts are in
[the review receipt](KPACK_WARMUP_GATE_RESULT.json).

This closes only `N=256,K=512` small-M/router coverage. It neither certifies
large-shape 100-ms warmup nor proves a within-5% choice on real model shapes.
Do not rerun this completed suite unchanged; the next device work is the
bounded real-shape/selected-parent check.

### Reproduction

Local validation: 100 host tests passed; HGCC compiled all 50 small parent
modules in 473.749 seconds with eight jobs, followed by 50 cache hits. Each
module exports all seven C entry points. Ten format/metadata-mode builds of
the type-parity test passed for TM8/TM16, dense/grouped and both schedulers.
These are compilation/ABI checks, **not PPU numerical or performance results**.
Compile-only JIT currently uses the source checkout and its pinned headers;
the development image cache is local to the SDK installation, not a relocatable
replacement for the published runtime bundle.

From an updated `develop` checkout, with the pinned submodule initialized:

```bash
(
  set -o pipefail
  export CUDA_VISIBLE_DEVICES=0
  SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK
  source "$SDK/envsetup.sh" &&
  python3 tools/run_kpack_warmup_gate.py \
    --sdk "$SDK" \
    --cache /workspace/kpack-warmup-v1-jit \
    --output /workspace/kpack-warmup-v1-results \
    --jobs 16 \
    2>&1 | tee /workspace/kpack-warmup-v1.console.log
)
```

The host needs `gguf`, `numpy`, and `torch` for the independent oracle. SDK
compilation needs no device and is separately available with `--compile-only`.
This builds **50 small parent modules**, checks a second compile-only pass is
all cache hits, then runs **50 numeric contexts** across 20 format/route pairs.
It tests dense M=1/9, grouped `[9,7,0,0]`, `[8,8,0,0]`, `[2,0,3,1]`, exact
cache replay, same-bucket router changes, and a zeroed-code negative for each
context. Outputs are poisoned before every correctness launch. Device timing
is full-output resident timing, not modeled Split-K time.

The gate uses a 5-second per-context soft budget to exercise all small gate
candidates; it does not certify the production 100-ms budget or full-shape
performance. No giant bundle or campaign is rebuilt. Existing compiled cache
entries survive a failed run. The output directory must be new/empty; use a
new name for a rerun while preserving the compiled cache.

Return `summary.json` and `results.json`. If it fails, return the console and
the named parent `build.log`; do not recompile all old bundles.

## Next gate: bounded real-shape selection

`tools/run_kpack_warmup_real.py` extends the completed small gate, using
`tools/kpack_warmup_real_plan.py` as its fixed denominator. This run is now
complete; the commands below are reproduction instructions, not another test
request. See the reviewed result below.

| Route | Contexts across Q2–Q6 | Coverage |
|---|---:|---|
| FQ dense | 30 | M=1/4, N/K=1024/5120, 8192/5120, 5120/25600 |
| SF dense | 20 | M=2048 on those three families, plus M=3072 at N1024/K5120 |
| FQ grouped | 20 | E=256, decode/prefill at N512/K3072; boundary/changed routers at N3072/K512 |
| SF grouped | 20 | Same actual expert-row vectors as FQ grouped |

The plan contains **90 contexts, 173 distinct compiled parents and 868
candidate-contexts**, never more than five parents / fifteen recipes per
context. Grouped decode/prefill have total rows 8/16384; boundary/changed
routers have 528/534 rows, max 129, and 9/12 active experts. Empty experts
remain explicit. Eighty-five incumbents are exact measured policy entries.
The five new-M controls explicitly borrow the M=2048 incumbent; they are
not labelled as measured M=3072 evidence.

Each context first runs the actual 100-ms soft-budget tuner and tests cache
replay. A **separate validation phase** measures the entire bounded pool in
forward then reverse order, with two warmups and at most five repeats per
sample. It includes the historical incumbent in the same run; historical
microseconds are not used as a cross-machine baseline. A previous bucket hint
may add one confirmation-only recipe. This phase tests what the budgeted tuner
missed, not a new Cartesian sweep. Candidate observations, full recipe identity,
resolved grid, shared/workspace bytes and occupancy are recorded. Per-context
`tuning_ms`, fixture time and total case time remain separate.

All output elements are checked against official GGUF dequantization with
the unchanged condition-scaled error bound `5e-3`. Weights are random, finite
GGUF blocks at real dimensions; these are not actual model checkpoint values.
The benchmark A is dense, FP16-exact, row-tagged and rank two, with four random
K categories. This gives an exact factorized CPU oracle for the **same A used
in timing**, avoiding a huge CPU GEMM for every candidate. Host tests compare
this oracle to a full independent FP64 GEMM. The vectorized fixture packer is
test-only: its low/high/units bytes and half metadata are checked against the
existing scalar reference for all five formats. It does not replace the
offline format or production converter. Every context also detects a zeroed
low-plane fault.

The bounded-pool verdict is `WITHIN_BOUNDED_POOL_5PCT`, `BOUNDED_POOL_GAP`, or
`TIMING_NOISE_REVIEW` (either round differs by over 5%). A numerical pass does
not certify a global optimum or authorize deployment. Performance gaps remain
results, not reasons to discard other measurements.

### Reviewed real-shape result

Archive `kpack-warmup-real-results.LJGzK2.tgz`, SHA-256
`10a00ae918f719eee980eabff0b6006370142c1f9ad1be24a4d032a13a044440`,
replays against `ef392a4`. All 90 case digests, the exact plan, source/kernel
hashes, SDK/compiler contracts, 173 generated module keys and runtime catalog
identities agree. Binary payloads were not uploaded; their bytes/sizes were
not independently checked locally.

- Numerical closure: 2,322 positive checks, 90 detected zero-low negatives,
  90 exact cache replays and 15 changed-M/router bucket checks. Maximum
  condition-scaled error `2.7683903911088925e-4`, minimum planted error
  `2.0179433535854226e-2`; bound remains `5e-3`.
- Warmup timed 571 candidate-contexts; independent confirmation covered all
  868, with no rejected or missing candidate. Original verdicts remain
  80 `WITHIN_BOUNDED_POOL_5PCT`, one `BOUNDED_POOL_GAP`, nine
  `TIMING_NOISE_REVIEW`.
- Selected medians are within 5% of the same-run pool in 89/90 cases. The nine
  noise flags originate from pool-wide spread; each of their selected
  candidates remains within 5% even comparing its slower sample with the
  fastest sample in the whole pool. This is a two-round observation, not a
  universal performance bound or grounds to erase the original flags.
- No selected median is slower than its same-run historical incumbent;
  80 choices retained the incumbent and ten changed. This is not an Xplane
  comparison or proof of unseen-shape heuristic quality.
- Total wall time 740.597 s; all 173 cold builds took 353.733 s. The 90 warmup
  calls together took 7.494 s, not 740 s. Warmup includes preparation, checks
  and synchronization; fixture/confirmation work is separate.

| Route | Median warmup ms | Maximum ms | Soft-budget exhaustion |
|---|---:|---:|---:|
| FQ dense | 21.957 | 46.689 | 0/30 |
| SF dense | 142.857 | 256.564 | 20/20 |
| FQ grouped | 103.295 | 148.179 | 12/20 |
| SF grouped | 107.949 | 171.606 | 15/20 |

The clear miss is Q6 SF grouped, N3072/K512/E256, 528 rows, max 129 and nine
active experts. The tuner measured only candidates 1–2 before its soft budget
expired at 107.697 ms. Selected persistent geometry `16x64x128_w16x16_s3_dn32`
took 34.156 us; candidate 4, nonpersistent `16x128x128_w16x32_s2_dn64`, took
29.704 us (14.988% selected overhead). The selected choice is still 4.964%
faster than the remeasured incumbent. Recorded profiling batches account for
1.126 ms; the log does not separately attribute the remaining preparation,
correctness, synchronization and host overhead. Do not invent that attribution.

The final choices use 52 distinct parents across all formats/routes. Within
this suite, Q4 dense FQ/SF uses four. These are useful reduction targets, not
proof that those parents cover every workload or that library bytes shrink in
the same proportion. No libraries have been trimmed. Single-parent cold-JIT
latency still requires a separate measurement; dividing parallel build wall
time by 173 would not measure it.

### Run on box (reproduction only)

Use one otherwise idle PPU. This first compiles **small cached modules**, not
the old sweep bundle; existing matching cache entries are reused. The default
run covers all five formats. The host needs the same NumPy/gguf/PyTorch and SDK
as the completed small gate.

```bash
(
  set -eo pipefail
  export CUDA_VISIBLE_DEVICES=0
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK
  OUT=/workspace/kpack-warmup-real-v1
  source "$SDK/envsetup.sh"
  python3 tools/run_kpack_warmup_real.py \
    --sdk "$SDK" --cache /workspace/kpack-warmup-v1-jit \
    --output "$OUT" --jobs 32 \
    2>&1 | tee "$OUT.console.log"
)
```

`KPACK_REAL_COMPILE`, `KPACK_REAL_FIXTURE`, `KPACK_REAL_CASE` and
`KPACK_REAL_PROGRESS` show progress. The 25 weight families run sequentially
in fresh processes; a failed family does not stop the others. A completed
context is atomically saved only after numeric, negative, cache and confirmation
checks. Use the **same output and `--resume`** after interruption; only unfinished
contexts run again. Source/plan/SDK must match the prior authority. A source fix
needs a new result directory, but unchanged compiled parents still hit the
image cache. No results are silently invalidated or deleted.

`--plan-only` needs no SDK/device. `--compile-only` builds the declared union
without launching a kernel; use `--resume` to continue into device execution.
Local verification compiled twenty representative parents (five formats ×
four routes, including large tiles and Q4 packed-A) with eight jobs in 276.6 s.
All twenty expose the seven declared C symbols and hit the compiled cache on
replay. The 116 host tests cover fixture-byte/metadata parity, the independent
oracle, actual driver orchestration, cache/resume and existing policy behavior.
These are host/ELF checks, not device launches; the local Conda/SDK dynamic-load
environment is not admitted, and no system libraries were changed to mask it.
This is a compile-only measurement, **not a box campaign duration estimate**.

Return the small evidence files, not the JIT `.so` cache:

```bash
(
  set -eo pipefail
  OUT=/workspace/kpack-warmup-real-v1
  tar -czf /workspace/kpack-warmup-real-v1-results.tgz \
    -C "$OUT" plan.json authority.json modules.json summary.json results.json
)
```

If the final status is `INCOMPLETE`, also send the console and `failures/*.json`;
they name the exact request and active tactic. An interrupted run may not yet
have summary files; resume it or send the console and completed `cases/` receipts.

## Tracked sequence

- [x] Freeze the measured policy and retain explicit timing exceptions.
- [x] Audit DeepGEMM grouped/JIT and the existing configuration ABI.
- [x] Implement bounded warmup, grouped-aware timing cache and compile-only cache.
- [x] Implement resident-pointer modules using existing collective types.
- [x] Add host/ABI/cache tests and a five-format, four-route device gate.
- [x] Small PPU gate: numerical closure, router/cache behavior and measured costs
  on the 50 fixed small contexts (2026-09-07).
- [x] Real-shape bounded experiment: 90 numeric contexts complete, full bounded
  pool confirmed; retain the one selection gap and timing/budget observations.
- [ ] Calibrate a small deterministic heuristic from existing evidence; return
  one complete tactic, with no default online tuning or Cartesian compilation.
- [ ] Bind/admit full tactic identity in the deployment loader; retain existing
  any-M admission until its runtime misses have a verified path.
- [ ] Integrate the admitted runtime into llama.cpp and update the single
  [integration handoff](LLAMA_CPP_KPACK_HANDOFF.md).

The old geometry-only `config_name` cannot encode AP, delivery-N, algorithm,
Split-K and grid recipe. New modules expose `quactlize_kpack_*_v1` in
`quactlize/runtime/abi.h`; the six-library ABI and published artifacts remain
unchanged. The next deployment step is a versioned loader for these full
identities, not silently translating a benchmark symbol to `config_name`.
