# K-pack initialization-time tuning

Opt-in development runtime. The published six libraries and llama.cpp are
unchanged; the device gate below must pass before integration.

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
The 100-ms policy has host tests but its device cost is not yet admitted.

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

## Tracked sequence

- [x] Freeze the measured policy and retain explicit timing exceptions.
- [x] Audit DeepGEMM grouped/JIT and the existing configuration ABI.
- [x] Implement bounded warmup, grouped-aware timing cache and compile-only cache.
- [x] Implement resident-pointer modules using existing collective types.
- [x] Add host/ABI/cache tests and a five-format, four-route device gate.
- [ ] PPU gate: numerical closure, router/cache behavior and actual warmup costs.
- [ ] Bind/admit full tactic identity in the deployment loader; retain existing
  any-M admission until its runtime misses have a verified path.
- [ ] Integrate the admitted runtime into llama.cpp and update the single
  [integration handoff](LLAMA_CPP_KPACK_HANDOFF.md).

The old geometry-only `config_name` cannot encode AP, delivery-N, algorithm,
Split-K and grid recipe. New modules expose `quactlize_kpack_*_v1` in
`quactlize/runtime/abi.h`; the six-library ABI and published artifacts remain
unchanged. The next deployment step is a versioned loader for these full
identities, not silently translating a benchmark symbol to `config_name`.
