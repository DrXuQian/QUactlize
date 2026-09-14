# Per-call prefill composition

This is a development candidate. Local ABI tests and PPU compilation pass;
the new composed GPU path still needs the gate below. It does not inherit
whole-model admission from independent component timings or decode tests.

## Execution contract

The [component policy](KPACK_COMPONENT_POLICY.md) now has executable callers:

| Choice | Per-call work |
| --- | --- |
| FQ | Selected canonical K-pack GEMM, including its actual reducer |
| SF | Selected measured scale/zero expansion, then selected SF GEMM |
| Full BF16 dense | BF16 A conversion, measured full-weight expansion, installed SDK cuBLAS, F32 output |
| Full BF16 grouped | GPU active-expert directory and BF16 gather, selected active-weight expansion, installed DeepGEMM BF16 grouped GEMM, F32 indexed output |

Small-M decode and Q8_0 never enter full expansion. Their admitted execution
DSO is reused byte-for-byte. No offline format changes or Xplane fallback.
The existing FP16 and new F32/BF16 decoding endpoints remain available.
The prefill adapters above are not a return of standalone decoding casts.

Scale/zero and full weights are expanded on every invocation, including
graph replay. Grouped full expansion writes only the experts present in the
current GPU offsets; it retains `[E,N,K]` addressing so DeepGEMM can use its
own grouped entry. The allocation reserves E strides, but unselected weight
bytes are not expanded. Workspace is caller-owned and reusable per stream,
not allocated per weight. Input/output/weight/workspace spans cannot alias.

`quactlize/prefill/api.h` provides query, prepare, run, destroy, resident
status and provider-image accessors. Prepare binds immutable shape, pointers
and stream outside capture. Callers must finish outstanding work before
destroying a handle. Run contains no compiler, Python invocation, owned
allocation, D2H or CPU synchronization. One eager provider call must precede
capture to finish any provider-owned lazy initialization.

DeepGEMM prewarm uses its installed Python
`m_grouped_gemm_bf16_bf16_bf16_nt_nopad` heuristic and public compile-only
switch. Meta tensors supply shapes without allocating another weight copy.
The helper rejects a changed launch ABI or nonempty tuning space; it verifies
the generated source and argument list before handing its one compiled image
to C++. No copied DeepGEMM GEMM implementation or separate tactic selector.
The provider cache is trusted like the installed SDK; image/source hashes and
per-M receipts are retained for diagnosis. cuBLAS uses BF16 A/B/output, FP32
compute, default algorithm and disabled reduced-precision reductions, matching
the component measurement configuration.

## Local build

```bash
python3 tools/build_kpack_prefill_runtime.py --sdk /path/to/PPU_SDK \
  --output /data/prefill-runtime --jobs 4
python3 tools/build_kpack_dispatch.py --sdk /path/to/PPU_SDK \
  --output /data/composed-dispatch --execution-bundle /data/admitted-execution \
  --prefill-runtime /data/prefill-runtime --jit-only
```

The package has three DSOs, about 4.5 MiB total: host selector, unchanged
decode/execution, and the new prefill composition. It also contains bound
decode policy, source/build receipts and the compile-only provider helper.
No GEMM Cartesian-product bundle is rebuilt. GEMM JIT remains selected-only.
Payloads are published via Git LFS on `artifacts/kpack-prefill-runtime-v1`,
separate from `develop`; the launcher pins the exact artifact commit/hash.

Published source: `5947352afcebfa9c0036f74e6829b719b6a0b74d`; artifact:
`10053e67585eaddb3ffd758f243da64274199e27`; private llama caller:
`03142a8b7bdf690ac8c1ecfc51c9b20d4fbafc8e`.
Local checks: 142 focused library ABI/policy/package/JIT tests (including
native cuBLAS and DeepGEMM call signatures with host stubs), 24 llama Python
tests, four llama CTest cases, and PPU compilation of `ggml-cuda.cu`,
`quactlize-execution.cu`, `quactlize-execution-lib.cu`. No PPU GPU execution
or model performance result is claimed by these checks.

## PPU gate

Use an idle PPU with the SDK and DeepGEMM Python package already installed:

```bash
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
CUDA_VISIBLE_DEVICES=0 bash tools/run_kpack_prefill_runtime_box.sh
```

This does not build Quactlize on the box, resweep configurations, run llama,
or rerun the admitted decoding denominator. The only possible compilation is
DeepGEMM's selected first-use JIT. The gate covers nine composed contexts:

- Q2/Q3/Q4/Q5/Q6 dense M4096/N1024/K5120, each an actual automatic full-BF16 choice.
- Q3 grouped E256/top8/tokens4096, N2048 or N3072 and K512, with shared and
  per-slot activations. These are the two measured grouped full-BF16 winners.
- Every supported SF expansion config for the same weights is checked
  separately against the packed-unit FP16 oracle.
- One eager call and three graph replays with changed A, expert IDs, row maps
  and a sparse profile. Every output and every active/untouched weight expert
  is checked; padding and workspace guards must survive.
- A zeroed-code negative must disagree with the independent GGUF-derived
  BF16 dot oracle. Normalized absolute-product error must be below 0.005.

Only JSON/logs are archived to the printed `*.results.tgz`. A failed context
does not become a timing result. The outer Docker shell stays open.

## llama integration and remaining admission

The caller lives on private `dev/quactlize-v0.3.0`, above the exact
release-plus-patch `dev/v0.3.0` base. No original feature branch is rebased.
Optional prefill API binding is additive. A missing provider narrows the
policy mask; an explicitly present broken package fails instead of silently
using different weight bytes. Grouped full BF16 requires
`QUACTLIZE_KPACK_DEEPGEMM_HELPER` pointing to the packaged helper and
`QUACTLIZE_KPACK_JIT_PYTHON` pointing to the installed provider environment.

Remaining after this gate: new-branch model accuracy, first-use-excluded
latency and complete provider trace attribution. In particular, the SDK
cuBLAS shim is not itself a device image; the trace reader conservatively
reports its untraced provider separately rather than treating a decode TC
kernel as proof of prefill execution. Router-dependent regret and adapter
costs remain the debts documented with the component policy.
