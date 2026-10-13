# Dynamic dense execution

This is an additive execution ABI, not a new selector, format or numeric
contract. Existing bound handles and all existing exports remain supported.
No device correctness/performance admission is implied by a host or compile
check. The PPU device gate must run against newly rebuilt modules.

## Selected tensor-core route

1. Use the existing dispatcher query: general, typed dense, explicit compute,
   or matched small-M. Keep its returned ticket, parent, algorithm and split.
2. Call `quactlize_kpack_dispatch_prepare_dynamic_v1(runtime, &choice, &plan)`
   outside capture. No A/C/workspace addresses are needed or retained.
3. Build a complete `qk_call_v1` for each invocation, then call
   `quactlize_kpack_dispatch_run_dynamic_v1(plan, &call)`.
4. Destroy with `quactlize_kpack_dispatch_destroy_v1` after device/graph uses
   have ended. A plan holds its loaded module alive even after runtime close.

Geometry, device, mapping, endpoint storage, compute precision and recipe are
fixed. A/C/low/high/metadata/zero/workspace/stream may change on every call.
The plan is immutable; launch parameters are private to each call. Run uses
the same producer and complete ordered reducer as the bound API. It does not
allocate, compile, query occupancy, initialize device resources or synchronize.
All launches use the supplied stream. Use disjoint output/workspace for
overlapping invocations, and retain allocations while a graph references them.
Graph replay uses the pointers captured into that graph; a subsequent run does
not rewrite an existing graph.

Single-parent modules also export `quactlize_kpack_prepare_dynamic_v1` /
`quactlize_kpack_run_dynamic_v1`. Typed modules export the corresponding
`quactlize_kpack_decode_dense_*_dynamic_v1`, accepting `qkd_dense_call_v2`.
Module-level preparation accepts a shape-only call: all data pointers and
workspace bytes must be zero; the optional stream is only a capture check.
Use the existing module destroy function. Bound and dynamic run functions
reject the wrong kind of handle.

This covers native dense FQ/SF (including Q8 SF/W8A16), and existing typed
M1..8 TC endpoints, including matched choices and explicit BF16 compute.
It does **not** expand the typed endpoint M domain or accept grouped tickets.
SF expansion is still supplied per invocation by the composition layer.
F32 storage does not mean FP32 internal compute: F16 plans still require
F16-representable activations. There is no clipping or silent BF16 switch.
SIMT/Q4 readers already accept complete dynamic calls and are unchanged.

Old modules still load for old callers. A new dynamic prepare on a module
without dynamic exports fails with a rebuild diagnostic, never with a
different kernel. Compiler source hashes invalidate old JIT entries; rebuild
the small dispatcher and selected parents together.

## Full dequant + BF16 provider

Use `quactlize_kpack_prefill_query_dynamic_v1`,
`quactlize_kpack_prefill_prepare_dynamic_v1`, and
`quactlize_kpack_prefill_run_dynamic_v1`. Prepare accepts a pointer-free
`qkp_call_v1`; query returns the required scratch size. Run receives the full
current call and stream (`call.weight.stream`). The arrangement, shape,
strides and dequant config remain fixed; current buffers are validated before
any enqueue. Expanded weights are not retained or reused across invocations.

Dense dynamic scratch includes an additional 4 MiB of caller-owned cuBLAS
workspace; the legacy scratch query is unchanged. Each enqueue sets stream
then workspace under a short host-side lock, with no device wait. This order
is required because setting a cuBLAS stream resets its workspace. Concurrent
streams have separate scratch. See the
[cuBLAS workspace contract](https://docs.nvidia.com/cuda/archive/13.0.3/cublas/index.html#cublassetworkspace).
The installed PPU provider must be tested independently; NVIDIA documentation
does not establish PPU runtime conformance.

The grouped full-BF16 provider also accepts dynamic GPU row maps and scratch;
this does not add a grouped TC dynamic API or enable llama MUL_MAT_ID.
Inspect status with `quactlize_kpack_prefill_device_status_dynamic_v1(plan,
&call, &device_status)` for that invocation. Do not use the bound status
accessor on a dynamic plan. Provider resolution/JIT is prepare-only. Warm the
installed provider eagerly before graph capture/timing, as with the old API.

## Validation and remaining admission

Host regression:

```bash
python3 -m pytest -q tests/test_kpack_dynamic.py \
  tests/test_kpack_native_dispatch.py tests/test_kpack_prefill_runtime.py
```

Compile the Q2..Q6/Q8 FQ/SF, F32 and BF16 typed matrix, plus an unchanged
grouped module (no device required; object files are not runnable artifacts):

```bash
python3 tools/check_kpack_dynamic_compile.py --sdk "$PPU_SDK" \
  --output /path/to/new-dynamic-compile-check --jobs 4
```

On PPU, a rebuilt dispatcher/module bundle can run the native F16 and typed
F32 endpoint gate (44 points, two address sets/streams, bound versus dynamic
bitwise checks, independent GGUF references and graph replay):

```bash
python3 tools/run_kpack_dynamic_gate.py --sdk "$PPU_SDK" \
  --bundle /path/to/rebuilt-bundle --output /path/to/new-dynamic-device-check \
  --jit-cache /path/to/selected-parent-cache
```

This gate checks the selected recipes, not every split/config. It is not
BF16-compute, full-provider or model admission; missing points fail coverage.

Device admission must compare bound versus dynamic execution with identical
selected recipes, independent numerical references, all supported qtypes,
S1/S2/S4/S8, changing A/C/weight/workspace addresses, two live streams,
capture/replay, guards and rejected shape/type/capacity changes. SF and full
dequant are per-call costs and must be tested as such. Test F32 values outside
the F16 range only against an explicitly suitable arithmetic contract, never
by accepting Inf/NaN or adding clipping.

Dynamic execution is **not** an all-M support proof. Load-time admission must
still establish a legal route throughout the declared M domain. Sampling a
few M values, or merely finding a dynamic symbol, is not sufficient. This
change deliberately does not replace a missing production choice with an
unmeasured tactic or invent an any-M guarantee.
