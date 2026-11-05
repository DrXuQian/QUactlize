# Flash-Next serving interfaces

Public headers: `quactlize/compat/api.h` (unchanged v1) and
`quactlize/compat/serving.h` (additive helpers). No framework files or existing
PPU matmul selection thresholds change. GPU qualification is separate from
compilation and host/fake/export tests; neither qualifies PPU performance.

## Canonical preparation for Flash-Next

Use the existing packing API in the same `libncp_quactlize.so`:
canonical arrangement -> sizes -> GPU repack. Q5_1 (qtype 7) now has its own
mapping `0x5135314b50000001`, with N%16/K%128. Its low4/high1 planes retain
unsigned codes and exact FP16 `{d,min}` bytes in `[E,K/32,N,4]`. It is not
Q5_K, does not requantize, and keeps the same total weight bytes as raw GGUF.
Q8_0 retains the existing K-pack2 bytes/identity with N%256/K%32; K640 and
K320 are legal. Query/prepare still reject incompatible consumer geometry.

`prepare_gate_up_dev_for_arrangement_v1` accepts separate N640 raw sources
when combined N1280 is legal, placing `[gate;up]` **within each expert** with
no concatenation allocation. This is not paired-N4/SwiGLU storage; Q5_1 N4
fusion remains unsupported. The format extension does not enable old F32
dense/MoE selectors for Q5_1 or relabel the compatibility kernel as tuned.

The native16 compatibility consumer supports these packed planes with its
existing F32 reconstruction -> one F16/BF16 weight rounding -> FP32 accumulation
-> native16 output contract. The Box suite
checks E512/top10 down N2560/K640 at M1..8/9/16/128, shared scratch, dynamic
IDs, inactive NaN inputs, sticky errors and repeated graph replay.

## Direct packed Q5_1 candidates

In the same public DSO/header, use `quactlize_ppu_matmul_packed_query_v1` and
`quactlize_ppu_matmul_packed_prepare_v1`, followed by the existing run/destroy.
Python binding: `Library(path, packed_route=0)`. Route 0 is a provisional
AUTO (SIMT M<=8, TC M>8); force 2/3 to qualify SIMT/TC on either side of that
boundary. This is not a measured optimal selector or an update of old tables.
Only canonical Q5_1 is admitted; Q8/K-quants and raw storage are rejected.

Both preserve F16/BF16 A/C, separate F32 weight multiply/add followed by one
native16 rounding, and F32 accumulation. SIMT vectorizes across N and reuses
K32 affine metadata. TC coalesces unique low/high/metadata words into a K128
shared tile and decodes directly into native PPU m16n16k16 registers. No global
native16 weights, grouped A or grouped C are materialized. This is a separate
direct-packed implementation, not a renamed full-dequant or Q5_K kernel.

Query reports route, N tile and exact workspace. SIMT and dense TC use zero
scratch; indexed TC uses only three 256-byte-aligned `4*(E+1)` route arrays
and one aligned `4*M*topk` slot map: 12,032 bytes at E512/top10/M128. Shared
memory is CTA-local (512 bytes SIMT; 10,304 bytes TC), independent of model
size. Use disjoint scratch/status/output for concurrent calls; serial layers
may reuse max(query). No run-time compilation/allocation/sync/D2H is added.

Native object compilation and host mapping/rounding tests are not PPU
correctness/performance admission. Existing default v1 remains the control.
The narrow Box correctness command is:

```bash
git fetch origin feat/ppu-flash-next
git show FETCH_HEAD:tools/run_flashnext_box.py | python3 - --ref FETCH_HEAD --device 0 --packed-only
```

It checks real K640/E512/top10, forced-route N/M tails, changing pointers and
IDs, inactive NaN input, sticky errors and concurrent graph replay. Upload
the printed review archive; speed/default promotion needs separate paired
same-device performance evidence after this correctness gate.

## Contracts

| Operation | Inputs → outputs | Semantics |
|---|---|---|
| `quactlize_serving_router_v2` | BF16/F32 logits `[capacity,E]` → F32 selected logits, I32 IDs, F32 weights `[capacity,topk]` | Original-logit descending top-k, larger ID wins exact ties; selected-only stable softmax × multiplier. Only softmax + renormalization, no bias. |
| `quactlize_serving_swiglu_v2` | Native F16/BF16 gate/up `[rows,H]` or `[gate;up]` per `[rows,2H]` row → native `[rows,H]` | `round_silu=1` rounds SiLU to native16 **before** multiplying up; `0` rounds only the final product. No clipping. |
| `quactlize_serving_gather_rows_v1` | Raw GGUF bytes + I64 IDs `[capacity]` → native16 `[capacity,columns]` | Q8_0 (34B/32), IQ4_NL (18B/32). F32 decode, one target rounding. IQ4_NL width160 is 90B/row. No canonical reinterpretation. |
| `quactlize_serving_merge_v1` | Native slot values `[M,topk,N]`, F32 weights → **F32** `[M,N]` | Fixed slot order, F32 accumulation. Framework BF16 output conversion, if wanted, is explicit. |

All row strides are **elements**, the inner dimension is contiguous. Row
lookup is fully contiguous. Capacity/byte sizes are host metadata; live counts
are optional **device I32 pointers**. NULL count means all rows. Inactive
router rows become `(selected=0,ids=-1,weights=0)` without reading logits;
inactive lookup rows become zero without reading IDs or raw. Indexed matmul
then propagates `-1` through both projections without reading invalid weights.
An active lookup ID `-1` is zero without error; duplicates are independent.

Sticky device bits: `BAD_ID=1`, `BAD_LOGIT=2`, `BAD_COUNT=4`. Caller clears status
once at chain/replay entry; helpers only OR bits. Invalid live count clears all
outputs; active nonfinite logits clear that router row. This fail-closed invalid
input behavior is not claimed as Flare numerical equivalence. E<=1024/topk<=32;
512/top10 is included. IQ4_NL lookup does **not** enable IQ4_NL matmul.

The helpers require no workspace. They never download route values, allocate,
JIT, tune, or synchronize in run. `quactlize_serving_prepare_v1(device,stream)`
must run outside capture to warm all helper variants. Use an explicit nonzero
stream. Repeat preparation after a device/context reset. Matmul uses the
existing query → prepare → run sequence; prepare overwrites its valid supplied
buffers and synchronizes. PPU small-M SIMT/grouped TC selection is unchanged.

All memory is caller-owned. Keep buffers, prepared matmul owners and DSO alive
through completion and graph destruction. A graph captures addresses/capacity,
not the current contents of logits/IDs/counts. Different concurrent chains need
disjoint output, status and scratch; serial layers can share
`max(query.workspace_bytes)`. Read-only weights can be shared. No global
writable device scratch is cached. Use a distinct CUDA matmul handle per
concurrent chain/stream (its cuBLAS state is mutable).

NVIDIA is a **separate compatibility binary**, not a PPU fallback. It shares
the decoder/numerical ABI and bounds expanded weights to one expert × at most
512 output columns, independent of M; input/output scratch is linear in
`R=M*topk`, plus 32MiB caller-owned cuBLAS workspace. It computes each expert's
bounded GEMM and scatters matching slots. This favors bounded memory and
correctness, not a claim of PPU-like throughput; it does not transplant CUDA
tile choices into the PPU route.

## Build and tests (no dependency installation)

For Box acceptance from an existing Quactlize checkout, use the uploaded runner:

```bash
git fetch origin feat/ppu-flash-next
git show FETCH_HEAD:tools/run_flashnext_box.py | python3 - --ref FETCH_HEAD --device 0
```

It creates detached worktrees without changing the caller's branch/worktree,
reuses the local pinned actlize object when available (otherwise SSH checkout),
builds via `build.sh`, runs the native compatibility and serving tests, then
M1/M4/M16 graphs at E512/top10/K2560/H640. Choose an idle PPU. Existing platform
Torch/SDK must be configured; nothing is pip-installed. Failures stop later
phases and still produce `*.review.tgz` with logs, summaries and build identity,
not source, binaries or caches. Source/package stay on Box for diagnosis.

Use the already installed platform Torch. These commands do not run pip,
modify Torch/FlashInfer/Triton, fetch submodules, or build `_C`.

```bash
# PPU: existing full public package, using the pinned actlize checkout.
bash build.sh --library --sdk "$PPU_SDK" --output /tmp/qz-ppu-serving --jobs 8

# NVIDIA: stock CUTLASS checkout supplied explicitly, different output/DSO.
bash build.sh --cuda-serving --cuda /usr/local/cuda \
  --cutlass /path/to/nvidia/cutlass --arch 80 --output /tmp/qz-cuda-serving

# Host ABI/reference/custom_op fake/export checks; no GPU proof.
python -m pytest -q tests/test_serving.py

# Same serving tests with REAL platform GPU Torch. Missing device is a failure.
QUACTLIZE_SERVING_LIBRARY=/tmp/qz-ppu-serving/libncp_quactlize.so \
  bash run_tests.sh --suite serving
# NVIDIA: replace only the library path with:
# /tmp/qz-cuda-serving/libquactlize_cuda_serving.so

# Existing raw/canonical PPU projection regression remains independently run:
bash run_tests.sh --suite accuracy --package /tmp/qz-ppu-serving
```

`PYTHONPATH=/path/to/quactlize` suffices for the thin Python binding; no host
extension import is performed. `quactlize.serving.Serving` owns the loaded
helper library and `Matmul` owns a prepared v1 handle. All custom ops declare
output/scratch/status mutations and register fake implementations. They load
nothing lazily. Export/capture must happen while these prepared owners are
registered. Pure C callers can use the same header without Python/Torch objects.

## Minimal graph and measurements

```bash
python examples/serving_graph.py \
  --library /tmp/qz-ppu-serving/libncp_quactlize.so \
  --m 4 --experts 512 --topk 10 --trace /tmp/qz-serving-trace.json
```

The example prepares then captures embedding + ready PLE staging + router →
indexed gate/up → rounded SwiGLU → indexed down → weighted merge. It changes
device inputs/counts at fixed addresses across replay, checks inactive outputs,
and reports component/replay times and workspace. Optional platform profiler
output reports kernel/device activity counts/names and allows inspection of runtime
API calls. Timing events/profiler synchronization are measurement scaffolding,
not graph nodes. Component event times can include host submission gaps; they
are not sums of isolated kernel durations. Do not infer absence of bubbles
solely from graph capture.

The default uses compact synthetic quantized weights; increase K/H explicitly
for actual projection geometry. No model/PLE table is downloaded. Tests include
512/top10, M1/2/4/8/16/32 router, native16 rounding witnesses, strided inputs,
64-bit host bounds, poisoned inactive inputs, duplicate/invalid IDs, canaries,
changing graph routes, serial scratch reuse and disjoint concurrent streams.
Weights tolerance is `rtol=3e-6, atol=2e-7`; IDs/selected values/status are exact.
SwiGLU allows one target ULP for exp implementation rounding at a midpoint and
also requires an input set distinguishing the two rounding modes.

Framework migration: replace only router/SwiGLU/lookup/merge bindings, select
`round_silu=True`, keep route weights F32, and pass device live counts. Keep
ordinary BF16 projections in platform cuBLAS. PLE hashing, host-table fetch and
staging readiness synchronization stay in the framework. Quantized projections
keep the existing native16 matmul ABI and -1 inactive slot convention.
