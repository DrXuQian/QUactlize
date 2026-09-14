# Q4 model integration and performance closure

Current scope: the private `dev/quactlize-v0.3.0` branch, canonical K-pack,
request batch 1, prompt 2048. Q4_K_M model files can contain Q5_K, Q6_K and
Q8_0 tensors; their actual paths must be covered, not just qtype 12.

## Current checklist, 2026-09-14

| Step | State | Completion evidence | Location |
| --- | --- | --- | --- |
| Release and patch base | Complete | `dev/v0.3.0` at `e73e2136b`, Quactlize child at `03142a8b7` | Local/private fork |
| Measured component selection | Complete within documented coverage | Small-M SIMT/TC policy and measured prefill FQ/SF/full-BF16 policy; no double-counted reducer | Existing PPU results |
| Library functional gates | Complete for declared denominators | Decode 208 dense endpoints and four real TC chains; prefill nine composed contexts | Reviewed PPU results |
| Decode fusion wiring | Open, current code task | Retain measured SIMT/TC choices through the shared MoE prepare/activation/finish path; direct F32 dense endpoints remain intact | Local implementation, then PPU gate |
| Model deployment entry | Open | Pin the new dispatcher/execution/prefill package and v0.3.0 caller, matching headers/JIT source and paired packer | Local build/package tests |
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

## Exact decode wiring gap

The caller already implements direct F32 dense TC endpoints, measured Q4
SIMT selection and direct F32 SIMT calls. The missing part is mixed SIMT/TC
MoE composition. At private llama commit `03142a8b7`, `prepare_moe()` rejects
any projection for which `Plan::direct` is true. The v1 chain interface takes
only indexed-bound TC handles. A direct SIMT call is not such a handle and
returns output in caller order, whereas the TC chain uses compact expert
order and FP16 completed values (or FP32 split partials).

Removing that rejection alone is incorrect. The bridge must preserve each
projection's input/output type, ordering, pointer lifetime and measured
recipe, including a SIMT projection adjacent to a TC projection. It must not
force TC merely to satisfy the old fusion interface. Existing all-TC chain
results do not admit this missing mixed path.

The old `run_kpack_fusion_box.sh` still defaults to the early
`kpack-fusion-v3/dispatch` package and the historical llama build directory.
Do not use that default as proof of this new integration. The final model
runner must bind the new package and branch explicitly, keep Asys separate
from reported timing, and exclude the first full PP/TG warmup pass.

Model inputs are the existing focused int4 plan:
`tools/kpack_batched_int4_2048.json` (Qwen3.5-35B-A3B-Q4_K_M MoE and
Qwen3-32B-Q4_K_M dense; NPL=1, PP=2048, TG=128). User files remain unchanged.
