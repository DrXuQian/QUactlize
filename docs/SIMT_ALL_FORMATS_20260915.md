# All-format SIMT: implementation and numerical receipt

The new reader is an explicit-config implementation, not a new policy winner.
It covers canonical Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_0 with vector N loads,
word reuse, half2 integer-code reconstruction and F32 group-affine dots.
The format bytes, existing Q4 selector and TC routes are unchanged.

## Numerical result

RTX5070/WSL,48 SMs,48MiB L2, CUDA12.8. The same compiled DSO is used in
all runs below:
`b7c42d4fbbf71f94c756d4695a6a0b9d1c0cbcbddb1e8962d566ab6ab7703035`.

Archive `simt-5070-numeric-results-v2.tgz`:
`ce58a1604a21bb3a325c7168c0a1d6ac9263cdc2b4ccc5decf0851cd5dc4123a`.
All result denominators, unique configuration/context keys and source hashes
were checked. The only modified files relative to the compiled-source
receipt are the development Python upload/runner files; kernel sources match.

| Format | Checks passed | Maximum condition-normalized GGUF dot error |
|---|---:|---:|
| Q8_0 | 6240/6240 | 2.660e-8 |
| Q2_K | 12480/12480 | 3.629e-8 |
| Q3_K | 12480/12480 | 4.322e-8 |
| Q4_K | 12480/12480 | 3.615e-8 |
| Q5_K | 12480/12480 | 2.876e-8 |
| Q6_K | 12480/12480 | 4.352e-8 |

Total68640 checks: every compiled runtime recipe, dense M1--8, indexed
token1--8 with shared/per-slot A, compact rows with empty experts, F16/F32
storage, S1/2/4/8, output/workspace guards. There are1152 changing-input
graph checks and384 zero-A negatives. F16-exact activation fixtures give
bit-identical F16/F32 endpoint results. The0.005 error bound was not relaxed.
PPU compilation has660 producer bodies plus the real reducers; CUDA ptxas
reports no register spills in this inventory. These facts do not prove PPU
device correctness, performance, or full-range F32 input arithmetic.

## Failed first run and actual test-harness fix

The original harness used default-stream `cudaMemcpy` from pageable NumPy
storage, then consumed inputs in an explicitly nonblocking stream. A sync
of the consumer stream alone does not establish copy completion. CUDA's
[copy synchronization contract](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)
allows pageable H2D to return before the final DMA finishes; the
[stream contract](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html)
does not implicitly synchronize a nonblocking stream with the legacy stream.

The fix queues H2D on the actual consumer stream and waits in untimed
fixture preparation, keeping the source array alive. No kernel, arithmetic,
threshold, shape, candidate or production launch path changes.

The first run failed in Q8/Q3/Q6 and passed Q2/Q4/Q5. Repeating the full
inventory with ordered uploads passes all six. A separate A/B changes only
`Runtime.copy`: reverting to the old copy reproduces Q6 error0.271976 at
token5/per-slot-A/F16, recipe v0/C4/W2/P2/S1; its ordered-copy counterpart
passes. The old-copy Q8/Q3 reruns happen to pass, consistent with missing
ordering being intermittent rather than a deterministic decode formula.
Do not erase the failed receipts or describe this as a production math fix.

The previous Qwen3-32B activation-range failure remains a separate open
issue. This reader still rounds F32 A to F16 in registers; this receipt
does not admit inputs above the F16 finite range, and clamping is not a fix.

## Performance and admission

The 5070 performance archive `simt-5070-perf-results-v1.tgz`, SHA-256
`40425526b42151f9251b43260f2d1565339d0894de940cda40d9956e9ff90928`,
contains 30 passing contexts (six formats, five dense/indexed shapes).
Each screens the declared inventory, then confirms two candidates per
implementation in four alternating rounds of 11 samples. Recomputed
medians, finite samples, independent GGUF checks and complete rotating-ring
traversals agree. Ring bytes exceed 2.25 times 48 MiB L2.

Selected complete-call medians for dense M1/N4096/K4096 on this NVIDIA GPU:

| Format | Old generic SIMT, us | New reuse SIMT, us |
|---|---:|---:|
| Q2_K | 62.654 | 18.326 |
| Q3_K | 92.286 | 25.583 |
| Q4_K | 52.820 | 19.163 |
| Q5_K | 69.163 | 24.149 |
| Q6_K | 90.475 | 29.137 |
| Q8_0 | 61.723 | 29.928 |

The old generic control is NOT the optimized Q4 incumbent. The latter takes
19.150 us on this large M1 shape, 2.603 vs new 3.645 us on M1/N512/K2048,
and 30.476 vs new 49.388 us on indexed T8/N2048/K512. Keep it in the candidate
set and retain its existing selection; no global reader replacement follows.
New and old use different legal dequantization rounding (documented above).

NCU/ACU and source address models must separate A, metadata, low and high
planes. Q2/Q3/Q6 scale groups are smaller than their packed-word K span;
requests can still repeat across workers even when N requests are contiguous.

The final PPU sweep must retain current typed TC choices with actual
Split-K reducers and required endpoints. A CUDA SIMT-only gain is not a
PPU route admission, native llama.cpp parity, or whole-model speed result.

## Callable library and composition

`tools/build_kpack_execution.py` now compiles the same inventory from
`quactlize/execution/simt_codegen.py`; the development harness imports that
generator. The existing scalar/pair and optimized Q4 entries remain exported.
Local PPU execution build: 72.754 seconds, approximately 16 MiB DSO,
`/root/autodl-tmp/simt-execution-20260915-v1/libquactlize_ppu_execution.so`.
A JIT-only dispatcher also compiles at
`/root/autodl-tmp/simt-dispatch-20260915-v1`. These are local builds, not the
historical model artifact being replayed for the activation diagnosis.

The new `qks_moe_endpoint_v3` accepts a `qkg_simt_config_v1` without changing
v1/v2 layouts or their selection. It composes mixed TC/SIMT gate/up/down,
including merged gate/up, using the existing GPU preparation/SwiGLU/finish.
Creation copies calls/configs; run performs no JIT, allocation or ID readback.
The caller supplies separate Split-K workspace and projection scratch.
Composition rejects workspace overlap with live intermediates, weights,
router inputs or weighted-finish storage. Old libraries decline the new
reader while remaining usable for the old interfaces.

Validation: 209 local tests pass, including 1,920 host composition cases
over six formats, token 1--8, all TC/SIMT masks and S1/2/4/8. Host mocks prove
dispatch and pointer contracts, not numerical execution of the mixed chain.
The earlier 68,640 CUDA checks cover the standalone dense/indexed kernels.
PPU numerical/performance admission and a full mixed-chain GPU gate remain
required; no policy is promoted and no model-range failure is closed here.

Native inspection of Q6 F32/v2/C4/W4/P8 finds vector `vmem.ld.b32x4`,
half2 code construction, F32 FMA, 102 vector registers, and stack size zero.
Q4 F32/v0/C8/W4/P4 uses 62 vector registers and stack size zero. These are
static properties of these two exact specializations, not dynamic bandwidth
measurements or a claim of no spills in every PPU candidate.
