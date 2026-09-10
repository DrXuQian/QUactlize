# MoE integration work

The target is complete, warmed model latency, not just producer GEMM time.
Changes stay on develop/private llama.cpp until numerical and device gates
close. No graph-capture-time compilation, D2H routing, or CPU scale cache.

| Work | State | Required closure |
| --- | --- | --- |
| Q8_0 W8A16, resident original FP16 d, dense/grouped intake | Implemented; four PPU parents compiled locally | Q8 GPU pack, native graph, model routing/accuracy and timing on PPU |
| Gate/up paired GPU pack | Implemented, 18 host exact cases pass, PPU producer compiled | PPU device bytes and consumer numerical gate |
| Merged weight intake and cache | Implemented; nonadjacent/reversed source spans, cold/warm and negative host tests pass | PPU model cold/warm gate |
| Fused route/gather/metadata/directory | Implemented per call and shared across the small gate/up/down chain; 5090 checks pass | PPU numerical/performance gate; larger prefill retains its original path |
| Ordered Split-K reduce + scatter | Implemented for that indexed path; 5090 checks pass | PPU guard/replay/numerical/performance gate |
| Gate/up completion + SwiGLU, expert-ordered down input | Implemented and wired through the closed llama graph; separate or already-merged weights | PPU producer/chain and full model checks |
| Top-k/preparation integration | Implemented for matching 256-expert small-decode graphs; softmax, sigmoid+bias+norm, delayed-softmax 5090 checks pass | Additional NaN/Inf cases, matched timing, PPU graph and model checks |
| 5090 fusion checks | Initial 8 per-call cells plus 7 complete SIMT-stage contexts pass, 7 changing-input replays each | These use completed-projection fixtures, not a PPU GEMM emulator; matched timing remains |
| PPU model benchmark and trace | Pending | Uploaded batched-bench axes, first full pass excluded, honest TP admission |

## Reuse llama.cpp's merged graph

In the local llama.cpp checkout, `convert_hf_to_gguf.py --fuse-gate-up-exps`
enables `conversion/base.py` to concatenate **unquantized** gate/up tensors
along N: `[E,N,K] + [E,N,K] -> [E,2N,K]`. Gate precedes up for each expert.
This is a conversion flag, not an inference-time flag.

`create_tensor_gate_up_exps()` already recognizes
`blk.X.ffn_gate_up_exps.weight`. `llama-graph.cpp` consumes it with one
`MUL_MAT_ID`, then gate/up views and `ggml_swiglu_split`.
This is the graph contract to reuse; do not add another projection ordering.

For already-quantized GGUF inputs, the paired GPU producer in
`quactlize/packing` virtually concatenates **raw blocks within each expert**
and writes the final canonical K-pack with N doubled. It neither requantizes
nor first materializes two separate K-pack tensors. Two K-pack low arrays
cannot simply be appended: their physical leading-N strides would be wrong.
Both sources must share qtype, N, K, E and the canonical descriptor. Unequal
qtypes retain separate GEMMs, though their routing can still be shared.

## Boundaries

- Common preparation, metadata and completion live in Quactlize. llama.cpp
  graph/stride-specific contracts live under `quactlize/integrations/llama`.
- Gate/up can share routing and gathered input. Down shares routing, not the
  pre-SwiGLU activation. It consumes the newly produced expert-ordered input.
- A directory may be shared for the same routing and TileM. Output pointers
  and strides depend on N/K/S and must be generated for each projection.
- Merged N is a different dispatch request. Do not call a reused split/grid
  the measured optimum until the merged shape is measured.
- Small decode can build routing, gather and typed metadata in one launch
  using an independently derived mapping in every CTA. Prefill needs a real
  global prefix boundary; CTA barriers cannot replace it.
- Preserve `fp32 ordered sum -> fp16 -> fp32` before scatter/SwiGLU, matching
  the unfused numerical contract. Do not replace it with an all-fp32 path.
- Top-k has model-specific normalization, bias, tie-breaking and clamp rules.
  Only matching graphs are fused; unsupported graphs retain the existing path.

## Local evidence (2026-09-10)

`tests/test_kpack_gate_up_pack.py` covers Q8_0 and Q2/3/4/5/6_K in 18 cases with E=1/2/3 and
different N/K extents, independent raw concatenation, all plane bytes and
guard regions. The whole-low-plane append negative is always red. A single
expert with one metadata unit is deliberately an identity control for the
metadata append negative, not a failed producer.

`tests/kpack_indexed_cuda.cu` launches the production preparation/completion
SIMT bodies on a 5090 using the existing development-only CUDA API bridge.
Eight TM8/TM16, S1/2/4/8 contexts pass seven changed-router/input graph
replays, including slot-specific down input, 32 routed rows, and a second
M-tile. Independent checks cover gathered FP16 values, directory records,
typed output descriptors, ordered reduction, final scatter and padding.
The all-FP32 completion negative differs, proving the FP16 boundary matters.

The graph-batched SIMT-only measurement places single-token preparation near
3.8 us and completion near 1 us on this 5090. This is not PPU or full-model
timing, nor a matched unfused-baseline speedup. PPU admission is pending.
The production GEMM, selected split/grid and accumulator/partial stores are
unchanged. Existing modules without the additive indexed binding and larger
requests retain the original path explicitly.

## Cross-projection execution candidate

`quactlize_kpack_dispatch_moe_create_v1(gate,up,down,...)` binds the existing
selected handles without selecting another parent. Passing NULL for `up`
uses an already-merged `[E,2N,K]` gate/up weight. The chain checks dimensions,
ID/source identity, the real CuTe shape/stride member offsets and disjoint
scratch. Standalone per-stream scratch is intentionally reused and is NOT
safe for retaining both projections' partials. The llama chain gets separate
retained allocations outside capture; it never JITs, allocates, copies routes
to the host or synchronizes during execution.

The shared preparation computes ranks once per CTA and publishes each
participant's own metadata/directory for its TileM/N/K/S. Gate/up share the
input load and conversion; distinct directories are still needed if TileM
differs. Ordered projection completion rounds through FP16 before SwiGLU.
SwiGLU writes down's expert-ordered FP16 input directly. Down reuses the
current route and has no second gather or metadata launch. Its completion
reduces and scatters directly to llama's F32 result.

The llama matcher reuses the existing four-node separate-weight and five-node
merged-weight graph. It checks external consumers, view order, input IDs,
activation, shape/strides and allocator overlap. Bias, OAI/clamped activation,
concurrent-stream or larger-row contexts retain the per-node path. A complete
top-k segment can additionally use `moe_run_router_v1`: the library kernel
derives IDs independently inside every CTA, publishes weights/IDs once, and
shares preparation. The single-CTA top-k allocator-overlap exception is NOT
borrowed for this multi-CTA fusion.

`tests/kpack_moe_chain_cuda.cu` passes seven 5090 contexts with zero descriptor,
gather, activation-half or scatter mismatches. It checks seven graph replays,
different splits and TileM per projection, empty experts, a second M tile,
padding/guards and invalid IDs. Router weights are compared with an independent
double-precision stable-sort oracle; ties use the lower expert ID. Projection
fixtures do not admit the PPU GEMMs. The GPU was shared with another workload,
so this run makes no matched latency claim.

Combined local Python/host tests: **139 passed**. The updated Q4 FQ and Q8 SF
grouped modules, dispatcher and the three affected llama CUDA translation
units compile with the real PPU SDK. Candidate small JIT-only package:
`/root/autodl-tmp/kpack-moe-dispatch-v1` (modules=0, device admission pending).
It has not replaced a published bundle.

## Paired load/cache closure and box delivery

`QUACTLIZE_KPACK_PAIR_WEIGHTS=1` now admits compatible source pairs through
the backend capability and uses the original merged graph. Raw source spans
are uploaded in bounded expert batches; there is no CPU concatenation or
intermediate individual K-pack allocation. The setter waits for source H2D
consumption only. Sources may be in different GGUF files; persistent caching
retains its existing single-file limitation explicitly.

Runtime cache v2 records the ordered gate/up source names, indices, offsets
and byte counts. No fake contiguous span or raw hash is used. Runtime v1 and
verified offline v3 reading remain compatible. Local tests cover reversed
file order with an unrelated tensor between sources, cache publication and
hit, exact plane bytes, missing/swapped/type/shape/offset/index negatives,
and Q8 original-scale cache. No separate-weight LoRA is admitted by this
opt-in merge; per-projection bias/scale tensors prevent pairing.

Delivery: `prebuilt/ppu0010/kpack-fusion-v1` (about 1.6 MiB including receipts),
`tools/run_kpack_fusion_box.sh`. No large sweep rebuild. The box runs six paired
GPU byte gates, seven Q8 W8A16 contexts, four real selected MoE chains with
changing-input graph replay, adapter/cache tests, and warmed model PP/TG.
The model benchmark uses separate reference/cold/hot processes, excluding
the first complete pass for each prompt length. `MODEL_NAMES=all` selects
the uploaded model inventory; unadmitted tensor parallel is NOT_TESTED.
Optional Asys is a separate warmed-request proof, not benchmark time.

Final local checks: 147 Python/host tests pass; llama loader/environment/cache
CTest entries pass (including 35 loader cases). The modified host loader and
three CUDA integration TUs compile. PPU numerical and model performance are
still pending: an implemented fusion is not a measured speedup or optimum.

## Doubled-N selector correction

The first box gate stopped before launch for merged Q4 N1024/K2048/E256,
M8/max_rows1 and M32/max_rows4. Both source-N512 requests selected successfully;
the previous selector required exact N in its historical grouped family.
This was missing production selection coverage, not a numerical mismatch.

Exact-family selection is unchanged. An uncovered grouped request may now
reuse a compatible N/2 family once, with whole output tiles and unchanged
qtype/K/E/M. It is labeled predicted; resource queries and runtime preparation
receive doubled N. No new offline mapping, GEMM body or activation is used.
Tests consume the same request generator as the device gate, cover Q2-Q6
FQ/SF at five token counts, reject recursive/cross-K/cross-E transfers, and
exercise actual C dispatch query/prepare with a doubled-N resource stub.
The 1,452 prior catalog requests retain exactly their previous selections.
Only the small host selector needs rebuilding; the JIT source hash is unchanged.
