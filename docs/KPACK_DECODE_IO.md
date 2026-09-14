# Decode input/output without standalone casts

Local implementation and PPU compilation are complete. Device numerical,
graph-replay and model-speed admission remain pending. This is an additive
endpoint change, not a new sweep or a new heuristic.

## Scope

| Path | Input conversion | Output conversion | Remaining kernels |
|---|---|---|---|
| Dense TC, M1..8 | F32 or BF16 storage to FP16 inside A load | Direct F32/BF16 epilogue, or ordered Split-K reducer | GEMM; reducer only for S>1 |
| Selected SIMT GEMV | Existing F32 caller input, register conversion | Existing F32 store | Existing selected recipe, unchanged |
| Indexed/chain MoE, tokens1..8 | F32 to FP16 inside fused prepare | Inside indexed finish/reduction; SwiGLU writes down's FP16 input directly | Routing/permutation is still necessary, but no separate dtype casts |

The original FP16 API is retained. BF16 here names storage, **not BF16 TC
compute**: the mixed-input core continues to use FP16 operands and FP32
accumulation. Values must be representable at that FP16 boundary. F32 final
stores are not rounded to FP16. Canonical weights, B readers, MMA and pipeline
cadence do not change. Prefill M>8 is outside this endpoint change.

The extension from 32 to 64 routed rows covers top8 tokens5..8. The old
single-token preparation remains unchanged. Shared row coordinates span both
warps; ranks and tile prefixes cannot be computed independently per warp.

## API and selection

- [Typed dense C API](../quactlize/decode/api.h): matching F32/F32 or BF16/BF16,
  contiguous `[M,K]` and `[M,N]`, E=1, M1..8. Pointers are 16-byte aligned;
  caller owns sufficient disjoint input/output/workspace storage.
- [Dispatcher](../quactlize/dispatch/api.h): `query_dense_io_v1` uses the
  existing ordinary or measured Q4 decode selector. `prepare_dense_io_v1`
  requires the matching typed ticket. It never silently consumes an old
  FP16 ticket or reinterprets BF16 as FP16.
- Typed and FP16 modules have different content-addressed identities. Only
  the selected endpoint is loaded/JIT-compiled; an unused FP16 module is not
  compiled first. Prepare/JIT stays outside capture; run does no allocation,
  compilation, host copy or host synchronization.
- The private llama adapter binds the optional pair together. Admitted dense
  decode calls point directly at the caller's tensors with zero A/output
  adapter allocation. Older packages and policy misses retain existing
  fallback behavior; this is not an all-shape no-fallback guarantee.

## Box gate

Wait for the active eight-card cost campaign to finish, or use a separate
checkout and an idle device. Do not update its measuring checkout in place.

```bash
cd /sim/eec/shared/junfu.qx/quactlize &&
git pull --ff-only &&
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
CUDA_VISIBLE_DEVICES=0 \
bash tools/run_kpack_decode_io_ppu_box.sh
```

The script pulls only the focused payloads with Git LFS, preserves the SDK
environment and leaves the invoking Docker shell open on failure. Requires
the same PPU SDK/runtime used by earlier successful box runs, Python, NumPy
and `gguf`. No model is needed. Compilers/JIT are not invoked by this gate.

The package contains 27 typed parents, five old-ABI/grouped controls and two
SIMT-stage test binaries. There are 104 policy requests with both endpoint
types, seven changing-input eager/graph checks per admitted dense request,
output/workspace guards, an independent GGUF oracle and a zero-input negative.
The captured dense graph must have one node for S1 or two for Split-K.
Separate tests cover 32 indexed-stage cases and 192 chain-stage cases; four
real selected gate/up/down chains exercise 64 routed rows, with and without
merged weights and router fusion. Policy misses are reported, not numerical
passes. Warm diagnostic timings exclude the first launch; they do not
establish model performance or cold-weight parity.

Return the printed `/workspace/kpack-decode-io.*.results.tgz`. Failures in one
phase do not discard successful phases. After device admission, check the
actual llama decode trace and steady-state model time before release.

Local build entry:

```bash
python3 tools/build_kpack_decode_io.py --sdk /path/to/PPU_SDK \
  --output /data/kpack-decode-io-v1 --jobs 8
```

Module cache reuse requires identical compiler/kernel identities. The two
test executables use native PPU runtime names with the production SIMT bodies;
they neither copy/emulate the GEMM body nor admit TC numerical correctness.
