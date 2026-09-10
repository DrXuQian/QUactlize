# Single-token fused router/preparation candidate

The small dispatcher binds the new selected-parent JIT source contract. The
execution DSO is byte-identical to v2. No precompiled GEMM closure is shipped.
GPU packing, offline formats and on-disk weight caches are unchanged.

Single-token, top8, 256-expert preparation uses one CTA. One warp selects
experts while the other warps gather the common activation, followed by
typed gate/up/down descriptors and directories. Other shapes retain the
general preparation. Both merged and separate gate/up remain supported.

RTX5090/5070 SIMT checks do not admit the PPU device or model path. The real
PPU grouped module compiles with the production packed CuTe types. The box
gate must validate selected GEMMs, router fusion and complete model timing.
First-use JIT and the first request remain excluded from steady timing.

Update private llama.cpp `feat/kpack-gpu-cache` for the router/input-view
matcher repair and explicit Asys target environment. Then run from Quactlize:

```bash
CUDA_VISIBLE_DEVICES=0 JOBS=192 RUN_Q8_SIMT=0 RUN_MODEL_TRACE=1 \
  TRACE_PROMPT=2048 TRACE_GENERATE=16 bash tools/run_kpack_fusion_box.sh
```

This bounded run does not re-sweep Q8. Use an existing matching Q8 recipe via
the full recipe gate when needed; an absent recipe retains W8A16 TC. This
candidate does not change Q8 selection or claim to close the decode gap.
