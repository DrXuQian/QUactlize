# Parallel MoE preparation and Q8 SIMT candidate

`dispatch/` contains the small host selector and PPU execution library.
There are no precompiled GEMM modules. Use Git LFS to fetch both libraries.
This package is compile-verified, not PPU/device-admitted.

The GPU converter remains `../kpack-fusion-v1/libquactlize_ppu_pack.so`;
offline bytes and persisted weight caches are unchanged. Runtime source
changes require fresh **selected-parent** JIT keys, not a full sweep rebuild.

Use Quactlize develop with private llama.cpp `feat/kpack-gpu-cache` at
`83e8efdfe` or later. From the Quactlize checkout, the box entry is:

```bash
CUDA_VISIBLE_DEVICES=0 JOBS=192 RUN_MODEL_TRACE=1 \
  TRACE_PROMPT=2048 TRACE_GENERATE=16 bash tools/run_kpack_fusion_box.sh
```

The default gate includes Q8 SIMT vs selected W8A16 TC on twelve contexts,
eight configurations, three alternating-order rounds. Only measured SIMT
winners are exported and used in the subsequent model run. Empty policies
are valid and retain TC. Reference/model timings exclude the first complete
pass; the optional Asys trace captures a warmed request.

See `docs/LLAMA_CPP_KPACK_HANDOFF.md` for scope, remaining gaps, and the
separate RTX5090 preparation measurements. NVIDIA timings are not PPU policy.
