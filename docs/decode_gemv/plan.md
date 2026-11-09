# Decode GEMV / small-M optimization plan (QUactlize on PPU)

Task workspace: `/workspace/quactlize-decode-gemv-20261108`
Source worktree: `source/` on branch `perf/ppu-decode-gemv-20261108`
(from `origin/develop` 5b5b9a455ce9800cf85d3a2f4d707fa5207b4072, no upstream tracking).

## Identity and environment

- QUactlize baseline commit: `5b5b9a45` (= `origin/develop`, also the deployed revision used by
  `/workspace/strata-ppu610/src/QUactlize`). `origin/feat/ppu-flash-next` (82388b15) is develop + 13
  commits (Q5_1, Q2_0, native16 MTP, compat/serving ABI); second-stage set only.
- actlize submodule `021c6930`; cutlass submodule not needed by the library build.
- SDK `2.2.1-184ee6` (cuda-12.9 variant) at
  `/workspace/actlize-la-gdn-fla20-20261106/sdk/2.2.1-184ee6-cuda129/PPU_SDK` (`env.sh`).
  The system SDK 2.0.0 rejects `-arch=ppu_10`.
- Devices: 2x PPU-ZW610 (PPU1.0 / ppu_10), 64 CU, 48 GiB, runtime `l2CacheSize` 64 MiB
  (`runs/environment-dev*/environment.json`). Both cards are shared with other agents'
  jobs; every formal capture waits for an idle card and is discarded if a foreign process
  appears during it (`tools/ab_rounds.py`).
- Caller rules: DrXuQian/llama.cpp `pr/kpack-ffn` 30cdcb24 (gate/up merge, TP split function,
  dense F16 context, MoE BF16).

## Workload manifest

`inventory/workload_manifest.json` (generator `tools/make_manifest.py`) from GGUF headers
fetched by range requests (`tools/fetch_gguf_headers.py`, no weights): 5 models, 55 unique
Quactlize shapes, plus every non-Quactlize matrix (IQ*, MXFP4 experts, N%256!=0, DeepSeek
grouped `attn_output_a`) listed separately (in this tree: `docs/decode_gemv/workload_manifest.json`).

## Measurement protocol

- Public ABI only (`quactlize_ppu_dense_*`, `quactlize_ppu_moe_matmul_*`), production
  `prepare_v1` selection; F16 dense context, BF16 MoE (caller contract).
- Cold weights: dense ring of distinct weight copies >= 768 MiB (12x reported L2); MoE walks all
  256 experts with distinct IDs (spread routing). Graph replay; JIT/module load, first graph
  upload and first replay excluded.
- Asys `-t hggc` GPU kernel activity; per-call kernel positions (main / prepare / reducer /
  helpers) are attributed between unique memset markers (`tools/parse_asys.py`).
- A/B: baseline and candidate packages in separate processes, alternating order per round,
  same physical card; >= 3 rounds; mean, median and distribution retained.
- `T_model(M) = sum(calls x mean complete-call GPU time)`; model reduction averaged with equal
  weight; target: M=1 average reduction >= 20% (ratio <= 0.80). M=2..8 reported separately.

## Iteration plan

1. Baseline: all 55 shapes x M=1..8 through production selection; record selected configs.
2. Pick representative shapes (largest time share per model / distinct bottleneck) and inspect
   the selected kernels (ISA, resources; ACU on one warmed kernel when needed).
3. Candidates (one controlled change each), validated with the independent GGUF oracle before
   timing; promote only measured winners through the existing selection tables.
4. Integrate winners into the default selection; regression-check affected prefill paths.
