# Draft: decode GEMV / small-M (M=1..8) on ZW610

## Baseline evidence (ZW610 dev0, SDK 2.2.1, develop 5b5b9a45, production `prepare_v1`)

- Streaming-read probe (`tools/bw_probe.cu`): 1.10-1.17 TB/s achievable read bandwidth.
  ACU reports DRAM peak ~1.38 TB/s (896 GB/s = 64.8%), CE clock 1.6 GHz, DRAM 1.8 GHz.
- M=1 production kernels (one Asys round, `runs/baseline-m1`): 170-930 GB/s effective.
  Large K-quant SIMT `common_reader`: 680-870 GB/s; TC `fqk_tc`/`closure`: 550-830 GB/s;
  Q8_0 small shapes 170-580 GB/s and pay a separate Split-K reducer (~1.0-2.3 us).
- ACU (forced-cold, one warmed launch each, `profile/baseline-acu`):
  - `common_reader<0,13,8,4,4>` Q5_K 5120x17408: IPC 3.75 of 4 (issue-bound), DRAM 63%,
    0.29 warp-instr/code (9.4 thread-instr per weight code).
  - `common_reader<0,12,4,2,8>` Q4_K 5120x25600: IPC 3.12, DRAM 65%, 6.25 thread-instr/code.
  - TC Q4_K 51200x5120 (`MainloopQuactlizeMixedInput`, AIU->TSM, m8n16k16): IPC 2.12,
    57% no-eligible cycles, occupancy 36%, DRAM 54% -> latency/occupancy bound.
- Native ISA of the K-quant reader: per weight code ~1 FMA + 1.2 `cnvt.f32.f16` + 1.1 shift +
  0.6 and + 0.6 or + 0.5 `add.f16x2`: conversion/extraction dominates, not memory.
- Model upper bound if every call ran at 1.1 TB/s: 32-47% reduction; with 2 us fixed cost per
  call 20-30%. The 20% target therefore needs near-roofline kernels on all large shapes and a
  lower fixed cost on small ones.

## Ranked directions

1. Register-direct MMA reader (candidate C1). Weights go global->registers, codes are expanded
   exactly (lop3/prmt/hfma2) directly into the m8n16k16 A fragment, activations are staged once
   per CTA in TSM in B-fragment order, tensor cores produce each group's exact-code dot product in
   FP32, and the group affine stays FP32 on the accumulator (SIMT arithmetic contract: rounded
   activation, exact code, FP32 group scale/min). Expected 2.5-3x fewer issued instructions per
   code than `common_reader`; M=2..8 reuse the same weight registers for up to 8 tokens.
   Device probe `tools/mma_probe.cu` confirmed the A/B/C fragment maps (f16 and bf16).
   Risks: register pressure (units + double-buffered weights), Q5/Q6 high-plane maps,
   per-lane metadata decode cost.
2. Lower fixed cost for small Q8_0 shapes: single kernel (no separate reducer), more warps/CTAs.
3. Rotate token-0 B rows across tiles so the M=1 affine/metadata work spreads over all lanes.

Counterexamples on record: the historical "1 row per block, plain LDG" SIMT rule and the TC
AIU path both stop below 70% of DRAM on this card; register count reductions alone did not
predict wins in earlier rounds.

## C1 register-direct MMA reader: result (v3, M=1 screen, single round)

- Correctness: 46/46 cases vs independent gguf-py reference (Q4_K/Q5_K/Q6_K/Q8_0 dense F16,
  Q4_K/Q5_K MoE indexed BF16, M=1..8, T=2/4/8, NB=2/4/8, WK>1, split 1..4), max rel err ~1e-8.
- Screen `runs/proto/screen-v3` (+ `screen-v3b/round0/v3.contaminated0`, foreign light pytest;
  two contaminated attempts agree within +-0.4% p5-p95): best per shape vs in-run production.
  Wins on large Q4/Q5 (51200x5120 0.817, 5120x8192 0.740, Q5 12288x5120 0.878), ties/losses on
  Q8 and small shapes (Q8 512x2048 1.37, Q4 1024x5120 1.13). Model estimate with best-of-screen
  winners only: qwen3-32b 13.4%, qwen38-27b 8.0%, qwen35-35b 2.1%, qwen35-122b 1.8%,
  deepseek 2.5%; equal-weight 5.6%. C1 alone cannot reach 20%.

## Step 1: why C1 tops out (access pattern, not in-flight bytes)

`tools/pattern_probe.cu` (no consumer), Asys kernel durations, cold ring, ZW610 dev1
(`profile/pattern/sweep1`), ACU forced-cold (`profile/pattern/acu`):

| pattern on a 125 MB K-pack plane (n=51200) | GB/s | ACU DRAM % |
|---|---|---|
| flat (contiguous chunk per CTA) | 1160-1181 | 85.1 |
| rows (one warp instr = 512 B of one plane row) | 1152 | 83.4 |
| frag (MMA-fragment order: 4 rows x 32-128 B per instr), best of T/WN/WK/U | 949-989 | 68.9-71.8 |

- Prefetch depth U=1..8 and occupancy 19-92% do not move the frag pattern; the row-segment
  layout of the warp instruction does. The prototype (870 GB/s, ACU DRAM 63%, IPC 2.02,
  achieved occupancy 38%, 400 CTAs = 6.25/CU) is at ~91% of its own pattern ceiling.
- Production TC (AIU->TSM) 728 GB/s DRAM read is consumer/barrier-bound (IPC 2.12, sync stalls,
  occ. 36%, DRAM active only 89%). Full step-1 evidence: `docs/step1-access-pattern.md`.
- Size floor (flat, best grid): 16 KB 0.92 us, 1 MB 1.94, 2 MB 2.90, 4 MB 4.72, 8 MB 8.38,
  16 MB 15.1, 32 MB 29.5 us. If every manifest call ran at this streaming floor:
  qwen3-32b 36.7%, qwen38-27b 35.1%, qwen35-35b 31.3%, qwen35-122b 31.7%, deepseek 38.0%
  (equal-weight bound 34.6%). 20% therefore needs ~60% of the streaming headroom everywhere,
  including single-kernel small calls.
- Direction for step 2: row-contiguous global->shared staging (cp.async/AIU, >=256-512 B per
  row per warp instruction) feeding the C1 consumer (exact codes, group rotation, FP32 window
  affine) as an actlize-extension collective.
