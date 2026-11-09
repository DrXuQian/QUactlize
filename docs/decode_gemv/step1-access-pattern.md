# Step 1: where the decode MMA-reader prototype loses bandwidth (ZW610)

Question: is the register-direct prototype (`quactlize/execution/mma_reader.cuh`, C1) limited by
how many bytes it can keep in flight, or by the DRAM access pattern of its loads; and how do the
prototype, the production TC route and a pure streaming read compare on the same shape.

Setup: PPU-ZW610 (64 CU, CE 1.6 GHz, DRAM 1.8 GHz; clock not locked to 1700 MHz), SDK 2.2.1
(cuda-12.9 variant). Probes, prototype and production TC ACU all on dev1 while it had no foreign
compute process (the production TC kernel was re-captured on dev1:
`profile/acu-v3/base-d-q12-n51200-k5120-m1.acurep`, 52.63% DRAM / 203 us, identical to the
11-08 capture in `profile/baseline-acu`). ACU: `--set detailed --cache-control all` (forced
cold, one kernel per report). Asys: GPU kernel activity, cold ring larger than the 64 MiB
L2/LLC.

## 1. Same shape, three kernels (Q4_K 51200x5120, M=1, ACU)

Probes (`tools/pattern_probe.cu`) read a 125 MiB buffer shaped like the Q4_K low plane of this
shape (`[1280 rows][51200]` uint16) with no arithmetic consumer:
- flat: each CTA one contiguous chunk, 16 B per lane;
- rows512B: one warp instruction = 512 contiguous bytes of one plane row;
- fragT2/T4/T8: the prototype's MMA-fragment order (lane m,t loads T columns of rows 2t, 2t+1):
  one warp instruction = 4 rows x 16*T bytes.

| metric | flat | rows512B | fragT2 | fragT4 | fragT8 | prototype t4c4012s1 | production TC fqk_tc |
|---|---|---|---|---|---|---|---|
| duration us | 111 | 114 | 133 | 138 | 138 | 170 | 203 |
| DRAM read MiB | 125 | 125 | 125 | 125 | 125 | 141 | 141 |
| DRAM GB/s | 1177 | 1154 | 987 | 952 | 953 | 866 | 728 |
| DRAM read % of peak | 85.2 | 83.5 | 71.4 | 68.9 | 68.9 | 62.7 | 52.6 |
| DRAM active cycles % | 98.3 | 97.0 | 98.6 | 98.4 | 98.0 | 95.3 | 88.8 |
| LLC bytes / request | 64 | 64 | 64 | 64 | 64 | 64 | 64 |
| L2 / LLC hit % | 0.1 / 0.0 | 0.0 / 0.0 | 20 / 34 | 6.9 / 0.1 | 0.1 / 0.0 | 1.2 / 0.5 | 10.4 / 4.0 |
| achieved occupancy % | 12 | 38 | 92 | 77 | 19 | 38 | 36 |
| IPC (active) | 0.33 | 0.30 | 0.94 | 0.51 | 0.33 | 2.02 | 2.12 |
| issue active % | 8 | 7 | 21 | 11 | 8 | 45 | 43 |
| top stall / issue | vmem_ld 9.5 | vmem_ld 38.6 | vmem_ld 28.5 | pipe_busy 17.4, vmem_ld 17.3 | vmem_ld 16.9 | vmem_ld 3.1, compute_dep 1.1 | sync 2.4, vmem_ld 0.8 |
| regs / grid x block | 32 / 64x256 | 32 / 200x256 | 32 / 1600x512 | 32 / 400x256 | 32 / 100x256 | 96 / 400x128 | 90 / 800x128 |

Full table: `profile/step1-q4-51200x5120.md` (`tools/acu_summary.py`).

Reading:
- No overfetch anywhere (DRAM bytes = data bytes) and every pattern issues 64 B LLC requests,
  so sector/request efficiency is not the difference.
- DRAM is busy ~98% of the time for every probe, but delivers 85% of peak bytes for contiguous
  or 512 B-per-row instructions and only 69-71% for the fragment order: the fragment order costs
  DRAM efficiency per active cycle. ACU exposes no row-buffer counters; the only thing that
  differs is the address order (four rows of 32-128 B per instruction), so DRAM row/bank locality
  is the inferred cause.
- It is not in-flight depth: the fragment order stays at 0.95-0.99 TB/s for prefetch depth
  U = 1..8 and occupancy 19-92% (Asys sweep below).
- The prototype reaches 866 GB/s = ~91% of its own pattern ceiling (fragT4 952). The rest is
  consumer/scheduling: issue active 45%, IPC 2.0, vmem_ld 3.1 and compute_dependency 1.1 stalls
  per issue, and 400 CTAs = 6.25 per CU (one partial wave).
- Production TC (AIU->TSM) does not saturate DRAM (active 89%, 52.6% of peak) and stalls on
  barriers (sync 2.4): its consumer, not its access pattern, is the limit.

## 2. How contiguous must a row segment be (Asys, `profile/pattern/sweep2`)

`seg`: each warp instruction reads R rows x 512/R contiguous bytes (ld.global, 16 B per lane).
`cpa`: the same pattern through `ppu.cp.async` into a per-warp shared ring, consumed by
ld.shared. Best over WK = 4/8/16 warps and depth U, n=51200 (125 MiB):

| R rows per instruction | bytes per row segment | seg GB/s | cpa GB/s |
|---|---|---|---|
| 1 | 512 | 1139 | 1124 |
| 2 | 256 | 1121 | 1119 |
| 4 | 128 | 1071 | 1008 |
| 8 | 64 | 985 | 935 |
| fragment order (4 rows x 32-128 B) | | 949-989 | |
| flat | | 1180 | |

On n=5120 (62.5 MiB, only 20-160 CTAs) the same holds once each CTA has 16 warps x depth 4-8:
seg R=2 1112, cpa R=2 1092, cpa R=4 1080, fragment order best 1020, flat 1137-1151.

So: at least 256 contiguous bytes of one plane row per warp instruction (a CTA spanning >= 128
columns of the uint16 plane) recovers ~95% of streaming bandwidth, and cp.async staging is as
good as direct loads at R <= 2.

## 3. Small calls (decode-sized reads)

Streaming floor (flat, best grid, Asys): 16 KB 0.92 us, 1 MiB 1.94, 2 MiB 2.90, 4 MiB 4.72,
8 MiB 8.38, 16 MiB 15.1, 32 MiB 29.5 us. With n=1024 and k split over S CTAs, row-contiguous
seg/cpa reach 3.5 / 5.2 / 8.6-8.8 / 15.6-15.7 us at 2 / 4 / 8 / 16 MiB, while the fragment order
without k split needs 3.6 / 6.4 / 11.5 / 22.4 us.

Q8_0 4096x2048 (8.5 MiB, ACU): prototype 13.4 us, production q8_refresh 14.2 us; both only
45-48% of DRAM peak with DRAM active 65-72% (not saturated) and 128 CTAs: small calls are bound
by CTA count / ramp-up / tail, plus the separate Split-K reducer (0.9-3.8 us per call in the
baseline Asys run), not by the access pattern. Table: `profile/step1-small-and-simt.md` (prototype and q8_refresh 4096x2048 on dev1; the 8192x2048 and
SIMT 5120x25600 columns are the 11-08 production captures, context only).

Upper bound for the target: if every manifest call ran at the streaming floor for its weight
bytes (single kernel), M=1 model time would drop 36.7% (qwen3-32b), 35.1% (qwen38-27b), 31.3%
(qwen35-35b-a3b), 31.7% (qwen35-122b-a10b), 38.0% (deepseek-v4-flash); equal-weight 34.6%.
The 20% target needs ~60% of that headroom on every model, small Q8/MoE calls included.

## 4. Conclusion of step 1

- Confirmed: on large shapes the prototype is limited by its access pattern (MMA-fragment order
  loads of the canonical K-pack planes cap DRAM at ~69-71% of peak = 0.95-0.99 TB/s), not by
  in-flight bytes. Row-contiguous global->shared staging with >= 256 B per row per warp
  instruction (cp.async or AIU) removes that cap (1.12 TB/s measured with cp.async).
- The production TC route already stages through TSM but is consumer/barrier bound (52.6% of
  DRAM peak); the prototype consumer (exact codes, group rotation, FP32 window affine) issues
  13% fewer instructions than it on the same shape.
- Small calls are a separate problem (DRAM not saturated; CTA count, tail, reducer launch).

## Artifacts and reproduction

- Probes: `tools/pattern_probe.cu` (patterns flat/rows/frag/seg/cpa, stdin config list),
  `tools/pattern_sweep.sh DEVS OUTDIR [CONFIGS]` (Asys, waits for an idle PPU),
  `tools/pattern_asys.py` (attribution), `tools/pattern_acu.sh DEV OUTDIR < configs`.
- Results: `profile/pattern/sweep1` (frag/rows/flat, size sweep), `profile/pattern/sweep2`
  (seg/cpa), `profile/pattern/acu`, `profile/acu-v3` (prototype + q8_refresh),
  `profile/baseline-acu` (production), summaries by `tools/acu_summary.py`.
- Prototype screen: `runs/proto/screen-v3` (+ `screen-v3b/round0/v3.contaminated0`).
- Not used for any conclusion: `source/quactlize/execution/mma_staged.cuh` (a cp.async-staged
  variant written before the hand-over; correctness 29/29 valid cases vs the gguf-py oracle,
  never timed; `runs/proto/screen-staged-v1` was aborted and is partial).
