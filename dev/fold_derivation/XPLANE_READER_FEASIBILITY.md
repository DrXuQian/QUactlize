# Shipping BC xplane whole-word reader feasibility

This is a host-only feasibility verdict for INBOX 169. It changes no kernel, placement, or artifact bytes.
The denominator is evaluated from `arrangement_supported_v`; it is not a copied support list.
L137 independently anchors every slot permutation to the production `xplane::place_from_map` writer.

## Verdict

* Direct CPW window in one word: **0/26 planes**.
* Fixed-row multiword closure: **20/26 planes**.
* Complete formats (all resident planes close): **11/17 arrangements**.
* Fast-plan versus scalar-address comparisons: **1,310,720 coordinates**, exhaustive.
* Total classified domain: **1,703,936 coordinates**; the remaining **393,216** are explicit `UNSUPPORTED`, never scalar fallback.
* Conclusion: **partial generalisation, not a universal reader**. The existing Q4 path is a multiword
  positive: 32 logical codes close over four words and use one fixed 8x4 register permutation.
  Literal consecutive-CPW blocks do not individually occupy one word.
* Criterion (c) is measured for RTX 5090 Q4/A64 and is deliberately not duplicated in this host-model
  generator.  The hash-bound timings, binary/result identities, and causal limits live in
  `BC_Q4_GEMV_5090_AB.md`; PPU instruction count and time remain unmeasured for this exact shipping path.
* Criterion (a) below counts source-level address/extraction primitives. The Q4 permutation is lowered on sm120,
  but this remains no claim about PPU code generation.

## Arrangement table

| T | ArtifactTileK | planes | fast | closure K | permutations | source before | source after | rejection |
|---|---:|---:|:---:|---:|---|---|---|---|
| Q2_K | 32 | 1 | NO | - | - | 14 slot terms + 1 scalar plane loads/code | explicitly unsupported | low:WORD_MIXES_LOGICAL_ROWS |
| Q2_K | 64 | 1 | YES | 64 | P2x64 | 14 slot terms + 1 scalar plane loads/code | 0 slot terms + 1/16 word loads/code | - |
| Q2_K | 128 | 1 | YES | 64 | P2x64 | 14 slot terms + 1 scalar plane loads/code | 0 slot terms + 1/16 word loads/code | - |
| Q2_K | 256 | 1 | YES | 64 | P2x64 | 14 slot terms + 1 scalar plane loads/code | 0 slot terms + 1/16 word loads/code | - |
| Q3_K | 64 | 2 | NO | - | P2x64, - | 28 slot terms + 2 scalar plane loads/code | explicitly unsupported | high:WORD_MIXES_LOGICAL_ROWS |
| Q3_K | 128 | 2 | YES | 128 | P2x64, P1x128 | 28 slot terms + 2 scalar plane loads/code | 0 slot terms + 3/32 word loads/code | - |
| Q3_K | 256 | 2 | YES | 128 | P2x64, P1x128 | 28 slot terms + 2 scalar plane loads/code | 0 slot terms + 3/32 word loads/code | - |
| Q4_K | 32 | 1 | YES | 32 | P4x32 | 14 slot terms + 1 scalar plane loads/code | 0 slot terms + 1/8 word loads/code | - |
| Q4_K | 64 | 1 | YES | 32 | P4x32 | 14 slot terms + 1 scalar plane loads/code | 0 slot terms + 1/8 word loads/code | - |
| Q4_K | 128 | 1 | YES | 32 | P4x32 | 14 slot terms + 1 scalar plane loads/code | 0 slot terms + 1/8 word loads/code | - |
| Q4_K | 256 | 1 | YES | 32 | P4x32 | 14 slot terms + 1 scalar plane loads/code | 0 slot terms + 1/8 word loads/code | - |
| Q5_K | 64 | 2 | NO | - | P4x32, - | 28 slot terms + 2 scalar plane loads/code | explicitly unsupported | high:WORD_MIXES_LOGICAL_ROWS |
| Q5_K | 128 | 2 | NO | - | P4x32, - | 28 slot terms + 2 scalar plane loads/code | explicitly unsupported | high:WORD_MIXES_LOGICAL_ROWS |
| Q5_K | 256 | 2 | NO | - | P4x32, - | 28 slot terms + 2 scalar plane loads/code | explicitly unsupported | high:WORD_MIXES_LOGICAL_ROWS |
| Q6_K | 32 | 2 | NO | - | P4x32, - | 28 slot terms + 2 scalar plane loads/code | explicitly unsupported | high:WORD_MIXES_LOGICAL_ROWS |
| Q6_K | 64 | 2 | YES | 64 | P4x32, P2x64 | 28 slot terms + 2 scalar plane loads/code | 0 slot terms + 3/16 word loads/code | - |
| Q6_K | 128 | 2 | YES | 64 | P4x32, P2x64 | 28 slot terms + 2 scalar plane loads/code | 0 slot terms + 3/16 word loads/code | - |

## Plane table

`permutation` lists logical K offsets in physical word/slot order. Consecutive chunks of CPW entries
are the exact within-word permutations.

| T | A | plane | bits | F | CPW | direct CPW word | minimum closed K | words | permutation | reason |
|---|---:|---|---:|---:|---:|:---:|---:|---:|---|---|
| Q2_K | 32 | low | 2 | 4 | 16 | NO | - | - | `-` | WORD_MIXES_LOGICAL_ROWS |
| Q2_K | 64 | low | 2 | 2 | 16 | NO | 64 | 4 | `0,8,16,24,32,40,48,56,1,9,17,25,33,41,49,57,2,10,18,26,34,42,50,58,3,11,19,27,35,43,51,59,4,12,20,28,36,44,52,60,5,13,21,29,37,45,53,61,6,14,22,30,38,46,54,62,7,15,23,31,39,47,55,63` | - |
| Q2_K | 128 | low | 2 | 1 | 16 | NO | 64 | 4 | `0,8,16,24,32,40,48,56,1,9,17,25,33,41,49,57,2,10,18,26,34,42,50,58,3,11,19,27,35,43,51,59,4,12,20,28,36,44,52,60,5,13,21,29,37,45,53,61,6,14,22,30,38,46,54,62,7,15,23,31,39,47,55,63` | - |
| Q2_K | 256 | low | 2 | 1 | 16 | NO | 64 | 4 | `0,8,16,24,32,40,48,56,1,9,17,25,33,41,49,57,2,10,18,26,34,42,50,58,3,11,19,27,35,43,51,59,4,12,20,28,36,44,52,60,5,13,21,29,37,45,53,61,6,14,22,30,38,46,54,62,7,15,23,31,39,47,55,63` | - |
| Q3_K | 64 | low | 2 | 2 | 16 | NO | 64 | 4 | `0,8,16,24,32,40,48,56,1,9,17,25,33,41,49,57,2,10,18,26,34,42,50,58,3,11,19,27,35,43,51,59,4,12,20,28,36,44,52,60,5,13,21,29,37,45,53,61,6,14,22,30,38,46,54,62,7,15,23,31,39,47,55,63` | - |
| Q3_K | 64 | high | 1 | 4 | 32 | NO | - | - | `-` | WORD_MIXES_LOGICAL_ROWS |
| Q3_K | 128 | low | 2 | 1 | 16 | NO | 64 | 4 | `0,8,16,24,32,40,48,56,1,9,17,25,33,41,49,57,2,10,18,26,34,42,50,58,3,11,19,27,35,43,51,59,4,12,20,28,36,44,52,60,5,13,21,29,37,45,53,61,6,14,22,30,38,46,54,62,7,15,23,31,39,47,55,63` | - |
| Q3_K | 128 | high | 1 | 2 | 32 | NO | 128 | 4 | `0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,1,9,17,25,33,41,49,57,65,73,81,89,97,105,113,121,2,10,18,26,34,42,50,58,66,74,82,90,98,106,114,122,3,11,19,27,35,43,51,59,67,75,83,91,99,107,115,123,4,12,20,28,36,44,52,60,68,76,84,92,100,108,116,124,5,13,21,29,37,45,53,61,69,77,85,93,101,109,117,125,6,14,22,30,38,46,54,62,70,78,86,94,102,110,118,126,7,15,23,31,39,47,55,63,71,79,87,95,103,111,119,127` | - |
| Q3_K | 256 | low | 2 | 1 | 16 | NO | 64 | 4 | `0,8,16,24,32,40,48,56,1,9,17,25,33,41,49,57,2,10,18,26,34,42,50,58,3,11,19,27,35,43,51,59,4,12,20,28,36,44,52,60,5,13,21,29,37,45,53,61,6,14,22,30,38,46,54,62,7,15,23,31,39,47,55,63` | - |
| Q3_K | 256 | high | 1 | 1 | 32 | NO | 128 | 4 | `0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,1,9,17,25,33,41,49,57,65,73,81,89,97,105,113,121,2,10,18,26,34,42,50,58,66,74,82,90,98,106,114,122,3,11,19,27,35,43,51,59,67,75,83,91,99,107,115,123,4,12,20,28,36,44,52,60,68,76,84,92,100,108,116,124,5,13,21,29,37,45,53,61,69,77,85,93,101,109,117,125,6,14,22,30,38,46,54,62,70,78,86,94,102,110,118,126,7,15,23,31,39,47,55,63,71,79,87,95,103,111,119,127` | - |
| Q4_K | 32 | low | 4 | 2 | 8 | NO | 32 | 4 | `0,8,16,24,1,9,17,25,2,10,18,26,3,11,19,27,4,12,20,28,5,13,21,29,6,14,22,30,7,15,23,31` | - |
| Q4_K | 64 | low | 4 | 1 | 8 | NO | 32 | 4 | `0,8,16,24,1,9,17,25,2,10,18,26,3,11,19,27,4,12,20,28,5,13,21,29,6,14,22,30,7,15,23,31` | - |
| Q4_K | 128 | low | 4 | 1 | 8 | NO | 32 | 4 | `0,8,16,24,1,9,17,25,2,10,18,26,3,11,19,27,4,12,20,28,5,13,21,29,6,14,22,30,7,15,23,31` | - |
| Q4_K | 256 | low | 4 | 1 | 8 | NO | 32 | 4 | `0,8,16,24,1,9,17,25,2,10,18,26,3,11,19,27,4,12,20,28,5,13,21,29,6,14,22,30,7,15,23,31` | - |
| Q5_K | 64 | low | 4 | 1 | 8 | NO | 32 | 4 | `0,8,16,24,1,9,17,25,2,10,18,26,3,11,19,27,4,12,20,28,5,13,21,29,6,14,22,30,7,15,23,31` | - |
| Q5_K | 64 | high | 1 | 4 | 32 | NO | - | - | `-` | WORD_MIXES_LOGICAL_ROWS |
| Q5_K | 128 | low | 4 | 1 | 8 | NO | 32 | 4 | `0,8,16,24,1,9,17,25,2,10,18,26,3,11,19,27,4,12,20,28,5,13,21,29,6,14,22,30,7,15,23,31` | - |
| Q5_K | 128 | high | 1 | 2 | 32 | NO | - | - | `-` | WORD_MIXES_LOGICAL_ROWS |
| Q5_K | 256 | low | 4 | 1 | 8 | NO | 32 | 4 | `0,8,16,24,1,9,17,25,2,10,18,26,3,11,19,27,4,12,20,28,5,13,21,29,6,14,22,30,7,15,23,31` | - |
| Q5_K | 256 | high | 1 | 1 | 32 | NO | - | - | `-` | WORD_MIXES_LOGICAL_ROWS |
| Q6_K | 32 | low | 4 | 2 | 8 | NO | 32 | 4 | `0,8,16,24,1,9,17,25,2,10,18,26,3,11,19,27,4,12,20,28,5,13,21,29,6,14,22,30,7,15,23,31` | - |
| Q6_K | 32 | high | 2 | 4 | 16 | NO | - | - | `-` | WORD_MIXES_LOGICAL_ROWS |
| Q6_K | 64 | low | 4 | 1 | 8 | NO | 32 | 4 | `0,8,16,24,1,9,17,25,2,10,18,26,3,11,19,27,4,12,20,28,5,13,21,29,6,14,22,30,7,15,23,31` | - |
| Q6_K | 64 | high | 2 | 2 | 16 | NO | 64 | 4 | `0,8,16,24,32,40,48,56,1,9,17,25,33,41,49,57,2,10,18,26,34,42,50,58,3,11,19,27,35,43,51,59,4,12,20,28,36,44,52,60,5,13,21,29,37,45,53,61,6,14,22,30,38,46,54,62,7,15,23,31,39,47,55,63` | - |
| Q6_K | 128 | low | 4 | 1 | 8 | NO | 32 | 4 | `0,8,16,24,1,9,17,25,2,10,18,26,3,11,19,27,4,12,20,28,5,13,21,29,6,14,22,30,7,15,23,31` | - |
| Q6_K | 128 | high | 2 | 1 | 16 | NO | 64 | 4 | `0,8,16,24,32,40,48,56,1,9,17,25,33,41,49,57,2,10,18,26,34,42,50,58,3,11,19,27,35,43,51,59,4,12,20,28,36,44,52,60,5,13,21,29,37,45,53,61,6,14,22,30,38,46,54,62,7,15,23,31,39,47,55,63` | - |

## Interpretation

Fast complete arrangements are Q2 A=64/128/256; Q3 A=128/256; Q4 A=32/64/128/256; and
Q6 A=64/128. Q5 is only a partial-plane opportunity: its low plane closes, but every supported high
plane word mixes logical rows, so silently retaining per-code `code_at` for the high plane would violate
the preregistered property. Q2 A=32, Q3 A=64, and Q6 A=32 fail for the same row-mixing reason.

The three reusable fixed permutations are P4x32, P2x64, and P1x128. P4x32 is byte-for-byte the
shipping `q4_group` formula `k=(p&3)*8+(p>>2)`. The generic Q4 reader now selects the whole-word path for
A=32/64/128/256; only dense CUDA A64 is admitted to the separately measured cooperative topology.

### Preregistered criteria (verbatim from INBOX 169)

* (a) **源码级**:每码指令数,逐 `(T, ArtifactTileK)`,改前/改后同表。与编译器无关,先报这个。
* (b) **覆盖**:走快读的 `(T, ArtifactTileK, High)` 组合数 / 受支持组合总数。分母来自 `arrangement_supported_v` 的枚举,**不是手写清单**。
* (c) **box 实测**:② 同 shape 的时间。**注意 ② 现在没有基线**,所以第一次跑要先补 baseline 再谈提升。

### Negative controls (verbatim from INBOX 169)

1. 快读与 `code_at` 必须在同一测试里逐码比对,**全部受支持组合、全部 (n,k)**,不是抽样。`code_at` 保留为 oracle,不许删。
2. 植入一个**字内置换错一位**的故障,必须判红。这是 `q4_group` 那类"physically contiguous 但逻辑 K 是转置"的实际失败形态。
3. 对 (b) 的分母植入一个"少枚举一个受支持组合"的故障,必须判红 —— 否则覆盖率可以靠缩分母刷。

### Scope limits (verbatim from INBOX 169)

* 不动 π,不动离线摆放,不动 artifact 字节。**prefill 的零成本性质是硬约束。**
* 不碰 ① 和 ③;它们是否退役是另一件事(见 167 末尾),本任务不预设。
* 2.75 条/对是 sm_120 的;PPU codegen 未测。本任务的 (a) 是源码级计数,**不许把它说成 PPU 实测**。
* TODO #58 与出货路无关(见第四节),**不要顺手做**。
