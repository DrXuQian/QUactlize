# Back-test targets: what was measured, under what conditions, and what to reproduce

Every performance figure this project has recorded, with the conditions that make it meaningful, so a box run can
be compared against the right one. Built by sweeping `memory/` and `dev/fold_derivation/SWEEP_STATE.md`
2026-08-05, preferring the newest record and taking the highest figure where several exist for one condition.

**WHY A TABLE RATHER THAN A NUMBER.** Six different int4 MFU figures exist between 53.1% and 65.0%, all correct,
all for different `(gs, TileK, warp shape, sweep grid)`. Quoting one without its conditions is how "we used to
get 65%" becomes an unfalsifiable claim. Each row below carries what it needs to be checked.

**PEAK is 500 TFLOP/s** (fp16). MFU = TF/s ÷ 500. Where a record gives only µs, the FLOP is `2·M·N·K`.

---

## A. DENSE — M=2048, N=K=4096, L=1

68.72 GFLOP per call (`2·2048·4096·4096`). "L=1" means the grouped kernel with one expert, which IS dense; the
records use both harnesses and they are the same measurement.

| # | width | gs | config | µs | MFU | recorded | source |
|---|---|---|---|---|---|---|---|
| A1 | **int4** | **32** | after adding `w64x32` | **211.33** | **65.0%** | 2026-07-28 | `ppu-q3-bconcat-tuned` |
| A2 | int1 | 32 | after `w64x32`, `PPU_B_CHUNK=1` | 215.23 | 63.9% | 2026-07-28 | same |
| A3 | int1 | 32 | `(64,128,64) w64x64 s2`, `PPU_B_CHUNK=1`, ScaleOnly | — | 63.7% | 2026-07-27 | `ppu-b-chunk-int1-637` |
| A4 | int2 | 32 | after `w64x32` | 233.76 | 58.8% | 2026-07-28 | `ppu-q3-bconcat-tuned` |
| A5 | Q3 (int2+int1 B-concat) | 32 | after all four levers | 255.47 | 53.8% | 2026-07-28 | same |
| A6 | int4 | 32 | `64x64:64 s4` — **before `w64x32` was in the grid** | 246 | 55.8% | 2026-07-25 | `ppu-aiu-n-contiguous-load` |
| A7 | int4 | 16 | `64x64:64 s3` — **before `w64x32`** | 259 | 53.1% | 2026-07-25 | same |
| A8 | int1 | 32 | `32,128,128` F=2 | — | 54.3% | 2026-07-25 | same |
| A9 | int2 | 32 | `64,64,64` F=2 | — | 53.2% | 2026-07-25 | same |
| A10 | Q3 B-concat | 32 | before the four levers | 822.60 | 16.7% | 2026-07-28 | `ppu-q3-bconcat-tuned` |

**A1 IS THE ONE TO REPRODUCE.** A6 and A7 are the same width at the same shape and are *superseded*: their sweep
grid did not contain `w64x32`, which the record measures at **+8.6 points for int4**. Reproducing 55.8% instead
of 65.0% therefore means the warp shape is missing again, not that the kernel regressed.

**gs=16 for int4 is NOT recorded post-`w64x32`.** The record gives the delta instead: moving int4 to gs=16 costs
**10.8%**, so A1 implies ≈234 µs / **58.7%**. That is arithmetic over a measurement, not a measurement — the box
run settles it. (A recollection of "60+% at gs=16" exists and is not what the arithmetic gives.)

## B. DENSE — other shapes and group sizes

| # | shape | gs | config | figure | recorded | source |
|---|---|---|---|---|---|---|
| B1 | 2048×4096×4096 | 128 | `64×64 / 32×32 / s4` | **305 TF/s = 61%** | 2026-07-22 | `ppu-cutlass-w4a16-actlize` |
| B2 | 2048×4096×4096 | 128 | default `32×32` tile | **25%** | 2026-07-22 | same — **the control** |
| B3 | 2048×4096×4096 | 128 | official finegrained, `64×64×128 / s3` | 56.6% | 2026-07-22 | same |
| B4 | 2048×4096×4096 | 128 | hand-written Marlin | 215 TF/s = 43% | 2026-07-22 | same |
| B5 | 4096³, dense L=1 | 128 | — | **62%** | 2026-07-22 | same |
| B6 | dequant→fp16 + dense cuBLAS GEMM | — | not fused | **59–66%** | — | `ppu-w4a16-path-by-m`, `ppu-aiu-int4-viable` |
| B7 | fused mixed-input, same shape | 32 | — | 40.6% | — | same — the 20–26 point fusion cost |

**B2 is the control that matters most.** If a re-run's `32x32` rows do not read about 25%, the thing being
measured now is not the thing measured then, and every other comparison is against the wrong baseline.

## C. MoE — grouped, ragged vs uniform

| # | fixture | rows/expert | gs | config | figure | recorded | source |
|---|---|---|---|---|---|---|---|
| C1 | A3B FC1, N=512 K=2048, 4096 tok, 256 exp top-8 (32,768 rows) | ragged | 32 | `i4 64x128:64 w64x16 s6` | 423.96 µs / 162.1 TF/s / **32.4% useful** | 2026-08-04 | `SWEEP_STATE.md` |
| C2 | A3B FC1, N=1024 K=2048, 2048 tok top-8 over 128 exp (16,384 rows) | ragged | 32 | cutlass | 416 µs / 165 TF/s / **33.0% useful** | 2026-07-22 | `ppu-moe-w4a16-cutlass-vs-handwritten` |
| C3 | same as C2 | **uniform** | 32 | cutlass | **49.2%** | 2026-07-22 | same |
| C4 | avg 512 rows/expert | ragged | 32 | cutlass | 39.1% | 2026-07-22 | same |
| C5 | A3B FC1 | ragged | 32 | hand-written AIU, `NST=4 BMR=128` | 0.416 ms / 165 TF/s / 33% useful, **46% issued** | — | `ppu-moe-q4k-aiu` |
| C6 | FC1 real config | — | — | `BMR=128 MWARPS=4 NST=4` | 0.524 ms / **issued 262 TF/s = 52%** | — | same |
| C7 | qwen3-30B-A3B, uniform (m=256=4×TM) | uniform | 128 | — | 55–59% | 2026-07-22 | `ppu-cutlass-w4a16-actlize` |
| C8 | ragged dropless FC1, N=1536 K=2048 | ragged | 128 | — | ~40% | 2026-07-22 | same |
| C9 | ragged dropless FC2, N=2048 K=768 | ragged | 128 | — | ~31% | 2026-07-22 | same |

**C1 and C2 are the same total work** — `2·16384·1024·2048 = 2·32768·512·2048 = 68.72 GFLOP` — so the 33.0%→32.4%
step across a redistribution is a 1.9% harness difference, not a regression.

**USEFUL vs ISSUED is not a presentation choice.** C5 reports both: 33% useful, 46% issued. The gap is masked
rows burning mma cycles. C6's 52% is an ISSUED figure and must never be compared against a useful one.

**Do not benchmark ragged against uniform.** C3's 49.2% has no masked rows; the recorded ~16-point gap to C1/C2 is
the structural masked-row tax and is implementation-independent (the hand-written AIU kernel ties the cutlass one
at 33% on the same ragged shape).

## D. Decode — GEMV, CUDA cores

Bandwidth-referenced, not MFU. Shape is the decode band; see `ppu-gemv-alu-bound-not-bandwidth`.

| # | width | time | GB/s | %HBM | config | recorded |
|---|---|---|---|---|---|---|
| D1 | int4 | 16.05 µs | 1310.7 | **47.4%** | `tileK s32/t64 N4 C2` | 2026-08-03 |
| D2 | int2 | 15.86 µs | 797.8 | 28.8% | `native s16/t128 N8 C2` | 2026-08-03 |
| D3 | int1 | 15.88 µs | 532.4 | 19.2% | `native s32/t64 N8 C2` | 2026-08-03 |
| D4 | grouped tensor-core GEMM at the same band | 20.74 µs | — | 37.5% | `i4 16x32:256 w16x16 s2` | — |
| D5 | best GEMV before the 2026-08-03 retune | 22.27 µs | — | 34.1% | `int4 native s16/t128 CtaN2 Chunk2` | — |

**D1–D3 within 1.1% of each other across a 2.5× byte range** is the finding: decode is ALU-bound, not
bandwidth-bound, so int4 is effectively free relative to int1.

---

## The order to back-test

One at a time, each against its own row, because a failure means different things at different rows.

1. **A1** — int4 gs=32. The headline. Checks the sweep grid contains `w64x32` and the collective still performs.
2. **B2** — the `32x32` control, read out of the same A1 sweep at gs=128. Costs nothing extra and validates the
   baseline before anything else is believed.
3. **dense ≤ grouped(L=1)** — the invariant (`analyse.py --invariant`). Needs no historical figure at all.
4. **A7 / gs=16** — settles the 58.7%-vs-"60+%" disagreement between the record's arithmetic and recollection.
5. **B1** — gs=128, a different sweep's number, to check the two sweeps still agree with each other.
6. **C1** — the MoE ragged figure, whose harness was already back-tested to within 1.9% on 2026-08-04.

## What would invalidate a comparison

* **A different `w64x32` presence.** The single largest recorded config effect (+8.6 points int4, +7.2 int2).
* **A different TileK.** `scale_first` is 256/bits — 64 for int4 — and the bench's default matches. A run with
  `TSK=` set is a different row and not comparable to A1.
* **`PPU_B_CHUNK`.** A2/A3 require it; A1 does not (the fold collective gates int4 out of chunking entirely).
* **useful vs issued** on any MoE row.
* **A stale config table.** The `.inc` carries `(bits, TileK)` and the binary `static_assert`s it, so this one
  cannot fail silently — it fails to compile.
