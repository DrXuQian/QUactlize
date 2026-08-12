# Back-test targets: what was measured, under what conditions, and what to reproduce

Every performance figure this project has recorded, with the conditions that make it meaningful, so a box run can
be compared against the right one. Built by sweeping `memory/` and `dev/fold_derivation/SWEEP_STATE.md`
2026-08-05, preferring the newest record and taking the highest figure where several exist for one condition.

**WHY A TABLE RATHER THAN A NUMBER.** Six different int4 MFU figures exist between 53.1% and 65.0%, all correct,
all for different `(gs, TileK, warp shape, sweep grid)`. Quoting one without its conditions is how "we used to
get 65%" becomes an unfalsifiable claim. Each row below carries what it needs to be checked.

**PEAK is 500 TFLOP/s** (fp16). MFU = TF/s ÷ 500. Where a record gives only µs, the FLOP is `2·M·N·K`.

> ## ⚠ SECTIONS A, B AND C ARE THE **scale_first** PATH — AND SO IS DENSE PREFILL AS SHIPPED.
>
> The registry gives each format two TileKs because they are two different consumers:
> `scale_first` (a pre-pass produces separate scale/zero planes; TileK = 256/bits, so 64 for int4) and
> `fully_quantized` (the GEMM consumes the packed GGUF metadata unit natively; TileK = 256 for Q2/Q4).
>
> **THE `.so` SHIPS BOTH, and an earlier version of this block claimed otherwise.** The exported entries are
> `quactlize_ppu_dense_lowbit*` (scale_first, dense) alongside `quactlize_ppu_dense_fully_quantized*` and
> `quactlize_ppu_grouped_fully_quantized*`. So section A is not merely a harness check — for dense prefill it is
> a measurement of a path that ships. The earlier wording ("reproduce 65% says nothing about what will ship")
> was wrong and would have mis-ranked the whole back-test.
>
> What remains true is narrower and still matters: **`fully_quantized` has no tensor-core prefill measurement on
> either operator**, and section E is everything that exists for it. That gap belongs to the packed-metadata
> consumer -- decode, and the grouped/MoE side, where there is no `*_grouped_lowbit` entry at all.

---

## A. DENSE — M=2048, N=K=4096, L=1
<!-- route: dense_lowbit -->


68.72 GFLOP per call (`2·2048·4096·4096`). "L=1" means the grouped kernel with one expert, which IS dense; the
records use both harnesses and they are the same measurement.

| # | width | gs | config | µs | MFU | recorded | source |
|---|---|---|---|---|---|---|---|
| A0 | **int4** | **32** | `64x64:64x32:s3` — **the DENSE bench, its first validated number** | **209.27** | **65.7%** | 2026-08-05 | box, `test_lowbit_dense_bench` |
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

**A0 WAS RE-RUN ON 2026-08-06 TO VALIDATE THE actlize EXTRACTION, and it is the only evidence that the move is
sound.** 209.36 µs / 65.6% against A0's 209.27 / 65.7% -- +0.09 µs, +0.04%, noise. Same box, same geometry; the
config string gained a TileK segment (`64x64:64x32:s3` -> `64x64x64:64x32:s3`) when TileK became a row field.

It is worth saying what this re-run does and does not establish, because it was the ONLY device evidence for a
change that moved every collective out of the actlize fork. It establishes that hgcc compiles the extracted tree
and that the kernel reached is the same one: `ppu_mixed_policy::ArtifactFoldedSchedule` now wraps
UNCONDITIONALLY (so an unfolded row lands on quactlize's collective rather than silently selecting actlize's),
and that wrapper is routing-neutral only by derivation -- `ArtifactLowFold=1` gives `HasFold=false` and
`BaseSchedule=Base`. A number 0.04% from the pre-extraction one is what routing-neutral looks like; a different
kernel would not land there. It establishes nothing about the other 631 rows, the grouped operator, or any
width but int4 at gs=32.

**A0 SETTLES THREE THINGS AT ONCE, and it is the first number this bench has ever produced.** Until 2026-08-05
`test_lowbit_dense_bench` asserted on its first gs=32 config, so every dense figure below came from the grouped
harness at L=1. A0 is the same geometry as A1, run through the dense kernel — plain `ProblemShape`,
`EpilogueSimtVectorized`, no pointer arrays, no GroupScheduler.

1. **The sub-four-warp quarantine was wrong, and now on hardware rather than in the compiler.** `w64x32` is two
   warps. `ppu_tactic_space.hpp` excluded it from the dense space on a device abort observed 2026-08-04; it does
   not abort. The exclusion was deleted the same day it was refuted, and this is the run that refutes it.
2. **`dense <= grouped(L=1)` is not violated, and the sign agrees with the source.** 209.27 against 211.33.
   ⚠ **THAT IS PARITY, NOT AN ADVANTAGE, AND THIS ROW MUST NOT BE CITED AS ONE.** 0.98% between two runs on
   different dates is far inside the cross-run spread this file's own selection procedure exists to defeat
   (13%, which is why the sweep takes a median over repeated passes). What A0 establishes is that dense reaches
   the same place grouped does. `DENSE_VS_GROUPED_L1.md` predicts dense should be the faster of the two — every
   structural asymmetry is grouped doing MORE (a pointer indirection per output tile, per-CTA m-tile prefix
   decode, host-built ptr/stride arrays, a workspace) — and the sign is consistent with that, which is worth
   something and is not a measurement of the gap. **Attributing the difference to grouped overhead needs a
   same-run, interleaved, higher-repetition A/B.** Until that exists, quote parity.
3. **The quarantine cost 5.1 points.** Dense's best reachable row under it measured 60.6%.

**A1 IS STILL THE ONE TO REPRODUCE** for the grouped harness, and A0 for the dense one. They are the same
computation; expect them within noise of each other, and treat a gap in EITHER direction as needing the same-run
comparison before it means anything.

**A1 IS THE ONE TO REPRODUCE.** A6 and A7 are the same width at the same shape and are *superseded*: their sweep
grid did not contain `w64x32`, which the record measures at **+8.6 points for int4**. Reproducing 55.8% instead
of 65.0% therefore means the warp shape is missing again, not that the kernel regressed.

**gs=16 for int4 WAS measured post-`w64x32`; the earlier version of this paragraph was false.** The same
grouped-L=1 sweep recorded **234.16 µs** for `(64,64,64) w64x32 s3`, then a later equivalence-refactor A/B
recorded **228.13 → 227.35 µs** at the same group size.  Those are historical device measurements, not the
old 58.7% extrapolation.  What remains missing is narrower: the current HEAD has not re-run that row through the
native dense operator, so it is a stale regression target rather than an absent measurement.  See
`dev/fold_derivation/TODO36_MEASUREMENT_GAPS.md` for the evidence and the queued fresh run.

## B. DENSE — other shapes and group sizes
<!-- route: dense_lowbit -->


> **⚠ EVERY ROW IN THIS SECTION IS gs=128, AND NONE HAS BEEN RE-RUN SINCE THE PATH THAT BROKE THEM WAS FIXED.**
>
> This block used to say gs=128 was unreproducible, and that is no longer true. On 2026-08-05 `--g=128` hit an
> unconditional `assert(false)` in `ppu_mma_aiu_multistage_mixed_input.hpp` -- the COARSE scale path
> (`Scale_TileK <= K_BLOCK_MAX`) copied the scale and then ran `if constexpr (false) {} else { assert(false); }`,
> whose else branch always executes. actlize `a7a8ea91` ("Finish mixed-input coarse scale path") implemented it.
> That file now has 45 `assert(` sites of which 45 are `static_assert`: **zero runtime asserts**. The dense bench
> builds all four group sizes including 128, and the full 293-row table compiles clean through the local syntax
> gate.
>
> So these rows are back-test CANDIDATES again. What is still missing is a RUN: nothing has executed gs=128 since
> the fix, so there is no confirmation of numerical correctness on the COARSE path, only that it instantiates.
> Verify correctness before quoting any timing from this section.
>
> Note for prioritisation, not as a reason to skip it: **gs=128 is not a GGUF k-quant group size.** The shipping
> registry has only 16 (Q2/Q3/Q6) and 32 (Q4/Q5), so no shipping path reaches COARSE. These figures were measured
> in July through the older `Kernels/general/w4a16_gemm/cutlass_w4a16` bench, and their value is as a comparison
> against that bench rather than as a product claim.


**WIDTH IS DERIVED, NOT ASSUMED.** This section had no width column at all, so B1 and B3 named a config and
no format and `ci/check_backtest_configs.py` reported them UNPARSED rather than guessing which table to look
in. Every row here is the W4A16 path: the section is routed `dense_lowbit` and its source memory is
`ppu-cutlass-w4a16-actlize`, i.e. 4-bit weights with fp16 activations. B6 is the only exception and is marked
`—`: it is dequant-to-fp16 followed by a dense cuBLAS GEMM, so it has no low-bit tactic at all.

| # | width | shape | gs | config | figure | recorded | source |
|---|---|---|---|---|---|---|---|
| B1 | int4 | 2048×4096×4096 | 128 | `64×64 / 32×32 / s4` | **305 TF/s = 61%** | 2026-07-22 | `ppu-cutlass-w4a16-actlize` |
| B2 | int4 | 2048×4096×4096 | 128 | default `32×32` tile | **25%** | 2026-07-22 | same — **the control** |
| B3 | int4 | 2048×4096×4096 | 128 | official finegrained, `64×64×128 / s3` | 56.6% | 2026-07-22 | same |
| B4 | int4 | 2048×4096×4096 | 128 | hand-written Marlin | 215 TF/s = 43% | 2026-07-22 | same |
| B5 | int4 | 4096³, dense L=1 | 128 | — | **62%** | 2026-07-22 | same |
| B6 | — | dequant→fp16 + dense cuBLAS GEMM | — | not fused | **59–66%** | — | `ppu-w4a16-path-by-m`, `ppu-aiu-int4-viable` |
| B7 | int4 | fused mixed-input, same shape | 32 | — | 40.6% | — | same — the 20–26 point fusion cost |

**B2 is the control that matters most.** If a re-run's `32x32` rows do not read about 25%, the thing being
measured now is not the thing measured then, and every other comparison is against the wrong baseline.

## C. MoE — grouped, ragged vs uniform
<!-- route: grouped_lowbit -->


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
<!-- route: gemv_lowbit -->


Bandwidth-referenced, not MFU. Shape is the decode band; see `ppu-gemv-alu-bound-not-bandwidth`.

**WIDTH IS A COLUMN AGAIN FOR D4-D9.** Those six rows used to put prose where the width goes, which made them
unback-testable: `ci/check_backtest_configs.py` cannot map a config to a table without knowing which format's
table to look in, and it reported them as UNPARSED rather than guessing. The width is DERIVED, not assumed:
D4's config names it (`i4 ...`), D5's does too (`int4 native ...`), and D6-D9 are all the same dense-bench
config (`16x16x256:16x16:s2`) measured when `lowbit_dense_configs.inc` was emitted at bits=4 and was the only
dense table there was -- per-format dense tables did not exist until 2026-08-07 (`c776d98`). The displaced
prose moved to a `note` column rather than being deleted.

| # | width | time | GB/s | %HBM | config | recorded | note |
|---|---|---|---|---|---|---|---|
| D1 | int4 | 16.05 µs | 1310.7 | **47.4%** | `tileK s32/t64 N4 C2` | 2026-08-03 |
| D2 | int2 | 15.86 µs | 797.8 | 28.8% | `native s16/t128 N8 C2` | 2026-08-03 |
| D3 | int1 | 15.88 µs | 532.4 | 19.2% | `native s32/t64 N8 C2` | 2026-08-03 |
| D4 | int4 | 20.74 µs | — | 37.5% | `i4 16x32:256 w16x16 s2` | — | grouped tensor-core GEMM at the same band |
| D5 | int4 | 22.27 µs | — | 34.1% | `int4 native s16/t128 CtaN2 Chunk2` | — | best GEMV before the 2026-08-03 retune |
| D6 | **int4** | **17.98 µs** | 1168 | **42.2%** | `16x16x256:16x16:s2` | 2026-08-06 | tensor-core at M=1 once TileK entered the search |
| D7 | int4 | 26.09 µs | 805 | 29.1% | `16x16x256:16x16:s2` | 2026-08-06 | compact-A capacity 1 — CONFIRMED, and the cause is known |
| D8 | int4 | 16.97 µs | 1237 | 44.7% | `16x16x256:16x16:s2`, `[A path] ordinary AIU + swzl` | 2026-08-07 | ordinary A, the A/B baseline for D9 |
| D9 | **int4** | 16.49 µs | 1273 | 46.0% | same, `[A path] PACKED cubes` | 2026-08-07 | PPU_A_PACK: +2.9% and numerically Passed at M=1 |

**D7 WAS RIGHT ALL ALONG; ONLY ITS EVIDENCE WAS MISSING.** It was filed as "compact A at capacity 1 is 45%
slower" with nothing in the output naming the capacity, and was marked UNATTRIBUTED for a day. Reproduced
2026-08-07 with the capacity printed on the result line (`a0`/`a1`): **26.31 µs / 28.9%** against **16.98 / 44.7%**
— within 1% of D7. The chain of guesses in between was wrong at every link: I argued the run might have measured
PPU_A_PACK, or nothing at all, because PPU_A_CPASYNC is inert against a table that passes an explicit ACR. That
last fact is true and the conclusion drawn from it was not — the table's ACR field WAS the live entry point, and
it existed on the day D7 ran.

**AND THE MECHANISM IS NOW KNOWN, which is what the row could never say.** `SmemLayoutACompact` was hand-built
with `make_layout` and never composed `InternalSmemLayoutAtomA`, so the compact A tile carried no swizzle:

    capacity 0   (_16,(_64,_4),_2):(_64,(_1,_1024),_4096)     K mode is the swizzle atom tiled
    capacity 1   ((_1,_16),_256,_2):((_256,_0),_1,_256)       K mode is flat; the atom is gone

A therefore left the AIU/`tsm.ld.swzl` hardware path for plain addressed loads. acu, same tile, same grid
(1,512,1) both ways: `tsm.ld` +640%, `v.shll.i` 176×, `v.cnvt` 241×, `v.or.i` 0 → 327,680, while `tsm.ld.swzl`
and `vmem.aiu.ld.tsm` each fell 80% and the MMA count was **bit-identical** (163,840). Registers 94 → 132 are a
symptom of the address arithmetic, not a cause. And the swizzle could not have been kept: `tile_to_shape` refuses
to tile an 8-row atom onto a 1-row tile. Compact A was deleted on 2026-08-07 (task #42).

**D8/D9 ARE THE SAME IDEA DONE ON THE OTHER SIDE.** PPU_A_PACK leaves the swzl read untouched and overlaps the
cube allocations instead, so only row 0 is ever written: allocation −5.6×, A's gmem→smem traffic −16× at
TileM=16, and the read instruction unchanged. That is worth +2.9% and `Disposition: Passed`, against compact A's
−55%. One idea, two implementations, and the difference between them is entirely whether A stays on the delivery
hardware.

⚠ **D8/D9 ARE AN INTERNAL A/B AND NOT COMPARABLE TO D6's ABSOLUTE NUMBER.** Two builds from one commit, one
command, both witnessing their A path. D6 read 17.98 µs / 42.2% for the same config and D8 reads 16.97 / 44.7%;
what moved between them is not established here and must not be attributed to the compact-A deletion without a
measurement.

⚠ **ONE SHAPE, ONE CONFIG.** A's share of staged bytes is TileM·TileK·2 against TileN·TileK·bits/8 — 4× B at
TileN=16, a third of B at TileN=128 — so the gain should be LARGEST at small TileN. That is falsifiable and the
66-fixture sweep answers it. Do not generalise +2.9% before it does.

**D6 IS WHAT THE TileK SPLIT WAS FOR, and it is NOT a like-for-like beat of D4.** Different shape: D6 is
Qwen3-32B's q projection (M=1, N=8192, K=5120) while D4 is this section's decode band, so the comparable figure
is the normalised one -- 42.2% against 37.5% -- and not the microseconds. What it does establish is that the
M=1 tensor-core path was search-space-limited rather than kernel-limited: with TileK pinned at 64 the best
reachable row sat near 20%, D4's `16x32:256` was unreachable by construction, and the row that actually wins is
neither of those but `16x16x256` -- TileN=16, which no previously recorded number names.

⚠ **ONE MANUAL RUN OF ONE CONFIG, not a search verdict.** `benchmarks/sweep_real_shapes.py` over the 66 dense
fixtures is what decides whether 16x16x256:16x16:s2 is the optimum at this shape or merely better than the two
configs anyone had tried. Until that lands, this row is evidence that the ceiling moved, not that it has been
found.

**D1–D3 within 1.1% of each other across a 2.5× byte range** is the finding: decode is ALU-bound, not
bandwidth-bound, so int4 is effectively free relative to int1.

## E. fully_quantized — the SHIPPING path, and the only data that exists
<!-- route: dense_fully_quantized, grouped_fully_quantized -->


| # | what | figure | conditions | source |
|---|---|---|---|---|
| E1 | packed/native scale tax vs `scale_first` | **+13.1%** (95% CI 1.111–1.150) | 21.52 vs 22.54 µs — a **small-M / decode-band** shape | `ppu-q4k-native-perf-bc` |
| E2 | of which transfer + store + barrier | +9.2% (1.076–1.109) | same | same |
| E3 | of which the arithmetic itself | +3.5% (1.025–1.045) | same | same |

That is one paired A/B (20 interleaved ABBA blocks, paired log-ratio, six intervals all excluding 1.0) run by
`tests/test_q4k_native_scale.cu`. **It is the only performance number the fully-quantized path has, and it is
relative, not absolute.**

**IT CANNOT BE EXTRAPOLATED TO PREFILL, and the record says so in its own words:**

> ⚠ 不能跨 M 外推:prepass 字节数与 M 无关,pack 的解码按 `Σ ceil(M_e/TileM)` 重复。M ≤ TileM 时 pack 无冗余;
> M=2048/TM=128 时重复 16 次。

The pre-pass cost is independent of M; the packed decode is repeated once per m-tile. At M ≤ TileM the packed
path has no redundancy at all, and at M=2048 with TileM=128 it repeats **16×**. So the mechanism predicts the
tax GROWS with M, and +13.1% is its value where the repetition factor is smallest.

**WHAT DOES NOT EXIST:** any fully-quantized figure at prefill M, on either operator. `tests/test_q4k_packed_gemm.cu`
exercises the path but has no timing; the `.so`'s `*_fully_quantized` entries are consumed only by
`tests/test_gguf_routes.py`, which is a correctness test. So there is no bench to run — building one is work,
not a command.

**WHY THIS MATTERS MORE THAN THE SECTION-A TARGETS.** Section A tells us the collective and the harness are
healthy. The shipping path is a different consumer of that collective with a different TileK and a decode whose
cost scales with the m-tile count. A 65% scale_first result is compatible with a fully-quantized result
anywhere from 57% to much worse, and nothing measured has narrowed that.

---

## The order to back-test
<!-- route: none -- prose about which rows to reproduce; it quotes no figure of its own -->


One at a time, each against its own row, because a failure means different things at different rows.

1. **A1** — int4 gs=32. The headline. Checks the sweep grid contains `w64x32` and the collective still performs.
2. **B2** — the `32x32` control, read out of the same A1 sweep at gs=128. Costs nothing extra and validates the
   baseline before anything else is believed.
3. **dense ≤ grouped(L=1)** — the invariant (`analyse.py --invariant`). Needs no historical figure at all.
4. **A7 / gs=16** — refreshes on current HEAD the historical post-`w64x32` 234.16 µs and 228.13→227.35 µs
   measurements; it no longer settles a measurement-versus-recollection disagreement.
5. **B1** — gs=128, a different sweep's number, to check the two sweeps still agree with each other.
6. **C1** — the MoE ragged figure, whose harness was already back-tested to within 1.9% on 2026-08-04.

## What would invalidate a comparison
<!-- route: none -- prose about comparison hygiene; it quotes no figure of its own -->


* **A different `w64x32` presence.** The single largest recorded config effect (+8.6 points int4, +7.2 int2).
* **A different TileK.** `scale_first` is 256/bits — 64 for int4 — and the bench's default matches. A run with
  `TSK=` set is a different row and not comparable to A1.
* **`PPU_B_CHUNK`.** A2/A3 require it; A1 does not (the fold collective gates int4 out of chunking entirely).
* **useful vs issued** on any MoE row.
* **A stale config table.** The `.inc` carries `(bits, TileK)` and the binary `static_assert`s it, so this one
  cannot fail silently — it fails to compile.
