# GGUF k-quant pipeline — handover

What exists, what is verified and by what, what is not, and the traps that cost a round trip each.

---

## 1. The execution routes

A GGUF k-quant checkpoint can reach the hardware four ways. They differ in **what gets materialised**, and that is
the whole basis for choosing between them.

| route | scheme | materialises | extra DRAM per expert (N=K=2048, gs=32, Q4_K) |
|---|---|---|---|
| **fallback** | `DEQUANT_FIRST` | the whole weight as fp16, then cuBLAS / DeepGemm | write 8.39 + read 8.39 MB, and the GEMM then reads 8.39 instead of 2.36 |
| **pre-pass** | `SCALE_FIRST` | the fp16 scale/zero planes, in a workspace | write 0.52 + read 0.52 MB |
| **packed** | `FULLY_QUANTIZED` | nothing; the collective decodes the scale in-kernel | none, plus an UNDETERMINED tax (see §3) |
| **native GEMV** | `FULLY_QUANTIZED` | nothing; the scale pair is consumed in registers | none |
| **resident GEMV** | `SCALE_FIRST` | packed code planes plus resident fp16 scale/zero planes | format-dependent |

The fallback's extra traffic is **16×** the pre-pass's and its GEMM reads **3.6×** more bytes, so the crossover is in
**M**, not in implementation quality: at decode it is absurd, in the middle band the pre-pass's 1.05 MB constant
trades against the packed path's rate, and only at large M does reuse amortise the dequantised weight far enough for
cuBLAS-grade efficiency to pay for the bytes. `formats.select_path` already takes `num_rows`; what it lacks is
measurement to set the thresholds with.

All four CUDA-core GEMV launches are callable from Python now (`native/scale-first × dense/MoE`). The production
raw-pointer `.so` is tested against official gguf for all five k-quants, with a separate planted fault for each launch.
See `docs/DECODE_GEMV_RESULTS.md` for the 20-case oracle and both required benchmark shapes.

The fallback sat at "a piece exists, the path does not" on a note saying the remainder was *host-side wiring*. The
note was accurate and it functioned as a blocker anyway: the remainder was `a @ w.T`, and the cost of not writing it
was that **no two routes had ever produced the same number in the same process**.

The resident scale-first artifact costs +9…52% depending on format. That remains a product/storage choice, not a
technical support claim: the scale-first dense and MoE GEMV cells are now validated, while native GEMV remains the
byte-minimal decode route.

---

## 2. What is verified, and by what

**The oracle is the official `gguf` Python package**, not our own parser. Every k-quant constant in
`gguf_scale_layout.hpp` was read off `tools/dump_real_weights.py`, so comparing the C++ against that script compares
two transcriptions of one belief. `gguf.quants.dequantize` is an independent implementation of the same spec.

`gguf.quants` exposes **only `dequantize()` — there is no scale accessor**, so the scale is *inverted out* of the
reference rather than compared to it:

1. set `dmin = 0`, killing the affine term, so `w = code × scale`
2. fill the code bytes `0x00` and `0xFF` — the only byte values uniform in every bit, so every element of a group
   takes the same code — giving two points
3. `scale = (w_hi − w_lo) / qmax`, where `qmax` is only the bit width
4. `c_lo = w_lo / our_scale` is asserted **integral**, which *discovers* the code offset instead of assuming it

Derived, never written down: `Q2 [0,3]`, `Q3 [-4,3]`, `Q4 [0,15]`, `Q5 [0,31]`, `Q6 [-32,31]` — all matching ggml.

⚠ The integer assertion is **offset discovery, not proof of scale**: for Q2/Q4/Q5 `c_lo = 0` and is integral for any
non-zero scale. The discriminator is the relative comparison that precedes it.

### Coverage today (`tests/test_gguf_golden.py`)

| check | covers | tolerance and why |
|---|---|---|
| scale/zero vs official | all five | 4.9e-4 = 2⁻¹¹, the fp16 output's floor |
| **vecdot as a DOT PRODUCT, CPU** | all five | 2e-5. Per-group tests cannot see element ORDER; a dot product can |
| **CUDA fp16/half2 vecdot** | all five | conditioned error `< 2^-11` against official fp64 weights; observed 1.05e-4..2.02e-4 |
| fp16 dequantise, elementwise | all five | 1e-3, fp16 rounding |
| codes/scale/zero split reconstructs | all five | plus the code range asserted, not printed |
| packed unit round trip | all five | **bit-exact** — same integers, same header, so any difference is a lost bit |
| four routes agree | all five | each route's own floor |
| real GGUF scale bytes | Q4_K | anchors the RECORD, which random bytes cannot |
| whole offline artifact | Q4_K | crosses `torch.save`/`load`; the consuming arm sees neither the raw blocks nor the codes |

---

## 3. Kernels

**Correctness is at 5/5 on every check; the native decode kernel is now timed locally.** PPU performance remains
unmeasured, so the 5090 results are directional rather than a claim about the box.

| kernel | hw | state | measured |
|---|---|---|---|
| `dequantize_kernel_warp` | 5090 | tuned | 779.4 → **58.4 µs**, 13.4×, 1.473 TB/s = **82.2% of peak**, bit-identical |
| `dequantize_kernel_warp_logical` | 5090 | kept slow on purpose | 196.6 vs 65.5 µs on a nontrivial reorder — the general path for layouts `right_inverse` cannot take |
| `prepass_kernel` (cooperative) | 5090 | tuned | speedup grows with size — 1.32× / 1.91× / 2.95× — the fixed-duration floor made visible |
| `vecdot_rows_kernel` | 5090 | **1659–3361 Gelem/s cold, 55–76% of peak AT A SHAPE 64× LARGER THAN A REAL LAYER** | rows=131072; runtime policy also covers N=K=2048 — see §3a |
| packed collective | PPU | **+12.8% native-format tax in paired blocks** | 95% CI 1.118..1.139× base; see below |
| `gemv_lowbit` | PPU | tuned earlier | 22.27 µs, and **ALU-bound, not bandwidth-bound** |

**The packed path's tax is determined only by paired blocks, not cross-run medians.** The same pinned configuration
drifted 17% between sessions, so old cross-run subtraction remains void. A ten-block interleaved run forms ratios
within each block: packed is +12.8% with a 1.118..1.139 95% interval, while the int4→fp16 dequant NOP reproduces at
−10.9% against the earlier −11.1%. `packfuse` removed 26.8% of `tsm.st`, 48.8% of `tsm.ld`, and 48.3% of conflicts
yet cost +0.46%; it exchanged shared work for ALU on a kernel at Compute 38.99% / Memory 29.87%.

The retained `vecdot_rows_kernel_serial` baseline has one output row per thread, so 32 lanes read addresses
`blocks_per_row × block_bytes` apart — 1152 B for Q4_K. The tuned kernel gives every lane contiguous packed words:
whole groups for Q2/Q3/Q6 and adjacent group pairs for Q4/Q5. It consumes fp16 activation, converts four packed
codes at once with actlize's shared converter, and keeps the cooperative products, sums, row accumulator and final
butterfly in half2. Output widens to fp32 only after that reduction. The fixed correctness contract is conditioned
error `< 2^-11` against official fp64 weights, not error divided by `abs(dot)`, which explodes under cancellation.

**THE OLD INSTRUMENT COULD NOT RESOLVE THE OPTIMISATIONS.** `cudaEventElapsedTime` on this 5090 advances in 2.048 µs
ticks. At 16384 rows the kernel took only 20–26 ticks, so the former one/two-tick rankings were below a 4–5% floor.
Cold launches cannot be batched: only the first launch after an L2 flush is cold. Warm launches are batched under
one event pair and divided by the batch size, which resolves sub-tick differences without pretending they are cold.
The benchmark now treats rows<=0 as rows=131072, and every large-shape A/B below uses 131072 rows × 8 blocks.

### 3b. The SIMT-vs-collective comparison CANNOT be settled on a 5090, and the reason is L2

A real SIMT MoE GEMV was built and measured: `gemv_lowbit`'s grouped arm — experts on the grid's z dimension,
ragged rows via `row_offsets`, real gather — through the REAL CMake unit generator (20 int4 units, no PPU SDK
needed; `dev/fold_derivation/gen_gemv_units_check.sh` drives the generator slice with `cmake -P`).

| shape | best W4 config | median | compulsory bytes | ÷ 1.792 TB/s |
|---|---|---:|---:|---:|
| MoE `L=8`, one row/expert, `N=K=2048`, gs=32 | `int4 tileK s8/t256 N8 C2` | **7.64 µs** | 21.04 MB | **153.7%** |
| dense `m=1`, `N=K=2048`, gs=32 | `int4 tileK s8/t256 N4 C2` | **2.60 µs** | 2.63 MB | 56.4% |

**153.7% of DRAM peak is not a result, it is a diagnosis.** The 21 MB operand fits the 5090's L2, so the
number is L2-served at an effective 2.754 TB/s. On PPU the same operand does not fit. The collective's 29.87%
Memory SoL is an HBM number; this is a cache number; **they have no common denominator**, and the gap between
them is not a performance difference at all. A cache-flushed counter run was attempted and `ERR_NVGPUCTRPERM`
blocked it — this container has no profiling-counter permission.

The grid mechanism is real and measured: `1 × 256 × 8 = 2048` CTAs, exactly **16×** the collective's 128,
because the tilings differ for the same problem. And PPU has already run this comparison at parity of memory
level: **grouped GEMM 20.74 µs / 37.5% HBM against `gemv_lowbit` 22.27 µs / 34.1%** — the SIMT GEMV had 16×
the CTA supply and was still **7.4% slower**. The 5090 does not reproduce that ordering, and cannot settle
it: the result is L2-served, no 5090 collective was measured against it, and this project has a recorded case
of configuration rankings inverting between the two machines at gs=32.

This earlier result is strictly `SCALE_FIRST`. The native GGUF MoE launch now exists separately and is measured with
the same rows/expert convention in `docs/DECODE_GEMV_RESULTS.md`.

Two things that did transfer, because they are properties of the binary rather than of the machine: the SASS
contains **zero tensor-op mnemonics and 287,278 half2 arithmetic instructions**, so "CUDA cores only" is
proven rather than asserted, and so is the fp16 accumulation reaching packed instructions instead of being
scalarised.

### 3a. Every GEMV number above is a LARGE-GRID number, and the shipping shape is not

`rows` is the OUTPUT dimension N. At `bpr=8` the tuning shape is N=131072, K=2048 — **sixty-four times a real
dense layer at decode**, which is N=K=2048, i.e. `rows=2048`. Q4_K with the runtime rows/bpr policy, sweeping only
size after the fp16 rewrite:

| rows | rpw | CTAs | CTA/SM | cold µs | Gelem/s | % of peak |
|---:|---:|---:|---:|---:|---:|---:|
| **2048** | **2** | **128** | **0.8** | 14.30 | 293 | **9.3%** |
| 8192 | 8 | 128 | 0.8 | 16.35 | 1026 | 32.3% |
| 16384 | 8 | 256 | 1.5 | 20.48 | 1638 | 51.6% |
| 32768 | 8 | 512 | 3.0 | 36.90 | 1819 | 57.3% |
| 65536 | 8 | 1024 | 6.0 | 71.71 | 1872 | 59.0% |
| 131072 | 4 | 4096 | 24.1 | 129.02 | 2081 | 65.5% |
| 262144 | 4 | 8192 | 48.2 | 245.76 | 2185 | 68.8% |

The dispatcher deliberately keeps both 2048 and 8192 rows at **128** CTAs by moving Q4 from rpw2 to rpw8. The
probe launches 256 threads = eight warps, so a CTA owns `8 x rpw` rows: 2048/16 and 8192/64 are both 128. Four
times the work costs only 1.14x. This is a launch/latency regime, not a throughput regime. The instrument forced the
large shape for throughput A/Bs; at 2048 rows cold is seven 2.048-us ticks, so differences below ~14% are unresolved.
Warm measurement batches up to 64 launches and can resolve the launch-policy crossover, while cold retains exactly
one post-flush launch.

**The per-format rows-per-warp defect is fixed structurally.** `vecdot_rows_per_warp<T>(rows, blocks_per_row)` now
dispatches the existing kernel instantiations instead of replacing one large-N constant with one small-N constant.
Warm-batched size sweeps plus bpr=1/4/8/16 sensitivity set the boundaries: Q2/Q6 use 2; Q3 uses 2 below 65536 rows
and 4 above when bpr<=8; Q4 uses 2 at rows<=2048, 8 in the middle, and 4 once `rows*bpr >= 786432`; Q5 uses 2 only
at rows<=2048 with bpr>=8 and otherwise 8. Fixed-rpw entry points remain as measurement witnesses.

Both the saturated defaults (`rows=131072`, bpr=8) and the runtime boundaries are now written beside the policy in
the header. The omitted tuning shape was how a large-grid optimisation became a shipping-shape regression.

**This also reframes the comparison against the PPU collective — grid and work must both be matched.** The pinned
collective launches 128 CTAs (Waves Per CU 0.44 = 128 over 72 CU × 4 theoretical block slots), from eight active
experts × `ceil(2048/128)=16` N tiles. Three SIMT points separate the confounders:

| Q4_K SIMT point | useful MoE-sized work | CTAs | cold % peak | warm % peak | what it matches |
|---|---:|---:|---:|---:|---|
| rows=2048, policy rpw2 | 1 × 2048 outputs | 128 | 9.3% | 17.0% | grid only; one eighth of the work |
| rows=16384, policy rpw8 | 8 × 2048 outputs | 256 | 51.6% | 69.9% | work only; twice the CTAs |
| rows=16384, fixed rpw16 | 8 × 2048 outputs | 128 | **36.9%** | 60.3% | work and grid |

The last row is the defensible comparison against collective Memory 29.87% / Compute 38.99%. Grid supply therefore
explains a large part of the apparent saturated-SIMT advantage (65.5% at rows=131072 falls to 36.9%), but not all of
it: seven cold bandwidth-percentage points remain, and the CTAs still have different delivery, dependency and
barrier chains. Warm SIMT is an L2-resident counterfactual at this size, not a PPU HBM comparison.

The bound described here is historical. `vecdot_rows_kernel<Grouped=true>` now adds grid.z expert bases and ragged
`row_offsets`; its direct native-MoE results supersede this dense extrapolation.

The collective's 0.44 is set first by **tile count**: at the pinned `16x128:256` shape, eight active experts times
`ceil(2048/128)=16` N tiles is 128 CTAs. acu's four-block/CU capacity is shared-memory-limited (57,344 B per CTA in
256 KB); registers admit that point, so neither register cutting nor smaller scale storage creates more work. TileN
can move the grid — TN64 gives 256 CTAs and TN32 gives 512 — but that ladder was measured and TN128, the row with the
fewest blocks and eight warps sharing each loaded B tile, won at 20.11 µs. Thus capacity is movable but is not the
measured lever in this band. More stages, the scale-reload chain, or a genuine persistent schedule address the
per-CTA dependency/fill cost; blindly adding CTAs repeats a losing experiment.

What transfers from SIMT is narrower than “reuse its decoder.” Packed-word plane merge, the shared byte4→half
converter, CUTE descriptions of the physical planes, and the conditioned fp16 error gate are reusable. Whole-group
lane ownership is not: the AIU collective requires its fixed interleaved register fragment and `tsm.ld.swzl`
delivery order. Feeding native GGUF words directly would need an explicit gather/shuffle into that fragment (or
would abandon the AIU path), so the offline xplane/fragment relayout remains required unless that transpose is
measured cheaper. CUTE can derive the required relayout; it cannot make the two ownership maps identical.

| | rpw | cold µs | cold Gelem/s | cold % peak | warm µs | warm Gelem/s | warm % peak | cold/warm | vs `f18ec9e` cold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q2_K | 2 | 79.872 | **3360.8** | 61.91% | 58.752 | **4569.0** | 84.16% | 1.359 | 1.513× |
| Q3_K | 4 | 116.736 | **2299.5** | 55.39% | 110.048 | **2439.3** | 58.76% | 1.061 | 1.491× |
| Q4_K | 4 | 129.024 | **2080.5** | 65.53% | 125.856 | **2132.9** | 67.18% | 1.025 | 1.714× |
| Q5_K | 8 | 143.360 | **1872.5** | 72.04% | 142.688 | **1881.3** | 72.38% | 1.005 | 1.843× |
| Q6_K | 2 | 161.792 | **1659.1** | 76.13% | 160.544 | **1672.0** | 76.72% | 1.008 | 1.342× |

Element rate, not byte-density-derived `% peak`, selected the implementations. Q2 remains the fastest element
kernel even though Q4/Q5 now report higher `% peak` because each of their elements carries more raw bytes.
These are rows=131072, bpr=8, cold single-launch and warm-batched medians over 50 samples. Q3..Q6 operands exceed
L2, so their near-one large-shape ratios mean both sides are DRAM-fed, not that memory is free. Q2's 88.6 MB
footprint fits close enough to this GPU's L2 to show a real warm uplift. At rows=2048 every format fits and the
meaningful cold/warm ratios are 1.75..2.24.

**Q4/Q5 paired byte map.** Four owner lanes consume both nibbles of one complete 32-byte run:

| lane | groups | low-plane bytes | use |
|---:|---:|---|---|
| 0 | 0, 1 | `qs[0..31]` | low nibble → g0, high nibble → g1 |
| 1 | 2, 3 | `qs[32..63]` | low nibble → g2, high nibble → g3 |
| 2 | 4, 5 | `qs[64..95]` | low nibble → g4, high nibble → g5 |
| 3 | 6, 7 | `qs[96..127]` | low nibble → g6, high nibble → g7 |

At rpw8 those are the row's four lanes. At rpw4 the row subgroup has eight lanes and lanes 4..7 are inactive for
that row; the four owners and byte map do not change. Q5 loads each matching `qh[0..31]` word once and selects the
two group bits from it. On the fp32 predecessor, pairing alone moved Q4 from 221.184 to 190.464 µs cold (1.161×);
Q5 was within two event ticks of its slice optimum and slightly worse warm. The composition with four-code word
extraction was decisive for both, and the later half2 rewrite reaches 129.024/143.360 µs cold.

**Four-code extraction.** Q2/Q4/Q5 block strides (84/144/176 B) preserve four-byte alignment. Q3/Q6 strides
(110/210 B) make every odd block only two-byte aligned, so the loader checks the address: one `uint32_t` load when
four-byte aligned, two `uint16_t` loads when only two-byte aligned, and byte assembly as the general fallback. Each
word is masked/merged while four codes remain byte-packed; the group loop retains an explicit scalar tail.

The code planes are described once as CUTE fields. `PackedField4` maps `(group, four-element word)` to its physical
word coordinate; each `CodeTraits<T>` names independent low/high field layouts and offsets. Static assertions prove
that every bit of each exact native plane is claimed once, including Q3/Q6's different high-plane shapes.
Both scalar `code_at_group` and cooperative word extraction consume those traits, so a high plane cannot silently
inherit a low plane's indexing rule.

A benchmark-only padded device stride (Q3 110→112 B, Q6 210→212 B) was also measured three times in alternating
order. It regressed Q3 from 1820–1846 to 1771–1795 Gelem/s, but improved Q6 from 1285 to 1473 Gelem/s cold. It is not
the shipped result: `ppu_backend::Api::vecdot` currently promises compact raw GGUF blocks and no repack workspace
exists behind the Python seam. Two neighboring aligned 32-bit loads on compact odd blocks were also neutral. Padding
Q6 is a worthwhile follow-up only together with an explicit device-layout/repack contract; the table keeps raw bytes.

The first `__byte_perm`/`ppu.prmt.b32` magic-u8-to-f32 variant was tested on the fp32 kernel and rejected:
Q2/Q3/Q5 cold times were unchanged, Q4 cost one 2.048 µs event tick, and Q6 warm regressed. Once products moved to
half2, the shared actlize operation became the right abstraction. `MixGemmByte4ToHalf<Bias,Interleave>` owns the
selectors, exponent constant and value bias; native GGUF requests logical `[0,1,2,3]` order, while the historical
int8 GEMM specialization preserves `[0,2,1,3]`. CUDA uses guarded `__byte_perm`; the new PPU arm is guarded by the
device-pass `__HGGC_ARCH__`, with a runtime box probe for the selected arm. No private converter remains.

**The public activation contract is fp16 on device and fp16/fp32 at the Python seam.** `VecdotActivation` defaults
to `half_t`; `GGUF_VECDOT_FP32_ACTIVATION=1` is a measurement-only build. `gguf_vecdot` accepts either dtype so the
CPU arm remains a tight fp32 oracle. A loaded PPU backend receives fp16, and `matmul_native_gemv` converts its one
K-vector before fan-out. To avoid charging fp32 for an rpw policy tuned for fp16, the price comparison sweeps
rpw1/2/4/8 independently for both dtypes and compares each one's best cold point at rows=131072, bpr=8:

| | fp32 x, best cold µs (rpw) | native fp16 x, best cold µs (rpw) | speedup | speedup at rows=2048 |
|---|---:|---:|---:|---:|
| Q2_K | 102.400 (4) | 79.872 (2) | 1.282× | 1.204× |
| Q3_K | 149.504 (4) | 116.736 (4) | 1.281× | 1.145× |
| Q4_K | 137.216 (8) | 129.024 (4) | 1.063× | 1.430× |
| Q5_K | 157.696 (8) | 143.360 (8) | 1.100× | 1.113× |
| Q6_K | 212.992 (2) | 161.792 (2) | 1.316× | 1.286× |

The 2048-row cold results are quantised, but the independently batched warm sweep agrees on the direction. The
device ABI is therefore fp16 by measurement, not merely because the accumulator happens to be fp16.

The direct CUDA golden dequantises official GGUF weights to fp64, multiplies the actual fp16 activation, and divides
absolute dot error by `sum(abs(w*x))`. Observed conditioned errors are Q2 1.30e-4, Q3 1.65e-4, Q4 1.55e-4,
Q5 2.02e-4 and Q6 1.05e-4, all below the fixed fp16 half-ULP floor `2^-11 = 4.88e-4`. Error divided by `abs(dot)`
reaches 0.121 on the same fixture because centred codes cancel; that is conditioning, not twelve-percent decode
error. The scalar CPU witness remains separate and tight rather than impersonating the arithmetic that ships.

The pre-extraction `f18ec9e` NOP-style probes kept every lane live and priced the terms that motivated this round.
`code_at` returns constant one; the scale
probe uses the block base as unit scale/min. Negative scale shares mean the counterfactual was slower, not that the
real unpack has negative cost. Probe `% peak` retains the production kernel's nominal byte count for normalization;
the code NOP deliberately removes raw-code traffic, so an effective figure above 100% is not a physical HBM claim.

| | code-NOP cold µs (% peak), share | code-NOP warm µs (% peak), share | scale-NOP cold µs (% peak), share | scale-NOP warm µs (% peak), share |
|---|---|---|---|---|
| Q2_K | 86.016 (57.49%), **28.8%** | 83.296 (59.37%), **27.7%** | 120.832 (40.92%), 0.0% | 116.000 (42.63%), -0.7% |
| Q3_K | 81.920 (78.93%), **52.9%** | 82.304 (78.57%), **51.3%** | 176.128 (36.71%), -1.2% | 171.040 (37.81%), -1.2% |
| Q4_K | 106.496 (79.40%), **51.9%** | 103.456 (81.73%), **52.4%** | 219.136 (38.59%), 0.9% | 217.088 (38.95%), 0.0% |
| Q5_K | 106.496 (96.98%), **59.7%** | 103.424 (99.86%), **59.7%** | 231.456 (44.62%), **12.4%** | 230.464 (44.81%), **10.3%** |
| Q6_K | 83.968 (146.70%), **61.3%** | 82.976 (148.45%), **61.0%** | 217.088 (56.74%), 0.0% | 212.992 (57.83%), 0.0% |

Those probes identified packed-code extraction as the target; the production table above is the resulting rewrite.
The NOP builds were repeated after the half2 rewrite to ensure the new packed pairs did not let dead lanes disappear;
each owner still consumes activation, the other metadata plane, both half accumulators and the final butterfly:

| | full cold/warm µs | code NOP cold/warm µs (cold share) | scale NOP cold/warm µs (cold share) |
|---|---:|---:|---:|
| Q2_K | 79.872 / 59.552 | 55.296 / 49.408 (30.8%) | 77.824 / 59.840 (2.6%) |
| Q3_K | 116.736 / 110.656 | 51.200 / 48.160 (56.1%) | 139.264 / 132.096 (-19.3%) |
| Q4_K | 127.008 / 126.912 | 104.448 / 103.840 (17.8%) | 120.832 / 120.576 (4.9%) |
| Q5_K | 143.360 / 142.752 | 71.680 / 69.792 (50.0%) | 145.408 / 146.432 (-1.4%) |
| Q6_K | 161.792 / 160.672 | 77.824 / 77.824 (51.9%) | 159.744 / 158.688 (1.3%) |

As before, a negative scale share means that counterfactual compiled slower; it is not a claim of negative work.
The old shares must not be read as a profile of the new kernel. Scale/min unpack was material only for Q5 in that
baseline. Large-shape cold/warm is not a compute-vs-memory classifier when the operand exceeds L2; the 2048-row
ratios are the cache-residency signal and show that memory still costs real time.

The regime was never reported, only the speedup, and a speedup measured against a one-row-per-thread baseline says
nothing about how much is left. Both halves are the same measurement run; only one of them is a target.

**Every number above is a 5090 number, and not one GGUF kernel has run on PPU.** The ordering does not transfer:
at gs=32 the 5090's ranking of GEMV configurations inverted against PPU's. Treat these as directional only.

**The two dequantiser kernels' loss was the store pattern**, and the GEMV's serial baseline had the same shape on the load
side. One thread per block puts a warp's lane addresses 512 bytes apart, so one instruction touches 32 separate
32-byte sectors for 64 useful bytes: **6.25% sector utilisation, exactly 16× worse** than lanes touching consecutive
elements. Partitioning fixes the sectors; native packed-code extraction remains the dominant GEMV term afterward.

The production source now builds `libquactlize_ppu.so` and the dlopen seam resolves its five required symbols. It is
compiled and run under nvcc in CI; `build.sh` registers the same source as an hgcc `cutlass_add_library` target.

---

## 4. The packed scale unit

GGUF's own packing is **not half-separable** — Q4_K's `get_scale_min_k4` takes groups 4..7 from bytes 8-11 *and* the
top two bits of bytes 0-3 — so a k-tile covering part of a superblock cannot read part of a block. The reordered
unit fixes that **at no cost in stored bytes**, and that byte-neutrality is the licence for the whole path.

Q4_K has always been reordered this way; what is new is that it is **named**, **generalised**, and **checked**.

| format | unit | vs GGUF's own scale metadata | active in the collective |
|---|---|---|---|
| Q4_K / Q5_K | `scu16x1` | 16 / 16 | ✅ |
| Q2_K | `scu20x1` | 20 / 20 | ✅ |
| Q3_K | `scu28x2` | 28 / 28 | ❌ staging |
| Q6_K | `scu36x2` | 36 / 36 | ❌ staging |

`scu<bytes>x<superblocks>` is in the layout vocabulary, so two arrangements of one format have different names — and
`scu14x1`'s own description says it **cannot be bulk-copied**, because 14 is 2 mod 4 and `ppu.cp.async` takes only
4, 8 or 16 bytes.

**The axis matters.** Pairing two *columns* needs the staged tensor recast to `(TN/2, 2·unit)` and changes which
column a thread owns — withdrawn. Pairing two *superblocks of the same column* needs neither: 28 and 36 bytes, no
padding, and a thread still owns exactly its own column. Each superblock keeps its own header, so a consumer wanting
one reads a contiguous run.

---

## 5. Open, with the blocking reason

1. **Q3_K / Q6_K in the collective** — the format, pack and decode are done and bit-exact; the staging still assumes
   one superblock per unit, so `scu28x2`/`scu36x2` need the tile and stage cadence to cover two k-tiles per copy.
2. **Device correctness for Q2_K, Q3_K, Q5_K, Q6_K** — these are *front-end instantiations*. The staging, the
   partial-word register assembly and **Q2_K's `ZMul = 0`** have never run. Q2_K is the risk: it is newly active, and
   `test_q4k_packed_gemm` cannot catch it because that fixture is Q4_K.
3. **`packfuse` unresolved** — acu reported Shared Store bank conflicts unchanged at 81,920. Every previous box run
   used an actlize that **did not contain the code**; the gitlink pointed at a commit without it. The decisive datum
   is the Shared Store **instruction** count, not the conflict count: a fall of ~36,864 means the word store was
   emitted and the bank model is wrong; no fall means two half stores or a stale binary. Timing cannot decide it.
4. **The unit is not yet an offline artifact** for the packed path — `test_q4k_packed_gemm.cu` builds it at load time
   with `put_code`. The whole-artifact test does cross a file boundary; the harness does not.
5. **The destination should be a cute Tensor/Layout, not a callable** — measured: partitioning physical output
   addresses and deriving logical source indices with `right_inverse` is **3× faster** than striping logical indices
   (65.5 vs 196.6 µs) on a non-affine layout.
6. **SCALE_FIRST × DENSE hides a mechanism** — `fpA_intB_ppu.cuh` hardcodes int4 and has no second-plane input.
   Q2_K needs an int2 dense instantiation; Q3_K/Q5_K/Q6_K need the separate plane joined in the dense converter.
   It remains PARTIAL rather than pretending the scale-first GEMV binding also filled a tensor-core launcher.

7. **The registry GEMV vocabulary is split now** — native/scale-first and dense/MoE have four distinct path names.
   The historical coarse `gemv` name remains for old harness reports, and `_CELL_PATH_IS_COARSE` remains as the
   refusal mechanism for any future neighbouring-kernel evidence share.

8. **`ci/registry.py`'s completeness sweep is `.cu` only.** Every CUDA harness must be declared; a Python one need
   not be. Turning it on requires declaring every existing `tests/*.py`, and a sweep that goes red the moment it is
   switched on gets switched off.

---

## 6. Traps, each of which cost a round trip

**A check that has only ever been observed passing is a check whose failure path is untested.** Two instances in one
night, symmetric: `ppu_portability_check.py` had a `NameError` in the branch that reports a violation, and a lint
returned `"ok"` where the runner wanted `"PASS"` — crashing only on the *passing* path. Both were fixed by planting
the fault, watching it report the right line, then removing it.

**39/39 local ≠ the box builds.** The syntax gate is nvcc's front end and catches only what *both* compilers reject.
hgcc is stricter: it fails on two unroll directives for one loop, which nvcc accepts. A lint covers that one
instance; the general gap has no local answer.

**A submodule needs its own push and a gitlink bump.** Every parent commit describing the fused work pointed at an
actlize where `kFusedScaleZero` appears zero times. Verified defines prove preprocessing; differing md5s prove the
files differ; a local type gate compiles *this machine's* tree. None is a statement about what the box received.

**`git add -A` while another agent works in the tree** sweeps its half-finished work into your commit. It is how
`tests/gguf_cuda_probe.cu` — a host-CUDA file that `hgcc` cannot compile — got committed and overlaid onto the box.

**A gate can compile dead code.** `kPackedScaleOn` needs `Scale_TileK == groups`, and the fixture only has 8 and 2,
so "all five formats compile" was three formats' decoders never being instantiated. `l103` asserts the path is
*active* and fails otherwise.

**Coalescing without partitioning is worth nothing.** The first warp dequantiser had every lane run the whole
traversal and keep its 32nd share: stores coalesced, loads and arithmetic replicated 32×, 1.2× instead of 13.4×.

**Time the kernel, not the bus.** The first timing harness wrapped `cudaMalloc` and 350 MB of transfers inside the
timed region and reported 0.95×.

**A bandwidth above DRAM peak is evidence a measurement leaks**, not evidence of speed — dirty output written back
during an untimed flush.

**Label recovery through the full layout chain is unreliable for sub-byte types**: `cvtword` permutes nibbles inside
a word, so `unpack_int4` does not restore element order. Per-step labelling is sound; whole-chain is not.

**An invariance argument requires that both changes actually touch the thing being measured.** "278,528 conflicts are
identical in base and pack, so they are not the scale read" is void: `pack` decodes *into the same fp16 planes* and
leaves the read side untouched, so their equality carries no information.

---

## 7. Where the reorder does and does not reach the scale planes

Measured, after two wrong answers in opposite directions:

- **k axis is safe.** The row permutation stays inside its own block — int4 is a true 32-element permutation with max
  displacement 18, int8 is 16 with displacement 6, and 100% of elements remain in their own 32-block. A plane indexed
  by `k//gs` at gs=32 cannot see it. *(The int4 number needs two label passes; 4-bit labels alias across a 32-row
  permutation and a single pass reports agreement it has not established.)*
- **n axis is safe for a different reason.** `mem_cacheline_col_tile_interleave` does not reorder n within n; it
  **folds** `interleave` adjacent columns into one and stacks them along the row axis. The kernel must recover the
  source n to write its output column, and having recovered it, indexes the scale in logical `(n, k//gs)` order.
- Corroboration: **nothing in this codebase preprocesses a scale tensor.** `preprocess_weights_to_layout` takes only
  the weight; `symmetric_quantize` returns its scales unprocessed.

**So the scale planes need no transformation for an offline-reordered weight.**
