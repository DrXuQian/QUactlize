# GGUF k-quant pipeline — handover

What exists, what is verified and by what, what is not, and the traps that cost a round trip each.

---

## 1. The four routes, and why there are four

A GGUF k-quant checkpoint can reach the hardware four ways. They differ in **what gets materialised**, and that is
the whole basis for choosing between them.

| route | scheme | materialises | extra DRAM per expert (N=K=2048, gs=32, Q4_K) |
|---|---|---|---|
| **fallback** | `DEQUANT_FIRST` | the whole weight as fp16, then cuBLAS / DeepGemm | write 8.39 + read 8.39 MB, and the GEMM then reads 8.39 instead of 2.36 |
| **pre-pass** | `SCALE_FIRST` | the fp16 scale/zero planes, in a workspace | write 0.52 + read 0.52 MB |
| **packed** | `FULLY_QUANTIZED` | nothing; the collective decodes the scale in-kernel | none, plus an UNDETERMINED tax (see §3) |
| **GEMV** | `FULLY_QUANTIZED` | nothing; the scale pair is consumed in registers | none |

The fallback's extra traffic is **16×** the pre-pass's and its GEMM reads **3.6×** more bytes, so the crossover is in
**M**, not in implementation quality: at decode it is absurd, in the middle band the pre-pass's 1.05 MB constant
trades against the packed path's rate, and only at large M does reuse amortise the dequantised weight far enough for
cuBLAS-grade efficiency to pay for the bytes. `formats.select_path` already takes `num_rows`; what it lacks is
measurement to set the thresholds with.

**Two of these are callable from Python now** (`quactlize/routes.py`), which is what makes that measurement possible
at all: `matmul_dequant_first` (dense and per-expert) and `matmul_native_gemv`. Both are checked against the official
`gguf` package for all five k-quants — worst relative error 1.05e-3 and 1.4e-7 respectively, each its own arithmetic's
floor — and the dense route is exercised on a CUDA tensor so its GEMM is the library rather than a CPU matmul.

The fallback sat at "a piece exists, the path does not" on a note saying the remainder was *host-side wiring*. The
note was accurate and it functioned as a blocker anyway: the remainder was `a @ w.T`, and the cost of not writing it
was that **no two routes had ever produced the same number in the same process**.

**The pre-pass removes the STORAGE objection, not the RESIDENCY one.** Materialising fp16 planes costs +9…52% stored
bytes depending on format, which the project forbids; a workspace does not count. But at decode the planes would be
rebuilt every token, so that band needs a native path — which is what the GEMV route is for.

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

### Coverage today (`tests/test_gguf_golden.py`, 44 tests)

| check | covers | tolerance and why |
|---|---|---|
| scale/zero vs official | all five | 4.9e-4 = 2⁻¹¹, the fp16 output's floor |
| **vecdot as a DOT PRODUCT** | all five | 2e-5. Per-group tests cannot see element ORDER; a dot product can |
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
| `vecdot_rows_kernel` | 5090 | **1285–2621 Gelem/s cold, at 44–64% of peak** | rows=131072; native bytes, CUDA cores only; see below |
| packed collective | PPU | **tax undetermined** | see below |
| `gemv_lowbit` | PPU | tuned earlier | 22.27 µs, and **ALU-bound, not bandwidth-bound** |

**The packed path's tax is not a number yet.** `CHECKPOINT.md` records the same pinned configuration measuring
`base` at 23.67 µs in one run and 20.11 µs in another — **17% apart, wider than `pack − base` itself**, which lands
at +2.4% against one baseline and +12.9% against the other. Cross-run comparison is therefore void and any single
figure quoted for it (this document said +11.6%) is a selection, not a measurement. The one term in that document
that does survive is `base − bdqnop = −11.1%`: the int4→fp16 dequant pipeline, 43% of dynamic instructions and 11%
of time, which is the ceiling for every dequant-side idea.

The retained `vecdot_rows_kernel_serial` baseline has one output row per thread, so 32 lanes read addresses
`blocks_per_row × block_bytes` apart — 1152 B for Q4_K. The tuned kernel gives every lane contiguous packed words:
whole groups for Q2/Q3/Q6 and adjacent group pairs for Q4/Q5. Reassociation tolerance is 2e-6, observed at 2.16e-7
across all five formats.

**THE OLD INSTRUMENT COULD NOT RESOLVE THE OPTIMISATIONS.** `cudaEventElapsedTime` on this 5090 advances in 2.048 µs
ticks. At 16384 rows the kernel took only 20–26 ticks, so the former one/two-tick rankings were below a 4–5% floor.
Cold launches cannot be batched: only the first launch after an L2 flush is cold. The benchmark now treats rows<=0
as rows=131072, and every A/B below uses 131072 rows × 8 blocks. Warm timings also use the larger problem.

| | rpw | cold µs | cold Gelem/s | cold % peak | warm µs | warm Gelem/s | warm % peak | cold/warm | vs `f18ec9e` cold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q2_K | 4 | 102.400 | **2621.4** | 48.29% | 91.360 | **2938.2** | 54.13% | 1.121 | 1.180× |
| Q3_K | 4 | 147.456 | **1820.4** | 43.85% | 135.360 | **1983.1** | 47.77% | 1.089 | 1.181× |
| Q4_K | 8 | 137.216 | **1956.3** | 61.62% | 134.688 | **1993.0** | 62.78% | 1.019 | 1.612× |
| Q5_K | 8 | 161.792 | **1659.1** | 63.84% | 157.792 | **1701.2** | 65.45% | 1.025 | 1.633× |
| Q6_K | 2 | 208.896 | **1285.0** | 58.97% | 203.680 | **1317.9** | 60.48% | 1.026 | 1.039× |

Element rate, not byte-density-derived `% peak`, selected the implementations. Q2 remains the fastest element
kernel even though Q4/Q5 now report higher `% peak` because each of their elements carries more raw bytes.

**Q4/Q5 paired byte map.** At `rpw=8`, four lanes own a row and each lane consumes both nibbles of one complete
32-byte run:

| lane | groups | low-plane bytes | use |
|---:|---:|---|---|
| 0 | 0, 1 | `qs[0..31]` | low nibble → g0, high nibble → g1 |
| 1 | 2, 3 | `qs[32..63]` | low nibble → g2, high nibble → g3 |
| 2 | 4, 5 | `qs[64..95]` | low nibble → g4, high nibble → g5 |
| 3 | 6, 7 | `qs[96..127]` | low nibble → g6, high nibble → g7 |

Q5 loads the matching `qh[0..31]` word once and selects the two group bits from it. Pairing alone moved Q4 from
221.184 to 190.464 µs cold (1.161×); Q5 was within two event ticks of its slice optimum and slightly worse warm.
The composition with four-code word extraction was decisive for both: production reaches 137.216/161.792 µs.

**Four-code extraction.** Q2/Q4/Q5 block strides (84/144/176 B) preserve four-byte alignment. Q3/Q6 strides
(110/210 B) make every odd block only two-byte aligned, so the loader checks the address: one `uint32_t` load when
four-byte aligned, two `uint16_t` loads when only two-byte aligned, and byte assembly as the general fallback. Each
word is masked/merged while four codes remain byte-packed; the group loop retains an explicit scalar tail.

A benchmark-only padded device stride (Q3 110→112 B, Q6 210→212 B) was also measured three times in alternating
order. It regressed Q3 from 1820–1846 to 1771–1795 Gelem/s, but improved Q6 from 1285 to 1473 Gelem/s cold. It is not
the shipped result: `ppu_backend::Api::vecdot` currently promises compact raw GGUF blocks and no repack workspace
exists behind the Python seam. Two neighboring aligned 32-bit loads on compact odd blocks were also neutral. Padding
Q6 is a worthwhile follow-up only together with an explicit device-layout/repack contract; the table keeps raw bytes.

The `__byte_perm`/`ppu.prmt.b32` magic-u8-to-f32 variant was tested and rejected: Q2/Q3/Q5 cold times were unchanged,
Q4 cost one 2.048 µs event tick, and Q6 warm time regressed. Production therefore uses ordinary exact conversions
and does not duplicate actlize's selector/magic constants. Its shared converter emits fp16 GEMM operand fragments
with PPU-only asm, while this common CUDA/hgcc GEMV needs fp32 codes immediately for fp32 activation products.

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
The old shares must not be read as a profile of the new kernel. Scale/min unpack was material only for Q5 in that
baseline. Warming now changes the final kernels by 1.9–12.1%, with the larger Q2/Q3 shifts consistent with the packed
versions becoming more memory-sensitive.

The regime was never reported, only the speedup, and a speedup measured against a one-row-per-thread baseline says
nothing about how much is left. Both halves are the same measurement run; only one of them is a target.

**Every number above is a 5090 number, and not one GGUF kernel has run on PPU.** The ordering does not transfer:
at gs=32 the 5090's ranking of GEMV configurations inverted against PPU's. Treat these as directional only.

**The two dequantiser kernels' loss was the store pattern**, and the GEMV's serial baseline had the same shape on the load
side. One thread per block puts a warp's lane addresses 512 bytes apart, so one instruction touches 32 separate
32-byte sectors for 64 useful bytes: **6.25% sector utilisation, exactly 16× worse** than lanes touching consecutive
elements. Partitioning fixes the sectors; native packed-code extraction remains the dominant GEMV term afterward.

**None of them run on PPU.** They compile under nvcc and run on a 5090; the PPU device path goes through
`build.sh`/hgcc and the dlopen seam (`ppu_backend.{h,cpp}`), which is wired but has no `libquactlize_ppu.so` behind
it yet.

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
6. **The GEMV's PPU launcher, not its kernel** — the host side is done: `routes.matmul_native_gemv` assembles a full
   `(1, n)` product and agrees with the oracle to 1.4e-7. It stays PARTIAL because that assembly tiles the
   *activation* k/256-fold on the host — correct, and useless as a timing. `vecdot_rows_kernel` is tuned and
   CUDA-validated now; `libquactlize_ppu.so` still needs the exported launch entry before Python reaches it on PPU.
   `gemv_lowbit` remains the separate tuned `SCALE_FIRST` route, not a native-scale substitute.

7. **The registry's path vocabulary is coarser than the matrix** — four path names, nine cells. Two shares are
   wrong enough to matter: `fully_quantized/gemv` would be approved by the fp16-*plane* GEMV harness, and
   `dequant_first/gemv` by a GEMM. Both are mapped anyway, because an *unmapped* cell is approved by default —
   the lookup that should refuse it finds nothing to check — and a test now refuses any VALIDATED claim in them.
   Splitting the vocabulary is the fix; it is a task, not a surprise.

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
