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
| **packed** | `FULLY_QUANTIZED` | nothing; the collective decodes the scale in-kernel | none, plus a measured +11.6% tax |
| **GEMV** | `FULLY_QUANTIZED` | nothing; the scale pair is consumed in registers | none |

The fallback's extra traffic is **16×** the pre-pass's and its GEMM reads **3.6×** more bytes, so the crossover is in
**M**, not in implementation quality: at decode it is absurd, in the middle band the pre-pass's 1.05 MB constant
trades against the packed path's 11.6% rate, and only at large M does reuse amortise the dequantised weight far
enough for cuBLAS-grade efficiency to pay for the bytes. `formats.select_path` already takes `num_rows`; what it
lacks is measurement to set the thresholds with.

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

| kernel | state | measured (RTX 5090) |
|---|---|---|
| `dequantize_kernel_warp` | landed | 779.4 → **58.4 µs**, 13.4×, 1.473 TB/s = **82.2% of peak**, bit-identical |
| `prepass_kernel` (cooperative) | landed | speedup grows with size — 1.32× / 1.91× / 2.95× — the fixed-duration floor made visible |
| `vecdot_rows_kernel` | landed | correctness only; not tuned |

**The loss both removed was the store pattern.** One thread per block puts a warp's lane addresses 512 bytes apart,
so one store instruction touches 32 separate 32-byte sectors for 64 useful bytes: **6.25% sector utilisation, exactly
16× worse** than lanes writing consecutive elements.

**Neither runs on PPU.** They compile under nvcc and run on a 5090; the PPU device path goes through `build.sh`/hgcc
and the dlopen seam (`ppu_backend.{h,cpp}`), which is wired but has no `libquactlize_ppu.so` behind it yet.

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
6. **The GEMV is wiring, not a kernel** — `gemv_lowbit` is validated and consumes exactly what the offline chain
   produces. Note it produces `SCALE_FIRST` execution, not native-scale `FULLY_QUANTIZED`.

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
