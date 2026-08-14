# BC Q4 GEMV on RTX 5090: resident bytes are not the gap

Date: 2026-08-14 UTC

This note binds one same-machine, same-shape experiment to the generated SASS.  It answers a narrow question:
why the first shipping-BC reader measured about five times slower than the user-supplied PDF Q4_K kernel, and
whether retaining the prefill-compatible xplane artifact prevents the decode kernel from reaching the PDF result.

The answer is **no**.  The prototype which kept the exact resident A64 xplane codes and byte-neutral packed units,
but used the PDF work decomposition and native NVIDIA half arithmetic, measured **7.621 us** against the PDF
kernel's **7.788 us** in the same binary.  That implementation has since been promoted behind the public dense
Q4/A64 dispatch.  The format is therefore not the performance obstacle.

## Bound experiment

Device and toolchain:

* NVIDIA GeForce RTX 5090, `GPU-bf1f779f-e900-eb25-708b-ec70a7b0a5f4`, compute capability 12.0
* driver `595.71.05`, reported peak `1792.128 GB/s`, L2 `100663296 B`
* repository HEAD while compiling: `5c3edf4efcc203c717f1d4804e8e06fd85408549`
* binary: `/workspace/bc-perf-diagnosis/bc2-build/q4k_pdf_vs_bc_5090`
  (`sha256=0d7f3d83e725ed7c06985d4e915744f3ed44357e94609f7ab47f89b3a4cb847d`)
* result: `/workspace/bc-perf-diagnosis/bc2-build/result-all-31.log`
  (`sha256=bc908d3d84ef164e2201a6b0d1996a50a69add097859ac32c861995c1278099d`)

Protocol:

* `M=1, N=4096, K=4096`, GGUF Q4_K, `ArtifactTileK=64`
* 24 distinct copies; raw and resident artifacts are both `9437184 B`, and the rotating set is `2.25x` L2
* packing is outside timing; the resident artifact is `8388608 B` xplane codes plus `1048576 B` packed units
* 31 raw CUDA-event samples per arm
* both positive and signed activation correctness arms ran before timing

The earlier first-result artifact remains useful because it binds the original reported gap without later worktree
changes: `/workspace/quactlize-q4k-pdf-vs-bc-5090-a64/result.log`
(`sha256=96b80dce913bbfe622e7962348b5a4e348a90dd38b5d813147a1d0c5296f2d52`) reported
`38.038666 / 7.785333 = 4.886x` for BC/PDF.

## Results

| arm | ownership | grid / threads | median us | weight GB/s | % peak | regs | spill | barriers |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PDF reference | CtaN2, Wn4, Wk1 | `(1,512,1) / 128` | 7.788 | 1211.760 | 67.616 | 64 | 0 | 1 |
| pre-fast shipping BC reader | A64, RPW8 | `(256,1,1) / 64` | 38.711 | 243.788 | 13.603 | 40 | 0 | 0 |
| BC resident bytes, PDF topology (`bc2`) | A64, CtaN2, Wn4, Wk1 | `(1,512,1) / 128` | **7.621** | **1238.259** | **69.094** | 64 | 0 | 1 |

In the same binary, `bc2 / PDF = 0.978600`: the resident-artifact arm is 2.14% faster.  Its positive fixture is
bit-identical to the independent golden (`rel_l2=0`); its signed fixture reports `rel_l2=0.000710245423`,
`max_abs=0.000534057617`, and `max_conditioned=0.0557575758`, all inside the predeclared gate.

This row was the benchmark-first proof.  The public `bc_vecdot::launch` now routes dense Q4/A64/fp16 CUDA calls to
`bc_q4_gemv::launch_default`; unsupported targets, formats, arrangements, grouped calls, tails, and oversized K
fall back to the generic reader.  The production route keeps the shipping bytes while replacing reader topology,
metadata arithmetic, activation delivery, and accumulator shape together.

### Production-bound rerun

The final run was compiled from code commit `17ee232e06a18392979989e97f3fd9b3bb5609b4`; each production source also
has its own hash in `identity.txt`.  The public dispatch is exercised specifically by the winning shipping arm.

* artifacts: `/workspace/quactlize-q4k-pdf-vs-bc-5090-17ee232`
* binary SHA256: `6952e52366180f3e5a3086fc50f5a1f370ee60c4b0cfaec5b10600fe2f8a0da3`
* result SHA256: `bc3deac794eb0fe3a97c29b29abe8e807a6b0aaeda0ab811b8f94f714cb3961f`
* protocol: 31 event samples, 24 cold copies = `2.25x` L2, packing outside timing

| arm | median us | weight GB/s | % peak | regs | local bytes/thread | barriers |
|---|---:|---:|---:|---:|---:|---:|
| PDF winner, CtaN2/Wn4/Wk1 | 7.793333 | 1210.930 | 67.57 | 64 | 0 | 1 |
| generic BC after native metadata, RPW8/T64 | 12.833333 | 735.365 | 41.033 | 40 | 0 | 0 |
| **public shipping BC, A64 CtaN2/Wn4/Wk1** | **7.625333** | **1237.609** | **69.058** | **63** | **0** | **1** |

The production ratio is `0.978443`, so the resident-artifact route is **2.16% faster** than the exact PDF reference
in the same binary.  Positive activation is bit-exact to the independent golden; signed activation remains
`rel_l2=0.000710245423`.  The wrong PDF magic control is 32/32 red, the legacy-artifact plant reaches all 12 legacy
configs, the one-bit shipping magic plant is 24/24 red, and a deliberately omitted arm is an explicit SKIP.

## Static attribution of the old reader

The exact old BC winner's SASS contains 2056 static instructions; the PDF CtaN2/Wn4/Wk1 kernel contains 496.  Of
the BC instructions, **1695/2056 (82.4%)** carry line attribution to
`third_party/actlize/include/cutlass/half.h`'s software conversions and operators.

The mechanism is explicit in the source:

1. `cutlass/half.h:254-304` uses the software half-to-float converter unless `__HGGC_ARCH__ >= 100`.
2. `cutlass/half.h:756-760` implements `half_t * half_t` as float conversions outside that PPU-only branch;
   unary minus has the same form at lines 738-743.
3. `packed_unit::unit_group_sb` calls `make_group_scale`, so Q4 metadata decode enters those operators.
4. `gguf_bc_vecdot.hpp` then calls `float(sz.scale)` and `float(sz.zero)` in the hot superblock loop, entering the
   software half-to-float converter again under NVIDIA `__CUDA_ARCH__`.

This is not merely a large source body.  It creates almost all of the old kernel's control instructions:

| SASS opcode family | whole old BC kernel | attributed to `cutlass/half.h` |
|---|---:|---:|
| all static instructions | 2056 | 1695 |
| `LOP3` | 348 | 287 |
| `IADD3` | 263 | 255 |
| `ISETP` | 259 | 253 |
| `BRA` | 226 | 222 |
| `BSSY` / `BSYNC` | 78 / 78 | 77 / 77 |
| `IMAD` | 136 | 109 |
| `SHF` | 124 | 99 |
| `MOV` | 119 | 106 |

The native production metadata decoder instead uses native half/half2 operations through the shared
`gguf_bc_q4_reader.hpp`; it does not numerically convert a `cutlass::half_t` through the portable software
implementation.  Its complete winner
kernel is **432 static instructions**, versus 496 for PDF, with the same 64 registers, no spill, and one barrier.
Thus the reader can be instruction-competitive without changing one resident byte.

The counts can be reproduced without performance-counter permission:

```bash
mkdir -p /workspace/bc-q4-static
cd /workspace/bc-q4-static
cuobjdump --dump-sass /workspace/quactlize-q4k-pdf-vs-bc-5090-a64/q4k_pdf_vs_bc_5090 \
  > old.sass
cuobjdump --extract-elf all \
  /workspace/quactlize-q4k-pdf-vs-bc-5090-a64/q4k_pdf_vs_bc_5090
# Run nvdisasm -g on the extracted sm_120 cubin.  Count instruction lines inside
# rows_kernel<Q4_K,64,8,false>, and maintain the most recent `//## File` record
# to group them by source.  The bound result is 2056 total / 1695 half.h.
```

Nsight Compute counters were not used for this conclusion: this host currently returns
`ERR_NVGPUCTRPERM`.  Latency, resource usage, correctness, and SASS are available; stall attribution is not.

## The second structural defect: RPW is not K parallelism

The first reader assigns a Q4 row only `Groups/2 = 4` work items:

```cpp
for (int pair = row_lane; pair < Groups/2; pair += LanesPerRow)
```

Consequences at `K=4096`:

* RPW8 has four lanes per output, so every lane works, but each lane consumes 1024 weights.
* RPW2 has sixteen lanes per output, but only four are useful; twelve of sixteen lanes are idle.
* RPW1 has thirty-two lanes per output, but only four are useful; twenty-eight are idle.

Therefore the old RPW sweep cannot express the PDF kernel's K ownership.  PDF CtaN2/Wn4/Wk1 uses all 32 lanes
for two outputs, so each lane consumes 256 weight products.  Lowering RPW in the old kernel creates more warps but
does not split the four-word group among them; it trades occupancy against inactive lanes.

The shipping cooperative topology proves the needed decomposition directly: eight lanes own the eight metadata groups of a superblock, each lane
loads the group's four resident words, four warps cover eight output columns, and the CTA stages the activation row
once.  This is also why the resident A64 map is compatible: one logical 32-code group is already one contiguous
`uint4`; only the proved P4x32 register permutation remains.

## What landed, and what remains

1. Native metadata and target-dispatched whole-word dequant are shared by the generic and cooperative Q4 readers.
2. The cooperative A64 topology is selected by the public CUDA dense dispatch; no raw-GGUF production operand exists.
3. Scalar `code_at` remains an independent oracle.  L187 exhaustively binds P4x32 against producer bytes and carries
   wrong-permutation and missing-denominator plants.
4. The same-binary sweep remains broader than the PDF reference and selects the production default from measured
   CtaN/Wn points rather than freezing the reference's single configuration.

Activation staging is part of the production point, not independently priced.  The pre-fast reader logically
reloaded the K vector per output, while the new route stages 4096 halves once per CTA and shares them across its
output columns.  Separating staging from K ownership would require a controlled A/B; this note assigns no invented
percentage to either mechanism.  PPU device code, other formats/arrangements, grouped ownership, and larger K remain
separately gated work.

## Scope

The conclusion is limited to RTX 5090 / sm_120, Q4_K, A64, `M=1,N=K=4096`, and this cold 2.25x-L2 protocol.  It does
not establish PPU code generation, other artifact tile widths, other quant formats, grouped MoE ownership, or warm
cache behavior.  It does establish the fact needed for the overnight task: **the actual CUDA production A64 route
can meet and slightly beat the PDF Q4_K target; the 4.9x loss belonged to the old reader implementation.**
