# Sweep state — agreed checkpoint, 2026-08-04

This is the single current statement of the dense and grouped low-bit sweep. It supersedes the stage and N-geometry
parts of `SWEEP_032_PRUNING_CODEX.md`; that file remains the derivation record, not the live specification.

Evidence labels used below:

- **CHECKED** — read from the current `develop` source or reproduced with a host-side generator/predicate.
- **MEASURED** — device result from ppu001 recorded in the coordination log; the arithmetic was rechecked locally,
  but this document's author did not rerun the device measurement.
- **PENDING** — a deliberate decision or measurement has not happened. It is not silently treated as true.

## Executable scope today

The active sweep is one stored schema, i4, at TileK=64. TileK remains a tactic/arrangement axis across separately
built binaries; equal fold does not make two TileK kernels performance-equivalent. Split-K is outside this sweep.

| Operator/build | Enumeration after the four-warp fix | Performance policy |
|---|---:|---|
| Dense i4/TK64, ordinary A | **227 stage configurations**: 65 primary, 162 guard-only | Pruned at generation time |
| Grouped i4/TK64, ordinary A | **402 executable stage configurations** from 76 legal shapes | Unpruned; classify saved samples afterward |

**CHECKED.** The dense count is emitted in `benchmarks/lowbit_dense_configs.inc`. For grouped,
`MOE_FORMATS=i4` generates 180 Cartesian source units (`3 WarpN * 4 TileN * 3 WarpM * 5 TileM`); the shared
predicate leaves 76 nonempty shapes. Stage/smem legality admits 76 rows at each of s2, s3, s4, and s6, 53 at s8,
and 45 at s12: **402 total**. Applying the current dense policy to that same legal grouped set retains the same
227 rows, so the queued policy-cost comparison has 175 outside-policy rows. The full five-format MoE generator
still creates 600 source units; that is a generator count, not 600 executable configurations.

These are the counts for the current ordinary-A binaries. Compact A capacities 1, 2, and 4 are separate kernel
types. The listed configurations can be built with those capacities where the compact reader exists, but the current
enumerators do not add configurations which become legal only because compact A reduces shared memory. Therefore
227 and 402 must not be presented as a complete compact-specific search count. **CHECKED limitation.**

### Stage axis

- Grouped/MoE keeps **{2, 3, 4, 6, 8, 12}**. Stage depth is an operator-specific axis, and s6 has already won a
  grouped fixture. It stays complete until grouped measurements justify a cut. **MEASURED decision.**
- Dense currently also compiles **{2, 3, 4, 6, 8, 12}**. This is coverage, not a claim that deep stages win on
  dense. The user has not re-decided dense's permanent stage scope after the grouped s6 result. **PENDING decision.**
- No stage scope is transferred from one operator to the other without operator-specific evidence.

### Legality before pruning

`ppu_tactic_space.hpp` is the common authority for dense and grouped. It checks atom alignment, warp divisibility,
the CTA warp bounds, accumulator and per-stage shared-memory ceilings, per-plane delivery/fold constraints, compact-A
reachability, and current producer reachability. Both launchers static-assert the kernel part; the dense emitter and
MoE `moe_ok` ask the same shared predicate.

The new lower CTA bound is **one 128-thread PPU warp group = four 32-thread warps**. **CHECKED implementation,
MEASURED boundary.** Before the fix, the MoE axis lists contained one- and two-warp cells; there was no independent
MoE check. It was not protected by the particular `MOE_WM_LIST`/`MOE_TN_LIST` values. The common gate now prevents
both operators from instantiating or launching them.

### Performance pruning currently defined

Legality is applied first. For every legal stage independently:

1. **Primary:** keep the largest legal WarpM for `(TileM, TileN, WarpN)` when `TileN/WarpN = 2`.
2. **H1 guard:** at every TileM, keep the next-smaller legal WarpM at the lightest and heaviest legal ratio-two
   TileN geometry.
3. **N-geometry guard:** at every TileM, keep the largest legal WarpM for **every legal TileN/WarpN ratio**.
   This deliberately includes ratio 8 and the interior TileM=64 geometry which produced the grouped winner.
4. Fully cross the operator's current stage axis. A guard inside the leader's uncertainty band expands the affected
   stratum before a result is called a winner.

**CHECKED.** In the dense i4/TK64 table these rules produce 65 primary plus 162 guard-only rows. Grouped is not
compiled with this pruning: its 402 rows run once and the exact saved samples are partitioned afterward into the
227 policy rows and the full 402, avoiding two timing populations.

H1 remains supported by the first grouped result: the winner uses WarpM=64, the largest legal WarpM at TileM=64.
Ratio two remains a primary sampling hypothesis, not a dominance claim. The former enumerated `{1,2,4}` N guard and
the former extreme-TileM-only guard are withdrawn because together they excluded the measured ratio-8/interior-TileM
winner.

## Workload fixtures

All operators use global token counts **{1, 2, 4, 64, 2048, 4096}**. Dense uses that value directly as M. MoE
routes it through 256 experts at top-k 8 and uses the resulting per-expert histogram, never the mean as a substitute.

The model-derived, per-card projection shapes currently encoded are:

| Model | Operator projections `(N,K)` |
|---|---|
| Qwen3-32B, TP1 | dense: q `(8192,5120)`, k/v `(1024,5120)`, o `(5120,8192)`, gate/up `(25600,5120)`, down `(5120,25600)` |
| Qwen3.5-35B-A3B, TP1 | dense: q `(4096,2048)`, k/v `(512,2048)`, o `(2048,4096)`; grouped: gate/up `(512,2048)`, down `(2048,512)` |
| Qwen3.5-122B-A10B, TP2 | dense: q `(4096,3072)`, k/v `(256,3072)`, o `(3072,4096)`; grouped: gate/up `(512,3072)`, down `(3072,512)` |

**CHECKED against repository fixture generation; checkpoint confirmation is PENDING.** These shapes were derived
from model `config.json` and the user-confirmed standard TP split. The quantized checkpoint tensor shapes remain the
authority and must be checked before spending the full sweep. The measured A3B row below directly exercises its
`(N,K)=(512,2048)` grouped gate/up shape.

The grouped distribution is named and versioned:

`token-topk-hot16x4-wor-sm64-s44-v1`

Each token chooses eight distinct experts without replacement. Experts 0–15 have lottery weight 4, the remaining
240 have weight 1, and a fixed SplitMix64 stream makes the histogram reproducible. At token counts
`{1,2,4,64,2048,4096}`, its checked `(active experts, Mmax)` ladder is
`{(8,1),(15,2),(30,3),(212,12),(256,239),(256,447)}`. Thus compact A capacities 1, 2, and 4 serve the first three
fixtures; the remaining three require the ordinary A path. A too-small compact build refuses the fixture rather
than widening or falling back.

## What has actually been measured

The first real grouped sweep result on ppu001 is:

```text
i4  64x128:64 w64x16 s6    423.96 us | 162.1 TF/s (32.4% useful MFU)   band: separated
```

**MEASURED.** Fixture: Qwen3.5-35B-A3B expert gate/up, `N=512`, `K=2048`, 4096 tokens, 256 experts, top-k 8,
the pinned router distribution above. It performs 32,768 real expert rows. The uncertainty band was separated,
so this was not reported from an unresolved tie.

The historical A3B FC1 record used `N=1024`, `K=2048`, 2048 tokens x top-8 over 128 experts = 16,384 rows and
reported 416 us, 165 TF/s, and 33.0% useful MFU. The useful work is exactly equal:

```text
2 * 16,384 * 1,024 * 2,048 = 2 * 32,768 * 512 * 2,048
                              = 68,719,476,736 FLOP
```

Redistributing the same 68.72 GFLOP changed 416 to 423.96 us (+1.9%), 165 to 162.1 TF/s, and 33.0% to 32.4%.
**CHECKED arithmetic over MEASURED/RECORDED inputs.** The new harness therefore reproduces the historical ragged
result within 2%. The 32.4% result is evidence for the ragged useful-work ceiling, not a failed attempt to reach the
49.2% uniform result; the roughly 16-point gap is the previously recorded masked-row tax.

## Known unmeasurable paths

- **Sub-four-warp CTAs:** a safe sweep cannot launch them. On ppu001 every tested one- or two-warp dense cell hit an
  unconditional device assert and every tested cell at four or more ran. A device assert poisons the context, so it
  cannot be handled as a refused row. The shared predicate excludes these cells until the exact failing mechanism is
  named and a replacement boundary is positively validated. **MEASURED.**
- **Fully-quantized prefill on the local CUDA machine:** the primary prefill branch is the PPU tensor-core GEMM and
  cannot execute on CUDA. The llama.cpp validation harness must decline it explicitly; recover/dequantize plus cuBLAS
  is only an opt-in cross-check and is not validation of the shipped branch. Only ppu001 can validate that path.
  **CHECKED platform limitation; PPU validation PENDING.**
- **PPU kernel performance or PPU-only correctness locally:** host generation and CUDA-side shuffle/recovery/GEMV
  checks are local, but any conclusion requiring the PPU collective, its tensor-core instructions, or ppu001 timing
  remains box-only.

## Open decisions and unknowns

There is no remaining disagreement between the two collaborators in this document. The open items are:

1. **Exact sub-four-warp assert site.** Actlize defines a PPU warp group as 128 threads, its universal GEMM adapter
   independently models a minimum of four warps, and the kernels launch the actual `size(TiledMma)` threads. These facts
   corroborate the four-warp gate, but the supplied device trace has no file:line and source inspection did not identify
   the firing assert. Because the finite geometry contains no three-warp CTA, today's evidence cannot distinguish a
   hypothetical `>=3` rule from `>=4`; no one may weaken the checked four-warp boundary on that basis. **UNKNOWN root
   cause, checked conservative gate.**
2. **Dense stage scope.** Six stages are compiled now, but the user has not decided whether dense keeps all six.
   Grouped's s6 result does not decide dense. **PENDING user decision.**
3. **Cost of the performance policy.** One unpruned A3B shape will compare the full 402-row result with the 227-row
   policy partition using identical timing samples. Until it runs, the 175 omitted rows have an unknown cost. **PENDING
   ppu001 measurement.**
4. **One generator for both operators.** **DONE.** CMake consumes the emitted grouped tables rather than a Cartesian
   product, and `DenseSpace` / `GroupedSpace` are now public aliases of one `TacticSpace` implementation. A structural
   type guard and full emitter-route gate prevent a second legality chain from returning silently. **CHECKED.**
5. **Checkpoint shape confirmation and compact-specific enumeration.** The source fixtures are reproducible, but the
   loaded quantized artifacts still need to confirm their shapes, and the generators do not yet enumerate the extra
   stage/topology cells made legal only by compact A's smaller footprint. **PENDING.**
