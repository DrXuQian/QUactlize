# Standalone Marlin alignment audit

## Current authority (post generic-path removal)

The current comparison no longer routes a Marlin topology through the generic
mixed-input collective.  It is one independent stack:

| Owner | Production file |
|---|---|
| packed W4 / gs128 artifact | `quactlize/include/marlin_format_ppu.hpp` |
| mainloop stage driver | `quactlize/include/quactlize_extensions/cutlass/gemm/collective/marlin_collective_ppu.hpp` |
| classic cp.async / A ldmatrix | `quactlize/include/quactlize_extensions/cutlass/gemm/collective/marlin_load_ppu.hpp` |
| W4 dequant and grouped scale | `quactlize/include/quactlize_extensions/cutlass/gemm/collective/marlin_dequant_ppu.hpp` |
| native PPU n16 MMA / C registers | `quactlize/include/quactlize_extensions/cutlass/gemm/collective/marlin_mma_ppu.hpp` |
| stripe decomposition and global-q lock lifecycle | `quactlize/include/quactlize_extensions/cutlass/gemm/kernel/marlin_scheduler_ppu.hpp` |
| 4->2->1 reduction, fp16 D-chain and final store | `quactlize/include/quactlize_extensions/cutlass/gemm/kernel/marlin_kernel_ppu.hpp` |

The first admitted row is intentionally exact:

```
M=1, N=4096, K=4096, L=1, gs=128
CTA=16x128x128, warp=16x64x32, stages=4
topology=1M x 2N x 4K, threads=256, load=classic cp.async
```

The device anchors remain 17.8 us / 17.5% of nameplate for the local PPU
classic kernel and 21.14 us / 14.5% for the retired generic `4N x 1K` arm.
No device timing is inferred from the local proofs below.

## Current fixed-row closure matrix

| Axis | Awesome-CuTe / PPU classic | standalone stack | State / evidence |
|---|---|---|---|
| W4 artifact and gs128 scale permutation | classic Marlin packed u32 plus 8x8/64-column scale transpose | same representation | **double-anchored locally**; L167 compares independent classic and Awesome formulas, both inverses, and actual asymmetric bytes. |
| CTA / warp topology | `16x128x128`, `16x64x32`, `1M x 2N x 4K`, 256 threads | same | **compile-closed**; L169 and the standalone target gate. |
| A global-to-shared | one predicate-scoped 16-byte cp.async per valid A vector; XOR destination | same | **source and byte-ledger closed**; L171.  The M=1 residue transfers 256 B/cell, not a physical 16-row AIU cube. |
| B global-to-shared | two 16-byte writes/thread = 8,192 B/cell | same | **source and byte-ledger closed**; L171.  There is one producer, not one per K cohort or per B consumer. |
| scale global-to-shared | 16 writers x 16 B = 256 B at each gs128/TileK128 cell | same | **source and byte-ledger closed**; L171. |
| total logical global-to-shared | 8,704 B/cell x 1,024 cells = **8,912,896 B** | **8,912,896 B** | **model-closed, device counter pending**.  L171 anchors the three production load points to both reference sources; it does not claim cache-line traffic. |
| shared-to-register / dequant | classic A x4 load, packed B vector load, two `lop3` biased-int4 converts and grouped scale | same source cadence; PPU x4 register order is bound explicitly | **local source-closed; generated opcode parity remains a device/toolchain postcondition**. |
| C register ABI | PPU classic native `FragC[4]`, eight FP32 values per n16 instruction | exact `FragmentC[4]`, 128 B/thread; no generic CuTe C fragment or whole-accumulator flat view | **type/source/map closed**; L175 binds the production type and three causal plants, while L139 proves the generic CuTe C map differs at 6,144/8,192 coordinates and the classic map/reduction is raw-bit exact. |
| MMA count | 64 n16 MMAs/cell on PPU | same | **65,536/launch locally fixed**; L168.  NVIDIA's reference expresses each n16 as two n8 instructions, so raw opcode counts must be normalized by output work. |
| four-stage issue cadence | three-stage prologue attempts, wait/prime, two B inner iterations, rolling refill | same | **event-ledger closed**; L168, including causal missing-attempt and ordering plants. |
| CTA-local reduction | FP32 4->2->1 shared tree; K0 survives | same | **cadence and ownership closed**; L168 plus kernel contract. |
| CTA stripe | K-fast, `G=max(Q,CU*B)`, reverse-q Awesome traversal, global q lock | same semantics; B=1 is default | **exact-once and lifecycle closed**; L170.  The three physically launched idle CTAs are explicitly invalid rather than allowed to construct an out-of-range descriptor. |
| CTA/segment address state | final per-thread A/B/scale bases and shared coordinates are CTA invariants; a segment only rebases global q/K | same lifetime and pointer equations | **exhaustively source-closed**; L178 checks all 4,325,376 legal fixed-target segments against independent classic and Awesome-CuTe equations.  It also rejects byte/code pitch confusion, local-q rebasing, stale WK deltas, tight-smem substitution and reintroduced hot-loop topology arithmetic. |
| cross-CTA partial | ordered fp16 chain through D, only final peer writes result; an unsplit tile enters no lock/partial protocol | same | **source/ABI closed**; L177 proves 98 split acquire/handoff/release operations with 66 arrive and 32 reset, while the `Q>=CU` whole-tile case performs exactly zero cooperative calls.  Device memory-order progress remains pending. |
| output | K0 stages fp16 into padded shared rows, all CTA threads coalesce the final write | same mechanism | **source closed; codegen/counter parity pending**. |
| launch bounds / shared | `(256,2)`, 50,176 B | `(256,2)`, 50,176 B | **compile-closed**; actual registers, spills and blocks/CU remain device measurements. |

The local byte ledger also normalizes the shared producer work to 4,352 B per
N-warp and 2,176 B per K cohort.  Those are amortized shares: A and scale are
physically produced once per CTA cell.  L171's causal plants make all four
regressions red: A copied once per K cohort, duplicate two-source B production,
stage-count-multiplied refills, and all-thread scale loads.

Awesome-CuTe's cooperative is a deliberate medium difference after that equal
mainloop ledger: a non-first peer cp.async-loads the prior fp16 C partial into
shared memory.  At 66 peer edges this adds `66*128*2 = 16,896 B`, for an
Awesome-CuTe total of 8,929,792 logical G2S bytes.  PPU classic and this
standalone kernel instead load the fp16 D-chain directly into registers, so
their cooperative contributes zero KVD-to-TSM bytes.  The standalone chain's
logical D read+write traffic is 33,792 B, but it belongs to global/register
traffic, not this counter direction.

`MarlinAiuLoadPolicyPPU` is currently only a named future axis.  The admitted
baseline statically requires `MarlinCpAsyncLoadPolicyPPU`; therefore an AIU
experiment cannot silently change the classic comparison.  When enabled it
must live in this standalone collective, with its own byte-map and generated
instruction oracle, rather than add branches to the generic collective.

## What local equality does not establish

The PPU compiler may lower the same source differently.  In particular, a
`lop3` LUT can be printed as a target-specific logical mnemonic (including an
XNOR-shaped form), and scheduler expressions can materialize extra `s.mov`,
`s.cmp` or branches.  Their mere names are not additional algorithmic work;
the fixed-shape disassembly must compare dynamic opcode counts against the PPU
classic symbol.  The remaining device-only questions are:

1. exact opcode counts and absence of accumulator spills;
2. registers and achieved blocks/warps per CU;
3. lock acquire/release progress under repeated launches;
4. KVD/TSM counters versus the 8,912,896-byte logical model; and
5. numerical result and 17.8--19 us timing.

The dequant source is independently bound to CalebDu/Awesome-CuTe commit
`9f166294bd639cad712a531ac6a5e7aeb983ed37`: both use
`D0,D1,S0,S1,M` within each logical n16 block.  The measured local PPU classic
source uses the same constants and arithmetic but orders those calls
`D0,S0,D1,S1,M`; L174 reports this difference rather than falsely calling all
three sources instruction-for-instruction identical.  Only target disassembly
can decide whether that source-order difference changes the final PPU opcode
schedule.

The hot scheduler descriptor is now 20 bytes (global q, K begin/count, peer
ordinal and cached flags), rather than the earlier 44-byte compatibility
record.  Split/first/final are consumed once in the kernel, and the shared
stage bases plus final per-thread source/smem coordinates are constructed
outside the segment loop.  These are source/combination facts, not a register
allocation claim: L176's exact shipping-symbol disassembly is still required
to establish the resulting `s.mov`/`s.cmp`/branch counts and spills.

Any discrepancy is reported against those named postconditions.  It must not
be attributed to the retired generic collective or repaired by putting Marlin
branches back into that collective.

## Historical pre-standalone audit (not current authority)

Everything below records the superseded attempt to align the generic
mixed-input collective.  It remains useful as failure history, but its
`retained-different` labels do not describe the standalone stack above.

## Historical arm axis-by-axis audit

The table below records why the measured historical arm was not a classic
comparison.  Later local work closed selected rows in the aligned target; the
closure matrix above is authoritative for current status.

| Axis | standalone classic | historical measured collective Marlin | Verdict | Evidence / consequence |
|---|---|---|---|---|
| Logical CTA tile | `16x128x128` | `16x128x128` | **same** | classic selects `(MB,NB,KB)=(1,8,8)` at `marlin_classic_ppu.cuh:841-845,893`; the production row is pinned in `l134_marlin_codegen.cu:16-25`. |
| FP16 MMA atom | PPU `m16n16k16`, FP32 accumulator | PPU `m16n16k16`, FP32 accumulator | **same** | classic calls `mma_n16` at `:513-548`; the builder selects the m16 atom for `TM=16/WM=16` at `quactlize_mma_builder.inl:335-355`. |
| Warp grid | `2N x 4K`: `warp_n=warp%2`, `warp_k=warp/2` | `4N x 1K`: `Layout<Shape<1,4,_1>>` | **different, load-bearing** | classic `:444,471-472`; shipping builder hard-codes `_1` at `quactlize_mma_builder.inl:363-368`. |
| CTA threads | 256 (8 warps) | 128 (4 warps) | **different, load-bearing** | classic `THREADS=256` at `:804`; collective launches `cute::size(TiledMma{})` at `ppu_aiu_gemm_mixed_input_marlin.hpp:64-79,218`. |
| C fragment per thread | 32 FP32 values, repeated for each K cohort | 16 FP32 values, no K cohort | **different, implied by topology** | classic `FragC[1][4]`, eight floats each at `:365-378`; collective owns `16*128/128=16` values/thread.  A `2N x 4K` collective gives `16*128/(2*32)=32` values/thread per K cohort, matching classic. |
| CTA-local K reduction | shared-memory FP32 tree across four K warps; only `warp_k==0` retains C | absent | **different, load-bearing** | classic `thread_block_reduce` at `:552-595`, followed by the `threadIdx/32 < thread_n_blocks/4` writers at `:602,648,689`; current kernel sends the unreduced fragment directly to CTA fixup/epilogue at `ppu_aiu_gemm_mixed_input_marlin.hpp:258-295`. |
| CTA stripe decomposition | equal-length K-fast stripes; a CTA may continue across output tiles | the same K-fast stripe core | **same by local proof** | classic `:247-285`; L126/L133/L134 bind the production scheduler and exhaust exact-once coverage. |
| Default launch protection | `G = Q` when `Q>=CU`, otherwise `G=CU` | implicit/default `blocks_per_cu=1` gives `G=max(Q,CU)` | **same** | classic `:887-889`; `ppu_tile_scheduler_marlin.hpp:68,111-153`.  Explicit B2/B4/B6 are diagnostics, not the default. |
| Cross-CTA peer identity/order | global output tile is the lock; peers enter in `slice_idx` order | global `q` is the lock; reverse peer indices are consumed in strict order | **same semantics** | classic `:253-279,769-774`; collective scheduler/fixup is pinned by L126/L134 and the lock-lifecycle fingerprint. |
| Cross-CTA partial medium | fp16 chain through C itself | FP32 workspace chain, predicated to valid residue | **different, intentional** | classic `:597-620`; collective `ppu_tile_scheduler_marlin.hpp:282-362`.  The collective preserves C/D/beta ABI and precision; this is not part of the first alignment change. |
| Pipeline stages | 4 | 3 | **different** | classic `MARLIN_STAGES=4` at `:792-804`; production Cfg has `St=3`. |
| Per-stage storage size | 12,544 B (`A=4096`, `B=8192`, scale=256) | 12,544 B for this shape | **same size, different layout/copy mechanism** | classic formula `:814-816`; collective's three-stage total is 37,632 B.  Aligning to four stages would make 50,176 B/CTA, equal to classic. |
| A global-to-shared path | per-thread 16-byte `cp.async`; manually XOR-swizzled shared indices | AIU bulk tensor copy with `.padz.swzl`; CuTe copy traits | **different, intentionally retained initially** | classic `:335-398,459-483`; collective `quactlize_mma_mixed_input.hpp:1066-1085`. |
| B global-to-shared path | per-thread 16-byte `cp.async` from classic Marlin packing into plain shared staging | AIU bulk tensor copy from the versioned xplane artifact | **different mechanism; byte-map equivalence unknown** | classic `:297-304,357-389,485-486`; collective `:1066-1085`.  No oracle currently compares classic's packed B bytes with xplane bytes, so this row must not be called equal. |
| B shared-to-register / converter | manual `frag_b_quant`, Marlin dequant, then grouped scale | fixed m16 int8 shadow load, `partition_S -> retile_D -> MixGemmEmit`, then scale/zero transform | **different, intentionally retained initially** | classic `:400-458,485-548`; collective `quactlize_mma_mixed_input.hpp:1125-1249`.  Both implement W4A16 gs128, but they are not instruction-identical. |
| Scale path | host-permuted classic scales, one int4/lane/stage; multiply inside K loop | versioned metadata layout, tiled copies, scale fragment/transform inside K loop | **different representation and load path; same mathematical placement** | classic `:400-458,520-543`; collective `:1291-1345,1359-1430`. |
| `b_sh_wr_iters` / K-cohort coupling | exactly 2; every per-K address includes `warp_k` where required | `K_BLOCK_MAX` is collective-derived, but topology has no `warp_k` today | **different; current WK>1 seam not implemented** | classic `b_sh_wr_iters=b_sh_stage/threads=2` at `:297-304`, and A's K tile explicitly adds `warp_k` at `:468-481`.  L123 proves changing only `AtomLayout.K` is insufficient. |
| Epilogue | K-reduced N warps stage FP32 C as fp16 into padded shared rows; all 256 threads stream coalesced stores | generic `EpilogueSimtVectorized` consumes every TiledMma thread | **different, load-bearing at WK>1** | classic `:622-719`; collective builder `test_lowbit_dense_bench.cu:214-226` and call site `ppu_aiu_gemm_mixed_input_marlin.hpp:288-295`.  After CTA-local reduction, only one K cohort may supply unique C fragments to the epilogue; the exact cohort must be encoded, not inferred at runtime. |
| Register contract | `__launch_bounds__(256,2)`, zero reported spill in the documented classic point; exact registers/thread not recorded | measured 160 registers/thread at 128 threads; occupancy API says six blocks | **unknown equality / known different launch contract** | classic `:226-241`; committed ACU transcription records the collective count.  Device codegen must report the aligned 256-thread kernel before comparing occupancy. |
| Mainloop instruction schedule | four-stage manual pipeline, two B write iterations, classic dependency chain | shared mixed pipeline driver, three stages, collective-derived B chunks | **different** | classic `:721-790`; collective `quactlize_mma_mixed_input.hpp:1202-1249`. |

## Classic-aligned target: local closure

The isolated target now aligns the topology rows that were intentionally left
different in the measured control: `2N x 4K`, 256 threads, four stages, 32
FP32 C values per thread, and a 64-thread K0 output cohort after CTA-local FP32
reduction.  The historical `4N x 1K` target remains unchanged.

Its B path does **not** introduce a new artifact.  L142 models the exact
production source and destination layouts; L143 independently exhausts 8,192
half2 destinations / 16,384 int4 codes and proves that direct `(t,t+4)` pair
selection consumes the shipping xplane bytes with zero differences (hash
`b89b157b5b1bd6c3`).  The sequential production-fragment order (`ea96...`),
compact-s16 order (`17df...`), whole `Converter<32>`, adjacent-nibble pairing,
and swapped sources are permanent red controls.  The real Cfg full-body gate
instantiates the WK4 branch; WK2 is an expected compile-time failure and WK1
retains its original branch and bytes.

Remaining unknowns are device codegen details (register count/spills), runtime
lock progress, occupancy, time and ACU counters.  The existing A/B copy
machinery, scale/zero path and cross-CTA FP32 workspace remain intentionally
different from standalone classic.

## L123 artifact gate (proved before kernel changes)

`run_l123_warp_nk_topology.sh` was rerun on this tree.  Its scope and result are:

- `WK=1` is exactly the shipping builder type and the shipping F1-int4 xplane
  map: `0/8192` differences.  This must remain a permanent assertion.
- `(WN,WK)` is genuinely two-dimensional.  Equal-warp controls `2N x 2K`
  and `1N x 4K` have different types, correct A/B K shards, C-slot identity,
  and distinct B maps (`fixed4-diff=6144`).
- The oracle includes the real `partition_S -> retile_D -> converter emission ->
  partition_B` chain, not only an abstract CuTe layout.  Its stale shadow-K and
  folded-K negative controls both fail.
- The compact destination used by L123 changes `6144/8192` logical slots for
  WK2 and WK4.  This proves that a single K base, a global vreg permutation,
  or their combination cannot repair that compact consumer.  It explicitly
  did **not** exclude a per-cohort/per-vreg consumer remap.
- L142/L143 found exactly that narrower remap for the production-layout WK4
  target.  Therefore WarpK is a consumer/tactic axis for this target, not an
  artifact descriptor axis: the shipping packer and bytes remain unchanged.
  Folded, two-plane, int1/int2 and other WarpK shapes remain unproved and fail
  closed; this result must not be generalized from pattern matching.
- Positive folded coverage exists for int2-F2 and Q3 low/high planes.  Int1,
  Q6, real-device numerical execution and performance are still **unknown**.

Therefore the aligned implementation calls an explicit WarpK consumer API,
but that API resolves to shipping bytes only for the exact L142/L143-proved
int4 target.  Every other non-default topology fails closed.

## Implemented boundary fixed by this audit

The local alignment change is deliberately narrower than a rewrite of the
mixed-input formats:

1. add a `(WN,WK)` TiledMma topology while preserving exact WK1 types;
2. add CTA-local FP32 reduction with register-index/coordinate identity proved
   before device timing;
3. route only the surviving K cohort into CTA fixup and the epilogue;
4. keep the shipping artifact and select its pairs in the production consumer;
5. keep converter, scale/zero, fold, B-chunk, two-plane and the FP32 cross-CTA
   workspace implementation otherwise unchanged;
6. measure the aligned kernel and B1/B2/B4/B6 in one box batch.

If that kernel does not converge toward the 17.8 us classic anchor, the
remaining explicit differences above -- A/B copy machinery,
converter/scale/metadata path, CTA-local reduction algorithm, cross-CTA
partial medium/order, epilogue, build/launch/register contract, and the
four-stage collective issue schedule -- are the next attribution set.  The
stage count itself is aligned and must not be relisted as an explanation.
Failure to converge is not evidence that the PPU has a generic Marlin ceiling.
