# Standalone Marlin alignment audit

This audit binds one exact comparison.  It does not generalize from a generic
"Marlin" name:

```
M=1, N=4096, K=4096, L=1, gs=128
CTA tile = 16x128x128
classic = Marlin<256, 1, 8, 8, 4, 8>
collective = Cfg<128,16,128,128,16,32,3>::MarlinKernel
```

The measured anchors are 17.8 us / 17.5% of nameplate for classic and
21.14 us / 14.5% for the current collective arm.  The 19% time gap must not be
attributed to the CTA stripe scheduler until every row below is either aligned
or retained as an explicit difference.

## Axis-by-axis audit

| Axis | standalone classic | current collective Marlin | Verdict | Evidence / consequence |
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
- The current WK1 artifact cannot directly serve WK2 or WK4.  F1-int4 changes
  `6144/8192` logical slots for either candidate, and WK2 and WK4 differ from
  one another.  A single K base, a global vreg permutation, or their
  combination cannot repair it.
- This is an **offline placement descriptor axis**, not a new quantization
  format.  The descriptor/packer must carry a normalized WK placement class
  beside TileK/fold.  The proof does not require the ABI field to be literally
  named `WK`, and it does not exclude a future, more complex kernel-side
  per-cohort/per-vreg remap.
- Positive folded coverage exists for int2-F2 and Q3 low/high planes.  Int1,
  Q6, real-device numerical execution and performance are still **unknown**.

Therefore the first aligned implementation may not silently point a WK4
consumer at the WK1 artifact.  It must either produce/route a WK-aware artifact
or fail closed.

## Implementation boundary fixed by this audit

The first alignment change is deliberately narrower than a rewrite of the
mixed-input formats:

1. add a `(WN,WK)` TiledMma topology while preserving exact WK1 types;
2. add CTA-local FP32 reduction with register-index/coordinate identity proved
   before device timing;
3. route only the surviving K cohort into CTA fixup and the epilogue;
4. make the offline artifact descriptor/packer select the proved WK placement;
5. keep converter, scale/zero, fold, B-chunk, two-plane and the FP32 cross-CTA
   workspace implementation otherwise unchanged;
6. measure the aligned kernel and B1/B2/B4/B6 in one box batch.

If that kernel does not converge toward the 17.8 us classic anchor, the
remaining explicit differences above -- stages, A/B copy machinery,
converter/scale path, epilogue and launch/register contract -- are the next
attribution set.  Failure to converge is not evidence that the PPU has a
generic Marlin ceiling.
