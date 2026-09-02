# K-pack incremental optimization backlog

This backlog starts after the exhaustive canonical K-pack denominator frozen
at `adbd5f8`.  Its executable build campaign was restarted at `754130e` after
repairing the grouped `DeliveryN` registry macro; the repair changes registry
wiring, not the denominator.  The items below are new implementation axes:
they require their own type identity and correctness closure before they may
enter an incremental performance sweep.  They do not justify changing or
pruning that baseline denominator.

## Candidate axes

| candidate | current canonical state | offline arrangement impact | admission work |
|---|---|---|---|
| K-pack `BChunk` | Both K-pack policies fix `BChunkRequest=0` and require `atom_at_a_time=false`. `DeliveryN=16/32/64` is the currently swept K-pack delivery axis. | None if it is only a conversion schedule; a new mapping is forbidden unless the global byte coordinate changes. | First prove that a K-pack BChunk schedule is distinct from `DeliveryN`, then add a typed policy, emitted-instruction check, raw-bit closure, and incremental timing. Do not revive a binary-global diagnostic macro. |
| BC/SIMT decode rows per warp | No canonical K-pack BC reader; `RPW=1/2/4/8` is therefore structurally unavailable in the baseline campaign. | The canonical bytes can remain unchanged if the BC reader consumes the versioned K-pack arrangement directly. | Implement the K-pack BC reader, close dense and grouped raw-bit correctness, then sweep `RPW=1/2/4/8` against the tensor-core decode winner. |
| AIU-plain global-to-shared plus `UniversalCopy` shared-to-register | Current K-pack uses the production AIU/swizzle delivery and swizzle matrix reader. The complete Q4 consumer map has now been derived and is not byte-compatible with the canonical K-pack4 map. PPU0010 m8 and m16 publish the same MMA-B fragment layout, so this Q4 map can cover decode and prefill in our collective. | A distinct experimental Q4 arrangement already exists: layout 3, mapping `0x51344e3136440001`, physical `[K/16][2*N] uint32`. It is an `N16 x K64` atom derived from our converter/MMA chain, not DeepGEMM's permutation. Q2/Q3/Q5/Q6 have no such arrangement yet and must each be derived from their actual one- or two-plane converter chain. | Keep layout 3 fail-closed until exact PPU raw-bit delivery and full converter/MMA numeric closure pass for dense and grouped kernels, then run isomorphic resource/timing A/B. Derive and version every additional qtype independently; never infer its bytes from Q4. |
| Q4 metadata publication | Q4 dense uses interleaved half2 publication; generic Q2/Q3/Q5/Q6 and grouped paths retain separate half planes. This is fixed per current product type, not swept. | Weight bytes do not change. Shared/workspace layout and descriptor identity can change. | Add a typed publication axis only where both implementations are correct; compare store/load conflicts, instructions, shared bytes, and end-to-end time. |
| Packed-A coverage | AP1 is admitted only for proved Q2/Q4 `TM=8, WM=8` decode rows; all other rows are AP0. | Weight arrangement is unchanged. | Extend one format/geometry at a time, with exact A-stage raw-bit closure before timing. |
| Fused Split-K completion | The baseline ranks producer time plus the versioned exact reducer lookup. | No weight-layout change; workspace/launch ABI changes. | Admit fused last-arriver only after counter reset/lifecycle correctness and compare complete end-to-end launches with the separate reducer. |
| N-stage expansion / scheduler order | The baseline expands current persistent grids and grouped persistent/nonpersistent policies, but it does not implement DeepGEMM's `N_EXPAND` pipeline. | Normally unchanged. | Add it as a scheduler type, not a hidden runtime heuristic; validate coverage, tail handling, and paired timing by shape. |

Already covered by the baseline and not backlog items: full legal
`TM/TN/TK`, `WM/WN`, stages `2/3/4/6/8/12`, K-pack `DeliveryN=16/32/64`,
legal AP0/AP1 rows, dense Split-K `S=1/2/4/8`, and runtime persistent grid
space.  Xplane `ArtifactTileK/Fold` is intentionally not a product axis.

## Deferred operand-provider boundary

Do not refactor the collective while the exhaustive canonical build campaign,
the K-pack-only heuristic sweep, or the layout-3 device admission is in
flight.  Those results are source-authority bound; mixing a structural rewrite
into the same closure would make code-generation changes indistinguishable
from configuration or layout effects.  Until those campaigns close, this item
is documentation only and must not change the mainloop.

The current collective is acceptable while it has one production arrangement
and one matching delivery path.  Introduce a provider abstraction only after a
second delivery path has passed correctness and is intended to remain.  The
motivation is clean ownership and impossible invalid pairings, not an assumed
performance gain.  Use explicit names rather than a generic `Legacy` label:

- `KpackSwizzleProvider`: the current canonical K-pack AIU/swizzle path;
- `N16DirectProvider`: Q4 layout 3 plus AIU-plain and `UniversalCopy`;
- `DeepGemmProvider`: a future DeepGEMM-compatible arrangement and its matched
  register/dequant reader, if independently admitted;
- `CpAsyncUniversalProvider`: only if a separately derived cp.async path is
  retained after measurement.

Each provider must own one compile-time-bound chain: arrangement and mapping
identity, global-to-shared writer, shared layout, shared-to-register reader,
raw register layout, decode, and placement into the MMA-ready B fragment.  The
mainloop continues to own pipeline/stage orchestration, MMA issue order,
accumulators, and epilogue.  A layout must not be selectable independently of
its reader.

Admission order after the frozen campaigns complete:

1. Wrap the current path as `KpackSwizzleProvider` and prove identical emitted
   instruction counts, registers, spills, shared/workspace ABI, raw bits, and
   paired timing.  The wrapper is rejected if it changes code generation.
2. Add `N16DirectProvider` only after its standalone PPU probe and dense/grouped
   numerical closures pass; compare it with the wrapped current path under an
   identical tactic and launch.
3. Add other providers as matched arrangement/reader pairs, with both crossed
   pairings planted as RED controls.
4. Keep provider selection compile-time.  Do not add runtime branches or
   binary-global experiment flags to the hot mainloop.

Provider-specific delivery may later recover a performance gap by eliminating
bridge permutations or improving the load/decode dependency chain, but that is
a separate device measurement.  The abstraction itself has a zero-overhead,
no-regression requirement.

## AIU + UniversalCopy arrangement proof

Changing the copy atom does **not** by itself prove an offline format.  The
decision is made by the composed coordinate map, not by the instruction name.
For the concrete Q4 path, L240 and L244 enumerated these maps over one complete
transport tile.  PPU0010's m8 and m16 fp16 MMA atoms have the same `BLayout`,
so the offline map is not an M-tile choice:

1. `O(n,k,plane) -> byte` -- the versioned offline arrangement.
2. `G(byte,lane,stage) -> smem` -- the AIU global-to-shared delivery.
3. `R(smem,lane,slot) -> register` -- the shared-to-register copy.
4. `F(n,k,plane,lane) -> mma_fragment` -- the fragment expected by the
   converter and MMA.

The existing Q4 arrangement would have been reusable iff
`R(G(O(...))) == F(...)` were a bijection over every plane and transport-tile
edge.  That equality is false for Q4 K-pack4 layout 1.  The resulting direct
map is therefore versioned separately as `q4-n16k64-direct`:

```
logical atom       N16 x K64 Q4
physical tensor    [K/16][2*N] uint32, N-word contiguous
atom bit map       p[0..9] = {k3,k4,k0,k5,n3,k1,k2,n0,n1,n2}
layout             3
mapping id          0x51344e3136440001
```

This derivation is specific to Q4.  Q2/Q3/Q5/Q6 use different converter
widths, and Q3/Q5/Q6 also have a high-plane composition whose register source
order participates in `R` and `F`.  Each therefore needs a new complete
enumeration and a new mapping identity; changing only the bit count in this
formula would not be a proof.

The proof must still report global transaction alignment and shared bank
multiplicity; a bit-correct gather that destroys coalescing is not an
admissible performance candidate.

Run the investigation in this order:

1. **Closed locally:** L239--L245 bind the AIU-plain writer, UniversalCopy
   reader, converter emission and real m8 MMA-B fragment; layout-1 reuse is a
   planted RED, and layout 3 has independent prepare/recover oracles.
2. **Built for device admission:** L248 requires exactly four AIU-plain writes
   and sixteen `tsm.ld.b32x4` reads and rejects any swizzled instruction.  Run
   its exact binary on PPU and require the poisoned raw-bit denominator to be
   completely overwritten.
3. **Built for device admission:** L254 applies the real converter into the
   actual m8 MMA-B owner over `N64 x K64`.  It requires every logical fragment
   value to equal the independent `code - 8` fp16 oracle and plants both
   layout-1 and DeepGEMM's native N64xK16 permutation as RED controls.
4. Repeat dense and grouped correctness, then an isomorphic performance A/B
   with identical tile, stage, provider, grid and numerical fixture.
5. Do not select layout 3 in `auto`, change the canonical mapping ID, or feed
   its bytes to the shipping reader before all four steps close.

## DeepGEMM comparison anchor

The local comparison is pinned to DeepGEMM-for-sail commit
`f89eae10c0e90c20630b50e4314448f01321bfba`, file
`deep_gemm/include/deep_gemm/w4a16_gemm_cutlass3.cuh`.

Its W4A16 B path is not “plain copy all the way”:

- global-to-shared still comes from `DefaultGemm_AIU_Operand` and its
  `GmemTiledCopyB`;
- shared-to-register is a custom
  `Copy_Atom<UniversalCopy<uint128_t>, int>`;
- the global B tensor is viewed as `(K/16, 2*N)` `int32` with stride
  `(2*N, 1)`, while the shared tile is `(BlockK/16, 2*BlockN)`;
- the copied integer registers are dequantized and scaled before MMA.

Its producer's `_perm` has a minimum `N64 x K16` permutation atom.  L254 uses
the common complete `N64 x K64` domain rather than folding either axis.  In
low-to-high physical-nibble bit order, DeepGEMM is
`{k3,n3,k0,n4,n5,k1,k2,n0,n1,n2,k4,k5}`, while layout 3 is
`{k3,k4,k0,k5,n3,k1,k2,n0,n4,n5,n1,n2}`.  The maps select the same physical
coordinate for only 64 of 4096 logical codes.  Thus the common outer tensor
shape does not make the offline bytes interchangeable.  This difference comes
from DeepGEMM's hand-built register/dequant view, not from m16 MMA itself: our
m16 path uses the same CuTe MMA-B owner as m8 and therefore retains layout 3.
DeepGEMM remains a reader-structure reference and a deliberate
wrong-permutation control; its packed bytes are not a Q4_K correctness oracle.
