# Packed metadata owner deficit

## Exact incident

The Q4_K fully-quantized decode row

```
M/N/K=1/1024/5120  A=64  TM/TN/TK=8/64/256
WM/WN=8/64  stages=3 or 4  AP=standard-aiu or packed-row
```

failed on PPU with `raw_bad=512` and `first_bad=32`. Both A providers failed,
so the common B/metadata path was the first shared seam.

The selected m8 tiled MMA has one warp: 32 CTA threads. The packed metadata
copy was declared with 64 thread slices, one per TileN column. Only slices
0..31 could execute, while the decoder read columns 0..63. Thus columns 32..63
were never written. Across N=1024 this gives exactly 512 bad outputs and the
first bad column 32.

## CuTe ownership rule

One packed metadata unit is one logical N column. Derive

```
owners = min(TileN, size(TiledMma))
columns_per_owner = TileN / owners
```

Keep the TiledCopy value tile equal to exactly one complete metadata unit.
CuTe places additional columns in the partition rest mode. For 32 owners and
64 columns, owner `t` receives columns `t` and `t+32`.

Do not widen the value tile to two adjacent units. An actual-CuTe oracle showed
that this crosses the logical column boundary inside the atom and then repeats
the traversal in the rest mode, duplicating coverage. The correct mapping must
come from `partition_S(identity_tensor)` and be shared by copy and decode.

For a partial N tile, predicate every column atom in the rest mode. A single
predicate taken from the first column is valid only when
`columns_per_owner == 1`.

## Surplus-publisher complement

A later Q4_K row exposed the opposite side of the same ownership contract:

```
M/N/K=1/1024/5120  A=64  TM/TN/TK=8/64/256
WM/WN=8/16  stages=2  CTA threads=128
```

For Q4_K, both the fp16 metadata plan and packed-column plan have 64 logical
owners. The historical code constructed slices as `thread_idx % 64` and then
let all 128 physical threads issue the operation. Every shared destination was
therefore published twice. Failures were intermittent, but each event changed
exactly one 32-output stripe and the producer FP32 partial was already wrong.
The same custom kernel also failed at S=1, excluding Split-K partitioning and
the reducer as necessary causes.

The exact passing contrast is `WM/WN=8/64`: its tiled MMA has one 32-thread
warp and 32 logical metadata owners, so the old wrapped protocol happens to be
exact. Local L114 locks both geometries:

- `TileN=64, groups=8, CTA=32`: owner-only and legacy-wrapped each make 512
  visits, one per destination;
- `TileN=64, groups=8, CTA=128`: owner-only makes 512 visits, while
  legacy-wrapped makes 1024 visits with 512 duplicate destinations.

An earlier `owner-only` experiment guarded only the packed raw async copy. It
did not guard the ordinary scale/zero copies or the initial `clear(tSsS)`, so
its dirty result was not a valid refutation of exact-once publication. Packed
decode writes the fp16 tile by column owner while initialization clears it by
scale-copy owner; on a multi-warp CTA those two maps also need one publication
edge after initialization. Otherwise a late clear may overwrite a decoded
scale even after raw-copy duplication has been removed.

The complete contract is therefore:

1. construct an in-range logical scale slice and packed-column slice for every
   physical thread;
2. only `ScaleCopyPlan::owns_physical_thread()` may clear/copy the fp16
   scale/zero partition;
3. only `PackedMetadataColumnOwnership::owns_physical_thread()` may issue the
   packed raw copy;
4. publish multi-warp packed initialization once before raw copy/decode;
5. if a diagnostic splits one column's decode across surplus threads, add a
   pre-decode CTA edge because those threads no longer issue duplicate copies.

## Required controls

- Legacy one-column ownership on `TileN=64, threads=32` must report exactly
  `copy_missing=32`, `first_copy_missing=32`, and `unowned_reads=32`.
- Correct ownership must cover `TileN/threads` pairs 32/32, 64/32, 128/32,
  64/64, 64/128, and 128/256 exactly once with no unowned reads.
- Legacy `TileN=64, threads=128` must report exactly 64 duplicate packed-column
  publishers; production must report zero.
- The Q4/A64 scale-copy oracle must keep CTA32 legacy-wrapped exact and CTA128
  legacy-wrapped red with two visits per destination.
- A partial-N residue must copy all and only live metadata bytes.
- Both standard-A and packed-row-A rows must pass; changing A cannot repair a
  common metadata defect.
- The fix must not prune WN. The m8 selector is only `TM=8 && WM=8`; WN remains
  governed by generic topology/resource constraints.

The local implementation gates are
`dev/fold_derivation/run_l217_packed_metadata_ownership.sh`, L114, and
`ci/check_fq_splitk_partial_path.py`. The narrow-CTA device closure remains the
four exact AP0/AP1 × stages3/4 rows driven by
`tools/run_fq_q4k_tm8_wn64_closure_box.sh`; the surplus-publisher closure is
`FQ_A_STAGE_CANDIDATE=exact-metadata-publication` in
`tools/run_fq_q4k_custom_split_count_box.sh`.

## Performance boundary

For a one-warp CTA with no surplus publishers, ownership predicates are
compile-time true and no initialization barrier is emitted. For a narrower
CTA, copying and decoding additional owner columns is necessary work that was
previously missing. For a wider packed CTA, exact ownership removes redundant
work but adds one initialization-only CTA edge; it does not add a per-K-tile
barrier or change MMA order. Device timing is still required after raw-bit
closure.
