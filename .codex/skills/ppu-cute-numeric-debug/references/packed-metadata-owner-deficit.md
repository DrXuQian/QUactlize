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

## Required controls

- Legacy one-column ownership on `TileN=64, threads=32` must report exactly
  `copy_missing=32`, `first_copy_missing=32`, and `unowned_reads=32`.
- Correct ownership must cover `TileN/threads` pairs 32/32, 64/32, 128/32,
  64/64, 64/128, and 128/256 exactly once with no unowned reads.
- A partial-N residue must copy all and only live metadata bytes.
- Both standard-A and packed-row-A rows must pass; changing A cannot repair a
  common metadata defect.
- The fix must not prune WN. The m8 selector is only `TM=8 && WM=8`; WN remains
  governed by generic topology/resource constraints.

The local implementation gate is
`dev/fold_derivation/run_l217_packed_metadata_ownership.sh`. Device closure is
the four exact AP0/AP1 × stages3/4 rows driven by
`tools/run_fq_q4k_tm8_wn64_closure_box.sh`.

## Performance boundary

For `columns_per_owner == 1`, preserve the established copy branch exactly.
For a narrower CTA, copying and decoding the additional owner columns is
necessary work that was previously missing; do not add a barrier, change the
A provider, alter the MMA order, or remove the WN candidate. Device timing is
still required after raw-bit closure.
