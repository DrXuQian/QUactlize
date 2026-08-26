# Packed metadata ownership and initialization ordering

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
therefore published twice. More importantly, the two operations touching the
same fp16 shared tile use different warp maps:

```text
scale clear warp 0: all N, groups 0..3
scale clear warp 1: all N, groups 4..7
scale clear warp 2: duplicate of warp 0
scale clear warp 3: duplicate of warp 1

packed decode warp 0: N 0..31, all groups
packed decode warp 1: N 32..63, all groups
```

The historical order was `clear`, initial async prefetch, per-thread wait,
packed decode, then the first CTA barrier. A per-thread wait does not order an
ordinary clear performed by another warp. Actual CuTe `partition_D` plus the
packed-column ownership map gives the exact intersection census:

```text
CTA32 passing row:  same-warp pairs=1, cross-warp pairs=0
CTA128 failing row: same-warp pairs=2, cross-warp pairs=6
                    active-owner cross pairs=2
                    surplus-warp cross pairs=4
                    overlap per cross pair=128 metadata values
```

Consequently an allowed schedule can decode N0..31, execute one late clear,
then decode N32..63. Exactly one 32-column half loses four scale groups. This
is the observed signature: intermittent, 32-aligned `raw_bad=32`, a small
fixed contribution delta, and a wrong producer FP32 partial. The same custom
kernel also failed at S=1, excluding Split-K partitioning and the reducer as
necessary causes.

The exact passing contrast is `WM/WN=8/64`: its tiled MMA has one 32-thread
warp and 32 logical metadata owners, so the old wrapped protocol happens to be
exact. Local L114 locks both geometries:

- `TileN=64, groups=8, CTA=32`: owner-only and legacy-wrapped each make 512
  visits, one per destination;
- `TileN=64, groups=8, CTA=128`: owner-only makes 512 visits, while
  legacy-wrapped makes 1024 visits with 512 duplicate destinations.

An earlier `owner-only` experiment guarded only the packed raw async copy. It
did not guard the ordinary scale/zero copies or the initial `clear(tSsS)`, and
it added no clear-to-decode edge. Its dirty result therefore did not test the
complete contract. Removing surplus publishers eliminates four cross-warp
pairs, but the two active-owner cross pairs remain while the separate fp16
clear exists. Exact ownership alone cannot order two different writer maps.

The conservative device-closed repair added a clear-to-decode CTA edge. The
shipping candidate now removes the second writer map entirely. Its complete
contract is:

1. construct an in-range logical scale slice and packed-column slice for every
   physical thread;
2. only the ordinary fp16 path uses `ScaleCopyPlan` to clear/copy scale/zero;
3. only `PackedMetadataColumnOwnership::owns_physical_thread()` may issue the
   packed raw copy;
4. the same packed-column owner writes every destination group: decode a valid
   N column, or write zero to an invalid tail column without reading raw bytes;
5. perform no packed-path fp16 clear and no prefetch-before barrier;
6. keep the existing post-decode CTA barrier that publishes decoded metadata
   to all MMA consumers.

## Required controls

- Legacy one-column ownership on `TileN=64, threads=32` must report exactly
  `copy_missing=32`, `first_copy_missing=32`, and `unowned_reads=32`.
- Correct ownership must cover `TileN/threads` pairs 32/32, 64/32, 128/32,
  64/64, 64/128, and 128/256 exactly once with no unowned reads.
- Legacy `TileN=64, threads=128` must report exactly 64 duplicate packed-column
  publishers; production must report zero.
- The Q4/A64 scale-copy oracle must keep CTA32 legacy-wrapped exact and CTA128
  legacy-wrapped red with two visits per destination.
- The same actual-CuTe oracle must report zero cross-warp clear/decode pairs
  for CTA32 and exactly six for CTA128: two active-owner plus four surplus,
  each intersecting on 128 metadata values.
- A partial-N residue must copy all and only live metadata bytes.
- Every partial-N destination `(n,group)` must still have exactly one packed
  decode-owner write; a missing-tail-zero plant must fail.
- Both standard-A and packed-row-A rows must pass; changing A cannot repair a
  common metadata defect.
- The fix must not prune WN. The m8 selector is only `TM=8 && WM=8`; WN remains
  governed by generic topology/resource constraints.

The local implementation gates are
`dev/fold_derivation/run_l217_packed_metadata_ownership.sh`, L114, and
`ci/check_fq_splitk_partial_path.py`. The narrow device closure is driven by
`tools/run_fq_q4k_tm8_wn64_closure_box.sh`: four WN64 ownership controls plus
the AP0/AP1 WN16/stage2 root-cause rows at aligned N=1024. The resident xplane
ABI requires N%256==0 and admitted TileN divides 256, so L217—not an illegal
device fixture—closes the future N-tail destination contract.

## Device causal closure

The exact legacy-versus-production A/B closed on 2026-08-26 at commit
`265033f5483f995cadb4ccda3b28a5af8a23d7b4`:

```text
artifact=/workspace/quactlize-fq-q4k-a-stage-root-265033f5-20260826T002715Z-2904796
verdict=PACKED_METADATA_CLEAR_DECODE_RACE_CLOSED
baseline_failure_attempts=2/2
selection=exact-metadata-publication
clean_candidates=exact-metadata-publication
shipping_s1_clean=4/4
```

The negative reconstructed all-thread modulo publishers and omitted the
initialization edge. The candidate used exact ownership plus the one-time
prefetch-before CTA edge and closed every custom S1/S2/S4 cell. This proves the
root cause and conservative repair. It does not by itself validate the later
total-overwrite cadence; that candidate needs a fresh narrow box closure.

After closure, the factorial runner, repeat-state controls, partial-plane
failure probes, stale-A/issuer oracle and counterfactual compile macros were
deleted. Retain only the production contract, local CuTe ownership gates, this
record, and the exact legacy macro
`PPU_MIXED_LEGACY_MODULO_METADATA_PUBLISHERS`.

## Performance boundary

For every full N tile, the packed path emits no initial clear or extra barrier;
the already-required decoder stores are the complete destination write. A tail
adds zero stores only for invalid columns owned by the same decoder threads.
This removes rather than adds work on the common aligned shapes, but device
timing and raw-bit closure are still required because compiler scheduling has
already mattered for this incident.
