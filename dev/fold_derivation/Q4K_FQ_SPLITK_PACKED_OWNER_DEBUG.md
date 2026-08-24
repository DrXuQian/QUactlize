# Q4_K fully-quantized Split-K packed-metadata publisher debug

## Frozen row

```text
shape       M/N/K=1/1024/5120, group_size=32
artifact    qtype=12 Q4_K, ArtifactTileK=64, bchunk=0
tactic      TM/TN/TK=8/64/256, WM/WN=8/16, stages=2
providers   AP0 standard-aiu, AP1 packed-row
splits      S=2 and S=4 (S=8 is arithmetically inadmissible)
symbols     fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap{0,1}
legacy sha  43fb02ba7c0d6340f21d8c1108941758815a25b5
```

The independent host oracle compares every fp32 partial plane before the
reducer.  The current device census is:

```text
AP0 S2  partial_value_raw_bad=288  first index=416 plane=0  25 -> 20
AP0 S4  partial_value_raw_bad=704  first index=288 plane=3   0 -> -6
AP1 S2  partial_value_raw_bad=0    direct-only intermittent failure
AP1 S4  partial_value_raw_bad=32   first index=160 plane=0   3 -> -2
```

Canaries are intact.  The bad counts are 9, 22 and 1 complete 32-output
stripes.  All first indices satisfy `index % TileN == 32`; this is the start of
the second metadata-decoder warp, not an arbitrary epilogue coordinate.

## Real CuTe ownership result

The exact tactic has 128 CTA threads but the packed raw-byte copy has 64 CuTe
thread slices.  Production maps every physical thread through
`get_slice(thread % 64)`:

```text
legacy:     1024 byte destinations, 2048 visits, every byte has 2 publishers
owner-only: 1024 byte destinations, 1024 visits, every byte has 1 publisher
```

The legacy second publisher overlaps every byte read by the 64 decoder owner
threads.  In particular, warp 3 republishes columns 32..63 while decoder warp
1 reads those columns after only its own per-thread `cp_async_wait`.  The CTA
barrier comes after decode, so it cannot order that overlap.  The executable
oracle is `l221_packed_metadata_publishers.cu`; its modulo-all arm is the exact
duplicate-owner negative.

This does not yet prove that the physical overlap causes the PPU value
corruption.  `PPU_PACKED_METADATA_OWNER_ONLY=1` is a factorial device arm that
changes only whether surplus physical threads issue the packed-metadata copy.
The legacy behavior remains the default until the device arm adjudicates it.

## Performance invariants for the candidate

Unchanged:

- A provider and its global/shared/register traffic;
- quantized B load, conversion, affine application and MMA count/order;
- packed metadata coordinates, bytes per logical column and decode arithmetic;
- pipeline depth, waits, fences and CTA barriers;
- accumulator and fp32 partial-workspace epilogue;
- grid, scheduler and reducer.

Changed only for `CTA threads > packed owner threads`:

- packed metadata publishers go from `CTA/owners` to one per destination;
- this exact row drops redundant raw metadata cp.async operations from 128 to
  64 per k-stage;
- one warp-uniform owner predicate is introduced; no intra-warp divergence and
  no new synchronization are added.

The optional `PPU_PACKED_SPLIT_GROUPS` experiment deliberately consumes the
duplicate owner set and is compile-time incompatible with this diagnostic.

## Device verdicts

The next one-build diagnostic reuses the hash-checked 43fb02b legacy artifact.
It emits one of:

- `OWNER_RACE_CLOSED_ALL_EXACT`;
- `OWNER_RACE_CLOSED_DIRECT_GAP_REMAINS`;
- `OWNER_ONLY_REFUTED`;
- `UNADJUDICATED_LEGACY_DID_NOT_REPRODUCE`.

Only a candidate with exact fp32 partial planes proceeds to production
closure.  Production closure must then rebuild legacy and candidate from one
SHA and report raw-bit correctness, latency, registers and spills.
