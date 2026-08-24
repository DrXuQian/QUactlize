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
stripes.  Those first three samples all started at `index % TileN == 32`, the
second metadata-decoder warp.  A later legacy repeat produced 64 bad outputs
starting at index 320: still exactly 32-stripe aligned, but local N=0.  Thus
the stable signature is the 32-output stripe origin, not one fixed TN64 half.
The earlier local-N=32 claim was a three-sample overfit.

## Real CuTe ownership result

The exact tactic has 128 CTA threads but the packed raw-byte copy has 64 CuTe
thread slices.  Production maps every physical thread through
`get_slice(thread % 64)`:

```text
legacy:     1024 byte destinations, 2048 visits, every byte has 2 publishers
owner-only: 1024 byte destinations, 1024 visits, every byte has 1 publisher
```

The legacy second publisher overlaps every byte read by the 64 decoder owner
threads: warp 2 republishes columns 0..31 while decoder warp 0 reads them, and
warp 3 does the same to columns 32..63 while decoder warp 1 reads them.  Each
decoder has waited only for its own copy; the CTA barrier comes after decode
and cannot order either overlap.  Which 32-column half loses is therefore a
scheduling outcome.  The executable oracle is
`l221_packed_metadata_publishers.cu`; its modulo-all arm is the exact
duplicate-owner negative.

This ownership result alone did not prove that the physical overlap caused the
PPU value corruption.  `PPU_PACKED_METADATA_OWNER_ONLY=1` was therefore kept as
a factorial device arm that changed only whether surplus physical threads
issued the packed-metadata copy.  The device verdict below refuted it as the
sole repair, so the shipping behavior remains the default.

## Performance invariants of the owner-only diagnostic

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

The owner-only device arm did **not** close the entire row.  It did, however,
separate the two A providers sharply:

```text
owner-only packed-row   S2  0/128 bad samples, partials exact
owner-only packed-row   S4  0/128 bad samples, partials exact
owner-only standard-A   S2  probe partials exact; one direct and one sync-only 32-output event
owner-only standard-A   S4 11/128 bad samples, 352 bad fp32 partial values
                            plane mask=0xd, every event one stripe at local N=32
```

Thus duplicate packed-metadata publishers were a cadence/contribution seam,
not the sole root cause.  The aggregate bad-sample rate changed from 28/256 to
11/512, but a production repair cannot retain a workaround that merely lowers
the probability.  `PPU_PACKED_METADATA_OWNER_ONLY` remains diagnostic-only.

The next factorial is strictly post-mainloop.  Both arms keep owner-only
metadata and the same standard-A/packed-A mainloop.  The control uses the
shipping shared R2S/barrier/S2R/vectorized partial epilogue; the candidate maps
the completed accumulator directly through the production TiledMma
`partition_C` view.  The local exact-ownership oracle and its duplicate-thread
and rotated-fragment negatives are in
`l222_fq_splitk_direct_accumulator_store.cu`.

The PPU box's `nvcc` delegates device preprocessing to `ppu_clang++`, enables
the HGGC FP8 include path, and cannot execute this NVIDIA/stub host oracle
without mixing incompatible SDK headers.  The box runner therefore validates
the exact committed L222 output from the result SHA instead of inventing an
`hggc_fp8.h` stub.  Both device arms are still built fresh with `hgcc`.

The hash-bound two-build diagnostic emits one of:

- `MAINLOOP_ACCUMULATOR_CORRUPTION_CONFIRMED`;
- `PARTIAL_EPILOGUE_CORRUPTION_CONFIRMED`;
- `DIRECT_STORE_NEGATIVE_CONTROL_FAILED`;
- `UNADJUDICATED_EPILOGUE_ARM_DID_NOT_REPRODUCE`.

Only that semantic verdict proceeds to the smallest production repair.
Production closure must then rebuild the historical and repaired path from one
SHA and report raw-bit correctness, latency, registers and spills.
