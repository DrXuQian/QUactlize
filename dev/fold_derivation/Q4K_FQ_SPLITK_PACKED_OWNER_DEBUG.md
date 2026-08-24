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

The post-mainloop factorial used the same standard-A/packed-A mainloop in both
arms.  Its first diagnostic run kept owner-only metadata to hold the preceding
factorial constant.  The control used the shared
R2S/barrier/S2R/vectorized partial epilogue; the candidate mapped the completed
accumulator directly through the production TiledMma `partition_C` view.  The
local exact-ownership oracle and its duplicate-thread and rotated-fragment
negatives are in `l222_fq_splitk_direct_accumulator_store.cu`.

The PPU box's `nvcc` delegates device preprocessing to `ppu_clang++`, enables
the HGGC FP8 include path, and cannot execute this NVIDIA/stub host oracle
without mixing incompatible SDK headers.  The box runner therefore validates
the exact committed L222 output from the result SHA instead of inventing an
`hggc_fp8.h` stub.  Both device arms are still built fresh with `hgcc`.

The hash-bound two-build diagnostic emitted:

```text
source sha  bc31c2e33c528a19e9987008ca68db88293515df
shared AP0/S4   5/256 bad samples, 160 bad fp32 partial values,
                plane mask=0xc, stripe origin=32
shared other    AP0/S2 and AP1/S2/S4 exact in this census
direct AP0/S2   0/256 bad samples, partials exact
direct AP0/S4   0/256 bad samples, partials exact
direct AP1/S2   0/256 bad samples, partials exact
direct AP1/S4   0/256 bad samples, partials exact
verdict         PARTIAL_EPILOGUE_CORRUPTION_CONFIRMED
```

This closes the causal boundary.  The completed mainloop accumulator is exact;
the reducer, workspace addressing and cross-kernel publication are not the
source.  Corruption enters only while the old partial epilogue redistributes
the fp32 fragment register-to-shared, synchronizes, reloads shared-to-register,
and writes global.  The intermittent complete 32-output stripes and changing
plane mask are an ordering/visibility signature, not a static CuTe coordinate
permutation.

The production repair does not need the obsolete handoff, but the backend
micro-root was subsequently investigated rather than inferred from that
bypass.  The exact L223 production-type CuTe oracle proves 512 R2S writers and
512 S2R readers, each exact-once, with no coordinate or value permutation.  Its
R2S-rotation negative produces 512 coordinate/value mismatches; its S2R-owner
negative produces 256 holes and 256 duplicates.

Two hash-bound device factorials then constructively excluded the first set of
dynamic hypotheses:

- legacy integer barrier 0 (effective hardware ID 6), its exact clone,
  reserved epilogue barrier ID 1 and full CTA barriers all retained the
  intermittent corruption;
- a pre-R2S CTA barrier, physically disjoint mainloop/epilogue shared storage,
  identity conversion, scalar R2S, scalar S2R and scalar R2S+S2R all retained
  it;
- most importantly, executing the shared round trip, discarding its result,
  and then publishing the original accumulator through the proven direct map
  still failed.

The last result is not proof that the mainloop produced a bad accumulator.  In
that arm, unused S2R/conversion values can be dead-code eliminated while R2S
shared writes, barriers and their compiler footprint remain.  Its honest
verdict is therefore `ACCUMULATOR_OR_COMPILER_FOOTPRINT_REMAINS`.

The next and narrower experiment is
`run_fq_q4k_split_shared_prefix_root_box.sh`.  It compiles a separate binary
for each prefix before the correct direct store: opaque accumulator liveness,
an extra register clone, CTA-only synchronization, exact-once flat shared
stores from constants or live accumulators, vector/scalar/snapshotted R2S, and
vector/scalar live S2R readback.  All data-path arms use disjoint shared
storage; the complete historical discard arm is rebuilt as the negative.
Every binary also records SDK `hgobjdump` register and spill usage.  This
separates a backend register-footprint threshold from a generic shared store,
a live-accumulator-to-shared dependency, CuTe `retile_S`/scatter lowering, and
S2R readback without changing the production hot path.

## Production repair and invariants

The fixed Split-K producer now stores each completed accumulator fragment
directly through the exact `tiled_mma.get_thread_slice(thread_idx).partition_C`
mapping.  `PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE=1` retains the old path as
the hash-bound negative control.  The default path:

- preserves the mainloop, MMA count/order, scheduler and split partition;
- preserves the split-major fp32 workspace bytes and deterministic reducer ABI;
- preserves every output's unique CuTe owner, with residue predication;
- performs no fp16 conversion or output-layout redistribution;
- deletes the shared R2S/S2R traffic and two CTA barriers;
- adds no fence, counter, atomic operation or synchronization;
- does **not** enable `PPU_PACKED_METADATA_OWNER_ONLY`.

The exact local ownership oracle visits all 576 test outputs once.  Its planted
bad-thread arm produces 432 holes and 144 duplicates; its rotated-fragment arm
produces 576 value mismatches.  Thus the direct map is not a scalar loop that
can accidentally pass a weak aggregate check.

The final box closure rebuilds both paths from one SHA with true shipping
metadata.  It requires high-repeat raw-bit exactness for the production arm and
times both paths in the same run.  The packed-row S4 producer is the performance
control; more than 3% regression is red.  This replaces the earlier requirement
to preserve an unnecessary shared epilogue merely to retain its cadence.

The semantic checker can still emit one of:

- `MAINLOOP_ACCUMULATOR_CORRUPTION_CONFIRMED`;
- `PARTIAL_EPILOGUE_CORRUPTION_CONFIRMED`;
- `DIRECT_STORE_NEGATIVE_CONTROL_FAILED`;
- `UNADJUDICATED_EPILOGUE_ARM_DID_NOT_REPRODUCE`.

Only `PARTIAL_EPILOGUE_CORRUPTION_CONFIRMED`, followed by a raw-bit exact and
non-regressing direct-store performance arm, admits the production repair.
