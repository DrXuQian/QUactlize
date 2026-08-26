# Q4_K fully-quantized packed-metadata race

Status: **root cause and decode-owner total overwrite raw-bit closed on
device; performance comparison pending**. The conservative repair is commit
`7124998`; its source/CuTe proof and exact device A/B are commit `265033f`.
The later decode-owner total-overwrite simplification removes that repair's
initial clear/barrier. Its implementation is commit `d557509`; the runner's
current-ABI correction is `3f24f8d`.

The closing artifact is:

```text
/workspace/quactlize-fq-q4k-a-stage-root-265033f5-20260826T002715Z-2904796
```

Its adjudicated result was:

```text
verdict=PACKED_METADATA_CLEAR_DECODE_RACE_CLOSED
baseline_failure_attempts=2/2
selection=exact-metadata-publication
clean_candidates=exact-metadata-publication
shipping_s1_clean=4/4
```

The must-red arm reconstructed the historical all-thread modulo publishers and
omitted the initialization edge. The candidate used the production exact
ownership plus one pre-prefetch CTA edge. The historical arm reproduced in
both attempts; the candidate was raw-bit exact for every custom S1/S2/S4 cell.

The later total-overwrite implementation closed on the current production
domain at:

```text
/workspace/quactlize-fq-q4k-tm8-wn64-closure-d557509e-20260826T021632Z-2254120
```

The hash-bound six-tactic checker reported:

```text
[fq-tm8-wn64-check] PASS shapes=1 tactics=6 census_rows=24 raw_bad=0
WN64-control=4 WN16-S1/S2/S4=6 tail=LOCAL-L217/CURRENT-ABI-N%256
```

Thus both A providers and the WN16 root-cause S1/S2/S4 cells are raw-bit
exact after removing the clear/barrier. The attempted N=992 fixture was
correctly classified as infrastructure, not a device result: the resident
xplane ABI requires N%256==0. L217 remains the tail contract authority.

## Root cause

The failing Q4_K/A64 row was:

```text
M/N/K       1/1024/5120
tile        8x64x256
warp        8x16
stages      2
CTA         128 threads
providers   standard-aiu and packed-row
```

The fp16 metadata clear and packed scale/zero decoder touched the same shared
tile through different physical-warp maps. Historical code also wrapped 128
physical threads onto 64 logical metadata owners.

Actual CuTe `ScaleCopyPlan::partition_D` mapped the clear as:

```text
warp 0: N0..63, groups 0..3
warp 1: N0..63, groups 4..7
warp 2: duplicate of warp 0
warp 3: duplicate of warp 1
```

Packed decode mapped work as:

```text
warp 0: N0..31, groups 0..7
warp 1: N32..63, groups 0..7
```

The old order was:

```text
clear fp16 metadata
initial async A/B/raw-metadata prefetch
per-thread async wait
packed decode
first CTA barrier
```

A per-thread async wait does not order an ordinary shared-memory clear issued
by another warp. The first CTA edge came after decode, so a late clear could
erase one decoded 32-column half. The MMA then consumed the cleared scale
values and wrote an already-wrong FP32 producer plane.

The L114 actual-CuTe composition gives the decisive census:

```text
CTA32  PASS: same-warp clear/decode pairs=1, cross-warp pairs=0
CTA128 FAIL: same-warp pairs=2, cross-warp pairs=6
             active-owner cross pairs=2
             surplus-warp cross pairs=4
             128 metadata values per cross pair
```

This exactly predicts the intermittent 32-aligned output stripe. Exact owner
filtering removes the four surplus-warp pairs, but two active-owner cross-warp
pairs remain; therefore ownership filtering alone is insufficient.

## Repair history and current contract

The device-closed conservative repair made the clear exact-owner-only and
published it with one pre-prefetch CTA barrier. That proved the root cause, but
retained two writer maps for one destination tile.

Both one-plane and two-plane collectives now use the simpler production
candidate contract:

1. the ordinary fp16-metadata path keeps its exact-owner clear/copy;
2. only the exact packed-column owner issues the packed raw copy;
3. in the packed path that same column owner writes the complete fp16
   destination column: valid N columns are decoded, invalid tail columns are
   explicitly zeroed;
4. the packed path performs no fp16 clear and no pre-prefetch CTA barrier;
5. the existing post-decode CTA barrier remains the publication edge for MMA
   consumers.

Thus every packed `(n, group, stage)` slot has one owner and one write. Full N
tiles execute only decode stores; a tail executes zero stores only for skipped
columns and never reads their unfilled raw slots. The old single-warp safety no
longer depends on warp lockstep. No accumulation order, stage-ring order,
Split-K partition, reducer or epilogue math changed.

The exact historical negative is retained behind
`PPU_MIXED_LEGACY_MODULO_METADATA_PUBLISHERS`. All other counterfactual macros,
factorial runners, repeat-state controls and failure-only partial-plane probes
were removed after closure.

The two retained negatives remain build-reachable through the ordinary target
when an exact generated registry is supplied:

```bash
PPU_DEFS=PPU_MIXED_LEGACY_MODULO_METADATA_PUBLISHERS=1 \
TARGET=test_fully_quantized_internal_sweep ./build.sh

PPU_DEFS=PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE=1 \
TARGET=test_fully_quantized_internal_sweep ./build.sh
```

## Why it looked Split-K-specific

The first reliable failures appeared in S2/S4 producer runs because changing
the K interval and output path changed kernel cadence and code generation. A
later same-custom-kernel diagnostic also reproduced at runtime S1. Split-K
planes are disjoint and the reducer was not required for failure: the wrong
FP32 value was already present in one producer plane.

Shipping S1 remained a useful control, but it is a different compiled kernel
with a product epilogue and cannot disprove a race in the common collective.

## Two real but independent defects found during the investigation

### Packed-A physical x4 footprint

The m8 CuTe A copy exposes two logical registers per lane, while the PPU0010
load instruction physically touches four. The old stage pitch described only
the logical x2 view and allowed hidden x4 reads to overlap another stage.

The repaired authority separates cube and stage pitch. For TK256/Stages2:

```text
cube_pitch  = 64 half
stage_pitch = 1216 half
old cross-stage physical intersections = 432
new cross-stage physical intersections = 0
```

L186 preserves this physical-footprint proof. The fix removed a genuine
fragility but did not close the metadata race.

### Split-K shared partial-output handoff

The historical partial epilogue unnecessarily moved a completed FP32
accumulator register-to-shared-to-register before writing the workspace. The
direct store uses the real `TiledMma::partition_C` ownership and was raw-bit
exact. It also avoided an independently observed shared-store backend/codegen
corruption trigger.

The direct store does not change accumulation or reduction order. Its measured
packed-row/S4 median was 11.300 us versus 11.320 us for the old shared path
(-0.177%, inside the preregistered 3% limit). The exact old path remains under
`PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE` as its one historical negative.

Neither independent repair alone closed the later metadata race.

## Refuted hypotheses

The retained evidence excludes these as the complete root:

- stale partial-workspace contents or overlapping host launches;
- producer-to-reducer publication;
- reducer plane addressing;
- A-provider choice;
- duplicate logical AIU issuers;
- A-before-B ordering and separate async groups;
- inline-asm memory clobbers;
- K iterator shape lifetime;
- packed-A stage pitch;
- shared partial epilogue and its synchronization policy.

Several of those arms changed failure incidence. That was schedule sensitivity,
not causal closure. The final diagnosis required composing the actual clear
and decode ownership maps and then testing the complete owner-plus-edge repair
against the exact legacy negative.

## Retained regression evidence

Long-lived local gates:

```bash
python3 -B ci/check_fq_splitk_partial_path.py
bash dev/fold_derivation/run_l217_packed_metadata_ownership.sh
python3 -B ci/local_gates.py -k l114_scale_copy_coverage --strict
bash dev/fold_derivation/run_l186_dense_m1_packed_a.sh
bash dev/fold_derivation/run_l223_fq_splitk_partial_abi.sh
```

The retained narrow closure at
`tools/run_fq_q4k_tm8_wn64_closure_box.sh` now covers four WN64 controls plus
the two WN16 root-cause tactics at aligned N=1024. The resident xplane ABI
requires N%256==0 and every admitted TileN divides 256, so a device N-tail is
outside the current production domain; L217 owns that future-ABI proof. The
large causal runner used for
the now-closed bug was deliberately deleted; its exact output and source remain
recoverable from commit `265033f` and the artifact path above. L217 additionally
plants a missing decode-owner tail zero and requires it to fail.

## Performance boundary

The packed full-tile path removes both the fp16 clear and the conservative
initialization-only CTA edge. Tail tiles replace the old whole-tile clear with
zero stores only for invalid owner columns. The six-tactic device run proves
raw-bit correctness of the new cadence, but correctness does not adjudicate
performance.

The d557509 run used 200 timing samples. Against the earlier same-shape/tactic
timing context in
`/workspace/quactlize-fq-q4k-partial-closure-219c15a-20260824T133319Z`
(contextual, not a hash-paired conservative-repair A/B), the medians are:

| provider | scope | old us | d557509 us | delta |
|---|---|---:|---:|---:|
| standard-aiu | S1 full output | 27.240 | 27.160 | -0.294% |
| standard-aiu | S2 producer | 16.240 | 16.280 | +0.246% |
| standard-aiu | S4 producer | 11.240 | 10.960 | -2.491% |
| packed-row | S1 full output | 28.800 | 29.280 | +1.667% |
| packed-row | S2 producer | 17.080 | 17.040 | -0.234% |
| packed-row | S4 producer | 11.300 | 11.240 | -0.531% |

All six absolute deltas are below the preregistered 3% boundary, so there is
no observed regression signal. Standard-aiu is lower-latency than packed-row
by 7.240%/4.460%/2.491% for S1/S2/S4 respectively.

Under the analysis policy requested for decode—reducer bandwidth at
`0.8 * 2766 GB/s`, launch time ignored, reading `S*M*N*4` partial bytes and
writing `M*N*2` fp16 output bytes—the reducer contributes 0.004628 us for S2
and 0.008330 us for S4. Modeled product times are therefore:

```text
standard-aiu S2  16.284627 us
standard-aiu S4  10.968329 us  <-- winner, 2.476x vs its S1
packed-row   S2  17.044627 us
packed-row   S4  11.248330 us  <-- 2.603x vs its S1
```

S2/S4 remain producer-only measurements; these product values are model
outputs, not measured reducer E2E.
