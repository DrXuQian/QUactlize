# Q4_K K-pack4 converter destination layout

## Exact incident

The first native K-pack4 performance pilot used:

```text
qtype=12  weight_layout=q4-kpack4-transpose-v1
mapping_id=0x51344b5034540001
shape=1x1024x5120  TM=8  TK=256  WM=8  AP=standard-aiu
typed_rows=72
```

Its S1 screen reported:

```text
MEASURED=37  RAW_FP16_MISMATCH=23  SHIPPING_SHARED_STORAGE=12
```

The 23 numeric failures were deterministic at repeat zero and exactly covered
four geometry groups:

```text
(TN,WN)=(32,32)   stages=2,3,4,6,8,12  raw_bad=512  first_bad=16
(TN,WN)=(64,32)   stages=2,3,4,6,8,12  raw_bad=512  first_bad=32
(TN,WN)=(64,64)   stages=2,3,4,6,8,12  raw_bad=1024
(TN,WN)=(128,64)  stages=2,3,4,6,8     raw_bad=512
```

The first wrong value was consistently `want=0x4e80, got=0xd100`.  Stage 12
for `(128,64)` and the remaining `SHIPPING_SHARED_STORAGE` rows were static
shared-memory capacity terminals, not numeric results.

## Root cause

K-pack4 transports a physical `(N,K/4)` b16 tile through an m16 transposed
loader and expands every four-bit code group into a logical `(N,K)` fp16 m8
MMA fragment.  One conversion iteration emits 32 logical fp16 values.  The
historical collective constructed the destination as:

```cpp
make_tensor(tCrB_mma(...).data(), cvt_in.layout())
```

This preserved the converter's correct 32-value mode-0 emission order, but it
also copied the physical loader's mode-1 N stride onto the logical compute
fragment.  Those rest-mode strides are not generally equal.

For `(TN,WN)=(32,32)`, for example:

```text
physical recast loader N stride = 32 halfs
logical compute fragment N stride = 128 halfs
legacy cohort bases             = 0,1,1,2,2,3,3,4,...
required cohort bases           = 0,4,1,5,2,6,3,7,...
```

Distinct `(N16,K64)` converter cohorts therefore overlapped in the raw MMA
register storage.  This is a destination placement defect after an exact
load and exact converter; it is not an offline-byte, metadata, stage, timing,
AIU publication or MMA defect.

Equal physical and logical cohort *sizes* were insufficient.  The missing
condition was equality of their rest-mode strides.

## Correct repair

Keep mode 0 from `cvt_in.layout()` because it is the proved converter emission
order.  Rebuild only the destination N mode as a compact layout rooted at the
compute fragment's own N stride:

```cpp
auto dst_n_stride = compact_col_major(
    shape<1>(cvt_in.layout()), stride<1>(tCrB_mma.layout()));
auto dst_layout = make_layout(
    shape(cvt_in.layout()),
    make_stride(stride<0>(cvt_in.layout()), dst_n_stride));
```

The register converter accepts separate static input/output layouts with equal
logical sizes.  Its emitted instructions and 32-value order are unchanged;
only each conversion iteration's compile-time destination base changes.

The exact historical negative is:

```text
PPU_Q4_KPACK4_LEGACY_LOADER_OUTPUT_LAYOUT=1
```

Do not replace this with a linear temporary/scatter or change the converter
order.  Neither is needed.

## CuTe proof

`dev/fold_derivation/l231_q4_kpack4_production_fragment.cu` composes the real:

- production K-pack4 physical `SmemLayoutB`;
- m16 transposed loader `TiledMMA` and `partition_fragment_B`;
- logical m8 compute `TiledMMA` and `partition_fragment_B`;
- four-bit converter emission map.

It enumerates all 12 admitted `(TN,WN)` groups.  The legacy mapping is
nonidentity for exactly the four device-failing groups and identity for the
other eight.  The compute-stride mapping is identity and bijective for all 12.
Two controls must turn red:

- rotate the 32-value destination emission by one;
- restore the loader N stride as the candidate destination.

Run it locally through:

```bash
bash dev/fold_derivation/run_l231_q4_kpack4_production_fragment.sh
```

This runner follows the nvcc/stub boundary in
`host-oracle-compiler-boundary.md`; never introduce a fake `hggc_fp8.h`.

## Device causal closure

The candidate and exact legacy arm closed on 2026-08-27 at commit
`604feb83c496672515efde2b97775f786a0ab615`:

```text
artifact=/workspace/quactlize-fq-q4k-kpack4-fragment-604feb83-20260827T044520Z-2657230
candidate=6/6 RAW-BIT/PASS
legacy=2/6 clean + 4/6 predicted RAW_FP16_MISMATCH
overlap=EXACT-L231
```

The six rows contained two identity controls and one stage-2 representative
from each failing geometry.  Both arms were built from the same source SHA;
the legacy arm differed only by the macro above.  The exact command is:

```bash
JOBS=16 CORRECTNESS_REPEATS=8 \
  bash tools/run_fq_q4k_kpack4_fragment_closure_box.sh
```

## Constructively excluded hypotheses

- **Pipeline timing or stale stage:** every failure occurred at repeat zero,
  persisted across stages, and grouped only by `(TN,WN)`.
- **Offline K-pack4 bytes:** prepare/recover fixture and mapping identity were
  byte-exact; passing and failing rows consumed the same artifact ABI.
- **Converter bit order:** the real converter emission map was exact within
  every 32-value cohort; rotating it makes the oracle red.
- **Metadata ownership:** failures followed fragment geometry and were already
  present in S1; packed metadata closure remained exact.
- **Split-K/reducer:** the failing screen was S1 `FULL_OUTPUT`.
- **Shared-memory capacity:** capacity terminals were separated from measured
  numeric rows.

## Scope scan

The syntactic pattern of aliasing MMA storage through `cvt_in.layout()` also
exists in generic, folded and two-plane collectives.  Do not mechanically
rewrite all of them:

- K-pack4 uniquely composes a physical m16 `(N,K/4)` loader with a logical m8
  `(N,K)` compute fragment, so its rest-mode mismatch is proved here.
- multi-delivery folded paths already use their own explicit artifact scatter;
- two-plane paths have independent delivery ratios and layout oracles;
- ordinary paths retain matching load/compute fragment contracts.

For any new specialization, compose its actual loader and compute
`partition_fragment_B` types and prove both cohort order and rest-mode stride.
A source-level match is a reason to audit, not evidence that the same repair is
correct.

## Performance boundary

The repair leaves unchanged:

- global/TSM load count and byte count;
- four-register converter count and code order;
- metadata operations;
- MMA count and order;
- barriers and scheduler;
- runtime branch count.

It changes only static register destination bases.  The six-row device closure
proves correctness, not latency.  Run the fresh 72-row pilot before claiming a
performance result; never resume the pre-fix artifact because its source
authority and numeric denominator are obsolete.
