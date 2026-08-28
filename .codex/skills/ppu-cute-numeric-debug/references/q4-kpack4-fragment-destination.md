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

It changes only static register destination bases.  The fresh 72-row pilot
subsequently closed with 18 confirmed symbols:

```text
shape=1x1024x5120
winner=TC_SPLITK_S4_MODELED_E2E
config=8x32x256_w8x16_s3_bc0_apstandard-aiu
producer=10.800000280 us
modeled_reducer=0.008329718 us
modeled_e2e=10.808329998 us
```

This is 1.123% slower than the prior cross-format decode winner at
10.688330016 us and 26.215% faster than the native A32/F2 decode result at
14.648329 us.  The S4 winner and runner had overlapping sample envelopes
(`gap=0.740%`), so the tactic verdict is correctly
`UNRESOLVED_OVERLAPPING_ENVELOPES`; this is not a correctness or physical
layout failure.  S8 was structurally unavailable for this K/TK partition.

Do not resume the pre-fix artifact: its source authority and numeric
denominator are obsolete.  This one-shape result admits the layout to the
inventory-owned M=1/2/4/8 real-shape decode scan; it does not establish
prefill performance.

## Current provider denominator

The 72-row counts above are intentionally frozen historical AP0 evidence.  The
current K-pack4 TM8 performance authority is 144/918: the same 72 tactic
geometries are instantiated for both `standard-aiu` (AP0) and `packed-row`
(AP1).  AP1 eligibility is derived from the layout-neutral A64 topology; it is
independent of the K-pack4 resident-B byte class.  New performance comparisons
must match AP across layouts and must not compare an Xplane AP1 winner with a
K-pack4 AP0-only result.

## PPU codegen reporting boundary

The PPU0010 source atoms spell the readers as
`ppu.tc01.ldmatrix...m8n8.x4.swzl...` and
`ppu.tc01.ldmatrix...m16n16.x1.swzl.trans...`, but `hgobjdump -line` normally
reports both after assembler lowering as backend `tsm.ld...` operations.  Do
not require a source `ldmatrix` mnemonic in final-object disassembly.  Bind
the K-pack4 reader identity through the retained
`KernelAiuQ4KPack4Transpose` kernel type.  Packed-A's schedule wrapper is not
a retained final-producer template parameter on every SDK, so bind AP0/AP1
instead through the exact one-row generated macro, the target compile ABI and
the runtime `FQ_TC_CELL provider` field.  Do not infer AP identity from a
missing demangled wrapper.  Then compare final instruction/resource counts
through the lowered `tsm.ld`, MMA, register, spill and ACU evidence.  If a
tool release preserves source mnemonics, their expected m8/m16 forms remain an
additional positive check.

## Resident-delivery bank-conflict follow-up

The fixed AP0/AP1 Xplane-versus-K-pack4 comparison later showed that K-pack4's
remaining N-wide regression is not extra load volume.  Shared Load instruction,
request and transaction counts were unchanged, but bank conflicts rose from
`344064` to `516096` (`+50%`).  Shared Store conflicts stayed `98304`, and all
other conflict classes stayed zero.  Therefore the complete extra-conflict
denominator is on the opaque TSM read side.

The resident row pitch was a plausible mechanism, but the matched device A/B
refuted it as the explanation for the counted conflicts.  D32 improved the
frozen row by about 6.2% (AP0) and 6.6% (AP1), while the Shared Load conflict
counter did not provide the predicted fall.  The same run reduced `No
Eligible` and warp cycles per instruction.  Treat D32 as a scheduler/
scoreboard critical-path improvement, not as the bank-conflict repair.  The
earlier 64-b16 pad was still non-diagnostic: it added a complete 128 B bank
period and could not change bank phase.

Do not add a CuTe-only XOR to this opaque AIU/TSM path and claim a fix.  The
repository's scale-plane XOR changed logical placement but measured no change
in Shared Load conflicts.  Instead change the matched writer and reader atom
together while preserving the complete stage:

- `auto64`: one N64/Kphys64 cube, 128 B row pitch;
- `D32`: two N32/Kphys64 cubes, 64 B row pitch;
- `D16`: four N16/Kphys64 cubes, 32 B row pitch.

The delivery value is a compile-time cap carried by
`KernelAiuQ4KPack4Transpose`, not an offline-layout field.  A smaller tactic N
resolves to itself.  The offline mapping ID, total shared bytes, converter
destination, MMA fragment, metadata, barriers and split workspace remain
identical.  L229 binds these type/storage invariants; L231 proves candidate
fragment identity for all 12 admitted geometries at D32 and D16.

Use the following device closure, which fixes shape, tactic, provider and S4,
runs cyclic `auto64/D32/D16` timing, and captures all six AP0/AP1 ACU reports:

```bash
JOBS=16 PERF_ITERATIONS=201 PERF_ROUNDS=3 CORRECTNESS_REPEATS=64 RUN_ACU=1 \
  bash tools/run_fq_q4k_kpack4_delivery_ab_box.sh
```

The box runner does not execute L229/L231 through its PPU-delegating `nvcc`.
It reads the hash-bound host evidence from its own result SHA, validates the
two positive delivery denominators and two RED controls, prints
`fresh_box_execution=0`, and only then builds the six shipping binaries with
fresh `hgcc`.  A direct box-host oracle is a compiler-boundary violation, not
a stronger proof.

Raw-bit equality is an admission gate.  The completed result did not satisfy
the 128 B phase hypothesis.  Choose D32 per tactic/provider from full kernel
time; do not cite the conflict model as its cause.

## Fused scale/zero store closure

Q4_K's packed decoder has a byte-neutral interleaved `(scale, zero)` shared
representation behind `PPU_PACKED_SCALE_FUSED`.  It fuses the two fp16
metadata stores into one packed word while preserving the established CuTe
reload path.

The frozen-row D32 device factorial admitted that fused-store path and rejected
the experimental one-word reload.  On AP1, balanced four-round timing improved
from `16.880000010` to `16.240000725 us` (`-3.791465%`), every paired delta was
negative, instructions fell `1703 -> 1660`, and TSM loads fell `67 -> 47`.
AP0 improved by `-2.296812%`.  Shared storage increased by 16 bytes and split
workspace did not change.  The extra reload experiment was `+14.647371%` on
AP0 and `+16.995065%` on AP1 versus fused-store, while its Shared Load bank
conflict count was unchanged.  It had no causal or performance value and was
deleted rather than retained as a dormant switch.

L232 now binds the retained shipping-auto AP0/AP1 fused-store word/half CuTe
maps, zero-plane overlay, allocation coverage, and raw packed-half identity.
ACU Duration from one profiled launch is diagnostic; the cyclic balanced
timing is the product-performance authority.

The deployment closure must not reuse the frozen one-shape row.  Sweep all 20
inventory-owned decode shapes with an exact matched factorial:

- five `(N,K)` families times `M={1,2,4,8}`;
- one 144-row K-pack4 manifest: 72 AP0 plus 72 packed-A AP1 tactics;
- the default auto64 delivery policy in both binaries;
- full raw-bit screen for both arms, then a union shortlist;
- scheduler and seven-sample confirmation on the same arm-union denominator.

Packed-A has capacity one row.  It is a real candidate at M=1 and an explicit
`PACKED_A_DECODE_ONLY_M_NOT_1` terminal at M=2/4/8.  Never delete those rows
from the denominator or compare an AP0-only K-pack4 board to an Xplane AP1
winner.

Do not promote the frozen D32 result globally.  An accidental full-D32
plain/store scan subsequently lost all 20 plain-arm median comparisons to the
archived Xplane baseline; its five per-family worst regrets were 57.7%--96.8%.
That is a direct RED control showing that one TN64 delivery win does not
transfer across tactics or M.  Delivery is a separate per-tactic search axis;
the fused-store deployment A/B must reproduce the original auto64 baseline and
change only the store macro.

```bash
INTERNAL_SWEEP_SPEC=/path/to/complete/inventory.json \
JOBS=16 FQ_CONFIGS_PER_UNIT=4 MATERIAL_THRESHOLD=0.02 \
  bash tools/run_fq_q4k_kpack4_fused_store_real_shapes_box.sh
```

The packed arithmetic remains the fast path.  Q4_K uses
`group_pair_of_words` (paired half2 sub/FMA) and the int4 converter's packed
LOP3/sub/FMA sequence; a high FMA count is expected for affine dequantization
and is not evidence that scalar dequant was selected.

## Prefill measurement boundary

No completed K-pack4 prefill timing existed before 2026-08-28.  Do not
mislabel either of these earlier results as K-pack4 prefill:

- the inventory-owned K-pack4 versus Xplane comparison covers only
  `M={1,2,4,8}` decode; K-pack4 won 12 of 20 median comparisons there;
- the `4096x5120x8192` result with `A32/native-F2=2663.840055 us` is a
  ScaleFirst folded-Xplane experiment, not the K-pack4 mapping.

The first registered K-pack4 prefill pilot deliberately admits one real
inventory shape before opening the full 15-shape denominator:

- shape `M=2048,N=1024,K=5120`;
- mapping `q4-kpack4-transpose-v1`, ID `0x51344b5034540001`;
- the complete TM64 denominator: 210 AP0 tactics out of 918 source-typed rows;
- all admitted tactics have TK256; packed-A/AP1 remains decode-only and is not
  silently borrowed at M=2048;
- shipping-auto delivery and plain metadata, so neither global D32 nor the
  fused-store experiment can confound the first result;
- full S1 raw-bit screen, S1/S2/S4/S8 scheduler census, and seven-sample
  confirmation; S>1 is ranked only after adding the registered zero-launch,
  80%-HBM reducer model.

The runner requires the complete inventory-v2 JSON and proves that the target
shape belongs to the registered 15-shape prefill plan.  A deterministic
numeric failure reopens the K-pack4 fragment/metadata diagnosis; a successful
pilot admits TM128 and the remaining real families, but does not by itself
establish the shipping operator or fused-store default.

The pilot was raw-bit clean.  Its nine confirmed TM64 candidates selected S1:

```text
shape=2048x1024x5120
config=64x128x256_w64x16_s2_bc0_apstandard-aiu
S1=101.759999990 us
S2 modeled E2E=116.917370303 us
S4 modeled E2E=134.339261431 us
S8=UNAVAILABLE
```

The archived Xplane/native-F2 result on the same cell was about
`69.320000708 us`, so the TM64 K-pack4 pilot was 46.797% slower.  This excludes
Split-K as the missing win on that row, but it does not separate K-pack4 format
cost from the pilot's single-TM denominator.

The admitted follow-up compiles the complete 918-row K-pack4 graph once and
selects exactly 774 AP0 prefill rows: TM16/32/64/128/256, all TK256.  It runs
all five inventory `(N,K)` families at `M={64,2048,4096}`, 15 shapes total,
with the same raw-bit-first three phases.  TM8/AP1 remains an explicit
decode-only complement of the manifest rather than a hidden missing row.

```bash
INTERNAL_SWEEP_SPEC=/path/to/complete/inventory.json \
JOBS=16 FQ_CONFIGS_PER_UNIT=4 \
  bash tools/run_fq_q4k_kpack4_prefill_pilot_box.sh
```

Full prefill command:

```bash
INTERNAL_SWEEP_SPEC=/path/to/complete/inventory.json \
JOBS=16 FQ_CONFIGS_PER_UNIT=4 \
  bash tools/run_fq_q4k_kpack4_prefill_real_shapes_box.sh
```
