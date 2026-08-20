# Q4_K/A32 folded-reader numeric incident

This is the durable debug record for the first real-shape Q4_K ScaleFirst
failure on an `ArtifactTileK=32` folded artifact.  It is intentionally kept
beside the layout oracles: the incident depends on production CuTe layouts,
the generated tactic authority, and one exact PPU specialization, so it is a
repository playbook rather than a general debugging skill.

## Status and binding

| Item | Binding |
|---|---|
| First observed revision | `851a374` |
| Disproved candidate | `981afaa` (`fix Q4 A32 folded reader register indexing`) |
| Format | GGUF Q4_K, `qtype=12`, `gs=32`, `bchunk=0` |
| Artifact | `ArtifactTileK=32`, folded `F=2` |
| Problem | `M=64, N=1024, K=5120` |
| Exact tactic | `64x64x128_w16x32_s8_bc0` |
| Exact symbol | `sf_q12_a32_tm64_tn64_tk128_wm16_wn32_s8_bc0` |
| Local verdict | Offline artifact and all statically composable reader/metadata maps are exact; root cause remains device-dynamic |
| PPU numeric verdict | `eb7d95c` reproduced the identical `61184`, `-3 -> -18` signature; `981afaa` is not the cause or fix |

Do not rewrite the device verdict from a host proof.  The remaining seam
crosses runtime cp.async/pipeline/codegen behavior; only the exact device
specialization closes the numeric claim.

## Failure record

The first full-graph failure was already a raw-bit correctness failure, not a
performance anomaly:

```text
SF_ATTEMPT shape=64x1024x5120 ...
  symbol=sf_q12_a32_tm32_tn32_tk64_wm16_wn16_s3_bc0
SF_FATAL ... state=RAW_FP16_MISMATCH raw_bad=65536
  first_bad=0 want=0xc200 got=0xd000
```

After the conservative candidate filter exposed the target used for the exact
diagnostic runner, the stable signature was:

```text
SF_SHARD qtype=12 artifact_tile_k=32 bchunk=0 typed_rows=490 ...
SF_ATTEMPT shape=64x1024x5120 ordinal=1/490
  symbol=sf_q12_a32_tm64_tn64_tk128_wm16_wn32_s8_bc0
SF_FATAL ... state=RAW_FP16_MISMATCH raw_bad=61184
  first_bad=0 want=0xc200 got=0xcc80
```

`0xc200` is fp16 `-3`; `0xcc80` is fp16 `-18`.  The count means that 4,352 of
65,536 outputs happened to agree.  Neither the count nor that ratio identifies
the broken map.  Timing from this arm is invalid because the arm failed exact
correctness.

The sparse exact fixture uses eight active K positions:

```text
11, 688, 1365, 2042, 2719, 3396, 4073, 4750
```

Its integer bound makes the fp16 result order-independent and exact.  That
removes reassociation as an explanation before inspecting any layout.

## Debug rule: stop the sweep at the first numeric failure

The full sweep is the wrong tool after `RAW_FP16_MISMATCH`.  It mixes thousands
of types and makes every rerun expensive.  The incident was reduced to one
generated row while retaining the full candidate authority:

1. Pin revision, qtype, artifact, tactic, shape, fixture, and raw fingerprint.
2. Generate the exact symbol from the normal authority with
   `--select-symbol`; do not hand-write a second row authority.
3. Compile one specialization and run one shape.
4. Resume a screen only after that row is raw-bit exact.

The generated manifest records both the full authority denominator and
`compiled_rows=1`.  An unknown symbol is a required red negative; a typo must
not silently produce an empty green target.

## What was proved and excluded

The investigation followed the physical chain instead of comparing two copies
of the same high-level layout.

| Layer | Oracle | What it establishes | Constructive negative |
|---|---|---|---|
| shared-to-register ownership | `l123_warp_nk_topology.cu` | Exact Q4/A32 `partition_S -> retile_D -> converter -> MMA fragment` ownership; no missing or duplicate slot | stale shadow `PermK` and multiplied folded compute `PermK` are red |
| metadata | `l211_q4_a32_metadata_map.cu` | All `(stage, group, n)` scale/zero values reach the expected fragment slots | encoded stage/group/n makes broadcast, rotation, or flattened-stage alias red |
| global-to-shared address | `l212_q4_a32_gmem_map.cu` | Every 32-byte quantum selected by the production operand has the independently computed artifact address | swapping descriptor `(coord_w, coord_h)` roles is red |
| output signature | `l213_q4_a32_failure_signature.cu` | Correct scatter reproduces raw-bit output; a delivery-major scatter corrupts it | one planted destination permutation is red |
| exact shipping type/body | `l214_q4_a32_exact_type.cu` | The exact generated `RowTypes<12,32,64,64,128,16,32,8,0>` reaches the full kernel body | unexpected non-vendor compile diagnostics are red |
| metadata apply composition | `l216_q4_a32_metadata_apply_map.cu` | Every converted code is paired with scale/zero from the same logical N | rotating metadata N by one is red |
| exact composed code reader | `l217_q4_a32_exact_composed_reader.cu` | Offline artifact slot through AIU destination, real tiled smem, TSM load, converter/scatter and MMA logical `(n,k)` is a bijection | rotating logical N by one is red |
| metadata global-to-shared | `l218_q4_a32_metadata_gmem_smem_map.cu` | All 40 K tiles and 10,240 values map through the production capped cp.async tiled copy to the intended stage/group/N slot | transposing source N/group is red |
| component device bisection | `FIXTURES=transport-only,code-only,scale-only,zero-only,metadata-only,exact tools/run_scalefirst_q4_a32_exact_box.sh` | Separates common transport, code reader, scale, zero, metadata interaction and exact failures in one specialization | every arm's prelaunch fixture identity must bind before a numeric classification |
| final device value | default single-exact mode of the same runner | The shipping PPU specialization is numerically exact | any nonzero `raw_bad`, missing one-row marker, or nonzero process status is red |

Recorded local positives:

```text
L123 Q4-A32-F2 WN=32 WK=1 entries=8192
  slot-diff=0 stored-byte-diff=0 cross-physical-row=0
  max-delta=1 max-vreg=1 pair-owner-bad=0 PASS

L211 metadata-map correct=1048576/1048576 bad=0
L211 Q4-A32 TM64/TN64/TK128/WM16/WN32/S8 flat=EXACT map=EXACT PASS

L212 distinct=160/160 swapped-coordinate-negative=RED result=PASS

L213 correct raw_bad=0 first=0;
  delivery-major raw_bad=65536 first=1.5 PASS

L214 exact q12/A32/64x64x128-w16x32-s8 body=REACHED
  vendor-asm-baseline=84 nonvendor=0 result=PASS

L216 metadata-apply same-N=32768/32768 bad=0 map_uncovered=0
  owner_conflicts=0; rotate-N-negative=64512/65536 result=PASS

L217 exact-reader map_diff=0/8192 owner_bad=0 predicted_raw_bad=0
  rotate-N-negative=64512 result=PASS

L218 metadata-gmem-smem tiles=40 values=10240 physical-duplicate=8 positive=EXACT
  transpose-negative=RED result=PASS
```

The exact L212 ABI detail matters: `coord_` is `(coord_w, coord_h)`, not
logical `(n, k)`.  Its flattened code address is
`coord_h * 256 + coord_w`.  Treating coordinate names as logical axes creates
a plausible but false map.

### Hypotheses ruled out before editing production code

- The producer artifact round trip was exact, but that only proves
  producer/inverse agreement.  It does not prove the shipping reader.
- Exhaustive simple subsets and multiplicities `{0,1,2}` of the eight active
  K stages, constrained to total eight, cannot reproduce the observed output
  signature.
- Missing or permuted whole K blocks cannot reproduce the signature.
- A delivery-major destination transpose is detectably wrong, but its planted
  signature does not match the device failure.
- Metadata global-to-shared delivery and shared-to-register application are
  exact under independently encoded oracles.
- Global-to-shared copy quanta are exact and unique under an arithmetic address
  oracle that does not call the producer placement function.
- The full offline-artifact-to-MMA code path is exact under a composed oracle;
  it reports `map_diff=0/8192` and exact-once ownership.

Consequently an offline placement change is not supported by the evidence.
The remaining seam is device-dynamic: actual cp.async issue/publication,
runtime pipeline stage reuse, or PPU lowering of the otherwise exact reader.
The exact tactic has 256 CTA threads but only 32 metadata-copy slots.  At
`79fba86`, production wrapped the thread id and issued eight identical clears
and asynchronous writes per logical slot.  That was statically complete, but
its device behavior was not proved by an address oracle.  The next experiment
replaces it with one physical publisher per proved slot without changing any
address, stage, converter, or MMA mapping.

## Three-arm device result and what it establishes

The exact/code-only/metadata-only run at `79fba86` classified the failure as
`COMMON_PIPELINE_OR_MULTIPLE_DEFECTS`:

| Arm | raw_bad | first want -> got |
|---|---:|---|
| code-only | 26,368 / 65,536 | `0x4000 -> 0xc580` |
| metadata-only | 65,536 / 65,536 | `0xc100 -> 0xc780` |
| exact | 61,184 / 65,536 | `0xc200 -> 0xcc80` |

Metadata-only holds every decoded code at one, so a pure code permutation
cannot explain that arm.  Code-only holds scale at one and zero at zero, so a
pure metadata-coordinate permutation cannot explain that arm.  Together they
reject an offline-format-only explanation, but do **not** prove that one defect
explains both arms; two device-lowering defects remain possible.

The exact type probe now pins `K_BLOCK_MAX=4` and
`K_ATOM_PER_COPY=2`.  A tempting ring-stage explanation was inspected and
rejected before editing the driver: this row has `bchunk=0`, so B conversion
and metadata application happen in `prepare`; `consume` only issues MMA.
Moving the consume-stage token cannot change this row and is not its fix.

## Exact-owner metadata publication experiment

`ScaleCopyPlan` now publishes both the physical-owner predicate and the
logical-slot map.  The folded collective uses the first 32 physical threads
to clear and publish the 32 Q4/A32 metadata slots exactly once; the remaining
224 threads issue no metadata write.  All threads retain the same shared
consumer path and synchronization cadence.

The local evidence is constructive rather than an idempotence assumption:

- L114 exhausts the exact plan: owner protocol is 256 values, 256 visits,
  zero duplicates, one hit each.  The old protocol is 256 values, 2,048
  visits, 1,792 duplicates, eight hits each and is required to report red.
- L218 applies the owner protocol to all 40 K tiles and 10,240 metadata values;
  the old 8x wrap and the source-coordinate transpose are independent red
  controls.
- The production-seam lint binds owner-only clear plus both preload and
  steady-state async issue points, then plants an all-thread publisher and
  requires it to fail.

This is still a **candidate cause** until the same three device arms run.  If
metadata-only turns green while code-only remains red, it has isolated the
metadata defect and there is a second B-reader defect.  If all three turn
green, repeated publication was the common root.  If none changes, the
experiment is falsified and must be reverted rather than retained as a
plausible cleanup.

The first rerun at `7dad9ac` is not admissible as that verdict.  Its line
labeled `metadata-only` reported `want=0x4000`, but the pinned metadata-only
fixture's host golden at output zero is necessarily `0xc100` (-2.5); `0x4000`
is the code-only golden (+2).  A device kernel can change `got`, never the host
`want`.  The old runner printed the fixture identity only after a successful
row, then attached the shell loop's label to a failing row.  It therefore
could not detect this contradiction and incorrectly retained the broad
`COMMON_PIPELINE_OR_MULTIPLE_DEFECTS` label.  Do not use the apparent
65,536-to-26,368 metadata change as kernel evidence.

The closure runner now emits `SF_FIXTURE` before launch and binds mode, first
golden and byte fingerprints for A, native/placed code, scale, zero and the
complete golden.  A wrong mode or the observed code-golden metadata plant is
`INFRA_FAIL`, not `NUMERIC_FAIL`.  It also decomposes the path with six arms:

| Arm | Varied quantity | Pinned output-zero golden |
|---|---|---:|
| transport-only | no B quantity; eight same-sign A impulses | `0x4400` (+4) |
| code-only | code | `0x4000` (+2) |
| scale-only | scale | `0x3800` (+0.5) |
| zero-only | zero | `0xc200` (-3) |
| metadata-only | scale and zero | `0xc100` (-2.5) |
| exact | code, scale and zero | `0xc200` (-3) |

`transport-only` is the branch point.  Failure there localizes the common A
load/stage/MMA/output path and the next experiment is one active K impulse at
a time.  A pass there makes code, scale and zero arms independent reader/apply
tests; only the failing component may then be instrumented.

## Disproved register-index hypothesis

`detail::run_mixed_pipeline` supplies `k_block` as a compile-time
`cute::Int<k_block>`.  Two helper boundaries erased that type:

```cpp
copy_B_and_extra_info(..., int k_block, ...)
transform_B_kblock(..., int k_block, ...)
```

The folded `FoldF > 1 && KBM > 1` path then indexed CuTe register fragments
through runtime integers and raw pointers.  The inner MMA atom loop had the
same dependency hidden behind `#pragma unroll`.

An unroll pragma is not a type-level guarantee.  Host layout proofs can show
that the intended indices form a bijection while PPU codegen still lowers a
dynamic register-array subscript differently.  This is a real source-hygiene
issue: the pipeline already owned a static index and the helper interface
discarded it.

However, the exact PPU row at `eb7d95c` (which contains `981afaa`) reproduced
the original signature byte for byte: `raw_bad=61184`, first `-3 -> -18`.
Therefore static-index erasure is neither the root cause nor a fix for this
incident.  It must not be used as the explanation for a later closure.

## Candidate change in `981afaa`

The patch preserves compile-time identity across the entire register path:

1. `copy_B_and_extra_info` and `transform_B_kblock` are templated on `KBlock`
   and use `KBlock::value`.
2. Every `tCsB`, `tCrB_load`, `tCrB_mma`, scale, and MMA-atom index derived
   from K block is a `cute::Int<>`.
3. Compile-time `cute::for_each(make_int_sequence<...>)` replaces loops whose
   indices address register-backed tensors.
4. Folded multi-delivery int4 converts one complete 32-code delivery with the
   already-shipping
   `MixGemmNumericArrayConverter<half_t, int4b_t, 32>`.
5. Converted elements land through the independently proved
   `MixGemmArtifactScatter::flat(E, KBlockIndex, II)` map.
6. Other bit widths retain their existing chunk emitter; legacy one-delivery
   and unfolded paths remain on their previous converter/write path.

The change is deliberately narrower than rewriting the artifact or changing
fold, but its device result is negative.  Keep its source-hygiene merit and
its incident causality separate; the latter has been constructively rejected.
The diagnostic tree therefore restores `ppu_mma_aiu_fold.hpp` byte-for-byte
to `851a374` before running the three fixture arms; a disproved candidate must
not become part of the new baseline merely because it is plausible.

## Local reproduction

Use a persistent directory below `/workspace`; none of these checks needs a
PPU.  L123, L211-L213, and L216-L218 are ordinary nvcc host executables with the
repository CuTe headers.  Run L123 directly here: its general-purpose shell
runner also contains unrelated shipping-builder type-equivalence admissions,
which are not part of this incident's evidence.  L214 deliberately expects the
known vendor asm diagnostics and rejects any additional diagnostic.

```bash
cd /sim/eec/shared/junfu.qx/quactlize

mkdir -p /workspace/quactlize-q4-a32-oracles
INC=(-Idev/fold_derivation/stub_inc -Ithird_party/actlize/include \
     -Ithird_party/actlize/tools/util/include -Iquactlize/include \
     -Itests -Ibenchmarks -Idev/fold_derivation)

nvcc -std=c++17 -arch=sm_80 -w "${INC[@]}" \
  dev/fold_derivation/l123_warp_nk_topology.cu \
  -o /workspace/quactlize-q4-a32-oracles/l123
/workspace/quactlize-q4-a32-oracles/l123

for id in 211 212 213 216 217 218; do
  nvcc -std=c++17 -arch=sm_80 -w "${INC[@]}" \
    "dev/fold_derivation/l${id}_q4_a32_"*.cu \
    -o "/workspace/quactlize-q4-a32-oracles/l${id}"
  "/workspace/quactlize-q4-a32-oracles/l${id}"
done

QUACTLIZE_L214_OUT=/workspace/quactlize-l214-q4-a32-exact \
  bash dev/fold_derivation/run_l214_q4_a32_exact_type.sh
```

If a toolchain glob selects more than one source, invoke each oracle by its
exact filename.  The evidence is the per-oracle marker, not merely a zero
shell status.

## Exact PPU closure

Run only the failed row; do not restart the full sweep first:

```bash
cd /sim/eec/shared/junfu.qx/quactlize
git pull --ff-only origin develop

OUT=/workspace/quactlize-q4-a32-components-$(git rev-parse --short HEAD)-$(date -u +%Y%m%dT%H%M%SZ) \
FIXTURES=transport-only,code-only,scale-only,zero-only,metadata-only,exact \
ITERATIONS=1 CORRECTNESS_REPEATS=8 JOBS=16 \
  bash tools/run_scalefirst_q4_a32_exact_box.sh
```

The six arms compile the exact shipping row once.  The final three retain the
historical broad classifier, while transport/scale/zero make that result
actionable:

- code-only fail, metadata-only pass: code reader or B pipeline;
- code-only pass, metadata-only fail: metadata load/apply;
- both isolated arms pass but exact fails: interaction or stage reuse;
- both isolated arms fail: common pipeline or multiple defects.

Admission requires all of the following from the same bundle:

```text
git.sha == the intended revision
manifest selection.mode == exact-symbol
manifest selection.compiled_rows == 1
exactly one SF_FIXTURE marker with the registered mode, first golden and input hashes
SF_COMPLETE status=COMPLETE shape=64x1024x5120 typed_rows=1
every measured row contains raw_bad":0
[q4-a32-exact] PASS
```

The runner stores the generated authority, build log, binary SHA-256, Git SHA,
and exact output below `OUT`.  A compile-only pass is not numeric closure.

### Device verdict log

Fill this table by appending evidence; do not alter the admission rule above.

| Revision | Device | Result | Artifact directory |
|---|---|---|---|
| `eb7d95c` | PPU | **FAIL**, identical `61184`, first `-3 -> -18` | `/workspace/quactlize-q4-a32-exact-eb7d95c-20260820T070953Z` |
| `79fba86` | PPU | **FAIL**, three-arm locus `COMMON_PIPELINE_OR_MULTIPLE_DEFECTS`; 26,368 / 65,536 / 61,184 bad | `/workspace/quactlize-q4-a32-bisect-79fba86-20260820T081739Z` |
| `7dad9ac` | PPU | **VOID**, failing metadata arm carried code-only host golden (`0x4000`, required `0xc100`) | artifact path not supplied |

## Reusable folded-artifact decision tree

For the next folded-reader mismatch, use this order:

1. **Classify the fixture.** Establish exact/order-independent output or keep
   numerical tolerance separate from layout diagnosis.
2. **Bind one row.** Record SHA, symbol, all tactic axes, artifact axes, shape,
   fixture seed, raw mismatch count, first mismatch, and fingerprint.
3. **Prove producer bytes.** Compare shipping artifact bytes and round trip,
   but do not mistake producer/inverse agreement for reader correctness.
4. **Prove global-to-shared coordinates.** Decode the descriptor ABI against
   independent arithmetic; include a coordinate-role negative.
5. **Prove shared-to-register ownership.** Compose real `partition_S`,
   `retile_D`, converter emission, and MMA fragment ownership.  Check
   exact-once coverage and a wrong-layout negative.
6. **Prove metadata independently.** Encode stage, group, and N into values so
   aliasing cannot remain invisible.
7. **Classify candidate scatters constructively.** Plant one map change at a
   time and compare the complete failure signature, not just a total count.
8. **Audit static identity.** Trace `cute::Int<>` indices across helper APIs.
   Any register-backed tensor index that becomes `int` is suspect even if a
   nearby pragma says unroll.
9. **Compile the exact production type/body.** A reduced type is insufficient;
   bind the generated shipping row.
10. **Run one exact device row.** Only raw-bit closure permits the wider sweep
    to resume.

## Things not to do

- Do not report latency, MBU, or MFU from a correctness-failing arm.
- Do not rerun thousands of configurations to diagnose one exact-row failure.
- Do not infer a map from `raw_bad`, a ratio, or the first output value.
- Do not use the artifact round-trip as the consumer oracle.
- Do not let the oracle call the same placement helper it is checking.
- Do not assume `#pragma unroll` preserves a compile-time register index.
- Do not call a host type/body compilation a PPU numeric result.
- Do not broaden the fix to other widths until each width has its own positive
  and wrong-map negative.

## Source inventory

- `dev/fold_derivation/l123_warp_nk_topology.cu`
- `dev/fold_derivation/l211_q4_a32_metadata_map.cu`
- `dev/fold_derivation/l212_q4_a32_gmem_map.cu`
- `dev/fold_derivation/l213_q4_a32_failure_signature.cu`
- `dev/fold_derivation/l214_q4_a32_exact_type.cu`
- `dev/fold_derivation/l216_q4_a32_metadata_apply_map.cu`
- `dev/fold_derivation/l217_q4_a32_exact_composed_reader.cu`
- `dev/fold_derivation/l218_q4_a32_metadata_gmem_smem_map.cu`
- `dev/fold_derivation/run_l214_q4_a32_exact_type.sh`
- `tools/gen_scalefirst_internal_units.py`
- `tools/run_scalefirst_q4_a32_exact_box.sh`
- `quactlize/include/quactlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_fold.hpp`
