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
| Source fix | `981afaa` (`fix Q4 A32 folded reader register indexing`) |
| Format | GGUF Q4_K, `qtype=12`, `gs=32`, `bchunk=0` |
| Artifact | `ArtifactTileK=32`, folded `F=2` |
| Problem | `M=64, N=1024, K=5120` |
| Exact tactic | `64x64x128_w16x32_s8_bc0` |
| Exact symbol | `sf_q12_a32_tm64_tn64_tk128_wm16_wn32_s8_bc0` |
| Local verdict | The register-indexing seam is identified, fixed, and covered by independent host/type oracles |
| PPU numeric verdict | **PENDING** until the exact one-row runner reports `raw_bad=0` |

Do not rewrite the pending verdict from a host proof.  The defect crossed a
PPU register-codegen boundary; only the exact device specialization closes the
numeric claim.

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
| final device value | `tools/run_scalefirst_q4_a32_exact_box.sh` | The shipping PPU specialization is numerically exact | any nonzero `raw_bad`, missing one-row marker, or nonzero process status is red |

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
- Metadata stage/group/N delivery is exact under an independently encoded
  oracle.
- Global-to-shared copy quanta are exact and unique under an arithmetic address
  oracle that does not call the producer placement function.

Consequently the remaining seam was after the proved load ownership and before
MMA consumption: indexing of register-backed fragments in the real device
body.

## Root seam

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
dynamic register-array subscript differently.  This is a real source defect:
the pipeline already owned a static index and the helper interface discarded
it.

The source-level defect and its fix are closed.  Its causal attribution to the
observed PPU values remains provisional until the exact device row passes.

## Fix in `981afaa`

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

This is deliberately narrower than rewriting the artifact or changing fold.
The artifact bytes and offline layout were already exact; the reader lost
static identity while consuming them.

## Local reproduction

Use a persistent directory below `/workspace`; none of these checks needs a
PPU.  L123 and L211-L213 are ordinary nvcc host executables with the
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

for id in 211 212 213; do
  nvcc -std=c++17 -arch=sm_80 -w "${INC[@]}" \
    "dev/fold_derivation/l${id}_q4_a32_"*.cu \
    -o "/workspace/quactlize-q4-a32-oracles/l${id}"
  "/workspace/quactlize-q4-a32-oracles/l${id}"
done

QUACTLIZE_L214_OUT=/workspace/quactlize-l214-q4-a32-exact \
  bash dev/fold_derivation/run_l214_q4_a32_exact_type.sh
```

If a toolchain glob selects more than one source, invoke L211, L212, and L213
by their exact filenames instead.  The evidence is the per-oracle marker, not
merely a zero shell status.

## Exact PPU closure

Run only the failed row; do not restart the full sweep first:

```bash
cd /sim/eec/shared/junfu.qx/quactlize
git pull --ff-only origin develop

OUT=/workspace/quactlize-q4-a32-exact-981afaa-$(date -u +%Y%m%dT%H%M%SZ) \
ITERATIONS=3 CORRECTNESS_REPEATS=8 JOBS=16 \
  bash tools/run_scalefirst_q4_a32_exact_box.sh
```

Admission requires all of the following from the same bundle:

```text
git.sha == the intended revision
manifest selection.mode == exact-symbol
manifest selection.compiled_rows == 1
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
| `981afaa` | PPU | **PENDING** | pending exact runner |

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
- `dev/fold_derivation/run_l214_q4_a32_exact_type.sh`
- `tools/gen_scalefirst_internal_units.py`
- `tools/run_scalefirst_q4_a32_exact_box.sh`
- `quactlize/include/quactlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_fold.hpp`
