# Q4_K/A32 folded-reader numeric incident

This is the case record for the first real-shape Q4_K ScaleFirst failure on an
`ArtifactTileK=32` folded artifact.  The reusable workflow lives in
`.codex/skills/ppu-cute-numeric-debug`; this file keeps only the evidence and
the exact production fix for this incident.

## Binding and final root cause

| Item | Binding |
|---|---|
| First observed revision | `851a374` |
| Format | GGUF Q4_K, `qtype=12`, `gs=32`, `bchunk=0` |
| Artifact | `ArtifactTileK=32`, folded `F=2` |
| Problem | `M=64, N=1024, K=5120` |
| Exact tactic | `64x64x128_w16x32_s8_bc0` |
| Exact symbol | `sf_q12_a32_tm64_tn64_tk128_wm16_wn32_s8_bc0` |
| Stable failure | `raw_bad=61184/65536`, first `0xc200 -> 0xcc80` (`-3 -> -18`) |
| Root oracle | `dev/fold_derivation/l220_q4_a32_prepare_consume_layout.cu` |
| Root cause | B's four-delivery coordinate was reused to index A's one-block, stride-zero copy view |
| PPU closure | `bf20f3e`: legacy negative reproduces the exact failure; shared A schedule is raw-bit exact |
| Artifact | `/workspace/quactlize-q4-a32-a-schedule-bf20f3e-20260821T010609Z` |

This is not a bad CuTe layout.  It is an invalid join between two independently
retiled CuTe views:

```text
B CPY_K extent = 4
A CPY_K extent = 1, stride = 0
MMA K atoms    = 8
```

The old mainloop called the A copy with B's `k_block` (`0..3`).  Only zero is
in A's logical CPY_K domain.  Physically all four coordinates alias the same
whole A fragment.  At the final-delivery wrap, preparing next-tile B delivery
zero therefore also reloaded all of A before current-tile delivery three had
consumed its final 16 A values.

## Constructive proof with the real layouts

L220 composes the exact production `partition_fragment_A/B`, tiled-copy
`retile_D`, and MMA-K modes.  Its result is:

```text
L220 A-fragment size=64 cosize=64
  layout=((_2,_2,_2),_1,_8):((_1,_2,_4),_0,_8)
L220 A-copy-view size=64 cosize=64
  layout=((_8,_8),_1,_1):((_1,_8),_0,_0)
L220 B-fragment size=128 cosize=128
  layout=((_2,_2,_2),_2,_8):((_1,_2,_4),_64,_8)
L220 wrap d3->d0 A_copy_K=1 A_prepare=64 A_consume=16
  A_overlap=16 B_prepare=32 B_consume=32 B_overlap=0
L220 schedule A_blocks=1 B_blocks=4 A_atoms=8 B_atoms=2
  loads/tile=1 delay-wrap=1 post-consume=1 matching-unchanged=1
L220 verdict=A_PREPARE_OVERWRITES_LIVE_D3
```

The important discriminator is `A_overlap=16` with `B_overlap=0`.  It rules
out the earlier theory that B scatter itself overlapped delivery zero and
delivery three.  The collision is A-only.

The actual pipeline chronology is:

1. Prime B delivery 0 and the one whole A block for the current K tile.
2. Prepare B deliveries 1, 2 and 3 while consuming the preceding delivery;
   A needs no additional load.
3. At current delivery 3, bind the next shared stage and prepare next B0.
4. Do **not** reload A yet; consume current B3 with current A atoms 6 and 7.
5. Reload the one whole A block from the next stage after B3 is dead.

No barrier, stage transition, B conversion, or MMA order changes.

## Why earlier observations looked broader

The six component fixtures at `789d25f` reported:

| Arm | Result | First output |
|---|---:|---:|
| transport-only | PASS | `+4 -> +4` |
| code-only | FAIL, 26,368 bad | `+2 -> -5.5` |
| scale-only | FAIL, 65,536 bad | `+0.5 -> 0` |
| zero-only | FAIL, 43,712 bad | `-3 -> -7.5` |
| metadata-only | FAIL, 65,536 bad | `-2.5 -> -7.5` |
| exact | FAIL, 61,184 bad | `-3 -> -18` |

`transport-only` made A K-invariant, so overwriting current A with next-tile A
was invisible.  Every other arm varied A by K segment and exposed the same A
lifetime failure.  Those results did not imply independent code, scale and
zero defects.

The one-box experiment at `7a1b4d8` then showed:

```text
legacy=NUMERIC_FAIL
typed_old_order=NUMERIC_FAIL
raw_consume_first=PASS
candidate=PASS
tail=PREVIOUS_TILE_STALE
```

That experiment proved the failure was cadence-sensitive, but it did not make
consume-first the right production fix.  Moving all preparation after consume
also moves valid B look-ahead and can cost performance.  L220 supplies the
missing operand-level proof: only the one A reload must move.

## Production fix

`detail/ppu_mixed_a_schedule.hpp` relates A and B through their common MMA-K
atom space.  It never treats one view's CPY_K coordinate as a coordinate in
the other view.

- Each A block is loaded once, immediately before the first B delivery that
  consumes one of its MMA-K atoms.
- When A has one whole-fragment block and B has multiple deliveries, only the
  steady-state B-last to next-B0 A reload is delayed until after consume.
- Equal A/B copy granularities retain the ordinary prepare-first schedule.
- `PPU_MIXED_LEGACY_B_INDEXED_A_COPY=1` is an exact device negative that
  restores the historical cross-view indexing and must reproduce the stable
  failure signature.

The same invalid direct A indexing syntax existed in the ordinary, folded and
two-plane collectives.  All three now use the shared schedule.  This does not
claim that every format previously failed: rows with matching A/B copy
granularity were unaffected.  It removes the mechanism from all three paths
instead of special-casing Q4/A32.

## Removed superseded code

The following diagnostic candidate code is deliberately absent from the final
hot path:

- typed half2 scatter (`Half2Layout`, `emit_to`);
- the pipeline-wide `ConsumeBeforePrepare` policy;
- Q4-only raw/typed and prepare/consume switches;
- L219's synthetic typed-cadence oracle and its source checker.

The B converter, dequant emitter and shared pipeline driver are restored
source-identically to the pre-candidate revision `5bf61dd`.  The final delta is
only the A scheduling seam plus its proof and exact negative.

## Performance invariants

For the exact Q4/A32 row:

- old A TSM loads per K tile: 4 aliases of the same 64-value destination;
- new A TSM loads per K tile: 1;
- B loads/conversion: unchanged;
- MMA count/order: unchanged;
- barriers and cp.async waits: unchanged;
- runtime branches: none; all schedule decisions are `if constexpr`;
- equal-granularity rows: compile to their existing prepare-first behavior.

Source invariants are not a device timing verdict.  After raw-bit closure,
record latency, registers, spill and ACU instruction mix.  A correctness-
failing arm never supplies performance evidence.

## Local validation

Use persistent output under `/workspace`:

```bash
cd /sim/eec/shared/junfu.qx/quactlize
mkdir -p /workspace/quactlize-q4-a32-oracles

python3 -B ci/check_mixed_a_register_schedule.py
python3 -B ci/check_q4_a32_fixture_components.py

python3 -B ci/local_gates.py -k l220_q4_a32_prepare_consume_layout --strict
python3 -B ci/local_gates.py -k test_fold_int2.cu --strict
python3 -B ci/local_gates.py -k test_fpA_kquant_dense.cu --strict
python3 -B ci/local_gates.py -k test_q3_bconcat_real.cu --strict
```

The three compile gates cover folded, ordinary and two-plane collective
instantiation.  The schedule checker plants a wrong wrap condition, bypasses
each required hook, and reintroduces direct B-coordinate indexing of A.

## Exact PPU closure: PASS

The two-arm batch on `bf20f3e` closed on the PPU:

```text
Q4_A32_CLOSURE verdict=PASS
  legacy_b_indexed_a=NUMERIC_FAIL
  candidate=PASS
  driver=PREPARE_FIRST
  b_converter=UNCHANGED
[q4-a32-closure] PASS: historical signature reproduced and the shared A
  schedule is raw-bit exact
```

Artifact:

```text
/workspace/quactlize-q4-a32-a-schedule-bf20f3e-20260821T010609Z
```

This is the required constructive closure: the legacy macro reproduced
`raw_bad=61184`, first `-3 -> -18`, while the default candidate returned
`raw_bad=0` in the same fixture/binary-generation protocol.  No second root-
cause box iteration is required.  Preserve the command for regression use:

```bash
cd /sim/eec/shared/junfu.qx/quactlize
git pull --ff-only origin develop

OUT=/workspace/quactlize-q4-a32-a-schedule-$(git rev-parse --short HEAD)-$(date -u +%Y%m%dT%H%M%SZ)
Q4_A32_CLOSURE=1 ITERATIONS=1 CORRECTNESS_REPEATS=8 JOBS=16 OUT="$OUT" \
  bash tools/run_scalefirst_q4_a32_exact_box.sh
echo "artifacts: $OUT"
```

Admission requires, in the same bundle:

```text
legacy-b-indexed-a = NUMERIC_FAIL
legacy signature   = raw_bad=61184, first -3 -> -18
candidate          = PASS with raw_bad=0
driver             = PREPARE_FIRST
B converter        = UNCHANGED
```

The legacy arm proves the fixture and binary still recognize this exact bug;
the candidate arm proves the replacement closes it.  Missing evidence is not
a pass.

## Historical device evidence

| Revision | Verdict | What it established |
|---|---|---|
| `eb7d95c` | FAIL, exact old signature | static-index cleanup alone was not a fix |
| `79fba86` | diagnostic | code/metadata/exact all fail, broad locus only |
| `7dad9ac` | VOID | fixture label and host golden contradicted each other |
| `789d25f` | diagnostic | transport constant-A arm masks the defect; varied-A arms fail |
| `8a2d8d5` | diagnostic | missing values concentrate in final 32-K delivery |
| `5bf61dd` | diagnostic | second tile is live; original tag was K-invariant |
| `7a1b4d8` | diagnostic | consume-first masks the failure; typed scatter is irrelevant |
| `bf20f3e` | **PASS** | exact legacy A-index negative fails and the shared MMA-K A schedule is raw-bit exact |

## Design rule for future collectives

For independently created CuTe views, equal shape labels do not establish a
shared coordinate system.  Compose them through a common semantic space (here
MMA-K atoms), and prove physical prepare/consume overlap with real production
layouts.  Never compare rebased subview offsets without adding each subview's
base offset; that can make distinct physical slots look identical.

## Source inventory

- `dev/fold_derivation/l220_q4_a32_prepare_consume_layout.cu`
- `ci/check_mixed_a_register_schedule.py`
- `ci/check_q4_a32_fixture_components.py`
- `tools/run_scalefirst_q4_a32_exact_box.sh`
- `quactlize/include/actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_a_schedule.hpp`
- `quactlize/include/actlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp`
- `quactlize/include/actlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_fold.hpp`
- `quactlize/include/actlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_mixed_input_2plane.hpp`
