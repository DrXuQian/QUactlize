---
name: ppu-cute-numeric-debug
description: Isolate PPU CuTe kernel raw-bit or numeric mismatches involving layouts, register fragments, mixed-input pipelines, metadata, or offline artifacts. Use when a shipping specialization fails correctness, when a workaround changes cadence but does not prove a root cause, or before editing performance-sensitive collective code to fix a suspected placement or lifetime bug.
---

# PPU CuTe Numeric Debug

## Objective

Reduce a numeric failure to one shipping specialization, prove the failing
semantic seam with real CuTe layouts, and close it with an exact negative plus
the smallest performance-preserving production change.

Read `references/q4-a32-case.md` when the failure resembles a mixed A/B
register-delivery or prepare/consume lifetime problem.

Read `references/packed-metadata-owner-deficit.md` when the first bad output is
an N-column boundary, especially when TileN exceeds the CTA thread count or a
packed scale/zero channel is decoded before its publishing barrier.

## Workflow

### 1. Freeze one exact row

Record the Git SHA, generated symbol, shape, qtype, group size, artifact axes,
tactic axes, fixture seed, mismatch count, first mismatch and output
fingerprint.  Generate that symbol through the shipping authority; do not
hand-write a second tactic authority.

Stop the full sweep at the first raw-bit failure.  Timing from a wrong result
is invalid.

### 2. Establish an independent numeric oracle

Prefer an order-independent, exactly representable fp16 fixture.  Split the
input into independently varied components such as transport, code, scale,
zero, metadata and exact.  Compute the host golden without calling the
producer placement or consumer mapping being checked.

Print prelaunch input and golden hashes.  A label/hash contradiction is an
infrastructure failure, not a device result.

### 3. Prove each physical chain separately

Check, in order:

1. native bytes to offline artifact and round trip;
2. global descriptor coordinates to shared addresses;
3. shared partition to register copy view;
4. converter destination to MMA fragment;
5. scale/zero stage, group and N ownership;
6. output ownership and fixup/reduction.

Every positive needs a wrong-coordinate, missing-owner or duplicate-owner
negative that turns red.

### 4. Compose the actual CuTe types

Use the production `TiledMMA`, copy atoms, `partition_fragment_A/B`,
`partition_S` and `retile_D`.  Print shape, stride, size and cosize.

For every prepare/consume pair, enumerate physical destination offsets and
intersect them.  Include the subview base offset before comparing subviews;
rebasing both to zero can manufacture a false overlap.

Treat independently retiled views as different coordinate systems.  Never
index A with B's CPY_K coordinate, even when both modes are named `k_block`.
Relate them through a shared semantic space such as MMA-K atoms.

### 5. Distinguish stale, current and future values

Use shifted absolute-coordinate tags.  Make tags vary along the suspected
axis; an N-only tag cannot distinguish K-stage freshness.  Register the exact
denominator and require a missing round or coordinate to fail closed.

### 6. Use workarounds only as bisections

A cadence change, typed destination, forced unroll or alternate scatter that
makes the row pass proves sensitivity, not causality.  Build a factorial when
two seams changed.  Then construct a layout oracle that uniquely separates
them before retaining production code.

### 7. Implement at the semantic seam

Fix the shared abstraction, not one qtype, when the same mechanism appears in
multiple collectives.  Preserve unrelated converters, B look-ahead, barriers,
MMA order and scheduler behavior.  Use compile-time scheduling so the hot path
has no runtime branch.

Keep one exact legacy macro or fixture that reproduces the historical failure
signature.  Delete superseded candidate code after the root is proved.

### 8. Close locally, then on device

Locally require:

- the real-layout oracle passes;
- all source-seam negative plants turn red;
- representative ordinary, folded and two-plane shipping bodies compile;
- no unexpected compiler diagnostic;
- `git diff --check` passes.

On the PPU require the legacy arm to reproduce the exact failure and the
candidate to be raw-bit exact in one hash-bound bundle.  Only then measure
latency, registers, spills and ACU instruction mix.

## Performance guard

Before editing, write down counts for loads, conversions, MMA, barriers and
runtime branches.  After the fix, prove which counts are unchanged and which
are reduced.  Source invariants do not replace a device timing verdict, but
they prevent an accidental broad rewrite from being called a correctness fix.

## Required report

Report:

- exact root cause, including both logical extents and physical overlap;
- hypotheses constructively excluded;
- files retained, restored and deleted;
- scope scan for the same mechanism;
- local positives and planted negatives;
- one direct device command and its admission markers;
- performance invariants and remaining device-only measurement.
