# Q4_K fully-quantized Split-K partial epilogue trigger record

Status: the rarer direct-store failure is localized to stale packed-A delivery
but its final causal seam remains open.  The earlier shared-output path was one
independent corruption trigger.  A proposed physical stage-footprint repair
at ca01dc6 removed every modeled cross-stage x4/read-to-writer overlap, yet the
same packed-row S4 producer failed at repeat 714 (`1.0 -> 6.0`).  The footprint
overlap is a real fragility, not the complete numerical root.  This is not
Split-K producer-to-producer synchronization.

## Refuted complete-root hypothesis: packed-A physical stage overlap

The exact failing symbol uses an m8 CuTe A copy atom whose semantic destination
has two registers per lane.  Its implementation nevertheless executes the
PPU0010 `m8n8.x4` swizzled shared load into a private four-register temporary,
then retains v0/v1.  The old packed layout compressed both cubes and stages at
64 fp16 elements per cube.  CuTe's logical x2 layout proved the live row values
were bijective, but it did not represent the two discarded registers' real
shared-memory reads.

An exhaustive hardware-address replay for TM8/TK256/Stages2 gives:

```text
logical CuTe source coordinates:       exact, coord_bad=0
historical stage pitch 256 half:       432 cross-stage physical read/write addresses
repaired stage pitch 1216 half:        0 cross-stage physical read/write addresses
writer -> calibrated reader values:    exact, value_mismatches=0
direct partial/reducer ABI:            exact
```

The conflicting writes are the other stage's row-0 cp.async transfers.  The
conflicting reads feed hidden x4 rows that are discarded or output-masked, but
the physical memory accesses still occur.  Thus the old layout contains a real
intra-CTA shared-memory fragility even though the mathematical row-0 mapping
and each Split-K output plane are disjoint.  Removing it did not close the
frozen numerical failure, so it cannot carry the root-cause verdict.

The repair separates cube and stage pitches:

```text
cube_pitch  = 64 half
stage_pitch = cube_pitch * (cubes_per_stage - 1) + 16 * 64
            = 1216 half for TK256
```

The writer, allocation, and `PPU0010_TSM_LD_SWZL_M8` operation now share that
authority.  TK256/S2 A storage becomes 2432 half (4.75 KiB), still much smaller
than the ordinary physical-m16 allocation of 8192 half (16 KiB).  L186 binds
the production type to `(cube_pitch, stage_pitch)=(64,1216)`, requires the old
layout to show crossings, and requires the repaired layout to show none.  This
is a physical-contract proof only, not a device numeric closure.

## Exact stale previous-A-tile fingerprint

The ca01dc6 failure snapshot found AP1/S4 plane 2, output column 32, expected
FP32 `1.0` and observed `6.0`.  The exact fixture contributions from its five
superblocks are `+3,-14,+1,-4,+15`.  Replacing sb13's complete A tile with the
immediately preceding tile changes its contribution from `-4` to `+1` and
uniquely gives `6.0`; the same substitution at sb10,11,12,14 gives
`-2,18,-14,-18` respectively.

L224 independently composes the exact CuTe fragment/copy layouts.  All 16 A
copy blocks map one-to-one to the 16 MMA K atoms, and prepare(next B delivery)
has zero logical register-offset overlap with consume(current B delivery).
Therefore the stale value is not explained by the source-level CuTe block map.
The remaining candidates are mutable stage binding, prepare-before-consume's
physical/backend register lifetime, the x4-to-x2 temporary, and packed-A
asynchronous publication/compiler ordering.  The one-box factorial isolates
these without changing the shipping default.

## Scope

This incident is deliberately narrower than "PPU epilogues are broken".
Ordinary S=1 output epilogues remain raw-bit-clean controls.  The failure was
observed in the fully-quantized fixed Split-K producer below:

```text
shape       M/N/K=1/1024/5120, group_size=32
format      Q4_K, ArtifactTileK=64, bchunk=0
tactic      TM/TN/TK=8/64/256, WM/WN=8/16, stages=2
providers   standard-aiu and packed-row
splits      S=2 and S=4
```

The S>1 route is not the S=1 kernel with one runtime integer changed.  S=1
delegates to the shipping GEMM and its product epilogue.  S>1 instantiates
`GemmUniversalMixedInputSplitKParallel`, traverses a shorter K interval, and
publishes FP32 partial planes before a deterministic reducer.  Consequently,
the evidence proves a failure in this Split-K producer's compiled context; it
does not prove that the split count alone would make an otherwise identical
S=1 binary fail.

## What failed

The mainloop accumulator was exact.  The old partial-output path then
redistributed the completed FP32 fragment register-to-shared, synchronized,
reloaded shared-to-register, and wrote the split-major workspace.  Independent
per-plane FP32 goldens found intermittent complete 32-output stripes already
wrong in the producer workspace.  Canaries were intact.  The reducer,
workspace plane addressing, and cross-kernel publication were therefore not
the source.

The production candidate stores the completed accumulator directly through the
real `TiledMma::partition_C` ownership into the FP32 partial plane.  This is a
same-type internal store: no conversion or output-layout redistribution needs
the shared round trip.  It sharply reduced the observed failure rate but did
not close correctness at 8192 repeats.

## Reopened high-repeat result

At `79ff0a3`, the first same-custom-kernel S1/S2/S4 direct arm reported:

```text
standard-aiu S1: REAL_CAN_IMPLEMENT (not launched)
standard-aiu S2: 8192 repeats clean
standard-aiu S4: 8192 repeats clean
packed-row  S1: REAL_CAN_IMPLEMENT (not launched)
packed-row  S2: 8192 repeats clean
packed-row  S4: raw_bad=32 at repeat 2142
```

The S1 result was an invalid diagnostic control, not a device verdict.
`make_cute_packed_stride` canonicalized singleton `L=1` to `stride_L=0`, but
the custom partial ABI requires one physical plane with `stride_L=M*N=1024`.
The next diagnostic explicitly supplies that plane stride.

The packed-row S4 failure initially reopened the producer/reducer boundary.
Failure-only partial snapshots prove the wrong value is already in one FP32
producer plane, and the exact value classifies it as stale A.  The physical
footprint change did not close it.  No inter-CTA publication fix is called for;
the remaining experiment is entirely inside the packed-A delivery pipeline.

## Why this is not a missing barrier

All supported synchronization boundaries were present or tested directly:

1. The mixed-input mainloop exits with `cp_async_wait<0>()` followed by
   `__syncthreads()`.  No asynchronous mainloop copy remains in flight when the
   epilogue starts.
2. The historical partial epilogue synchronizes after register-to-shared and
   again after shared-to-register.
3. Separate binaries replaced those calls with the historical user barrier,
   the reserved epilogue barrier, and full CTA `__syncthreads()`; every arm
   retained intermittent corruption.
4. An extra full CTA barrier before the first shared store did not close it.
5. Mainloop and epilogue shared allocations were made physically disjoint and
   the failure remained.
6. A CTA-only prefix with no shared store was clean.
7. A device completion boundary followed by D2H inspection found the wrong
   FP32 values already resident in individual producer planes.  No
   producer/reducer fence or launch ordering can repair bytes that are already
   wrong before the reducer starts.

The decisive prefix wrote one constant per owner to disjoint shared memory,
used two CTA barriers, never read those constants, and then executed the proven
direct accumulator store.  That binary still failed.  A missing producer to
consumer ordering edge cannot explain corruption of unrelated live
accumulator values in this arm.

An undocumented backend or hardware scoreboard hazard may internally be
ordering-sensitive, but no supported barrier variant repaired it.  Calling
this a source-level missing-barrier bug would therefore overstate the evidence
and point at a refuted fix.

## Earlier shared-output trigger (separate from the stale-A incident)

The earlier, now superseded 512-repeat device arm reported:

```text
legacy shared output: 776 / 8192 bad samples
production direct:    raw-bit exact, standard-aiu + packed-row, S=2 + S=4
verdict:              SHARED_STORE_BACKEND_OR_FOOTPRINT_CAUSAL
```

Exact-symbol disassembly kept 16 MMA instructions, 34 TSM stores, 67 TSM
loads, and zero reported stack in the clean direct binary.  The first failing
flat-constant binary kept the same MMA/load counts and added exactly one
`tsm.st.b32x4` store.  Thus, within the frozen custom S>1 kernel, the old
shared-output path had an independent corruption trigger:

- removing the unnecessary shared-memory partial-output handoff is a
  performance-neutral candidate that eliminates one strong trigger but is not
  a correctness closure;
- the first isolated operation that can trigger corruption is the additional
  vector TSM-store lowering (or the compiler/hardware footprint it changes);
- ordinary S=1 correctness did not isolate whether that earlier trigger also
  required the custom kernel context, because S=1 changed the compiled kernel,
  FP16 versus FP32 output, copy ownership and register/shared footprint.

The rarer direct-store failure contains no shared-output round trip.  Its exact
stale-A value is independent of that historical epilogue trigger, while the
final A-stage causal seam remains to be adjudicated.

The direct-store candidate does not change accumulation or reduction order.  Its
same-run packed-row S4 median was 11.300 us versus 11.320 us for the legacy
shared path (`-0.177%`), within the preregistered 3% regression limit.

## Similar-path audit

The CuTe copy-op audit found one owned operation that publishes fewer
registers than its hardware opcode touches: `PPU0010_TSM_LD_SWZL_M8` (logical
x2, physical x4).  Ordinary A uses natural cube/stage spacing and is not
exposed.  Every production Q2/Q4 packed-row specialization reaches the same
typed m8 atom, so the independent stage-pitch fix covers the family rather
than only the frozen Q4/S4 symbol.  L186 instantiates the complete seven-cell
Q2/Q4 packed-A matrix and the exact Stages2 failure type.

The packed-A gmem writer remains a narrow custom cp.async loop because the
PPU swizzle/run permutation is not a plain affine CuTe layout.  Its source is
the real CuTe `gA(row,k,k_tile)` tensor, and its destination is checked against
an independent hardware-calibrated reader for every byte, cube and stage.
Replacing that loop with a cosmetic logical CuTe tensor would not encode the
hidden x4 footprint and therefore would not prevent this class of bug.  The
design rule is instead: logical value mapping through CuTe, plus an explicit
physical-footprint lifetime contract for any opcode whose physical accesses
exceed its logical fragment.

After cleanup, owned code has no other post-mainloop
`retile_S(accumulators)` shared round trip.  The fixed Split-K hot path is
direct; the exact historical collective call remains only under
`PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE` as a negative control.

Actlize's ordinary vectorized, array, and EVT epilogues contain structurally
similar register/shared/register redistribution, but their shipping S=1 paths
have independent correctness coverage and may require it for conversion or
elementwise work.  They are controls, not implicated bugs, and must not be
blanket-rewritten.

`ppu_epilogue_vectorized_parallel.hpp` is the closest structural analogue for
same-type partial publication.  Quactlize's fixed Split-K path retains its
type only for Params/stride ABI and bypasses its call operator.  Any future
owned producer that publishes same-type FP32 partials must either use direct
MMA ownership or earn its own high-repeat raw-bit closure before reusing that
shared path.

The grouped MoE Split-K kernel is adjacent but not identical.  It invokes the
ordinary pointer-array epilogue for every K slice, converts each partial to
FP16, and later accumulates those FP16 planes in FP32.  Its benchmark compares
S>1 output with S=1 under a numerical tolerance, so it covers the intended
math but is not a high-repeat per-plane raw-bit test for intermittent shared
handoff corruption.  Keep that path unchanged; before treating it as a
shipping correctness authority, add a high-repeat partial-plane/canary census
or give its completed same-type stage a direct-store route.

The generic vendor `GemmUniversalParallel` plus `EpilogueParallel` route is
also structurally exposed.  No owned fixed Split-K launch uses its call
operator after this repair, so it is not a current regression.  A future
caller must not infer admission from type formation alone.

The current device closure is implemented by
`tools/run_fq_q4k_custom_split_count_box.sh`.  It launches the exact generated
AP0/AP1 `GemmUniversalMixedInputSplitKParallel` symbols at runtime S=1/2/4
across two binaries.  The preceding eight-binary factorial at `3e83a45`
left every counterfactual dirty: baseline, prepare-after-consume,
explicit-stage, direct-x4, separate-A-group and synchronous-store were 0/8
clean S>1 cells; compiler-fence was 4/8 and A-before-B 6/8.  Those incidence
changes are code-layout sensitivity, not closure.

The first reduced closure compares baseline with one exact contract arm:

- `asm-memory-contract` retains the physical x4 opcode and adds actual
  `memory` clobbers to the fp16-A AIU/packed cp.async producer, commit/wait and
  m8 ldmatrix consumer;

The already source-proved `logical-x2-scalar` arm, which removes the physical
x4 swizzle opcode and loads only its two semantic b32 values, is deliberately
not part of this first Box run.  It is the next one-variable probe only if the
exact memory-contract arm remains dirty.

L186 proves that reserved arm's 512 `(coord_h,slice,lane,vreg)` addresses against the
independent calibrated PPU0010 model and requires a one-word-offset negative
to turn RED.  Two independent 32768-repeat attempts run for both current arms;
numeric failure never truncates the closure.

The missing inline-asm memory side-effect declarations are a real compiler
contract defect, but they are not yet the device root.  Pure address arithmetic
may be hoisted or commoned only while preserving the runtime stage value, so
the source defect alone does not prove the proposed stale-stage mechanism.  A
causal verdict requires the x4-preserving memory-contract arm to close while
the exact frozen baseline incident reproduces.  If it remains dirty, run the
reserved scalar arm separately; do not mix that opcode change into this first
contract adjudication.

Until one arm closes while the baseline reproduces, the honest claim is: the
legacy shared handoff is unsafe and direct store removes that trigger; the
remaining direct-store incident is an exact stale previous-A-tile delivery,
its logical CuTe block map is exact, ca01dc6 stage separation did not repair
it, and the final physical/backend delivery seam still requires one box run.

## Retained regression evidence

- `ppu_splitk_direct_accumulator_store.hpp`: production implementation.
- `PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE`: exact historical negative.
- `l222_fq_splitk_direct_accumulator_store.cu`: exact-once ownership oracle,
  including duplicate-owner and rotated-fragment negatives.
- `ci/check_fq_splitk_partial_path.py`: source contract for the direct path,
  terminal mainloop synchronization, and ordered producer/reducer timing.

The one-off owner, barrier, handoff, and prefix factorial programs are retained
in Git history, not in the shipping tree.
