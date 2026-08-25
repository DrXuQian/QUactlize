# Q4_K fully-quantized Split-K partial epilogue trigger record

Status: the rarer direct-store failure is localized to one wrong 32-output
band in a completed producer FP32 partial; its final causal seam remains open.
That places the defect between operand delivery and the direct store, but does
not separately observe or certify the accumulator.  The earlier shared-output
path was one independent corruption trigger.  A proposed physical
stage-footprint repair at ca01dc6
removed every modeled cross-stage x4/read-to-writer overlap, yet the same
packed-row S4 producer failed at repeat 714 (`1.0 -> 6.0`).  That value pair
admits one stale-A explanation, but the full failure family does not.  The
footprint overlap is a real fragility, not the complete numerical root.  This
is not Split-K producer-to-producer synchronization.

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

## One stale-A-compatible signature, not a family root

The ca01dc6 failure snapshot found AP1/S4 plane 2, output column 32, expected
FP32 `1.0` and observed `6.0`.  The exact fixture contributions from its five
superblocks are `+3,-14,+1,-4,+15`.  Replacing sb13's complete A tile with the
immediately preceding tile changes its contribution from `-4` to `+1` and
uniquely gives `6.0`; the same substitution at sb10,11,12,14 gives
`-2,18,-14,-18` respectively.  This proves compatibility, not causality.

L224 now enumerates all 20 active K superblocks, all eight fixture column
residues and every complete source-A/destination-B tile pairing.  Adjacent
previous-A substitutions can produce the observed `+5` and `-5` deltas, and
some non-adjacent complete-A substitutions can produce `-4`.  No complete
A-tile substitution can produce the observed `-6` delta.  Therefore the
earlier statement that the failure family was “classified exactly as stale
A” was false; only one captured value pair had that explanation.

L224 independently composes the exact shipping CuTe fragment/copy layouts,
including the builder's Q4/TK256 `MmaPermK=64`.  The A and B fragments each
contain 16 MMA K atoms and their copy views each contain four delivery blocks;
delivery block `i` maps exactly to atoms `[4*i,4*i+4)`.  The reconstructed B
load/conversion view is also exact, and prepare(next B delivery) has zero
logical register-offset overlap with consume(current B delivery).  An older
L224 accidentally used the atom's K16 default and reported 16 one-atom A copy
blocks; that was not the shipping type and is not evidence.  With the corrected
type, the exact block map still neither explains nor rules out the device
failure.  The remaining candidates are operand publication/delivery,
physical/backend register lifetime, the x4-to-x2 consumer and MMA/accumulator
state.  The reduced closure isolates the remaining publication/consumer seams
without changing the shipping default.

The same exact `partition_C` oracle maps M=1 output ownership as 16 consecutive
N values per warp: N0-15/W0, N16-31/W1, N32-47/W2, N48-63/W3.  One aligned
32-output band is therefore two adjacent output warps.  It does not, by
itself, identify an A/B copy issuer or prove a wait/barrier defect.

## Scope

This incident is deliberately narrower than "PPU epilogues are broken".
Ordinary shipping S=1 output epilogues remain raw-bit-clean controls.  The
failure was observed in the fully-quantized fixed Split-K producer below:

```text
shape       M/N/K=1/1024/5120, group_size=32
format      Q4_K, ArtifactTileK=64, bchunk=0
tactic      TM/TN/TK=8/64/256, WM/WN=8/16, stages=2
providers   standard-aiu and packed-row
splits      S=2 and S=4
```

The shipping S>1 route is not the S=1 kernel with one runtime integer changed.  S=1
delegates to the shipping GEMM and its product epilogue.  S>1 instantiates
`GemmUniversalMixedInputSplitKParallel`, traverses a shorter K interval, and
publishes FP32 partial planes before a deterministic reducer.  Consequently,
the evidence proves a failure in this Split-K producer's compiled context; it
does not prove that the split count alone would make an otherwise identical
S=1 binary fail.  The current diagnostic bypasses that dispatch and runs the
same custom producer type at runtime S=1/2/4; its earlier parser accidentally
dropped S1, so the S>1-only premise still requires one device census.

## What failed

For the earlier shared-output trigger, the accumulator feeding a clean direct
arm was exact.  The old partial-output path then redistributed the completed
FP32 fragment register-to-shared, synchronized,
reloaded shared-to-register, and wrote the split-major workspace.  Independent
per-plane FP32 goldens found intermittent complete 32-output stripes already
wrong in the producer workspace.  Canaries were intact.  The reducer,
workspace plane addressing, and cross-kernel publication were therefore not
the source.

The production candidate stores the completed accumulator directly through the
real `TiledMma::partition_C` ownership into the FP32 partial plane.  This is a
same-type internal store: no conversion or output-layout redistribution needs
the shared round trip.  It sharply reduced the observed failure rate but did
not close correctness at 8192 repeats.  In that rarer direct-store incident,
a wrong FP32 partial leaves the source before or at the direct store; it does
not prove that the accumulator itself was exact.

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
producer plane.  One exact value pair admits stale A, but the later `-6`
family member rules out complete stale-A substitution as the common root.  The
physical footprint change did not close it.  No inter-CTA publication fix is
called for; the remaining experiment is inside the mainloop operand/
accumulator path.

## What synchronization is closed, and what remains open

Producer-to-producer and producer-to-reducer synchronization are closed:

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

The barrier variants above were epilogue/prefix tests.  They do not directly
adjudicate whether PPU async-shared writes require
`fence_view_async_shared()` between the mainloop's per-thread
`cp_async_wait<Stages-2>()` and ordinary shared consumers.  That one proxy-
visibility question remains open and is tested separately from the m8 x4
consumer.  Calling the bug a source-level missing barrier before that arm
closes would still overstate the evidence.

One proposed waiter diagnosis read the wrong `cute::copy_aiu` branch.  For
`__HGGC_ARCH__ == 100`, warp 0 issues both ordinary A and B; the warp0/A,
warp1/B split exists only in the alternate branch.  Packed A uses its explicit
logical copy threads, and the wrapped extra-input partition gives CTA threads
metadata-copy work.  `PPU_PACKED_SPLIT_GROUPS` changes only which duplicate
metadata owners decode which groups.  It can change cadence, but it does not
make previously nonparticipating A/B waiters participate and therefore cannot
adjudicate that A/B hypothesis.

The Stages2 driver does issue one redundant copy of the final K tile into the
retired, physically disjoint write stage while draining.  Removing it changes
async traffic and cadence but does not repair a proved address or ownership
violation; a clean result would therefore be another sensitivity result, not a
root-cause verdict.  It is intentionally excluded from the first causal
closure.  Only if both exact candidates remain dirty should it be used as a
second-stage localization arm, followed by a more direct producer/consumer
proof before any production change.

## Earlier shared-output trigger (separate from the direct-store incident)

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

The rarer direct-store failure contains no shared-output round trip.  Its
producer-partial corruption is independent of that historical epilogue
trigger, while the final operand/accumulator seam remains to be adjudicated.

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
across one uncontaminated baseline and two orthogonal candidates.  The
preceding eight-binary factorial at `3e83a45`
left every counterfactual dirty: baseline, prepare-after-consume,
explicit-stage, direct-x4, separate-A-group and synchronous-store were 0/8
clean S>1 cells; compiler-fence was 4/8 and A-before-B 6/8.  Those incidence
changes are code-layout sensitivity, not closure.

The repeated host harness was also audited. Every round is ordered as
producer, reducer, device synchronize, then D2H output inspection, so launches
cannot overlap across repeats. The FP32 workspace is reused without clearing,
however, and historical runs inspected its planes only after a reduced FP16
failure. This leaves a narrower harness-state possibility: corrupt planes in
one apparently clean round could cancel during reduction and survive into the
next round. Simple reuse of a previously correct plane is already inconsistent
with the captured values (`25->20`, `22->18`, `1->6`, `0->-6`), because none
of `20,18,6,-6` is a correct S1/S2/S4 partial for the frozen fixture.

The host diagnostic now supports `reuse`, `target-poison`, and
`control-poison` repeat-state modes and can inspect every FP32 plane on every
round. `control-poison` applies the same memset/synchronize cadence to an
equally sized disjoint workspace tail. Target poison is causal only when it is
clean and control poison remains dirty; both clean means cadence masking, and
both dirty excludes ordinary global workspace carry-over.

The completed `254a3e4` bundle made that exclusion.  Reuse, control-poison and
target-poison each failed exactly twice in about 198k--202k geometric exposure,
always in packed-row/S4 and always as one actively written 32-value FP32
producer band.  The target-poison failures contained small fixture values
(`-6`, `-2`, `6`), not poison.  Launch overlap, prior workspace contents and a
missing partial store are therefore excluded.  In this exact binary AP0 and
S2 remained clean, which narrows this codegen but does not erase their older
failures under different code layout.

One locally provable contract mismatch remains at the AIU helper boundary.
`Copy_Traits<PPU0010_AIU_LOAD>::ThrID` is `Layout<_1>` and its source comment
states that one thread issues the opaque bulk load.  The two rvalue
`cute::copy_aiu` paths used by the frozen mixed-input kernels nevertheless let
all 32 lanes of warp 0 issue the same operation.  L224 proves their CuTe
coordinates are identical and separately proves that the candidate physical
issuer count is one; it includes a two-issuer negative.  Coordinate identity
does not establish that duplicate asynchronous bulk issues are legal.  The
`single-aiu-issuer` diagnostic gates only these two overloads to CTA thread 0;
the FA lvalue overload and all descriptors, scalar packed-A copies, stage
geometry, waits/barriers, MMA and partial stores are unchanged.

The frozen closure is:

```bash
FQ_A_STAGE_CANDIDATE=repeat-state \
PROBE_REPEATS=32768 PROBE_ATTEMPTS=2 \
bash tools/run_fq_q4k_custom_split_count_box.sh
```

The issuer-cardinality closure is:

```bash
FQ_A_STAGE_CANDIDATE=single-aiu-issuer \
PROBE_REPEATS=32768 PROBE_ATTEMPTS=2 \
bash tools/run_fq_q4k_custom_split_count_box.sh
```

Because each cell stops at its first failure, probability uses the geometric
exposure `sum(failure_repeat + 1)`, not the requested repeat count. Two
baseline attempts estimate standard-aiu S2/S4 at about 0.078%/0.018% per
launch and packed-row S2/S4 at about 0.427%/0.409%. Each estimate contains
only two events and is an order-of-magnitude diagnostic, not a stable
production rate.

The first reduced closure at `d6e5589` compared baseline with one exact
contract arm:

- `asm-memory-contract` retains the physical x4 opcode and adds actual
  `memory` clobbers to the fp16-A AIU/packed cp.async producer, commit/wait and
  m8 ldmatrix consumer;

The first remaining arm is the source-proved `logical-x2-scalar` path, which
removes the physical x4 swizzle opcode and loads only its two semantic b32
values.  The second adds only `fence_view_async_shared()` after each
`cp_async_wait<Stages-2>()` and before packed decode/CTA publication.  Neither
combines with the failed clobber arm.

L186 proves that reserved arm's 512 `(coord_h,slice,lane,vreg)` addresses against the
independent calibrated PPU0010 model and requires a one-word-offset negative
to turn RED.  Two independent 32768-repeat attempts run for both current arms;
numeric failure never truncates the closure.

The missing inline-asm memory side-effect declarations are a real compiler
contract defect, but the device run rejected it as a complete root.  Baseline
and `asm-memory-contract` each produced zero clean S>1 cells out of eight;
all sixteen dirty cells localized to wrong producer FP32 partials and no other
failure class.  The first captured error did not repeat the narrow
plane-2/index-32 location, but baseline attempt 2 reproduced its complete
value signature at index 576: one aligned 32-output stripe, output
`0x4e80 -> 0x4fc0`, and producer plane 2 FP32
`0x3f800000 -> 0x40c00000` (`1.0 -> 6.0`).  Requiring index 32 made the first
runner verdict a false nonreproduction.  The corrected rule freezes symbol,
fixture, provider, S, plane and value pair while recording the aligned stripe
index separately.  The corrected baseline admission no longer depends on that
one value pair: all four AP0/AP1 x S2/S4 cells must reproduce a producer-
partial corruption across the independent attempts.  Each candidate must
keep custom S1 clean and close every S2/S4 cell.

Pure address arithmetic may be hoisted or commoned only while preserving the
runtime stage value.  The source defect alone did not prove the proposed
stale-stage mechanism, and adding the exact clobbers did not change the device
failure denominator.  The next isolated boundaries are the physical x4
consumer and async-shared proxy visibility.  One uniquely clean arm selects a
repair; two clean arms remain cadence-sensitive and unadjudicated; two dirty
arms retire both hypotheses.

Until one arm closes while the baseline reproduces, the honest claim is: the
legacy shared handoff is unsafe and direct store removes that trigger; the
remaining direct-store incident is a producer-side aligned 32-output band,
its logical CuTe block map and output ownership are exact, ca01dc6 stage
separation did not repair it, and one box run remains to adjudicate the final
two publication/consumer seams.

## Retained regression evidence

- `ppu_splitk_direct_accumulator_store.hpp`: production implementation.
- `PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE`: exact historical negative.
- `l222_fq_splitk_direct_accumulator_store.cu`: exact-once ownership oracle,
  including duplicate-owner and rotated-fragment negatives.
- `ci/check_fq_splitk_partial_path.py`: source contract for the direct path,
  terminal mainloop synchronization, and ordered producer/reducer timing.

The one-off owner, barrier, handoff, and prefix factorial programs are retained
in Git history, not in the shipping tree.
