# Q4_K fully-quantized Split-K partial epilogue trigger record

Status: the production correctness repair is closed, but the internal root is
not.  The prefix factorial localizes a corrupting trigger inside the custom
fixed Split-K compiled context.  It does not yet explain why an ordinary S=1
epilogue is clean, because the public S=1 route uses a different kernel,
output type and epilogue.  Do not label the compiler/hardware mechanism root
cause complete until the same custom kernel has been run at runtime S=1.

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

The production repair stores the completed accumulator directly through the
real `TiledMma::partition_C` ownership into the FP32 partial plane.  This is a
same-type internal store: no conversion or output-layout redistribution needs
the shared round trip.

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

## Narrowest proven trigger and remaining causal gap

The final hash-bound device closure reported:

```text
legacy shared output: 776 / 8192 bad samples
production direct:    raw-bit exact, standard-aiu + packed-row, S=2 + S=4
verdict:              SHARED_STORE_BACKEND_OR_FOOTPRINT_CAUSAL
```

Exact-symbol disassembly kept 16 MMA instructions, 34 TSM stores, 67 TSM
loads, and zero reported stack in the clean direct binary.  The first failing
flat-constant binary kept the same MMA/load counts and added exactly one
`tsm.st.b32x4` store.  Thus, within the frozen custom S>1 kernel:

- removing the unnecessary shared-memory partial-output handoff is a closed,
  performance-neutral production repair;
- the first isolated operation that can trigger corruption is the additional
  vector TSM-store lowering (or the compiler/hardware footprint it changes);
- this is not yet a cross-context micro-root: ordinary S=1 correctness does
  not isolate the split count because it also changes the compiled kernel,
  FP16 versus FP32 output, copy ownership and register/shared footprint;
- available SDK evidence cannot yet distinguish whether the necessary
  condition is the custom kernel context itself or its runtime S>1
  decomposition.

The direct-store fix does not change accumulation or reduction order.  Its
same-run packed-row S4 median was 11.300 us versus 11.320 us for the legacy
shared path (`-0.177%`), within the preregistered 3% regression limit.

## Similar-path audit

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

The remaining decisive experiment is implemented by
`tools/run_fq_q4k_custom_split_count_box.sh`.  It launches the exact generated
AP0/AP1 `GemmUniversalMixedInputSplitKParallel` symbols at runtime S=1/2/4,
once with the production direct store and once with the legacy shared
negative.  S=1 uses one FP32 partial plane plus the same deterministic reducer;
the shipping S=1 kernel is deliberately bypassed.

Its two decisive outcomes are:

- `RUNTIME_SPLIT_DECOMPOSITION_NECESSARY`: legacy custom-S1 is clean while
  custom-S2/S4 reproduce.  The S>1 K-range/work-grid decomposition is a
  necessary trigger for this row (not merely an arbitrary shared store).
- `CUSTOM_KERNEL_CONTEXT_SUFFICIENT`: legacy custom-S1 also reproduces.  The
  split count is not necessary; the shipping S1 control stayed clean because
  its compiled kernel/output epilogue was different.

Until one of those verdicts is hash-bound on device, the honest claim is only:
the legacy shared handoff is unsafe in the observed custom fixed Split-K S>1
context, while the direct-store repair is exact and non-regressing.

## Retained regression evidence

- `ppu_splitk_direct_accumulator_store.hpp`: production implementation.
- `PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE`: exact historical negative.
- `l222_fq_splitk_direct_accumulator_store.cu`: exact-once ownership oracle,
  including duplicate-owner and rotated-fragment negatives.
- `ci/check_fq_splitk_partial_path.py`: source contract for the direct path,
  terminal mainloop synchronization, and ordered producer/reducer timing.

The one-off owner, barrier, handoff, and prefix factorial programs are retained
in Git history, not in the shipping tree.
