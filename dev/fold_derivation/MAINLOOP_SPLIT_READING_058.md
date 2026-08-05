# Three-mainloop source reading (INBOX 058)

Scope: a source reading of the three shipping collectives, not a refactor:

* `ppu_mma_aiu_multistage_mixed_input.hpp` (ordinary one-plane)
* `ppu_mma_aiu_mixed_input_2plane.hpp` (two-plane, with independent per-plane folds)
* `ppu_mma_aiu_fold.hpp` (folded one-plane)

## Judgment

The user's diagnosis is right: needing to port `PPU_B_CHUNK` is evidence that the mainloop orchestration was copied
at the wrong boundary. The cp.async ring, register prefetch cadence, scale publication/reload, MMA issue order, and
accumulation are substantially the same algorithm in all three files.

“One mainloop plus three B-provider policies” is close, but is not a sufficient interface by itself. There are at
least four independent choices in the current code:

1. **A provider:** AIU A, compact cp.async A, or packed-row A.
2. **B provider:** ordinary one-plane, folded one-plane, or two-plane. The two-plane provider itself accepts separate
   fold factors, so fold and plane count are composable facts rather than perfectly disjoint variants.
3. **Metadata provider:** fp16 scale/zero planes or packed GGUF metadata, including its publication layout.
4. **Conversion schedule:** eager conversion of a complete delivery or atom-at-a-time chunking.

The shared object should therefore be a pipeline driver with provider hooks, not a monolithic body parameterized only
by a B loader/converter. A three-way B policy is still a sensible first extraction, but putting A and scale branches
back into that policy would merely move the coupling. `PPU_B_CHUNK` belongs to conversion scheduling, with a
provider/config-specific capability predicate.

There is no epilogue difference to preserve here. Fold deliberately presents its physical `(N/F, F*K)` B tile through
an ordinary logical `(N,K)` MMA view, and all three accumulate through the same tiled MMA contract.

## What is genuinely different

| Concern | Ordinary one-plane | Folded one-plane | Two-plane |
|---|---|---|---|
| B storage/view | Plain `(N,K)` | Physical `(N/F,F*K)`, logical `(N,K)` MMA view | Two physical planes, each with its own fold and logical view |
| B copies | One gmem/smem copy atom | One copy atom over folded delivery | Independent B1/B2 atoms, descriptors, tensors, copy slots and stage loads |
| Conversion | One-plane numeric converter | Width-generic chunk emitter or ordinary one-plane converter | `MixGemm2Plane`, including low/high pairing and code bias |
| Copy-step relation | One B copy-slot space | One folded B copy-slot space | Plane 2 has fewer copy slots; `P2_DIV` maps a B1 step to a B2 slot |
| Deferred-conversion hazard | B1 slot for `k_block` survives B1 prefetch of `k_block_next` | Same, but scale stage must be captured before ring advance | B2 can have one slot shared by several B1 steps; eager prefetch can clobber it, so chunked conversion rereads B2 from the captured consume stage |
| A source | AIU, `PPU_A_CPASYNC`, or `PPU_A_PACK` | AIU only | AIU only |
| Metadata source | fp16 or packed GGUF metadata | fp16 only | fp16 or packed GGUF metadata |

The folded mainloop's comment is accurate: after its provider constructs the logical B view, the compute loop is not
fold-specific. The two-plane path has more than a different converter: its provider must own the lifetime rule for
plane 2. In the chunked path it must suppress the normal B2 register prefetch and issue a consume-stage reread just
before conversion. A `load()`/`convert()` policy that does not express register-slot lifetime would be unsound.

Packed metadata is not a B-provider concern. It stages and decodes scale/zero information and can coexist with either
one or two weight planes. Its absence from the fold collective is a capability gap produced by the current class
split, not a mathematical requirement of folded weights.

## What is copied

The following is one algorithm repeated three times:

* initialize `smem_pipe_read = 0` and `smem_pipe_write = Stages-1`;
* issue `Stages-1` prologue copies, copy metadata, fence, and advance the K iterator;
* partition A/B MMA fragments, build the S8 B-load fragment, retile shared-to-register copies, and allocate the
  accumulator outside the provider-specific B details;
* wait for the first stage, publish decoded metadata if present, synchronize, and prefetch register fragments;
* for every static `k_block`, wait/synchronize at the stage boundary, prefetch `k_block_next`, issue the next gmem
  stage at `k_block == 0`, and advance the ring at `K_BLOCK_MAX-2` (or immediately when it is one);
* transform A, present one B atom, and call the same `cute::gemm` in the same order;
* drain cp.async and synchronize.

The scale machinery is also copied, despite comments describing a shared entry point. Each class owns its own
`make_scale_fragment`, host-visible fragment-layout witness, `partition_extra_mma_info`, retile, coarse
`copy_B_and_extra_info`, fine-group reload arithmetic, and scale/zero application. Fold and two-plane have locally
consolidated their eager and chunked arms through `apply_scale_atom`; ordinary still has its separate
`bdq_transform` branches because it also implements scale prefetch and packed/fused metadata. None of those helpers
is physically shared across the three collectives.

`load_init_B` cannot simply become common code: its shape and stride are provider facts. The ring that invokes it and
the MMA loop that consumes its logical fragment can be common.

## Optimisation and flag drift

| Facility | Ordinary | Fold | Two-plane | Classification / consequence |
|---|:---:|:---:|:---:|---|
| `PPU_B_CHUNK` | no | yes | yes | Production performance mechanism; no offline-layout change; should become a tactic field after all providers expose a capability predicate |
| Compact A (`PPU_A_CPASYNC`) | yes | no | no | Runtime/shared-memory choice; performance-only and shape-conditional; either a search axis or an explicitly retired mode |
| Packed-row A (`PPU_A_PACK`) | yes | no | no | Same ownership problem as compact A; belongs to an A provider, not the one-plane B class |
| Scale swizzle | yes | no | no | Shared-memory mapping only; applicability is configuration-dependent; a candidate search field if retained |
| Scale padding | yes | no | no | Shared-memory mapping/size only; recorded measurement says it lost, so retire it or search it, but do not leave a one-copy build flag |
| Scale prefetch | yes | no | no | Scheduling-only; measured small win; a direct example of an optimisation stranded in one mainloop |
| Packed GGUF scale | yes | no | yes | Changes the input artifact/schema, so it is a scheme/capability choice rather than a tactic bit; fold's absence is still artificial |
| Fused scale/zero publication | yes | no | no | Reorders only the shared publication and preserves global bytes; performance-only if retained |
| Split packed decode groups | yes | no | no | Scheduling-only packed-metadata optimisation; its ownership invariants belong to the metadata provider |
| Packed pair converter | yes | no | no | Fast implementation plus an explicit bisection switch; the `PAIR=0` arm is diagnostic, not a production tactic |
| `PPU_PACKED_SCALE_NOP` | yes | no | no | Timing-only ablation with deliberately artificial values; not a tactic |
| `PPU_B_DEQUANT_NOP` | yes | no | no | Timing-only, deliberately wrong results; not a tactic and not a missing optimisation elsewhere |
| MMA probe | no | yes | yes | Diagnostic only |
| Chunk full-fragment bisection | no | yes | no | Diagnostic only; correctly separated from the production chunk selector now |

This inventory supports the user's rule, with one qualification: not every macro should become a search dimension.
Wrong-result ablations and debug probes should be excluded, and artifact-changing packed metadata remains a scheme.
The remaining performance-only choices either need tactic fields plus validity predicates, or should be deleted after
a documented losing measurement.

`PPU_B_CHUNK` is not yet a portable Boolean. Fold additionally requires a width/emitter and fragment-delivery match;
two-plane currently admits every supported pair and asserts its delivery relation later; ordinary has no implementation.
Fold's explicit int4 exclusion is an old performance assumption, not a capability proof. Removing that bit-width
check alone would not make shipping TileK=256 int4 legal: the existing fold emitter also requires the full B fragment
to be exactly one delivery. A generalized atom emitter/capability check is needed before int4 can be measured. The
eventual tactic field therefore needs three outcomes per configuration: unsupported, supported/off, and supported/on.

## Correctness cost already visible

### 1. Scale-copy coverage is fixed in only one copy

Two-plane caps its scale-copy thread layout at the CTA size:

```
Scale_ThrH = min(Scale_TileN / 8, Scale_NumThreads / Scale_TileK)
```

and asserts that requested slots do not exceed CTA threads. Ordinary and fold still use
`(Scale_TileN/8) * Scale_TileK` slots and wrap `thread_idx` modulo that count. The two-plane comment records the prior
failure: at `(64,128,128)`, warp `64x64`, group size 16, 128 slots were requested from 64 threads and half the scale
groups were never loaded, yielding plausible wrong output.

The current four-warp tactic guard excludes that particular two-warp configuration, so this reading does not claim a
known bug in today's admitted shipping set. It does prove that the safety invariant is local to one copy and can
silently recur when the candidate set changes. Slot coverage belongs in a shared scale-copy constructor.

### 2. Deferred conversion needs a consume-stage contract

Fold and two-plane both capture `b_consume_stage` before the ring advances because atom-at-a-time conversion happens
after the advance. Using the current `smem_pipe_read` loaded scales from the next stage when `K_BLOCK_MAX == 1`; the
fold comment records every mismatch selecting group 2 where group 0 was correct.

Two-plane has an additional B2 RAW/clobber hazard. Because plane 2 can have fewer register copy slots, prefetching the
next B2 step can overwrite the only slot before the current chunk converts it. Its chunked branch therefore skips the
normal prefetch and rereads B2 from `b_consume_stage` inside the MMA loop. This is correct today, but it is an invariant
hidden in one orchestration copy. A provider must return data whose lifetime covers conversion, or explicitly request
a consume-time load; the driver must pass an immutable consume-stage token.

### 3. The scale rule has already drifted within copies

Fold documents a former eager/chunk discrepancy: one arm tested `Scale_TileK > 1`, the other `> K_BLOCK_MAX`, which
was right only for the then-int1-only chunk set. Fold and two-plane now route both schedules through a local
`apply_scale_atom`, but ordinary independently implements coarse/fine reload, prefetch, scale, and zero branches.
The three classes still derive `FINE`, atoms-per-group, reload boundaries, and scale fragment access separately.

The comments also record a stale-register bug: copying an owning `make_fragment_like` result into local `auto tCrS`
snapshotted values before a fine-group reload. Fold/two-plane now read the tuple's owning fragment directly. Ordinary
does that in its fine branches but has another implementation. One tested scale policy should own reload and apply.

### 4. Two-plane source pairing has duplicated eager/chunk expressions

The high-plane source must include both delivery/N position and B1-to-B2 step position. The two-plane file records a
past version that left half the folded high-plane vregs unread, producing finite but wrong output. It now centralizes
part of this in `HiPlaneSrc` and gates eager and chunked maps together, but it still has two conversion bodies whose
source and destination views must agree. That relationship belongs inside one two-plane provider implementation and
should be tested once for both schedules.

### 5. Capability skew is silent

Ordinary alone can compact A and tune scale scheduling; fold alone lacks packed metadata; two-plane alone carries the
scale-copy coverage fix. All three classes compile and run, so absence is not reported by the type system or tactic
enumeration. This is already a correctness risk where an absent local guard permits a candidate, and a performance
risk everywhere else.

## Recommended seam, without doing the refactor now

The minimum safe target is conceptually:

```
PpuPipeline<APolicy, BPolicy, MetadataPolicy, ConversionPolicy>
```

The driver owns stage indices, waits/fences/barriers, static K-block traversal, issue order, and accumulator calls.
Policies expose compile-time layouts/storage plus small operations such as `prologue_load`, `prefetch_next`,
`prepare_atom(consume_stage, k_block, atom)`, and `publish_metadata`. `prepare_atom` is where eager and chunked paths
meet; B2's consume-time reread remains private to the two-plane provider. The metadata policy owns the single scale
fragment/reload/apply rule and validates copy-slot coverage.

Do not start by mechanically merging all three classes. First extract dependency-light invariants/helpers that can
be compiled against every current instantiation: scale-copy coverage, consume-stage token, scale group split, and B
provider delivery capability. Then put the existing bodies behind identical policy contracts and differential
compile/layout witnesses. Only after the ordinary provider supports atom preparation should `BChunk` enter the tactic
record and validity query.

No runtime polymorphism is required. These are compile-time policies over already-template-selected kernels, so the
abstraction need not add a branch, allocation, kernel launch, or offline-layout transformation. The hard part is
stating the lifetime and layout contracts; the current source comments supply the counterexamples those contracts
must reject.
