# Dense/grouped mixed-input parity map (INBOX 061-063)

This is the review checkpoint requested before a shared-driver extraction. It inventories the shipping tensor-core
paths and the three collectives they instantiate. Test-only harnesses are mentioned only where they expose a stale
production-shaped launcher; independent test implementations are not treated as duplication to remove.

Verdicts used below:

* **Unified**: dense and grouped already consume one definition.
* **Unifiable**: the distinction is accidental or is duplicated glue; the named seam can be shared.
* **Irreducible**: operator or artifact semantics genuinely differ; the type/API must make the difference explicit.
* **Retire**: diagnostic or legacy code should not become a policy axis.

## Two distinctions that change the diagnosis

`PPU_B_CHUNK` is not literally a grouped-only feature. `CollectiveBuilder` selects a mainloop from the weight
provider, not from the operator:

| Weight provider | Dense scheduler | Grouped scheduler | Collective selected |
|---|---|---|---|
| unfolded one-plane | yes | yes | `ppu_mma_aiu_multistage_mixed_input.hpp` |
| folded one-plane | yes | yes | `ppu_mma_aiu_fold.hpp` |
| two-plane, independently folded | yes | yes | `ppu_mma_aiu_mixed_input_2plane.hpp` |

Thus chunking is present for folded/two-plane formats under **both** operators and absent for ordinary int4 under
**both**. It is provider drift caused by the three copied collective bodies. The gs ladder failure was different:
that was launcher/harness drift before the common builder was reached.

The second distinction is between a tensor's arrangement and a kernel policy. Dense and expert weights are different
tensors and may deliberately use different measured TileK/fold arrangements. Once an artifact says which arrangement
it has, the consumer derivation, metadata decoder, conversion schedule, and legality rules should be shared.

## The actual COARSE predicate and the instantiated configurations

The ordinary collective reaches COARSE exactly when

```text
Scale_TileK <= size<2>(smem_thr_copy_B.retile_D(tCrB_load))
```

The right-hand side is the K-mode of an actual CuTe copy view made from `SmemCopyAtomB` and the S8 `TiledMma`.
It is not `TileK/group_size`, the atom's printed K width, or a group-size threshold.

The real-object probe used the same `TiledMMA`, `PPU0010_TSM_LD_SWZL` atom and `make_tiled_copy_B(...).retile_D(...)`
construction as the collective. Its result for the current generated domains is:

| Instantiations | `copy_view_K` | gs16 `Scale_TileK` | gs32 | gs64 | gs128 |
|---|---:|---:|---:|---:|---:|
| all 227 dense-table rows, int4 TileK=64 | 1 | 4, FINE | 2, FINE | 1, COARSE | 1, COARSE |
| MoE i4 shape product, TileK=64 | 1 | 4, FINE | 2, FINE | 1, COARSE | 1, COARSE |
| MoE i4 shape product, TileK=128 override | 2 | 8, FINE | 4, FINE | 2, COARSE | 1, COARSE |

TileM, TileN, warp shape, and stage count do not change those values in the current generated products. The default
MoE i2/TileK=64 units select the folded collective, and q3/q5/q6 select the two-plane collective, so they do not
instantiate the ordinary assertion site.

The old dense bench omitted `ScaleTileShape`. The collective's default is `(TileN, 1)`, so its actual
`Scale_TileK` was **1 for every runtime `--g`**. The live gs32 failure was therefore `1 <= 1`, not `2 <= 2`.
The fixed bench explicitly instantiates the table for each gs and selects it at runtime. gs32 now takes FINE;
gs64 and gs128 deliberately still take COARSE, whose completed scale copy is now exercised. COARSE additionally
static-asserts that `copy_view_K % Scale_TileK == 0`, and every not-handled marker in the ordinary collective is a
compile-time failure rather than a context-poisoning device assert.

## Operator adapter and specialization parity

| Capability / policy | Dense | Grouped | Verdict and current consequence |
|---|---|---|---|
| Collective selection from code planes and fold | `fpA_intB_ppu` passes `ElementB`, optional `PlaneB2`, and a folded schedule | same tuple/schedule seam | **Unified in `CollectiveBuilder`.** There is no dense-specific converter or packed decoder. |
| Group-size tag and `ceil(TileK/gs)` | `ppu_group_schedule.hpp` | same | **Unified by 1d2a621.** Bench, dense header, grouped header and both `.so` tactic launchers use the trait. This fixes the live gs32 case and makes gs64/128 dense-reachable. |
| Quant mode and `ElementBInfo` tuple assembly | local enum/helper and nested conditional | a second identical enum/helper/conditional | **Unifiable:** one `QuantMode` and `MixedOperandInfo<mode,low,high>` trait. The grouped copy previously forced every two-plane ScaleOnly call into ScaleZero, proving correctness drift here. |
| Folded schedule wrapper | `FPA_FOLD` plus `FPA_SCHED` macro | `MOEG_FOLD` plus `MOEG_SCHED` macro | **Unifiable:** `FoldedSchedule<bits,TileK,BaseSchedule>` beside `delivery_fold_v`. The arithmetic is shared; the wrapper expression is still copied. |
| Interleaved-vs-plain B layout | `n%256 && k%256` in `filter_and_run` | same test | **Unifiable:** a shared layout selector. The shipping ABI already restricts both to the interleaved 256 domain. |
| Mainloop builder arguments | duplicated `CollectiveBuilder` stack | duplicated, byte-for-byte equivalent for the same policy | **Unifiable:** a common mixed-mainloop type factory. This is the smallest extraction and does not touch scheduler code. |
| Dense scheduling | rank-4 universal problem plus `SplitKSerialScheduler`, contiguous D | n/a | **Irreducible:** one problem, serial split-K reduction, non-EVT epilogue. Keep in a dense operator adapter. |
| Grouped scheduling | n/a | `GroupProblemShape`, per-expert D pointers, ragged A offsets, uniform 3-D fast path, prefix-search ragged path | **Irreducible:** these are the operator. Keep in a grouped adapter; pass a consume-ready problem view to the shared mainloop. |
| In-kernel grouped K slicing | dense scheduler owns its own serial split-K | grouped grid.z writes `L*splitk` output planes | **Irreducible implementations**, but one shared K-slice validity helper can state divisibility and full-stride rules. |
| Tactic legality | `DenseSpace` alias of `TacticSpace` | `GroupedSpace` alias of the same type | **Unified structurally.** A type-identity assertion rejects a second implementation; the local gate also compares both table routes and proves both controls fire. |
| Exact instantiated smem/compact validity | `QueryOnly` asks the actual kernel type | same | **Unified property, duplicated call glue.** Preserve exact-type queries; do not replace them with host arithmetic. |
| Packed-metadata contract witness | `ExpectPackedScale` static-asserts `is_packed_scale` | same | **Unified property, duplicated call glue.** This is the right kind of loud scheme guard. |
| Compiled tactic inventory | dense default `64x64:32x32:s3` plus four bootstrap rows | grouped default `16x128:16x16:s2` plus the same four bootstrap rows and CUDA vecdot | **Irreducible selection.** Different operator distributions measured different winners. Separate enumeration ABIs make the difference visible; equality would be wrong. |
| Failure return | `bool` propagates can-implement/workspace/init failure | `launch` returns bool, but outer `filter_and_run` is void and maintains a global fail counter | **Unifiable:** return one result enum from both adapters. Empty grouped launches were timed as winners before the counter existed. |
| Header problem-domain tails | dense helper rejects N/K not divisible by 64 | grouped helper permits the non-interleaved tail domain | **Drift outside the shipping ABI.** The `.so` rejects both unless N/K are multiples of 256. Decide whether the header API intentionally has a wider grouped domain, then encode it in an operator-domain trait. |
| Shipping tensor-core surfaces | scale-first dense; fully-quantized dense and grouped | no scale-first grouped raw `.so` entry | **Surface gap, not a mainloop gap.** The grouped header/harness has scale-first. Nothing currently breaks for native GGUF shipping, but the capability matrix must not imply a grouped `.so` symbol that does not exist. |
| Symmetric GPTQ surface | dense and grouped header paths accept int4/fp16-scale buffers; dense bench now runs all four gs tags | grouped real-weight harness supplies gs128 | **Kernel capability now parallel; production binding remains separate.** GPTQ qtype 1000 is not a row in the GGUF-only `.so` registry, so neither raw device ABI dispatches it by qtype today. |
| Legacy uniform-M `moe_gemm_ppu.cuh` | n/a | test-only batched launcher, int4 only, gs64/128 only, silent void failures | **Retire/migrate.** It is not the shipping grouped scheduler and is a third copy of old dense glue. Do not widen it into another policy owner. |

## Axis 1: A loading

| Capability / policy | Ordinary one-plane | Folded one-plane | Two-plane | Verdict and consequence |
|---|---|---|---|---|
| AIU gmem-to-smem plus swizzled read | yes | yes | yes | **Unifiable orchestration.** Atom/layout types belong to an A provider; ring invocation is copied. |
| Ragged expert base (`group_row_offsets`) | yes | yes | yes | **Copied correctness logic.** Dense passes null; grouped passes offsets. Move offset selection into one A provider used by all three. |
| Compact cp.async A, row capacities 1/2/4 | yes | no (`compact_a_rows=0`) | no (`compact_a_rows=0`) | **Provider drift.** It is independent of B folding/plane count and should be an A-provider choice. Today launchers reject the inactive macro via the type witness, so the gap is loud rather than silently ignored. |
| Packed row-0 A (`PPU_A_PACK`) | yes | no | no | **Provider drift plus current correctness asymmetry.** Grouped rejects `Mmax>1`; dense only excludes it from the universal fallback and can launch a non-default packed build at M>1 even though only row 0 is real. This must be fixed before treating it as a tactic. |
| Packed A pitch | builder read atom and ordinary write path use `PPU_A_PACK_PITCH` | n/a | n/a | **Unified within the provider** by a shared macro plus static assertions; retain as a provider invariant, not an operator setting. |
| A diagnostics (`MOEG_SMEM`) | no dense equivalent | launcher-only grouped log | same selected collective | **Retire as a capability.** It is observability, not kernel behavior; a generic type-witness dumper can replace the grouped-only environment hook. |

## Axis 2: B layout, delivery, and lifetime

| Capability / policy | Ordinary one-plane | Folded one-plane | Two-plane | Verdict and consequence |
|---|---|---|---|---|
| Physical B view | `(N,K)` | `(N/F,F*K)`, logical `(N,K)` | two physical views with independent F1/F2 | **Irreducible provider data.** Express as B-provider layouts and one logical atom interface. |
| Fold derivation and delivery legality | `F=1` | `delivery_fold_v`, `CheckDelivery` | same per plane | **Unified derivation already;** duplicated launcher stride construction is unifiable. |
| Global B descriptor/stride initialization | local `load_init_B` | another copy plus folded pitch | B1 and B2 copies | **Unifiable behind provider hooks.** The driver should not know physical row count or pitch. |
| Prologue/ring/prefetch cadence | local body | copied body | copied body | **Unifiable driver.** This is the largest identical region and the source of cross-copy fixes not propagating. |
| Captured consume stage | implicit/eager only | required for deferred conversion after ring advance | required, plus B2 reread when a shared slot would be clobbered | **Irreducible lifetime contract, shared driver token.** A provider must declare whether data survives prefetch or requires a consume-time load. |
| Plane-2 copy-slot ratio (`P2_DIV`) | n/a | n/a | yes | **Irreducible provider mapping.** Keep entirely inside the two-plane provider. |
| Offline arrangement choice | operator's tensor header/config | operator's tensor header/config | per-plane header/config | **Irreducible artifact choice.** Dense and MoE may choose different measured TileK/fold; producer and consumer must share the artifact header, not a universal operator value. |

## Axis 3: metadata sourcing and scale policy

| Capability / policy | Ordinary one-plane | Folded one-plane | Two-plane | Verdict and consequence |
|---|---|---|---|---|
| fp16 scale and optional zero planes | yes | yes | yes | **Copied and unifiable.** One metadata provider should own tiling, partition, copy, publication and fragment access. |
| Per-column and groupwise schedules | yes | yes | yes | **Unified tags**, but each collective still reimplements the reload arithmetic. |
| COARSE/FINE predicate and application | local implementation, now completed and compile-time guarded | second implementation | third implementation | **Correctness-critical drift.** The live unconditional assert existed only in ordinary; fold/two-plane happened to handle their supported modes. Extract before performance work. |
| Scale-copy thread coverage | uncapped original construction | uncapped original construction | caps slots to CTA threads and asserts coverage | **Correctness drift.** A formerly admitted two-warp shape loaded only half its groups until the two-plane copy was fixed. The current tactic gate masks that exact row; changing the candidate set can revive it. |
| Scale fragment construction/layout witness | shared `MixedMetadataPolicy` | shared | shared | **Unified.** `PPU_SCALE_FRAGMENT_API` remains a harness stale-submodule gate and now lives beside the one implementation. |
| Packed GGUF unit source | yes | no | yes | **Provider drift.** Packed metadata is orthogonal to folded B; fold's absence is artificial. This changes the artifact/schema, so it is a scheme capability rather than a tactic bit. |
| Packed-unit format selection | `PPU_PACKED_FORMAT` trait | n/a | same trait | **Unified trait where present.** Five format-specific builds are a deliberate artifact contract, not five decoders. |
| Scale shared-memory swizzle | yes | no | no | **Performance-only drift.** No offline-layout change; make a metadata-policy tactic field or retire from measurements. |
| Scale padding | yes | no | no | **Performance-only drift.** Recorded loser; retire or expose as a searched metadata policy, not a one-copy macro. |
| Fine scale prefetch | yes | no | no | **Performance-only drift.** Scheduling-only, measured small win, and stranded in one body. |
| Fused scale/zero publication | yes | no | no | **Performance-only drift within packed metadata.** Same global bytes; metadata provider policy. |
| Split packed decode groups | yes | no | no | **Performance-only drift.** Scheduling-only packed-provider policy. |
| Packed pair converter | fast pair plus scalar bisection | n/a | basic packed decode path | **Implementation drift.** Keep the fast implementation in the metadata provider; `PPU_PACKED_PAIR=0` is diagnostic, not a tactic. |
| Not-handled failure mode | all ordinary markers are dependent `static_assert`s after a7a8ea91 | 12 `assert(false)` sites | 12 `assert(false)` sites | **Correctness/diagnostic drift, high priority.** An unsupported folded/two-plane mode can still poison the device context and surface as a later occupancy error. Convert these while extracting the metadata policy. |

## Axis 4: eager versus chunked conversion

| Capability / policy | Ordinary one-plane | Folded one-plane | Two-plane | Verdict and consequence |
|---|---|---|---|---|
| Eager full-delivery conversion | yes | yes | yes | **Unifiable driver schedule** with provider-specific conversion. |
| Atom-at-a-time `PPU_B_CHUNK` | no | yes for the admitted delivery/emitter predicate | yes, including consume-stage B2 reread | **Performance drift.** It changes no resident bytes and must become a tactic field once ordinary exposes `prepare_atom`. It is not safe as a bare Boolean; capability is provider/config dependent. |
| One scale rule shared by eager/chunk within a file | ordinary has independent branches | local `apply_scale_atom` | local `apply_scale_atom` | **Correctness drift.** Fold once used different FINE predicates between eager/chunk. One metadata policy must apply the atom regardless of conversion schedule. |
| Numeric conversion | single-plane converter | folded width-generic emitter | low/high `MixGemm2Plane` composition | **Irreducible provider implementation.** All must return the same logical fp16 B-atom contract. |
| Chunk full-fragment bisection | no | `PPU_B_CHUNK_BISECT` | no | **Retire as a capability.** Diagnostic only. |
| MMA probe | no | `PPU_MMA_PROBE` | `PPU_MMA_PROBE` | **Retire as a capability.** A common compile/layout witness can serve debugging. |
| Dequant NOP | `PPU_B_DEQUANT_NOP`, deliberately wrong | no | no | **Retire as a capability.** Timing ablation only; never a tactic. |
| Packed-scale NOP | ordinary packed path, deliberately artificial | no packed provider | no equivalent | **Retire as a capability.** Timing ablation only. |

## Tuning flags: disposition

The flags that preserve resident bytes are not allowed to remain global, per-collective build choices. Their target
homes are:

| Flag/facility | Disposition |
|---|---|
| `PPU_A_CPASYNC` capacity, `PPU_A_PACK` | A-provider tactic fields with exact type validity; fix dense M>1 first |
| `PPU_B_CHUNK` | conversion-schedule tactic field with provider capability result `{unsupported, off, on}` |
| `PPU_SCALE_SWIZZLE`, `PPU_SCALE_PAD`, `PPU_SCALE_PREFETCH` | metadata-policy fields; remove measured losers rather than preserve dead axes |
| `PPU_PACKED_SCALE_FUSED`, `PPU_PACKED_SPLIT_GROUPS` | packed-metadata scheduling fields |
| `PPU_PACKED_SCALE`, `PPU_PACKED_FORMAT` | scheme/artifact selection, not tactics; eventually separate format-specialized library variants or generated registrations |
| `PPU_PACKED_PAIR=0`, `PPU_B_CHUNK_BISECT`, `PPU_MMA_PROBE`, `PPU_B_DEQUANT_NOP`, `PPU_PACKED_SCALE_NOP` | diagnostics/ablations; keep out of capability and tactic manifests |
| `MOEG_FORCE3D`, `MOEG_PROBE`, `MOEG_SMEM` | grouped scheduler diagnostics; not mainloop capabilities |

## Extraction order proposed from this map

1. **Shared type factory at the launcher boundary.** Move `QuantMode`, operand-tuple construction,
   `FoldedSchedule`, group schedule/scale shape, plane bit traits, and mainloop builder aliases into one header.
   Dense and grouped retain only problem/scheduler/epilogue/arguments. This is dependency-light and immediately
   prevents another gs/tuple/fold launcher divergence.
2. **Shared metadata policy.** Extract scale copy coverage, fragment construction, COARSE/FINE split, reload and
   apply. Convert all remaining not-handled device asserts to dependent static assertions. This addresses the known
   correctness divergences before changing the ring.
3. **A and B providers behind a common atom contract.** A provider owns ordinary/compact/packed loading and ragged
   base selection. B provider owns physical layouts, descriptors, plane mapping and lifetime. Each produces logical
   A/B atoms.
4. **One pipeline driver.** Move prologue, stage ring, prefetch cadence, static K traversal, MMA issue and drain into
   one body. It passes an immutable consume-stage token; provider hooks decide eager data lifetime or consume-time
   reread.
5. **Turn surviving performance-only macros into tactic fields.** Only after every provider exposes capability and
   the exact instantiated type answers validity.

No step requires runtime polymorphism or changes the offline artifact by itself.

## Guard required after extraction

The guard should be a compile-time `MixedPolicyDescriptor` produced by the shared factory, containing at least:

```text
A provider, B provider/F1/F2, metadata provider/format,
conversion schedule, group tag, ScaleTileK, compact-A capacity,
chunk capability, packed-metadata capability
```

For a common `(format, tile, group size, build policy)`, both operator adapters must instantiate that same descriptor;
only scheduler, epilogue, problem view and output view may differ. Tactic legality is shared by type identity; the
separate mixed-policy parity gate still enumerates descriptors and uses its own planted-divergence proof.

The compiled tactic inventories remain intentionally separate. Their public enumeration ABI is the loud mechanism:
different defaults/rows are visible data, not hidden policy booleans. Artifact-changing metadata formats likewise
remain separate scheme registrations. Everything else in the descriptor is either shared by construction or rejected
at compile time.

## Extraction checkpoint

Step 1 is now implemented in `ppu_mixed_policy.hpp`. Dense, grouped, and the dense sweep no longer construct their
own operand tuple, low-plane fold wrapper, A/B layouts, alignments, or mixed-input `CollectiveBuilder` argument list.
They pass their common policy inputs to `MainloopPolicy`; only their problem scheduler and epilogue remain local.
The public namespace-local aliases preserve existing caller spelling while making both adapters name the same type.

`MixedPolicyDescriptor` records the selected collective, base and folded schedules, operand tuple, A/B providers,
tile/warp/scale shapes, plane widths/folds, quant mode, compact-A capacity, packed-metadata activation, and whether
atom-at-a-time conversion is active. `KernelPolicyGuard` is the one launch-site guard for instantiated thread count,
scale-copy coverage, operator tactic legality, and both delivery checks. The dense-only quarantine remains explicit
as the `TacticSpace` argument; it is not smuggled into the otherwise common descriptor.

`l112_mixed_policy_parity.cu` compares dense and grouped descriptors over ordinary, folded, two-plane, ScaleOnly,
and ScaleZero instantiations. The local tier also compiles a planted grouped-only B-layout change and requires that
the descriptor equality assertion reject it. This is the step-1/launcher-boundary guard; it does not claim that the
three collective bodies are shared yet.

Step 2's common metadata rules are now implemented in
`actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_metadata_policy.hpp`. Every collective publishes one
`MetadataPolicy` type.
That type owns the scale-copy coverage assertion, scale-fragment construction/layout, the COARSE/FINE predicate and
divisibility invariants, the group-boundary/index split, flat-versus-`(group,stage)` addressing, and the positional
scale/zero reload contract. Fold and two-plane also use its common scale/zero application primitive for both eager
and atom-at-a-time conversion. Ordinary's packed-scale prefetch and deliberately-wrong dequant timing ablation remain
local conversion-provider behavior; they consume the shared FINE policy and reload contract rather than redefining
either.

`l113_mixed_metadata_policy.cu` pins COARSE/FINE boundary/index behavior and both storage-address policy types.
`MixedPolicyDescriptor::MetadataPolicyType` plus l112 extends the operator parity guard through that seam.

Steps 3 and 4 are now implemented as one hook-based provider boundary in
`actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_pipeline.hpp`. The ordinary, folded and two-plane
collectives supply five
storage-specific operations: bind a read-stage view, publish completed asynchronous data, prepare the next logical
A/B delivery, prefetch the next physical tile, and consume the prepared atom(s). Those hooks are the provider policy
parameters; the shared driver owns the one-time register prime, wait/barrier placement, static K-block traversal,
global-to-shared issue point, fence, iterator advance, ring advance, immutable pre-advance consume-stage token and
final drain. The prime bit passed to `prepare` preserves the two-plane chunk provider's original one-time plane-2
load while allowing its steady-state consume-time reread.

Physical B shapes/descriptors, fold mapping, plane-slot mapping and numeric conversion remain in their provider
collective, as do ordinary/compact/packed A loads. They are deliberately not restated in the driver. What used to be
three complete pipeline bodies is now three provider hook packs around one cadence body (162 net lines removed from
the collectives in this extraction).

`MixedPolicyDescriptor::PipelineDriverType` makes the driver part of the dense/grouped compile-time policy witness.
The local-tier shared-pipeline lint additionally requires exactly one delegation from each shipping mixed collective,
forbids local stage counters/waits/K-tile loops, and demonstrates the check by rejecting a planted bypass. Together
with l112's planted adapter-policy drift, this closes 061's guard: adding an operator-local mainloop policy either
changes the shared descriptor and fails compilation, or bypasses the shared cadence and fails the local tier.

The performance-only macro-to-tactic-field work in step 5 remains a tuning-interface cleanup, not an operator parity
hole: today those switches select behavior inside a collective that both adapters instantiate through the same
`MainloopPolicy`. They should still move into typed tactic fields before being admitted to a searched inventory.
