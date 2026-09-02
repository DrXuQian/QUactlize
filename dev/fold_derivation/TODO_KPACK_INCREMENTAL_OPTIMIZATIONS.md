# K-pack incremental optimization backlog

This backlog starts after the exhaustive canonical K-pack denominator frozen
at `adbd5f8`.  Its executable build campaign was restarted at `754130e` after
repairing the grouped `DeliveryN` registry macro; the repair changes registry
wiring, not the denominator.  The items below are new implementation axes:
they require their own type identity and correctness closure before they may
enter an incremental performance sweep.  They do not justify changing or
pruning that baseline denominator.

## Candidate axes

| candidate | current canonical state | offline arrangement impact | admission work |
|---|---|---|---|
| K-pack `BChunk` | Both K-pack policies fix `BChunkRequest=0` and require `atom_at_a_time=false`. `DeliveryN=16/32/64` is the currently swept K-pack delivery axis. | None if it is only a conversion schedule; a new mapping is forbidden unless the global byte coordinate changes. | First prove that a K-pack BChunk schedule is distinct from `DeliveryN`, then add a typed policy, emitted-instruction check, raw-bit closure, and incremental timing. Do not revive a binary-global diagnostic macro. |
| BC/SIMT decode rows per warp | No canonical K-pack BC reader; `RPW=1/2/4/8` is therefore structurally unavailable in the baseline campaign. | The canonical bytes can remain unchanged if the BC reader consumes the versioned K-pack arrangement directly. | Implement the K-pack BC reader, close dense and grouped raw-bit correctness, then sweep `RPW=1/2/4/8` against the tensor-core decode winner. |
| AIU-plain global-to-shared plus `UniversalCopy` shared-to-register | Current K-pack uses the production AIU/swizzle delivery and swizzle matrix reader. The complete Q4 consumer map has now been derived and is not byte-compatible with the canonical K-pack4 map. PPU0010 m8 and m16 publish the same MMA-B fragment layout, so this Q4 map can cover decode and prefill in our collective. | A distinct experimental Q4 arrangement already exists: layout 3, mapping `0x51344e3136440001`, physical `[K/16][2*N] uint32`. It is an `N16 x K64` atom derived from our converter/MMA chain, not DeepGEMM's permutation. Q2/Q3/Q5/Q6 have no such arrangement yet and must each be derived from their actual one- or two-plane converter chain. | Keep layout 3 fail-closed until exact PPU raw-bit delivery and full converter/MMA numeric closure pass for dense and grouped kernels, then run isomorphic resource/timing A/B. Derive and version every additional qtype independently; never infer its bytes from Q4. |
| Q4 metadata publication | Q4 dense uses interleaved half2 publication; generic Q2/Q3/Q5/Q6 and grouped paths retain separate half planes. This is fixed per current product type, not swept. | Weight bytes do not change. Shared/workspace layout and descriptor identity can change. | Add a typed publication axis only where both implementations are correct; compare store/load conflicts, instructions, shared bytes, and end-to-end time. |
| Packed-A coverage | AP1 is admitted only for proved Q2/Q4 `TM=8, WM=8` decode rows; all other rows are AP0. | Weight arrangement is unchanged. | Extend one format/geometry at a time, with exact A-stage raw-bit closure before timing. |
| Fused Split-K completion | The baseline ranks producer time plus the versioned exact reducer lookup. | No weight-layout change; workspace/launch ABI changes. | Admit fused last-arriver only after counter reset/lifecycle correctness and compare complete end-to-end launches with the separate reducer. |
| N-stage expansion / scheduler order | The baseline expands current persistent grids and grouped persistent/nonpersistent policies, but it does not implement DeepGEMM's `N_EXPAND` pipeline. | Normally unchanged. | Add it as a scheduler type, not a hidden runtime heuristic; validate coverage, tail handling, and paired timing by shape. |

Already covered by the baseline and not backlog items: full legal
`TM/TN/TK`, `WM/WN`, stages `2/3/4/6/8/12`, K-pack `DeliveryN=16/32/64`,
legal AP0/AP1 rows, dense Split-K `S=1/2/4/8`, and runtime persistent grid
space.  Xplane `ArtifactTileK/Fold` is intentionally not a product axis.

## Deferred operand-provider boundary

Do not refactor the collective while the exhaustive canonical build campaign,
the K-pack-only heuristic sweep, or the layout-3 device admission is in
flight.  Those results are source-authority bound; mixing a structural rewrite
into the same closure would make code-generation changes indistinguishable
from configuration or layout effects.  Until those campaigns close, this item
is documentation only and must not change the mainloop.

The current collective is acceptable while it has one production arrangement
and one matching delivery path.  Introduce a provider abstraction only after a
second delivery path has passed correctness and is intended to remain.  The
motivation is clean ownership and impossible invalid pairings, not an assumed
performance gain.  Use explicit names rather than a generic `Legacy` label:

- `KpackSwizzleProvider`: the current canonical K-pack AIU/swizzle path;
- `N16DirectProvider`: Q4 layout 3 plus AIU-plain and `UniversalCopy`;
- `DeepGemmProvider`: a future DeepGEMM-compatible arrangement and its matched
  register/dequant reader, if independently admitted;
- `CpAsyncUniversalProvider`: only if a separately derived cp.async path is
  retained after measurement.

Each provider must own one compile-time-bound chain: arrangement and mapping
identity, global-to-shared writer, shared layout, shared-to-register reader,
raw register layout, decode, and placement into the MMA-ready B fragment.  The
mainloop continues to own pipeline/stage orchestration, MMA issue order,
accumulators, and epilogue.  A layout must not be selectable independently of
its reader.

Admission order after the frozen campaigns complete:

1. Wrap the current path as `KpackSwizzleProvider` and prove identical emitted
   instruction counts, registers, spills, shared/workspace ABI, raw bits, and
   paired timing.  The wrapper is rejected if it changes code generation.
2. Add `N16DirectProvider` only after its standalone PPU probe and dense/grouped
   numerical closures pass; compare it with the wrapped current path under an
   identical tactic and launch.
3. Add other providers as matched arrangement/reader pairs, with both crossed
   pairings planted as RED controls.
4. Keep provider selection compile-time.  Do not add runtime branches or
   binary-global experiment flags to the hot mainloop.

Provider-specific delivery may later recover a performance gap by eliminating
bridge permutations or improving the load/decode dependency chain, but that is
a separate device measurement.  The abstraction itself has a zero-overhead,
no-regression requirement.

## Post-campaign architecture audit

This audit is frozen as documentation while the `754130e` exhaustive build,
the K-pack heuristic run, and layout-3 device admission are outstanding.  `P0`
below means "close before admitting a second production delivery provider",
not "change the source under the running campaign".  `P1` is required before
the curated main product is cut; `P2` may remain as an isolated compatibility
layer if it is not reachable from the product path.

The current canonical K-pack implementation is fail-closed and is not known to
have a silent correctness defect.  In particular, shipping schedules are still
statically locked to the existing AIU-swizzle/TSM-swizzle pair, layout 3 is
rejected by compute admission, and `BoundBDelivery` compares the concrete
shared layout, element, stage bytes, atom sizes, and alignment.  Preserve those
properties while addressing the debts below.

| priority | debt in the current tree | extension risk | required closure |
|---|---|---|---|
| P0 | The B-delivery seam is only partially integrated.  The single-plane collective dispatches G2S through `issue_mixed_b_g2s`, but S2R, converter and MMA-fragment placement still use the fixed collective body.  The two-plane collective mostly carries a policy plus a static lock and still issues its old copies directly. | A second provider would add special cases on both sides of the K loop, and a change could reach one-plane but not two-plane formats. | Carry one complete bound provider from the builder through G2S, physical shared storage, S2R, raw-register ownership, decode and MMA-fragment placement.  Keep stage advancement, wait/barrier order and MMA orchestration in the mainloop. |
| P0 | `ComposedBDelivery` declares compatibility from the coarse `SharedEncoding::{Plain,AiuSwizzled}` tag.  Many byte-incompatible plain layouts satisfy that tag, and the schedule carries this tag composition rather than the concrete `BoundBDelivery`. | AIU-plain or cp.async could be paired with the wrong Universal reader even though both say `Plain`. | Make the concrete bound provider, including its physical shared contract and arrangement identity, the selected type.  A coarse encoding may remain diagnostic metadata but must not admit a pairing. |
| P0 | Arrangement identity is checked separately from the compile-time reader.  `ProductionBDelivery` is one global default, while `mapping_id` lives in runtime routing. | A valid reader type can still be launched on bytes belonging to another arrangement unless every host branch is updated perfectly. | Give each provider an exact arrangement/mapping trait and make each compiled tactic row name the provider.  Runtime admission must match `(qtype, descriptor, provider, tactic)` before launch. |
| P0 | The complete C offline producer depends on compute policy: `ppu_unit_pack.cpp::product_arrangement` chooses layout from qtype and calls `matches_compiled_tactic` with the format-wide fully-quantized TileK.  Plane-level `ppu_dense_layout.cu` can prepare/recover layout 3 while the complete ABI rejects it. | Adding an offline arrangement requires editing a current-reader decision, and a producer capability can disappear when tactics change. | Split pure `validate_offline_arrangement` from `provider_accepts_arrangement`.  Complete prepare/recover must depend only on the former. |
| P0 | There are two complete GGUF-to-artifact implementations.  `ppu_unit_pack.cpp` exports complete prepare/recover, while `gguf_prepass_ops.cpp` independently unpacks codes, loops over experts, calls plane placement and packs units.  The torch `Api` loads only the plane-level producer even though the runtime bundle requires the complete ABI. | Python/torch and loader integrations can produce different bytes or support different arrangements. | Make the complete C ABI the sole production implementation.  Torch allocates tensors, calls it and attaches the returned canonical descriptor; independent Python/reference code remains an oracle, not a second shipping producer. |
| P0 | Config records v3/v4 contain geometry and Split-K but no stable tactic/provider identity.  Generic dense, grouped and Q4 use separate registries; TileK is still partly a format-wide property. | Two providers with identical geometry cannot be selected, validated or measured independently without overloading a string or adding another registry. | Generate one structured tactic record containing stable `tactic_id`, arrangement/provider ID, TM/TN/TK, WM/WN, stages, scheduler/Split-K, DeliveryN, A provider, BChunk and metadata publication.  Names are display-only; TileK belongs to a row, not the GGUF format. |
| P0 | Workspace query and launch do not consume one resolved plan.  Queries reselect null/default policy, while launch can accept an explicit config; grouped workspace validation is tied to the default. | A new provider or scheduler with different partial/counter/shared requirements can be launched with space computed for another row. | Add a host `select_plan` result containing stable tactic ID, exact workspace bytes/alignment and provider capability.  Device launch consumes and validates that same plan. |
| P0 | The committed measured policy is dense exact-point lookup only, grouped falls back to its default, Q4 dense uses a separate policy, and the table is marked `unverified-sdk`. | The object called a heuristic does not yet route the full product domain, cannot compare new providers, and unmeasured intermediate M silently uses a generic fallback. | Regenerate from accepted SDK evidence.  Emit dense and grouped exact tables plus an explicitly validated bounded fallback, including Q4.  Never interpolate merely because two sampled endpoints agree; validate the regret envelope first. |
| P1 | `ppu_dense_backend.cu` is a 3168-line dispatcher with repeated `PPU_PACKED_FORMAT`/`PPU_PACKED_SCALE` ladders and several layout branches whose final fall-through means Q4. | Every new qtype/layout/provider must be added consistently to inventory, validity, workspace, host launch and device launch; omission may select another route. | Use an exhaustive `(layout,qtype) -> RouteTraits` authority and per-format/provider instantiation TUs.  No semantic `else means Q4`, and no repeated preprocessor ladder in each ABI entry. |
| P1 | Policy vocabulary overlaps: `BProvider`, `WeightLayout`, `BDelivery`, nested schedule wrappers, boolean axes and zero sentinels all describe related pieces.  Missing `BDelivery` propagation may silently become `void` and then the production default. | Additional axes multiply template parameters and allow a policy field to be lost while unwrapping schedules. | Rename the axes by responsibility (`CodePlaneTopology`, `Arrangement`, `DeliveryProvider`, `MetadataPipeline`, `AProvider`, `Scheduler`).  Require explicit types; remove `void` fallback, magic zero meanings and a global `ProductionBDelivery`. |
| P1 | Single-plane and two-plane collectives are 2175 and 1980 lines and independently implement closely related metadata publication, B preparation and transform logic. | A correctness fix or provider hook can land in one format family and be omitted from the other, while merging the whole bodies would endanger code generation. | Extract only stateless semantic helpers and typed policies with per-step ISA/resource equality gates.  Keep genuinely different one-plane/two-plane register pipelines separate. |
| P1 | Shipping types retain dormant diagnostics: grouped Arguments/Params carry `probe` and read `MOEG_PROBE`; Stream-K carries `DiagnosticState*` and diagnostic branches; Split-K prepared objects retain fused/publish-only diagnostic state and methods. | Product ABI and potentially generated code depend on debug facilities; future scheduling changes become harder to compare. | Move diagnostics into test-only wrapper kernels/types.  The selected product kernel and public Arguments/Params contain no dormant probe state.  Prove the removal with codegen/resource and device timing gates where it touches a kernel type. |
| P1 | Product and experiment builds share a 2880-line CMake graph, global `PPU_EXTRA_DEFS`, and source lookup across product, tests, benchmarks and `dev`.  The optional-source helper currently performs `list(GET _hits 0 ...)` before handling zero hits. | New providers expand an already fragile Cartesian build, and a diagnostic define or source-resolution bug can alter a product binary. | Split a minimal product CMake target with explicit sources/includes/fixed definitions from opt-in sweep/probe targets.  Fix the zero-hit ordering locally after the frozen campaign; never apply experimental definitions globally to the product library. |
| P1 | Develop exposes Xplane and unadmitted layout 3 through public constants, routes and offline transforms. | A main port can accidentally retain archived bytes or advertise a producer with no shipping reader. | Main contains only the canonical K-pack descriptors and admitted readers.  Keep Xplane and unadmitted layouts in development/reference tooling until a product decision explicitly replaces the canonical arrangement. |
| P2 | C ABI has config v1-v4, several near-duplicate host/device launch signatures, inconsistent argument order, magic integer error codes, and a manually extended `Api`/`dlsym` table.  `PlacedArtifact` subclasses tuple and loses its descriptor when converted to a plain sequence. | Every new capability requires synchronized edits across headers, loader and torch ops, while callers can strip identity accidentally. | Preserve old symbols only in a compatibility shim.  A future versioned request/function table uses one field order and typed status values; Python uses an immutable `PlacedWeight` record containing planes, descriptor, qtype, N/K and experts. |
| P2 | Format and mapping facts still have hand-maintained copies in C/C++ and Python despite `ppu_format_config.inc`. | Adding a format/layout can produce plausible but divergent descriptors. | Generate production descriptors and language bindings from one machine-readable registry.  Keep the independent reference converter hand-written so it remains a real second oracle. |

### Target boundary

The intended dependency direction after those closures is:

```text
FormatTraits<QType>
  bits, high bits, group size, packed-unit semantics
          |
ArrangementTraits<MappingId>
  canonical descriptor, plane byte maps, prepare/recover, shape legality
          |
DeliveryProvider<Arrangement, Tactic>
  G2S writer, shared contract, S2R reader, raw fragment, decode/placement
          |
MainloopPolicy
  A provider, metadata pipeline, B provider, pipeline, scheduler, tile
          |
TacticRecord -> SelectedPlan(workspace) -> launch
```

Metadata unit preparation/publication remains a reusable `MetadataPipeline`;
the B provider receives its typed metadata view when its register layout needs
to apply scale/zero.  It must not duplicate GGUF metadata parsing.

One resident tensor has one arrangement.  A runtime heuristic may choose only
tactics whose provider consumes that exact mapping.  It cannot choose between
N16-direct and DeepGEMM-compatible providers if they require different bytes,
unless the model deliberately stores or repacks both artifacts.  Keep
`canonical_arrangement(qtype)` (what new checkpoints write) separate from
`supported_arrangements(qtype, operator)` (what an admitted reader can consume).

After the frozen campaigns finish, either complete this boundary because a
second provider has been admitted, or remove the inert candidate aliases and
collapse back to the single current path.  A permanently half-integrated
provider scaffold is not an acceptable product endpoint.

## AIU + UniversalCopy arrangement proof

Changing the copy atom does **not** by itself prove an offline format.  The
decision is made by the composed coordinate map, not by the instruction name.
For the concrete Q4 path, L240 and L244 enumerated these maps over one complete
transport tile.  PPU0010's m8 and m16 fp16 MMA atoms have the same `BLayout`,
so the offline map is not an M-tile choice:

1. `O(n,k,plane) -> byte` -- the versioned offline arrangement.
2. `G(byte,lane,stage) -> smem` -- the AIU global-to-shared delivery.
3. `R(smem,lane,slot) -> register` -- the shared-to-register copy.
4. `F(n,k,plane,lane) -> mma_fragment` -- the fragment expected by the
   converter and MMA.

The existing Q4 arrangement would have been reusable iff
`R(G(O(...))) == F(...)` were a bijection over every plane and transport-tile
edge.  That equality is false for Q4 K-pack4 layout 1.  The resulting direct
map is therefore versioned separately as `q4-n16k64-direct`:

```
logical atom       N16 x K64 Q4
physical tensor    [K/16][2*N] uint32, N-word contiguous
atom bit map       p[0..9] = {k3,k4,k0,k5,n3,k1,k2,n0,n1,n2}
layout             3
mapping id          0x51344e3136440001
```

This derivation is specific to Q4.  Q2/Q3/Q5/Q6 use different converter
widths, and Q3/Q5/Q6 also have a high-plane composition whose register source
order participates in `R` and `F`.  Each therefore needs a new complete
enumeration and a new mapping identity; changing only the bit count in this
formula would not be a proof.

The proof must still report global transaction alignment and shared bank
multiplicity; a bit-correct gather that destroys coalescing is not an
admissible performance candidate.

Run the investigation in this order:

1. **Closed locally:** L239--L245 bind the AIU-plain writer, UniversalCopy
   reader, converter emission and real m8 MMA-B fragment; layout-1 reuse is a
   planted RED, and layout 3 has independent prepare/recover oracles.
2. **Built for device admission:** L248 requires exactly four AIU-plain writes
   and sixteen `tsm.ld.b32x4` reads and rejects any swizzled instruction.  Run
   its exact binary on PPU and require the poisoned raw-bit denominator to be
   completely overwritten.
3. **Built for device admission:** L254 applies the real converter into the
   actual m8 MMA-B owner over `N64 x K64`.  It requires every logical fragment
   value to equal the independent `code - 8` fp16 oracle and plants both
   layout-1 and DeepGEMM's native N64xK16 permutation as RED controls.
4. Repeat dense and grouped correctness, then an isomorphic performance A/B
   with identical tile, stage, provider, grid and numerical fixture.
5. Do not select layout 3 in `auto`, change the canonical mapping ID, or feed
   its bytes to the shipping reader before all four steps close.

## DeepGEMM comparison anchor

The local comparison is pinned to DeepGEMM-for-sail commit
`f89eae10c0e90c20630b50e4314448f01321bfba`, file
`deep_gemm/include/deep_gemm/w4a16_gemm_cutlass3.cuh`.

Its W4A16 B path is not “plain copy all the way”:

- global-to-shared still comes from `DefaultGemm_AIU_Operand` and its
  `GmemTiledCopyB`;
- shared-to-register is a custom
  `Copy_Atom<UniversalCopy<uint128_t>, int>`;
- the global B tensor is viewed as `(K/16, 2*N)` `int32` with stride
  `(2*N, 1)`, while the shared tile is `(BlockK/16, 2*BlockN)`;
- the copied integer registers are dequantized and scaled before MMA.

Its producer's `_perm` has a minimum `N64 x K16` permutation atom.  L254 uses
the common complete `N64 x K64` domain rather than folding either axis.  In
low-to-high physical-nibble bit order, DeepGEMM is
`{k3,n3,k0,n4,n5,k1,k2,n0,n1,n2,k4,k5}`, while layout 3 is
`{k3,k4,k0,k5,n3,k1,k2,n0,n4,n5,n1,n2}`.  The maps select the same physical
coordinate for only 64 of 4096 logical codes.  Thus the common outer tensor
shape does not make the offline bytes interchangeable.  This difference comes
from DeepGEMM's hand-built register/dequant view, not from m16 MMA itself: our
m16 path uses the same CuTe MMA-B owner as m8 and therefore retains layout 3.
DeepGEMM remains a reader-structure reference and a deliberate
wrong-permutation control; its packed bytes are not a Q4_K correctness oracle.
