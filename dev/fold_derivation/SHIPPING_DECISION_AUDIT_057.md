# Shipping-path single-decision audit (INBOX 057)

Scope: production `.so`, host producer/torch boundary, packer/manifest, tactic emitter/table, and the headers they
share. Tests and independent oracles are deliberately excluded. “Unified” means one shipping definition is consumed
everywhere; a consistency test between two definitions does not qualify.

## Verdict map

### 1. Per-format TileK — unifiable, substantially unified in C++, cross-language work remains

Definition: `quactlize/include/ppu_format_config.inc`, with distinct `scale_first_tile_k` and
`fully_quantized_tile_k` columns. They are distinct decisions: scale-first uses the minimum legal 32-byte delivery
of its narrowest plane; fully-quantized Q2/Q4 deliberately consume a complete 256-code packed metadata superblock,
Q3/Q5 need 256 for the int1 plane, and Q6 retains its validated 128-K high-plane arrangement.

Now derived from it: dense and grouped launches, their validity queries, the C++ layout producer/recoverer, the
fully-quantized unit producer/recoverer, tactic-space format rows, and the v2 per-problem config record. Grouped has
no TileK template default left.

Still restated: `tools/pack_gguf.py::_tile_k`, scale-first benchmark defaults, the torch fixed-arrangement wrapper,
and `emit_tactic_configs`' raw `<bits> <tile_k>` arguments. These should parse the registry or take a format+scheme
and derive the value. The old offline-layout gate currently fails because it searches dispatch for numeric literals;
it should inspect the registry. Divergence is now loud at the library/tuner boundary through
`quactlize_ppu_config_v2.tile_k`, but is not yet loud in those Python/benchmark consumers.

Measurement status: the scale-first values are hardware delivery minima, not tuning preferences. The
fully-quantized Q2/Q4 256 choice is backed by the packed-unit/full-superblock measurement; Q6 128 is also a
correctness boundary because the 256 inverse map is incomplete. The distinction must not be collapsed to one
per-width value.

### 2. GroupSize — unifiable; registry and format traits both own it today

Definitions today: `ppu_format_config.inc` says 16/32 per qtype, while `gguf_scale_layout.hpp::Traits<T>` independently
states the same GGUF semantic. The backend validity domain and CUDA vecdot query consume the registry, but dense and
grouped template dispatches still pass literal 16/32. `gguf_prepass_ops.cpp` also reconstructs 16/32 from qtype in
several shipping entry points.

Make the registry the cross-language definition and make each `Traits<T>::kGroupSize` either consume it or
`static_assert` equality through a generated `KType`/registry mapping. Generate the backend type cases from the same
rows and pass `format.group_size`; make torch shape checks use the same exported metadata. Until then a registry
change can make the validity query disagree with the launcher, and no current shipping assertion catches it.

Measurement status: none needed. GroupSize is GGUF format semantics, not a tactic.

### 3. qtype -> low/high code planes — unifiable; repeated in every layer

Definition intended: the `low_bits`/`high_bits` columns in `ppu_format_config.inc`.

Restatements include `schemes.py::CODE_PLANE`, `tools/pack_gguf.py::_low_bits/_high_bits`, C++ launch type switches,
`ppu_unit_pack.cpp`'s `LowBits/HighBits`, and three `dispatch_ktype` lambdas plus the scale-first path in
`gguf_prepass_ops.cpp`. `gguf_scale_layout.hpp::CodeTraits<T>` is the official byte-map layer and independently
implies the widths.

Generate the qtype/type dispatch cases from the registry and assert its widths against `CodeTraits<T>` at compile
time. Python should parse the registry rather than translate `CODE_PLANE` strings back to integers. Boundary shape
checks make a one-sided mismatch loud, but producer and consumer can still drift together and accept the same wrong
shape, so those checks are not a single source.

Measurement status: none needed. Plane decomposition is format semantics.

### 4. Delivery fold — C++ unified; Python remains a separate implementation

C++ definition: `fold_traits.hpp::DeliveryFold` / `delivery_fold_v`. Contrary to the older three-copy description,
both `fpA_intB_ppu.cuh` and `moe_grouped_ppu.cuh` now consume that template; they do not own independent formulas.
`ppu_tactic_space.hpp::fold_for` and `formats.py::fold_for` still repeat the arithmetic.

Move the host-only arithmetic used by tactic space into the same small dependency-free header as `DeliveryFold`, and
have Python obtain the derived fold through an exported layout-description query (or generated registry fields for
the fixed shipping arrangements). The manifest may print fold for humans but readers should continue deriving and
checking it. There is currently no shipping failure if Python and C++ fold differently; wrong finite output is
possible.

Measurement status: the value is derived from the 32-byte delivery floor. Measurements justify retaining folded
tactics as search candidates, not the formula itself.

### 5. Config identity and spelling — unifiable; library is canonical, tuner still reparses names

Definition: the strings and geometry in `ppu_dense_configs.inc` and `ppu_grouped_configs.inc`; each X-macro already
drives both enumeration and dispatch. `QUACTLIZE_PPU_GROUPED_CUDA_CONFIG_NAME` similarly unifies the CUDA family.

`benchmarks/analyse.py` constructs `schema TMxTNxTK:WMxWN:sS`; the library string omits schema and TileK;
`tools/tune.py::canonical` parses both into a tuple. That parser is a bridge between two identities. New tuning code
should match the sample's component fields against the library's v2 record `(family, tile_m, tile_n, tile_k,
warp_m, warp_n, stages)` and carry the library-provided opaque `name` into the tactic table. Then spelling is never
reconstructed. Today a typo becomes an empty table (loud), but two syntactically valid spellings can still be joined
by a permissive parser rather than by identity.

Measurement status: none; this is identity. TileK in the sample is measured, but its spelling is not.

### 6. qtype -> `PPU_PACKED_FORMAT` and per-format library — unifiable, and the shipping loader is not wired

Definition: `ppu_format_config.inc::packed_format`; the device binary now uses `for_packed_format` and statically
rejects an unknown build value. Preprocessor branches still restate which C++ types go with format 0..4 and should be
generated or guarded by width/group/type static assertions.

More seriously, `ppu_backend.cpp::load_format(fmt)` implements per-format library resolution but has no caller.
Fully-quantized torch entries call `load()` for every qtype. A mixed GGUF therefore cannot select its registered
format library through this path despite the loader comments claiming that it can. The device entry rejects the
wrong qtype with rc=33, so divergence is loud but the advertised multi-format route is absent. The caller should
derive `fmt` from the registry and call `load_format(fmt)`; the library should also export its selected qtype/format
so a wrong file name cannot masquerade as the right binary.

Measurement status: none; this is build/routing identity.

### 7. Packed metadata unit geometry — C++ unified, Python presentation is unifiable

Definition: `gguf_packed_unit.hpp::Unit<T>` derives unit bytes, superblocks per copyable unit, group runs, field bit
positions, and byte neutrality from `gguf_scale_layout.hpp::Traits<T>`. `gguf_unit_pack.hpp`, the kernel staging
paths, vecdot, and torch shape checks consume those traits. Static assertions make non-byte-neutral or non-integral
layouts loud through the trait's exact bit tiling, although `Unit` should state that local contract as equality
rather than its weaker `kUnitBytes <= header + kBlockBytes` assertion.

`schemes.py::_packed_unit_name` independently rederives pairing and byte count from `formats.py::BLOCKS`. It only
names/presents the unit today, but a name can still lie. Have Python read the already exported C++ scale/unit traits
and mint the token from those values. Also make `selected_fully_quantized_qtype` derive paired-K alignment from
`Unit<T>::kSbPerUnit`, not `packed_format == 3 || packed_format == 4`.

Measurement status: unit byte count and pairing follow GGUF bytes plus legal 4/8/16-byte copies. The performance of
the staging strategy is measured separately; the unit geometry is not a tuning choice.

### 8. Raw GGUF block size and offsets — unifiable within C++, irreducibly mirrored at the external format boundary

C++ definition should be `gguf_unit_pack.hpp::Raw<T>`. `ppu_backend.cu::raw_block_bytes`,
`gguf_prepass_ops.cpp::kRaw`, and other qtype switches repeat block sizes. They can consume `Raw<T>` directly.
Python's `formats.py::BLOCKS` must represent the external GGUF schema independently enough to inspect files, but a
generated/exported comparison must fail when it differs from C++.

Current entry points reject a caller-provided wrong `block_bytes`, which catches a one-sided error. It does not catch
all internal C++ copies agreeing on the wrong number. The official GGUF numbers, not a benchmark, determine this.

### 9. Offset-binary code bias and placed-code ZMul — unifiable and correctness-sensitive

`ppu_unit_pack.cpp` and `gguf_prepass_ops.cpp` each restate code bias (Q3=4, Q6=32, others 0).
`formats.py::PLACED_CODE_ZMUL` separately records the converter correction (0, -4, 8, 8, -24), while device scale
prepass paths take ZMul template/runtime arguments. These values are related format/converter semantics and have
already produced wrong-but-finite output when defaulted.

Add `code_bias` and the scale-first/fully-quantized converter correction to the format registry, generate both C++
dispatches, and have Python read them. Assert the selected converter's compile-time correction against the registry.
There is no adequate shipping loudness today if producer and consumer repeat the same wrong bias.

Measurement status: correctness semantics, established by official dequantization and planted faults, not speed.

### 10. The value 256 in shape admission — irreducibly two decisions that must be named separately

The same literal currently means both GGUF's 256 weights per K-quant superblock and the AIU resident artifact's
256-row interleave/alignment. They happen to agree but are not one concept: a future non-GGUF format can still need
AIU placement, and a different placement can consume 256-weight GGUF blocks.

Define named `kGgufSuperblockK` and `kAiuInterleaveRows` owners, derive all K/unit counts from the former and all
resident N/K admission from the latter, and assert their required composition at producer/consumer boundaries.
For paired Q3/Q6, derive K alignment as `kGgufSuperblockK * Unit<T>::kSbPerUnit`; do not spell 512. Current repeated
entry checks make most one-sided mistakes loud, but do not say which 256 they enforce.

Measurement status: external format and hardware layout constraints, not performance choices.

### 11. Default tactic identity and fallback coverage — coverage unified; identity/order unifiable

The actual fallback is `DenseConfigId::Default` / `GroupedConfigId::Default`, and its exact kernel instantiation now
statically proves shared-storage fit and the unrestricted ordinary-A path. That is the correctness definition.

The `.inc` files additionally require the default row to be first, while diagnostic strings use `kConfigs[0].name`;
there is no `static_assert` that `Default` has underlying index zero. Generate a default lookup or assert the index,
and stop using row order as a second identity. The current default geometries came from measured configurations, but
the files do not link “first/default” to evidence that they are the best universal fallback; only legality is proven.

### 12. Config-dependent workspace maxima — unifiable; grouped still restates geometry

Dense derives minimum TileM/TileN from `kDenseConfigs`. Grouped workspace sizing hardcodes a 16-row/64-column grid
formula even though the grouped inventory includes TileN=32. Its 64-byte multiplier appears to make the current
allocation conservative, but that is arithmetic a reviewer must rediscover rather than a property of the config set.

Derive grouped maxima from `kGroupedConfigs`, or expose the maximum exact `get_workspace_size` across the valid v2
records. Add an assertion that the chosen formula dominates every compiled config. This is correctness capacity,
not a measured knob.

### 13. Offline tactic legality versus exact compiled-kernel legality — irreducibly separate, now loud

`ppu_tactic_space.hpp` is a dependency-light pre-build grid/filter; it cannot know the exact instantiated
`GemmKernel::SharedStorageSize`. The new per-shape query instantiates the real type in query-only mode and applies the
same exact guard as launch. These are legitimately two layers, not competing definitions: the former prevents
known-impossible compilation, the latter is authoritative for profiling.

Divergence is now loud in both directions: an offline false positive is removed by the v2 valid list (and launch
still refuses it); a generated row that cannot instantiate fails a compile-time assertion. The tuner must not copy
either predicate into Python. Runtime enumeration is a cheap host call—no context, allocation, or device launch—so
there is no cost argument for a second implementation.

Measurement status: none; validity precedes measurement.

### 14. Compiled tactic set versus benchmark coverage set — irreducibly separate decisions, identities can be unified

The benchmark's large generated set is a coverage experiment; the `.so`'s small X-macro set is a deployment choice.
They should not be the same set. Within each library, enumeration and dispatch are already unified by one X-macro.
The handoff should be generated from coverage output into the `.inc` rows, and the tuner should read v2 records from
the binary rather than a separately typed shipped-name file. A tactic absent from the binary is then structurally
unselectable rather than detected by name canonicalization.

Measurement status: the non-default shipped set should be justified by the regret/coverage measurement. The current
files describe it only as a bootstrap set.

### 15. Three mainloop implementations and performance-only flags — unifiable architecture, detailed under INBOX 058

The ordinary one-plane, two-plane, and folded collectives separately implement substantial pipeline/load/transform
logic. This is not itself one scalar decision, but it is the mechanism that lets decisions such as `PPU_B_CHUNK`
exist in only two copies. A performance option that does not alter resident bytes must become a tactic/search field,
not a build flag. Current int1 chunking has measurement; the int4 exclusion at shipping TileK=256 does not.

The safe target shape is shared orchestration plus B-provider/converter policies only if a source reading confirms
that the pipeline and scale/accumulate semantics are genuinely identical. INBOX 058 requests that reading before a
refactor; its optimisation/correctness inventory is a separate follow-up report.

## Priority from the audit

1. Finish the TileK cross-language consumers and update the gate to read the registry.
2. Wire `load_format(registry.packed_format)` and export binary identity; otherwise mixed-format deployment cannot
   reach the five per-format libraries.
3. Generate/assert qtype -> `(Low, High, GroupSize, packed_format, code_bias, ZMul)` from the registry across device,
   torch, and Python. These mismatches can be finite and silent.
4. Remove the remaining Python fold implementation or make it query the library.
5. Derive paired-K alignment, raw block sizes, grouped workspace, and default identity from their named owners.
6. Use the INBOX 058 mainloop reading to decide whether shared orchestration plus provider policies is sound before
   making `PPU_B_CHUNK` a tactic field.
