# Internal full-sweep axis census

This is the finite denominator for the internal performance-first sweep.  It
is a source-bound census, not a recommendation that every Cartesian coordinate
will compile.  A complete run emits every coordinate and gives it exactly one
terminal state (`MEASURED`, `INADMISSIBLE`, `BUILD_REJECT`, or `UNSUPPORTED`).
Static rejection is evidence about a coordinate; omitting the coordinate is
not pruning and cannot support a claim that the best configuration was found.

The executable authority is
`python3 -B ci/check_internal_sweep_axis_census.py --self-test`.  The separate
`--audit-current` mode is intentionally fail-closed until the component
runners cover the denominator below.

## Compile-time tensor-core axes

| axis | complete domain | source authority | classification |
|---|---|---|---|
| TileM | 8, 16, 32, 64, 128, 256 | `ppu_tactic_space.hpp::kTileM` | free |
| TileN | 16, 32, 64, 128, 256 | `ppu_tactic_space.hpp::kTileN` | free |
| TacticTileK | 32, 64, 128, 256 | `ppu_tactic_space.hpp::kTileK` | free |
| WarpM | 8, 16, 32, 64 | `ppu_tactic_space.hpp::kWarpM` | free |
| WarpN | 16, 32, 64, 128 | `ppu_tactic_space.hpp::kWarpN` | free |
| stages | 2, 3, 4, 6, 8, 12 | full Q8 table stamp and `fully_quantized_internal_matrix.py::STAGES` | experiment axis |
| BChunk request | 0, 1 | `ppu_tactic_space.hpp::kBChunkModes` | free request; implementation may reject |

The historical emitter default `{2,3,4}` is not the full stage authority.  It
is a convenient default for older commands.  Replacing the full ladder with
that default loses s6/s8/s12 and must make the census check red.

The raw Cartesian product is

```text
6 * 5 * 4 * 4 * 4 * 6 * 2 = 23,040 coordinates
```

for each `(format, ArtifactTileK)` pair, before any divisibility, warp-count,
pipeline-depth, copy-coverage, or resource guard.  FoldN is not an independent
knob.  It is derived from `(low bits, high bits, ArtifactTileK)` and remains in
the reported layout identity.

## Format and artifact axis

`ppu_format_config.hpp::artifact_tile_k_supported` defines the producer ABI:

| qtype | format | supported ArtifactTileK |
|---:|---|---|
| 8 | Q8_0 | 32 |
| 10 | Q2_K | 32, 64, 128, 256 |
| 11 | Q3_K | 64, 128, 256 |
| 12 | Q4_K | 32, 64, 128, 256 |
| 13 | Q5_K | 64, 128, 256 |
| 14 | Q6_K | 32, 64, 128 |

That is 18 supported pairs.  The registry's one ScaleFirst or
FullyQuantized default is a shipping default, not the layout-search domain.
Using only the canonical default is the artifact-axis negative control.

## ScaleFirst algorithm and grid axes

For every workload/qtype/layout/topology coordinate the algorithm denominator
contains:

1. ordinary non-persistent DP;
2. persistent pure-DP capacity grids;
3. persistent pure-DP balanced grids; and
4. fixed Split-K producer-only S=2, S=4, and S=8.

Persistent grids are generated from the exact final kernel's
`maximum_active_blocks()`.  For every `b=1..occupancy` they contain both:

```text
capacity: G = min(Q, CU*b)
balanced: G = ceil(Q / ceil(Q/(CU*b)))
```

Equal `G` values may be measured once, but their capacity/balanced provenance
bits remain attached.  For the established `Q=2048, CU=72, occupancy=8`
witness, capacity contributes `G=576` and balanced contributes `G=512`; a
single `capacity+balanced` row is accepted only when the two formulas really
deduplicate to the same G.

Fixed Split-K rows time the producer only.  S=1/full-output and
S=2/4/8/producer-only never share a leaderboard.  Reducer cost is neither
silently included nor subtracted later.

## FullyQuantized algorithm axes

The tensor-core branch uses the same seven raw compile axes and the same
ArtifactTileK support map.  Its A-provider is conditional rather than a global
Cartesian axis:

* `standard-aiu` is always represented where the format route exists;
* `packed-row` is represented only where its structural admission rule holds;
* S=1 is full output; and
* S=2, S=4, S=8 are producer-only.

The placed BC GEMV branch is a separate full-output competitor.  Its shipping
free axis is `RowsPerWarp={1,2,4,8}` from `gguf_bc_vecdot.hpp`.  `Threads` is
derived: Q4 with RowsPerWarp=4 uses 128, every other case uses 256.  `Grouped`
is a route capability, not a dense free axis; dense fixes it false and grouped
gets its own workload denominator.

There is no generic BC `CTA_N` axis.  The CTA_N/WARPS_N/WARPS_K template in
`gguf_bc_q4_gemv.hpp` is a separate CUDA-only Q4 specialization.  Likewise,
the axes in `gemv_lowbit/gemv_tactic_space.hpp` belong to the older nonshipping
route.  Borrowing either set would inflate the denominator with candidates the
shipping generic BC kernel cannot express.

## Workload, provenance, and output identities

The compile census is crossed with the committed multi-GGUF inventory, not a
hard-coded q/k/v/o list.  Every component must carry the exact model-to-GGUF
hash map, TP world/rank/partition, dense/grouped route, stable inventory
`shape_id`, official group size (or literal `UNKNOWN` for a future unknown
qtype), and grouped E/active/ragged identity.

Layers that share one inventory dedup shape contribute to
`source_tensors[]`; their names do not split a performance decision.  The same
MNK with a different qtype, TP partition, route, group size, or grouped identity
does split it.

Shape directory names come from the model catalog's hashed
`shape_directory` templates.  Components carry those templates in provenance,
the merger exact-compares them, and output rendering consumes them.  The
merger contains no second copy of the dense/grouped naming literals.

## Current integration verdict (2026-08-17)

`--audit-current` reports `INCOMPLETE`, with six concrete gaps:

1. the ScaleFirst device runner is Q8/A32-only rather than q8/q10--q14;
2. it lacks the ScaleFirst S2/S4/S8 producer denominator;
3. its 2,501-row Q8 table publishes only legal rows instead of all 23,040 raw
   coordinates plus named rejection states;
4. no default FullyQuantized device runner exists at the expected top-level
   path;
5. the plan-only FullyQuantized fallback cannot produce measured mergeable
   cells; and
6. the ScaleFirst component has no multi-GGUF/model/TP/grouped interface.

Thus the current component scripts are useful implementation pieces, but they
are not yet a complete overnight sweep.  The orchestration merger must reject
their partial summaries rather than publish a winner.

## Constructive falsification

The local checker plants five independent denominator losses:

* remove stage 6;
* reduce BChunk requests to the historical zero default;
* replace the supported ArtifactTileK set with one canonical default;
* reduce BC RowsPerWarp to 4; and
* remove ScaleFirst S4.

Each must fail.  This specifically prevents a small internally consistent
table from masquerading as the full search space.
