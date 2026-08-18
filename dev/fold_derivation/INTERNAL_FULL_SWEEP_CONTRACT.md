# Internal full-sweep orchestration contract

This is the publication boundary for the overnight performance-first sweep.
It is an orchestration and adjudication layer only.  It does not define a
kernel, collective, layout, tactic, or shape denominator, and it must never be
used to turn a bounded smoke table into a claim over the generated tactic
space.

The shape denominator comes from the committed multi-GGUF inventory authority.  In
particular, the old q/k/v/o 12-cell JSON and the 172-row Q8 table are smoke
controls, not the final denominator.  Both component runners must consume the
same shape-manifest and the exact same nonempty `gguf_hashes {model_id: sha256}`
map.  `gguf_set_sha256` is the SHA-256 of that canonical map, but is not a
substitute for member comparison: the merger compares both and rejects a
missing/substituted model even if a caller forges the aggregate token.

## Four leaderboards, never one

| ranking group | competitors | timed quantity |
|---|---|---|
| `SCALEFIRST_FULL_OUTPUT` | non-persistent plus every admitted persistent capacity/balanced grid policy | complete GEMM output |
| `SCALEFIRST_SPLITK_PRODUCER_ONLY` | ScaleFirst fixed Split-K S=2/4/8 | producer only; reducer excluded |
| `FULLY_QUANTIZED_FULL_OUTPUT` | placed BC GEMV plus direct tensor-core S=1 | complete output |
| `FULLY_QUANTIZED_SPLITK_PRODUCER_ONLY` | tensor-core S=2/4/8 | producer only; reducer excluded |

The two producer-only tables are deliberately not product E2E tables.  An
untimed reducer may establish output correctness, but reducer time never
enters or leaves the producer number.  This is essential for the already
measured ScaleOnly xplane S=8 path: it is ScaleFirst, not FullyQuantized.  The
merger validates scope from `metric_scope`, rejects a cell whose algorithm and
scope disagree, and never lets ScaleFirst and FullyQuantized compete.

Within one table, candidates compete only inside the exact `(model_id,
tp_world,tp_rank,tp_partition,problem_route,grouped identity,group_size,
shape_id,qtype,M,N,K,L)` identity.  `shape_id` is the inventory dedup key;
different layer names at the same decision shape are retained only in the
`source_tensors[]` provenance list.  The grouped identity retains `E`, active
experts and ragged policy (each may be explicitly `UNKNOWN`); it can therefore
never alias a dense row.  Layout, FoldN, tactic, grid policy, and Split-K factor
are candidate axes, so they are reported on the winner and runner-up rather
than used to split the comparison.

## Component terminal-state contract

Every generated denominator coordinate occurs exactly once and ends in one of:

* `MEASURED` -- positive raw samples, a median inside their range, MFU, MBU,
  and a positive correctness verdict;
* `INADMISSIBLE` -- a named static/runtime admission reason;
* `BUILD_REJECT` -- named compiler evidence for that exact generated type; or
* `UNSUPPORTED` -- an explicit format/algorithm boundary such as the absence
  of a shipping Q8 FullyQuantized reader.

`MISSING`, `RUN_FAILED`, and `CORRECTNESS_FAILED` are not terminal-state
aliases.  They make the component top level `INCOMPLETE` and prevent all winner
publication.  This preserves the distinction between losing, being rejected,
and never being generated.

For every ScaleFirst `(model,TP,route,grouped identity,group size,shape_id,
qtype,shape)` identity the denominator explicitly
contains non-persistent, persistent `capacity` and `balanced` policies, and
ScaleFirst fixed Split-K S=2/4/8 producer coordinates.  For every
FullyQuantized identity it explicitly contains placed BC GEMV, tensor-core S=1,
and tensor-core S=2/4/8.  A route that cannot implement one of them emits an
`UNSUPPORTED` coordinate; it does not shrink this algorithm denominator.
This applies equally to qtypes first discovered in a future GGUF (for example
an unknown qtype 6 or 20): the inventory remains visible and the component
states why it is unsupported.  Neither merger nor runner hard-codes the current
model's shape or qtype list.

The component summary must contain:

```text
schema, component, status=COMPLETE
expected_cells (or denominator.total_cells)
status_counts, missing=[], failures=[]
cells[]
provenance {
  root_sha, actlize_sha,
  shape_manifest_sha256,
  gguf_hashes { stable_model_id: sha256... },
  gguf_set_sha256,
  shape_directory { dense: template, grouped: template },
  device { measured identity fields... },
  source_hashes { path: sha256... },
  binary_hashes { target: sha256... }
}
```

Each cell carries `model_id`, stable `shape_id`, `source_tensors[]`, TP
world/rank/partition, dense/grouped route,
quantization group size, grouped `E/active/ragged` identity when applicable,
`qtype`, shape, layout descriptor, `ArtifactTileK`, FoldN, algorithm, config,
S, grid/policy, and status/reason.  A measured cell also
carries raw samples, median, MFU, MBU and correctness.  MBU must state its kind
(for example `DISTINCT_BYTE_MODEL`); a model is not silently relabelled as a
device traffic counter.

Inventory counts deliberately have two denominators: `logical_tensor_count`
includes every physical tensor rank plus a materialized tied-output alias,
whereas `tensors[]` and `rank2_or_rank3_logical_tensor_count` contain only
rank-2/3 rows that can participate in matrix/lookup/non-matmul routing.  The
authority validator checks both identities independently; it never equates
the all-rank count with the rank-2/3 publication.

## Runner and outputs

The production entry point resolves the three catalog models, reads their
actual GGUF headers, freezes that inventory inside the output bundle, and then
invokes both component runners.  A fresh box invocation is:

```bash
export QWEN35_35B_A3B_Q4_K_M_GGUF=/workspace/models/Qwen3.5-35B-A3B-Q4_K_M-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf
export QWEN3_32B_Q4_K_M_GGUF=/workspace/models/Qwen3-32B-Q4_K_M_GGUF/Qwen3-32B-Q4_K_M.gguf
export QWEN35_122B_A10B_Q4_K_M_GGUF=/workspace/models/Qwen3.5-122B-A10B-GGUF-Q4_K_M/Q4_K_M

OUT=/workspace/quactlize-internal-full-sweep-$(git rev-parse --short HEAD)-$(date -u +%Y%m%dT%H%M%SZ) \
JOBS=16 ITERATIONS=7 CORRECTNESS_REPEATS=2 \
bash tools/run_internal_full_sweep_box.sh
```

Each binding may be one GGUF file or one complete standard split directory.
The resolver fails on an incomplete or mixed shard family.  To continue an
interrupted run, reuse the exact `OUT` with `RESUME=1`; the frozen catalog,
resolved file set, inventory, runner hashes and source identity must still
match.  A changed authority is rejected before either component executes.
For a split family, shard zero owns model metadata.  Later shards may omit
those descriptive/model keys, as standard GGUF writers do; any key they repeat
must match shard zero byte-for-byte, they may not introduce a new metadata
authority, and every shard must still carry a complete, consistent `split.*`
identity.
The catalog itself is copied into `inputs/catalog.json`; after that first
atomic write the bundle is self-contained, so a vanished external catalog is
not required for resume.  If an external catalog is still readable, its bytes
must match the frozen copy.  Imported resolved/inventory files are accepted
only when their exact ordered model membership, TP policy, catalog hash and
GGUF member map match that frozen catalog; a one-model import cannot satisfy
the three-model production catalog.

`INTERNAL_SWEEP_CATALOG`, `GGUF_SET`, and `INTERNAL_SWEEP_SPEC` import
prebuilt authorities for development only.  `SCALEFIRST_RUNNER`,
`FULLY_QUANTIZED_RUNNER`, and the two `*_SUMMARY_REL` variables are likewise
development seams.  Any override requires explicit
`INTERNAL_SWEEP_DEV_MODE=1`; that bundle records
`publication_mode=development` and prints `DEVELOPMENT-COMPLETE`, never the
production `PASS` witness.  Production uses the checked-in three-model
catalog, its binding environments, and repository component defaults.  All paths are strict
`/workspace` children, created with `mkdir`; the runner uses neither `/tmp`
nor `mktemp` and refuses an existing bundle unless `RESUME=1` is explicit.
Every orchestration attempt has a unique ID.  Both component summaries must
publish that ID in provenance, so a zero-returning/no-op runner cannot reuse a
summary from an earlier attempt.  Merge output is built under the attempt
directory and renamed into place only after validation.  A pre-existing
partial `results/` tree is preserved under that attempt rather than blocking
recovery.  `completion.json` binds both component summaries and a deterministic
recursive manifest of every file under `results/`; an exact completed resume
validates the whole tree and returns idempotently without rerunning device work.

The durable outputs are:

* `results/summary.json` -- complete cells, exact denominator and provenance;
* `results/cells.tsv` -- one line per denominator coordinate;
* `results/winners.tsv` -- winner and runner-up gap for each ranking identity;
* `results/models/<model_id>/<dense-shape-directory>/` -- dense
  `cells.tsv`, `winners.tsv`, and `scope.json`;
* the grouped equivalent
  `<grouped-shape-directory>/`; qtype and layout
  remain rows inside that folder rather than becoming path-level identities;
* component raw bundles and runner logs; and
* `orchestration.provenance.txt` with runner, summary and merged hashes.

The two shape-directory strings are not duplicated here or in the merger.
They come from the hash-bound model catalog, are copied into both component
provenance records, exact-compared, then used to render the paths.  Known
qtypes use their official group size.  A future unknown qtype uses the literal
`UNKNOWN` (therefore `gUNKNOWN` under the current catalog), never the misleading
numeric surrogate `g0`.

## Local falsification

`python3 -B tools/merge_internal_full_sweep.py self-test` creates all four
leaderboards in memory, exercises all four terminal states, and requires the
FullyQuantized full-output runner-up to be present.  Its negative controls
plant:

1. one denominator cell missing;
2. a whole FullyQuantized algorithm removed while the declared denominator is
   reduced to make the arithmetic look closed;
3. the entire ScaleFirst S=4 algorithm removed with the same forged closure;
4. an unknown `MISSING` terminal status;
5. a Split-K producer mislabelled as full output; and
6. mixed device provenance;
7. equal claimed GGUF-set digests with a substituted member map;
8. equal claimed GGUF-set digests with one model missing; and
9. differing shape-directory authorities despite otherwise matching
   provenance; and
10. a sixth Q8 FullyQuantized cell added after forging the declared
    denominator closed (Q8 owns exactly BC + TC S1/S2/S4/S8, all unsupported).

It also proves that two layer names sharing one `shape_id` merge into one
decision while qtype, TP, route, and grouped changes remain isolated, and that
a catalog template change changes the rendered folder.  All ten planted
contract failures must be rejected.  Device execution remains the component runners'
responsibility; this local test proves only the publication contract.

`python3 -B ci/check_internal_full_sweep_runner.py` additionally runs the real
catalog resolver and GGUF parser on a synthetic GGUF, feeds hash-bound fake
device summaries through the production orchestration and merger, and
requires all four boards under `results/models/<model>/<shape>/`.  Its
negative changes only the catalog before a resume and must fail before either
component executes.  Additional negatives require: exact three-model
membership, stale-summary rejection through the attempt ID, self-contained
completed resume after deleting the external catalog, recovery from a
pre-existing result tree, truncated immutable provenance rejection, and a
failed authority write stopping before measurement.
