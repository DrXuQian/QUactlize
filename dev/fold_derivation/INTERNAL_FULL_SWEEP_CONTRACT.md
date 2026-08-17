# Internal full-sweep orchestration contract

This is the publication boundary for the overnight performance-first sweep.
It is an orchestration and adjudication layer only.  It does not define a
kernel, collective, layout, tactic, or shape denominator, and it must never be
used to turn a bounded smoke table into a claim over the generated tactic
space.

The shape denominator comes from the committed GGUF inventory authority.  In
particular, the old q/k/v/o 12-cell JSON and the 172-row Q8 table are smoke
controls, not the final denominator.  Both component runners must consume the
same shape-manifest and GGUF hashes; the merger rejects a mixed pair.

## Three leaderboards, never one

| ranking group | competitors | timed quantity |
|---|---|---|
| `SCALEFIRST_FULL_OUTPUT` | non-persistent plus every admitted persistent capacity/balanced grid policy | complete GEMM output |
| `FULLY_QUANTIZED_FULL_OUTPUT` | placed BC GEMV plus direct tensor-core S=1 | complete output |
| `FULLY_QUANTIZED_SPLITK_PRODUCER_ONLY` | tensor-core S=2/4/8 | producer only; reducer excluded |

The third table is deliberately not a product E2E table.  An untimed reducer
may establish output correctness, but reducer time never enters or leaves the
producer number.  The merger validates that fact from `metric_scope` and
rejects a cell whose algorithm and scope disagree.  ScaleFirst and
FullyQuantized also never compete with each other in this checkpoint.

Within one table, candidates compete only inside the exact
`(qtype,tensor-or-route,M,N,K,L)` identity.  Layout, FoldN, tactic, grid policy,
and Split-K factor are candidate axes, so they are reported on the winner and
runner-up rather than used to split the comparison.

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

For every ScaleFirst `(qtype,tensor,shape)` identity the denominator explicitly
contains non-persistent plus persistent `capacity` and `balanced` policies.  For
every FullyQuantized identity it explicitly contains placed BC GEMV, tensor-core
S=1, and tensor-core S=2/4/8.  A route that cannot implement one of them emits
an `UNSUPPORTED` coordinate; it does not shrink this algorithm denominator.
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
  shape_manifest_sha256, gguf_sha256,
  device { measured identity fields... },
  source_hashes { path: sha256... },
  binary_hashes { target: sha256... }
}
```

Each cell carries `qtype`, shape, layout descriptor, `ArtifactTileK`, FoldN,
algorithm, config, S, grid/policy, and status/reason.  A measured cell also
carries raw samples, median, MFU, MBU and correctness.  MBU must state its kind
(for example `DISTINCT_BYTE_MODEL`); a model is not silently relabelled as a
device traffic counter.

## Runner and outputs

After the two component runners exist, the complete box invocation is:

```bash
OUT=/workspace/quactlize-internal-full-sweep-$(git rev-parse --short HEAD)-$(date -u +%Y%m%dT%H%M%SZ) \
GGUF=/path/to/model.gguf \
INTERNAL_SWEEP_SPEC=/path/to/committed-shape-manifest.json \
JOBS=16 BENCH_REPS=3 \
bash tools/run_internal_full_sweep_box.sh
```

`SCALEFIRST_RUNNER`, `FULLY_QUANTIZED_RUNNER`, and the two `*_SUMMARY_REL`
variables are development seams only.  Production runs use their repository
defaults.  All paths are strict `/workspace` children, created with `mkdir`;
the runner uses neither `/tmp` nor `mktemp` and refuses to overwrite an old
bundle.

The durable outputs are:

* `results/summary.json` -- complete cells, exact denominator and provenance;
* `results/cells.tsv` -- one line per denominator coordinate;
* `results/winners.tsv` -- winner and runner-up gap for each ranking identity;
* component raw bundles and runner logs; and
* `orchestration.provenance.txt` with runner, summary and merged hashes.

## Local falsification

`python3 -B tools/merge_internal_full_sweep.py self-test` creates all three
leaderboards in memory, exercises all four terminal states, and requires the
FullyQuantized full-output runner-up to be present.  Its negative controls
plant:

1. one denominator cell missing;
2. a whole algorithm removed while the declared denominator is reduced to
   make the arithmetic look closed;
3. an unknown `MISSING` terminal status;
4. a Split-K producer mislabelled as full output; and
5. mixed device provenance.

All four must be rejected.  Device execution remains the component runners'
responsibility; this local test proves only the publication contract.
