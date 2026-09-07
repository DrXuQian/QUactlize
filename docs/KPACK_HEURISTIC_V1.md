# Deterministic K-pack selection

The host path now selects **one complete tactic**, then prepares that one
prebuilt/cached parent. It does not run `Tuner.warmup`, benchmark candidates,
or compile an entire candidate pool. The kernel, offline bytes and published
six-library bundle are unchanged. Loader/device admission remains separate.

## Selection and data

The same code serves Q2–Q6 and FQ/SF dense/grouped:

1. Use a recent exact measurement when available.
2. Otherwise use the historical exact tactic, retaining its historical label.
3. With explicit prediction opt-in, choose the nearest eligible measured
   load profile **within the same qtype/route/N/K/expert-count family**.
4. If there is no eligible choice, return `FallbackRequired`. Never switch
   resident layouts, invoke an arbitrary compiled default, or start tuning.

Dense distance uses M. Grouped distance uses total rows, maximum rows and
active experts. A common integer relative-distance formula avoids handwritten
shape branches and floating-point tie differences. Grouped profiles only rank
tactics: actual expert rows determine tile count and the capacity/balanced
grid. Same profile does not imply measured performance on another router.
AP1/M1, TM8 route boundaries, split eligibility and K-loop depth are checked
before a transfer; the exact module still owns final resource/legality checks.

This is an **empirical heuristic backed by tactic data**, not a universal
analytic latency model. It has no fitted coefficients or decision tree.
It deliberately does not interpolate absolute microseconds between different
measurement campaigns. Unknown weight families decline, rather than borrowing
a superficially similar N/K geometry.

The data is still substantial and is not described as "only a few rules":
2,982 historical contexts plus five newly measured M values give **2,987 exact
contexts**. Ninety recent contexts override the old choices/measurement epoch.
The effective global kernel closure is still **247 parents**. This step does
not claim that all of them can be replaced by a handful of universal kernels.

## Calibration result

Input archive `kpack-warmup-real-results.LJGzK2.tgz`, SHA-256
`10a00ae918f719eee980eabff0b6006370142c1f9ad1be24a4d032a13a044440`,
contains the already reviewed 90-context real-module gate. The importer
checks the exact plan/module union, source and compiler receipts, case digests,
complete numerical/negative checks, and rederives timing summaries from the
two confirmation samples for every candidate. No device execution is performed
during this import; the uploaded archive does not include ELF payloads.

The old tactic is retained when it is within 5% of the same-run bounded-pool
best and its own round spread is within 5%. Otherwise a measured alternative
is selected. Unrelated noisy candidates do not invalidate a stable alternative.
Original gate verdicts (80 within-pool, one gap, nine noise reviews) remain in
the report; this selection does not rewrite the original warmup result.

| Result | Value |
|---|---:|
| Recent selected contexts within both 5% limits | 90 / 90 |
| Maximum selected median regret | 4.153523% |
| Old exact choices above 5% corrected | 9 |
| Parents selected for these 90 contexts | 55 |
| Previous candidate-pool parents | 173 |
| Runtime candidate timing / fitted coefficients | 0 / 0 |

These are **post-hoc calibration** results, not an independent rerun or a
global optimum proof. The two-sample recent bound is not the historical
three-round bound; historical performance guarantees do not transfer to the
new module wrapper. The Q6 SF grouped boundary now uses the measured faster
alternative that the online warmup's soft budget missed.

The attempted shallow global fit was rejected: blind fallback replay had 486
known regrets above 5%, 866 missing costs and a worst known regret of 89.14%.
No such tree is deployed. Missing observations are not numerical failures or
performance passes. Profile transfers remain explicitly unvalidated.

## Files and query

- `tools/kpack_heuristic.py`: Python host reference and query CLI.
- `policies/kpack_zw810_heuristic_v1.json`: recent calibration overlay and
  module binding; depends on the digest of `kpack_zw810_runtime_v1.json`.
- `policies/kpack_zw810_heuristic_v1.hpp`: SDK-free C++17 selector; includes
  the matching historical header, without duplicating its configuration data.
- `policies/kpack_zw810_heuristic_v1.report.json`: calibration counts and the
  nine historical corrections.
- `quactlize/runtime/dispatch.py::prepare_selected`: single-parent resident
  module binding, with no tuner/measurement/compiler call.

For a host-reference query, from the repository root:

```sh
python tools/kpack_heuristic.py \
  --base policies/kpack_zw810_runtime_v1.json \
  --model policies/kpack_zw810_heuristic_v1.json \
  --route sf-dense --qtype 12 --m 3072 --n 1024 --k 5120
```

For grouped, supply `--rows-file` (one actual expert count per line), not M.
`--allow-prediction` permits a host proposal on a miss; it is **not** device
admission. The CLI uses the recorded identity for reference queries and says
so. An application must supply actual device/module identities.

The C++ namespace is `quactlize_kpack_heuristic_v1`. `select(Query, false)`
serves exact data; `select(Query, true)` additionally permits a proposal.
The statuses are `MeasuredRecent`, `MeasuredHistorical`,
`HeuristicUnvalidated`, `FallbackRequired`, `InvalidQuery`, `BindingMismatch`.
Every returned config includes AP, delivery-N, algorithm, split and grid policy;
do not translate its symbol into the old partial `config_name` ABI.

`Query.kernel_source` and `sdk_digest` now use the resident module's
`record.identity.kernel/sdk` hash scope, not the older sweep's differently
scoped hashes. The C++ `kModuleContract` also identifies the **complete**
compiler receipt; the loader must compare it before binding a module. Python
`prepare_selected` performs the full receipt and parent-tuple comparison.

The returned `grid` follows the resident module ABI: **zero for nonpersistent
launch**, including historical SF `ordinary` records. Persistent grids are
recomputed from the actual rows. Logical CTA counts from old SF logs must not
be copied into the module's persistent-grid field.

Compile/load only the selected parent, then call
`prepare_selected(backend, request, selection)` outside graph capture. Reuse
the handle via `backend.run(handle)` and close it through the owning backend.
Preparation rejects a missing parent, changed request, changed compiler/SDK,
wrong device, or unadmitted prediction; it does not secretly try alternatives.
The prepared module's resource query/can_implement still runs.

## Reproduction and remaining admission

The calibration command takes the archived results, not a PPU device:

```sh
python tools/calibrate_kpack_heuristic.py \
  --base policies/kpack_zw810_runtime_v1.json \
  --results kpack-warmup-real-results.LJGzK2.tgz \
  --output /tmp/kpack-heuristic-review/policy.json
python -m pytest -q tests/test_kpack_heuristic.py
```

Choose a fresh output path; existing outputs are not overwritten. The generated
header also needs the original historical header on its include path. Python/
C++ parity covers 17,922 queries (all exact contexts, perturbed M/routers,
prediction disabled/enabled and bad bindings). Negative tests cover changed
source, missing results, invalid samples, numerical failures and dispatch
contract/capture errors. These are host checks, not new GPU results.

Next is a short **selected-dispatch** device/loader gate, not another full
sweep or 173-parent tuning run. The 55 selected parent identities are already
in the real-shape module cache; kernel headers and the compiler generator have
not changed. Unknown-profile transfers need separate targeted checks or an
admitted K-pack fallback. Do not admit an any-M loader solely from this table.
The six-library selector and llama.cpp binding have not been changed here.

Choosing one module reduces compilation/loading of unused candidates; it does
not by itself reduce the bytes of the published DSOs. AOT packaging must prune
its actual emitted kernel closure before claiming a size reduction. This
90-context subset is not proof that the full 247-parent closure can shrink to 55.
