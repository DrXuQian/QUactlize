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

The selected-dispatch gate is now ready; see the command below. It is not a
full sweep or 173-parent tuning run. The 55 selected parent identities are
already in the real-shape module cache; kernel headers and the compiler
generator have not changed. Unknown-profile transfers need separate targeted
checks or an admitted K-pack fallback. Do not admit an any-M loader solely from
this table. The six-library selector and llama.cpp binding remain unchanged.

Choosing one module reduces compilation/loading of unused candidates; it does
not by itself reduce the bytes of the published DSOs. AOT packaging must prune
its actual emitted kernel closure before claiming a size reduction. This
90-context subset is not proof that the full 247-parent closure can shrink to 55.

## Run the selected-dispatch gate on box

`tools/run_kpack_selected_gate.py` fixes the denominator at 90 real-shape inputs:
Q2–Q6, FQ dense (30), SF dense (20), FQ grouped (20), SF grouped (20). Each input
has **one** selected configuration. Default execution only reads the 55 existing
cached modules; it does not construct a compiler, profile a shortlist, or update
the selector. A cache miss names the missing parents and stops before device
work. Restoring the original cache is preferable to rebuilding it.

For each input the gate:

- Reselects using the loaded module's actual device/SDK/kernel identity, then
  calls the real `prepare_selected` helper and checks the actual resolved grid.
- Checks every output against official GGUF dequantization, replays the same
  handle, and compares with direct preparation of the **same parent/tactic**
  bit-for-bit. The unchanged condition-scaled bound is `5e-3`.
- Detects a zero-low-plane fault. A launch failure, NaN, or unwritten poisoned
  output cannot count as a detected numerical negative.
- Takes three five-repeat samples of that fixed configuration, after two
  warmups. These are validation timings, never inputs to an online choice.

This reuses the previously tested fixture: random finite GGUF blocks at real
dimensions, dense FP16-exact row-tagged A with four K categories, and an
independent factorized official-GGUF oracle. It is not a checkpoint-weight test
or exhaustive input-value validation. Expected closure is 270 positive checks,
90 detected negatives, 90 raw replay/direct matches and 25 clean worker exits.

Run from the **develop checkout root**, on one otherwise idle PPU:

```bash
(
  set -eo pipefail
  test "$(git branch --show-current)" = develop
  git pull --ff-only
  SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK
  OUT=/workspace/kpack-selected-dispatch-v1
  source "$SDK/envsetup.sh"
  export CUDA_VISIBLE_DEVICES=0
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  python3 tools/run_kpack_selected_gate.py \
    --sdk "$SDK" --cache /workspace/kpack-warmup-v1-jit \
    --output "$OUT" --resume \
    2>&1 | tee -a "$OUT.console.log"
)
```

The subshell prevents a failed command from exiting the calling Docker shell.
Do not point `--cache` at the old sweep bundle: this is the cache used by the
completed real-shape module gate. Only if modules really are missing, explicitly
add `--compile-missing --jobs 32`; this rebuilds missing selected parents only,
and rejects a different compiler/kernel contract. It is not the default path.

`PLAN`, `CACHE`, `FIXTURE`, `CASE`, `PROGRESS`, and `DONE` records show progress.
Twenty-five weight groups run in fresh processes. A failed group does not stop
other groups. Results are saved per case, and `--resume` retains them when
plan, source, module payload, SDK and physical device identity agree. A clean
worker-exit receipt is required as well: if cleanup or the coordinator died
after case results were saved, one case is rechecked to close that group.
Changed source/authority requires a fresh output directory, not deleting prior
results. A directory lock prevents two runners sharing one output.

`setup-timing.json` separates cache verification and optional compilation.
Per-case results separate module loading, fixture preparation, host selection,
handle preparation and resident-kernel timings. `wall_seconds` is the current
invocation's wall time, not a sum across resumed runs. Historical calibration
times are labelled separately; no cross-run ratio is called a same-run 5% pass.
The principal success record is:

```text
KPACK_SELECTED_DONE { ... "status": "PASS", "completed": 90, "clean_workers": 25 ... }
```

Return the output directory (receipts and logs only, no cached `.so` files):

```bash
(
  set -e
  OUT=/workspace/kpack-selected-dispatch-v1
  test -s "$OUT/summary.json"
  tar -czf /workspace/kpack-selected-dispatch-v1-results.tgz -C "$OUT" .
)
```

`--plan-only --output /a/fresh/directory` needs no SDK/device. Local tests cover
the actual driver with a host backend, forbidden tuner/compiler calls,
four routes, wrong output/NaN/missed-store/launch-error plants, cache corruption,
and fail-closed resume/worker completion. **Device execution remains pending.**
