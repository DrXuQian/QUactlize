# K-pack measured policy

## Tactic shortlist prototype and next box run

The next host prototype is `policies/kpack_zw810_tactics.json`, queried by
`tools/kpack_tactic_model.py`. It uses exact measured tactics, nearby-family
parent hints and separate parent/runtime scores. It does not replace the
compact JSON/C++ selector below or change any `.so`. Read the
[calibration results and limitations](KPACK_HEURISTIC_REVIEW.md#implemented-host-prototype-and-calibration)
before using its suggestions. A cache miss is a shortlist requiring admission
and measurement, not a 5%-guaranteed default.

```bash
python3 tools/kpack_tactic_model.py policies/kpack_zw810_tactics.json \
  --route fq-dense --qtype 11 --m 48 --n 256 --k 3072
```

This exact measured query returns the same `32x32x256_w16x16_s2` parent with
**S4**, not the stale S1 choice. Add `--no-cache` to inspect the proposed
shortlist instead. Grouped queries additionally require `--rows-file` with
one actual expert row count per line; total/max counts alone are insufficient.
Runtime binding must check the returned device/source/SDK/mapping requirements.

The next box run has **212 requests / 285 existing parents**, no compilation.
Keep the original completed campaign and its compiled cache on the box. Once
the current development checkout contains the new suite, run:

```bash
python3 -u tools/run_kpack_policy_validation.py \
  --suite policies/kpack_zw810_tactics.validation.json \
  --campaign /workspace/kpack-overnight-c0c1361 \
  --output /workspace/kpack-tactics-validation-v1 \
  --sdk /workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
  --devices 0,1,2,3,4,5,6,7
```

It runs 3x11 fresh timing rounds, correctness_repeats=1, with source/SDK/payload
checks and progress/ETA. No new full sweep starts. Failures are isolated;
repeat the command with `--retry-failures` to retry just rejected requests.
Return `results/summary.json` under the new output and keep `phases/` for audit.
Do not reuse the previous validation's output directory or overwrite its timings.

Reproduce the host calibration from extracted archives (new output paths):

```bash
python3 tools/kpack_tactic_evidence.py /path/to/overnight \
  --epoch overnight --output /path/to/evidence/overnight.json
python3 tools/kpack_tactic_evidence.py /path/to/followup \
  --epoch validation --output /path/to/evidence/validation.json
python3 tools/fit_kpack_tactic_model.py \
  --evidence /path/to/evidence/overnight.json \
  --evidence /path/to/evidence/validation.json --output /path/to/new-fit
python3 tools/plan_kpack_tactic_validation.py \
  --campaign /path/to/overnight --validation /path/to/followup \
  --evidence /path/to/evidence/overnight.json \
  --evidence /path/to/evidence/validation.json \
  --model /path/to/new-fit/model.json --output /path/to/new-suite.json
```

The query tool needs only the Python standard library; fitting additionally
uses NumPy. No new Python package, PPU SDK, kernel build or device is needed
for this local calibration. Generated cache rows are measured data, not a
promise that the production ABI can already express these full identities.

## Current compact selection

The current host prototype merges near-equal measured choices. It still
requires both per-round regret and cross-round spread to stay within **5% at
every served observation**. Missing costs are not treated as acceptable choices.

| Fit | Rule leaves | Runtime variants | Compiled parents | Mean median-time increase |
|---|---:|---:|---:|---:|
| Initial independent-family fit | 999 | 484 | 311 | 0.3917% |
| Source-owned dense grid recipes | 938 | 422 | 303 | 0.4066% |
| Pooled, favor fewer branches | 738 | 331 | 261 | 0.6352% |
| Selected: pooled, favor fewer parents | **788** | **312** | **219** | **0.7012%** |

The selected model serves the same **2,717 requests**, with maximum training
round regret **4.982932%**; 45 requests remain blocked. It has 20 shared rule
trees and 152 exact family guards, **not just 20 rules**. A parent is one
independently compiled specialization; runtime variants add schedule/grid/Split-K.
The greedy tree and parent cover are a maintenance tradeoff, not a claim of
globally minimal rule count or globally optimal kernel performance.

Why are hundreds of rules still necessary? Confirmation costs are sparse:
a neighboring shape's choice often was never timed at this shape. A common
choice must satisfy every observation, including grouped aliases. Missing
measurements do not prove a performance gap, but cannot justify a safe merge.
The next run fills prioritized holes instead of scanning the Cartesian product.

Current files:

- [Compact policy JSON](../policies/kpack_zw810_compact.json).
- [SDK-free C++17 header](../policies/kpack_zw810_compact.hpp).
- [Fit report](../policies/kpack_zw810_compact.report.json).
- [Targeted validation suite](../policies/kpack_zw810_compact.validation.json).

Use the query examples below with `policies/kpack_zw810_compact.json` and
include `policies/kpack_zw810_compact.hpp` for the current selection. The two
example choices remain the same. Dense persistent grid integers now resolve
from source-owned recipes in `scalefirst_persistent_policy.hpp`:

```text
Q = ceil(M / TM) * ceil(N / TN)
capacity(Q, CU, b) = min(Q, CU * b)
balanced(Q, CU, b) = ceil(Q / ceil(Q / (CU * b)))
```

Every recipe is verified against raw grid, occupancy and capacity/balanced
masks across all three rounds. Grouped grids remain fixed measured choices:
total_rows/max_rows do not determine the exact per-expert tile sum. Unknown
M/router queries remain proposals, not admission. The original fixed-grid
leave-one-point-out counts below are **not validation of this pooled model**.
Grid recipes alone changed those counts to 1,503 within / 214 outside / 597
missing / 365 abstained; the new pooled tree has no all-point holdout claim.

## Targeted box validation: no compilation

**2026-09-07 result:** the run below completed and all 945 raw logs were
replayed. No numerical/launch failures occurred. Only 85/110 new-M predictions
met both performance limits; 76 grouped merge targets had stale fixed grids.
The conservative cross-epoch refit did not reduce the 788 leaves and lost
coverage, so it was not adopted. See [the review and proposed hybrid
heuristic](KPACK_HEURISTIC_REVIEW.md). The command remains a reproducibility
entry, not a request to rerun the same experiment.

The frozen suite contains **315 requests**: 45 original blockers, 160
adjacent-choice merge requests, and 110 previously unmeasured M boundaries.
It selects 909 parent/workload pairs and reuses **306 existing parents** from
the original confirmation bundle. This is the first prioritized merge wave,
not all 1,703 available opportunities or a universal interpolation proof.
Merge candidates retain a same-run incumbent; new M proposals are challenged
by both neighboring measured-rule parents where admissible.

On the original box, retain the completed campaign and its compiled cache:

```bash
git pull --ff-only origin develop &&
python3 -u tools/run_kpack_policy_validation.py \
  --campaign /workspace/kpack-overnight-c0c1361 \
  --output /workspace/kpack-policy-compact-v1 \
  --sdk /workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
  --devices 0,1,2,3,4,5,6,7
```

Source/SDK and payload hashes are checked before running. Missing payloads
fail explicitly; **there is no compilation fallback**. The runner uses three
fresh 11-sample rounds, correctness_repeats=1, isolated failures and resumable
receipts. Rerun the same command with `--retry-failures` to retry rejected
requests while retaining valid measurements. Original overnight timings are
never overwritten. Progress prints every 30 seconds; remaining time is advisory,
with future-round estimates based on completed fresh rounds, not launch counts.

Return `/workspace/kpack-policy-compact-v1/results/summary.json` first and
retain the new `phases/` logs/receipts for verified refitting. A complete run
can report slow or unavailable proposals; it cannot silently substitute a
different winner and call the frozen proposal a pass. No policy is updated
automatically, and **no new production `.so` is delivered by this host fit**.

To reproduce the compact fit locally, use new output paths:

```bash
python3 tools/refine_kpack_grid_policy.py \
  --source /path/to/extracted/kpack-overnight-c0c1361 \
  --out /path/to/new-grid-evidence
python3 tools/compact_kpack_policy.py \
  --evidence /path/to/new-grid-evidence --out /path/to/new-compact-fit
python3 tools/plan_kpack_policy_validation.py \
  --campaign /path/to/extracted/kpack-overnight-c0c1361 \
  --evidence /path/to/new-grid-evidence \
  --policy /path/to/new-compact-fit/covered-policy.json \
  --output /path/to/new-validation-suite.json
python3 tools/check_kpack_policy.py policies/kpack_zw810_compact.json
```

`covered-policy.json` is the selected parent-cover alternative;
`pooled-policy.json` favors fewer branches. Python/C++ parity covers 7,984
queries. This is host validation, not PPU execution. Kernel source identity
and the compiled module cache remain unchanged.

## Initial v1 baseline and evidence

The remainder documents the reproducible initial fit. Its files and holdout
results are retained as a baseline, not the current compact selection.

This is a **host selector prototype on develop**, fitted from the completed
`c0c1361` overnight search. It covers Q2/Q3/Q4/Q5/Q6 and all four routes:
FullyQuantized dense/grouped and ScaleFirst dense/grouped. It does not switch
between routes, run a kernel, change the offline format, or update a deployed
shared library. Xplane is not a candidate.

## Deliverables and use

- [policy JSON](../policies/kpack_zw810_v1.json): queryable data and source receipts.
- [C++17 header](../policies/kpack_zw810_v1.hpp): the same selector, SDK-free.
- [fit report](../policies/kpack_zw810_v1.report.json) and
  [45-request recheck plan](../policies/kpack_zw810_v1.recheck.json).
- [fitter](../tools/fit_kpack_tuner_policy.py): original-raw replay and rule fitting.
- [query tool](../tools/kpack_policy.py): no profiling and no compiled-default fallback.

For example, from the repository root:

```bash
python3 tools/kpack_policy.py policies/kpack_zw810_v1.json \
  --route fq-dense --qtype 12 --m 1 --n 8192 --k 5120

python3 tools/kpack_policy.py policies/kpack_zw810_v1.json \
  --route sf-dense --qtype 12 --m 2048 --n 1024 --k 5120

python3 tools/kpack_policy.py policies/kpack_zw810_v1.json \
  --route fq-grouped --qtype 12 --total-rows 8 --max-rows 1 \
  --experts 256 --n 512 --k 2048
```

The first two select, respectively:

| Route / shape | CTA / warp | Stages | AP / delivery-N | Algorithm |
|---|---|---:|---|---|
| Q4 FQ, 1×8192×5120 | 8×128×256 / 8×32 | 2 | AP0 / 32 | Split-K S4 |
| Q4 SF, 2048×1024×5120 | 64×64×64 / 64×32 | 3 | AP0 / 64 | Persistent, grid 512 |

Do not truncate the returned identity to an old geometry name. The selected
parent symbol, route, AP, delivery-N, stages, Split-K and scheduler/grid are
all significant. An ordinary SF grid is shape-derived and checked against the
recorded formula; persistent grids remain fixed measured choices.
The JSON `persistent` field is the parent's type restriction (`-1` means that
parent can emit either schedule), not a runtime boolean. The selected
`algorithm` owns the runtime schedule; C++ names the restriction
`parent_persistent` to distinguish them.

The query device defaults are PPU-ZW810 / 72 CUs. Integrators must pass the
actual device and arrangement identity, not assume those defaults prove a match.

```cpp
#include "policies/kpack_zw810_v1.hpp"
using namespace quactlize_kpack_policy;
Query q{12, 8192, 5120, 32, Route::FqDense};
q.m = 1;
q.mapping_id = 0x51344b5034540001ULL; // obtain from the canonical arrangement
auto choice = select(q);
// choice.config describes a kernel; it is NOT a function pointer.
// Bind its full identity to an admitted inventory before launching anything.
```

The result has three states:

| State | Meaning |
|---|---|
| `MEASURED_POLICY` | The rule meets its cost constraints on the observed public feature point. For grouped, this is conditional on the tested router fixtures, not every distribution with those features. |
| `INTERPOLATED_PROPOSAL` | A rule predicts an unmeasured point inside the same N/K/expert family. This is **not performance or correctness admission** for that query. |
| `NO_MEASURED_POLICY` | Unknown family/device/mapping, out-of-range features, noisy/conflicting measured key, or known invalid candidate. No compiled default is substituted. |

The C++ enum uses the same three states. The current library's any-M admission
promise must not be replaced with this partial policy's coverage. A consumer
must explicitly decline a miss/proposal or perform its own validated admission;
it must not discard a fallback weight representation on this policy's authority.

## Fit and evidence

The uploaded archive SHA-256 is
`03e81a5e53e445d5ba4f2813f087775f6b3eab17673c0f801673d51fd8e938f1`.
The initial review replayed all 13,572 raw logs across stages. The reproducible
fitter independently replays the **8,286 confirmation logs** and reconstructs
all 2,762 final decisions from their three rounds of 11 samples. Structural
exclusions and absent costs never become measured timings.

Each format/route/N/K/expert family is fitted separately. Dense rules split on
M; grouped rules split on total rows and maximum expert rows. Candidate costs
are regret against each round's best common confirmed variant, plus the
candidate's across-round spread. A leaf needs a common measured candidate
within 5% on **every** training observation. The fitter minimizes leaf count,
then regret. It is not a classifier trained only on the winner's name.

Forty public feature keys have multiple router fixtures. Their candidates are
coalesced by minimax constraints. Router names, private row arrays, tokens,
top-k and hashes are not selector features. Leave-one-point-out validation
removes the whole alias class to prevent leakage.

| Format | Input requests | Served by measured rules | Maximum training round regret |
|---|---:|---:|---:|
| Q2_K | 438 | 432 | 4.9593% |
| Q3_K | 438 | 435 | 4.9080% |
| Q4_K | 1010 | 993 | 4.8350% |
| Q5_K | 438 | 431 | 4.7684% |
| Q6_K | 438 | 426 | 4.9386% |
| Total | 2762 | 2717 | 4.9593% |

There are 152 families, 999 leaves, 484 runtime variants and 311 distinct
parent symbols. On the served observations, the mean median-time increase
relative to the original selected winner is **0.3917%**. These are training
cost bounds within the measured confirmation set, not global optimality.

The 45 blocked requests are:

- The original 42 noisy confirmations.
- One otherwise stable Q6 SF-grouped observation sharing its public key with
  a noisy observation; both must be blocked consistently.
- Two stable Q6 SF-grouped permutations at N2048/K512/E256,
  total_rows=534/max_rows=129 with no common qualifying confirmed choice.

This last case is a performance/evidence conflict, **not a numerical failure**.
The separate permutations were numerically correct.

## Interpolation is not yet a shipping default

Leave-one-public-point-out produces:

| Outcome | Points |
|---|---:|
| Measured prediction within both 5% limits | 1463 |
| Measured prediction outside a limit | 229 |
| Predicted candidate not measured at the held-out point | 622 |
| Explicit abstention, including range endpoints | 365 |

Of the 229 failures, 228 exceed regret and seven exceed spread (six overlap).
Worst observed holdout regret is 93.34%. In particular, simply interpolating a
persistent grid across M can be expensive. The full-data policy retains those
points as anchors; holding them out shows why unmeasured M values must remain
proposals. Neither the 622 missing costs nor the 365 abstentions count as passes.

Next refine interpolation around tile/wave/grid transitions and validate its
new candidates. Do not label the current interval tree a universal 5% heuristic.

## Reproduce locally

```bash
python3 tools/fit_kpack_tuner_policy.py \
  --source /path/to/extracted/kpack-overnight-c0c1361 \
  --out /path/to/new-policy-result

python3 tools/check_kpack_policy.py policies/kpack_zw810_v1.json
```

The fitter creates `policy.json`, `report.json`, `replay.tsv`, `replay.json`,
`holdout.json`, `followup.json`, `source-receipts.json` and `recheck-plan.json`.
The output directory must be new; original results are not edited. Absolute raw
log paths are relocated only in memory under the supplied extracted root.
Original parser hashes must match the campaign receipt.

The host checker compiles the exported C++ selector and compares it with Python
at measured points, blocked points, interpolation points, domain boundaries and
device/mapping negatives: **7,984/7,984 queries agree**. This is not PPU execution.

The recheck plan contains only the 45 blocked requests, 235 parent/workload
pairs and 149 already-compiled parents. It unions candidates across router
aliases; no new parent type is required. It is not the interpolation or shipping
gate, and fresh timings must remain separate from the original campaign.

All these changes are outside the kernel source closure. Its identity remains
`f00789d47f082ad47cea826c7ac38ac7b4c558e5a6c64a65cbb18924a36751cf`:
the existing tuner object/DSO cache is not invalidated.

## Runtime handoff still required

The old six-library release does not contain this selector or all its winners.
Its config v3/v4 descriptors cannot express the complete AP/delivery/scheduler
identity. Do not pass the generated benchmark symbol as an old `config_name`.

Remaining integration work is a versioned full-identity selector/inventory
binding, selected-parent production builds and selected-config device replay.
SF/FQ route choice and prepass amortization are separate. See the single
[llama.cpp handoff](LLAMA_CPP_KPACK_HANDOFF.md) before consuming a library.
