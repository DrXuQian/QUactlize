# K-pack heuristic review after targeted validation

Date: 2026-09-07. Host implementation/calibration only: no kernel, production
selector, public ABI or shared library changed. The reference is the local
`GEMM 配置选择方法综述与对比.md`, especially sections 4.3 and 5.1–5.3.
Its reusable idea is a measured fast path plus a small ranked candidate set.
Its descriptions of other libraries are not proof that their constants,
resource limits or exact configuration choices transfer to these kernels.

## Evidence from the new run

Archive: `kpack-policy-compact-v1-results.tgz`, SHA-256
`f687bbb27fda8f92c8914fa5b6afb13839a233507064192c8ea85f1a62f7e372`.
All 945 raw logs were hashed and reparsed using the original parser. Suite,
phase plans, source/SDK receipts, eight distinct device identities, sample
counts and final decisions agree. Three rounds of 315 requests completed in
186.26 seconds. There are 15,801 measured runtime-cell records and 240
structural records, with no numerical or launch failure. Binary payloads were
not included in this upload and were not independently inspected locally.

The request-level `roles` field applies to all targets in that request.
Distinguishing the frozen incumbent by its config ID gives:

| Target kind | Within both 5% limits | Outside a limit | Exact runtime unavailable | Total |
|---|---:|---:|---:|---:|
| Newly proposed merges | 85 | 159 | 76 | 320 |
| Previously selected incumbents, remeasured | 113 | 47 | 0 | 160 |
| Previously unmeasured M | 85 | 25 | 0 | 110 |

For new M, 15 predictions exceed 5% regret and ten additional predictions
fail only the cross-round spread limit. The worst new-M regret is 78.98%.
These 110 points were deliberately chosen around transitions; they are not
a random sample or a population-wide error-rate estimate.

Of the 45 original rechecks, 30 have a stable selected-set winner this time
and 15 remain noisy. Clearing a request-level noise flag is not automatically
proof that all aliases of its public grouped key share one admissible choice.

All 76 unavailable targets are grouped persistent grids: the same parent and
schedule were measured in all three rounds, but the fixed grid copied from a
different workload was absent from the new workload's generated grid set.
This is a proposal-identity problem, not evidence that the parent cannot run.
Other measured grids remain usable evidence; they cannot retroactively turn
the original, different grid into a validated prediction.

## Why the present tree is not the final heuristic

A conservative shadow refit retained both timing epochs as separate
constraints. It did not mix absolute times, invent missing costs, discard
older rejections or promote the best alternate grid as the frozen proposal.
It still produced 788 leaves / 216 parents and reduced original-request
coverage from 2,717 to 2,671. This fit was rejected as a replacement: fewer
covered points are not successful rule compression.

The following direct comparisons identify decisions that should be modeled
separately from parent geometry:

| Query | Same compiled parent geometry | Frozen choice | Measured alternative |
|---|---|---|---|
| Q3 FQ, M48/N256/K3072 | 32x32x256, warp16x16, s2, AP0/DN32 | S1, 32.28 us | S4, 18.24 us |
| Q5 FQ, M48/N256/K3072 | 32x16x256, warp16x16, s2, AP0/DN16 | S1, 26.72 us | S4, 16.52 us |
| Q4 SF, M3072/N25600/K5120 | 64x64x64, warp64x32, s3, AP0/DN64 | Nonpersistent | Persistent, about 19.2% lower frozen-choice regret |

The last row's 19.2% is frozen-choice overhead relative to the comparison
reference, not a 19.2% runtime reduction. All references are the same-run
selected set, not a globally exhaustive optimum.

## Proposed reusable heuristics

These are design hypotheses to calibrate and replay locally. They are not
new shipping rules or an asserted universal 5% bound.

1. **Keep complete parent tuples.** Generate a small route/format-specific
   pool from measured coverage, with TM/TN/TK/WM/WN/stages/AP/delivery bound
   together. Do not recreate their Cartesian product. Historical winners
   remain challenges when a smaller pool is evaluated.
2. **Separate eligibility regimes before interpolation.** AP1 is a distinct
   M1 path. The current SF dense TM8 admission changes at M8; the current FQ
   dense Split-K admission changes at M64. Predictions between M4 and M8 or
   M32 and M64 must respect these implementation boundaries, not just use a
   geometric midpoint. These are current route restrictions, not new offline
   format constraints and not a blanket rule that every small M needs TM8.
3. **Rank parent geometry by work, tails and traffic.** For dense,
   `Q = ceil(M/TM) * ceil(N/TN)` and useful output fraction is
   `M*N / (Q*TM*TN)`. For grouped, use the actual
   `Q = sum_e ceil(M_e/TM) * ceil(N/TN)` from scheduler-owned row metadata.
   Total rows and maximum rows alone do not determine Q. Padding waste, K-loop
   depth, A/B/metadata traffic and actual compiled resource limits are useful
   features; minimum wave count alone is not a time model.
4. **Select Split-K using parallelism and completion cost.** Evaluate the
   small admissible set of S values for each shortlisted parent, using
   `Q*S`, `K/(TK*S)`, resident capacity and a calibrated reducer/workspace
   cost. The scoring target is producer plus real reducer time. Do not copy
   an S1 decision from M64 to M48 simply because the parent matches, or
   assume that more parallel blocks always compensate for extra completion.
5. **Resolve grids from a policy, not a neighboring grid integer.** Reuse the
   existing `capacity(Q,CU,b)` and `balanced(Q,CU,b)` functions. Calibrate b
   within the actual parent occupancy. Grouped needs its actual tile count
   after selecting a parent. Test fixture names and router hashes must not
   become heuristic features. Correct resolution of Q/grid is not proof of
   the fastest grid.
6. **Use measured tactics for exceptions and uncertainty.** Favor an admitted
   exact cache/table hit. Otherwise rank a small set (initial target: 3–5
   parents, to be justified by replay) and resolve their runtime choices.
   Optional warmup-time tuning measures only that set and caches the result.
   Latency-sensitive execution uses an admitted K-pack fallback when tuning
   is unavailable; its performance is explicitly unvalidated on a cache miss.

The simple existing scores in `tools/kpack_tuning_plan.py::estimates` were
designed to diversify a search shortlist. They explicitly are uncalibrated,
and should not silently become a top-1 production latency predictor. Fit a
small number of coefficients from the original measurements and evaluate
whole held-out public points, including their grouped aliases. Measurements
used to revise the model are no longer an untouched validation set.

## Runtime and maintenance boundaries

Represent a tactic as complete parent identity plus Split-K, scheduler/grid
policy and resolved workspace requirements. The kernel inventory, not the
heuristic, owns legality. Numerical failures must be isolated as correctness
failures, not learned as slow timings. A fallback is another admitted K-pack
tactic; it does not change resident weights or reintroduce Xplane.

Exact measured choices and exceptions belong in generated data/cache, while
the common selection code stays small. This reduces handwritten branches;
it does not magically eliminate the number of compiled parents or cache
records. Grouped cache admission must account for routing workload identity:
a match on total/max rows does not prove every expert permutation has the
same performance. Cache validity also binds device, source/kernel inventory,
SDK, arrangement and the full tactic identity.

Current production config v3/v4 cannot express the whole identity. No new
benchmark symbol should be sent to the old `config_name` entry. Public ABI
binding and final selected-parent production/device admission remain separate
work items in the integration handoff.

## Next local evaluation

Report separately: measured top-1 regret, top-3/top-5 near-optimal recall,
missing candidate costs, cache/table coverage, fallback rate and model size.
Retain every known winner when scoring recall; do not make the shortlist look
good by omitting hard queries. Missing sparse-matrix costs are unknown, not
passes. Only request new device measurements for the residual uncertain
candidate/query pairs. No new full Cartesian campaign is proposed.

## Implemented host prototype and calibration

The new query entry is `tools/kpack_tactic_model.py`; generated data is
`policies/kpack_zw810_tactics.json`. The old compact policy remains unchanged.
The selection pipeline is:

1. An exact measured tactic cache, including the actual grouped expert rows.
2. On a miss, up to two parent hints from the same measured N/K family. Dense
   uses an M bracket; grouped uses total/max-row distance only for hints.
3. A calibrated geometry score fills a shortlist of at most five complete
   parents. A separate runtime score retains up to three S/scheduler/grid
   choices per parent. This means at most **15 proposed runtime choices**,
   not five guaranteed-fast launches.
4. Runtime inventory admission is still required. If short tuning is not
   available, the caller needs an already admitted K-pack tactic; this host
   tool neither invents a compiled default nor falls back to Xplane.

No neighboring S/grid value is copied. Grouped CTA counts use actual rows.
All 161,871 persistent grid/mask witnesses in the original and follow-up
confirmation logs agree with the source capacity/balanced formulas. The
input reader reparses 9,231 confirmation logs; the two timing epochs remain
separate observations. No absolute-time average is formed across epochs.

The model has 20 route/format parent-scoring blocks and four route runtime
blocks, with 848 fitted coefficients. It has no branch tree, but this is **not
zero data or only 24 rules**: there are 2,770 exact measured cache entries,
2,733 family hint records and an existing 836-parent candidate inventory.
The cache currently references 346 parents. Reducing selection-code branching
does not by itself minimize the eventual shipping kernel set.

Each cache entry has a common runtime within both 5% limits in every supplied
observation of that exact context. Of 2,872 distinct contexts, 80 remain blocked
by noise and 22 lack such a common runtime. The larger context count is not
directly comparable to the old selector's public-feature-key count: actual
grouped row vectors now distinguish router aliases.

Calibration uses centered log full-output times and fixed ridge regularization
0.01. Parent targets use the best stable runtime of that parent; runtime
targets compare variants within one parent/query. All formats, routes, router
aliases and epochs of a public point are held in the same fold. The five-fold
results are internal model-development evidence, **not an untouched final
validation set** or a guarantee on arbitrary shapes.

| Actual candidate generation, stable observations | Top-3 contains a measured good tactic | Top-5 contains a measured good tactic |
|---|---:|---:|
| Geometry score alone | 1,512 / 2,985 | 1,885 / 2,985 |
| Family hints plus geometry score | 2,449 / 2,985 | 2,569 / 2,985 |

For the hybrid Top-5 set, the other 416 observations have insufficient costs;
they are unknown, not passes or proven losses. Ninety-two noisy/unconfirmed
observations remain separately reported. Blind Top-1 is **not admitted**:
1,579 within both limits, 601 known outside, 805 unknown; worst known regret
161.82%. Thus the model supplies a short tuning set, not an unconditional
single-tactic replacement.

Restricting evaluation to parents already timed at each query would produce
2,984/2,985 Top-5 coverage, but 2,515 queries have at most five measured parents.
That largely trivial conditional number is not the actual shortlist result.
The report preserves both denominators rather than presenting it as 99.97%
generalization. Detailed evidence is in `kpack_zw810_tactics.report.json`.

## Next bounded device challenge

`policies/kpack_zw810_tactics.validation.json` freezes **212 requests**:
102 exact cache misses plus 110 genuinely new M values (one per dense family).
It reuses 285 already compiled parents in 1,115 parent/workload pairs. No new
parent, kernel source change, all-config Cartesian product or rebuild is needed.
Historical winners remain same-run challengers, including across epochs; a
different winning runtime cannot retroactively validate the frozen proposal.

The existing runner measures all legal runtime variants of each selected
parent as challengers, which can exceed the 15 frozen predictions. The fixed
controls remain three 11-sample rounds and correctness_repeats=1. Progress,
failure isolation and resumption come from the existing runner. See
[the executable command](KPACK_POLICY.md#tactic-shortlist-prototype-and-next-box-run).

The previous 315-request run took 186 seconds, but that is not a time guarantee
for this different workload. There is no hour-scale compilation step; use the
runner's observed ETA. Upload the new results summary and retain phase logs
for raw replay. A new production library and full-identity ABI binding remain
separate tasks after this challenge.
