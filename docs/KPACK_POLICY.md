# K-pack measured policy v1

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
