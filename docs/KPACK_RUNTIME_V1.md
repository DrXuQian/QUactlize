# K-pack runtime policy v1

Frozen 2026-09-07 on `develop`. This closes the first **host selection policy**,
not the production DSO/inventory integration. No new sweep or compilation is
requested to close this version. Kernel source remains
`f00789d47f082ad47cea826c7ac38ac7b4c558e5a6c64a65cbb18924a36751cf`.

## Deliverable

- `policies/kpack_zw810_runtime_v1.hpp`: SDK-free C++17 exact-table selector.
- `policies/kpack_zw810_runtime_v1.json`: equivalent data and evidence receipts.
- `policies/kpack_zw810_runtime_v1.report.json`: coverage and exception counts.
- `tools/kpack_runtime_policy.py`: standard-library-only query reference.
- `tools/freeze_kpack_runtime_policy.py`: reproducible evidence-to-table export.

The runtime contains **no fitted coefficients, branch tree, profiler or
candidate search**. It has 2,982 exact-input entries, 509 runtime recipes and
247 compiled parent identities. Nineteen deduplicated row vectors describe
the measured grouped fixtures. These are generated data, not 2,982 handwritten
rules. The earlier compact tree and 848-coefficient ranker are retained as
development artifacts; they are not part of this runtime header.

The shortlist scorer remains useful offline. It is not an unconditional
fallback or a claim that an unmeasured shape is within 5% of its best config.

## Last planned box challenge

Archive `kpack-tactics-validation-v1-results.tgz`, SHA-256:
`6e61f4c989c66cbb95534b89a5e247227655cbecd694678419f473303a0f3751`.
The 212-request suite matches the frozen source suite exactly. All 636 raw
logs were hashed and reparsed. Plans, per-phase receipts, source/SDK identity,
eight distinct physical devices and 3x11 samples agree with the final summary.
The run took 307.515 seconds: 102.844 / 102.270 / 102.401 seconds per round.
There were **no numerical, launch or infrastructure failures** and no missing
proposed runtime/grid. The archive does not contain the kernel binaries;
their payloads were not independently inspected locally.

| Frozen shortlist | Within both 5% limits | Regret within 5%, spread outside | No proposed tactic within 5% regret |
|---|---:|---:|---:|
| 102 previous cache misses | 86 | 7 | 9 |
| 110 genuinely new M values | 107 | 1 | 2 |

Blind Top-1 passed both limits for 106/212 requests. The shortlist contained
a passing choice for 193/212; allowing all measured same-run challengers gave
198/212 requests with a passing choice. The pooled fastest-winner status was
stable in 190 cases and noisy in 22; a noisy fastest candidate does not rule
out a different, independently measured stable candidate.

The summary's 468 within-budget and 1,907 outside-budget records are individual
**candidate** verdicts, not failed numerical requests. Most candidates in a
comparison may be slower than the best one. The policy selects one measured
tactic per input instead of requiring every proposed alternative to pass.

## Selection and timing claims

Three evidence epochs contribute 3,289 observations at 2,982 distinct exact
inputs. No absolute times are averaged across epochs. For each input:

1. Find runtime choices actually measured within both 5% regret and 5% spread
   at **every recorded epoch**. Greedily share compiled parents across these
   sets, without weakening either bound. This is a cover heuristic, not a
   proof of globally minimal parent count.
2. If no such common choice exists, retain the newest epoch's measured
   choice: prefer candidates meeting both limits there, then minimize that
   epoch's worst-round regret. Preserve all historical missing-cost, regret
   and spread exceptions. Do not keep a known slow incumbent just because it
   happened to appear in every older experiment.
3. Look up exact inputs at runtime. No interpolation is silently labeled
   measured. Grouped lookup compares actual row vectors, not only total/max
   rows; its hash is only an accelerator and full equality is still checked.

| Frozen result | Exact inputs |
|---|---:|
| `MEASURED_WITHIN_5PCT`, both bounds at every recorded epoch | 2,889 |
| `MEASURED_EXCEPTION`, numerical evidence retained but no cross-epoch timing bound | 93 |
| Total | 2,982 |

The 2,889 bounded inputs have maximum recorded regret 4.962783%. A pooled
winner's noise label is not used as a blanket veto when the selected alternative
itself satisfies the unchanged bounds in every epoch. This recovers twelve
inputs that the earlier cache builder excluded unnecessarily.

Among the 93 exceptions, reasons overlap: 49 have missing older costs for the
selected choice, 17 have measured historical regret above 5%, and 48 have
recorded spread above 5%. Across all inputs, the selected choice meets both
limits in its **latest available epoch** for 2,968/2,982. The remaining fourteen
comprise seven spread-only, two regret-only and five combined cases. Maximum
latest-epoch regret is 7.082630%; maximum measured historical exception regret
is 19.758065%. These are recorded-data statistics, not a population-wide or
unseen-shape performance guarantee. Missing older measurements are never passes.

For example, Q5 SF M128/N4096/K3072 previously chose a persistent 16x32x256
parent at about 39.7 us. The new run measured a nonpersistent 32x64x256 parent
at about 25.6 us. Requiring an all-epoch *measured* intersection would wrongly
retain the old choice at 56.55% overhead in the new comparison. The frozen
policy uses the freshly measured faster choice, but explicitly marks its
missing old costs as an exception rather than inventing a historical pass.

## Runtime contract

The header namespace is `quactlize_kpack_runtime_v1`. Set actual route, qtype,
N/K/group size, M or expert rows, device name/CU count, canonical mapping ID,
kernel-source fingerprint and SDK digest. The result is:

| Status | Meaning |
|---|---|
| `MeasuredWithin5` | Exact table hit with the recorded cross-epoch timing bound. |
| `MeasuredException` | Exact hit using a numerically measured choice; timing caveats remain explicit. |
| `FallbackRequired` | Input was not measured. Caller must use its admitted K-pack fallback or explicitly tune. |
| `BindingMismatch` | Device, source, SDK or mapping differs. Do not use this table. |
| `InvalidQuery` | Invalid shape or inconsistent grouped row vector. |

This v1 deliberately does **not** implement the caller's fallback kernel or
claim its performance. It does not return an arbitrary compiled default.
Unknown grouped routing vectors may require that fallback even if their
total/max rows match a measured vector. Empty workloads should be handled by
the caller before querying this positive-size policy. Calibration is FP16
full-output on PPU-ZW810/72 CUs, not BF16 or a route/prepass-amortization policy.

The returned config includes full parent symbol, AP, delivery-N, Split-K and
grid recipe. Bind this identity to a matching compiled inventory, obtain
workspace and run `can_implement` before launch. Do not translate it into an
old geometry-only config name. Grid formulas remain source-owned and use
actual grouped tile counts.

Local validation: all 3,289 observations replay against their chosen entries;
bounded entries satisfy both limits everywhere, and exceptions have the exact
selected measurement in their newest epoch. SDK-free C++ and Python selection
agree on **11,928 queries** covering every entry, wrong binding, unseen N and
invalid K. This validates the selector, not a rebuilt production library.

```bash
python3 tools/kpack_runtime_policy.py policies/kpack_zw810_runtime_v1.json \
  --route fq-dense --qtype 11 --m 48 --n 256 --k 3072
```

The CLI is a reference query using the policy's identity, not a probe of a
loaded DSO. A real consumer must supply its actual identity.

## Reproduce

First produce verified evidence from each extracted archive with
`tools/kpack_tactic_evidence.py`, using epoch labels `overnight`, `validation`
and `tactics`. Then supply files **oldest to newest**, with a new output prefix:

```bash
python3 tools/freeze_kpack_runtime_policy.py \
  --evidence /path/to/overnight.json \
  --evidence /path/to/validation.json \
  --evidence /path/to/tactics.json \
  --prefix /path/to/new-runtime-policy
python3 -m pytest -q tests/test_kpack_runtime_policy.py
```

## Why the development path looked heavier

K-quant plane/metadata variants, FQ/SF routes, decode providers and ragged
grouped scheduling are real requirements. They do not require a large runtime
fitting system. The earlier approach also exposed many independent tuning
axes, tried to compress sparse/noisy measurements into one universal tree,
and mixed development audit work with runtime selection concerns.

In the local DeepGEMM-for-sail checkout, some paths constrain warp geometry
from the CTA tile (`common_bf16.hpp`), and FP4 includes bounded tile tables
(`common_fp4.hpp`). There are also separate autotuners and saved config files.
Comparing our complete tuning/audit pipeline with only their runtime chooser
overstates the required runtime difference. Their particular candidate counts
or shape rules are not universal guarantees and are not copied blindly here.

This first version separates the concerns: tuning/evidence tools stay offline;
the runtime is a small lookup plus validity/grid logic. Known small differences
are merged as data, exceptions remain data, and no new model or full sweep is
required for first-version closure. Full-identity production binding and the
final DSO/device gate remain the next, separate integration tasks.
