# Dense CTA-local token reuse: Q2–Q5 and Q8

## Next sweep: actual model shapes, not all-format expansion

Use `COHORT=token-cta-model-gaps` for the remaining model scope. The source is
the existing `policies/dense_prefill_v1.json` registry: **45 real (q,N,K)
families × M1–8 = 360 points**, from the five requested models (including TP
local shapes). At N1024/K5120 the models contain Q4/Q5/Q6/Q8 only. The previous
Q2/Q3 tests at that N/K are synthetic format checks, not model coverage.

The gap cohort excludes the 32 already measured real-model points at that
N/K, leaving **41 families / 328 points**: Q8 136, Q3 16, Q4 80, Q5 48, Q6 48.
It contains **9,337 arms**, at most 37 per point, with 23 TC modules and 149
vector modules after compile deduplication. The plan keeps every current C++
selection, previous exact control and same-point near-winner. It adds the
declared token-CTA geometries/Split-K choices; TC geometry transfers and their
Split-K neighbors are excluded explicitly. This is not an exhaustive TC search.

No historical times are reused or relabelled as new passes. The plan records
the excluded families and old evidence IDs; changed measured kernel bodies
invalidate gap-only reuse. `COHORT=token-cta-models` explicitly reruns all 360
real points when fresh full coverage is needed. Neither cohort forms a format
Cartesian product, adds MoE, or promotes the resulting choices automatically.

```bash
(
    set -e
    cd /sim/eec/shared/junfu.qx/quactlize
    git pull --ff-only origin develop
    unset RUN RESUME_RUN PLAN_ONLY BASELINE QTYPES N K
    COHORT=token-cta-model-gaps GPUS=0,1,2,3 \
    PPU_SDK=/usr/local/PPU_SDK L2_BYTES=67108864 \
    IDLE_CONFIRMED=1 JOBS=8 \
    bash tools/run_smallm_regressions_asys_box.sh
)
```

Set GPUS to otherwise-idle physical cards. No llama rebuild or GGUF model read
is needed: independent GGUF-format fixtures use the registered local shapes.
The scheduler distributes complete paired points by estimated work, so one
card is not assigned M8 for every family. That score is not a measured latency.
Production Q8 fixed/measured/S1 entrypoints remain intact as controls.

Asys-only, six alternating rounds × five samples, each helper/producer/reducer
timed separately then summed by arithmetic mean. Same physical GPU per paired
point; source commit/tree, dependency, SDK, binary and raw intervals are bound
to the result. Request 1700MHz; actual clocks are advisory. Warmup/JIT/upload
are excluded. Native llama comparison and model TPOT are separate experiments.

Upload the printed `results/review-lite-*.tar.xz`, not the Asys/SQLite trees.
To resume only failed/missing points (preserve the original source/cache):

```bash
COHORT=token-cta-model-gaps RESUME_RUN=/workspace/kpack-smallm-asys.XXXXXXXX \
PPU_SDK=/usr/local/PPU_SDK L2_BYTES=67108864 IDLE_CONFIRMED=1 JOBS=8 \
bash tools/run_smallm_regressions_asys_box.sh
```

The resume reads the original GPU list automatically. `PLAN_ONLY=1` generates
the plan without GPU work. The earlier 40-point command below remains a format
experiment; do not use it to claim model-shape closure.

## Earlier format extension and exact admission

Bounded follow-up to [the measured Q6 experiment](PPU_Q6_TOKEN_CTA.md).
The explicit experimental carrier is retained. The reviewed mean winners now
have exact production bindings as described below; there is no public ABI
change. Dense F32 endpoints, F16 activation rounding, FP32 accumulation,
canonical planes, per-row operation order and the real Split-K reducer remain.
MoE/BF16 and other N/K timings are not admitted by this cohort.

## Exact production binding

`policies/kpack_token_cta_v1.json` records all forty decisions, candidate means,
component identities, numerical errors, module hashes and source/profiler
provenance. It imports archive SHA256
`28a8d7dbaea294311ac1d2b61c329b39068c246f8bf3a585e0a146069a3c1631`,
measured at source `1048a176a468e07abf30685f25344a4bae7c3410`.
Selection minimizes the arithmetic mean of isolated component sums, including
Split-K reduction. It is not an end-to-end or model latency claim.

Thirty-five exact decisions change: 32 choose the token-sharing body; Q2/M1,
Q5/M1 and Q4/M2 choose better existing configurations. Q4/M8 and Q8/M8 keep TC.
Q4/M3 is a 0.03% mean near-tie, not a demonstrated material speedup. Q8/M7 is
also marginal (0.51%). These remain mean-selected under the requested policy.
No MoE, BF16-compute, M>8, other N/K, or bucket donor/eligibility is modified.

`token_cta.inc` is shared by the host explanation and production launcher.
The ordinary SIMT ABI still validates workspace and dynamic addresses; launch
uses the unchanged measured template plus its existing reducer. Package
validation requires the exact body capability, not merely matching V/C/W/P/S.
Public plan names append `:tokensM:windowW` for the new implementation.

Regenerate/check on the host:

```bash
python3 tools/generate_token_cta.py --check
python3 tools/generate_decode_selection.py --check
bash run_tests.sh --suite host --sdk "$PPU_SDK"
```

Rebuild the complete public library/runtime package with `build.sh --library`
before using this selection. `run_tests.sh --suite accuracy --package PACKAGE`
includes all forty final public calls (two streams, changed addresses, padded
strides, graph replay and output guards). This production-library device gate
is separate from the uploaded experimental timings; host tests do not admit
new model TPOT or native parity for formats without a native baseline.

## Inventory

N1024/K5120, Q2_K/Q3_K/Q4_K/Q5_K/Q8_0, each M1–8: **40 points**.
M1 is the unchanged control. At M2–8:

- Q2–Q5: C4/W8/P8, C4/W2/P8, C8/W4/P4 × S1/2/4/8: 336 new cells.
- Q8: C4/W8/P4, C8/W4/P4 × one/two-half load windows × S1/2/4/8:
  112 new cells. Retain Q8's scale-only units and fast packed code reconstruction.

The 1,093 total cells comprise 448 new candidates and 645 current-body controls
and alternatives, including 60 TC controls and nine specialized Q4 cells.
All current C++ choices, previous confirmed choices and same-point historical
winners/within-5% alternatives are kept and timed afresh. TC geometry transfers
and Split-K neighbors are explicitly excluded (204 fewer cells); this is not
another TC sweep. The plan records every exclusion. There are 14 TC modules
and 153 vector modules; runtime Split-K does not require four compilations.

## Delivery and arithmetic

Each CTA owns all tokens for one N tile/K partition. Grid `[N/(C*P),1,S]`
replaces `[N/(C*P),M,S]`. Lane N is `tile*C*P+(lane%C)*P`; K group worker is
`thread/C`, starting at `partition*(W*32/C)+worker`, stepping `S*(W*32/C)`.
Q2/Q3/Q6 groups have 16 K values; Q4/Q5/Q8 groups have 32. K%256 ensures
whole-warp active/inactive tails for these mappings, including empty partitions.
Each token retains its own F32 dot, activation packet, warp partial and output.

Q2–Q5 instantiate the same CommonFormat/maps/metadata/code primitives as Q6;
Q2/Q4 compile out the high plane. Q5 keeps the N-bit3/K-bit7 high-plane shuffle,
not a naive one-bit Plane. Q8 uses its existing `scales` and `code_pair`
primitives, incremental weight/scale pointers and explicit load-window size.
The Q8 half→segment→slot→residue FMA order does not change between windows.
Packed-mantissa half2 reconstruction is exact for the code domain; accumulation
is FP32. No scalar per-code integer-to-float decoder or activation INT8 is added.

First-warp first-residue **source request** model, 128B plane alignment:

| Format | C4 low/high requested→unique bytes | C8 low/high requested→unique bytes |
|---|---|---|
| Q2 | 512→128 / absent | 256→64 / absent |
| Q3 | 512→128 / 512→64 | 256→64 / 256→64 |
| Q4 | 512→512 / absent | 256→256 / absent |
| Q5 | 512→512 / 512→128 | 256→256 / 256→64 |
| Q6 | 512→256 / 512→128 | 256→128 / 256→64 |
| Q8 (P4) | 256→256 / absent | 256→256 / absent |

Q2–Q6 low requests occupy their 32/64B sectors completely and half their
128B sectors. Q5/C8 **high** requests occupy only 50%/50%/25% of 32/64/128B
sectors because of its physical shuffle. Q8/C4 low and aligned scale requests
occupy 100%/50%/25%; C8 occupies 100%/100%/50%. Sparse Q2/Q3/Q6 metadata remains
strided. C4/group32 A has two float4 source loads, each half-sector populated;
do not claim an emitted float8 or add another transpose to hide that model.

The change removes repeated B/metadata/code work **across tokens**; it does
not eliminate the remaining within-warp duplication, fix all coalescing, or
prove DRAM-byte reduction by M. A loads, useful dot work and output stores
still scale with M. Tests derive 32/64/128B footprints, including legal Q8
two-byte scale alignment. Candidate receipts record actual plane/A/output/
workspace base modulo128. Native codegen, not source width alone, is checked.

## Local evidence and limits

177 vector units compiled with PPU1.0 SDK2.1.1 HGCC: 112 new format bodies,
21 generalized Q6 bodies and 44 controls. Q6's 21 producer instruction bodies
match their previously measured binaries after removing local label numbering
and trailing section alignment. Q6 control/reducer instructions also match.
No TC source, production table, packing layout or caller was changed.

Reported native stack: Q2 and Q8 candidates all zero; Q3/C4/M8 8B,
Q4/C4/M8 24B, Q5/C4/M7 24B and M8 204B. C8 candidates all report zero stack.
These alternatives stay visible for measurement, not silently promoted or
assumed faster. Q8/C8 M8 uses 154/160 vector registers for half/full windows.
Packed code construction remains shared while F32 FMA sites scale with M.

Host checks exercise the generated query ABI for every format, raw GGUF code
extraction against physical loaded-word addresses (including Q5 shuffle),
exact fast-code reconstruction, K tail coverage, partial ownership, load order,
entry/grid checks, current winner retention, and the real CLI/resume path.
This is not a GPU numerical result. The Box independent-GGUF, changed-input,
wrong-plane, zero-input, omitted-producer and redzone gates must still pass.
Focused host suite: 197 passed; 138 opt-in tests skipped (the 133 candidate
object compilations were run separately above). Local SDK installation has the
compiler but lacks the four runtime link libraries; local shared-library link
was therefore not verified. Box builds/links with its complete SDK before
running numerical gates. No prebuilt executable payload is claimed here.

## Box command

```bash
(
    set -e
    cd /sim/eec/shared/junfu.qx/quactlize
    git pull --ff-only origin develop
    unset RUN RESUME_RUN PLAN_ONLY BASELINE QTYPES
    COHORT=token-cta-formats GPUS=0,1,2,3 \
    PPU_SDK=/usr/local/PPU_SDK L2_BYTES=67108864 \
    IDLE_CONFIRMED=1 JOBS=8 \
    bash tools/run_smallm_regressions_asys_box.sh
)
```

Confirm these cards are idle before setting IDLE_CONFIRMED. No llama rebuild
is needed. The wrapper freezes the commit/dependency and builds the bounded
modules on Box, with no nested network submodule fetch. Measurements use only
Asys GPU intervals, separate helper/producer/reducer kernels, alternating six
rounds × five samples, rotating >2.25×L2 weights, and arithmetic mean sums.
Warmup/upload/JIT are excluded. 1700MHz is requested; actual deviations are
recorded, not failed. Adjacent timestamp overlap is not an admission gate.

Upload only the printed `results/review-lite-*.tar.xz`. Full Asys/SQLite remain
under `results/gpu-*/attempts/`. Resume with the printed RESUME_RUN and same
cards; only failed/missing points rerun. This is new-vs-existing Quactlize;
native Q5/Q6 baseline receipts remain separate and bound to their old source.
Native Q2/Q3/Q4/Q8 comparisons and production admission are subsequent gates,
not claims made by this plan.
