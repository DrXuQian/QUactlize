# Dense CTA-local token reuse: Q2–Q5 and Q8

Bounded follow-up to [the measured Q6 experiment](PPU_Q6_TOKEN_CTA.md).
This is an explicit experimental carrier, not a public ABI or production
selector change. Dense F32 endpoints, F16 activation rounding, FP32 accumulation,
canonical planes, per-row operation order and the real Split-K reducer remain.
MoE/BF16 and other N/K timings are not admitted by this cohort.

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
