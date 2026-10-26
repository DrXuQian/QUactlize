# Q6 dense CTA-local token reuse

Experimental candidate, not a production selector update. Base: `654efb1c`.
Primary cohort: Q6_K N1024 K5120, M1–8; focus regressions M5–7.
No new packing, ABI, llama.cpp dependency or INT8 activation path.

## Contract and mechanism

F32 input/output, existing F16 activation rounding, FP32 group-affine dot and
accumulation. Canonical low/high/unit planes, per-token group/half/slot/residue
FMA order and `[row,split,N]` partials remain unchanged. S2/4/8 uses the existing
SIMT reducer, not the TC split-major reducer. M1 remains the old implementation.

The new grid is `[N/32,1,S]`, rather than `[N/32,M,S]`. Each CTA owns all M rows
of its N/K partition. Packed loads, sparse metadata exchange and code extraction
are outside the token loop. Each token has its own activation packet and FP32
accumulators. No new shared staging: only the existing warp-partial fold grows
to M rows, with one CTA barrier. Less repeated weight work trades against fewer
CTAs and more registers; no speedup is inferred from traffic alone.

For C columns, P values/lane and W warps:

- Lane column: `tile*32 + (lane%C)*P`; K worker: `thread/C`.
- Group ownership: `partition*(W*32/C)+worker`, stride `S*(W*32/C)`.
- Q6 low word: `((k/32)*8+k%8)*N+n`; high word uses `k/64` instead.
- Metadata: `(g/32*N+n)*36`, existing phase-dependent 3/4-word packets.
- A: `row*a_stride + g*16 + (lane%C)*(16/C)`, 16B/C4 or 8B/C8 loads.

Per-residue first-warp source footprint (128B-aligned plane bases):

| Mapping | Low lane/unique bytes | High lane/unique bytes | A unique bytes |
|---|---:|---:|---:|
| C4/P8 | 512/256 | 512/128 | 512 |
| C8/P4 | 256/128 | 256/64 | 256 |

Low/high requests still have 2x/4x inter-group duplication and 64B contiguous
N runs. Their modeled 32/64B sector utilization is 100%, 128B utilization 50%.
This change removes repetition **across tokens**, not that existing duplication.
Metadata remains strided: a scalar word from each 36B unit is not a contiguous
warp load. A is unchanged per token. Actual ring pointer alignment is recorded
by the existing fixture; source request models are not HBM traffic measurements.
The inherited `common_codes` uses packed half2 bit construction/subtraction and
then FP32 operands, not scalar integer-to-FP32 conversion per code or half FMA.

## Local checks, not device admission

All 21 candidate instantiations (M2–8 × three mappings) compile with the local
PPU1.0 SDK2.1.1 HGCC. Three controls compile too. After ignoring disassembler
local-label numbering and trailing section padding, reducer instructions match
the controls. The old producer and production selection files are unmodified.

| M | C4/W8/P8 vregs | C4/W2/P8 vregs | C8/W4/P4 vregs |
|---|---:|---:|---:|
| 5 | 188 | 188 | 124 |
| 6 | 212 | 212 | 138 |
| 7 | 248 | 248 | 152 |
| 8 | 254 | 254 | 166 |

M2–7 and M8/C8 have zero reported stack. M8/C4 reports 28 bytes of stack;
retain it as an explicitly measured alternative, not an assumed improvement.
These are local compiler observations, not Box occupancy or latency evidence.
For C8/M7, FP32 FMA sites grow 68→476 (=7x useful row math), while packed
code-construction sites stay 32 and scale conversion sites stay 4. This supports
real code/metadata reuse; whole-body static instruction counts are not cycles.

Host tests cover real generated query ABI rejection, exact row count, padded
strides, whole-warp K tails, unique partial ownership and actual captured grid.
Device independent-GGUF, changed-input, zero, wrong-plane, omitted-producer and
redzone checks must pass before any latency is admitted.

## Box

```bash
(
    set -e
    cd /sim/eec/shared/junfu.qx/quactlize
    git pull --ff-only origin develop
    unset RUN RESUME_RUN PLAN_ONLY
    COHORT=q6-token-cta GPUS=0,1,2,3 \
    PPU_SDK=/usr/local/PPU_SDK L2_BYTES=67108864 \
    IDLE_CONFIRMED=1 JOBS=8 \
    bash tools/run_smallm_regressions_asys_box.sh
)
```

Only confirm idle after checking those devices. Override GPUS if necessary.
This is 8 points / 246 configurations: 84 new candidate cells plus 162 existing
control/candidate cells. It builds 2 TC and 25 vector modules; Split-K is a
runtime axis, not four compilations. The ordinary ten-point recovery remains
the default when COHORT is unset. No network submodule fetch is attempted.

Every point keeps its current C++-selected config and all matched per-token
SIMT controls. Use Asys GPU durations only, six alternating rounds, five samples,
full cold-weight ring traversals >2.25x verified L2. JIT/upload/warmup are excluded.
Producer/reducer/helper are independently timed and summed, with no fabricated
launch penalty. Mean ranks the configs; minimum/spread remain diagnostics.
1700MHz is requested; actual clock deviations are recorded, not auto-failed.

Resume only failed points using the printed RESUME_RUN and the same devices.
The run retains frozen source, source commit/tree, dependency commit, SDK,
physical device, binary hashes and raw GPU intervals. Upload only the printed
`results/review-lite-*.tar.xz`; full Asys reports stay in
`results/gpu-*/attempts/*/trace.asysrep` on Box.

This cohort answers whether CTA-token reuse beats the current Q6 implementation.
It does not by itself establish native llama.cpp parity, other N/K performance,
MoE correctness, model TPOT or production admission. Do not compare its Asys
means to historical native event numbers as if they were paired measurements.
