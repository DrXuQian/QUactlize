# MoE sweep preparation

This is a **host planning/coverage gate**, not a timing runner, a new production
selector, or a claim that the dense PR now executes MoE. Keep the caller PR
unchanged during this stage.

## Current priority and backup list (2026-10-27)

Active: optimize the indexed SIMT decode reader for tokens 1–8, first with a
bounded same-expert two-row reuse experiment. Keep canonical K-pack bytes,
direct BF16 A rounding, original FP32 B group-affine math/accumulation/output
and existing Split-K partial order.
The existing common/selected reader is the control; no production admission
or table update follows from host tests or compilation alone.

Backup, retained but not on this experiment's critical path:

- Merged/paired gate-up and fused SwiGLU configuration scanning, including
  physical 2N shapes and each output/rounding contract below.
- Broader fused TC/SF/full-provider implementation and large-M MoE scanning.
- New MoE table/bucket admission and caller integration after fresh evidence.

The fusion inventory and recall tools remain available. Deferral does not
reinterpret an old dense result as a fused result or remove fusion coverage
from the eventual scan.

### Bounded decode experiment

The opt-in `moe-token-pair` cohort reuses the existing Asys-only runner; it
does not add a production selector or change an installed kernel. Its runtime
row-pair mapping is shared across N/K, not an exact-shape implementation.

- Qwen35 separate Q4 gate/up N512/K2048, and Q5 down N2048/K512.
- Per family: M1 spread, M4 repeat2, M8 spread and M8 cluster. These are named
  synthetic router controls, NOT a claim to have captured real model IDs.
- Two geometries, S1/S4, current selected body plus matched single-row/pair
  bodies: 8 points, 64 arms, 13 vector modules at the recorded baseline.
- M1 keeps the old body. For M>1, only matching experts share B/code/metadata.
  A, outputs and split partials retain original routed row and stride semantics.
- Pairing scans at most 64 IDs in each warp, without a new helper launch. It
  increases live accumulators/registers; zero-collision and high-collision
  results must both be reported. No universal speedup is assumed.

Local plan-only invocation (no device or performance claim):

```bash
python3 tools/run_smallm_regressions_asys.py \
    --cohort moe-token-pair --plan-only --output /tmp/moe-pair-plan
```

After these sources are committed and delivered to Box, the existing frozen
runner accepts `COHORT=moe-token-pair`. It uses `IDLE_CONFIRMED=1`, `GPUS`,
verified `L2_BYTES`, fresh Asys samples, separate reducer timing, the commit
and transitive source/binary identities. An old checkout does NOT contain the
new cohort; do not execute its archived HEAD and call it this experiment.
Only `review-lite-*.tar.xz` needs uploading initially; raw traces stay on Box.
GPU correctness, measured benefit, all-M/model coverage, native comparison
and production admission remain separate pending gates.

Local checkpoint: 170 host/regression tests passed; 15 PPU1.0 SDK2.1.1
Q2–Q6/geometry instantiations compiled with zero stack. The embedded common
reader/reducer code retained 75 identical machine-code bodies across the
initial and simplified carriers. These are not GPU numerical/performance
results. C4/P8 pair registers rise by 18–26 over the one-row body; C8/P4 by
12–14. Pairing is deliberately not enabled in production.

At aligned bases, the unchanged source footprint model gives low-plane
32/64/128-byte unique-sector use of 100/100/50 percent for both geometries.
Q5 high is 100/100/50 for C4/P8 but 50/50/25 for C8/P4; Q4/Q5 A improves
from 50/50/50 to 100/100/100 when moving C4/P8 to C8/P4. This is why both
geometries remain controls. Pair reuse halves B/code/metadata requests for a
matched pair without improving these per-instruction footprints; A requests
and useful FLOPs are unchanged. It is not a measured HBM-traffic reduction.

## Rule-first, then reveal historical references

```bash
python3 tools/plan_moe_strategy.py --output /tmp/moe-rule-audit.json
```

The output must be a new file. Exit 0 means the declared historical recipe
recall passed, not performance or complete target-model coverage. A miss exits
1 and remains visible; it is never fixed by appending the missing champion.

The catalog is projected to configuration/geometry records with old times,
sample distributions, admission labels and timing priorities removed. Generate
candidates while withholding the entire target N/K family across M, qtype,
precision and router. Afterwards audit effective production choices,
individual router-profile winners and historical prefill winners. Fused
gate/up recipes have their own recall audit, not a tile-only alias.

Rule refinements retain indexed low-warp Q4 readers, one-axis SIMT column and
reader peers, distinct prefill M/N tile sizes, and portable SF delivery peers.
Every candidate still needs compiled resource and independent numeric gates.
These are sparse neighbors of blind seeds/observed template families, not a
Cartesian product of axes or per-shape champion exceptions. The ordinary
dense prescreen keeps its previous defaults.

`--seed-width 1` is a host budget experiment. It must pass the same recall
gate; a smaller pool is not useful if it drops known winners. Candidate and
module counts are reported before timing. Historical audit points are NOT
automatically scheduled on Box.

## Actual model and fusion denominator

Optionally supply `--models models.json --llama-source /path/to/llama.cpp`.
`models.json` is a list of `{ "name": "...", "path": "/path/first-shard.gguf",
"tp": 1 }`. TP2/TP4 use the unchanged caller split function in a CPU carrier,
not a model load or inferred `N/TP, K/TP`. The caller source hashes and complete
GGUF shard headers are bound to the inventory; unknown extents stay blocked.
Expert-axis partitioning is not admitted by this N/K-TP inventory.

Token anchors are 1–8, 16/32/64/128/256, 512/1024/2048/4096/5120. Only actual
quantized expert weights become model points; Q2–Q8 are not multiplied into
every model. Gate/up input channels=1; down channels=topk. Token T is not the
per-expert row count or the total routed row count T*topk. Real routing capture
remains separate; old fixtures called `real` were synthetic distributions.

The scan must include these execution forms, in addition to down:

| Form | Physical weight N | Output N | Extra contract |
|---|---:|---:|---|
| Separate gate/up | N each | N each | Two projections plus SwiGLU |
| Merged gate/up | 2N | 2N | Canonical per-expert gate-then-up; separate SwiGLU |
| Gate/up + SwiGLU | 2N | N | G4/U4 paired layout; F32 original-row output |
| Gate/up + SwiGLU | 2N | N | G4/U4 paired layout; BF16 compact-row output |

For Qwen3.5-35B-A3B headers, the two Q4 projections have N512/K2048;
their fused physical weight is N1024/K2048. Down is Q5 N2048/K512.
Gate/up pairing requires matching format, K/N, expert extent and local rank
partition. Native merged GGUF weights are not doubled again. Incompatible
pairs retain an explicit reason and their unfused controls.

Fusion keys include layout, projection rounding, output type/order, consumer
shape and epilogue. Even identical dense N/K does not supply its measurement.
Candidate templates hold out BOTH logical and physical target N/K. Generic
paired recipes are generated from the API/template family, not looked up in
the historical selector. New shape-specialized reader instantiations and
paired-register TC epilogues require compilation and numerical validation.
The current fused indexed TC processes one routed row at a time; expert-grouped
multi-token reuse and broader SF/full-provider composition remain explicit
implementation gates. `execution_ready`/`sweep_ready` stay false at this stage.

## Fresh measurement only

All scan times must be new **Asys GPU activity** samples. Every result must bind
the full Quactlize commit, actual source digest, actlize revision, compiler/SDK,
runtime, physical device, binary digest, recipe/body, router fixture and Asys
trace/range. The host plan reports `<commit>-src-<digest>`; this is a result tag,
not a Git tag or a substitute for a Box binary receipt. Dirty source is
fingerprinted rather than silently called the committed binary.

Old timings never enter the new table. Resume is only for valid results from
the same fresh campaign and implementation/measurement identity. Prepare,
metadata/full expansion, main, reducer and finish are profiled separately;
sum actual calls once. A fused reducer/SwiGLU is one measured component, not
an estimate plus the same standalone helper again. Native uses identical raw
weights/A/IDs and the same output boundary; record its actual arithmetic.
JIT, provider tuning and first-use warmup are outside measured samples.

Sequence: close host inventory/rule coverage → implement and validate the
BF16 indexed/fused candidates → fresh bounded Asys/native scan → measured
table/bucket admission → caller integration and model validation. No stale
timing transfer, automatic admission, or runtime configuration tuner.
