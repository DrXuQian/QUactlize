# Measured dense prefill helpers

This is a helper integration, not a new GEMM route table or a model-level
performance admission. `quactlize/dense/runtime.cpp` calls the same compiled
policy and typed runtime used by the diagnostic. Public caller ABI, offline
K-pack bytes, NCP wrapper and llama.cpp code are unchanged.

## Initial R4 scope (TC/SF unchanged)

PPU-ZW810, 72 CUs, contiguous dense F32 endpoints, M=2048. Other devices,
M, strides and unlisted N/K retained their previous helpers. MoE and decode
are unchanged. Full-dequant's existing production route remains BF16 with
FP32 accumulation, even in the default F16 context. TC F16 stays F16.

| Q | N | K | Typed dequant | F16 IO | BF16 IO |
|---|---:|---:|---:|---:|---:|
| Q4_K | 25600 | 5120 | 21 | 3 | 3 |
| Q4_K | 8192 | 5120 | 21 | 3 | 3 |
| Q4_K | 5120 | 8192 | 20 | 1 | 1 |
| Q4_K | 5120 | 25600 | 8 | 1 | 1 |
| Q4_K | 1024 | 5120 | 20 | 2 | 2 |
| Q6_K | 5120 | 25600 | 21 | 1 | 2 |
| Q6_K | 1024 | 5120 | 20 | 2 | 2 |

The single source is `quactlize/prefill/dense_helpers.hpp`, exposed through
`quactlize_kpack_dense_helper_policy_v1`. The query checks device/shape and
does no GPU work. The build receipt binds its hash to the private prefill
runtime; ncp still builds/installs the ordinary public package.

DQ20/21 = cooperative metadata + separate low/high shared-bank maps,
N64/K128 or N64/K256. DQ8 retains the prior N32/K128 pair reader.
IO1 = vec8; IO2 = two warp-contiguous vec4 chunks; IO3 = four chunks.
Do not extrapolate IO3 to small outputs: R4 measured a 10–12.5% regression.
Q4 5120x25600 retains DQ8 because the new full-call result was inconsistent.

Evidence: `review-1792644908006383291.tgz`, SHA256
`4f80fee84ea87413da6a11808aac7ce7cca83f6c039749400605b0ff1c59820a`,
source `afe3ee611cedd3d5b8e985bbea28588fedec2690`. Seven points / 112 arms /
40 ACU captures passed. Raw-bit IO/dequant, changed addresses, two streams,
redzones and independent GGUF dot checks passed. These are measured helper
choices; no whole-model latency is inferred from their component gains.

## Re-sweep against native

Use `tools/run_prefill_model_asys_box.sh` as documented in
`PPU_PREFILL_ASYS_SWEEP.md`. It remeasures all retained FQ/SF configs with
the production IO helpers, plus old model/current helper controls, old Full
BF16 configs, and the production typed helper combination for both Full
F16 and Full BF16. It does not cross every helper variant with every GEMM.

The typed full arms use production `prepare/run` entries. Every SF call
expands scales online; TC cases, including the old-helper controls, use
production enqueue order SF -> input -> GEMM/reducer -> output. Old-helper
controls are labelled `legacy-*`. Older diagnostic times with input before
SF are not imported into this run or relabelled as its control times.

Native must actually call cuBLAS with F16 A/B/C and FP32 accumulation;
the name `build-ci-32F-cublas` is not sufficient. Compare `same_dtype_*`
summary fields first. A faster BF16 arm is not relabelled as F16 parity.
Full F16 is a new explicit sweep candidate, not a silent production dtype
change; model numerics are still required before such a route is admitted.

## Next instruction-pressure experiment (not shipped)

In R4 N64/K256 native ISA, Q4 has 52 vector shifts / 39 integer MADs;
Q6 has 118 / 61. These are static instruction-body counts, not dynamic
instruction totals or fractions of latency. Shifts serve code/scale bit
extraction as well as address/swizzle calculations. Integer MADs include
address strides. MOVs include register setup/zero initialization, not HBM
transfers. The source still computes per-iteration producer/consumer indices.

Candidate: hoist common plane bases and strength-reduce repeated index
arithmetic, then inspect emitted code. Keep mandatory bit extraction,
separate low/high bank layouts, canonical bytes and exact affine rounding.
Investigate initialization/MOV provenance before removing it; do not disable
the compiler's `ppu-fix-uninit` correctness workaround to improve a count.
Confirm raw bits, occupancy and same-frequency cycles before promotion.

## Retired experiments

The high-plane unique-read R5 candidate was not admitted: paired cycle medians
changed by -1.21%, +0.08%, -0.67%, +6.16% for Q6 large/small and Q5 large/small.
Two samples per arm do not establish the small improvements; Q5 small regressed.
The R1–R5 runners and unselected typed helper bodies have been removed from
develop. Source and historical commands remain at commit
`8a004d42a91a23664e69762509d34c3b76bbf8e8`; do not run them against a new library.

The retained R4 production helpers are typed IDs 0/8/20/21: direct, pair N32/K128 and cooperative
N64/K128,K256. Their IDs, descriptor ABI and the selection table above are
unchanged. IO0..3 and the separate general SF/full-dequant config namespace
are retained. All removed typed IDs fail validation rather than falling back.
Independent GGUF, ownership/tail and changed-address/stream tests now live
together in `tests/test_prefill_dense.py`.

## Completed Full-only admission, 2026-10-23

The helper campaign is complete: 910 component cells, 700 Asys capture
batches, 175 shape/M points and 350 separately typed choices. Two labelled
Q2 extension families are included alongside 33 observed model families;
no DeepSeek TP4 shapes have been invented. The compact source of truth is
`policies/dense_full_helpers_20261023.json`; regenerate/check its C++ table
with `python3 tools/generate_dense_full_helpers.py --check`.

`quactlize_kpack_dense_full_helper_policy_v1` selects the measured Full
helper pair for exact qtype/N/K/M/storage, M=128/256/512/1024/2048, contiguous
dense endpoints on ZW810/72CU. Its evidence binds two complete-call GPU-sum
confirmations per typed point, not just the sum of isolated components.
The public dense provider and typed Full diagnostic both query this entry.
Unknown devices/M/shapes/strides retain their old helpers. The old
`quactlize_kpack_dense_helper_policy_v1` remains unchanged for TC/SF adapters.
The runtime and public manifest bind both the policy header and generated
table, and the public build rejects a stale table.

This does NOT select a different GEMM, change Full's default BF16 arithmetic,
enable automatic Full-Q8 routing or interpolate new buckets. Q8 typed Full
is still an explicit diagnostic endpoint. Q8's two >5% Full-F16/native
regressions remain visible: M512/N8192/K2048 (95.101 vs90.100us), and
M128/N1024/K5120 (27.58025 vs26.160us). No universal native-parity or model
TPOT admission is claimed. Raw SQLite remains on Box; the local merge
audited sealed receipts/proofs/samples, not re-exported GPU intervals.

Archive SHA256s:
`d3d8b1253ffacc29a1bd7af8f5aec449499c7720c8a89a83d560b9d95003ede2`
and `d1e85e19658192c279a83e9a6ea57f542f088999995bb29436d94f38216c63e1`.

### Q8 exchange return: rejected

Return `review-1792738826567943419.tgz`, SHA256
`83ddbd57ae0882e8eec476712e0e119d431c4e0d7962a11b43d8611445a0dba6`,
source32a8320:60 numerical cells,48 timing cells and8 ACU reports all pass
coverage/correctness. Every one of the12 typed points still prefersDQ22;
DQ24 regresses0.91--3.10%, DQ25 also loses. RetainDQ22; no production promotion.

On profile5 BF16(N8192/K2048), equal1.7GHz ACU cycles29213->30649(+4.92%).
Shared-load transactions-40%, but dynamic instructions+15.1% and scalar
instructions+89.2%. Global low/store traffic is unchanged. Register reduction
44->40 does not raise theoretical occupancy (both100%). Control shared-load
bank conflicts are0; store conflicts are49152 for both. Reducing on-chip
reads was not the dominant gain here. These forced-cold counters are separate
from rotating-event medians29.782->30.301us(+1.74%).

### Bounded fast-dequant / CTA / scale-load experiment

KeepDQ22's global/shared/output layout and discard the lane-exchange change
from this experiment. IDs26--34 in `dense_q8_tuning.inc` enumerate9 explicit
candidates; the DQ22 control makes10 timed configs, sixN/K families and both
precisions. Three independent factors:

- Exact fast FP32 construction `bitcast(0x4b000000 | raw) - 8388736.0f`
  before the unchanged FP32 multiply and final F16/BF16 RNE. No F16
  intermediate inBF16. All256codes x63488finiteF16scales x2outputtypes
  are raw-bit checked locally; native fast bodies must have no i32->f32.
- 128/256/512-thread CTAs with the sameN64/K128tile and no dummy blocks.
  Low-code and output tasks are repartitioned exactly once. Shorter block
  phases/more waves are hypotheses, not proof of lower latency. The2048CTA
  launch has1.78waves at16CTAs/CU;8/4CTAs/CU give3.56/7.11waves. Under a
  uniform linear model all three still have11.1% empty slot area. The ACU
  warning's50% is potential partial-wave duration, NOT50% wasted runtime.
- Four adjacent scale halfwords per8-byte load. Old scalar scale reads
  occupy64B/warp, half of a128B footprint; the vector mapping fills two
  128B segments perwarp. Low-plane16Bloads and each16Boutputstore already
  cover complete32/64/128B segments and are kept. This is checked per native
  instruction, not by combining two sparse instructions into a full row.
  Q8 N8192K2048 reported17MiB useful input versus18MiB LSU transaction
  bytes and32MiB output transactions. The128B footprint model predicts less
  scale request amplification;32/64B sectors are already full in the scalar
  mapping. Hardware transaction granularity/counters decide the benefit,
  not vector width alone. Compulsory HBM data does not change. Every actual
  input/output ring pointer's16/32/64/128B residue is recorded; this experiment
  requires128B alignment. Host tests also prove no duplicated global bytes
  across all tasks and both output stores, not only full sector counts.

Ordinary-convert and scalar-scale controls remain for causal comparison.
One verified idleZW810; no llama build/model input required:

```bash
PPU_SDK=/usr/local/PPU_SDK GPU=1 L2_BYTES=67108864 \
IDLE_CONFIRMED=1 Q8_EXCHANGE=0 Q8_FAST_TAIL=1 PROFILE_ACU=1 \
bash tools/run_dense_dequant_box.sh
```

120timing cells,30 alternating samples each. ThenACU on the two priority
N/K shapes, bothprecisions: baseline, fast-only, vector-scale-only,
ordinary-convert CTA-grain control and finalist (at most6profiles per typed
point,16--24total). One direct launch/profile; no cold loop or graph inACU.
Rawreports stayBox; compactreview includes counters/proofs. Source/config/
device checks prevent resuming older exchange results as fast-tail results.

`profile-5-24-s1` aggregate counters have0DRAM write bytes but32MiB L2/LLC
writes. Its bandwidth timeline cannot by itself distinguish CTA retirement
from producer/consumer phase changes; raw timelines were requested. Do not
call combined logical read+write bytes divided by that ACU time measured HBM
bandwidth. At handoff, candidate performance and model-level benefit were pending.

### Fast/grain/scale return: scoped gains, no default promotion

Return `review-1792741309128779355.tgz`, SHA256
`7922754a27eeffd21ebf3e130fc7ece010f8c7ff3ec8311705d45ad92d6db262`,
source `cd8b9c0`: all132 numeric cells,120 timing cells (3600 samples) and
18 ACU exports pass the evidence audit;1264 source hashes match. ACU clocks
are all1.7GHz; event before/after samples report1700/1800MHz core/memory,
not continuous clock telemetry. Device is ZW810/72CU, PCI0000:7E:00.0.
The compact upload includes rawCSV/proof hashes, not DSOs or full acurep.

Rotating input/output-ring event medians, microseconds; minimum measured
config in this bounded inventory, not a model/GEMM/full-call claim:

| N x K | F16 DQ22 -> best | Change | BF16 DQ22 -> best | Change |
|---|---:|---:|---:|---:|
|512 x2048|3.9707 ->3.7525 (28)|-5.50%|4.0045 ->3.7425 (28)|-6.54%|
|2048 x512|4.0054 ->3.7799 (28)|-5.63%|4.0006 ->3.7471 (28)|-6.34%|
|4096 x2048|15.5741 ->15.5741 (22)|0|15.5394 ->15.5394 (22)|0|
|8192 x2048|29.7989 ->29.6500 (30)|-0.50%|29.8456 ->29.5700 (29)|-0.92%|
|2048 x4096|15.5794 ->15.5794 (22)|0|15.5153 ->15.5153 (22)|0|
|1024 x5120|10.1471 ->10.0339 (28)|-1.12%|10.2004 ->9.9704 (27)|-2.25%|

The small-shape gain is mainly256-thread work partitioning; fast-only26
versus22 is marginal/mixed. The ordinary256-thread27 is already about5--6%
faster on the two smallest shapes. Preserve both27/28; their small difference
is not proof that fast conversion broadly wins. At N1024K5120, ordinary27
also lowers ACU cycles3.54%/1.22% and lifts active warps about50%->84%.

For N8192K2048/BF16, vector-scale31 truly improves global request packing:
LSU read transaction bytes18->17MiB, requests49152->36864; DRAM remains
about17MiB. But shared-store bank conflicts49152->114688, dynamic
instructions+2.25%, registers44->48, and ACU cycles29892->31493(+5.36%).
Event latency29.8456->30.3056us also loses. These coexisting changes do not
prove that one counter exactly cancels another. The source publishes each
thread's four neighboring scales to `A::offset(col+v,g)` at16-byte strides;
native still emits scalar shared stores. Next investigate paired scale
publication/consumer layout while retaining the proved global/output maps;
do not call a plain float4 cast on these noncontiguous addresses correct.

512-thread29 increases large-shape waves1.78->7.11 and active warps84.1%->
88.6%, but instructions+20.06%; BF16 ACU cycles+1.03% while rotating time
is0.92%lower. This does not establish a large tail-removal gain. Raw timelines
were not included, and forced-cold ACU has0DRAM output writes. Do not replace
the rotating ranking with this separate cache regime or normalize between
the two. Keep DQ22 as production default; preserve scoped256-thread candidates,
reject global promotion of vector-scale/512-thread/fast-only changes.

### Previous exchange handoff (immutable controls)

IDs24/25 preserve the DQ22/23 producer and 2N*K8 output map. Adjacent lanes
read complementary uint4s and exchange packed words with xor1, eliminating
the duplicated shared-code reads across two K8 phases. Arithmetic and
rounding are unchanged. No new shared buffer, output transpose or barrier.
Exchange offsets are applied BEFORE the shared swizzle, not to its physical
base; a coordinate negative and independent raw-bit oracle cover this seam.
Naive per-thread K16 stores were rejected locally: SDK2.1.1 lowers a32-byte
vector to two16-byte stores, whose per-instruction lane map has holes.

Both candidates are explicit only; default helpers remain the measured table.
One idle ZW810, no llama rebuild or model weights required:

```bash
PPU_SDK=/usr/local/PPU_SDK GPU=1 L2_BYTES=67108864 \
IDLE_CONFIRMED=1 Q8_EXCHANGE=1 PROFILE_ACU=1 \
bash tools/run_dense_dequant_box.sh
```

The script freezes sources and locally available actlize (no network submodule
fetch), requests1700MHz, runs six Q8 N/K families with four configs, two
precisions and30 alternating samples/config/precision. Setup/JIT and first-use
are excluded. The ordinary component timer uses rotating input/output rings
>=2.25L2; it is not a model or a complete GEMM-call timer.

After timing, ACU profiles N8192/K2048 and N1024/K5120, F16 and BF16 separately.
Each capture has one ordinary kernel launch, one weight copy, no CUDA graph
or cold-ring loop. It records the current DQ22 and fastest new candidate;
if K256 wins, it also captures DQ23 as the same-tile control. Both candidate
and controls must pass the oracle first. Kernel symbol/grid/block and one
launch are checked in the raw export. Forced-cold ACU cycles/counters are
separate from unprofiled rotating-event time.

Upload `results/review-*.tgz`; it contains timings, raw counters, identities,
numerical proofs and logs. `.acurep` files and the DSO stay in the printed
Box directory. A resume preserves old successful timing points and names
new ACU reports uniquely. No automatic candidate promotion follows a PASS.

## One-command remaining-helper closeout

`tools/run_dense_helpers_box.sh` covers the 33 returned model families at
M128/256/512/1024/2048, plus two labelled Q2 format-extension families. These
are not inferred DeepSeek TP4 shards. The inventory and prior Full-F16 helper
recipes (not historical times) live in `dev/dense_helpers/cohort.json`.

Explicit candidates22/23 now cover Q2/Q3/Q8 paired-plane N64 K128/K256.
Q4/Q5/Q6 retain0/8/20/21, including each old full-call winner. Both F16 and
BF16 are checked/timed independently. Q2 preserves its d/dmin and16 groups;
Q3 retains its odd14-byte-superblock header; no artifact or rounding changes.

The runner builds one small component probe, partitions whole families onto
the requested GPUs, and measures each dequant N/K/storage once per card and
each IO M/width/direction/storage once per card. Single-kernel rotating graphs
are allowed here, never around cuBLAS. Each scalar/direct baseline remains.
Two combinations shortlisted from isolated components proceed automatically
to complete ordinary calls under Asys, along with prior and production-helper
controls and native F16/FP32. Forward and reverse pools plus bracketing native
captures expose order drift. First complete warmup/primer rings are excluded.
No fresh TC sweep; no automatic production selector, dtype or bucket changes.

```bash
PPU_SDK=/usr/local/PPU_SDK \
LLAMA_CI_DIR=/sim/eec/shared/junfu.qx/llama.cpp.ncp \
LLAMA_CI_BUILD_DIR=/sim/eec/shared/junfu.qx/llama.cpp.ncp/build-ci-32F-cublas \
GPUS=1,2,3,4,5,6,7 L2_BYTES=67108864 IDLE_CONFIRMED=1 \
bash tools/run_dense_helpers_box.sh
```

Use only available idle GPUs. A resume supplies `RESUME_RUN` printed by the
command and the same GPU list/SDK/caller build. Successful component cells
and successful Asys batches are verified and reused. A bad component blocks
only its card's finalist stage; other cards continue, all errors stay visible.
The top-level `review-*.tgz` contains numeric proofs, stage/sample times,
source/device receipts and short error logs. Weights, DSOs, SQLite and GUI
reports stay on Box. The tables distinguish isolated sums from full-call
Asys timings; neither is a whole-model TPOT result.
