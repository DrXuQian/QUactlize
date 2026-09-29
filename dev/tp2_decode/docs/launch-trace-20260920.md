# Repaired TP2: returned steady-decode timeline

The newly uploaded reports belong to `kpack-tp2.yj9GRn`. Their exports exactly
reproduce its per-symbol kernel counts and duration sums, not just total file
sizes or timestamps. This closes the missing-timeline item in
`launch-result-20260920.md`; it does not replace the unprofiled ABBA authority.

| Report | SHA256 |
|---|---|
| `native_proof.asysrep` | `ca2cda6f85831c3fe4799b64b905f11b48432df2df4c6d0f594201633cd41480` |
| `reference_proof.asysrep` | `993d4df4a33107b88e19b329f67dd18679d29b5b100aa41c5a14550acbe7c9be` |

Local SDK export output is under
`/tmp/kpack-launch-trace-20260920.sqJk2Z/{native,reference}.sqlite`.
The uploaded names replace the old root-level reports; the old export remains
separate at `/tmp/kpack-tp2-trace-review-20260920.NE2QTb/`.
Nothing was captured or executed on a GPU during this analysis.

## Scope and validation

The native export contains 67,490 kernels, reference 76,402. Every symbol's
count and total duration matches the corresponding `kernel-times.json`.
Physical devices are 6 and 7, PCI `00000001:AA:00.0` and
`00000001:CE:00.0`, matching the requested assignment.

Use the last eight consecutive output-head-to-head intervals on each device,
from the preceding head's end through the next head's end. Each report has
16 head calls per rank. Native head is Q6 TC, grid `(1,1940,1)`; reference
head is raw Q6 `mul_mat_vec_q`, grid `(124160,1,1)`. All 16 selected intervals
have identical per-arm launch counts, no overlapping kernel intervals on
the same rank, and exclude first graph-capture/replay work.

This trace is PP2048/TG16, second request. The longer warmed PP2048/TG128
benchmark remains **reference 12.618078 versus K-pack 11.232953 ms/token**.
Do not replace it with the profiled intervals below.

## What really runs in steady decode

Per rank/token, native launches 1,851 kernels, down from the old 1,898;
reference launches 2,080. Native has:

- 47 snapshot prepares and one original no-snapshot prepare;
- 48 standalone chain SwiGLU and 48 weighted-finish calls;
- 48 measured Q4 gate/up, 48 measured Q5 down and 48 paired shared-Q8 calls;
- **zero** independent `topk_moe_cuda`, gather or scatter calls;
- 96 PCCL all-reduces, the same count as reference.

The residual 192 topk calls in whole-report statistics are outside these
steady decode windows. They must not be described as duplicated decode work.
The new ordinary Q8 public entry really reaches `measured_decode_kernel`.

## GEMV time and effective weight bandwidth

Times are pooled per-call medians over both ranks' final eight tokens. S8 rows
include the immediately following same-stream reducer. Pairing is checked;
producer-only timing is not substituted. Gate/up projection excludes SwiGLU,
whereas the shared-Q8 paired row includes it.

MBU is useful packed weight bytes divided by full selected-call time and an
assumed 2700 GB/s per-device peak. It is not ACU DRAM utilization. Indexed
rows charge eight active expert slices, not the 256-expert allocation.

| Operation | Local N x K | Calls/rank/token | Time us | Modeled MBU |
|---|---|---:|---:|---:|
| Q8 QKV, S8 with reducer | 6144 x 3072 | 36 | 18.480 | 40.192% |
| Q8 attention gate, S8 with reducer | 4096 x 3072 | 36 | 12.800 | 38.684% |
| Q8 output, S8 with reducer | 3072 x 4096 | 48 | 12.960 | 38.207% |
| Q8 attention Q, S1 | 8192 x 3072 | 12 | 21.200 | 46.713% |
| Q8 small K/V, S1 | 256 x 3072 | 24 | 6.100 | 5.073% |
| Shared Q8 gate/up with SwiGLU | 2 x 1024 x 3072 | 48 | 10.960 | 22.589% |
| Shared Q8 down, S1 | 3072 x 1024 | 48 | 8.040 | 15.397% |
| Routed Q4 gate/up projection | 1024 x 3072 | 48 | 18.880 | 27.769% |
| Routed Q5 down, S1 | 3072 x 512 | 48 | 16.520 | 19.395% |
| Q6 output head, TC S1 | 124160 x 3072 | 1 | 332.983 | 34.801% |

The three separate cold component winners are still those in the preceding
review. Their rotating-event times and this model's profiled times are not
interchangeable. In particular the new 12.835556 us Q5 P4-fixed candidate
has not yet replaced the model's P8 implementation.

## Why fewer helper launches do not ensure a faster chain

Match each Q4 projection to its preceding prepare/topk, following activation,
Q5 down and finish before the next all-reduce. In the old trace, layers0..46
use topk + ordinary gated activation + weighted multiply + eight-way sum.
The repaired trace substitutes prepare + chain SwiGLU + weighted finish.
The gate/up and down kernels themselves are almost unchanged in duration.

| Layers0..46, per chain | Old unfused helpers us | Repaired helpers us |
|---|---:|---:|
| Router / prepare | 6.760 | 7.440 |
| Gated activation | 1.880 | 5.240 |
| Weighted finish | 4.800 (two launches) | 2.560 (one launch) |
| Median of actual per-chain helper sums | 13.4405 | 15.2800 |
| Median of projection + helpers + down | 48.880 | 50.600 |
| Median first-helper-start to finish-end | 49.920 | 51.440 |

Individual medians need not sum to the median of the sum. The average helper
increase is 1.6652 us per affected layer, about 0.0783 ms/rank/token over
47 layers. This comparison is descriptive across two captures on different
device pairs, not a controlled repair-only performance admission. It does
identify the costly current stage: chain SwiGLU contributes about
0.249 ms/rank/token across all48 layers.

The existing BF16 chain activation has generic SIMT/TC reads, optional Split-K
reduction, indirect row maps, validity handling and BF16 projection rounding.
The old ordinary activation is not proof that all that work can simply be
deleted. Preserve arithmetic, row identity, invalid-route and lifetime
contracts when optimizing or fusing it.

## Communication and gaps remain separate costs

These are **means** over each rank's eight windows, so the three components
sum to the window duration. No same-rank kernel overlap was found.

| Arm / physical device | Non-PCCL kernels ms | PCCL kernels ms | No-kernel gaps ms | Window ms |
|---|---:|---:|---:|---:|
| K-pack / 6 | 9.007941 | 3.077556 | 2.095187 | 14.180684 |
| K-pack / 7 | 9.006814 | 1.299940 | 3.876297 | 14.183051 |
| Reference / 6 | 10.373829 | 2.972594 | 1.551166 | 14.897589 |
| Reference / 7 | 10.318736 | 1.280800 | 3.298118 | 14.897654 |

The two ranks' non-PCCL work is nearly equal. One rank spends more time inside
PCCL, the other outside kernels; these are parts of one coordinated execution,
not two independent costs to add. PCCL duration includes synchronization
waiting and is not pure fabric transfer time. No-kernel gaps do not by
themselves prove host-launch overhead. Large head-to-next-token gaps also
occur in reference. Runtime-event dependency analysis or a controlled test
is needed before attributing the difference to a particular caller operation.

The current trace supports substantial savings versus reference in GPU work,
but not an additional large GEMV gain over the previous selected K-pack
implementation. Some Q8 shapes improved in this capture and others did not.
Together with the helper cost and TP2 waiting/gaps, this refutes the assumption
that correcting launch identity and removing one helper launch per layer must
automatically yield a large further TPOT reduction.

## Gate/up + SwiGLU fusion: existing mechanism, missing TP2 admission

Yes: perform SwiGLU after the final gate/up sums, before writing the next
activation. This does not require a new architecture layer.

The existing pieces are:

1. `quactlize/fusion/paired_n4.hpp`: paired G4/U4 weight order places matching
   values within the same tile. The current merged-N projection alone does
   not guarantee gate/up values land in the same output owner.
2. `quactlize/fusion/store.cuh::SimtFinish`: for S1, the final owner reduces
   both gate and up, calls `activate`, and writes only the activated half-width
   output. `activate` preserves requested BF16/F16 projection rounding.
3. The same file's `reduce_gate_up`: for S>1, nonlinear activation happens
   only after summing all partitions. Activating each partial is incorrect.
4. The caller's `prepare_moe` already queries paired selection, prepares the
   paired weights, and binds the paired stage into the MoE handle. It need not
   add a second per-token weight conversion or another graph-fusion framework.

Current TP2 Q4 scope is per-projection N512/K3072, E256/top8, BF16, M1.
`policies/kpack_decode_winners_v1.json` contains its paired candidate, but
`automatic` is false. `tools/generate_decode_selection.py` retains this
explicit exclusion because the old candidate comparison did not establish
a matched full-chain win over the selected separate projection + activation.
The historical paired candidate takes22.786060 us under rotating timing;
comparing it directly with this trace's18.88 +5.24 us would mix conditions.

Next bounded comparison should keep the same router/prepare, down, finish,
BF16 boundaries, active experts and weight-ring regime in both arms:

- Current: prepare -> measured merged gate/up -> chain SwiGLU -> down -> finish.
- Candidate: prepare -> paired gate/up + SwiGLU -> down -> finish.

Retain the measured merged projection as baseline. Validate changing IDs,
BF16-range values, token1..8 correctness, TP2 K/N shards, alias/cache replay
and invalid routes; rank only the measured M scope. Include the paired
implementation's possible register/occupancy cost in the timing. Removing
the standalone SwiGLU launch does not save its entire duration if the paired
producer becomes slower. A passing complete-call winner can then enable
that policy entry and remove the redundant activation stage in the already
available binding path. Canonical and paired layouts must remain explicitly
identified; do not reinterpret one as the other.

This review changes no runtime/caller code, selector or artifact pin. It closes
the report-availability question and identifies the next causal experiment.
