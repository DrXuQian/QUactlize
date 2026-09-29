# Real-model GEMV reader experiment review

Seven of eight points have a faster candidate in all six confirmation rounds.
Keep the Q6 output-head TC incumbent. This is numerical and isolated M1
performance evidence, not an updated production selector or model TPOT result.

## Authority and validation

Return: `model-gemv.mZfskR.results.tgz`, SHA256
`0362d61e0a9c2e32786298ef59f48e79a0af2d4b84c9a7f542f66444a9ea38f9`.
Source: `2d8e5030afbbe757ab6759c0d2a37440310f1a32`.
Artifact: `9df3eb212289d05c8dbee6368d8b63621f1c8305`, manifest SHA256
`f54f6f627cc62f60f81664372d31b7e29c3bf9dedd406f88db331488d64301db`.
The immutable execution/fusion incumbents are those recorded in
[the paired model review](GATE_UP_MODEL_REVIEW_20260917.md), not rebuilt stand-ins.
The three TC controls retain their exact selected parent, compute type and split.

The archive has 107 safe regular-file/directory entries. All eight point
identities, source hashes, runtime hashes, candidate selections and numerical
denominators match. Every point uses ordinal0, PCI `0000:08:00.0`. Device
idleness is not independently proven by the upload.

- 68 candidate bodies plus eight incumbents: 76 provider/point checks.
- 510 numerical control rows: M1/M2/M8, changed inputs/IDs, and routed BF16
  large-input controls containing 243383.484. Maximum error is 0.001298824;
  the unchanged threshold is 0.005. Guards, zero-input and invalid-route/status
  negatives, and five same-geometry M1 clone comparisons pass in the pinned
  runner. This is not a new model-level accuracy test.
- All 24 confirmed arms have six rounds of 15 finite samples: 2160 samples.
  Medians, round deltas, modeled MBU, finalists and winner configs were
  independently recomputed from the per-point JSON.
- All 16 actual ACU reports were re-imported locally. The raw counters match
  the returned CSV, including all 20 producer/reducer launches and geometry.
  ACU replay/cache-control timing is not substituted for the event timings.

Performance uses M1, F32 storage endpoints, FP32 accumulation, and a rotating
active-weight working set at least 2.25 times the declared 64 MiB L2. Only the
eight used experts count toward routed traffic. Graph upload/first use is
excluded; Split-K reduction and paired SwiGLU are included in complete calls.
Dense/shared compute is F16; routed compute is BF16. No offline bytes changed.

## Recomputed complete-call results

N is physical weight N for paired points: logical output N is 512. Other
points use their normal output N. MBU is distinct packed weight bytes divided
by complete-call time and 2700 GB/s, not measured DRAM throughput.

| Point | N x K; active experts | Incumbent us | Candidate us | Delta | Candidate weight MBU | Decision |
|---|---|---:|---:|---:|---:|---|
| Q4 paired routed gate/up | 1024 x 2048; 8 | 16.860001 | 13.610625 | -19.273% | 25.68% | Exact M1 candidate |
| Q5 routed down | 2048 x 512; 8 | 11.072593 | 9.420740 | -14.918% | 22.67% | Exact M1 candidate |
| Q8 paired shared gate/up | 1024 x 2048; 1 | 7.271765 | 5.319412 | -26.848% | 15.51% | Exact M1 candidate |
| Q8 shared down | 2048 x 512; 1 | 4.087059 | 3.752353 | -8.189% | 11.00% | Exact M1 candidate |
| Q8 SSM output | 2048 x 4096; 1 | 9.313529 | 9.051765 | -2.811% | 36.47% | Exact M1 candidate, retain S8 |
| Q8 QKV | 8192 x 2048; 1 | 15.426111 | 14.591111 | -5.413% | 45.25% | Exact M1 SIMT candidate over TC S8 |
| Q8 attention gate | 4096 x 2048; 1 | 10.740000 | 9.343529 | -13.003% | 35.33% | Exact M1 SIMT candidate over TC S8 |
| Q6 output head | 248320 x 2048; 1 | 417.053118 | 421.840623 | +1.148% | 36.63% | Retain TC |

All seven improvements have the same sign in every paired round. The Q6
candidate's per-round deltas are +2.54, -5.42, -4.51, +2.21, +6.38, +8.44%:
there is visible drift, not a stable win. Its one ACU invocation is not a
reason to override the repeated complete-call result.

Winning configurations, using C=columns, P=values/thread, W=warps and S=split:

| Point | Arm | Config and changes |
|---|---:|---|
| Q4 paired routed | 3 | V3 C4/P8/W8/S1; H32; fixed N/K |
| Q5 routed down | 1 | V3 C4/P8/W2/S1; existing H32/index/fold; fixed N/K |
| Q8 paired shared | 10 | V5 C4/P4/W8/S1; hoisted words; TileN16 paired finish |
| Q8 shared down | 6 | V5 C4/P4/W2/S1; hoisted words |
| Q8 SSM output | 1 | V5 C8/P4/W4/S8; fixed N/K; ordered reducer |
| Q8 QKV | 6 | V5 C8/P4/W4/S1; hoisted words; fixed N/K |
| Q8 attention gate | 8 | V5 C4/P4/W8/S1 |
| Q6 experimental challenger only | 5 | V3 C4/P8/W4/S1; direct d/int8-scale metadata |

## What ACU supports

The numbers below are producer counters from separately profiled invocations.
Stall ratios are per active issue, not fractions of elapsed time; they can
exceed one. L1/L2 bytes and KVD/L2 transactions include the SDK's AIU/ldgsts
accounting. Do not add those loads a second time or call them DRAM bytes.

| Point | Emitted dynamic instructions | DRAM read MB | L1/L2 traffic MB | Achieved warp occupancy | Mechanism supported by counters |
|---|---:|---:|---:|---:|---|
| Q4 paired | 6.381M -> 4.607M | 9.487 -> 9.482 | 18.535 -> 18.588 | 43.30 -> 43.11% | Less address/metadata work, not fewer weight bytes |
| Q5 down | 2.984M -> 2.408M | 5.809 -> 5.804 | 12.152 -> 12.152 | 21.97 -> 21.90% | Fixed-shape address simplification |
| Q8 paired shared | 0.380M -> 0.467M | 2.254 -> 2.249 | 4.734 -> 9.474 | 5.34 -> 10.81% | 32 -> 64 CTAs; more parallelism despite more internal traffic |
| Q8 shared down | 0.239M -> 0.229M | 1.126 -> 1.133 | 2.384 -> 4.639 | 5.55 -> 5.50% | 64x128 -> 128x64 grid/block; shorter per-CTA work |
| Q6 output | 188.227M -> 187.536M | 418.961 -> 417.244 | 1861.726 -> 823.256 | 48.01 -> 48.42% | Much less internal traffic alone does not ensure a full-call win |

Q4's FP32 ALU work, vector-load instruction count and 94 vector registers
remain unchanged. Its scalar-ALU count falls by about 80%; total instructions
fall 27.8%. Q5 total instructions fall 19.3%, while vector registers rise
114 -> 216. A simple "fewer registers is faster" rule would choose wrongly.
The lower-register Q5 P4/fixed finalist is only 0.4% slower; keep it as a
future model-concurrency control, not an unmeasured replacement for the winner.

The narrower shared Q8 tile improves latency even though L1/L2 transactions
approximately double. Source B accesses still cover 32 contiguous bytes per
K-worker group versus the original 64; format and extraction family are
unchanged. Plain P8 hoisting alone was slower in screen. The winning change
combines a shorter register window with enough CTAs to cover more CUs.

For Q8 QKV the candidate producer is slower in ACU (15.736 vs 14.520 us), but
the TC call also requires a 1.739 us reducer. Compare complete calls: the
rotating result improves 5.4%. Attention gate similarly removes its reducer.
SSM output must not follow that blanket S1 rule: the confirmed S1 finalist is
11.635 us, 24.9% slower than the S8 incumbent. Its fixed-shape S8 candidate
reduces instructions without changing traffic or the ordered reducer.

No point reaches this experiment's declared MBU target. The manifest used
40% below 16 MiB and 60% above; this experiment label does not change the
user's small/large-shape objective. Raw-GGUF/Xplane parity was not rerun.

## Integration boundary and next work

1. Transfer the seven exact M1 winners to production/JIT selection while
   retaining N/K, compute/storage type, paired layout and routing guards.
   Preserve BF16 routed math, F16 dense/shared math and FP32 accumulation.
   No extrapolation to M2..8 from numerical-only controls; no global S1 rule.
2. Keep Q6 TC. Preserve direct-metadata SIMT as an experimental contender;
   investigate its dependency/issue cost with the already returned ACU data.
3. Rebuild only affected small libraries/JIT glue, run the integration gates,
   then do one paired whole-model timing/Asys run with first use excluded.
   Source-level specialization may compile differently after integration.
4. No router/prepare or caller changes were made in this review. Shared Q8
   still has a small-grid limit, and these gains do not close the MBU goal.

Using the previous trace's per-token frequencies (40 each except attention
gate30 and output1), the seven isolated differences sum to about 0.373 ms.
This is a component projection, not measured TPOT or a guaranteed reduction:
model cache reuse, scheduling and compilation context can change the result.

Structured evidence: [measurement receipt](measurements/model_gemv_20260917.json).
Reproduce the local audit with `tools/review_model_gemv.py --results <extracted/results>
--output <new-review-dir> --acu <host-compatible-acu>`. The host import may
need a private SDK-compatible loader on older build hosts; no device is used.
The original box ACU files remain under
`/workspace/model-gemv.mZfskR/results/sweep/*.acurep`.
