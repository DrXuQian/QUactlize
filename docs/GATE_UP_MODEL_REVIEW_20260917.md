# Paired-N4 model result review

Reviewed return: `kpack-q4-model.ehl4Bn.results.tgz`, SHA256
`827a5071d13751e8fa45ddce4c23c8376ef84d46337462bec1db9c3f665c430b`.
The 998 archive entries are regular files/directories under `results/`.
Source `51abe991fe64ced1fbed937d72690b7a2b20f87d`, caller
`9c6e7d756875c24d0ca3f6a0433bfe81077c804a`, and runtime manifest
`1ca7cdd7810317d3e032a2f9ba3518ecd857a1931a1d1f049ac8dd43367f3279`
match the published pin. The caller worktree receipt is clean. Device ordinal0,
PCI `0000:08:00.0`; one visible PPU. No binary or selector changed in this review.

## Declared model performance

Qwen3.5-35B-A3B Q4_K_M, one request, PP2048/TG128, graphs enabled. ABBA order;
each of four processes discards its first complete pass. The two subsequent
samples per process give four measurements per arm. Recomputing from raw
`t_pp`, `t_tg` and token denominators reproduces every summary median.
These are unprofiled random-token `llama-batched-bench` results, not task accuracy.

| Metric | Native reference | Paired K-pack | Latency change |
|---|---:|---:|---:|
| Prefill, us/token | 140.470459 | 106.157715 | -24.4270% |
| Decode, ms/token | 7.677785 | 6.989402 | -8.9659% |

K-pack decode is 143.07 tokens/s. Its four samples span6.984461..7.002172ms;
reference spans7.609555..7.710070ms. First K-pack passes were16.28..17.10ms/token
and are explicitly excluded, not averaged into resident performance.
The report does not independently prove that no other GPU workload was active.

Previous `KUCg49` return measured7.329109ms/token for K-pack and7.681074ms for
reference. The new K-pack result is0.339707ms/token (4.6350%) lower. Prefill
remains106.19 vs106.16us/token. This is a historical cross-run comparison,
not a same-binary paired-fusion-on/off experiment.

## Device execution, not selection-only evidence

The two Asys captures use identical input-token and request hashes, PP2048
and16 generated tokens. Both capture the second request in the same process;
first-use packing/JIT is excluded. The upload includes exported kernel times,
proofs and logs, but not the complete `.asysrep` or an ACU report.

80 exact paired plans cover all40 layers: Q8 shared/F16 and Q4 routed/BF16,
each M1/SIMT/S1/W8. The actual fusion symbols each execute600 times, consistent
with40 layers x15 decode forwards after the prompt produces the first token.

| Device operation | Calls | Mean us/call |
|---|---:|---:|
| Q4 routed paired gate/up + SwiGLU | 600 | 18.1654 |
| Q8 shared paired gate/up + SwiGLU | 600 | 8.5977 |
| Q5 routed down | 600 | 13.3203 |
| Fused router/prepare | 600 | 7.7617 |
| Weighted finish | 600 | 2.4688 |

Against the previous same-scope exported trace:

- Independent `moe_chain_swiglu_compute` calls:600 ->0. The old Q4 projection
  averaged18.1720us plus4.0832us for activation; the new projection includes
  activation in18.1654us. Down and prepare remain essentially unchanged.
- Shared Q8 projection calls removed:1200, replaced by600 paired calls.
  Generic SiLU call count falls1270 ->670. Its aggregate also contains other
  shapes/prefill, so do not use its global mean as the removed decode SiLU time.
- Total captured kernels:27839 ->26039, a reduction of1800 calls, consistent
  with120 fewer launches per decode forward across40 layers.
- Standalone top-k remains40 calls, consistent with the prefill part; this is
  not a claim of duplicate decode top-k. Generic gather/scatter in the full
  capture also includes out-of-scope prefill paths.

Whole-capture summed kernel time309.516ms includes prefill and decode. It is
not TPOT, wall-clock request latency, or a substitute for the unprofiled table.

## Numerical scope and remaining admission boundary

New integration gates pass:4 canonical repacks,144 unique mapped-output cells,
288 changed-input replays and148 detected negatives. Maximum relative error
is1.066e-7. All16 routed-chain cases pass (tokens1..8 x router absent/present),
with48 changed-input replays. Maximum chain error is0.002811 below the existing
0.005 threshold. Existing router alias, production Q8, BF16 capability746/746,
five-format metadata and selected-Q4 overflow/rounding controls also pass.

The model likelihood run has finite outputs and matching coverage/token hashes
within each three-arm comparison. It does not establish bitwise equivalence:

| Token batch | Scored tokens | PPL ratio | Mean KLD | Max KLD | Same top token |
|---|---:|---:|---:|---:|---:|
| 1 | 254 | 1.001358 | 0.010135 | 0.847899 | 98.425% |
| 2048 | 2046 | 0.997325 | 0.003416 | 0.299845 | 98.583% |

Reference self-checks have100% top-token agreement and near-zero KLD. The
native quantized reference and W*A16 K-pack path have different arithmetic;
these differences cannot be assigned to this fusion from the uploaded data.
The prior `KUCg49` model run has no model-likelihood comparison. Therefore
retain `accuracy_admission=PENDING_REVIEW`: finite metrics and runner rc0 are
not a task-level accuracy threshold. No new GSM8K evaluation was run.

## Next work

1. If final accuracy attribution is required, compare paired=0/1 using this
   same runtime/caller/corpus. This needs only a bounded numerical run, not
   another full kernel sweep or bundle rebuild. Do not clip or relax the oracle.
2. Router/prepare still costs7.76us per decode layer. Fusion removed an
   activation launch but did not optimize this helper or the Q5 down reader.
3. Keep the optional paired route at its confirmed shape/type/M scope. The
   auxiliary weight copies still cost about11.33GiB for this40-layer model;
   prefill retains canonical weights. No default/main promotion in this review.

Structured evidence: [measurement receipt](measurements/gate_up_model_20260917.json).
Full box traces remain at
`/workspace/kpack-q4-model.ehl4Bn/results/trace/qwen35-35b-q4km/{reference,native}/proof.asysrep`.
