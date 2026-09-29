# Selected TP2 decode: returned Asys audit

The private trace retry succeeds. It also exposes an execution-entry gap that
the earlier selector audit did not cover: ordinary F16 Q8 calls use the v1
generic launcher, not the admitted implementation available through v2.
MoE whole-chain fusion covers only the final layer. Neither finding changes
the already recorded unprofiled ABBA result; neither is fixed by this report.

## Evidence and scope

| Input | SHA256 |
|---|---|
| `kpack-model-asys.xj6rrq9j.results.tgz` | `c5f9dc49057501df51e440f766338a2d17b10996fb88d9c6a7009753d03d893b` |
| `native_proof.asysrep` | `d8082260e50b5bf92200d39209a14a1595f5ed313d31fe1022d05c0b9aa55eaa` |
| `reference_proof.asysrep` | `e5b596e2c793b8e136ef927673fbe60f18673145575b21ef85a35c623a02d4ec` |

The retry records the same runtime manifest
`5bda5fabf565c54494a358479a3ce9923bcde8d204338aede0742517491a4aab`
and caller build as `kpack-tp2.8VYueY`. Source trace handoff is `69af891`;
caller remains `1f7da3b`. Both arms capture the second request in their own
process, PP2048/TG16, request batch 1, devices 0 and 1. Input/request/response
hashes agree. This short identical response is not renewed task accuracy.
Private capture succeeds without restarting the shared profiler services.

Local read-only Asys export reproduces the receipt kernel counts exactly:
native 68,806, reference 76,402. These counts include prefill and both ranks.
The reports contain timings and launch geometry, not ACU DRAM counters.

For steady decode, use the last eight consecutive output-head-to-output-head
intervals on each rank. Each report has 16 output-head launches per rank.
Native head: Q6 TC, `MainloopPPUAiuMixedInput2Plane<2,...>`, tile16x64x128,
grid `(1,1940,1)`. Reference head: `mul_mat_vec_q<(ggml_type)14,1,...>`,
grid `(124160,1,1)`. Select kernels after the preceding head's end and through
the next head's end. All eight intervals have identical launch counts and
exclude first graph capture/replay. Do not substitute these profiled intervals
for the unprofiled PP2048/TG128 ABBA TPOT.

## Ordinary Q8: selection is correct, implementation dispatch is not

The caller chooses the promoted geometry, including C8/W4/P4/S8 for QKV,
attention gate and output. But actual Q8 producers are
`q8_vector::kernel<1,0,1,8,4,4,false>` or its hoisted/generic variants.
There are no Q8 `measured_decode_kernel<8,...>` launches. Q4 and Q5 indexed
MoE do execute their measured wrappers; shared-expert Q8 uses its paired
tile16 implementation.

The source path explains this discrepancy:

1. `ggml-cuda/quactlize-execution.cu`, `run()`: a reused SIMT plan with F16
   compute calls `simt_run` (v1). Only BF16 calls `simt_run_compute` (v2).
2. `quactlize/execution/simt.cpp`: v1 runs `qkg_simt_launch_Q`; v2 runs
   `qkg_simt_launch_v2_Q` after their respective validation.
3. `quactlize/execution/simt_codegen.py`: only the generated v2 launcher
   consults `measured_decode`. The v1 launcher goes straight to the generic
   reader with dynamic split. The same geometry is not the same implementation.
4. `dev/gemv_simt/production.py` binds `bind_simt_compute` even for F16.
   Consequently the prior selected-entry numerical test exercised v2, not the
   actual F16 caller entry. The query audit did not prove launch-body identity.

Fix priority: share the admitted implementation resolution between v1 and v2,
while preserving v1 validation and F16 compute semantics. Add actual v1/v2
dispatch parity coverage and assert implementation identity in the trace gate.
Do not solve this by converting every dense operation to BF16 or by adding
another shape table. The new body's model latency remains unmeasured until
the corrected caller path is exercised.

## Decode kernel time and modeled bandwidth

M=1, local per-rank N/K, pooled last-eight-token samples from both ranks.
For Split-K, time is the median of each producer-plus-following-reducer sum;
it excludes inter-launch gaps. All other rows are producer time. Q4 projection
does not include SwiGLU; shared Q8 gate/up does. Thus scopes differ across
those rows. This is useful-weight MBU using 2700 GB/s per device as the assumed
peak, not measured DRAM utilization or an idle-device/cold-weight benchmark.

| Operation | Local N x K | Active experts | Calls/rank/token | Time us | Modeled MBU |
|---|---:|---:|---:|---:|---:|
| Q8 QKV, S8 including reducer | 6144 x 3072 | 1 | 36 | 18.560 | 40.02% |
| Q8 attention gate, S8 including reducer | 4096 x 3072 | 1 | 36 | 12.200 | 40.59% |
| Q8 output, S8 including reducer | 3072 x 4096 | 1 | 48 | 12.360 | 40.06% |
| Q8 attention Q, S1 | 8192 x 3072 | 1 | 12 | 19.200 | 51.58% |
| Q8 shared gate/up, including SwiGLU | 2 x 1024 x 3072 | 1 | 48 | 10.960 | 22.59% |
| Q8 shared down, S1 | 3072 x 1024 | 1 | 48 | 8.240 | 15.02% |
| Q8 attention K/V, S1 | 256 x 3072 | 1 | 24 | 6.660 | 4.65% |
| Q4 merged MoE gate/up | 1024 x 3072 | 8 | 48 | 18.880 | 27.77% |
| Q5 MoE down | 3072 x 512 | 8 | 48 | 16.640 | 19.25% |
| Q6 output head, TC S1 | 124160 x 3072 | 1 | 1 | 332.620 | 34.84% |

Useful bytes are `N*K*active_experts*block_bytes/block_elements`:
Q8 34/32, Q4 144/256, Q5 176/256, Q6 210/256. Both shared weight matrices
are counted. These are canonical compact-weight byte counts, not measured
reload traffic; activation, output and Split-K partial traffic are excluded.
The Q8 K/V launch has only 16 CTAs of 256 threads. A low modeled MBU on this
small workload does not establish a memory-coalescing root cause.

## MoE coverage and other overhead

Per rank and steady decode token:

- 48 Q4 gate/up and 48 Q5 down producers execute the promoted implementations.
- Shared-expert Q8 gate/up is paired in all 48 layers.
- Only one `prepare_detail::once` (7.120 us), one chain SwiGLU (5.280 us)
  and one weighted finish (2.641 us) execute. All four recorded chain-plan
  lines refer to `blk.47`; none refers to layers 0..46.
- 47 independent `topk_moe_cuda` calls remain, median 6.760 us each.
- No gather, scatter or grouped-device-metadata kernel occurs in these decode
  intervals. Such names in the complete report belong outside this scope.

Therefore whole-chain fusion is not generalized across this model's layers.
Its rejection reason is still open: the caller has graph/consumer, memory
alias, concurrent-event and endpoint-admission guards. Instrument their
specific rejection paths before changing them. Do not remove a guard or assume
that eliminating every topk launch saves its full duration: the fused prepare
also performs work and its net chain cost must be compared.

Kernel calls fall from 2080 to 1898 per rank/token. Excluding PCCL, summed
kernel durations fall from about 10.26 to 8.86 ms per rank/token. PCCL has
96 calls per rank/token in both arms. Its native summed duration is about
3.037 ms on rank0 versus 1.424 ms on rank1; reference is about 2.885 versus
1.344 ms. This rank asymmetry includes collective waiting, not just transport
work. Do not add both ranks' durations to predict TPOT or treat them as GEMV
bandwidth differences.

## Next bounded work

1. Correct ordinary F16 SIMT v1 dispatch; verify v1 and v2 select and launch
   the same admitted body for the same F16 request, including table misses.
2. Explain layer0..46 chain rejection, preserving alias/lifetime guards.
   Decide on fusion using complete comparable chain timings.
3. Reuse the valid performance/numerical results and warmed caches. After any
   changed runtime/caller, run targeted entry tests plus model ABBA and this
   private trace. No new unrestricted configuration sweep is justified.
4. Then revisit the still-low Q8 small projections and Q4/Q5 MoE efficiency
   with ACU on the actual executed bodies. This upload alone does not prove
   how much either correction will lower TPOT.

Unprofiled performance authority remains the preceding report: reference
12.700852 versus K-pack 11.239113 ms/token, -11.5090%; prefill -20.2191%.
This audit adds trace evidence and narrows the next changes; it does not modify
production code, runtime artifacts or the precision admission decision.
