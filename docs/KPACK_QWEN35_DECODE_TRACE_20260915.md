# Qwen3.5-35B-A3B Q4_K_M decode trace review

## Conclusion

The Q4 SIMT selector is connected, but the model is not using a measured
decode winner for every weight. The dominant uncompleted path is Q8_0 dense:
250 logical projections per decode step use the initial TC/Split-K policy.
Its SIMT policy file is unset. Native gate/up/GLU fusion is also lost for
K-pack Q8 shared experts. In the routed MoE, Q4 gate/up gets faster, but most
of that benefit disappears when comparing equal work, including SwiGLU.
Q5 down's producer is essentially unchanged relative to the reference.

This trace review is not performance admission for a new implementation.
The follow-up below is separate from the captured baseline; GEMM selection
has not changed.

### Follow-up implementation, device admission pending

After this review, the M1 helper was changed to skip unused SIMT-side TC
descriptors. An additive, immutable MoE finish binding now combines output
completion/placement, routing-weight multiply and the ordered eight-slot sum.
The graph matcher only admits a closed, exact tail and preserves existing
TC FP16 completion and SIMT FP32 output semantics. Unsupported graphs retain
the ordinary execution path. Local host tests and hgcc compilation pass;
the small `tools/run_kpack_moe_finish_box.sh` gate is the next device check.
No model speedup is claimed. The Q8 shared-expert fusion/policy regression,
TC metadata optimization and separate routed SwiGLU remain open.

## Full timeline update

Both full reports are now available and exported locally. Their kernel counts
and summed nanoseconds exactly match the original archive:

- Native report SHA256: `f1b180bd8801f88506f8ddc696bea1b76bad016e01624faddf9ebf48f3499ac0`.
- Reference report SHA256: `b1d527291c2f80bb41c88d9d6e145c4ed569c4b5b413849cc527a0e6f2a2a1c7`.
- Derived data and reproducible read-only analysis:
  `/root/autodl-tmp/qwen35-decode-trace-20260915.nFBJ9n/`.
- `decode-events.csv`: every kernel, geometry, resource use and preceding gap.
- `decode-gemms.csv`: every selected dense/routed GEMM mapped to layer and shape.
- `decode-layers.csv`: every layer's attention/FFN component time.
- `analysis.json`: complete 15-step and per-symbol statistics.

Each arm has 16 identical output-head endpoints. Excluding the first
prefill endpoint leaves exactly 15 M1 steps, each with 40 layers and 81
normalization boundaries. Native has 1775 kernels/step; reference has 1644.
Each arm uses one stream, with no overlapping kernel intervals in these steps.
This removes the M4-tail ambiguity of the original aggregated evidence below.

### Actual M1 GEMM effective weight bandwidth

MBU here is logical packed weight bytes / device kernel duration / 2700 GB/s.
It includes quantization metadata, but not A/output traffic, repeated requests,
or measured DRAM transactions. It is not an ACU memory-controller counter.
For routed GEMM, count 8 selected experts, not all 256 experts. All means
below use the complete 15-step decode population, not one screen selection.

| Work | N x K | Calls/step | Producer us | Reducer us | Weight MBU | CTA x threads |
|---|---|---:|---:|---:|---:|---|
| Q8 TC S2 | 8192 x 2048 | 40 | 20.006 | 1.146 | 33.00% | 256 x 128 |
| Q8 TC S4 | 4096 x 2048 | 30 | 11.946 | 1.108 | 27.63% | 256 x 128 |
| Q8 TC S8 | 2048 x 4096 | 40 | 11.707 | 1.040 | 28.20% | 256 x 128 |
| Q8 TC S8 | 512 x 2048 | 100 | 6.879 | 0.944 | 6.00% | 64 x 128 |
| Q8 TC S4 | 2048 x 512 | 40 | 4.970 | 1.061 | 8.30% | 128 x 128 |
| Q4 merged gate/up SIMT | 1024 x 2048, active E=8 | 40 | 12.928 | none | 27.04% | 256 x 256 |
| Q5 down TC S1 | 2048 x 512, active E=8 | 40 | 15.464 | none | 13.81% | 256 x 128 |
| Q6 output head TC | 248320 x 2048 | 1 | 393.578 | none | 39.26% | 1940 x 128 |

Q8 uses 64 registers/thread and 12832 shared bytes/CTA. Q4 SIMT uses
92 registers/thread and 1024 shared bytes/CTA. Q5 down uses 94
registers/thread and 43008 shared bytes/CTA. These are allocation metadata,
not measured achieved occupancy or proof of an occupancy bottleneck.

### Adapter and postprocessing costs

| Work | Calls/step | Mean us | Total us/step | Launch and useful traffic |
|---|---:|---:|---:|---|
| Router + M1 prepare | 40 | 9.726 | 389.029 | One 256-thread CTA; no weights or activation gather in this mixed arm |
| Routed SwiGLU | 40 | 3.921 | 156.853 | 16 CTAs; 32 KiB F32 reads + 8 KiB F16 writes |
| Q5 S1 indexed finish | 40 | 1.596 | 63.842 | 64 CTAs; 32 KiB F16 reads + 64 KiB F32 writes |
| Q8 Split-K reducers | 250 | 1.030 weighted | 257.540 | 8/32/64/128 one-warp CTAs; 18--96 KiB per call |
| Added shared-expert SwiGLU | 40 | 1.488 | 59.538 | One CTA; absent as a standalone operation in reference |
| Q6 input/output casts | 2 | 1.341 / 2.549 | 3.890 | Legacy output-head adapter |

The prepare kernel still produces full 256-expert descriptors for the SIMT
gate side, although that consumer does not use the TC shape/output/stride
tables. Router selection is serial across the eight selected slots, and all
warps wait at two CTA barriers. Its approximately 10-us cost is a
latency/instruction/dependency problem to investigate, not bulk weight traffic.
No ACU stall, transaction or achieved-occupancy counters are in an asys report.

SwiGLU's effective useful traffic rate is about 10.45 GB/s; finish's is about
61.59 GB/s. These tiny operations are not credible HBM-saturating workloads.
Eliminating the passes/launches matters more than forcing high standalone MBU.
The graph still executes routed weight multiply and the eight-expert sum
after finish: 2.392 + 2.360 us/layer. They are also paid by reference, but
can be composed with our finish if graph ownership and rounding are preserved.

There are **zero generic gather/scatter, mm_ids_helper, separate grouped
metadata, or directory-builder kernels in these 15 native M1 steps**. Their
large costs in the full-request summary belong to prefill and its tail.
The F16-to-F32 Q5 output placement remains under the `indexed_finish` name;
renaming it does not remove that work.

### Equal-work comparison and where the regression comes from

| Scope | Reference us | Native us | Difference |
|---|---:|---:|---:|
| Routed chain: top-k/prepare through down finishing, per layer | 43.103 | 43.635 | +0.532 |
| Shared-expert gate/up + A conversion/reduction + SwiGLU, per layer | 6.984 | 17.197 | +10.213 |
| Shared-expert down including A conversion/reduction, per layer | 5.010 | 6.031 | +1.021 |
| All kernel-active time, per decode step | 7192.143 | 7586.815 | +394.672 |

The routed-chain comparison includes reference's two Q8_1 activation
quantizers and fused Q4 SwiGLU. Both chains have five kernel launches.
Q8 shared-expert gate/up instead grows from two launches (quantize + fused
matvec/GLU) to five (two producers, two reducers, GLU). Across 40 layers,
this alone adds 408.46 us of kernel work; other improvements partly offset it.
Reference also fuses the residual add into 10 attention-output projections;
the native TC endpoint leaves that add standalone.

Steady captured steps 3--15 have kernel-span times of 7455.185 us/reference
and 7869.461 us/native, with in-step gaps of 262.863 and 282.855 us.
Thus steady intra-step gap growth is only about 20 us: the main difference
is added kernel work, not a giant repeated GPU bubble. Do not equate these
spans with server latency; CPU sampling and inter-step work are outside them.

The capture also includes unusually long gaps before step 2 (19.244 ms
reference / 23.584 ms native), plus slower launch scheduling in native's
first decode step. Keep these recorded outliers separate; they are not
evidence of JIT, and this report does not silently discard them from the
full 15-step statistics. Host GraphLaunch intervals overlap GPU execution
and must not be added to summed kernel time.

## Original archive review and authority

- Archive: `/root/kpack-model-asys.frud843g.results.tgz`.
- SHA256: `76147bd723267b4d98c5602f015f72006f14f148db7db2816f36fe318e9ff38a`.
- Box results: `/workspace/kpack-model-asys.frud843g/results`.
- Model: `Qwen3.5-35B-A3B-Q4_K_M.gguf`.
- Both callers report `b10671-8fe799551`.
- Runtime manifest SHA256:
  `e6ec8f4a29d364de63b9de2dcb86fa6d0686f550b34b298cad99146ff3e83896`.
- Both arms used the same prompt/request hashes and generated the same text.
  This short text agreement is not a model accuracy gate.
- Each process completed one warmup request before capture. The captured
  request has PP=2048, generation=16, one sequence, graphs/fusions enabled,
  and no numerical callback. First-use/JIT is outside capture.
- The server's decode timer covers 15 decode steps: the first generated
  token comes from prefill. Its reported per-token timer divides by 15,
  not by the `predicted_n=16` field.
- The original archive has full-symbol aggregated kernel counts/times and selection
  logs, but **no `.sqlite` or `.asysrep` event timeline**. It cannot directly
  isolate every M1 event from prefill's small-M tail or measure host bubbles.
  The full-report update above supersedes these original timing limitations.

Captured server observations, with profiler overhead present:

| Quantity | Reference | K-pack |
|---|---:|---:|
| Decode milliseconds per step | 9.7154 | 10.6084 |
| Decode change | baseline | +9.19% |
| All-request GPU calls, including prefill | 30155 | 31351 |
| All-request summed kernel milliseconds | 387.772 | 322.465 |

The last two rows are not decode-only or critical-path timings. Prefill's
improvement must not be used to claim decode improvement.

## 1. Q8_0 is using an initial, not measured, decode policy

`native/proof-request/0-native.selection.json` binds every M1 Q8 projection
to `sf_q8_a0_tm16_tn64_tk64_wm16_wn16_s2_bc0_ap0_dn64`, `policy=6`.
All use the same typed module key:
`a851078e9fb78769af9a1e66ee956f16bb52b31132b8a8b38248b8ad906f8858`.

| N | K | Projections per decode step | Split |
|---:|---:|---:|---:|
| 8192 | 2048 | 40 | 2 |
| 4096 | 2048 | 30 | 4 |
| 2048 | 4096 | 40 | 8 |
| 512 | 2048 | 100 | 8 |
| 2048 | 512 | 40 | 4 |

These total 250 TC producers plus 250 reducers per decode step. The source
in `quactlize/dispatch/policy.hpp` explicitly labels this `QKS_Q8_INITIAL`.
It picks the split from estimated tile count, with a target of 144 blocks
and at least two K tiles per slice; it does not minimize a measured cost.
For example, N512/K2048 is sent to S8 without a measured S1/S2/S4 comparison.

`QUACTLIZE_KPACK_GEMV_POLICY` is absent in both profiled environments.
`ggml_quactlize_gemv_config()` therefore builds an empty map for the generic
Q8 SIMT entry, so Auto cannot select it in this run. This does not prove
SIMT would win; it proves no measured Q8 SIMT-vs-TC decision is applied.

The full symbol of the user's 250-call, 2.513-ms kernel is the Q8
`GemmUniversalMixedInputSplitKParallel` producer. It is not Q4 GEMV.

Aggregated small-M Q8 evidence (M1 plus prefill's M4 tail, not M1-only):

| Work | Reference calls / ms | K-pack calls / ms |
|---|---:|---:|
| Small-M Q8 matvec/producers, including unreplaced Q8 | 4360 / 36.504 | 4960 / 42.979 |
| Typed Split-K reducers | none for this reference path | 4000 / 4.176 |

These rows exclude input quantization. The reference quantizes A to Q8_1;
our required W8A16 path does not. They are therefore useful component costs,
not an assertion that every Q8 projection regressed by the same percentage.
Future alternatives must retain the requested unquantized 16-bit compute
contract; silently changing to W8A8 would not be an equivalent fix.

## 2. Shared-expert gate/up fusion was not replaced

The reference has fused Q8 matvec calls. The K-pack caller deliberately
rejects native mmvq fusion when a weight is a K-pack artifact: otherwise
mmvq would interpret reordered bytes as raw GGUF. That guard is correct.
However, the replacement MoE-chain matcher covers routed expert GEMMs,
not the ordinary Q8 `ffn_gate_shexp` / `ffn_up_shexp` dense pair.

Consequently those shared-expert pairs are separate TC calls plus a
standalone GLU. The archive shows 600 additional small-M Q8 producers and
600 additional `unary_gated_op_kernel<op_silu,float,true>` calls, consistent
with 40 shared-expert layers over 15 decode steps. Separate GLU calls total
670 in the reference and 1270 in K-pack; their summed time increases from
4.009 to 4.940 ms. These totals also contain unchanged work.

## 3. Compare routed MoE with the reference's already-fused operation

For M1, the selected Q4 gate/up is
`q4_decode::kernel<1,2,7,8,8,4,1024,2048>`. It is the measured SIMT recipe.
The Q5 down selection is
`fqg_q13_l2_tm8_tn64_tk256_wm8_wn16_s2_ap0_dn64_nonpersistent`, S1,
`policy=5`. It is not a Split-K down producer in this case.

| Comparable work | Reference | K-pack |
|---|---:|---:|
| Gate + up + SwiGLU | 17.865 us in one fused Q4 matvec | approximately 12.919 + 3.921 = 16.840 us |
| Q5 down producer | 15.476 us | 15.464 us |

The reference rows and K-pack SwiGLU/down are means of 600 M1-only calls.
The K-pack gate/up value is the user's one-step M1 window. The archive's
640-call gate/up aggregate also contains 40 M4 tail calls, so its 14.445-us
mean must not be labelled M1 latency. The first row is an indicative
component comparison, not an exact paired critical-path measurement.

`mmvq.cu` applies SwiGLU to gate/up accumulators before writing the reference
output. Our `run_mixed()` invokes separate prepare, gate/up producer,
SwiGLU, down producer and finish phases. Thus faster Q4 multiplication does
not provide the entire standalone-kernel speedup to the model. The old
28.68-us K-pack TC call with adapters is not the native reference's
17.87-us fused matvec baseline.

Other routed-MoE costs:

- `moe_chain_prepare_m1`: 600 calls, mean 9.726 us. It includes the fused
  router in this run, so comparing it to a metadata-only helper is invalid.
  The reference still pays top-k and activation quantization separately.
- `indexed_finish<1>`: 640 calls, mean 1.625 us including an M4 tail. S1
  has no partial reduction here; this is indexed placement/F16-to-F32
  output finishing. The native down matvec writes its F32 output directly.
- Prepare does not copy weights. Merged SIMT gate/up skips activation gather.
  It still constructs 256-expert shape/stride/output descriptors for both
  projections. The SIMT side only needs a subset of that state downstream;
  removing unused descriptors is a concrete optimization candidate, not a
  measured explanation of all 9.726 us. The router and barriers also cost time.

## 4. The 394-us kernel is the output head, not a hidden prefill GEMM

The inventory contains one Q6 tensor, `output.weight`. It misses the native
policy and retains the legacy K-pack FQ path. Its symbol occurs 16 times,
once at the end of prefill and once per subsequent decode step.

Its mean is 393.637 us versus reference Q6 matvec's 478.696 us (16 calls
each). It is not evidence of a regression, but the fallback means this run
is not a fully selected model: `fully_selected=false`. The
`PASS_SHORT_REQUEST` trace label only proves selected dense/grouped kernels
executed, not that every tensor used the selector.

## Next work, in priority order

1. Benchmark Q8 W8A16 F32-endpoint SIMT and TC on the five actual M1 shapes
   above, with real reducers and matching cache conditions. Replace the
   initial rule only with measured choices.
2. Restore shared-expert dense gate/up/GLU fusion in a K-pack-aware path.
   Do not remove the raw-GGUF fusion safety guard.
3. For routed Q4 SIMT, compare a paired-output/activation epilogue with the
   current producer plus SwiGLU. Both gate and up must be owned by the
   same compute unit before removing the standalone activation.
4. Reduce M1 prepare's unused SIMT-side descriptors; examine direct indexed
   F32 output from Q5 down to eliminate its separate S1 finish.
5. Read the existing `reference/proof.sqlite` and `native/proof.sqlite` to
   split the captured request into the 15 decode intervals, attribute Q8
   durations by grid/split, and quantify idle gaps. No device rerun is
   needed to export those existing files. Aggregated kernel times alone
   cannot establish how much of the remaining slowdown is host/graph overhead.

The Qwen3-32B large-activation/F16 overflow investigation is separate from
this Qwen3.5 trace diagnosis and remains unresolved.
