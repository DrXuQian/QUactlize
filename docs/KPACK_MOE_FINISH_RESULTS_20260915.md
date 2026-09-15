# MoE helper gate: GPlpUf

## Verdict

Correctness passes. The weighted finish improves the measured TC S1 and
SIMT S1 cases, but regresses every measured TC S2/S4/S8 case. Do not enable
the new finish unconditionally. These are helper-only results, not model
or selected-GEMM admission.

## Authority and coverage

- Archive: `/root/kpack-moe-finish.GPlpUf.results.tgz`.
- Archive SHA256: `00da971e89eb2bdef3df5f7f51c3d2e8ae02c3e2bb22a30d23e8970f2acf18f3`.
- Binary SHA256: `8d7efd2b92e04cfeb023a81ea587cc3d4a8e85ddea8045f1dd3003a737a72649`.
- All eight recorded input hashes match the tested source at `ba0160d`.
- Both invocations pass 80 mixed-stage cells with seven replays per cell,
  including guards, changing IDs and invalid-route checks. All have `bad=0`
  and `activation_half_bad=0`.
- Router equivalence: 16 modes, 256 fixtures/mode; ID/weight raw-bit mismatch0.
- Weighted finish:15 cells x7 replays; raw-bit mismatch0; reversed slot-order
  negative differs in482468 values. TC FP16 completion is retained.
- Timing:15 alternating samples/arm, each64 graph repetitions, first five
  warmups excluded. Every sample is finite/positive and recomputed medians
  match the reported medians. Reference is equivalent unfused helper stages,
  **not the llama binary**. The measurement is resident helpers, not cold B.

## Complete finish timings

All cases use N2048/top8. Delta is fused/reference minus1; negative is faster.
TC reference has three kernels; SIMT reference has two; fused has one.

| Tokens | Down | Reference us | Fused us | Delta |
|---:|---|---:|---:|---:|
| 1 | TC S1 | 4.880625 | 2.566250 | -47.42% |
| 1 | TC S2 | 4.935000 | 7.113750 | +44.15% |
| 1 | TC S4 | 4.979375 | 9.418750 | +89.16% |
| 1 | TC S8 | 4.983125 | 14.318750 | +187.34% |
| 2 | TC S1 | 4.829375 | 2.598125 | -46.20% |
| 2 | TC S2 | 4.820000 | 7.256875 | +50.56% |
| 2 | TC S4 | 4.847500 | 9.458124 | +95.11% |
| 2 | TC S8 | 4.913125 | 14.453125 | +194.17% |
| 8 | TC S1 | 5.019375 | 2.628750 | -47.63% |
| 8 | TC S2 | 5.021875 | 7.241250 | +44.19% |
| 8 | TC S4 | 5.385625 | 9.533125 | +77.01% |
| 8 | TC S8 | 6.324375 | 14.373124 | +127.27% |
| 1 | SIMT S1 | 3.305625 | 2.385000 | -27.85% |
| 2 | SIMT S1 | 3.250625 | 2.419375 | -25.57% |
| 8 | SIMT S1 | 3.332500 | 2.422500 | -27.31% |

Every regression has disjoint reference/candidate sample ranges. It is not
a marginal timing-noise decision. Kernel count alone is not a useful verdict.

The source gives a concrete performance hypothesis: the fused kernel launches
16 CTAs for M1/N2048, each output thread processes all eight slots and the
runtime-length split loop. The old first stage launches64 CTAs, processes
expert rows independently and has compile-time split unrolling. Both read
contiguous columns; this evidence does not establish uncoalesced memory.
Next candidates should parallelize expert-slot work and specialize split
count, while preserving split order, FP16 completion and ordered slot sum.
No instruction/stall root cause is claimed without matching counters/ISA.

## Prepare remains open

Merged gate/up SIMT + TC down S8, E256/top8/K2048, includes the router:

| Tokens | Prepare us |
|---:|---:|
| 1 | 7.820625 |
| 2 | 15.224999 |
| 4 | 13.053124 |
| 8 | 34.951252 |

The original model trace uses TC down S1, not this S8 fixture. This gate also
has no paired old/new prepare baseline, so comparing7.82us to the original
9.73us trace is **not** a measured prepare speedup. Add a matched S1 case and
counter/geometry evidence before making that claim. No weight copy occurs
in prepare. Active-expert-only TC descriptors, router latency and repeated
multi-token metadata work remain optimization targets.

## Integration boundary

The next caller integration should enable only the measured S1 finish
domain first. An unavailable or rejected finish must keep the original
fused MoE chain and execute the tail normally; it must not drop router/
activation fusion or skip uncomputed graph nodes. The current caller work
is still local and no new runtime/model package is admitted by this result.
Q8 shared-expert fusion/policy and the separate Qwen3-32B nonfinite-input
investigation are unchanged.
