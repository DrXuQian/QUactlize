# Q4 SIMT: FP32-matched comparison

The performance target is K-pack versus **FP32-accumulating Xplane** on the
same GPU, with independently selected configurations. The old Xplane reader
uses half2 partial sums inside each 32-element group, followed by FP32
cross-group and warp reduction. Its old numbers are historical context,
not the acceptance threshold for this comparison.

## Comparator construction

`dev/gemv_cuda/build_xplane_accum.py` builds two named libraries:

- `fp16-original`: the unmodified historical header body;
- `fp32-control`: the same body with FP32 multiply-accumulate chains and
  FP32 intra-group merging. FP16 weights and A, metadata decoding, offline
  producer, addressing, shared A stage, launch geometry and final FP32
  warp reduction are unchanged.

The manifest binds both generated header and library hashes. Passing
`--xplane-arithmetic fp32` to the comparison requires the matching FP32
receipt; renaming the old half library cannot satisfy it. This is a new
FP32 comparator, not a claim that the historical published kernel was FP32.

K-pack normally retains its existing signed-code / `ZMul=8` half affine
boundary and FP32 dot order. Xplane uses an unsigned-code half affine.
Thus equal accumulation precision does **not** make this a pure layout-only
causal experiment or guarantee identical weight rounding. The explicitly
named `cuda-q4-n4-unsigned` control also removes the signed-code compensation
inside the SIMT reader. It does not change the stored bytes, but it changes
that rounding path; it is not promoted to production.

## Scope and current status

The first matched campaign covers Q4 M=1 with `(N,K)`:
`(512,2048)`, `(1024,5120)`, `(4096,2048)`, `(4096,4096)`,
`(8192,5120)`, `(5120,8192)`, separately warm and rotating more than 2.25 L2.
These are six dense families, **not grouped-performance coverage**.

Each configuration must pass the independent official-GGUF FP64 dot
oracle with the unchanged 0.005 conditioned-error bound. Complete call
timing includes Split-K reduction. Graph upload and initial warmups are
excluded. Xplane and K-pack recipes are independently screened and then
confirmed in alternating order. No performance claim is based on an NCU
replay duration.

Experiments remain development-only. Neither PPU production libraries nor
the offline arrangement nor the shipping heuristic is changed.

| Experiment | Finding on RTX5090 |
| --- | --- |
| C4/C8 N2 scheduling | Useful for some families; insufficient to close the gap |
| Shared A only | No gain in the two original anchors; rotating cases slower |
| FP16 intra-group K-pack arithmetic control | No general gain; not an acceptable FP32 target substitute |
| On-chip b16 transpose into a contiguous-K shared tile | Correct on measured dense cases, but slower; not selected |
| Three-dimensional grid | Removes repeated integer grid decoding; small gain |
| Hoisted type/alignment guards | Material gain while preserving exact existing FP32 output bits |
| Native packed-half metadata products | No broad gain beyond the hoisted-guard reader |
| Four N columns per lane, b64 load | Further gains, with small-family regressions; requires per-shape selection |

The aligned N2 and N4 candidates preserve weak-alignment fallbacks. The N4
gate covers 672 contexts across Q2–Q6, including dense, ragged grouped,
indexed top8 with 1/2/3/4 tokens, aligned and misaligned FP16/FP32 inputs,
output guards, graph replay and wrong-expert negatives. All common-recipe
outputs are bit-identical to the N2 reference. The extra N4 C1/C2 domain is
separately checked against the independent oracle during the shape sweep.

**The multi-GPU performance goal is still open.** A large-family result
within 5% does not admit all shapes, the warm regime, other formats, grouped
performance or a PPU policy change.

The accepted latency tolerance is **5% slower than the FP32 Xplane
comparator**, separately for each shape, device and cache regime. Further
work prioritizes N=512/1024 and retains the other families as regressions.
S1 is a first-class candidate, not a missing algorithm: the N2/N4 warp-tree
experiments increase intra-CTA K parallelism and use FP32 warp/CTA reduction
to avoid an extra reducer when S1 wins. Split-K is optional and must earn its
place using complete-call timing. The changed FP32 addition order is checked
against the independent oracle, not required to reproduce serial sums bitwise.

## RTX5070 small-shape S1 result

The fixed-choice confirmation is separate from the recipe screen: six
alternating rounds, 15 graph-event samples per arm/round, with graph upload
and initial warmups excluded. Positive delta means K-pack is slower.

| M,N,K | Cache regime | Xplane FP32, us | K-pack FP32, us | Delta | K-pack C/W/S |
| --- | --- | ---: | ---: | ---: | --- |
| 1,512,2048 | warm | 2.1745 | 2.1275 | -2.16% | 4/8/1 |
| 1,512,2048 | >2.25 L2 rotation | 2.8176 | 2.6198 | -7.02% | 4/8/1 |
| 1,1024,5120 | warm | 4.8100 | 4.9325 | +2.55% | 4/10/1 |
| 1,1024,5120 | >2.25 L2 rotation | 6.8956 | 6.7543 | -2.05% | 4/10/1 |

**These four points satisfy the 5% target using S1 only.** No separate
reducer or new offline layout is necessary. This is not all-shape admission:
the larger-family warm cases still have approximately 5–9% gaps. The 5090
machine was shut down before it could repeat the new small-shape variants;
no 5090 parity claim is made for them.

The changes are in the development CUDA reader: b64/N4 loading, FP32
warp/CTA reduction, cooperative `float4` shared partials, and two bounded
F16/M1/S1 shape specializations. The N=1024/K=5120 row has 160 K-groups;
W=10 with C=4 assigns exactly two groups to every K worker. The standalone
N2/warp-owned-output and forced-64-register controls did not improve the
overall comparison and are not selected.

More CTAs were not the entire answer. The specialized N=1024 reader uses
56 registers versus 71 in the generic N4 tree and 63 in this Xplane control.
Its profiled instruction count is 988,544, compared with Xplane's
1,075,712–1,127,424 across the two recipes. NCU confirms one producer and
no reducer in each of the eight final profiles; profiler durations are
not substituted for the event timings above.

Additional indexed checks passed 116 F16-aligned and 116 F32-weak-alignment
cases across Q2–Q6, including independent GGUF dots, output/workspace guards,
ordered inter-CTA reduction where requested, graph replay, and wrong-expert
and missing-code negatives. Maximum conditioned error was about 1.15e-4,
below the unchanged 0.005 bound. These are single-token/top8 numeric checks,
not new grouped-performance or all-input-ABI admission.

The [compact receipt](measurements/q4_fp32_small_s1_5070_20260911.json)
contains raw confirmation samples, NCU counters, source/library/fixture
hashes and the six-family regression campaign. Production libraries,
llama.cpp routing and canonical packing are unchanged.

## Reproduction

Development/NVIDIA only:

```bash
python dev/gemv_cuda/build_xplane_accum.py --output /data/q4-xplane-fp32
python dev/gemv_cuda/build.py --reader cuda-q4-n4 --jobs 9 --output /data/q4-n4
python dev/gemv_cuda/prepare_fixtures.py --q4-dense-wide --output /data/q4-fixtures
python dev/gemv_cuda/compare_xplane.py \
  --fixtures /data/q4-fixtures \
  --xplane-library /data/q4-xplane-fp32/fp32-control/libq4_xplane.so \
  --xplane-arithmetic fp32 \
  --kpack-library /data/q4-n4/libkpack_gemv_cuda.so \
  --expected-reader cuda-q4-n4 --output /data/q4-comparison.json
```

`tune_q4_standalone.py` provides the CUDA-only, no-PyTorch runner path.
`--batch-runner` reuses fixture packing and allocation across configurations
of one arm; every recipe still gets its own warmup, graph, correctness check
and 15 event samples. It requires the new batch-capable `profile_xplane`.

The final small-shape reader is built with
`build.py --reader cuda-q4-small-balanced`. `confirm_q4_small.py` confirms
the two fixed S1 recipes without searching again; `check_standalone.py`
provides the CUDA-only indexed numerical gate, and
`run_xplane_ncu.py --retuned-fp32 --small-shapes` selects the measured small
families for profiling. None of these commands is a PPU production runner.

## Prefetch interpretation

Warm-versus-rotating measurements bound a possible benefit, not a net
prefetch speedup. After the reader is sufficiently efficient, revisit
prefetching the already-selected experts' down weights during gate/up.
Include all required planes, contention with the current kernel and the
complete current-plus-target interval. A faster target alone does not admit
the optimization; neither does an experiment excluding the preload cost.
