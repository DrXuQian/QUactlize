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
These are six dense families, **not grouped-performance coverage**. The
RTX5070 S1 candidate now meets the 5% target on all twelve shape/cache points
in a separate fixed-recipe confirmation; see the larger-shape result below.

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
work retains N=512/1024 and the other families as regressions.
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
reducer or new offline layout is necessary. At that checkpoint, the larger
families still had approximately 5–9% warm gaps; the next section closes those
four measured gaps on RTX5070. This is not all-shape admission. The 5090
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

## RTX5070 larger-shape S1 result

`cuda-q4-n4-static` extends the F16/M1/S1 constant-geometry reader to the four
larger families. It preserves the existing small-shape kernels and all generic
fallbacks. Full-loop specialization is bounded to C4/C8 and W4/5/8/10/16;
other input types, alignment, modes, shapes and split counts still use the
previous paths. The stored K-pack bytes and FP32 accumulation are unchanged.

The two layouts are independently tuned over the original candidate domain.
The selected recipes are then held fixed for another six alternating rounds,
15 event samples per arm/round, without another search. These are the latter
confirmation numbers, not NCU replay times:

| N,K (M=1) | Warm Xplane FP32, us | Warm K-pack FP32, us | Warm delta | Rotating delta | Warm K-pack C/W/S |
| --- | ---: | ---: | ---: | ---: | --- |
| 4096,2048 | 5.8185 | 5.5330 | -4.91% | -3.84% | 8/8/1 |
| 4096,4096 | 10.0780 | 9.4965 | -5.77% | -3.60% | 4/8/1 |
| 5120,8192 | 22.1240 | 21.3720 | -3.40% | +0.39% | 4/8/1 |
| 8192,5120 | 21.6840 | 20.1385 | -7.13% | +1.31% | 4/10/1 |

Positive delta means K-pack is slower. All six dense families pass both
regimes, including the two small regressions; the worst confirmed regression
is +2.32% on N1024/K5120 warm. All twelve selected K-pack recipes are S1.
Thus Split-K is **not required** to reach the measured FP32 Xplane level.

The four K-pack warm profiles use 56–58 registers per thread versus 64
for the matched Xplane kernels, and each profile contains exactly one
producer. Instruction counts are not uniformly lower than Xplane; these
results do not justify attributing every gain to fewer instructions or
higher occupancy. The measured DRAM traffic is zero or small in this warm
regime, so logical weight bytes divided by latency must not be called
physical DRAM utilization. The named NCU counters are retained in the receipt.

All forty new static specializations and four small-shape controls pass the
independent GGUF dot, output/workspace guards, graph replay and zeroed-code
negative. Another 116 F16-aligned and 116 F32-weak-alignment indexed checks
pass across Q2–Q6. These 276 checks do not imply new grouped performance
coverage. The [six-family receipt](measurements/q4_fp32_large_s1_5070_20260911.json)
contains raw confirmation samples, the numerical records and profile counters.
Full sources, binaries and raw profiles are preserved on the RTX5070 host in
`E:\q4-large-V928jM-evidence.tgz` (WSL `/mnt/e/q4-large-V928jM-evidence.tgz`)
(SHA256 `14c2889599c26d10f369efa38ea2e7f893b7d15ede34942257bc3b41c7d83cae`).
The compact receipts are also local under
`/root/autodl-tmp/q4-large-20260911.V928jM/`; a redundant 198 MiB full-archive
copy is not required to use the committed, hash-linked results.

This remains a development CUDA result, not a shipping policy change or a
PPU performance claim. Grouped/multi-token performance, other quantized
formats, and a recheck on RTX5090 are still open. Per the current workflow,
PPU comparison is deferred until the local investigations are collected.

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

The six-family candidate uses `build.py --reader cuda-q4-n4-static`.
`confirm_q4_retuned.py` takes the complete six-family tuning receipt via
`--recipes` and re-measures its fixed winners without searching.
`check_q4_static.py` exercises all forty new large static specializations
plus four small-shape regressions, including zeroed-code negatives.
`run_xplane_ncu.py --retuned-fp32 --large-warm` profiles the two measured
winners for each of the four larger warm cases.

## Prefetch interpretation

Warm-versus-rotating measurements bound a possible benefit, not a net
prefetch speedup. After the reader is sufficiently efficient, revisit
prefetching the already-selected experts' down weights during gate/up.
Include all required planes, contention with the current kernel and the
complete current-plus-target interval. A faster target alone does not admit
the optimization; neither does an experiment excluding the preload cost.
