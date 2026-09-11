# Q4 K-pack native-word reader, 2026-09-11

The previous N2 instruction/coalescing overhead is not an intrinsic cost of
the canonical K-pack format. A CUDA-only Q4 instruction replacement, with
identical weight bytes, thread mapping, Split-K and FP32 dot order, reduces
the rotating M1/N4096/K4096 complete call from 33.318 to 19.528 us on RTX5070.
This is development evidence, not a PPU production-reader promotion.

## Controlled change

`dev/gemv_cuda/build.py --reader cuda-q4-n2` changes only the Q4 pair body:

- Read each 16-byte packed metadata unit with `uint4` when aligned; retain
  the byte-load fallback for all other public pointer alignments.
- Decode metadata with native CUDA half operations. Preserve the canonical
  rounded products and the single FP32-expression/FP16-rounding `ZMul=8`
  correction. Do not substitute the historical unsigned Xplane affine.
- Extract both columns' nibbles as one packed word operation for each fixed
  K slot; keep fused half2 weight dequantization and FP32 dot accumulation.
- Load FP16 A as half2 when aligned and convert with native CUDA operations;
  retain scalar weak-alignment loads and the same FP16 boundary for FP32 A.

The production `quactlize/execution/gemv.cu`, `reader.hpp`, arrangement,
code-plane addresses, N2 lane ownership, output strides and reducer do not
change. There is no shared-memory B transpose, new weight copy, expanded
scale workspace, activation quantization, or FP16 dot accumulation. Other
formats keep the existing N2 body in the experimental library.

The comparison is specific to the CUDA lowering of the current generic
reader. PPU already has native half support; its instruction change and
benefit must be established separately. These results also do not imply
that CuTe compile-time layout algebra inherently incurs this overhead.

## Exactness

On RTX5090, `q4_native_check.cu` compares against the canonical host
`packed_unit::unit_group<Q4_K,8>` and independent signed-code conversion.
All 4,718,592 half values agree exactly, across six metadata offsets
`0/1/2/4/8/15`, all 6-bit scale/min code pairs, eight group positions and
headers including signed zero and subnormals.

`check_inputs.py --baseline` passes 360 Q2/Q3/Q4/Q5/Q6 contexts: dense
M1/M2/M4, grouped with empty experts, indexed top8 with 1/2/3/4 tokens and
broadcast/per-slot A, FP16/FP32 A, strided output and weak pointer alignment.
Every candidate output is bit-identical to N2 for the same recipe. All
independent official-GGUF FP64 dots pass the unchanged 0.005 conditioned
error bound; maximum observed error is 0.000184753. Guards, graph replays,
ordered downloaded-partial reduction and wrong-expert negatives pass.

The standalone Q4 indexed test additionally rejects zero-code and
wrong-expert inputs. This is not an all-shape numerical proof or PPU admission.

## Complete-call timing

RTX5070/WSL, 48 SMs, 48 MiB L2. Four alternating-order rounds, 15 CUDA-event
samples each. Each sample contains at least 32 graph calls; the first five
graph replays are excluded. Rotating sets are 108 MiB, 2.25 times L2, not a
claim that every cache line is cold. No forced clocks; Windows display active.
5090 was running another GPU task, so it supplied numerical validation only,
not these timing or counter numbers.

All N are 4096 and M is one. Recipes are the previous 5090 winners, fixed
for both N2 arms; this is not a 5070 retune or a new globally optimal policy.
The S8 columns include the separate reducer. The S1 columns use the previous
S1 recipe, identical between baseline and candidate.

| K | Cache mode | N2 S8 (us) | Candidate S8 (us) | N2 S1 (us) | Candidate S1 (us) | Xplane S1 (us) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2048 | warm | 17.844 | 10.072 | 18.250 | 9.112 | 5.023 |
| 2048 | rotating | 18.883 | 11.679 | 19.544 | 10.979 | 9.362 |
| 4096 | warm | 33.210 | 17.845 | 35.020 | 16.981 | 8.102 |
| 4096 | rotating | 33.318 | 19.528 | 37.012 | 19.672 | 16.987 |

Xplane retains partial FP16 dot accumulation, so its remaining gap is not a
layout-only comparison. Candidate K4096 rotating is still about 15% slower;
warm-cache gaps are larger. The no-slower-than-reference target remains open.

## NCU: same K4096 rotating recipes

Application replay, `--profile-from-start off --cache-control none
--clock-control none`. Every pass recreates the rotation history before the
profile range. These producer counters are separate from the unprofiled
complete-call timings above.

| Producer metric | Original N2 | Candidate | Xplane |
| --- | ---: | ---: | ---: |
| Executed warp instructions | 11,663,360 | 6,129,664 | 2,019,328 |
| DRAM throughput (GB/s) | 286.56 | 481.46 | 525.76 |
| DRAM sustained peak (%) | 43.35 | 72.90 | 79.61 |
| ALU-heavy sustained peak (%) | 75.97 | 41.48 | 13.43 |
| Registers/thread | 64 | 52 | 63 |
| Achieved occupancy (%) | 57.85 | 65.12 | 55.09 |
| DRAM bytes | 9,509,120 | 9,459,712 | 9,488,896 |

SourceCounters reports excessive global sectors falling from
1,972,224/2,893,824 (68.15%) to 6,144/796,672 (0.77%). This is a request-side
coalescing diagnostic, **not** an equivalent reduction in DRAM bytes. The B
plane word addresses and thread ownership did not change. Metadata-load and
A-conversion changes are combined here; the experiment does not attribute
the entire improvement to one load site.

The candidate's NCU guidance now identifies memory as more heavily utilized
than compute; the original N2 had much higher ALU-heavy activity. Thus the
instruction cleanup moves this case toward its bandwidth limit without
changing the resident format. The reducer still executes 23,168 warp
instructions in both N2 arms and is not the source of this gain.

## Next steps

1. Retune this instruction-clean reader's bounded S1/Split-K domain and A
   reuse before assuming the old N2 S8 recipe remains optimal.
2. Attribute remaining load stalls and compare A/metadata staging. Keep a
   packed-B shared-memory exchange as a measured alternative, not an assumed
   requirement now that global excessive sectors are below 1% on this case.
3. Compare equivalent accumulator precision with Xplane to separate dot
   arithmetic from layout ownership costs.
4. Extend profitable word-level changes to each two-plane format using its
   actual map. PPU compilation, numerical validation and ACU/timing are
   required before changing production selection or the llama.cpp bundle.

## Reproduction and evidence

Build a separate candidate; never overwrite the measured baseline library:

```bash
python3 dev/gemv_cuda/build.py --cuda /usr/local/cuda-12.8 \
  --output /work/q4-native --jobs 8 --reader cuda-q4-n2
OPENBLAS_NUM_THREADS=1 python3 dev/gemv_cuda/check_inputs.py \
  --library /work/q4-native/libkpack_gemv_cuda.so \
  --baseline /work/n2/libkpack_gemv_cuda.so \
  --fixtures /work/input-fixtures --output /work/q4-native-inputs.json
python3 dev/gemv_cuda/compare_q4_native.py \
  --runner /work/profile_xplane --fixtures /work/binary-fixtures \
  --recipes docs/measurements/q4_xplane_kpack_5090_20260911.json \
  --xplane /work/libq4_xplane.so --baseline /work/n2/libkpack_gemv_cuda.so \
  --candidate /work/q4-native/libkpack_gemv_cuda.so --output /work/q4-native-ab
```

The timing runner binds the baseline DSOs to the published receipt and the
candidate to its source/build manifest. It rejects changed identities,
wrong shapes/configs, nonfinite samples, wrong medians and oracle failures.
The standalone profiler build and NCU options are documented in
[the matched profiling report](Q4_SIMT_XPLANE_KPACK_NCU.md).

- [Timing, counters and hashes](measurements/q4_native_n2_20260911.json).
- [Raw counters, numeric receipts and source manifest](measurements/q4_native_n2_20260911.logs.tgz).
- Local binary/evidence: `/root/autodl-tmp/q4-native-20260911-KQnuUN/`.
- Full NCU reports: `E:/q4-native-20260911-KQnuUN/` on the profiling machine.
- Candidate library SHA-256:
  `dd3523851eee78ba1b2844bc1d20e13f26847590fb0e88f8bce5f6d59807ddf4`.
