# Qwen3-32B decode numerical blocker

This is not fixed or admitted. Do not clamp activations, relax the numerical
bound, drop the failing row, or call a different route to hide the failure.

## Frozen evidence

Input archive: `/root/kpack-first-nonfinite.Um0LH5.results.tgz`.
The first nonfinite tensor is `ffn_out-2`, Q6_K down projection,
M1/N5120/K25600, ordinary TC with Split-K8:
`fqk_tc_q14_l2_a0_tm8_tn128_tk128_wm8_wn32_s2_bc0_ap0_dn32`.

- All 25,600 saved `ffn_swiglu-2` F32 values are finite.
- Exactly one value overflows F16 round-to-nearest: index5613,
  `243383.484375`, F32 bits `0x486daddf`, becomes F16 `0x7c00`.
- Input SHA256:
  `7c63a6ef0c22ef3296e555dc38f158e0063855d0a6a5f7452b82f0b2ecd30755`.
- Output:37 NaNs,2500 positive infinities,2583 negative infinities;
  zero finite values. Output SHA256:
  `1b159824a9f87dd7e53abf356311722909d59bb1410d0016eeff19477591c4cb`.

`tests/kpack_decode_input_host.cpp` now feeds the frozen outlier through the
actual `copy_decode_a` writer at its original K coordinate, then reads the
physical TSM layout with the independent calibrated reader. It reproduces
the F16 infinity. The same writer with BF16 storage produces finite
`0x486e`; this is a range counterfactual, not a BF16 MMA implementation or
device admission.

## What is and is not established

The typed dense endpoint changes pointer storage but retains the F16 TC
compute boundary. `copy_decode_a` converts through `Compute(float(raw[i]))`,
and the selected compute type is `cutlass::half_t`. The old llama gather
also uses `__float2half`. Removing gather/scatter is therefore not, by
itself, a proven explanation for the new model failure. Current SIMT F32
views also round through F16 and need the same range audit.

The narrowed outlier is a sufficient mechanism for nonfinite products;
the origin of the large SwiGLU value is not yet proved. The uploaded source
was read AFTER the failing node. There is no paired reference intermediate
in this archive, and an earlier finite-but-wrong gate/up output has not been
excluded. Quantized weights do not bound intermediate activation range.
F32 accumulation cannot repair an infinity already produced by input
narrowing. Changing only the storage label to BF16 cannot fix an F16 core.

## Why the native Q6 decode route has a different range

In the inspected v0.3.0 caller, ordinary quantized M1 selects MMVQ, not
the BF16 GEMM path. `mmvq.cu` calls `quantize_row_q8_1_cuda` on F32 A.
`quantize.cu` computes `d = max(abs(x))/127` per32 values, then stores
int8 codes and the scale. The Q6 dot uses integer DP4A and F32 scaled
accumulation; `vecdotq.cuh` reads only the low scale half of `ds` for Q6.
It does not convert the unscaled activation itself to F16.

For the saved outlier's actual32-element group, the F32 scale is
1916.4053955078125, stored F16 scale1916 and int8 code127, reconstructing
243332 in F32. Those values remain finite. The separate stored Q8_1 sum
would overflow in this group, but this Q6 dot does not consume that sum;
do not generalize this explanation to every quantized route or every input.

True F32 activation arithmetic for SIMT or a BF16 TC compute specialization
would preserve the outlier's range. Neither is selected as a fix before the
paired upstream comparison. Clipping was considered and then paused by the
user; no production clipping change was made. Any numerical-contract change
needs independent tests and new device admission.
The existing F16 ABI and historical timings are separate contracts; no
production change or claim of unchanged performance is made by this note.

## Next bounded comparison

The existing diagnostic accepts `SNAPSHOT_UNTIL=ffn_swiglu-2`. It runs the
original corpus/KL command in two fresh processes, reference then native,
but saves computed floating-point intermediates only through the first
token's third SwiGLU result, before the failing down projection. Snapshots are taken at each producer before
downstream consumers, with original strides and per-file hashes.

The callback changes graph partitioning, as in the earlier tensor diagnostic;
this is neither a timing run nor whole-model accuracy admission. A finite
target exits87; a numerical failure still exits86 and remains a failure.
Missing target, dump, coverage or mismatched tensor shape is not a pass.
Bounds are512 nodes and64MiB per arm. No weights or model copy is archived.

```bash
(
    set -e
    cd /sim/eec/shared/junfu.qx/quactlize
    PREVIOUS_RUN=/workspace/kpack-q4-model.c6etoF \
    LLAMA_CI_DIR=/sim/eec/shared/junfu.qx/llama.cpp \
    SNAPSHOT_UNTIL=ffn_swiglu-2 \
    CUDA_VISIBLE_DEVICES=0 JOBS=192 \
    bash tools/run_kpack_first_nonfinite_box.sh
)
```

Update the caller and runner first. The command reuses the existing runtime,
weights, corpus, JIT cache and builds the caller incrementally through
`.aoneci`; it does not rebuild Quactlize GEMM modules. The result archive
includes `activation-range.json` with paired min/max/error and F16 overflow
counts. Overflowing coordinates, including index5613 if reproduced, show
both arms' values and bits at that SAME index, the captured gate/up operands,
and an independent host-F32 SwiGLU recomputation. Host exp is not a device
bitwise oracle. `KPACK_ACTIVATION_RANGE` prints the third FFN's three rows.
A missing full-model fix remains open regardless of this diagnostic
process's successful exit.

Fix acceptance requires locating any upstream divergence, choosing a
compute representation that supports the actual values (not saturation),
replaying the failed input through the real producer and Split-K reducer,
and passing the original model numerical test with no fallback. The small
F16-exact SIMT tests do not close this blocker.
