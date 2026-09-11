# Q4 SIMT: historical Xplane versus current K-pack, 2026-09-11

The historical optimized Xplane reader is substantially faster than the
current development K-pack N2 reader on both matched dense shapes. This
comparison does not change the production format, library or selector.

## Matched measurements

Same RTX 5090 (170 SMs, UUID `43f92c1b-db7c-88c6-e466-be075ce7c166`),
driver 580.95.05, CUDA 12.8. Both arms consume the same original GGUF weight
values, identical packed metadata, F16 activations and F32 output. There is
no activation quantization, scale-first workspace, gather or scatter.
Offline packing and copies are outside timing.

Each cache mode independently screens the historical 12 Xplane recipes
(`CtaN=1/2/4/8`, `WarpsN=2/4/8`, `WarpsK=1`) and 24 current N2 recipes
(`columns=16/32`, `warps=2/4/8`, `split=1/2/4/8`). "Best" means best in these
explicit domains, not a proof of a global optimum. N2 columns count thread
positions, with two output columns per thread position.

Five screening samples select candidates; confirmation uses six alternating
arm-order rounds, eleven CUDA-event samples per round. Each sample contains
32 graph calls in warm mode, or two complete weight rotations. Five initial
graph replays are excluded. Times below are medians of round medians, in us.

| M x N x K | Cache | Xplane best, S1 | K-pack best S1 | K-pack best overall | K-pack overall delta |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 x 4096 x 2048 | Warm | 2.909 | 9.658 | 7.381 (S8) | +153.71% |
| 1 x 4096 x 2048 | Rotating >L2 | 4.880 | 11.849 | 9.441 (S8) | +93.46% |
| 1 x 4096 x 4096 | Warm | 4.044 | 17.933 | 12.553 (S8) | +210.45% |
| 1 x 4096 x 4096 | Rotating >L2 | 7.819 | 21.409 | 14.807 (S8) | +89.38% |

Xplane S1 and K-pack S1 each launch one kernel. K-pack S8 times include the
producer and its separate reducer; they are not producer-only times.
Removing the reducer by choosing S1 does not close this gap.

The CUDA runtime reports 96 MiB L2. The rotating set uses 48 address-distinct
copies of the 4.5 MiB weight or 24 copies of the 9 MiB weight: 216 MiB per
arm, exactly 2.25 times L2. Copies contain the same logical fixture values.
This is a capacity-pressure rotation, not an explicit cache flush and not a
hardware-counter measurement of DRAM bandwidth.

The previous Xplane A64 historical winning recipe (`CtaN2/Wn4/Wk1`) is
included and confirmed separately. Its 4096x4096 rotating time is 7.825 us,
consistent with the older 7.623 us record on a different 5090. The warm
2048-K winner is CtaN2/Wn8; the other Xplane winners are CtaN2/Wn4.

## Correctness and attribution limits

All 144 screened cells pass the same independent official-GGUF FP64 dot
oracle at conditioned error <0.005. Maximum observed error is 7.90e-5.
Output/workspace guards and ordered downloaded-partial reduction pass;
captured graph output is checked after timing. Sixteen zero-code negative
checks are rejected. Xplane placement/recovery is exact and its packed
metadata is byte-identical to the K-pack fixture.

However, **the intermediate arithmetic is not identical**: the unmodified
historical Xplane kernel accumulates part of each group in half2, then adds
group contributions in FP32. Current N2 uses FP32 dot accumulation after
FP16 affine dequantization. Both pass this oracle; this is not a bitwise
arithmetic-equivalence claim. Input reuse, lane ownership and load topology
also differ. Therefore the result establishes a current implementation gap,
not that canonical K-pack must be this much slower for every SIMT reader.

The older small-delta Xplane/K-pack FQ GEMM tables cannot substitute for this
SIMT comparison. Nor can these dense Q4 results decide grouped execution,
other quantization types, PPU performance or whole-model performance.

## Source and result authority

- [Full screen and confirmation samples](measurements/q4_xplane_kpack_5090_20260911.json).
- Xplane calls the original `gguf_bc_q4_gemv.hpp` kernel. Its body is unchanged
  from historical commit `c6b30af9a6445bb5c625164712b23a98c23e7927`.
- Xplane kernel header SHA256:
  `4a9a8c5df0e8c24f15a7308b0836a386c0e7fad6076d2fd1f55cd260cd3652e1`.
- Xplane reader header SHA256:
  `d7fa01446a88eab7ffd9ca872dee7a1f58ee42debefea212d0b9fca1a299ff7f`.
- Xplane comparison DSO SHA256:
  `df713ebafedf5851efc179193d48b708b0cd43ffdec49f30ce07db90c60cf1fd`.
- K-pack reuses the manifest-bound N2-v2 development DSO from the NCU run:
  `eec36616718b24490237f441edcd3099a6629b0323ca7d8b22ac1bfa54780955`.
- [Comparison runner](../dev/gemv_cuda/compare_xplane.py) and
  [unchanged-kernel wrapper](../dev/gemv_cuda/xplane_compare.cu).

No new production SO or PPU bundle is needed for this result. The next
attribution experiment should control accumulation precision and data reuse
before deciding which part is caused by layout.

## Reproduce on an NVIDIA development machine

From the repository root, with fresh output paths and no competing GPU job:

```bash
mkdir -p /workspace/q4-layout-ab
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
from pathlib import Path
from dev.gemv_cuda.prepare_fixtures import export
out = Path('/workspace/q4-layout-ab/fixtures')
out.mkdir()
for k in (2048, 4096):
    export(out, 12, 4096, k, 1, [0], 1)
PY

/usr/local/cuda-12.8/bin/nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 \
  --expt-relaxed-constexpr -Xcompiler=-fPIC -shared --cudart=shared \
  -Ibenchmarks -Iquactlize/include -Ithird_party/actlize/include \
  -Idev/fold_derivation/stub_inc dev/gemv_cuda/xplane_compare.cu \
  -o /workspace/q4-layout-ab/libq4_xplane.so

python3 dev/gemv_cuda/build.py --reader cuda-n2 --jobs 8 \
  --output /workspace/q4-layout-ab/n2
OPENBLAS_NUM_THREADS=1 python3 dev/gemv_cuda/compare_xplane.py \
  --fixtures /workspace/q4-layout-ab/fixtures \
  --xplane-library /workspace/q4-layout-ab/libq4_xplane.so \
  --kpack-library /workspace/q4-layout-ab/n2/libkpack_gemv_cuda.so \
  --output /workspace/q4-layout-ab/result.json
```
