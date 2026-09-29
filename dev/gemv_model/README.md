# Real-model GEMV reader A/B

Prebuilt-only, eight M1 points, 68 bounded candidate specializations. This
does not replace the model runtime, its selection table, or llama.cpp.

The returned `model-gemv.mZfskR` run is [reviewed](../../docs/MODEL_GEMV_REVIEW_20260917.md):
seven exact M1 candidates improve in all six rounds; Q6 retains TC. All 16
ACU reports were re-imported. These are isolated results, not production or
whole-model performance admission.

| Point | Logical N × K | Contract | Incumbent |
|---|---:|---|---|
| Q4 routed gate/up | 512 × 2048, E256/top8 | paired-N4 + SwiGLU, BF16 | fused SIMT C4/P8/W8 |
| Q5 routed down | 2048 × 512, E256/top8 | canonical K-pack, BF16 | SIMT C4/P8/W2, H32 |
| Q8 shared gate/up | 512 × 2048 | paired-N4 + SwiGLU, F16 | fused SIMT C4/P8/W8 |
| Q8 shared down | 2048 × 512 | canonical K-pack2, F16 | hoisted SIMT C8/P4/W4 |
| Q8 SSM output | 2048 × 4096 | canonical K-pack2, F16 | SIMT S8 + fast reducer |
| Q8 QKV | 8192 × 2048 | canonical K-pack2, F16 | exact SF TC S8 + reducer |
| Q8 attention gate | 4096 × 2048 | canonical K-pack2, F16 | exact SF TC S8 + reducer |
| Q6 output head | 248320 × 2048 | canonical K-pack, F16 | exact FQ TC S1 |

All use F32 input/output storage and FP32 accumulation. BF16 inputs are never
silently rounded through F16. M2/M8 are numerical controls only; this round
does not promote performance rules for them. TC tuples match the uploaded
paired-model selection receipt exactly; the three small TC modules are
recompiled from unchanged source, not re-tuned. The SIMT/fusion incumbent
images are copied byte-for-byte from the admitted model artifact.

## Box

On an otherwise idle PPU, from this source checkout:

```bash
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
CUDA_VISIBLE_DEVICES=0 L2_BYTES=67108864 \
bash tools/run_model_gemv_box.sh
```

The script fetches only the pinned experiment binaries using Git LFS. No
box compilation and no model load/JIT. `L2_BYTES` is the operator-confirmed
64 MiB fallback when SDK reports zero; a conflicting positive SDK value is
rejected. SDK/runtime hashes are recorded with results rather than requiring
the same compiler executable on the box.

Default: correctness → cheap screen → incumbent/two finalists, six alternating
rounds of fifteen samples → ACU incumbent/best candidate. Graph upload and
first replay are excluded. Every timed graph traverses active weight bytes
covering at least 2.25 L2. Split-K reduction and paired SwiGLU are included.
ACU uses explicit cache-control=all and kernel replay; its times are not the
rotating full-call performance result. Raw counters and reports are retained.

`GEMV_POINTS=q4-paired-routed,q5-routed-down` restricts points. `PROFILE=0`
skips ACU. `GEMV_BUNDLE=/absolute/verified/bundle` avoids fetching. Use
`RESUME_RUN=/workspace/model-gemv.XXXXXX` with identical source, binaries,
device selection and point list to retain completed points/profiles. A failed
point does not stop the other fresh-process points. Numerical failures do not
participate in timing or selection. Runtime failures abort that process.

Send the printed `*.results.tgz`: it includes `summary.tsv`, point results,
numerics, access models, static resource/ISA summary, `.acurep` and raw CSV.
No model weights or source data are included.

## Local build and checks

```bash
python3 -m unittest discover -s tests -p test_model_gemv.py -v
python3 dev/gemv_model/build.py --sdk /root/ppu-sdk/2.1.1 \
  --baseline /path/to/immutable/model-runtime --output /new/empty/output --jobs 8
python3 dev/gemv_model/inspect_native.py /new/empty/output
```

The builder refuses an existing output and changed incumbent hashes. The
inspector verifies 68 emitted bodies, FP32 FMA, fast-code LOP3 and vector
loads. It reports actual registers and encoded shared-memory allocation
fields, without treating those fields as bytes or predicting speed from them.

Current static findings: Q4 C8/P4 reduces VREG 94→68; Q5 P4 reduces 114→80
(fixed-N/K P4:60, but fixed-N/K P8:216); Q6 direct metadata reduces 102→76.
Q8 paired hoist alone increases 78→118, whereas C8/P4 gives 60 without
hoist and 78 with hoist. Every candidate has zero reported stack bytes.
These are compile-time observations, not measured device improvements.

The paired half-width candidate keeps complete G4/U4 pairs and the same
sequential inter-warp sum. For shared Q8 M1 it raises the grid from 32 to 64
CTAs, at the cost of more CTAs/metadata requests and a smaller contiguous B
footprint per K worker group. It is S1-only and uses an experimental finish;
the production fusion implementation is unchanged.
