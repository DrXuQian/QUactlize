# Q4 small-M production selection

The Q4 decode sweep returned 372/372 passing cases in 91.04 minutes. The
review checked the result/file hashes, independent GGUF controls, all
210,465 samples and 18 ACU reports (96 kernels). The 2,501 structural TC
exclusions are not numerical failures. This admits a bounded selection
change, not a claim of whole-model speed or global optimality.

## Selection

The host uses N, K, operator and M/token count. Indexed requests require
E256/top8 and shared or slot-specific F32 A. IDs remain on the GPU. A config
must have six-round confirmation in every measured context indistinguishable
to this host query. Router names and fixture hashes are never policy features.
Choose the smallest worst measured regret subject to no more than 5% regression
against the old production control; merge adjacent M ranges only when the
confirmed recipe stays within 5% of their choices.

| Scope | SIMT cases | TC cases | Within 5% of measured best |
| --- | ---: | ---: | ---: |
| Dense | 41 | 55 | 96/96 |
| Indexed MoE | 204 | 72 | 230/276 |

The merged dense policy's maximum regret is 2.147%. No selected case in this
cohort is slower than its old production control. There are 64 automatic
range records, generated data rather than 372 handwritten rules. TC-only
fallback recipes are also retained for the explicit decode query.

**The remaining 46 cases are all token8 MoE.** Identical public shapes have
different best implementations when experts repeat. This first version
retains a conservative TC choice for four ambiguous families and does not
add a D2H routing histogram or online timing. Worst regret against a
router-specific oracle is 62.08%, with no improvement over the previous
control for that case; it is not a 5%-optimal MoE policy. These gaps remain
performance debt. Unknown qtypes/shapes and prefill keep their old selection.

## Production interface

- `quactlize/execution/q4_decode.h`: host-only automatic SIMT query and
  enqueue-only S1 execution. Its versioned config includes reader, variant,
  warps, values and columns; it does not reinterpret the old GEMV config.
- `quactlize/dispatch/api.h`: additive `query_decode_v1` selects the measured
  TC recipe. Ordinary `query_v1` and forced FQ/SF behavior remain unchanged.
  Separate ticket-cache keys prevent old/new selection from aliasing.
- The execution library includes only 40 selected SIMT N/K/recipe
  specializations, not the full candidate inventory. Their reader bodies and
  selected native memory/dequant/FP32-operation counts match the measured
  images. Offline bytes, weight cache and arithmetic are unchanged.
- Dense TC uses the measured flat F32/F16 casts. Indexed TC at 40-64 rows
  uses the measured GPU rank/gather and scatter kernels. TC metadata,
  directory and the actual reducer remain inside the native module. This
  does not pretend that the <=32-row fused-indexed ABI was enlarged.

The private llama adapter queries the new interface automatically when both
host and execution libraries provide it. An incomplete/mixed package fails
at loading. Older complete packages retain the old path. No policy-file
environment knob is needed for these Q4 choices. Q8 and other formats keep
their existing routes. The adapter never treats a direct SIMT plan as a
prepared TC chain. Direct projections use GGML's ordinary surrounding
operations; **whole-chain fusion and model latency must be checked on box**.

## Reproduction and handoff

`docs/measurements/q4_decode_policy_20260913.json.gz` retains the compact
confirmed inputs, raw-result hashes and original module receipts. The raw
archive was `q4-decode-sweep-ppu.ogZPsj.results.tgz`, SHA-256
`7d77d8cce080169978c2fb80aa27fc3d57c46b226086891923a6bb13c9ea3411`.

```bash
python tools/fit_q4_decode_policy.py \
  --evidence docs/measurements/q4_decode_policy_20260913.json.gz
python -m pytest -q tests/test_q4_decode_policy.py
```

The prebuilt package is `prebuilt/ppu0010/q4-decode-policy-v1`. Fourteen TC
images are reused byte-for-byte; only the small host and execution library
were rebuilt. It also supports the existing one-parent JIT for unlisted
requests, outside graph capture. The selected decode gate does not JIT.

```bash
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK CUDA_VISIBLE_DEVICES=0 \
  bash tools/run_q4_decode_policy_ppu_box.sh
```

This runs one selected call for 233 policy-boundary/router contexts, not a
new sweep. It checks independent numerics, immutable SIMT output bits,
guards, zero codes/A and changed-input graph replay. Timing excludes setup,
first launch and graph upload. Cases fail independently and a matching
`RESUME_RUN` retains completed results. Upload the printed result archive.
The gate requires an idle PPU; local compilation is not device admission.

## Next: independently timed large-M baselines

After this selection change, add full BF16 weight dequantization followed
by installed cuBLAS dense / DeepGEMM grouped GEMM. Per the measurement
contract, time and diagnose SF metadata expansion and full weight expansion
**independently of GEMM**, then measure the warmed/JIT-ready GEMM separately.
Old modeled SF expansion is not directly comparable to measured dequant.
Report read/write bytes and achieved bandwidth for each dequant kernel.
Any sum of separately measured stages used by a heuristic is a cost estimate,
not a measured end-to-end latency or a proof of matching cache state.
