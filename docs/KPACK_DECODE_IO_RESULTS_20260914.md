# Decode endpoint device results, 2026-09-14

**PASS for the tested library endpoints and graph replays.** Model-level
accuracy, trace and performance admission remain separate.

Input archive: `kpack-decode-io.auxlBV.results.tgz`, SHA256
`9177a41062e105d268823a0b98610883a9cc5df29a073cc2fd11dbfaf5838644`.
The [compact receipt](measurements/kpack_decode_io_20260914.json) records
member hashes, the published package identity and the reviewed denominator.
No archive files were extracted over the source tree.

## Reviewed coverage

| Check | Result |
|---|---|
| Ordinary and typed device queries | Both report PPU-ZW810, ordinal 0, 72 CUs; explicit SM attribute also 72 |
| Paired GPU packer | Correct `kpack-fusion-v1` digest and four required exports; canonical/size queries pass |
| Dense F32/F32 | 104/104 requests pass, M1..8 |
| Dense BF16/BF16 storage | 104/104 requests pass, M1..8 |
| Original FP16 ABI control | Q4 M1/N1024/K5120 passes |
| Indexed prepare/finish | 32/32 SIMT-stage cases pass, seven changing-input/route replays |
| MoE router equivalence | 16 modes, 256 fixtures, exact IDs and weights |
| MoE prepare/activation/finish | 192/192 SIMT-stage cases pass, seven replays and rounding/gate-up negatives |
| Real selected gate/up/down chains | 4/4 pass: tokens8/top8/E256, paired/unpaired weights and router on/off, three graph replays |

The 208 dense records exactly match the 104 requests in the published gate
manifest, once for each storage type, with no duplicates or policy misses.
All 27 typed parent identities occur. Raw `device.log` records equal the
JSON summary for both dense and chain results. No failures are recorded.

Dense input/output and workspace guards, changed-input eager/graph execution,
and planted zero-A checks pass. The graph has exactly one kernel for S1
(84 records) or two for S4/S8 (124 records): GEMM plus the reducer when needed.
There are no separate input/output conversion kernels inside these calls.
S2 does not occur among these selected typed-dense requests; the synthetic
indexed/MoE stage tests do cover S2, but are not typed-dense S2 admission.

## Numerical bounds

Errors use the gate's independent GGUF dot oracle, normalized by the sum of
absolute input-weight products. The bound is `0.005`, not a pointwise relative
error against a possibly near-zero dot product.

| Format | Dense records, both storage types | Max F32 error | Max BF16 error |
|---|---:|---:|---:|
| Q8_0 | 16 | 1.04573e-7 | 3.25246e-4 |
| Q2_K | 32 | 9.15336e-5 | 2.01994e-3 |
| Q3_K | 32 | 2.71828e-5 | 3.34121e-4 |
| Q4_K | 64 | 1.68749e-4 | 3.75287e-3 |
| Q5_K | 32 | 1.45141e-4 | 3.78737e-3 |
| Q6_K | 32 | 9.41358e-5 | 3.62582e-4 |

All formats cover N1024/K5120. Q4 also covers N512/K2048 and N4096/K4096
at M1/M8, including the separate measured decode-policy query. Q2--Q6 cover
FQ and SF; Q8 uses its supported scale route. BF16 denotes caller storage;
the TC operands still round to FP16 and accumulate in FP32.

The real chains use Q4 gate/up N512/K2048 and Q5 down N2048/K512. Paired
gate/up doubles N. Their largest normalized error is `4.5635340859e-4`.
The separate stage logs marked `PPU_GEMM_ADMISSION=NOT_TESTED` are not used
as GEMM evidence; the four real selected chains supply that narrower check.

## Boundaries and next work

- The device-query mismatch and old-packer missing-symbol failures are closed
  for this package. Keep old failed runs as infrastructure diagnostics.
- The two SIMT-stage executables isolate metadata, conversions and postops;
  their timings do not price the real selected MoE chain's preparation.
- Dense timings have three warm samples and explicitly exclude first launch.
  They are functional diagnostics, not a cold-weight sweep or a model-speed
  comparison. Do not change heuristics from these timings.
- The library-level no-standalone-cast claim is verified. The actual llama
  adapter must still be checked by a warmed model trace and paired accuracy /
  performance run, including the separately migrated v0.3.0 branch.
- The archive records the runtime source contract, execution/packer digests
  and module keys, but not the runner checkout SHA. No missing checkout SHA
  is inferred from the filename or upload timestamp.
