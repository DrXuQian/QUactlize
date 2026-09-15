# Explicit BF16 compute

This is an additive computation contract, not a new weight format or a claim
that BF16 has passed device numerical or performance admission. The existing
FP16 APIs, metadata bytes, canonical mappings, and FP16 selection records keep
their meaning.

The motivating finite activation is `243383.484375`: ordinary FP16 conversion
overflows, while BF16 rounds it to the finite value `243712` (`0x486e`). No
clipping is performed.

## Supported computation

| Path | Formats | Domain | Arithmetic / storage |
| --- | --- | --- | --- |
| Grouped FQ TC | Q2_K–Q6_K | Existing ordinary, persistent, compact, and Split-K legal domains; no new M limit | BF16 A/B, FP32 accumulation, BF16 final projection |
| Grouped SF TC | Q2_K–Q6_K, Q8_0 | Same existing grouped legal domains, including prefill | BF16 A/B; existing FP16 scale/zero planes |
| Typed dense TC | Q2_K–Q6_K FQ/SF, Q8_0 SF | Existing decode M=1..8; AP0 | F32 or BF16 I/O; BF16 compute; FP32 accumulator and split partials |
| Register-reuse SIMT | Q2_K–Q6_K, Q8_0 | Existing dense/indexed/grouped token limits, including 1..8 | BF16 A, original F32 group-affine weight arithmetic, FP32 accumulation/output |
| Fused indexed/MoE helpers | TC and SIMT projections | Existing small-row helper domain, at most 64 routed rows | F32 external I/O; explicit BF16 TC projection / activation / down boundaries |

The all-M claim belongs to the grouped TC interface. It does not extend the
small-row fused router to prefill. A prefill caller using its ordinary grouped
adapter must supply BF16 A and consume BF16 projection output. The existing
full-weight-dequant + BF16 provider path is not changed here.

AP1 packed-A is not admitted for BF16: its separate provider contract still
requires FP16. BF16 compile requests for AP1 explicitly decline. BF16 M8 uses
the same b16 coordinate map as FP16, with a separately named BF16 MMA
instruction; M16 uses the already available BF16 instruction.

## Numeric boundaries

- Canonical low/high planes and packed units are unchanged.
- FP16 metadata is converted by value; it is never interpreted as BF16 bits.
- Integer code extraction can reuse exact FP16 integer magic. Bounded codes
  are converted by value to BF16 in the same register positions.
- TC rounds metadata multiplication and zero addition separately into BF16,
  retaining the existing two-operation order. The accumulator remains FP32.
- SIMT retains its existing F32 group-affine algebra; only A's named compute
  boundary changes to BF16. It does not claim bitwise equality with TC's
  per-weight reconstruction.
- MoE TC projection completion rounds before SwiGLU. BF16 mixed SIMT
  projections also round at that chain boundary. SwiGLU computes in F32, then
  produces BF16 down input (or F32 storage subsequently rounded by the BF16
  SIMT reader). Weighted finish restores slot order and accumulates in F32.
- Split partials remain FP32. BF16 reducers preserve slice summation order and
  round only at the projection boundary.

## Caller and module interface

All outer structs require their exact version and `sizeof` value. Embedded
v1/v2 structs keep their existing versions. Compute is explicit:
`F16=0`, `BF16=1`.

| Interface | Call / identity | Lifecycle |
| --- | --- | --- |
| Dense typed TC | `qkd_dense_call_v2`, `quactlize_kpack_decode_dense_identity_v2` | `query_v2`, `prepare_v2`; existing `run_v1`, `destroy_v1` |
| Grouped TC | `qk_compute_device_call_v3`, `quactlize_kpack_compute_identity_v3` | `grouped_query_v3`, `grouped_prepare_v3`; existing `run_v1`, `destroy_v1` |
| SIMT | `qkg_simt_call_v2` | `quactlize_kpack_simt_query_v2`, `quactlize_kpack_simt_run_v2` |
| TC MoE private protocol | `qk_moe_projection_v2`, `qk_moe_plan_v2` | `moe_projection_v2`, `moe_stage_v2` |
| Mixed TC/SIMT MoE | `qkg_moe_compute_v2` | `moe_mixed_stage_v2`, `moe_weighted_finish_v2` |

Dense storage remains `QKD_F32=1` or `QKD_BF16=2`. SIMT adds
`QKG_SIMT_BF16=2` only for the explicit-compute v2 endpoint; its v1 endpoint
still rejects that storage type. All SIMT outputs are F32.

`DecodeCompiler(..., compute_type="bf16")` and
`GroupedComputeCompiler(..., compute_type="bf16")` generate distinct compute
identities, source contracts, and cache keys. Default `DecodeCompiler` remains
FP16. BF16 modules reject legacy query/prepare entry points, so an old caller
cannot silently feed FP16 buffers through a BF16 module.

Caller integration must additionally:

1. Include compute identity in policy requests, prepared-handle keys, module
   verification, graph reuse, and MoE projection compatibility checks.
2. Never rank BF16 candidates using measurements recorded for FP16 compute.
3. Load the v2/v3 entry points explicitly, validate the returned compute tag,
   and preserve the original fallback path if capability admission fails.
4. Keep all linked TC and SIMT MoE projections on the same named intermediate
   contract; pass BF16 down storage to a BF16 reader, not to legacy F16 APIs.
5. Teach dispatch/catalog tooling that `grouped-explicit-compute-v3` is not a
   typed dense module. This directory does not publish or update production
   manifests, policy tables, or caller headers.

## Local gates

These commands compile independently of a device. Use a fresh output directory
and a private cache, not a shipping bundle or an in-use JIT cache.

```sh
python dev/bf16_compute/matrix.py --sdk /path/to/PPU_SDK \
  --cache /data/bf16-cache --output /data/bf16-grouped-compile --jobs 4
python dev/bf16_compute/matrix.py --sdk /path/to/PPU_SDK \
  --cache /data/bf16-cache --output /data/bf16-dense-compile --kind dense --jobs 4
python -m pytest -q tests/test_bf16_compute.py
```

The grouped matrix covers 22 complete modules: all formats, FQ/SF where
supported, and TM8/TM64. Each module compiles its existing ordinary,
persistent, compact, and Split-K implementation branches; resource/shape
admission still belongs to runtime query. The dense matrix covers 11 typed
modules. These are bounded compile checks, not a configuration sweep.

The host tests execute real scalar metadata and MoE boundary functions. They
check the large finite activation, an FP16-overflow negative, a metadata-bit
reinterpretation negative, ABI sizes, cache separation, and wrong-type/version
rejection. The compile probe checks that BF16 and FP16 M8 A readers preserve
the same coordinate map. None proves GPU MMA arithmetic or GPU scheduling.

## Required device admission

Before enabling BF16 in a production caller:

- Check Q2_K–Q6_K and Q8_0, dense decode and grouped small/large M, both
  native BF16 and F32 source storage where supported.
- Check ordinary/persistent/compact routes and S1/S2/S4/S8 when query admits
  them, including non-first M tiles, partial tiles, empty experts and ragged
  counts. Compare with an independent typed oracle, not BF16-as-FP16 bits.
- Replay graphs with changing A and routing IDs; cover pure TC, pure SIMT,
  and mixed gate/up/down plans and the merged gate/up layout.
- Include the real Q6 down shape M1/N5120/K25600 with the captured finite
  SwiGLU activation. Then rerun the first-nonfinite and model numerical gates.
- Rerun legacy FP16 gates. This change preserves contracts, but does not
  substitute host checks for a device regression.
- Time BF16 independently after JIT/warmup. The FP16 performance authority is
  not BF16 performance authority.

## Dedicated device gate

`package.py` builds a bounded inventory of 40 TC modules, a six-format SIMT
smoke library, and the real MoE helper library. It does not build a config
sweep. The 40 TC modules include both FQ parent identities, both small and
large tiles, one FP16 grouped control per format, and separate FP16/BF16
Q6 dense modules for the captured outlier shape.

```sh
python dev/bf16_compute/package.py --sdk /path/to/PPU_SDK \
  --cache /data/bf16-cache --output /data/bf16-device-package --jobs 4

CUDA_VISIBLE_DEVICES=0 python dev/bf16_compute/run.py \
  --sdk /path/to/PPU_SDK --package /data/bf16-device-package \
  --output /data/bf16-device-results
```

An already built execution library can replace the smoke/helper builds with
`--execution /data/current-execution-build`. Reuse requires the original
manifest, matching payload and source hashes, all six generic SIMT recipes,
and the explicit BF16 exports. This is not an unchecked `.so` override.

The default 746-case denominator is:

| Family | Cases | Coverage |
| --- | ---: | --- |
| Grouped TC | 226 | Six formats, FQ/SF where defined, small/large M, compact/persistent/noncompact, S1/S2/S4/S8; six FP16 controls |
| SIMT | 396 | Tokens 1..8, F32/BF16 input, dense/grouped/indexed shared-A and per-slot-A; FP16 controls |
| MoE chain | 116 | Six formats, tokens 1/4/8, pure TC/pure SIMT/mixed, merged/unmerged; Q4 gate/up + Q5/Q6 down |
| Q6 outlier | 8 | M1/N5120/K25600, S1/S2/S4/S8, BF16 positive and real FP16-overflow negative |

The current v3 ordinary request becomes compact when experts <= 1024.
Therefore the noncompact cases use experts=1025, including an active last
expert; the gate does not relabel compact timing as ordinary execution.
Weight data uses 17 distinct, documented expert patterns to keep fixture
construction bounded, while active IDs, row counts, and A change on replay.

The oracle starts from raw GGUF and the official GGUF dequantizer. Dyadic
metadata makes these fixture weights exactly representable in both FP16 and
BF16, including the TC multiply-then-add boundary. Every output is checked.
Random non-dyadic metadata conversion has a separate host scalar gate; this
device fixture does not claim all rounding cases have been exhausted.

MoE additionally checks the prepared BF16 A bits, stable expert row map,
individual projections, SwiGLU/down conversion, weighted-sum order, and the
complete changed-input graph. `expf` differences may move the random SwiGLU
oracle across one BF16 ULP; a separate exp(-482)=0 case checks exact finite
conversion above the FP16 range. Individual dots use the existing 0.5%
relative-L1 threshold; the complete three-projection chain uses an explicit
2% propagation bound. Neither is a replacement for the model numerical gate.

Each format/family runs in a fresh process. A numerical/runtime failure stops
that child, not the remaining formats. `--resume` preserves successful
children and reruns failed children only with the same package/options.
`--family grouped --qtype 14` selects a diagnostic subset and reports its own
denominator. Default `--repeats 2 --samples 0` verifies changing data and graph
replay without a performance sweep; optional samples are diagnostic only and
exclude setup/correctness/warmup. The root `summary.json` reports exact passed
and expected counts and never promotes partial coverage to device admission.
