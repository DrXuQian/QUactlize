# BF16 fast-path contracts

This extends existing kernels; it does not establish a BF16 performance policy.
FP16 entry points and offline weight bytes are unchanged. No value is clipped.

| Path | BF16 behavior | Domain |
| --- | --- | --- |
| Selected Q4 SIMT | F32 or BF16 A is rounded/read as BF16; FP32 affine weight reconstruction, accumulation and output | Existing selected dense/indexed shapes, tokens 1–8 |
| Explicit Q4 indexed S1 | F16/F32/BF16 storage, explicit BF16 A compute; FP32 affine/accumulator/output | Existing 16 readers/configurations and six shapes |
| TC packed-A (AP1) | Typed BF16 packed writer, M8 reader and BF16 MMA | Existing Q2/Q4 dense M=1, TM=WM=8 domain; FQ and SF |

The Q4 metadata reader previously used FP16 per-weight affine arithmetic. Its
BF16 specialization decodes the original GGUF header to FP32 and applies the
affine transform in FP32, matching the other BF16 SIMT readers. Code unpack
still uses exact integer-to-half magic; BF16 data is never interpreted as
half bits. Metadata remains stored in its original FP16 format.

The existing B addresses, load vector widths, warp ownership, K grouping,
reduction order and kernel launch geometries are retained. BF16 activation
conversion and metadata-reader affine instructions differ; equal performance
must not be inferred from unchanged addresses. FP16 kernel names retain their
old template arity. New selected kernels are named `kernel_bf16<...>` and the
explicit S1 kernels `indexed_kernel_bf16<...>`.

## Interfaces and integration

Production execution exports additive
`quactlize_kpack_q4_decode_select_v2` / `run_v2`, accepting `qkg_simt_call_v2`.
Version 1 retains its original semantics. Version 2 rejects BF16 storage with
FP16 compute. Geometry transferred from the FP16 selector is **INITIAL**, not
a measured BF16 winner.

`dev.gemv_ppu.moe_s1.source_v2` generates additive
`quactlize_q4_s1_query_v2` / `run_v2` for explicit reader experiments. Its old
`source` generator still produces only the version-1 experiment and denominator.
The production selected library does not export these explicit-sweep symbols.

Caller integration must bind the new selected entry points and preserve the
compute identity. A fused MoE path receiving a Q4 recipe must invoke these
typed entry points rather than rejecting it or silently substituting generic
SIMT. A caller that vendors headers must also vendor `simt.h` alongside the
updated Q4 header. No llama.cpp or MoE-chain entry points are changed here.

AP1 retains the source parent's provider identity instead of silently changing
it to AP0. Invalid AP1 shapes remain rejected. Compiler, source and compute
identity changes produce distinct cache keys. Existing FP16 measurements are
not re-labelled as BF16 measurements.

## Local checks

`tests/test_bf16_fastpath.py` executes the actual packed-A writer and M8 physical
reader map on the host, checks guarded storage and exact ownership, and exercises
the real typed activation transport for F32/BF16 storage. It includes the finite
243383.484375 outlier and an FP16-overflow negative. These are host proofs, not
device GEMM accuracy results.

The normal `tools/build_kpack_execution.py` builds all selected Q4 shapes for
both compute types. For one complete explicit S1 family:

```sh
python dev/bf16_fastpath/compile_s1.py \
  --sdk /root/ppu-sdk/2.1.1 --output /tmp/bf16-q4-s1-build
```

The compile-only receipt requires 32 old FP16 and 48 BF16 kernels, all four
versioned exports, and a nonempty FP32-dot native image. Source changes during
compilation invalidate the receipt.

## Device admission still required

Check typed independent oracles for all three Q4 reader families, dense and
indexed tokens 1–8, F32/BF16 inputs, duplicate/ragged IDs in each supported
contract, plus FP16 controls. Exercise real finite-large activations through
the complete MoE chain. Check AP1 Q2/Q4 FQ/SF at legal Split-K settings against
independent BF16 oracles. Finally measure cold-weight latency against the
existing generic BF16 route and the corresponding FP16 baseline; no speedup
is claimed by compilation or host tests.
# Typed selected-Q4 device gate

`build_gate.py` packages the actual `q4_decode_select_v2/run_v2` implementation,
not the generic SIMT reader. Reuse a source-matching execution receipt on PPU:

```sh
python dev/bf16_fastpath/build_gate.py --platform ppu --sdk "$PPU_SDK" \
  --execution /path/to/execution-build --output /path/to/q4-bf16-gate
python dev/bf16_fastpath/gate.py --sdk "$PPU_SDK" \
  --bundle /path/to/q4-bf16-gate --output /path/to/q4-bf16-results
```

For an independent NVIDIA development check, omit `--execution`, pass
`--platform cuda --sdk /usr/local/cuda --arch sm_120 --jobs 4`. This compiles
only the production-selected Q4 closure. It never substitutes NVIDIA results
for PPU device admission. Neither command changes a production policy.

The frozen auto inventory declares 192 geometry requests: 129 selected Q4
requests, 63 expected `QKG_SHAPE` declines to TC, and 258 BF16 numerical cells
(F32/BF16 input storage). Dense covers every auto-policy shape and tokens 1..8;
indexed covers E256/top8, channels 1/8 and tokens 1..8 at N/K = 512/2048,
512/3072, 2048/512, 3072/512 plus merged gate/up 1024/2048 and 1024/3072.
All 40 compiled shape/recipe combinations are covered. A TC decline is not a
numerical pass or fallback.
Only eight distinct expert patterns are generated on CPU; the remaining planes
are copied D2D. Different expert IDs select different patterns.

Every selected request checks eager execution, changed A/IDs in graph replay,
restored graph results, row padding, allocation guards, zero A, wrong-value
negatives and invalid IDs. It also runs the actual F16 v1 implementation and
checks v2 delegation bit-for-bit. A finite activation of 243383.484375 must
remain finite under BF16 and provoke the known F16 overflow negative. Gold is
computed from official GGUF dequantization with independent FP64 category sums
and a separate large-activation column, not from the placed weight reader.

Each weight shape has a separate process/log; a failed process does not stop
other shapes. Missing cases, graph evidence or negative controls fail final
admission. `summary.json` contains numerical evidence only: no kernel timings,
BF16 performance claim, or heuristic-winner promotion.
