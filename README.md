# Quactlize

Weight-only quantized GEMM/GEMV for T-Head PPU.

PPU1.0 ZW610 and ZW810E reuse ZW810's exact selection tables and buckets.
Recorded latencies remain ZW810 measurements, used as selection estimates on
these devices. Launch grids, occupancy and resource checks use the actual
device properties; ZW810E's CU count is not hard-coded to 72.

`develop` contains production code, admitted selection tables, package builders
and product correctness tests. `main` is the separately curated release line.
Debug kernels, benchmarks, Asys/ACU sweeps, historical results and their tests
are maintained on [`experiments/ppu-tuning`](https://github.com/DrXuQian/QUactlize/tree/experiments/ppu-tuning).
The complete pre-cleanup snapshot is `c436ce5e4dc14ec60149b973342749d16125dba4`.

## Build and test

Checkout the pinned `third_party/actlize` gitlink using your environment's Git
credentials. The public builder never fetches or modifies dependencies.

```bash
PPU_SDK=/usr/local/PPU_SDK bash build.sh --library \
  --output /workspace/quactlize-package-RUN_ID --jobs 8

CUDA_VISIBLE_DEVICES=0 PPU_SDK=/usr/local/PPU_SDK \
  bash run_tests.sh --suite test --package /workspace/quactlize-package-RUN_ID

# Host contracts / test inventory (no GPU execution).
PPU_SDK=/usr/local/PPU_SDK bash run_tests.sh --suite host
bash run_tests.sh --suite test --list
```

Build output must be a new directory. Tests without `--output` create a unique
temporary directory and print its path. Both entrypoints honor `PYTHON`.
See [CI integration](docs/CI.md) for prerequisites and result handling.

Install the complete package: `libncp_quactlize.so`, `include/`,
`quactlize-runtime/` and `manifest.json`. ncp_flash_lib only builds/installs it;
packing, config selection and execution remain in Quactlize. The public ABI is
declared in [packing/api.h](quactlize/packing/api.h),
[dense/api.h](quactlize/dense/api.h), [moe/matmul.h](quactlize/moe/matmul.h)
and [ffn/api.h](quactlize/ffn/api.h).
The independent MoE projection takes F32 activations/output and GPU IDs, with
BF16 operands and FP32 accumulation. The library owns routing, preparation,
selection and reduction; the caller only supplies per-stream scratch.

The `pr/mxfp4-w4a16` feature adds native GGUF MXFP4 (type 39) independent MoE
projections. It losslessly repacks E2M1 codes and E8M0 scale bytes into Sail
W4FA16 layout; resident weight bytes do not grow. It does **not** convert a
Q4_K_XL GGUF to MXFP4, quantize activations, or enable MXFP4 FFN/SwiGLU fusion.
Its provider heuristic is unmeasured, separate from existing Q4_K tables.
PPU accuracy/performance validation is still required before product admission.

MXFP4 prepare needs the installed `DeepGEMM-for-sail` Python W4FA16 entry with
uint8 E8M0 support (implementation checked against `f89eae10c0e90c20630b50e4314448f01321bfba`)
and that package's `actlize_v1.0.0` headers. Quactlize's actlize gitlink is not
replaced. The helper compiles one provider-selected recipe outside capture;
native execution passes current A/IDs/C/workspace/stream on every call. Missing
or incompatible providers fail preparation with diagnostics, never requantize
or reinterpret weights. `DG_CACHE_DIR` remains the provider's JIT cache.
The exported product is still one `libncp_quactlize.so`; ncp needs no wrapper.

After building a fresh package with the command above, run the focused gate:

```bash
CUDA_VISIBLE_DEVICES=0 PPU_SDK=/usr/local/PPU_SDK \
  QUACTLIZE_CI_PACKAGE=/workspace/quactlize-package-RUN_ID \
  python3 -m pytest -x -v -s tests/ppu/test_mxfp4_library.py
```

It checks every packed bit, TP/EP slices, top-6 routing, M1..8 and prefill,
changed pointers, two streams and graph replay. Packing all E8M0 byte values
is distinct from arithmetic validation; the zero-exponent test also detects
silent subnormal flushing. No model is required for these tests.

The FFN API composes separate or canonical merged gate/up weights with SwiGLU,
Down and ordered expert weighting. It shares routing and consumes actual
producer partials without duplicate reductions. Gate/up merging produces one
ordinary 2N projection with its measured selector. SwiGLU remains a separate
transition for every M, followed by Down. GEMM-internal activation fusion is
not enabled or linked into the public library.

Dense prefill selection tries exact rows, bounded M/N/K buckets, then one
N-doubling transfer for merged projections. The transfer keeps M/K/dtype and
actual-shape legality, reports a bucket prediction, and never extrapolates FQ
past M64. It is not an exact measurement of the doubled shape.

## Source layout

| Directory | Responsibility |
|---|---|
| `quactlize/` | Formats, kernels, dispatch, JIT and public runtime |
| `policies/` | Admitted measurements, exact tables, buckets and required validation inputs |
| `tools/` | Package builders, deterministic table generators, GGUF metadata and JIT prewarm |
| `reference/` | Independent GGUF/K-pack format oracle |
| `ci/`, `tests/` | Product CI, reusable numerical fixtures and ABI/selection regressions |

PPU1.0/1.5 use pinned actlize; [PPU1.7](quactlize/backends/ppu17/README.md)
remains an isolated backend. Canonical formats are Q8_0 K-pack2 and
Q2_K–Q6_K per-plane K-pack. Consumers validate versioned arrangements; they
must not guess an unknown layout.

## Performance work

Use a separate worktree so profiling does not change the integration checkout:

```bash
git fetch origin
git worktree add --detach ../quactlize-tuning origin/experiments/ppu-tuning
```

Run sweeps and inspect their documentation there. Promote only validated
kernel changes, selection tables and focused regression tests to `develop`;
do not merge the experimental branch wholesale. Keep result archives and
compiled packages outside source branches. Repository cleanup is not a new
GPU performance admission.
