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

The additive native16 [Flash-Next serving ABI](docs/serving.md) provides raw
GGUF matmul, device-count router/row lookup and explicit SwiGLU rounding.
It also supports Q5_1 and Q2_0 canonical packing, Q8 K640/K320 and native16
indexed execution. Existing v1/tuned paths are unchanged. See
[wheel installation](docs/PPU_WHEEL.md) for Python resource discovery without
device initialization. The NVIDIA compatibility backend remains on its
separate feature branch; it is not built or installed by this PPU package.

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
