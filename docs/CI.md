# Quactlize CI / manual tests

CI has two shell entrypoints, usable from any working directory:

- **`bash build.sh --library`** builds the public package; no device tests.
- **`bash run_tests.sh --suite test --package PATH`** runs host contracts and
  PPU accuracy against that package; it never rebuilds the package.

Nothing clones dependencies, downloads models, launches a sweep or updates a
selection table. Host harnesses still compile small test binaries, and device
tests may JIT selected kernels; test-only does not mean compilation-free.

## Requirements

- Python 3.10+, Bash, Git, GCC/G++ (C++17), binutils and the PPU SDK.
- Install `ci/requirements.txt`. Supply Torch in the image; do not let pip
  replace a vendor PPU Torch wheel. Host tests use CPU tensors, no `_C.so`.
- Checkout actlize at Quactlize's `third_party/actlize` gitlink. Authenticated
  SSH checkout belongs to CI; scripts never perform nested HTTPS fetches.
- `PPU_SDK` defaults to `/usr/local/PPU_SDK`. Host tests need its headers;
  build/device tests also need the compiler and actual runtime/link DSOs.
- Accuracy requires one visible PPU (`CUDA_VISIBLE_DEVICES=0`). This is not
  performance admission; no idle/clock requirement or profiler is imposed.

`PYTHON=/path/to/python` selects the interpreter for both entrypoints.
`--jobs` defaults to 4 for builds.
`--timeout` is seconds per phase (default 7200, including cold JIT).

## Commands

```bash
# Build job: PPU compiler/runtime libraries needed, no GPU needed.
PPU_SDK=/usr/local/PPU_SDK bash build.sh --library \
  --output /workspace/quactlize-package-RUN_ID --jobs 8

# Test job: consumes the build job's complete package; does not rebuild it.
CUDA_VISIBLE_DEVICES=0 PPU_SDK=/usr/local/PPU_SDK \
  bash run_tests.sh --suite test \
  --package /workspace/quactlize-package-RUN_ID \
  --output /workspace/quactlize-tests-RUN_ID
```

The build job publishes the whole package; the test job also checks out the
same Quactlize commit and pinned actlize and uses the same SDK. Results belong
in a separate directory. No llama checkout is required.

For local diagnosis or an existing combined CI job:

```bash
bash run_tests.sh --suite test --list  # inventory only, no dependencies needed
bash run_tests.sh --suite host        # no GPU; SDK headers + pinned actlize
CUDA_VISIBLE_DEVICES=0 bash run_tests.sh --suite accuracy --package /workspace/quactlize-package-RUN_ID
CUDA_VISIBLE_DEVICES=0 bash run_tests.sh --suite all --jobs 8 --output /workspace/quactlize-ci-RUN_ID
```

`test` and `accuracy` require an explicit `--package`; missing or invalid
packages fail rather than triggering a build or searching for old libraries.
`all` retains host -> build -> accuracy for compatibility; it is **not**
test-only. Omitting `--suite` retains the host-only default.

Output/build paths must be new. Without test `--output`, a unique temporary
results directory is printed. Existing results and packages are never overwritten.
Pass the **whole** exported package to accuracy: `manifest.json`, `include/`,
`libquactlize_ppu.so`, `quactlize-runtime/` including JIT headers. Payload and
SDK identities are checked before loading. No llama checkout is required.

## Suites and coverage

`ci/suites.py` is the single file inventory; no test list in provider YAML.

| Suite | Scope |
|---|---|
| `host` | Real host ABI/oracles, packing, selectors, dynamic binding, BF16 arithmetic, JIT/cache/package contracts, helpers and runner negatives. Device calls are explicit doubles. |
| `test` | Host + accuracy against an existing package, fail fast. CI test job; no package build. |
| `build` | Existing build.sh, complete public library and receipt/export verification. |
| `accuracy` | Real GPU tests under `tests/ppu/`, sharing one buffer/ABI/oracle helper. |
| `all` | All three product suites, fail fast. Not every historical experiment. |
| `tooling` | Other Python harness/analysis regressions; may require their documented extra inputs. Not release admission. |

Device coverage:

- Q8_0, Q2_K..Q6_K: ordinary and gate/up pack, three experts, byte-exact
  oracle, guards, mapping identity and NULL high planes.
- Public dense F16/BF16, M=1..8 and 9/16/128/256: actual selected plans,
  changed inputs/addresses, two streams and disjoint workspace, strided
  decode, replay, zero-input negatives, insufficient workspace, BF16 range.
- Explicit full-dequant+BF16 provider for Q2_K..Q6_K at M128. Q8 is not
  supported by that candidate ABI; its default route is tested above.
  **cuBLAS uses ordinary calls, not CUDA Graphs.**
- Indexed SIMT MoE: six formats, tokens1..8, both compute types, shared and
  per-slot inputs, changing IDs, Split-K1/2/4/8 and actual reducers.
- Existing independent grouped FQ/SF TC oracle: Q4/Q5, decode and prefill.
  Existing selected gate/up/SwiGLU/down/finish chain oracle: F16/BF16,
  tokens1/2/8, with and without fused router. No duplicated reference code.

This is bounded regression coverage, **not** every shape/config, every MoE
TC format, multi-device TP, model accuracy or a performance sweep. Host PASS
must not be relabelled as device evidence. Historical Box runners remain
available for those experiments. The `separate` section in `--list` records
legacy Torch/six-library, private llama and PPU1.7 tests and their requirements.
`ci/local_gates.py` is retained and labelled a legacy development runner.
Completed dense-helper round scripts and unselected Q8 kernel variants have
been retired. Their product numerical, layout, negative and dynamic-call checks
live in `tests/test_prefill_dense.py` and `tests/test_dense_q8_paired.py`.
Product host CI checks all256Q8 codes against all63488finiteF16scale patterns,
paired distinct scales, signed zeros, subnormals and raw GGUF mappings. Typed
prepare checks retired IDs and rejects BF16 on the F16-only helpers38/41.
The reusable isolated runner retains only the current product cohort; it is
optional tooling, not a second CI entrypoint. No archived experiment PASS
replaces a live correctness gate.

## CI artifacts and verdict

Exit 0 requires every requested phase to pass. Nonzero includes missing
prerequisites, zero tests, skipped/xfail tests, timeout and missing reports.
Ambient `PYTEST_ADDOPTS`/auto-loaded plugins cannot filter the test selection.

Archive `summary.json`, `inventory.json`, `*.command.json`, `*.xml`, `*.log`
from the output directory. JUnit is emitted per phase, including setup/build
failures. Keep binaries and `jit-cache/` in separate artifact/cache jobs;
do not upload the entire output directory. PASS never rewrites package
admission fields or production policies. No hosted CI is enabled here.
