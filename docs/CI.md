# Quactlize CI / manual tests

Integration entry: **`bash run_tests.sh`**, from any working directory.
Build entry remains **`bash build.sh --library`**. Nothing clones dependencies,
downloads models, launches a sweep or updates a selection table.

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

`PYTHON=/path/to/python` selects the interpreter. `--jobs` defaults to 4.
`--timeout` is seconds per phase (default 7200, including cold JIT).

## Commands

```bash
# Inventory only; includes separately retained historical tests.
bash run_tests.sh --suite all --list

# No GPU; SDK headers and pinned actlize are required.
PPU_SDK=/usr/local/PPU_SDK bash run_tests.sh --suite host

# Complete product CI: host -> build -> accuracy.
CUDA_VISIBLE_DEVICES=0 PPU_SDK=/usr/local/PPU_SDK \
  bash run_tests.sh --suite all --jobs 8 --output /workspace/quactlize-ci-RUN_ID

# Separate build and device jobs, also usable from ncp_flash_lib CI.
bash run_tests.sh --suite build --package /workspace/quactlize-package-RUN_ID \
  --output /workspace/quactlize-build-tests-RUN_ID --jobs 8
CUDA_VISIBLE_DEVICES=0 bash run_tests.sh --suite accuracy \
  --package /workspace/quactlize-package-RUN_ID \
  --output /workspace/quactlize-device-tests-RUN_ID
```

Output/build paths must be new. Without `--output`, a unique temporary
directory is printed. Existing results and packages are never overwritten.
Pass the **whole** exported package to accuracy: `manifest.json`, `include/`,
`libquactlize_ppu.so`, `quactlize-runtime/` including JIT headers. Payload and
SDK identities are checked before loading. No llama checkout is required.

## Suites and coverage

`ci/suites.py` is the single file inventory; no test list in provider YAML.

| Suite | Scope |
|---|---|
| `host` | Real host ABI/oracles, packing, selectors, dynamic binding, BF16 arithmetic, JIT/cache/package contracts, helpers and runner negatives. Device calls are explicit doubles. |
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
The unused `ci/codex_liveness.py` agent process monitor was removed, not tests.

## CI artifacts and verdict

Exit 0 requires every requested phase to pass. Nonzero includes missing
prerequisites, zero tests, skipped/xfail tests, timeout and missing reports.
Ambient `PYTEST_ADDOPTS`/auto-loaded plugins cannot filter the test selection.

Archive `summary.json`, `inventory.json`, `*.command.json`, `*.xml`, `*.log`
from the output directory. JUnit is emitted per phase, including setup/build
failures. Keep binaries and `jit-cache/` in separate artifact/cache jobs;
do not upload the entire output directory. PASS never rewrites package
admission fields or production policies. No hosted CI is enabled here.
