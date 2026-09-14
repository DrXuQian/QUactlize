# llama.cpp v0.3.0 development branches

This is a selective development port in `DrXuQian/llama.cpp`, not an upstream
PR, a main-branch product release, or a rebase of `feat/kpack-gpu-cache`.

| Branch | Commit | Contents |
|---|---|---|
| `dev/v0.3.0` | `e73e2136b` | Official v0.3.0 plus the supplied PPU patch, unchanged |
| `dev/quactlize-v0.3.0` | `7bddc62e7` | That base plus 40 Quactlize commits and the v0.3.0 compatibility correction |
| `feat/kpack-gpu-cache` | `b312a0955` | Original published feature history retained; not rebased |

Official release commit: `c1d0e7a004015f23bc0233470b747b596f29b264`.
Supplied input: `/root/ppu_dev-vs-master.diff`, SHA256
`0ce55e7beb27ebf3f06cacca997daa72da61324ec6751eb311e66b2aa9be8b33`.
Applying it to an independent index at that release produces tree
`42d1067362c361226f34c1c3e8384e6393158b0d`, exactly the tree of `dev/v0.3.0`.
The patch's six pre-existing trailing-whitespace warnings were not rewritten.

The Quactlize port includes canonical intake/sidecar caching, selected dense
and grouped dispatch, Q8 and MoE integration, measured Q4 decode selection,
and the F32/BF16 dense endpoints. It excludes the old whole-PPU snapshot,
unrelated quantization-budget changes, and the uncommitted new large-M
full-BF16 provider integration. Tests remain in these development branches.

## Compatibility decisions

- Preserve v0.3.0's `llama_load_mode`, tensor reshape, graph fusion and backend
  APIs. Do not restore removed split-buffer APIs from the older branch.
- The model-loader test now calls the load-mode constructor with no allocation
  and no MTP loading, matching its original fixture scope.
- The supplied PPU base extends `ggml_cuda_launch_mm_ids_helper` with
  `write_inverse`. Quactlize's gather uses compact-row-to-input coordinates,
  so it explicitly requests the forward map with `false`.

## Local validation

- Host library/loader/sidecar build succeeds; four CTest entries pass.
- Native-evidence, numerical-runner and GSM8K-runner unit suites pass:
  23 + 33 + 16 = 72 tests. Their generated performance/answer records are
  test doubles, not new model measurements.
- The PPU SDK compiles `ggml-cuda.cu`, `quactlize-execution.cu` and
  `quactlize-buft.cu` against the new tree. This is compile-only evidence,
  not a fully linked new PPU server binary or device execution.
- The separate [decode endpoint box result](KPACK_DECODE_IO_RESULTS_20260914.md)
  admits the tested Quactlize library endpoints. It does not automatically
  admit the migrated llama adapter or its changed graph scheduling.

Next device work is a warmed, single-request model accuracy/trace/performance
comparison on this branch with the pinned endpoint package. Keep that
separate from the still-in-progress large-M provider composition gate.
