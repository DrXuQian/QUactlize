# Main productization status

Status: source candidate prepared; main promotion waits for PPU regression
evidence and the compatibility boundary below. No device or performance
admission is implied by compilation.

| Reference | Commit |
|---|---|
| Development input | `b799ca0a95cd083d6885c69da4b38bc3c96ef80e` |
| Remote main base | `1ae3ada58d7abe21eb2d5c429d9bd49386c4f849` |
| Product candidate | `f33c676c640a79da18575b7e6136cc85c0241d66` |
| Candidate tag | `candidate/main-product-20261001` |
| actlize dependency | `021c69300864e03926108867d318572871a44d35` |

No third maintained branch is needed. Keep main and develop; the candidate
tag is a source snapshot for validation, not a binary artifact branch.

## Included

- Q2_K–Q6_K canonical K-pack and Q8_0 K-pack2 producers and readers.
- Dense/grouped TC and SIMT, typed FP16/BF16 compute, FP32 decode endpoints.
- Measured small-M exact selection and constrained fallback, Split-K costs,
  prefill FQ/SF/full-BF16 composition, MoE preparation/postprocessing, paired
  gate/up kernels and their measured selection.
- Content-addressed JIT, a source-only build entry, bounded configuration
  scanning, host/device tests, and a rewritten public README.
- Portable GGUF reference retained without production SDK dependencies.

## Kept on develop

- The legacy six-library and Torch-extension compatibility layer. The
  current llama.cpp loader still needs the old `fmt*.so` entries, including
  host conversion and all-M admission. The new product runtime is not a
  drop-in replacement for that loader yet. Do not redirect a model runner to
  it merely because the new dispatcher is present.
- PPU1.7/CUTLASS 4.3 work pending native-platform admission.
- Experiments, retired layouts, device diagnostics, profiler campaigns,
  collaboration state, archived binaries, and NVIDIA-only support.

The candidate removes those files only from its own tree. Development and
historical commits remain available; no existing model/runtime bundle was
deleted.

## Completed local checks

- 37 host tests passed, including byte-map/reference, ABI, policy and build
  transaction tests. The 84 PPU tests were not executed locally.
- 1,852 measured inventory decisions and 6,480 boundary/local-shard requests
  exactly match the development selector output.
- 17 representative native TC translation units compiled: six dense decode,
  six typed grouped, and five BF16 SF grouped modules.
- Seven pack/dequant/metadata/fusion translation units compiled.
- The host dispatcher linked with an empty compile-only catalog; this is not
  a deployable runtime package.
- Python compilation, documented plan commands, and patch whitespace checks
  passed. No binary payload, retired-layout reader, or development-state
  directory is included in the candidate.

Before removing the dormant grouped routing probe, three representative
native ISA/resource dumps were identical. Removing the probe changes the
ordinary grouped parameter layout and generated code. In the checked Q8
grouped specialization, one kernel's vector-register count rose from 78 to
82; Q4 dense remains ISA-identical. Do not describe the final change as
byte-identical for all kernels.

The local SDK compiler works but its runtime libraries are absent, so the
complete shared libraries must be built on the PPU box.

## Box gate

Create detached worktrees for the candidate and the development-input commit,
initialize their actlize submodules, then run from the candidate:

```bash
CUDA_VISIBLE_DEVICES=6 python3 tools/validate.py \
  --sdk /workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
  --output /workspace/quactlize-main-validation \
  --baseline-source /path/to/development-input-worktree \
  --jobs 192
```

Use an idle ZW810. This builds the five product libraries, runs host/device
regressions, then compares 48 grouped cases against the same configurations
compiled from the development input. Both FP16/BF16, S1/S2 and small/larger
groups are covered. Each comparison uses four alternating-order rounds,
11 samples per round and 64 native repeats per sample. Compilation is outside
timing; SF preparation is excluded from this same-kernel comparison. Results
use warm weights and are not a cold-weight bandwidth or model TPOT claim.

Return the validation directory's `results/`. It contains source identities,
runtime receipts, numerical results, raw timing samples and logs. It contains
no shared libraries. `--resume` preserves complete builds and reruns tests.

## Remaining before promotion

1. Review the source-built package and all PPU regression results; resolve any
   numeric failure or same-config performance regression above 5%.
2. Keep the current llama.cpp deployment on develop until its loader is
   migrated from the six-library interface. A separate caller update and
   model/TP regression are required to replace that deployment.
3. Fast-forward remote main to an admitted candidate. Preserve the existing
   archived local-main history; do not force-overwrite either main history.

Measured layout-performance tradeoffs remain technical debt. Repository
cleanup does not change their performance verdict or grant new optimality
claims.
