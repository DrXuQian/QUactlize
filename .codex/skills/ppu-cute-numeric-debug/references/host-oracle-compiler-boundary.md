# Host CuTe Oracle vs PPU Box Compiler Boundary

Use this boundary for layout/type oracles that compile CUTLASS/CuTe with
NVIDIA `nvcc` plus `dev/fold_derivation/stub_inc` but do not execute a PPU
kernel.

## Repeated failure signature

On the PPU box, the executable named `nvcc` is not evidence that this fixture
is runnable.  It can delegate device preprocessing to `ppu_clang++`, which
defines `__HGGCCC_VER_MAJOR__`.  That activates `PPU_FP8_ENABLED` in
`third_party/actlize/include/cutlass/float8.h`, followed by:

```text
fatal error: hggc_fp8.h: No such file or directory
```

The same oracle passes on a normal CUDA machine because NVIDIA `nvcc` does not
define that HGGCC version macro, so the PPU fp8 bridge is not selected.

## Hard rules

- Never add `hggc_fp8.h` to `stub_inc`.  Stub include directories precede SDK
  directories and would shadow the real PPU SDK type declarations.
- Never decide from `which nvcc` or `nvcc --version`.  Compile a probe that
  includes `cuda_fp16.h` and uses `threadIdx`, `blockIdx`, and `__hadd`.
- If that probe fails, classify the local oracle as `SKIP`/unsupported.  Do not
  mix repository stubs with selected real SDK headers to force it through.
- Generate exact oracle evidence on a complete NVIDIA CUDA toolchain, commit
  its canonical output, and bind it to the Git SHA.
- On the PPU box, consume that committed evidence through a fail-close checker
  using `git show "$sha:<evidence>"`.  Print `fresh_box_execution=0`.
- The committed host oracle never replaces device admission: build the
  shipping target fresh with `hgcc`, whose command must carry both the PPU SDK
  top-level include and `targets/<triple>/include` directories.

## New-runner checklist

Before publishing a box runner that references a host CuTe oracle:

1. Add the real compiler-capability probe to the standalone oracle runner so a
   direct invocation returns an explicit skip instead of the fp8 include error.
2. Add a canonical `.expected.txt` file and a checker with positive denominator
   and planted-negative validation.
3. Make the box runner reject direct execution of the host oracle and consume
   only evidence from its own result SHA.
4. Keep the fresh `hgcc` build and raw-bit device closure after that evidence
   check.

Existing examples are `ci/local_gates.py::_sdk_target_includes`,
`ci/check_l208_q8_committed_evidence.py`, and the L210 evidence seam in
`tools/run_scalefirst_q4k_real_shapes_pruned_box.sh`.
