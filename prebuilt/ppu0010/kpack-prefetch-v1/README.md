# Cache-prefetch probe (PPU 1.0)

This 29 KiB auxiliary library is experimental, not a production dependency.
It reuses the two already-admitted Q4/S4 and Q5/S1 modules named in the
manifest. No GEMM module is rebuilt. All three auxiliary kernels use zero
shared memory; device execution and performance remain unvalidated.

The manifest's per-file SDK receipt expands the original combined digest;
the helper and GEMM binaries are unchanged. Prebuilt execution permits different
build tools when runtime libraries match. An explicit `--allow-unverified-sdk`
option allows an experimental runtime mismatch, records it, and keeps all
launch, payload and numerical checks mandatory.

Run `bash tools/run_kpack_prefetch_box.sh` from the repository after setting
`PPU_SDK` and selecting one idle device with `CUDA_VISIBLE_DEVICES`.
The box does not compile. See the [protocol](../../../docs/KPACK_PREFETCH_EXPERIMENT.md)
for cache pressure, separate benefit/interference timing and profiler caveats.
