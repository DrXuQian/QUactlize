# Quactlize documentation

Use `develop` for current integration. Start with the repository
[README](../README.md) and [CI / manual tests](CI.md).

## Product and integration

- [Selection, buckets and prewarm](KPACK_SELECTION.md)
- [Dynamic execution ABI](KPACK_DYNAMIC_EXECUTION.md)
- [Dense prefill route selection and evidence](DENSE_PREFILL_ROUTING.md)
- [Dense dequant and conversion helpers](PPU_PREFILL_DENSE_PRODUCTION.md)
- [Latest helper admission and final model validation](PPU_DENSE_CONTIGUOUS_IO.md)
- [Branch and build policy](BRANCHES.md)
- [Isolated PPU1.7 backend](../quactlize/backends/ppu17/README.md)

Build with `bash build.sh --library`; test with
`bash run_tests.sh --suite test --package PATH`. No additional CI build
script, model download or sweep is required. See CI.md for prerequisites
and the distinction between host checks and real PPU accuracy.

## Performance work

- [Model-shape Asys sweep](PPU_PREFILL_ASYS_SWEEP.md)
- `tools/run_native_components_box.sh`: isolated native/candidate components.
- `tools/run_dense_dequant_box.sh`: retained typed dequant helpers.
- `docs/measurements/`: source-bound historical evidence, not current commands.

Earlier experimental implementations and commands are recoverable from Git;
they are not installation instructions. The old W4A16 actlize/Marlin overview
formerly at this path is preserved at commit `618298d6f8761b8eb4489ad2f1a217dd57f0bd47`.
