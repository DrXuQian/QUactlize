# Selected prefill calls and standalone Split-K reduction

These are two independent, resumable experiments. Neither changes production
selection nor performs online tuning. The package contains the actual C++
dispatcher and its 12 selected parent modules, plus a small carrier of the
production reducer types. Box execution compiles nothing.

## 1. Complete heuristic-selected FQ/SF calls

`bash tools/run_kpack_prefill_measurement_ppu_box.sh gemm`

The denominator is the same 44 Q4/Q5 workload/token points as the
[component board](KPACK_PREFILL_COMPONENT_COSTS.md): tokens 2048/4096, five
dense and six E256/top8 grouped families. Both FQ and SF are measured, so
there are **88 measurements**. Current policy selects **S1 for all 88**.
This answers how the deployed heuristic performs, not whether every possible
configuration has been exhausted or a newly measured global optimum proved.

- Each request goes through the production dispatcher, including its grouped
  `max_rows=tokens` bound. The actual router is identical to the BF16 study.
- All required canonical low/high/unit bytes, SF plane bytes and independent
  BF16 weight hashes match sealed receipts from the previous dequant run.
  Activations use the same BF16-exact coefficients, represented in FP16.
  Numerical output is checked against official GGUF FP32 weights, with the
  existing conditioned error limit 0.005, not against the kernel itself.
- The timed call includes any actual Split-K completion and internal GPU
  directory/metadata work. SF scale/zero is resident and never expanded in the
  timed graph. Existing separately measured SF expansion time can be added
  once. External gather/scatter and precision adapters remain excluded.
- Weight input addresses rotate through a complete ring larger than 2.25x
  L2. Every graph traverses it twice. Setup, prepare and first graph replay
  are excluded. Three rounds of five GPU-event samples are recorded.
- Every output is checked, with zero-input, changed-input/graph, NaN poison
  and guard controls. No old dequant, cuBLAS or DeepGEMM sweep is repeated.

## 2. Standalone production reducers

`bash tools/run_kpack_prefill_measurement_ppu_box.sh reducer`

The same output shapes at S2/S4/S8 deduplicate to **48 points** by
`(dense/grouped, total M, N, S)`. Qtype and K do not affect this operation.
These are diagnostic S>1 costs, not forced choices for the S1 prefill calls.

The carrier includes, rather than rewrites, the production header:

| Route | Reducer |
|---|---|
| Dense FQ/SF | `PpuMixedInputSplitKParallelM1FastReduction<2>` |
| Grouped FQ/SF | `PpuMixedInputSplitKParallelCompactReduction<2>` |

Inputs are FP32 `[S,M,N]` partial planes, added in increasing S order and
converted to FP16 once. Dense uses an allocator-aligned workspace; grouped
also tests the weaker 16-byte alignment of its real workspace. All outputs
are checked bitwise, allowing equivalent signed zero, against deterministic
FP32-exact partials. Zeroing one split plane must make the oracle fail.

Partial buffers rotate through a complete ring larger than 2.25x L2; setup
and uploads are untimed. This is **reducer-only with synthetic partials**,
not the cache state immediately after an actual producer. Do not add this
time to a full-output GEMM time that already includes reduction, or claim it
is an exact attribution of that producer-consumer chain. No routing, gather,
scatter or dequantization is executed in this experiment.

## Execution and resume

Use the same idle physical PPU as the previous component measurements. The
runner retains the SDK environment and verifies the runtime libraries and
device receipt. Default SDK: `/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK`.
It selects Python before SDK setup, which may otherwise shadow the installed
NumPy/GGUF environment. No profiler or first-use JIT time enters samples.

Each command prints its result directory and a small `*.results.tgz` archive.
Send both archives. They contain `summary.tsv`, per-case JSON, logs, package
and input identities. Failed cases remain explicit; successful cells survive
unrelated failures. For the same checkout/package/environment:

```bash
RESUME_RUN=/workspace/kpack-prefill-gemm.PRINTED_SUFFIX \
  bash tools/run_kpack_prefill_measurement_ppu_box.sh gemm
```

Use the directory actually printed by the runner, not this placeholder.
Resume skips existing complete cells. A changed measurement protocol requires
a new result directory. Both commands preserve the enclosing Docker shell
on failure. Reported remaining time is an observed case-average estimate,
not a promised full-run duration.

The 12 parent modules compiled locally in about four minutes. Local tests
cover policy closure, independent fixture hashes, oracle negatives, result
validation, shell parsing and parent-shell survival. PPU numerical/timing
admission is still the purpose of these two box runs.
