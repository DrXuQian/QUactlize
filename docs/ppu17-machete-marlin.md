# H800 Machete / Marlin comparison — target remains open

Goal: outperform **both** implementations at the declared comparable shapes,
not merely improve the initial Quactlize backend. No default production route
was changed. PPU1.0/1.5 remain on the existing actlize backend.

The experiment, source pins and reproduction commands are in
[the reference harness](../tests/ppu17/reference/README.md). This compares
resident prepared GPU calls on a physical H800 PCIe: FP16/BF16 input/output,
FP32 accumulation, complete reduction/reset costs, rotating weights exceeding
2.25x50MiB L2, >=250ms warmup and stable1755MHz SM/1593MHz memory clocks.
The logical group64 affine fixture is a common subset, not arbitrary GGUF
group32 support in Machete. Every output passed the typed CPU dot oracle.

## Completed steady-state screen

Twelve points and240 configuration measurements are complete. Numbers below
are microseconds, using each implementation's measured fastest configuration.
Quactlize here uses the typed-RN, four-warp cached decode candidate. Large M
still uses the initial TMA/WGMMA baseline. This is not winner admission.

| A/C | M | N | K | Machete | Marlin | Quactlize |
|---|---:|---:|---:|---:|---:|---:|
| FP16 |1|512|2048|11.75|12.08|13.88|
| FP16 |8|512|2048|11.94|12.02|14.51|
| FP16 |128|512|2048|12.71|18.10|234.50|
| FP16 |1|4096|4096|16.39|12.03|29.55|
| FP16 |8|4096|4096|16.67|11.97|100.36|
| FP16 |128|4096|4096|24.27|29.66|623.46|
| BF16 |1|512|2048|11.94|12.12|13.69|
| BF16 |8|512|2048|11.85|12.06|15.16|
| BF16 |128|512|2048|12.70|18.90|253.16|
| BF16 |1|4096|4096|16.65|11.69|29.70|
| BF16 |8|4096|4096|16.67|11.65|109.02|
| BF16 |128|4096|4096|24.27|30.87|645.37|

The initial M1/N512/K2048 FP16 baseline was164.35us. Caching whole packed
words/affine metadata reduced it to20.56us, then typed native MUL/ADD and
input-storage specialization reduced it to13.88us. Both typed rounding
boundaries remain explicit; neither FP16 accumulation nor affine factoring
after the dot was used. The large improvement does **not** meet the goal.

Artifacts:
`/workspace/quactlize-ppu17-20260926/reference-ab/results/steady-cohort`.
Library build: `typed-rn-build/893b0e62359a1dd07a19` under that task root.

## Last bounded candidate

Two16-warp occupancy challenges retain the same N4/N16 word mapping and
canonical bytes. Their library is
`warp16-build/6ecd11b3ece7937bb493` under the task root.

- 216 compiled specializations pass host/ABI, source-target and CUDA SM90a
  codegen checks. Actual PPU1.7 native compilation is still not admitted.
- 220/220 physical numerical cases pass: M1..8, distinct group32 metadata,
  N257 tails, ragged/empty experts, FQ/SF, FP16/BF16 compute and three A/C
  storage types.
- The memory/synchronization gate stopped after2/30 passing cases because
  another GPU workload appeared. There is no reported candidate memory
  failure, but the remaining28 cases and its timing reconfirmation are pending.
- No performance claim is made for the16-warp candidates. They are explicit
  experimental inventory, not a default heuristic.

## Next work, in order

1. Complete the30-case sanitizer gate in an idle H800 window, then use
   `reference/reconfirm.py` against the measured reference leaders. Keep all
   regressions; do not select a convenient reference configuration.
2. For M1, use the occupancy challenge to decide whether the critical issue
   is insufficient independent K work or remaining metadata/load overhead.
   For M2..8, add reuse of each B fragment across rows: the current per-row
   SIMT decomposition rereads weights and cannot be the final large-N design.
3. Prefill needs a real mainloop change. The baseline serializes raw load,
   dequantized shared-memory staging, WGMMA, wait and refill. Adapt a
   warp-specialized pipeline with multiple in-flight stages; investigate
   transposed RS WGMMA to keep quantized B conversion in registers, as in
   Machete, while deriving the reader from the canonical K-pack addresses.
   This must remain isolated from the actlize include graph and ABI.
4. Validate register ownership, group16/32 metadata lifetime and typed
   rounding before timing. Expand to model shapes and grouped operations
   only after the dense comparison is admitted. H800 performance is not
   proof of native PPU1.7 performance or correctness.

NCU hardware counters were unavailable under the current driver permissions.
Address-derived coalescing counts are hypotheses, not measured bandwidth.
Early short-warmup screens and interrupted runs are retained for diagnosis,
but the table above uses only the completed steady-clock cohort.
