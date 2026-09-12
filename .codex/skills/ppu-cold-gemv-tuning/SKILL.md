---
name: ppu-cold-gemv-tuning
description: Design and review PPU GEMV performance experiments that compare K-pack, Xplane, and raw-GGUF readers under controlled cache conditions. Use for cold-weight parity tuning, interpreting ACU reports, or extending an admitted thread mapping across shapes; not for general product cleanup or a numeric bug investigation.
---

# PPU cold-weight GEMV tuning

Keep the canonical offline format fixed while finding a measured reader and
thread mapping. Separate a useful implementation result from a pure layout
comparison and from production admission.

Read [the C4/C8 case](references/q4-c4-c8-20260912.md) when diagnosing a
K-pack/Xplane gap, vector-memory pressure, or transferring the Q4 C8 result.
It contains evidence, not a universal C8 policy.

## Scope and comparison contract

- Resolve the requested axes independently: shapes, qtypes, M, dense/grouped,
  cache regime, and implementation/configuration. “All shapes” does not mean
  all formats, all M, or a Cartesian-product search. Reuse the declared shape
  registry and print the exact denominator.
- If the user has chosen cold weights as the acceptance regime, do not use a
  warm-cache win to close a cold regression. A rotating ring controls weight
  residency; it does not prove every other input is cold or reproduce a model.
- Preserve the actual per-shape best baseline, including its config and
  binary. Do not substitute one global recipe. Reuse old selected recipes,
  but measure all comparison times again in the new cohort.
- Name arithmetic separately from storage: input type, dequant rounding,
  accumulator/output type, and reduction order. FP32 accumulation alone does
  not make per-weight FP16 reconstruction and FP32 group-affine arithmetic
  identical. Do not require cross-arm raw-bit equality when the legal
  floating-point operation order changes.
- Keep one-kernel S1, inter-CTA Split-K plus reducer, and preprocessing outside
  the kernel distinguishable. Compare full declared call scope, not a faster
  producer that omits necessary work.

## Evidence before interpretation

Bind returned data to source, DSO, fixture, physical device, runtime and config.
Check complete/unique rows, finite samples, recomputed medians, independent
GGUF correctness, negative controls, guards, and report/log hashes. A failed
numeric cell has no usable timing. Preserve valid independent cells and
retry only failed cells in fresh processes; a poisoned GPU context must not
contaminate later arms.

Use a verified L2 capacity rather than an ABI-mismatched properties struct or
an assumed GPU name. The existing PPU harness uses a ring at least 2.25 times
that capacity and complete ring traversals. Keep this setting fixed across
arms unless deliberately testing the cache policy itself. Exclude setup,
JIT, graph upload and first-use warmup from resident timing. State when an
environment was not proven idle.

Alternate arm order and retain repeat distributions. Six rounds of fifteen
samples are the current bounded Q4 confirmation protocol, not a required
setting for every experiment. Estimate expanded runs from measured wall
time, including fixture/child/profiler overhead, not kernel time alone.

## ACU analysis

Import the actual reports locally; do not ask the user to transcribe counters
when the files are available. Use the SDK's `acu --import REPORT --page raw
--csv`. A compatible private loader/runtime can be used for host report
import without changing the system libraries or requiring a GPU.

Distinguish rotating event timing from forced-cold profiler replay. Do not
replace one with the other. Check the exact kernel symbol and launch geometry.

Compare at least:

- DRAM bytes versus L1/L2 traffic and transactions;
- vector load instructions and issued-memory pressure;
- registers, shared memory, achieved occupancy and barriers;
- bank conflicts and instruction mix when the changed path uses them.

Stall-per-issue ratios are not wall-time percentages. Effective weight
bandwidth (`weight_bytes / event_time`) is a model, not measured DRAM traffic.
Do not add AIU traffic twice when it is already part of a total metric. Read
the installed SDK metric definitions if a sum is ambiguous.

## Choose the next experiment

Write down lane-to-N/K ownership, contiguous request widths, workers, passes,
tail activity, CTA count and reduction ownership before changing code.
Look for excess on-chip transactions even when DRAM bytes are already near
the minimum. Higher occupancy is not intrinsically better if it increases
memory-issue pressure. Conversely, fewer CTAs do not intrinsically help.

Prefer a small change whose body, decoder and offline bytes can be compared
with an untouched baseline. Shared staging, AIU or asynchronous copy can
remove global requests while adding barriers, shared-memory conflicts or
instruction overhead; retain them only after a full-scope measured gain.
NVIDIA results guide hypotheses, but PPU lowering and counters decide PPU
acceptance. Do not assume a transplanted winner stays optimal.

For the user's current tolerance, a complete shape passes when its selected
K-pack time is at most 1.05 times **each** required contemporaneous control.
Report missing data separately from measured regressions, and retain the old
K-pack winner when the new mapping loses. A single passing shape does not
authorize a global selector change or admission of untested M/grouped/qtypes.

Keep bounded experiment libraries isolated from shipping selection. A box
handoff should execute an already built, hash-verified payload, retain
successful rows on failure, preserve the caller's Docker shell, and package
the summary plus raw evidence. This skill does not grant new push, remote
execution, or production-mutation authority.
