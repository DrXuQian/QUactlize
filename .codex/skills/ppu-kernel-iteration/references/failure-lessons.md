# PPU failure lessons

Use the category that matches the observation; these are not reasons to add
every diagnostic to every kernel.

| Observation | First discriminating check | Do not conclude |
|---|---|---|
| Outputs remain poison; sync returns success | Clear/check prior status, immediate launch status, then sync/deferred status; run an existing marker in the same carrier | Decode formula is wrong just because output is zero |
| Immediate status200 / InvalidKernelImage, including marker | Inspect SDK/compiler/runtime identity and native image admission; compare known runnable carrier | Header-defined kernel, aggregate ABI or CuTe must be broken |
| Failure appears after adding F32 endpoints | Inspect first nonfinite tensor and exact activation/dequant precision boundaries | F32-to-F16 cannot overflow, or clipping is harmless |
| Changing-input replay intermittently wrong | H2D must be ordered on the consumer stream and host memory kept alive | Passing once excludes a race |
| Error at second M8 tile, N-column stripes | Enumerate actual epilogue ownership; first CTA must not write next tile's row | Rebase the A pointer without evidence |
| Profiler checker rejects a new implementation | Import saved reports; match exact producer, geometry and required reducer for that call | All symbols observed in a whole request must occur in one M1 profile |
| Query says JIT source differs | Rebuild the small dispatcher against the candidate source and refresh the package | Disable hashes or point an old manifest at a replacement DSO |
| Benchmark is days rather than hours | Measure wall-time fixture/build/child overhead and prune a declared inventory | Kernel duration times the cell count predicts campaign duration |

SDK environment changes once caused even standalone markers and dequant to
fail image admission. A later machine/environment passed the same production
header bodies. Never infer a code fix from a machine switch alone.

An allocated MoE graph reused dead router logits storage. Whole-chain alias
rejection blocked most fused prepares; the bounded M1 fix relies on a register
snapshot before any store. This does not authorize generic overlap or deleting
live-input guards in multi-token cases.

Use fresh child processes after runtime/launch errors. Keep failed logs and
mark their timings invalid. Preserve unrelated valid experiments. If the same
environment failure recurs after targeted checks, report it and continue safe
host analysis; do not reset devices, kill another workload or rewrite the SDK.

Keep diagnostics small: the exact first bad cell, immediate runtime code,
source/binary identity, and saved detailed logs are more useful than thousands
of repeated warnings. A CPU/profiler receipt is not device admission.
