# Model GEMV reader experiments

Baseline source: `4f181a071ebf3715f90b2898033497342f9af4ca`.
Immutable model artifact: `6a9b89a322f1ccd5cf1b724325294f7d2ae129b1`.
The paired integration measurements are in `docs/GATE_UP_MODEL_REVIEW_20260917.md`.

## Scope and arithmetic

Priority: M1 paired Q4 routed gate/up (logical N512 K2048 E256 top8,
BF16 compute); Q5 routed down (N2048 K512 E256 top8, BF16); paired Q8
shared gate/up (logical N512 K2048, F16); Q8 shared down (N2048 K512,
F16). Extension: Q6 output head (N248320 K2048, F16) and the dense Q8
N2048 K4096, N8192 K2048, N4096 K2048 points, with their current TC or
SIMT complete-call baselines. All use F32 input/output storage and FP32
accumulation. Paired layouts, projection rounding and SwiGLU are unchanged.
No clipping, activation quantization, caller edits or production selection changes.

## Bounded changes

1. Q4 paired: restore the H32 metadata reader; separate the unsigned-index
   and fixed-N/K axes. Compare C4/P8 and C8/P4 register partitions.
   A separate TileN16 finish retains complete G4/U4 pairs and the existing
   sequential inter-warp order; it doubles the CTA count without reformatting.
2. Q8 paired: hoisted vector loads versus the current non-hoisted body;
   compare C4/P8 and C8/P4. Also compare C4/P4 with the isolated half-width
   paired finish (64 CTAs versus the incumbent 32 for shared M1).
3. Q5 down: fixed-N/K and smaller per-thread output windows, retaining H32.
4. Q6 output: load the one required header/scale instead of communicating
   the entire metadata unit. Compare with the existing generic reader and TC.
5. Dense Q8: a bounded neighboring SIMT inventory retaining current winners;
   complete Split-K reductions are timed, not estimated.

## Validation and selection

Host tests cover inventory, source seams, metadata byte addressing, output
pair ownership, affine/rounding contracts and access-sector accounting.
Local compile and ELF/resource inspection precede delivery. Device numerics
use an independent GGUF oracle, changed inputs, guards, graph replay,
zero-input and invalid-ID negatives, and same-geometry clone equality.
M2/M8 are numerical controls, not claims of performance promotion.
Every candidate must pass correctness before timing. Preserve independent
completed points when one fails; isolate points in fresh processes.

Use a verified-L2 rotating active-weight working set >=2.25 L2. Screen
cheaply, then compare finalists and the immutable incumbent in alternating
rounds. Exclude upload/JIT/first replay; include reducers and fused activation.
Report full-call modeled weight MBU at 2700 GB/s separately from ACU DRAM
counters. Small-shape 40% / large-shape 60% MBU are goals, not pass claims.
Target <=5% regression against the current winner before any promotion.
ACU profiling is outside timing, for incumbent and admitted finalist only.

## Handoff boundary

This turn ends with host-tested, locally compiled, immutable experimental
binaries and one box command, not an unmeasured production replacement.
No deadline was requested. Record build wall time; device runtime estimates
must remain advisory until the first point completes.

## Returned result and remaining work

`model-gemv.mZfskR` is complete. The [review](../../../docs/MODEL_GEMV_REVIEW_20260917.md)
recomputes 2160 event samples and re-imports all 16 ACU reports. Seven exact M1
points improve in all six rounds; retain Q6 TC. Numerics cover M1/M2/M8 but
performance admission does not extend to M2..8. No production/caller changed.

- [x] Validate source/runtime/numerical and timing denominators.
- [x] Re-import actual ACU reports, separate DRAM/internal traffic and instruction costs.
- [x] Record measured candidate decisions and failed hypotheses.
- [ ] Integrate only the seven exact M1 candidates; retain fallback scope and BF16 routing.
- [ ] Verify rebuilt integrated kernels, then repeat whole-model timing and Asys.
- [ ] Continue MBU tuning; this round does not meet the 40%/60% objectives.
