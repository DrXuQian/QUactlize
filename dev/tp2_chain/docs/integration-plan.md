# Measured paired selection integration

Source baseline: `0d7361eac2a3becdd53aa29b8b84d29cdc481fb8`.
Caller baseline: `1f7da3bd94b1fc6b9520e156456c89221fddd07f`.
Evidence: `result-20260920.md`; preserve its raw archive and immutable package.

## Scope

- Promote only Q4 logical N512/K3072, one token, E256/top8, F32 endpoints,
  BF16 computation and projection rounding, original-slot SIMT down input.
  Use the confirmed C4/W4/P4/S1 reader with Changes=3 and paired-16 finish.
- Preserve the N512/K2048 paired incumbent, existing Q8 choices, M2..8
  choices and all ordinary decode exact/bucket decisions.
- Keep canonical and paired-N4 bytes unchanged. Preserve F32 accumulation,
  BF16 projection rounding before SwiGLU and existing down/finish semantics.
- No new search, arithmetic rewrite, TC mainloop change or model-name rule.

## Steps

1. Freeze host selection snapshots and import reviewed full-chain evidence.
2. Generate paired requests, complete implementation recipes and selection
   rows through the existing decode generator. Keep ordinary decode output
   byte-identical. Separate capability validation from performance selection.
3. Add versioned recipe selection/query/execution and chain binding without
   changing old structs/symbols. Validate the semantic key and consumer contract.
4. Cache the resolved recipe in the existing chain. Port the measured body
   without changing reduction order. Keep old callers compatible.
5. Update the caller's thin API binding, workspace preparation and trace proof.
   Prepare paired planes before capture; no per-token allocation or host wait.
6. Run host regression, narrow PPU compilation, package/source/ABI checks and
   prepare a resumable box handoff on devices 6,7. Local compilation is not
   numerical or model-performance admission.

## Gates

Local: reproduce all current ordinary choices, freeze old paired choices,
reject semantic/layout/recipe drift, preserve old ABI behavior, verify complete
new API groups and selected implementation identity. Reuse existing caller tests.

Box: production-entry numerical controls for tokens 1..8, changing IDs, guards,
BF16 range and graph replay; matched rotating complete-chain A/B on both points;
warmed model A/B and Asys for the 122B TP2 and 35B control. Exclude first JIT,
graph upload and warmup. Preserve independent successful cells on failure.

Performance criterion: retain the best contemporaneously measured complete
chain. New TP2 evidence is 43.366286 -> 39.847999 us; the existing 35B winner
is 30.855499 us. These are component targets, not guaranteed model TPOT.

No elapsed-time promise is inferred from kernel microseconds. Finish local
implementation and validation, then hand off the device-only checks.
