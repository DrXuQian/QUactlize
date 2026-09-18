# Resume

State: patch integrated and compiled on dev/gemv-model-tuning; packaging next.
Production selector bounds, arithmetic and caller remain unchanged.
Baseline and acceptance checks: docs/plan.md.
105 related host tests PASS;40 actual-router host comparisons and a rejected
unstable-tie mutation. Execution build179.327s; seven GEMV native opcode and
resource comparisons PASS, zero stack. Candidate PPU device admission pending.
Execution: /tmp/moe-prepare-execution-20260918.
Device gate: /tmp/moe-prepare-gate-r2-20260918 (immutable pre-patch control).
Assembled runtime: /tmp/moe-prepare-runtime-20260918.
Whole-model source pin and command will follow publication. Do not overwrite
or performance-only resume the old runtime's results for this changed helper.
