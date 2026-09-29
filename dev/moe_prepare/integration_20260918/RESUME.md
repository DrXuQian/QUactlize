# Resume

State: patch integrated, compiled and packaged on dev/gemv-model-tuning.
Production selector bounds, arithmetic and caller remain unchanged.
Baseline and acceptance checks: docs/plan.md.
268 related host/runner tests PASS;40 actual-router host comparisons and a rejected
unstable-tie mutation. Execution build179.327s; seven GEMV native opcode and
resource comparisons PASS, zero stack. Candidate PPU device admission pending.
Execution: /tmp/moe-prepare-execution-20260918.
Device gate: /tmp/moe-prepare-gate-r2-20260918 (immutable pre-patch control).
Assembled runtime: /tmp/moe-prepare-runtime-20260918.
Runtime: artifacts/kpack-model-prepare-v1 at87b996559ff64d0fd17123ab8402617075cf613a.
Manifest:0bcf7ebf01ed534ea47fd73ca2dc384ac998ca4ef07f970cb8186267190aa624.
59 binary payloads retained byte-for-byte; four unique new binaries44,992,056B.
Use tools/run_kpack_q4_model_box.sh with MODEL_GATE_UP=1 MODEL_COMPUTE=bf16
MODEL_PHASES=perf MODEL_NAMES=qwen35-35b-q4km, reusing the existing caller build.
Do not overwrite
or performance-only resume the old runtime's results for this changed helper.
