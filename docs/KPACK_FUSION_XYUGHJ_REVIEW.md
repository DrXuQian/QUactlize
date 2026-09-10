# Focused K-pack model run: XYUgHJ

## Authority and admission

Input: `kpack-fusion.XYUgHJ.results.tgz`, SHA256
`6e5cc56466f9b4db4157c0eb622afe58b8b462822612c02e2d5434331d023776`.
Quactlize source: `c373f766368a825567c017c6cd4a0e38436f045b`.
llama.cpp source: `558e5a06b1f6e7e148873a50ecafd89c74773c3a`.
Device gates report PPU ordinal 0, PCI `0000:08:00.0`.

- Q8 GPU pack/native W8A16 gate: 7/7 pass, no failures.
- Paired GPU pack: all six formats pass (Q8_0, Q2/3/4/5/6_K).
- Selected MoE chain: 4/4 pass (separate/merged gate-up, tokens 1/4,
  router disabled/enabled respectively), including changing-input graph replay.
- llama adapter CTests: 8/8 pass.
- Both models' reference and two K-pack processes exit with rc=0.
- All four K-pack model arms fail **plan-receipt admission**. No Asys capture
  runs, because the wrapper stops after the failed benchmark stage.

Bounded device correctness is established for these fixtures, not full-model
accuracy, fusion coverage, or an end-to-end performance improvement.

## Cause of the missing receipts

All model command receipts omit `--verbosity`. llama's common logger defaults
to 3, but `common_log_default_callback` maps GGML_LOG_INFO to level 4. Both
`[quactlize-plan]` and `native policy miss` use GGML_LOG_INFO. Benchmark JSON
uses level 0 and remains visible. The JIT helper logs directly to its own
output and also remains visible: each 35B K-pack process has 8 cache hits,
each 32B K-pack process has 12, with no selected-plan lines in any of them.

A host probe linked the unmodified llama `common/log.cpp`: the same INFO
plan/fallback messages are absent at level 3 and present at level 4. Benchmark
output is present in both; per-kernel DEBUG remains absent at level 4.
This is a logging/protocol defect, not evidence that dense wiring is absent.
JIT resolution alone does not prove subsequent kernel execution or fusion.

The runner now explicitly requests `--verbosity 4` in both arms and preserves
selection receipts even on admission failure. Empty plans/fallbacks and an
invalid Q8 W8A16 contract still fail. No selector, kernel, JIT source contract,
binary package, or llama C++ changes are needed.

## Provisional raw timings, not admitted K-pack comparisons

PP=2048, TG=128, NPL=1, token batch/ubatch=2048. The first complete pass is
excluded in every process. Reference has one measured sample; the K-pack
column is the median of the two processes' measured samples. These numbers
are retained observations, not substitutes for the missing route receipts.

| Model | Reference prefill, total ms | K-pack-labelled prefill, total ms | Delta | Reference decode, ms/token | K-pack-labelled decode, ms/token | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-35B-A3B Q4_K_M | 636.374 | 241.964 | -61.98% | 7.725 | 10.523 | +36.23% |
| Qwen3-32B Q4_K_M | 663.289 | 817.308 | +23.22% | 33.516 | 29.627 | -11.60% |

The next bounded rerun keeps this workload and existing JIT parents, collects
the missing receipts, and reaches the warmed Asys capture. Inspect 35B decode
for actual small-chain fusion and retained per-node preparation. The 32B
prefill regression also remains open; this logging fix cannot improve either
kernel's performance by itself. Do not reclassify the old run as admitted.
