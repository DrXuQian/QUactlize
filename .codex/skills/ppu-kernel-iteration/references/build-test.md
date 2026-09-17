# Build and test the candidate, not a stale package

Run from an isolated Quactlize worktree. Commands below use explicitly assigned
paths; SDK points at the PPU SDK root, not only its `CUDA_SDK` subdirectory.
Confirm `bin/hgcc`, `bin/hgobjdump`, runtime libraries and `asight/bin/acu`.
Keep inherited SDK/module setup. No framework-wide Conda installation is needed.

## Small production execution library

```bash
python3 tools/build_kpack_execution.py \
  --sdk "$PPU_SDK" --output "$TASK/runs/execution-r1" --jobs 192
python3 tools/check_kpack_model_decode_updates.py \
  --sdk "$PPU_SDK" \
  --library "$TASK/runs/execution-r1/libquactlize_ppu_execution.so" \
  --output "$TASK/outputs/production-numeric-r1.json"
```

The output directory must be new. This builds the execution library, not the
large TC sweep. Actual parallelism is bounded by translation-unit count and
memory;192 jobs is a limit, not proof that192 cores are busy. For each candidate
prefer the smallest existing development build that instantiates its body.
The production gate covers three measured recipes plus M2 controls, not every
new config or format. Extend the independent oracle to the actual new scope.

## Compose a model package only after a kernel win

`BASE_RUNTIME` is the already verified published model package. Do not treat
an arbitrary `build/` directory as that package.

```bash
python3 tools/refresh_kpack_model_execution.py \
  --sdk "$PPU_SDK" --base "$BASE_RUNTIME" \
  --execution "$TASK/runs/execution-r1" --output "$TASK/outputs/runtime-r1"
python3 tools/verify_kpack_dispatch.py "$TASK/outputs/runtime-r1" --sdk "$PPU_SDK"
python3 tools/run_kpack_decode_updates.py \
  --sdk "$PPU_SDK" --bundle "$TASK/outputs/runtime-r1" \
  --output "$TASK/outputs/q8-production-gate-r1"
```

The refresh reuses unchanged TC modules, packer and prefill payloads. Source or
SDK incompatibility is an error to resolve, not a check to remove. Never copy a
new execution `.so` over the old package without regenerating its manifest.

## Bounded experiment entrypoints

- `dev/gemv_simt/build.py` / `run.py`: all-format explicit-config experiments;
  inspect `--help`, precision, ring and inventory before launching.
- `dev/gemv_simt/build_model_followup.py` / `model_followup.py`: historical
  five-point experiment. Its old recipe/identity cannot stand in for today's
  admitted baseline. Rebase the experiment contract explicitly if reusing it.
- `dev/gemv_simt/build_q8_topology.py` / `run_q8_topology.py`: Q8 topology work.
  Keep S1 and full Split-K calls, including the correct reducer.
- `dev/gemv_ppu/run_cold_shapes.py`: Q4 cold parity history. Reuse independent
  fixture/checking logic, not an unverified historical winner label.

For ACU use the installed SDK, not NCU command/metric names by substitution:

```bash
"$PPU_SDK/asight/bin/acu" --import "$REPORT" --page raw --csv
```

Find available metrics and definitions in the installed tool/section files.
Normal rotating event timing and profiler-forced-cold replay are separate
experiments. Keep original reports so parser repairs need no GPU rerun.

## Caller integration is a later, separately authorized step

If only a compatible execution library changed, the existing caller can be
reused. If caller code or ABI changed, compile through llama.cpp's
`.aoneci/scripts/build.sh`, normally via `tools/build_kpack_model_ci.py`.
Use its `--local-llama`, `--reuse-llama-build` and `--reuse-ncp-build` options
after inspecting `--help`. The NCP reuse path is the checkout with `build/`,
not the `build/` directory itself. Do not put llama binaries in Quactlize.

To test a local candidate runtime, use `tools/run_kpack_batched_bench.py`
with an explicit `--bundle`, the INT4/PP2048 plan, `--order abba`, and
`--require-selected`. The published one-key `run_kpack_q4_model_box.sh` fetches
the pinned artifact; it does not automatically use your locally built library.

Keep native and candidate precision, model, token count and request batch
identical. Exclude the first complete warmup/JIT pass in each process. Collect
Asys separately through `tools/run_kpack_model_validation.py --phase trace`,
then reproduce exact selected operators with `tools/profile_kpack_model_decode.py`.
Use `--recheck-existing` for a host-only review of saved ACU reports.

Component gates, kernel full-call performance, whole-model numerics and warmed
TPOT are four different deliverables. Passing one must not relabel the others.
