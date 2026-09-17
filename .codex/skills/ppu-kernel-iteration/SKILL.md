---
name: ppu-kernel-iteration
description: Run a bounded, resumable PPU kernel optimization task with measured candidate selection, ACU evidence, and a reproducible handoff. Use for overnight GEMV or MoE decode iteration; not for launching an unrestricted sweep or changing production routing without admission.
---

# PPU kernel iteration

Use the KDA loop: contract, plan, candidate, correctness, timing, profiling,
then keep/revise/reject. The agent chooses the next experiment; a shell loop
that repeats benchmarks is not an optimizer.

## Start and resume

- Work in a dedicated worktree and branch at a recorded source commit. Keep
  the admitted binary/configuration immutable. Never edit a shared build tree.
- Write the concrete workload list, arithmetic contract, baselines, validation
  commands, performance criteria and deadline before the first kernel edit.
  Distinguish the priority cohort from optional extensions.
- Maintain `docs/plan.md`, `candidates.jsonl`, `benchmark.csv` and `RESUME.md`
  in the task directory. Record candidate parent, change, hypothesis, commands,
  source/diff/binary hashes, evidence paths and decision. Resume from these
  files and actual processes; do not restart completed valid experiments.
- Read [the current handoff](references/quactlize-baseline-20260917.md) for
  the initial Quactlize task, and [build/test entrypoints](references/build-test.md)
  when compiling or composing an integration package. Historical observations
  are context, not fresh baseline timings or universal selectors.
- Read [failure lessons](references/failure-lessons.md) when an experiment
  stalls at admission, launch, numerics or report parsing.

## Select the right supporting skill

- `../ppu-cold-gemv-tuning/SKILL.md`: mandatory for cold GEMV performance
  decisions, address-pattern analysis and ACU interpretation.
- `../ppu-cute-numeric-debug/SKILL.md`: read when a numerical mismatch appears;
  pause timing of that specialization and isolate its failing seam. Preserve
  already-valid independent cells and run other cells in fresh processes.
- `../ppu-main-productization/SKILL.md`: only when explicitly preparing a
  product/main admission. It is not a prerequisite for an isolated experiment.

## Spend the night on experiments, not infrastructure

1. Reproduce a current baseline and time a minimal compile/test/profile cycle.
2. Rank bottlenecks by useful potential latency reduction, then test a bounded
   inventory that retains every existing per-shape winner. Prefer a single
   causal change to a reader rewrite plus a new scheduler and new arithmetic.
3. Screen cheaply with independent correctness and a few alternating samples.
   Promote only finalists to repeated confirmation and detailed ACU capture.
4. After an unproductive hypothesis, record the evidence and change direction.
   Do not wait for another user message while safe in-scope work remains.
5. Reserve the final hour of a typical 12-hour run for regression checks,
   clean replay and handoff. At the deadline report incomplete targets honestly;
   the deadline is a resource boundary, not a performance verdict.

Compile narrowly; reuse unchanged TC modules and packers. Estimate time from
observed wall time including fixture preparation, process startup and profiler
replay. Do not extrapolate an overnight campaign from microsecond kernel time.
CPU parallel compilation may overlap work on a different GPU, but never
overlap two timed workloads or a profiler with a timed workload on one GPU.
Resolve physical device identity before assigning multiple devices. Reconfirm
final paired baselines on the same device, even for identical GPU models.

## Candidate and promotion boundaries

Keep canonical offline bytes and public ABI fixed. Preserve separate dense
F32 storage/F16 activation compute and MoE BF16 compute contracts where the
task requests them. Do not silently insert F16 into a BF16 decode path, clip
large activations, quantize A to INT8, or weaken the oracle to win a benchmark.

Count the complete selected call, including Split-K reduction. Record helper
and model timing separately. The 40% small / 60% large modeled bandwidth goals
are goals, not evidence that every small launch can attain the bandwidth roof.
Classify shapes before measurement; do not move a losing shape to another bin.
Failure to reach a goal must stay visible.

An experimental winner is not automatically a shipping selector. Propose only
the measured shape/type/M/operator scope. Keep a safe admitted fallback and
require regression coverage before changing an exact table or bucket rule.
Do not merge main, publish replacement binaries, edit the caller, clean other
workspaces, or alter other GPU jobs unless the active task authorizes it.

## Durable handoff

Leave one result table with scope, precision, config, incumbent/candidate/
reference time, full-call MBU, correctness and evidence status. Include rejected
ideas, missing cells, the next highest-value experiment and exact resume/test
commands. Bind raw results to source, SDK/runtime, device and binary identity.
Keep logs/reports/binaries out of source commits; publish artifacts only through
the explicitly authorized artifact path. Never export credentials or raw chat
history as task memory.
