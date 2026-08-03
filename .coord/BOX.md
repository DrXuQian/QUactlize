# Commands that need ppu001, batched

Neither Claude nor codex can reach the box. Everything here is for the user to run in one go; each entry says
exactly what output settles the question, so a partial paste is still useful.

## OPEN

1. Re-run the python tier after `git pull` -- b0273ae should remove the two "In file included from
   gemv_wformat.hpp:35" errors, and 50023d4 should turn the device-vs-CPU-arm failure green.

       cd /sim/eec/shared/junfu.qx/quactlize && git pull && ./benchmarks/run_batch.sh pytest

   WANTED: the two summary lines (one per pass) and, if anything is red, the FAILED/ERROR ids -- these are
   named now that run_batch passes -rfE.
   NOTE: run it through run_batch, not a bare `pytest tests`. A single invocation puts the sixteen
   cpu_reference tests on the device arm, which is what they are marked to avoid, and they report as ERRORs.

## SETTLED

- `$OUT/dense_python_oracle.log` -> "5 passed, 5115 warnings in 4.81s" (2026-08-03). Five passing dense-oracle
  cases on ppu001, zero skipped. This is the evidence for promoting scale_first/dense.
