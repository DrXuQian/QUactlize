# Commands that need ppu001, batched

Neither Claude nor codex can reach the box. Everything here is for the user to run in one go; each entry says
exactly what output settles the question, so a partial paste is still useful.

## OPEN

(nothing open -- the tier is green on ppu001 as of 2026-08-03)



- `$OUT/dense_python_oracle.log` -> "5 passed, 5115 warnings in 4.81s" (2026-08-03). Five passing dense-oracle
  cases on ppu001, zero skipped. This is the evidence for promoting scale_first/dense.

2. THE LAST RUN WAS AGAINST A STALE EXTENSION -- nothing in it is evidence.

   50023d4 changed gguf_prepass_ops.cpp / ppu_backend.cpp / ppu_backend.h; the box pulled them but the host
   extension was not rebuilt, so the whole tier ran on the previous binary. run_batch now detects this and
   rebuilds automatically (commit above), so a plain re-run is enough:

       cd /sim/eec/shared/junfu.qx/quactlize && git pull && ./benchmarks/run_batch.sh pytest

   WANTED: the two pass summaries, plus any FAILED/ERROR ids.
   TWO Q2_K FAILURES ARE PENDING THIS and must not be diagnosed before it:
     - test_device_decode_routes... "Q2_K: native dense oracle missed planted row-0 reuse"
     - test_the_two_routes_agree_on_identical_bytes[Q2_K] "worst nan"
   Both may be artefacts of the stale build. If they survive a fresh one, they are real and Q2_K-specific.
