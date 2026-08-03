# Commands that need ppu001, batched

Neither Claude nor codex can reach the box. Everything here is for the user to run in one go; each entry says
exactly what output settles the question, so a partial paste is still useful.

## OPEN

1. **ON THE CRITICAL PATH RIGHT NOW.** codex has the Q4_K packed-dense C++ API done and is blocked on device
   numerical validation, which needs PPU_PACKED_SCALE=1 and therefore needs you. Nothing it does next can be
   called validated until this runs.

       cd /sim/eec/shared/junfu.qx/quactlize && git pull && ./benchmarks/run_batch.sh pytest

   USE run_batch, NOT a bare `pytest tests`. Two of today's runs were against a stale extension and produced
   plausible garbage -- "worst nan" on all five formats and a planted fault going uncaught, both about a .so
   built before the commit under test. run_batch rebuilds; a bare pytest did not. (As of this commit a bare
   pytest REFUSES to run when the extension is stale rather than reporting, so the trap is closed either way.)

   WANTED: the two pass summaries, plus any FAILED/ERROR ids.
   If codex has by then posted the packed-dense route names, the fully_quantized oracles stop skipping and this
   same command validates the new cell.




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
