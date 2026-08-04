# Commands that need ppu001, batched

Neither Claude nor codex can reach the box. Everything here is for the user to run in one go; each entry says
exactly what output settles the question, so a partial paste is still useful.

## OPEN

0. **TWO THINGS, ONE PULL.** (a) the sixth per-format oracle is newer than the last run -- expect
   `6 passed, 24 skipped` instead of 5/20; (b) the packer now exists and can be pointed at a real checkpoint.

       cd /sim/eec/shared/junfu.qx/quactlize && git pull --recurse-submodules
       ./benchmarks/run_batch.sh pytest && ./tools/failures.sh

       SO=$(find third_party/actlize/build_w4a16_compare -name libquactlize_ppu.so | head -1)
       python3 tools/pack_gguf.py --dry-run \
         /sim/eec/shared/AI_workspace/llm-models/Qwen3.5-35B-A3B-Q4_K_M-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf /tmp/bc

   --dry-run touches no device and writes nothing. WANTED from it: the `type mix` line. It answers the question
   that has been open all session -- whether a _K_M checkpoint really carries more than one k-quant, and so how
   many format-specific libraries a deployment of it needs. Drop --dry-run (with QUACTLIZE_PPU_LIB=$SO) to
   actually pack.


**WHEN SOMETHING IS RED, PASTE THIS.** No arguments, safe any time, output sized for a chat message:

    ./tools/failures.sh

It reads every log run_batch wrote and prints, per failing one: the summary, the failed test ids, and the
ASSERTION TEXT. That last part is what has been missing -- twice today a ppu001 round trip was spent asking for
a message that was already in the log, once three lines above where anyone looked.


0. **UNBLOCKS NINE MATRIX CELLS. One paste.** The run already happened and all five formats passed; codex is
   holding the promotion because the condition was "read the logs yourself" and those files exist only on the
   box. It is right to hold -- a relayed screenshot is weaker evidence than the file -- so the file is what is
   needed.

       cd /sim/eec/shared/junfu.qx/quactlize && tail -3 ~/ab/fully_quantized_*.log

   WANTED: the summary line from each of the five. Expected shape per format: `2 passed, 8 skipped` with the
   skips reading "this library is built for packed format N". If any says something else, the promotion is
   withdrawn rather than adjusted.


1. **THE ONLY THING BETWEEN 51/60 AND 60/60.** All five k-quants now have FULLY_QUANTIZED dense AND grouped
   implemented, local gates green on both sides. Nine cells sit at PARTIAL solely because no ppu001 oracle has
   run against them; there is no further code to write for those cells.

       cd /sim/eec/shared/junfu.qx/quactlize && git pull --recurse-submodules && ./benchmarks/run_batch.sh pytest

   It builds the device library FIVE TIMES -- once per PPU_PACKED_FORMAT, because no binary runs more than one
   format -- runs each format's dense and grouped oracle against official gguf with a planted fault required to
   fail FIRST, and restores the default-format library at the end.

   USE run_batch, NOT a bare `pytest tests`. Three runs today produced plausible garbage from a stale host
   extension; a bare pytest now refuses rather than reporting, but only run_batch rebuilds the DEVICE library and
   passes PPU_PACKED_SCALE=1, without which the whole cell answers rc=34.

   WANTED: the per-format summary lines, plus any FAILED/ERROR ids (named now, -rfE).

## SETTLED

- **The whole python tier was green on ppu001 once today**, which closed four things at once: codex's vecdot
  activation-contract fix confirmed ON THE DEVICE, the two stock-CUTLASS stand-in build errors gone, the sixteen
  cpu_reference ERRORs gone, and the two Q2_K failures ("native dense oracle missed planted row-0 reuse",
  "worst nan") shown to be STALE-BUILD ARTEFACTS -- they were never about Q2_K. Refusing to diagnose them before
  the rebuild was the right call.
- `$OUT/dense_python_oracle.log` -> "5 passed, 5115 warnings in 4.81s": five passing dense-oracle cases, zero
  skipped, on ppu001. That is the evidence scale_first/dense was promoted on.
- The FULLY_QUANTIZED build needs PPU_PACKED_SCALE=1 to reach the compile command, and for a while it did not:
  quactlize_ppu was built by cutlass_add_library and so sat outside the wrapper that applies PPU_EXTRA_DEFS.
  Fixed; run_batch now asserts the define reached the compile before running anything.
