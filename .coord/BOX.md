# Commands that need ppu001, batched

Neither Claude nor codex can reach the box. Everything here is for the user to run in one go; each entry says
exactly what output settles the question, so a partial paste is still useful.

## HELD -- do not run yet

H1. **THE M-SWEEP (INBOX 025), and any other tactic sweep.** The user set the ordering on 2026-08-04: **no box
    sweep is requested until the dense AND MoE option sets are BOTH ready and written down.** The reason is not
    politeness about box time. The command below covers **int4 dense only**, and a sweep that covers the options
    which happen to compile today reports a winner over a truncated space -- which reads exactly like a winner
    over the whole space. That is the defect that produced the incomplete Q6 tactic and my own l105 rows that
    classified all-zero buffers.

    WHAT UNBLOCKS IT: codex answering INBOX 029 (dense pins the low plane at F=1 via
    `static_assert(P1_FOLD == 1)`, while a measurement I labelled "dense int1 TK64/F4" says otherwise -- one of
    the two is wrong), and then enumerating the complete legal (bits, TK, tile, warp) set for dense and MoE
    separately, marking what is reachable today. Dispatched. When it lands, this becomes one batched command
    that names its own coverage, and any excluded cell carries its reason in a clause.

    Kept verbatim below so it can be merged rather than rewritten:

0. **ONE DENSE TENSOR ACROSS SEQLEN (INBOX 025).** This is the cheap decisive version: one int4 scale-only
   dense weight at fixed N=K=4096, swept at the requested M values. Every candidate TileK is 64, 128 or 256 and
   therefore consumes the SAME `xp4f1` bytes; a moving tactic winner would not imply a moving resident
   arrangement. One build runs all five shapes.

       cd /sim/eec/shared/junfu.qx/quactlize && git pull --recurse-submodules
       TARGET=test_fpA_intB_ppu ./build.sh
       BIN=$(find third_party/actlize/build_w4a16_compare -type f -name test_fpA_intB_ppu -perm -u+x -print -quit)
       test -n "$BIN"
       for M in 1 8 64 512 2048; do
         "$BIN" "$M" 4096 4096 32 | grep '^  WINNER m='
       done

   WANTED: the five `WINNER` lines. The config name contains TileK. If it changes, the first M with the new
   winner is the measured tactic crossover; regardless, all candidates remain one F=1/TK<=256 layout class, so
   no repack or duplicate tensor follows from the change.

## OPEN -- runnable now

These are correctness and inspection, not tactic sweeps, so the hold above does not apply to them.

1. **TWO THINGS, ONE PULL.** (a) the sixth per-format oracle is newer than the last run -- expect
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
