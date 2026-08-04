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

2. **THE SWEEP'S PROBLEM SHAPES. Everything measured so far was N=K=4096, which is none of the three targets.**
   The user fixed the workload on 2026-08-04: Qwen3-32B (dense), Qwen3.5-35B-A3B (MoE), Qwen3.5-122B-A10B
   (MoE, 2-card TP), at n-token in {1,2,4,64,2048,4096}. Shapes are deliberately NOT written into
   benchmarks/workloads.py until they come off the checkpoints -- a plausible 5120 in a file reads like a record.

       cd /sim/eec/shared/junfu.qx/quactlize && git pull --recurse-submodules
       python3 tools/inspect_models.py gguf /sim/eec/shared/AI_workspace/llm-models/Qwen3.5-35B-A3B-Q4_K_M-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf
       python3 tools/inspect_models.py gptq /sim/eec/shared/AI_workspace/llm-models/Qwen3.5-122B-A10B-GPTQ-Int4/ow7_224_ca
       ls /sim/eec/shared/AI_workspace/llm-models/ | grep -i 32b    # Qwen3-32B not yet located

   WANTED per model: hidden_size, intermediate_size, moe_intermediate_size, num_experts, num_experts_per_tok,
   and the distinct per-layer (n, k). For the 122B ALSO the serving TP convention -- which axis the 2-card split
   halves -- because both conventions exist and assuming one tunes a GEMM that never runs.

3. **FIRST FEEL OF THE SWEEP -- i4 (Q4_K) MoE, real Qwen shapes, one compile wave (~20 s).** The user asked for
   the single-schema slice to get a feel. This is the MoE half; the dense half is not ready (see the caveat).

       cd /sim/eec/shared/junfu.qx/quactlize && git pull --recurse-submodules
       MOE_FORMATS="i4" TARGET=test_lowbit_moe_bench ./build.sh
       BIN=$(find third_party/actlize/build_w4a16_compare -type f -name test_lowbit_moe_bench -perm -u+x -print -quit)
       test -n "$BIN"

       # Qwen3.5-35B-A3B expert FFN, gs=32, ragged (mode 2 is the default and IS the skewed generator).
       # args: L(experts) Rows(avg rows/expert) N K gs mode
       echo "== 35B-A3B expert_gate/up  n=512 k=2048 =="
       for R in 128 64 2; do "$BIN" 256 $R 512 2048 32 2 | tail -20; done
       echo "== 35B-A3B expert_down     n=2048 k=512 =="
       for R in 128 64 2; do "$BIN" 256 $R 2048 512 32 2 | tail -20; done
       echo "== 122B-A10B after TP=2    n=512 k=3072 / n=3072 k=512 =="
       for R in 128 64 2; do "$BIN" 256 $R 512 3072 32 2 | tail -20; done
       for R in 128 64 2; do "$BIN" 256 $R 3072 512 32 2 | tail -20; done
       echo "== DECODE: only M*8 experts are touched, ONE row each =="
       for E in 8 16 32; do "$BIN" $E 1 512 2048 32 2 | tail -20; done

   Rows comes from n-token: rows/expert = M*topk/experts = M/32 for both models, so R=128 is M=4096, R=64 is
   M=2048, R=2 is M=64. M in {1,2,4} does not reduce Rows below 1 -- it reduces the number of experts TOUCHED,
   which is why the last loop varies L instead.

   180 units, one compile wave at MOE_CORES=192; the file's own measurement is 14-19 s wall clock, and codegen
   is 5.6% of it. Stage counts are inside a unit, so they cost nothing extra.

   WANTED: the per-format best lines. **AND READ THEM AS A LANDSCAPE, NOT A VERDICT** -- see the caveat below.

   CAVEAT, and it is the reason this is "a feel" and not the sweep: the winner is still selected by a SINGLE
   timing (`if (tf > best_tf)`), and the recorded cross-run spread is 13%. Any two configs within 13% of each
   other are being ordered by noise. Use this run to see the shape of the landscape and whether anything is
   grossly off; do not quote a winner from it. The repeat/confidence-band harness is in progress.

   THE DENSE HALF IS NOT READY. bench_cutlass_w4a16.cu carries 17 hand-written configs plus a hand-written
   dispatch macro with one line per config; the pruned i4 set is 110 shapes. Generating that list is the next
   piece of work and it is mine, not codex's.


**WHEN SOMETHING IS RED, PASTE THIS.** No arguments, safe any time, output sized for a chat message:

    ./tools/failures.sh

It reads every log run_batch wrote and prints, per failing one: the summary, the failed test ids, and the
ASSERTION TEXT. That last part is what has been missing -- twice today a ppu001 round trip was spent asking for
a message that was already in the log, once three lines above where anyone looked.


## SETTLED

- **The two entries below were answered by the 60/60 run and are kept only so the ask is not re-issued.**
  All five formats passed and the nine PARTIAL cells were promoted on that evidence (`f52cc82`). An OPEN list
  that still contains answered items is an OPEN list nobody trusts, which is how a real request gets skipped.

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
