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
       BIN=$(find build_ppu -type f -name test_fpA_intB_ppu -perm -u+x -print -quit)
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

       SO=$(find build_ppu -name libquactlize_ppu.so | head -1)
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

3. **SUPERSEDED BY ENTRY 4 -- the two gaps named here are closed.** Kept for the reasoning only.
   (was: HELD at the user's request, "还是等你完全支持之后我一个一个跑吧")

   ORIGINAL NOTE: They would rather wait
   for the complete thing than spend a round on a half-ready one, which is the right call -- the two gaps below
   are exactly the kind that make a first look misleading rather than merely incomplete. Kept here because the
   command is correct and only the harness underneath it is not; when the two gaps close this moves to OPEN and
   is run one shape at a time.

   TWO GAPS, BOTH MINE: (a) the winner is chosen by a single timing against a 13% cross-run spread; (b) the
   dense half has 17 hand-written configs where the pruned set needs 110.

   i4 (Q4_K) MoE, real Qwen shapes, one compile wave (~20 s):

       cd /sim/eec/shared/junfu.qx/quactlize && git pull --recurse-submodules
       MOE_FORMATS="i4" TARGET=test_lowbit_moe_bench ./build.sh
       BIN=$(find build_ppu -type f -name test_lowbit_moe_bench -perm -u+x -print -quit)
       test -n "$BIN"

       # PINNED ROUTER fixture; args: L(experts) TOKENS N K gs mode=4 top-k=8.
       # The banner prints its versioned name, active experts, Mmax, and required compact capacity.
       echo "== 35B-A3B expert_gate/up  n=512 k=2048 =="
       for T in 1 2 4 64 2048 4096; do "$BIN" 256 $T 512 2048 32 4 8 | tail -20; done
       echo "== 35B-A3B expert_down     n=2048 k=512 =="
       for T in 1 2 4 64 2048 4096; do "$BIN" 256 $T 2048 512 32 4 8 | tail -20; done
       echo "== 122B-A10B after TP=2    n=512 k=3072 / n=3072 k=512 =="
       for T in 1 2 4 64 2048 4096; do "$BIN" 256 $T 512 3072 32 4 8 | tail -20; done
       for T in 1 2 4 64 2048 4096; do "$BIN" 256 $T 3072 512 32 4 8 | tail -20; done

   The old command asserted rows/expert and special-cased decode by shrinking L. This one keeps all 256 experts and
   routes T tokens through top-8. The pinned collision tail gives Mmax 1/2/3 at T=1/2/4 (compact capacities 1/2/4)
   and 12/239/447 at T=64/2048/4096 (ordinary A only). A compact binary below Mmax refuses the fixture.

   180 units, one compile wave at MOE_CORES=192; the file's own measurement is 14-19 s wall clock, and codegen
   is 5.6% of it. Stage counts are inside a unit, so they cost nothing extra.

   WANTED: the per-format best lines. **AND READ THEM AS A LANDSCAPE, NOT A VERDICT** -- see the caveat below.

   CAVEAT, and it is the reason this is "a feel" and not the sweep: the winner is still selected by a SINGLE
   timing (`if (tf > best_tf)`), and the recorded cross-run spread is 13%. Any two configs within 13% of each
   other are being ordered by noise. Use this run to see the shape of the landscape and whether anything is
   grossly off; do not quote a winner from it. The repeat/confidence-band harness is in progress.

   THE DENSE HALF IS NOT READY. test_lowbit_dense_bench.cu carries 17 hand-written configs plus a hand-written
   dispatch macro with one line per config; the pruned i4 set is 110 shapes. Generating that list is the next
   piece of work and it is mine, not codex's.

4. **THE SWEEP. Ready to run one shape at a time, as the user asked.** Everything the earlier HELD note was
   waiting for has landed: the ragged distribution is pinned (mode 4, `token-topk-hot16x4-wor-sm64-s44-v1`, with
   Mmax derived by the router rather than asserted -- ladder 1/2/3/12/239/447 across the six token counts, so
   compact A capacities 1/2/4 cover exactly the three decode points); both benches select by MEDIAN over
   interleaved repeats with a [min,max] band and report TIES instead of naming a winner inside the noise; the
   dense config table is generated from the shared tactic rules rather than 17 hand-written rows.

   DO NOT TYPE THE SHAPES. They are derived from the three target models and the standard TP split:

       cd /sim/eec/shared/junfu.qx/quactlize && git pull --recurse-submodules
       python3 benchmarks/fixtures.py                     # the table: which shapes and why
       python3 benchmarks/fixtures.py --emit moe          # 24 runnable lines
       python3 benchmarks/fixtures.py --emit dense        # 66 runnable lines

   SHAPES FIRST, and this is codex's stated blocker: the numbers above come from each model's config.json, not
   from the checkpoints we load. Confirm them before spending a sweep on them (entry 2 above).

   RUN ONE, THEN LOOK:

       MOE_FORMATS="i4" TARGET=test_lowbit_moe_bench ./build.sh
       BIN=$(find build_ppu -type f -name test_lowbit_moe_bench -perm -u+x -print -quit)
       export BENCH_JSONL=~/sweep.jsonl
       # ...one line from `--emit moe`, e.g. the 4096-token expert_gate shape...
       python3 benchmarks/analyse.py ~/sweep.jsonl

   BENCH_JSONL is what makes the run analysable: the bench emits one sample per (fixture, config, pass) and
   analyse.py decides. Without it the run still prints a table, and the 13% cross-run spread is consumed rather
   than left on disk -- so "was that separation real?" would need a new run instead of a second look at a file.

   MOE_REPS / BENCH_REPS default to 5. Setting either to 1 is allowed and the banner then says the output is
   NOT a ranking, because it cannot be.

   WANTED: the .jsonl (or analyse.py's output). The interesting line is not the leader -- it is whether
   anything TIES it. A tie is a guard that has to be expanded before the winner is a winner.

5. **ONE-SHAPE MoE POLICY-COST DIFF (INBOX 039).** Worth one shape, and only one timing run. The current MoE
   generator is the unpruned legal Cartesian product; partition its saved samples afterward so the full and
   policy-pruned verdicts see IDENTICAL timings. This uses the 4096-token A3B expert-gate/up shape which reproduced
   the historical 33% useful MFU and exposed the ratio-8, stage-6 winner.

       cd /sim/eec/shared/junfu.qx/quactlize && git pull --recurse-submodules
       MOE_FORMATS="i4" TARGET=test_lowbit_moe_bench ./build.sh
       BIN=$(find build_ppu -type f -name test_lowbit_moe_bench -perm -u+x -print -quit)
       test -n "$BIN"
       POLICY_DIR=$(mktemp -d "$PWD/moe-policy-diff.XXXXXX")
       test -n "$POLICY_DIR" && test -d "$POLICY_DIR"
       export BENCH_JSONL="$POLICY_DIR/unpruned.jsonl"
       "$BIN" 256 4096 512 2048 32 4 8 | tee "$POLICY_DIR/unpruned.stdout"

       python3 - "$BENCH_JSONL" "$POLICY_DIR/pruned.jsonl" <<'PY'
       import collections, json, pathlib, sys

       src, dst = map(pathlib.Path, sys.argv[1:])
       records = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
       samples = [r for r in records if r.get("rec") == "s"]
       if not samples:
           raise SystemExit("no samples: BENCH_JSONL was not populated")

       # Derive policy membership from the configurations which ACTUALLY RAN. Group by fixture as well as
       # schema/TileK/stage so an unsupported row cannot influence another fixture's guard extrema.
       fixture_fields = ("fixture", "dist", "n", "k", "gs", "experts", "rows", "mmax")
       config_fields = ("schema", "tm", "tn", "tk", "wm", "wn", "st")
       def fixture(r): return tuple(r[k] for k in fixture_fields)
       def config(r): return tuple(r[k] for k in config_fields)
       def dconfig(c): return dict(zip(config_fields, c))

       by_fixture = collections.defaultdict(set)
       for r in samples:
           by_fixture[fixture(r)].add(config(r))

       kept = set()
       for fx, configs in by_fixture.items():
           wms = collections.defaultdict(set)
           for c in configs:
               q = dconfig(c)
               base = (q["schema"], q["tk"], q["st"])
               wms[(base, q["tm"], q["tn"], q["wn"])].add(q["wm"])

           primary = set()
           primary_tn = collections.defaultdict(set)
           for c in configs:
               q = dconfig(c); base = (q["schema"], q["tk"], q["st"])
               ladder = sorted(wms[(base, q["tm"], q["tn"], q["wn"])])
               if q["wm"] == ladder[-1] and q["tn"] == 2 * q["wn"]:
                   primary.add(c); primary_tn[(base, q["tm"])].add(q["tn"])

           for c in configs:
               q = dconfig(c); base = (q["schema"], q["tk"], q["st"])
               ladder = sorted(wms[(base, q["tm"], q["tn"], q["wn"])])
               ptn = primary_tn[(base, q["tm"])]
               h1 = (q["tn"] == 2 * q["wn"] and len(ladder) >= 2 and q["wm"] == ladder[-2]
                     and ptn and q["tn"] in (min(ptn), max(ptn)))
               n_guard = q["wm"] == ladder[-1]  # every legal N ratio at every TileM; H1 still prunes WarpM
               if c in primary or h1 or n_guard:
                   kept.add((fx, c))

       out = [r for r in records if r.get("rec") != "s" or (fixture(r), config(r)) in kept]
       dst.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in out))
       all_cfg = {(fixture(r), config(r)) for r in samples}
       print(f"policy partition: {len(all_cfg)} measured configs -> {len(kept)} retained", file=sys.stderr)
       PY

       python3 benchmarks/analyse.py "$POLICY_DIR/unpruned.jsonl" | tee "$POLICY_DIR/unpruned.verdict"
       python3 benchmarks/analyse.py "$POLICY_DIR/pruned.jsonl"   | tee "$POLICY_DIR/pruned.verdict"
       diff -u "$POLICY_DIR/pruned.verdict" "$POLICY_DIR/unpruned.verdict" || true
       echo "artifacts: $POLICY_DIR"

   WANTED: both verdicts, the `policy partition` count, and the diff. The cost is the pruned leader's median versus
   the unpruned leader's median; different leaders with overlapping bands are unresolved, not a measured penalty.

6. **SHIPPING-PREFILL DEVICE ABI BUILD GATE (INBOX 046 + 053).** This is the ppu001-only compile/export half of the
   llama validation. It deliberately does not claim numerical coverage of the new primary branch; that requires
   the user's regenerated llama patch and a real-model run, which cannot be named here until that patch exists.

       cd /sim/eec/shared/junfu.qx/quactlize && git pull --recurse-submodules
       PPU_DEFS="PPU_PACKED_SCALE=1" TARGET=quactlize_ppu ./build.sh
       SO=$(find build_ppu -type f -name libquactlize_ppu.so -print -quit)
       test -n "$SO" && test -f "$SO"
       for SYM in \
         quactlize_ppu_dense_fully_quantized_workspace_bytes_v1 \
         quactlize_ppu_dense_fully_quantized_dev_v1 \
         quactlize_ppu_grouped_fully_quantized_workspace_bytes_v1 \
         quactlize_ppu_grouped_fully_quantized_dev_v1 \
         quactlize_ppu_list_grouped_configs \
         quactlize_ppu_grouped_fully_quantized_config_v1 \
         quactlize_ppu_grouped_fully_quantized_dev_v2 \
         quactlize_ppu_vecdot_moe_config_v1; do
         nm -D --defined-only "$SO" | grep -q " $SYM$" || { echo "MISSING $SYM"; exit 1; }
         echo "exported $SYM"
       done
       python3 - "$SO" <<'PY'
       import ctypes, sys
       lib = ctypes.CDLL(sys.argv[1])
       class Config(ctypes.Structure):
           _fields_ = [("enable_cuda_kernel", ctypes.c_bool), ("name", ctypes.c_char_p),
                       ("tile_m", ctypes.c_int32), ("tile_n", ctypes.c_int32),
                       ("warp_m", ctypes.c_int32), ("warp_n", ctypes.c_int32),
                       ("stages", ctypes.c_int32)]
       dense = lib.quactlize_ppu_dense_fully_quantized_workspace_bytes_v1
       dense.argtypes = [ctypes.c_int] * 4
       dense.restype = ctypes.c_int64
       grouped = lib.quactlize_ppu_grouped_fully_quantized_workspace_bytes_v1
       grouped.argtypes = [ctypes.c_int] * 5
       grouped.restype = ctypes.c_int64
       d = dense(7, 256, 512, 12)
       g = grouped(6, 256, 512, 4, 12)
       assert d > 0 and g > 0, (d, g)
       assert dense(7, 256, 512, 10) == -1
       print(f"Q4 device workspaces: dense={d} grouped={g} bytes; wrong-format query declined")
       inventory = lib.quactlize_ppu_list_grouped_configs
       inventory.argtypes = [ctypes.POINTER(ctypes.POINTER(Config))]
       inventory.restype = ctypes.c_int32
       configs = ctypes.POINTER(Config)()
       count = inventory(ctypes.byref(configs))
       names = [configs[i].name.decode() for i in range(count)]
       assert count == 6 and names[0] == "16x128:16x16:s2", (count, names)
       assert not any(configs[i].enable_cuda_kernel for i in range(count - 1)), names
       assert configs[count - 1].enable_cuda_kernel and names[-1] == "vecdot_moe", names
       print(f"grouped inventory: {count - 1} tensor configs + CUDA {names[-1]}; default={names[0]}")
       PY

   WANTED: eight `exported` lines, the workspace line and the grouped-inventory line. A successful build means hgcc
   instantiated both device-pointer families and exported the grouped selection layer; it does not replace the later
   llama numerical run through those entries.

8. **FRESH SUB-FOUR DENSE DIAGNOSIS AFTER INBOX 066. This is two correctness launches, not a tactic sweep.** The
   historical 047 aborts cannot have been split-K-specific: the exact `Cfg::Kernel` that produced them was a plain
   `GemmUniversal<Problem,Main,Epi>` with no `SplitKSerialScheduler` template argument and no split axis. What remains
   unknown is whether that non-grouped two-warp failure still exists after the ordinary COARSE path was completed.

   Use the bench's fixed ScaleZero path so TILE/WARP environment values instantiate the requested kernel directly;
   the generated ScaleOnly tactic inventory deliberately quarantines the two-warp row and is not modified here.
   Build/run the four-warp control first, then the two-warp cell in its own process:

       cd /sim/eec/shared/junfu.qx/quactlize && git pull --recurse-submodules

       TILE_M=64 TILE_N=64 WARP_M=32 WARP_N=32 STAGES=3 TARGET=test_lowbit_dense_bench ./build.sh
       BIN=$(find build_ppu -type f -name test_lowbit_dense_bench -perm -u+x -print -quit)
       test -n "$BIN"
       set +e
       "$BIN" --m=2048 --n=4096 --k=4096 --g=32 --mode=2 --iterations=1 2>&1 | tee /tmp/dense_4warp_control.log
       echo "four-warp rc=${PIPESTATUS[0]}"
       set -e

       TILE_M=64 TILE_N=64 WARP_M=64 WARP_N=32 STAGES=3 TARGET=test_lowbit_dense_bench ./build.sh
       BIN=$(find build_ppu -type f -name test_lowbit_dense_bench -perm -u+x -print -quit)
       test -n "$BIN"
       set +e
       "$BIN" --m=2048 --n=4096 --k=4096 --g=32 --mode=2 --iterations=1 2>&1 | tee /tmp/dense_2warp_probe.log
       echo "two-warp rc=${PIPESTATUS[0]}"
       set -e

   WANTED: both rc lines and the last 20 lines of each log, including the assertion text if present. Control pass plus
   two-warp abort proves the current failure is on the non-split dense kernel/epilogue route. Both passing means the
   old quarantine may be stale or ScaleOnly-specific; it still must not be lifted until the exact ScaleOnly winner is
   probed. Both aborting means this ScaleZero diagnosis hit a separate unsupported path and says nothing about warps.


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

7. **DENSE vs GROUPED(L=1): AN INVARIANT, NOT A COMPARISON.** The user's acceptance criterion, and it is
   sharper than a target number because it needs no reference figure to be believed.

   **grouped with one expert IS dense.** Same math, same collective family, one group -- but it additionally
   pays a scheduler, a pointer-array epilogue, and (on the ragged path) a boundary decode. So:

       time(dense)  <=  time(grouped, L=1)      always, at the same shape

   If dense is SLOWER, dense has a defect. That direction cannot be explained by grouped overhead, by the
   masked-row tax (one group has no masked rows), or by the ragged distribution (there is none at L=1). It is
   the one comparison whose wrong answer is unambiguous.

   Run both in ONE session, one build, one machine:

       cd /sim/eec/shared/junfu.qx/quactlize && git pull --recurse-submodules

       TARGET=test_lowbit_dense_bench ./build.sh
       BIN=$(find build_ppu -type f -name test_lowbit_dense_bench -print -quit)
       "$BIN" --m=2048 --n=4096 --k=4096 --g=32  --search_configs
       "$BIN" --m=2048 --n=4096 --k=4096 --g=64  --search_configs
       "$BIN" --m=2048 --n=4096 --k=4096 --g=16  --search_configs
       "$BIN" --m=2048 --n=4096 --k=4096 --g=128 --search_configs

       TARGET=test_lowbit_moe_bench ./build.sh
       MOE=$(find build_ppu -type f -name test_lowbit_moe_bench -print -quit)
       #      L  Rows   N     K    gs  mode
       "$MOE" 1  2048  4096  4096  32   2

   Then let the analyser state the verdict rather than reading it off two logs:

       cd /sim/eec/shared/junfu.qx/quactlize
       python3 benchmarks/analyse.py --invariant /tmp/sweep/dense_g32.jsonl /tmp/sweep/grouped_L1.jsonl
       # exits non-zero and names both configs if dense is the slower side
       python3 benchmarks/analyse.py /tmp/sweep/dense_g32.jsonl        # leader, band, ties per fixture

   WANTED BACK: every `==== WINNER:` line in full (tile AND warp AND stages), the --invariant table, the
   grouped L=1 figure, and the readings for the `32x32` rows.

   THREE THINGS TO CHECK, in order of what they would overturn:

   a. **dense <= grouped(L=1).** Violated => a dense-path defect. INBOX 058 names the most likely shape of one:
      three separately-maintained mainloops, with at least one optimisation (PPU_B_CHUNK) present in two of them
      and absent from the one dense runs.

   b. **The winner carries `w64x32`.** The record says that warp shape was missed by an entire earlier sweep and
      is worth +8.6 points for int4. Our table has 29 rows with it, on tiles 64x128 and 128x64 -- `64x64` with
      `w64x32` is only two warps and the four-warp legality gate excludes it. If the measured winner IS
      `64x64:64x32`, then that gate is wrong and that matters more than the sweep result, because the same gate
      protects grouped.

   c. **`32x32` reads about 25%.** The record's own control. If it does not, the thing being measured now is not
      the thing measured then, and (a) and (b) are being read against the wrong baseline.

   RECORDED TARGETS, with their conditions, because three different numbers exist and they are not comparable:

       M=2048 N=K=4096  gs=32   int4  211.33 us / 65.0%    after adding w64x32
       M=2048 N=K=4096  gs=32   int4  55.8% (64x64:64 s4)  BEFORE w64x32 was in the grid
       M=2048 N=K=4096  gs=16   int4  53.1% (64x64:64 s3)  BEFORE w64x32
       2048x4096x4096   gs=128  int4  305 TF/s / 61%       a different sweep, 64x64/32x32/s4
       4096^3           gs=128  dense L=1  62%

   The gs=32 65.0% is the one to reproduce. The user recalls a 60+% figure at gs=16 as well; the record's own
   arithmetic gives 58.7% (65.0% divided by the recorded 10.8% int4 cost of moving to gs=16), so gs=16 is run
   here to settle that rather than to confirm it.
