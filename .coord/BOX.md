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

   `run_batch.sh` now binds each just-built packed binary through both loader domains: the generic base remains
   `QUACTLIZE_PPU_LIB`, while the arrangement-aware dense reader receives the matching
   `QUACTLIZE_PPU_LIB_FMT{0..4}` (Q4/Q5/Q2/Q3/Q6 = 0/1/2/3/4).  Do not simplify this back to a base variable only:
   `load_format(fmt)` would splice `_fmtN.so` onto that path and test a nonexistent library instead of the build.

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

---

## rowC diagnostic ledger — E1 remains; the falsely labelled E2/E3 "unrun" experiments are retired

This section was added on 2026-08-07, but the 2026-08-01 derivation record already contained both alleged missing
runs: `PPU_PACKED_PAIR=0` made rowC MATCH, and rebuilding with `PPU_F16X2_EARLYCLOBBER=0` still passed. The old
`./build.sh && $BIN` recipe had a second bug: a child script cannot export its `BIN`, so an empty outer `$BIN` was a
successful no-op. The controlled fixture is `tests/data/q4k_packed.bin`; the old `real_weight/` path is absent.

### E1 — retained only as a recurrence diagnostic

rowC is intermittent. E1 is informative only after the same current-tree baseline reproduces a failure; one MATCH
does not indict the packed arithmetic because this switch also changes instruction count, register pressure and
scheduling. Build into distinct directories, resolve each binary explicitly, and require rowA/rowB to MATCH:

    PPU_BUILD_DIR="$PWD/build_e1_base" \
      PPU_DEFS="PPU_PACKED_SCALE=1" TARGET=test_q4k_packed_gemm ./build.sh
    BASE=$(find build_e1_base -type f -name test_q4k_packed_gemm -perm -u+x -print -quit)
    test -n "$BASE"
    "$BASE" tests/data/q4k_packed.bin

    PPU_BUILD_DIR="$PWD/build_e1_scalar" \
      PPU_DEFS="PPU_PACKED_SCALE=1 PPU_PACKED_PAIR=0" TARGET=test_q4k_packed_gemm ./build.sh
    E1=$(find build_e1_scalar -type f -name test_q4k_packed_gemm -perm -u+x -print -quit)
    test -n "$E1"
    "$E1" tests/data/q4k_packed.bin

### E2 — retired

The pressured mainloop was already rebuilt with `PPU_F16X2_EARLYCLOBBER=0` and still passed; the isolated alias
probe also found 0/5 differences. Keep the conservative `"=&r"`, but there is no pending experiment and no switch.

### E3 — retired on end-to-end evidence, not assembler grammar

The fresh same-run ppu001 build ran rowC — the only packed/f16x2 path — against an independent host golden five
times with `bad=0/4096`. Flushing this fixture's subnormal `d` values predicts `3626/4096` bad, so the relevant Q4_K
input-flush theory is excluded. A build failure for guessed `sub.noftz`/`fma.rtte.noftz` spellings would only reject
the first exact mnemonic the assembler reported; it could not establish the default instruction's physical FTZ
behaviour. The guessed E3 switch was therefore deleted instead of promoted into another permanent diagnostic.

---

## INBOX 088 — PPU_B_CHUNK row axis and the Mmax=3 decode artifact

Local generation/front-end gates pass, but this workspace has no PPU SDK or device. The shortest device proof is Q3,
stage 2 only, on the eight-unit decode artifact:

    cd /sim/eec/shared/junfu.qx/quactlize && git pull --ff-only origin develop
    MOE_FORMATS=q3 PPU_DEFS=MOE_STAGES_2 TARGET=test_lowbit_moe_decode_bench ./build.sh
    D=$(find build_ppu -type f -name test_lowbit_moe_decode_bench -perm -u+x -print -quit)
    "$D" 256 4 512 2048 32 4 8 | tee /tmp/088_decode_t4.log

The first line must say `table_band=decode table_mmax=3 actual_mmax=3`. Report one `bc0->...` row and its otherwise
identical `bc1->...` row verbatim; the right side is read from the instantiated collective, so `bc1->0` is a real
negative result rather than a missing build switch. Also report their timings so the old low-bit numbers can be
classified as lower bounds or not.

The guard is part of the artifact contract. This command must print `actual routed Mmax=12` and exit 2 before any pack,
allocation-sized output, or `-> <row>` launch line:

    "$D" 256 64 512 2048 32 4 8; test $? -eq 2

Finally prove prefill still has the unpruned table. This is a separate build because linking both unit sets into one
binary would save no compile time:

    MOE_FORMATS=q3 PPU_DEFS=MOE_STAGES_2 TARGET=test_lowbit_moe_bench ./build.sh
    F=$(find build_ppu -type f -name test_lowbit_moe_bench -perm -u+x -print -quit)
    "$F" 256 64 512 2048 32 4 8 | tee /tmp/088_full_t64.log

Its first line must say `table_band=full table_mmax=0 actual_mmax=12`, and it must reach row launches. Wanted back:
both build exit codes, the two first banner lines, the decode refusal line+exit code, and one matched bc0/bc1 timing pair.

---

## INBOX 097 — capped scale-copy numerics and newly reachable Q3/Q5 w64x64

The local witness uses the real uint128 atom and proves complete coordinate coverage, but this change moves scale loads
between threads. Device numerics are therefore a required gate. Pull the SHA named in the handoff, then run both
independent goldens:

    set -euo pipefail
    cd /sim/eec/shared/junfu.qx/quactlize
    git pull --ff-only origin develop

    TARGET=test_q3_bconcat_real ./build.sh
    Q3=$(find build_ppu -type f -name test_q3_bconcat_real -perm -u+x -print -quit)
    test -n "$Q3"
    "$Q3" real_weight/real_q3k_concat.bin | tee /tmp/097_q3_correct.log

    TARGET=test_q65_bconcat_real ./build.sh
    Q65=$(find build_ppu -type f -name test_q65_bconcat_real -perm -u+x -print -quit)
    test -n "$Q65"
    "$Q65" | tee /tmp/097_q5_correct.log

Q3's process exit code is determined by its last rung, so that alone is not this gate. Wanted verbatim from
`/tmp/097_q3_correct.log`: rung 4 `(64,128,128) w64x64` and rung 6 `w64x64 ScaleOnly` must each say `MATCH`, the ladder
must say `all rungs MATCH`, and the final native golden must say `bad=0/... MATCH`. From `/tmp/097_q5_correct.log`, the
new `(64,128,256) w64x64 s2 capped scale copy` row must say `bad=0/... MATCH`, and the summary must say
`0 failing configuration(s)`.

Measure the actual generated full-table rows at the A3/A5 problem shape. Configure generates full and decode together;
the `16;64` TileM and WarpM sets are intentional, because decode retains only TileM16 and a `64`-only restriction makes
configure fail with zero decode shapes. The exact `MOE_ONLY` strings below isolate one generated row apiece:

    MOE_FORMATS='q3;q5' MOE_TM_LIST='16;64' MOE_TN_LIST=128 MOE_WM_LIST='16;64' MOE_STAGES=2 \
      TARGET=test_lowbit_moe_bench ./build.sh
    MOE=$(find build_ppu -type f -name test_lowbit_moe_bench -perm -u+x -print -quit)
    test -n "$MOE"

    MOE_REPS=5 MOE_VERBOSE=1 MOE_ONLY='q3 64x128:256 w64x64 s2 bc0->0' \
      "$MOE" 1 2048 4096 4096 32 0 | tee /tmp/097_q3_w64_bc0.log
    MOE_REPS=5 MOE_VERBOSE=1 MOE_ONLY='q3 64x128:256 w64x64 s2 bc1->1' \
      "$MOE" 1 2048 4096 4096 32 0 | tee /tmp/097_q3_w64_bc1.log
    MOE_REPS=5 MOE_VERBOSE=1 MOE_ONLY='q5 64x128:256 w64x64 s2 bc0->0' \
      "$MOE" 1 2048 4096 4096 32 0 | tee /tmp/097_q5_w64_bc0.log
    MOE_REPS=5 MOE_VERBOSE=1 MOE_ONLY='q5 64x128:256 w64x64 s2 bc1->1' \
      "$MOE" 1 2048 4096 4096 32 0 | tee /tmp/097_q5_w64_bc1.log

Every run must print its exact `->` tag, five samples, and one verdict row; `no row matched`, `did not run`, an output
witness failure, or a device error is a failure. Wanted back: all four verdict rows with median/band, µs and MFU. Compare
them explicitly with A3 (`int1`, same shape/gs, w64x64 s2 bc1, 63.7% MFU) and A5 (`Q3`, same shape/gs, 255.47 us /
53.8% MFU). Those records used smaller consumer TileK, so the comparison is a regression/ceiling reference, not a claim
that the new shipping `TK256` row must equal them.

---

## #37 — one A64 resident artifact consumed at larger single-plane TileK

The local l115 gate proves the exact physical-slot owner map, and nvcc instantiates all three real collective bodies,
but neither is a PPU numerical result. Build the dedicated A64 mode once; its table contains only the three cross-T
rows, and both the offline packer and launcher print `ArtifactTileK=64`, so a same-T repack cannot masquerade as this
gate. The first line binds every result to the checked-out SHA.

    set -euo pipefail
    cd /sim/eec/shared/junfu.qx/quactlize
    git pull --ff-only origin develop
    echo "gate-sha=$(git rev-parse HEAD)"

    PPU_DEFS=FOLD_ARTIFACT_TILEK=64 TARGET=test_fold_int2 ./build.sh
    X=$(find build_ppu -type f -name test_fold_int2 -perm -u+x -print -quit)
    test -n "$X"

    FOLD_BITS=2 FOLD_TM=64 FOLD_TN=64  FOLD_TK=128 FOLD_SVARY=1 \
      "$X" 256 256 32 | tee /tmp/037_i2_a64_t128.log
    FOLD_BITS=1 FOLD_TM=64 FOLD_TN=128 FOLD_TK=128 FOLD_SVARY=1 \
      "$X" 256 256 32 | tee /tmp/037_i1_a64_t128.log
    FOLD_BITS=1 FOLD_TM=64 FOLD_TN=128 FOLD_TK=256 FOLD_SVARY=1 \
      "$X" 256 256 32 | tee /tmp/037_i1_a64_t256.log

    set +e
    NFOLD_STD=1 FOLD_BITS=2 FOLD_TM=64 FOLD_TN=64 FOLD_TK=128 FOLD_SVARY=1 \
      "$X" 256 256 32 | tee /tmp/037_negative_unfolded_bytes.log
    neg_rc=${PIPESTATUS[0]}
    set -e
    test "$neg_rc" -eq 1
    echo "negative-control-exit=$neg_rc"

Wanted back: `gate-sha=...`; each positive banner (all must name `ArtifactTileK=64`); and these three summary rows,
each with `bad=0/65536 MATCH`:

    fold int2 (64,64,128) w32x32 F=2
    fold int1 (64,128,128) w32x64 F=4
    fold int1 (64,128,256) w32x64 F=4

The negative control must print `MISMATCH`, have a nonzero bad count, and end with
`negative-control-exit=1`. A positive result is not accepted if its banner names any artifact TileK other than 64;
that would test repacking, not resident-byte reuse.

---

## #37 — two-plane Q3/Q5/Q6 cross-T numerical gate

Run this section only after the separate single-plane section above passes.  l115 proves that every logical code has
the same resident physical owner, but it cannot prove the device converter, `P2_DIV`, scale reload, or chunk path.
These two targets have independent numerical goldens and explicitly pass ArtifactTileK to both the offline writer and
the grouped launch.  They therefore consume the resident A64/A128/A32 bytes with T256 collectives; they do not repack
the fixture at T256.

    set -euo pipefail
    cd /sim/eec/shared/junfu.qx/quactlize
    git pull --ff-only origin develop
    echo "gate-sha=$(git rev-parse HEAD)"

    TARGET=test_q3_bconcat_real ./build.sh
    Q3=$(find build_ppu -type f -name test_q3_bconcat_real -perm -u+x -print -quit)
    test -n "$Q3"
    "$Q3" real_weight/real_q3k_concat.bin | tee /tmp/037_2plane_q3_cross_t.log

    TARGET=test_q65_bconcat_real ./build.sh
    Q65=$(find build_ppu -type f -name test_q65_bconcat_real -perm -u+x -print -quit)
    test -n "$Q65"
    "$Q65" | tee /tmp/037_2plane_q56_cross_t.log

Wanted back from the Q3 log: rung 8, labelled `A64/F2/F4 -> T256`, with `bad=0/... MATCH`; `ladder: all rungs
MATCH`; and `last rung (8) ... bad=0/... MATCH`.  Wanted back from the Q5/Q6 log: the first three labels under
`--- Q6 = int4 + int2 ---` and the last under `--- Q5 = int4 + int1 ---`, all with `bad=0/32768 ... MATCH`, followed
by `0 failing configuration(s)`:

    A128/F1/F1 -> T256 w64x64 s2
    A64/F1/F2  -> T256 w64x64 s2
    A32/F2/F4  -> T256 w64x64 s2
    A64/F1/F4  -> T256 w64x64 s2

The Q3 and Q5 A64 rows cover the fold pairs absent from Q6.  The Q6 A128 row is also the exact geometry used to
release the former `ConsumerMap` T256 exclusion; its local structural prerequisite is the l115 line
`Q6_K A=128 T=256 ... logical=32768/32768 ... owner_diff=0 writer_diff=0 COMPLETE`.

---

## Five-format MoE host-link overflow: preserve and identify the payload section

Do this inventory **before** running another default build in the failed build directory. `build.sh` recreates its
build directory, and the failed object is evidence we cannot reconstruct locally. The relocation messages prove that
the final x86-64 ELF layout exceeded signed PC-relative reach; they do not by themselves prove that
`.hggcFatBinSegment` is the large payload rather than the small host-visible wrapper.

    set -euo pipefail
    cd /sim/eec/shared/junfu.qx/quactlize

    # Point FAILED_BUILD at the preserved tree if it was not build_ppu.
    FAILED_BUILD=${FAILED_BUILD:-build_ppu}
    FAIL_OBJ=$(find "$FAILED_BUILD" -type f -name 'test_lowbit_moe_bench_4a7b0aff.o' -print -quit)
    test -n "$FAIL_OBJ"
    echo "failed-object=$FAIL_OBJ"
    size -A "$FAIL_OBJ" | tee /tmp/moe_failed_object.size
    readelf -SW -rW "$FAIL_OBJ" > /tmp/moe_failed_object.readelf
    grep -Ei 'hggc|fatbin|tm_clone|GOTPCREL|PC32' /tmp/moe_failed_object.readelf \
      | tee /tmp/moe_failed_object.relevant

    # One object is not the full link payload: aggregate every PPU object by section name.
    find "$FAILED_BUILD" -path '*/ppu_obj/*.o' -type f -print0 \
      | xargs -0 size -A \
      | awk '$2 ~ /^[0-9]+$/ {bytes[$1]+=$2} END {for (s in bytes) print bytes[s], s}' \
      | sort -nr | tee /tmp/moe_failed_sections.total

    # Ask the installed driver whether it has the SDK-authored equivalent of nvcc's gen-lcs facility. Do not infer
    # support from nvcc-compatible spelling elsewhere in hgcc.
    hgcc --help 2>&1 | grep -i -A4 -B2 'host-linker-script\|gen-lcs' \
      | tee /tmp/hgcc_host_linker_script.help || true

Now build a successful single-format control in a **different** directory, leaving the failed tree intact:

    git pull --ff-only origin develop
    PPU_BUILD_DIR="$PWD/build_moe_size_i4" MOE_FORMATS=i4 MOE_CORES=192 \
      TARGET=test_lowbit_moe_bench ./build.sh
    ONE=$(find build_moe_size_i4 -type f -name test_lowbit_moe_bench -perm -u+x -print -quit)
    test -n "$ONE"
    echo "single-format-binary=$ONE"
    size -A "$ONE" | tee /tmp/moe_i4_binary.size
    find build_moe_size_i4 -path '*/ppu_obj/*.o' -type f -print0 \
      | xargs -0 size -A \
      | awk '$2 ~ /^[0-9]+$/ {bytes[$1]+=$2} END {for (s in bytes) print bytes[s], s}' \
      | sort -nr | tee /tmp/moe_i4_sections.total

Wanted back: the total byte count and every `hggc`/`fatbin`-named row from both `*.size` files; the ten largest rows
from each `*.total`; the relevant relocation lines; and whether hgcc advertises a host-linker-script generator. If
the large payload is addressed through an absolute relocation while only a small wrapper is PC-relative, an
SDK-generated linker script remains a possible later root fix. If hgcc lacks that facility, the per-format binaries
are the formal solution; `-mcmodel=medium` and `--no-relax` do not repair the already-observed crt `PC32` relocation.

The data-producing path is now:

    SWEEP_DIR=/tmp/sweep bash benchmarks/sweep_all_formats.sh

It must report all five keys (`q3 q5 q6 i2 i4`) in `ran:` and no `FAILED:` line. The five binaries contain the five
complete tactic grids; only the final link unit is split.

DO NOT pass or export BENCH_JSONL here. The script derives the file name from its own arguments and prints both the
path and the exact analyse command as its last line; an inherited BENCH_JSONL is now reported and ignored. This
paragraph used to set it, and on 2026-08-10 a complete five-format run was filed under `grouped_L1.jsonl` -- the
name of the L=1 grouped-as-dense control -- because a stale `export` from an earlier step was still live in the
shell. Nothing detected it: the records carry their own fixture identity, so the analysis was correct while every
human reading the path was told the wrong experiment. Use SWEEP_DIR to choose the directory; the basename is not a
choice. Then analyse that one file (the script prints its exact name):

    python3 benchmarks/analyse.py /tmp/sweep/prefill_L256_r4096_n512_k2048_gs32_*.jsonl --coverage

---

## INBOX 104 — kernel-only C1/decode capture (asys timeline, not process wall time)

> **SUPERSEDED as a MEASUREMENT PROTOCOL by "104b — one capture, every config" below.** The shapes, the argv, the
> build lines and the distinct-byte cross-checks in this section are still correct and still wanted. What is
> replaced is the per-config `MOE_ONLY` / 21-launch / drop-one / mean-of-20 procedure: that reconstructs a kernel
> time from host wall-clock, and one asys capture gives it directly for every config at once.
>
> **One correction to rule 5 before anyone runs it.** The expected S068 total of 4,759,552 B is the **ScaleOnly**
> figure. A build in ScaleZero carries a second metadata plane and totals **5,283,840 B**; the run already observed
> reports 256 GB/s at 20.62 us, which is exactly that. Rule 5 says a mismatch invalidates the row, so as written it
> would throw away a correct measurement. Report which mode the binary was built in and check against that number.

This gate deliberately does not ask the operator to run `test_gemv_perf` for S068--S071.  That harness hard-codes
uniform `L=8 x 1 row` shapes and cannot represent the real `E=256, active=8, empty=248` histogram; its grouped grid
also launches `grid.z=num_experts`, so substituting the synthetic case would hide 31/32 of the scheduler work.  There
is no truthful same-shape GEMV command until that benchmark entry point accepts the real histogram.  Do not use its
`N=K=2048` row as S068, and do not reduce `E` from 256 to 8.

Build the exact ScaleOnly/gs32 C1 row and the TileK=128 decode table in separate directories.  Keep both binaries:

    set -euo pipefail
    cd /sim/eec/shared/junfu.qx/quactlize
    git pull --ff-only origin develop
    echo "gate-sha=$(git rev-parse HEAD)"

    PPU_BUILD_DIR="$PWD/build_104_c1" MOE_FORMATS=i4 \
      PPU_DEFS='MOE_TK=128 LOWBIT_QMODE=1' TARGET=test_lowbit_moe_bench ./build.sh
    C1=$(find build_104_c1 -type f -name test_lowbit_moe_bench -perm -u+x -print -quit)
    test -n "$C1"

    PPU_BUILD_DIR="$PWD/build_104_decode" MOE_FORMATS=i4 \
      PPU_DEFS='MOE_TK=128 LOWBIT_QMODE=1' TARGET=test_lowbit_moe_decode_bench ./build.sh
    DEC=$(find build_104_decode -type f -name test_lowbit_moe_decode_bench -perm -u+x -print -quit)
    test -n "$DEC"

The argv below are the processes to place under the box's **asys per-kernel timeline/activity capture**.  The asys
front-end spelling is installation-specific and is not present in this checkout, so it is intentionally not guessed
here.  Capture each argv as one pass; a process-total attribution is not an acceptable substitute.

First C1, isolating the previously reported wall-clock champion.  `MOE_REPS=1` still produces exactly 21 target
launches: one warm-up followed by 20 timed launches.  `MOE_ACU` must be absent because it changes that to one cold
launch.

    unset MOE_ACU MOE_ONLY MOE_ABCAST MOEG_FORCE3D BENCH_JSONL
    MOE_REPS=1 MOE_VERBOSE=1 \
      MOE_ONLY='i4 32x128:128 w32x32 s3 bc0->0' \
      "$C1" 256 4096 512 2048 32 4 8 | tee /tmp/104_c1_wall_and_identity.log

Then the four real tokens=1 shapes.  These runs must retain the whole decode table; in particular, do not filter out
any of the 18 compiled `16x128:128` rows before profiling.

    unset MOE_ACU MOE_ONLY BENCH_JSONL
    MOE_REPS=1 MOE_VERBOSE=1 "$DEC" 256 8  512 2048 32 3 8 | tee /tmp/104_S068_decode.log
    MOE_REPS=1 MOE_VERBOSE=1 "$DEC" 256 8  512 3072 32 3 8 | tee /tmp/104_S069_decode.log
    MOE_REPS=1 MOE_VERBOSE=1 "$DEC" 256 8 2048  512 32 3 8 | tee /tmp/104_S070_decode.log
    MOE_REPS=1 MOE_VERBOSE=1 "$DEC" 256 8 3072  512 32 3 8 | tee /tmp/104_S071_decode.log

Extraction protocol, identically for every pass:

1. Group timeline activities by the exact generated grouped-GEMM kernel identity/config; do not use whole-process
   elapsed time or whole-process device attribution.
2. Each selected config must have exactly 21 target-kernel activities.  Drop activity 1 (warm-up), average activities
   2--21, and report that mean in microseconds.  A different count invalidates the row.
3. Exclude `bench_floor_nop`, allocation/H2D/memcpy, `initialize`, host launcher/setup, output poison/witness, and all
   synchronization/idle gaps.  None of those names or durations may enter the 20-launch sum.
4. For C1, the exact selected tag must appear and no second grouped-GEMM config may appear.  Report kernel-only time
   beside the existing 399.74 us host-wall value; do not replace or silently mix the two.
5. For S068--S071, return every `16x128:128` config's exact tag and 20-launch mean, plus the fastest legal row per
   shape.  Compute ScaleOnly distinct bytes as `A + active*(W+S) + C`, with 8 real rows and 8 active experts; padded
   TileM rows do not enter MBU.  The independent expected totals are S068=4,759,552 B, S069=7,135,232 B,
   S070=4,759,552 B, and S071=7,135,232 B; a mismatch invalidates the row.  Report absolute time and MBU against
   2766 GB/s.

If this asys installation exposes only a process aggregate, stop rather than reporting it as kernel-only.  The usable
fallback is a per-launch device-activity timeline (or CUDA-equivalent events around only the generated kernel); the
current host `time_it` value is not a fallback because it includes launch and final synchronization.

Wanted back: `gate-sha`; both build exit codes and binary paths; the C1 exact tag, 21 raw target durations and mean of
the last 20; for each S068--S071, the banner proving `E=256, active=8, zero-row=248`, the count of `16x128:128` rows,
and the fastest row's exact tag, 21 raw durations, 20-launch mean, distinct bytes and MBU.  Also state explicitly that
the real-histogram grouped-GEMV half remains unmeasured rather than filling that cell with the synthetic `L=8` result.

---

## INBOX 107a — dense persistent scheduler, same-binary kernel-event A/B

This is the device half of 107a.  The implementation is deliberately a one-row target: it does not compile the full
1571-row dense table, does not touch the grouped scheduler, and does not route shipping `fpA_intB` through the new
kernel.  Each binary contains the same geometry twice: the existing flat non-persistent kernel and the serial
persistent work loop selected by `--persistent`.

Two accounting corrections are already closed locally and must not be reinterpreted from the timing:

1. Plain persistence keeps the mainloop/epilogue union.  They execute serially per tile, so real shared bytes are
   `max(main,epi)`, not `main+epi`.  The binary prints the sum only as `overlap-sum-counterfactual`.
2. Static persistence schedules the same full output tiles and therefore cannot remove #10's tail.  The ~11.1% is an
   ACU-measured wave geometry on the same A0 shape through grouped-L=1, not an already measured persistent speedup and
   not a plain-dense ACU capture.  107a is the mechanism/overhead gate for StreamK.

Build BACKTEST A0 and the exact #10/ACU rung in separate directories.  Both are tiny builds:

    set -euo pipefail
    cd /sim/eec/shared/junfu.qx/quactlize
    git pull --ff-only origin develop
    git submodule update --init --recursive
    echo "gate-sha=$(git rev-parse HEAD)"

    PPU_BUILD_DIR="$PWD/build_107a_a0" BENCH_GS=32 QUANT=int4 \
      TILE_M=64 TILE_N=64 WARP_M=64 WARP_N=32 STAGES=3 \
      TARGET=test_lowbit_dense_persistent_ab ./build.sh
    A0=$(find build_107a_a0 -type f -name test_lowbit_dense_persistent_ab -perm -u+x -print -quit)
    test -n "$A0"
    "$A0" --list_configs | tee /tmp/107a_a0_config.log

    PPU_BUILD_DIR="$PWD/build_107a_rung3" BENCH_GS=32 QUANT=int4 \
      TILE_M=64 TILE_N=128 WARP_M=32 WARP_N=32 STAGES=2 \
      TARGET=test_lowbit_dense_persistent_ab ./build.sh
    R3=$(find build_107a_rung3 -type f -name test_lowbit_dense_persistent_ab -perm -u+x -print -quit)
    test -n "$R3"
    "$R3" --list_configs | tee /tmp/107a_rung3_config.log

First prove scheduler coverage on a residue and on rank-4 `L=2`.  `D` is poisoned to half NaNs before the target
launch, outside timing, so a missed tile cannot inherit a plausible value.  Every log must contain exactly one
`Disposition: Passed`; process rc alone is not this gate.

    "$A0" --m=2051 --n=4096 --k=4096 --l=1 --g=32 --mode=1 --iterations=0 \
      | tee /tmp/107a_residue_np.log
    "$A0" --m=2051 --n=4096 --k=4096 --l=1 --g=32 --mode=1 --iterations=0 --persistent \
      | tee /tmp/107a_residue_p.log
    "$A0" --m=257 --n=4096 --k=4096 --l=2 --g=32 --mode=1 --iterations=0 \
      | tee /tmp/107a_l2_np.log
    "$A0" --m=257 --n=4096 --k=4096 --l=2 --g=32 --mode=1 --iterations=0 --persistent \
      | tee /tmp/107a_l2_p.log
    for f in /tmp/107a_residue_np.log /tmp/107a_residue_p.log /tmp/107a_l2_np.log /tmp/107a_l2_p.log; do
      test "$(grep -c 'Disposition: Passed' "$f")" -eq 1
    done

Then run five interleaved passes per geometry.  `PpuTimer` places device events around exactly 100 `gemm.run()` calls;
allocation, pack, the one target warm-up, the independent reference GEMM, verification, launcher setup and final host
reporting are outside those events.  This is the existing dense kernel-event protocol, not process wall time.

    run_107a_variant() {
      local bin="$1" label="$2" side="$3" out="/tmp/107a_${label}_${side}.log"
      if [ "$side" = p ]; then
        "$bin" --m=2048 --n=4096 --k=4096 --l=1 --g=32 --mode=1 --iterations=100 --persistent | tee -a "$out"
      else
        "$bin" --m=2048 --n=4096 --k=4096 --l=1 --g=32 --mode=1 --iterations=100 | tee -a "$out"
      fi
    }
    for label in a0 rung3; do : >"/tmp/107a_${label}_np.log"; : >"/tmp/107a_${label}_p.log"; done
    for pass in 1 2 3 4 5; do
      if [ $((pass % 2)) -eq 1 ]; then order="np p"; else order="p np"; fi
      for side in $order; do run_107a_variant "$A0" a0 "$side"; done
      for side in $order; do run_107a_variant "$R3" rung3 "$side"; done
    done

Parse only the scheduler-labelled metric rows; do not combine them with `Avg runtime` or any profiler process total:

    python3 - <<'PY'
    import pathlib, re, statistics
    pat = re.compile(r'\[CUTLASS .* scheduler=(non-persistent|persistent)\].*? ([0-9.]+) us')
    rows = {}
    for label in ('a0', 'rung3'):
        for side, want in (('np', 'non-persistent'), ('p', 'persistent')):
            text = pathlib.Path(f'/tmp/107a_{label}_{side}.log').read_text()
            vals = [float(us) for sched, us in pat.findall(text) if sched == want]
            assert len(vals) == 5, (label, side, vals)
            rows[label, side] = vals
            med = statistics.median(vals)
            tf = 2*2048*4096*4096/(med*1e-6)/1e12
            print(f'{label:5s} {side:2s}: raw={vals} median={med:.3f} us '
                  f'band=[{min(vals):.3f},{max(vals):.3f}] TF/s={tf:.3f} MFU={tf/500*100:.3f}%')
        ratios = [a/b for a, b in zip(rows[label,'np'], rows[label,'p'])]
        print(f'{label:5s} paired speedup NP/P={ratios}; median={statistics.median(ratios):.5f}x')
    PY

The diagnostic lines are part of the gate, not decoration:

- A0 must print `main=31488 epi=4112 union=31488 overlap-sum-counterfactual=35600`.
- rung3 must print `main=25600 epi=16400 union=25600 overlap-sum-counterfactual=42000`.
- Non-persistent physical CTA must equal logical CTA: A0 2048, rung3 1024.
- Persistent physical CTA must equal `min(logical_cta, cu * occupancy_api)`, and must be greater than one CTA/CU
  for these shapes.  A repeated `grid=72`/one-CTA-per-CU result means the old failure remains.
- Report `occupancy_api`, warps/CTA and resident warps/CU for BOTH symbols; the persistent loop can change register
  billing even though its shared bytes do not change.

Go/no-go for 107b: correctness, grid and occupancy must all pass.  On exact rung3, the measured static-persistent
overhead must also leave room for the tail: `tP/tNP < 1/(1-0.111) = 1.125` is a necessary, not sufficient, condition.
If it exceeds 1.125, even an impossible zero-cost StreamK that recovers the entire 11.1% cannot beat the current
kernel, so 107b is not justified by #10.  Also compare the fastest persistent result against same-session A0 NP;
winning only against its slower same-tile control is not a shipping win.

Wanted back: gate SHA; both binary paths/build rc; both `--list_configs` rows; four correctness disposition lines;
one representative scheduler+smem diagnostic for each of the four symbols; all 20 raw event means; the parser output;
and any device/compiler error verbatim.  There is no acceptable wall-clock substitution.

---

## 104b — ONE capture, every config: kernel-only time from the asys sqlite

**SUPERSEDED BY INBOX 113; DO NOT RE-RUN THIS CAPTURE PROTOCOL.** One capture produced the S068 anchor below, but
repeated captures report `can't find time calibration info for device id ...` with changing 64-KB-aligned ids and
export zero KERNEL/MEMCPY/MEMSET rows (33,529 RUNTIME rows survive). This is profiler/driver state, not a harness
fallback. The one usable result remains evidence; the repeatable measurement is now the in-harness event span.

**This replaces the per-config `MOE_ONLY` / 21-launch / drop-one / mean-of-20 procedure.** That procedure exists to
reconstruct a kernel time out of host wall-clock by repeating until the launch overhead averages out. It does not
average out: `time_it` wraps the launch and `hggcDeviceSynchronize` in the host clock, and the grouped path runs
`initialize` plus a blocking prefix H2D on **every** iteration, so the overhead sits inside each of the 20 timed
samples rather than being amortised across them.

The error is not small. On S068 the whole decode table's winner reads **20.62 us** of wall-clock while moving
**5.28 MB**; the same binary at N=K=2048 moves **21.0 MB** in **20.74 us**. Four times the work, the same time.

**And the ranking is the point, not the winner's timestamp.** With roughly 13 us of fixed cost sitting on a ~7 us
kernel, two configs whose kernel times differ by 2 us read 20.6 against 22.6 at the host -- inside this harness's
recorded 13% cross-run spread. Every ranking ever taken through that timer is a ranking of noise plus a constant.
Re-reading the same runs from the timeline can change which row is called the winner, and that is the finding.

### Run

**Capture ONE CONFIG at a time.** The original instruction here said to omit `MOE_ONLY` and take the whole table
in one capture. On this box that produces a report with **zero** kernel activities; batching a few configs by
prefix fails the same way. A single config works and yields 223 kernel rows -- 201 `bench_floor_nop` launches plus
the config's 21 -- so the limit is on activities per capture, not on the profiler.

Two configs carry every current conclusion, so take those first and treat a full-table sweep as optional:

    C1 prefill winner : MOE_ONLY='i4 32x128:128 w32x32 s3 bc0->0'   against "$C1" 256 4096 512 2048 32 4 8
    decode winner     : MOE_ONLY='i4 16x32:256 w16x16 s3 bc0->0'    against "$DEC" 256 8 512 2048 32 3 8

**Check `SELECT COUNT(*) FROM HGPTI_ACTIVITY_KIND_KERNEL` after every export, before reading anything.** An empty
kernel table is what a failed capture looks like, and it looks exactly like a successful one from the outside.

    cd /sim/eec/shared/junfu.qx/quactlize && git pull --ff-only origin develop
    echo "gate-sha=$(git rev-parse HEAD)"
    # build exactly as the (superseded) section above specifies; keep both binaries

    unset MOE_ACU MOE_ONLY BENCH_JSONL
    for S in "512 2048 S068" "512 3072 S069" "2048 512 S070" "3072 512 S071"; do
      set -- $S
      asys profile --hggc-memory-usage=true -t hggc,hgtx,acdnn,acblas -f true -o /tmp/104b_$3 \
        env MOE_REPS=1 MOE_VERBOSE=1 "$DEC" 256 8 $1 $2 32 3 8 | tee /tmp/104b_$3.log
    done
    # and C1 the same way, full table, no MOE_ONLY:
    asys profile --hggc-memory-usage=true -t hggc,hgtx,acdnn,acblas -f true -o /tmp/104b_C1 \
      env MOE_REPS=1 MOE_VERBOSE=1 "$C1" 256 4096 512 2048 32 4 8 | tee /tmp/104b_C1.log

The capture line above is the form used on this box (user-supplied 2026-08-10). An earlier version of this section
omitted `hgtx` and `-f`. `-f true` overwrites an existing report, which matters because each shape is captured
separately and a stale report is worse than none.

**A capture can succeed, be 18 MB, and still contain no kernel timing.** Observed the same day: repeated
`[error][activity_buffer.cpp:244] can't find time calibration info for device id 3600041488 / 3301327104` during
capture, and the exported sqlite then had `HGPTI_ACTIVITY_KIND_KERNEL` at **0 rows** while
`HGPTI_ACTIVITY_KIND_RUNTIME` held 33,529 host-API rows and `StringIds` held 11. Device activities are dropped when
calibration fails; the size comes from the runtime rows alone. Note the two device ids in the error against the
single row in `TARGET_INFO_PPU` -- they do not agree, which is where to look first.

So **check the kernel table before trusting anything**: `--schema` prints per-table row counts, and the reader now
refuses outright rather than falling through to the runtime table, which satisfies every shape test and would have
produced a plausible ranked table of host-side API durations.

### Read

**`asys -o X` writes `X.asysrep`, not sqlite.** Observed 2026-08-10: an 18.5 MB `/tmp/104b_S068.asysrep`. So there
is an export step, exactly as with nsys, and the reader now says so instead of raising sqlite3's one-size-fits-all
"unable to open database file". Check `asys export --help`; the nsys-shaped form is

    asys export --type sqlite -o /tmp/104b_S068.sqlite /tmp/104b_S068.asysrep

    python3 tools/asys_kernel_time.py --schema /tmp/104b_S068.sqlite      # once, to see the schema
    python3 tools/asys_kernel_time.py /tmp/104b_S068.sqlite --log /tmp/104b_S068.log

**`can't find time calibration info for device id ...` during capture is not automatically fatal.** It is the
profiler failing to align DEVICE timestamps to HOST time; a kernel duration is `end - start` within one clock
domain and can survive it. The 18.5 MB report proves activities were collected. Decide from the exported data, not
from the warning: if the kernel table has rows and their start/end (or duration) are neither null nor zero, the
capture is usable. Only an empty or all-zero kernel table makes it a real failure.

The reader detects the kernel table, the name column (resolving a strings table when names are ids) and either a
duration column or a start/end pair; `--table/--name-col/--dur-col` override it and the choice is always printed.
It excludes `bench_floor_nop`, memcpy and memset **and prints what it excluded with counts**, splits the timeline
into contiguous per-config runs, drops the warm-up launch of each, and prints mean/median/min/max/spread ranked by
kernel-only time. Verified locally against a synthetic capture in the hardest shape (ids + start/end + a decoy
table), including three failure paths: a tag/segment count mismatch warns loudly and falls back to kernel names
rather than mislabelling, and a database with no kernel table errors with rc=1 instead of printing an empty table.

### Wanted back

`gate-sha`; for each shape the `[asys] table=... name=... time=...` line (so the reader's choice is auditable), the
exclusion list, and the ranked table. **Also state whether the binary was built ScaleOnly or ScaleZero**, because
the distinct-byte cross-check differs (S068: 4,759,552 B against 5,283,840 B).

If this asys build only exposes a process aggregate, stop and say so rather than reporting it as kernel-only. The
host `time_it` value is not a fallback -- it is the thing 104b exists to replace.

---

## INBOX 111 — ppu001 m8n16k16 atom, red-before-green numerical gate

This is the hard prerequisite for 112, not a performance run. One script builds two fresh trees and preserves all
artifacts under a printed `/tmp/quactlize-m8n16-111.*` directory. The ppu001 side audits the generated **hgcc
build.make** (not an ArchTag or configure message), requires its only device flag to be `-arch=ppu_10`, requires an
`m8n16k16` symbol, then runs G1 and G2. G2 passes only if the physical-16-row AIU+x4 projection is bit-exact **and**
a 32-row guard control goes green at `(0,0)` but produces the expected mismatch when the **same x4-swzl helper on the
same base** receives the NVIDIA m8n8 address coordinates. The guard height keeps every planted x4 access in range;
using the production 16-row cube here would let an out-of-range read masquerade as the expected red. The ppu0015
side compiles only the raw atom, so an AIU diagnostic cannot hide the required `Cannot select ... m8n16k16` ISel
failure. “Nonzero mismatch” is not sufficient by itself: all 512 bad-arm halfwords must first equal the independent
shifted-coordinate golden, then exactly 120/128 projected values must differ from the origin. Thus clamp/garbage or
an invalid read cannot masquerade as the expected red; the box also decides whether the x4 interface really honors
the NVIDIA formula's 16-byte column component rather than assuming it from the host model.

Separate vendor defect, fail-closed by 114: ppu001's six plain-LDSM sites in `cute/arch/copy_ppu.hpp` and six in
`cutlass/arch/memory_ppu.h` formerly carried assembler-rejected `ppu.tc01.ex.ldmatrix` spellings. The working
swizzled opcode does **not** establish the plain x1/x2/x4 N/T grammar, so 114 does not guess a replacement: all 12
direct ppu001 entries now fail at the C++ call site, while the two legacy CuTe helpers carry dependent
`static_assert`s. A local four-arm compile gate proves unused ppu001 headers compile, all 12 ppu001 direct calls stop
with 12 deleted-function diagnostics and both helpers stop with their two reasoned assertions before assembly, while
ppu0015 retains CuTe+CUTLASS copies of all six tc02 forms plus the exact direct-function ABI.
G2 still must not directly name the legacy header or instantiate/call those atoms (an unrelated actlize trait
includes that header transitively). Any future implementation needs its own SDK compile plus numerical gate.

The named symbol is the requested provenance marker, not opcode evidence by itself. The instruction identity is
cross-checked by executing the raw-atom golden on ppu001 and by the isolated ppu0015 compiler naming the
`m8n16k16` intrinsic it cannot select; neither conclusion is inferred from the kernel's name.

    set -euo pipefail
    cd /sim/eec/shared/junfu.qx/quactlize
    git pull --ff-only origin develop
    git submodule update --init --recursive
    echo "gate-sha=$(git rev-parse HEAD)"
    bash tools/run_m8n16_111_box.sh

Do not open 112 unless the final line is:

    [111] PASS: positive arch + G1 + G2 green/red + negative arch all proved

Wanted back: `gate-sha`; the ppu001/ppu0015 unique-arch lines; the `m8n16k16` symbol line; G1's one-hot sweep and
asymmetric-case summaries; `[G2-control-path]`, `[G2-green-detail]`, `[G2-green]`, `[G2-negative-detail]`, and
`[G2-negative]`; the ppu0015 `Cannot select` diagnostic; final PASS; and the printed artifact directory. A ppu0015
nonzero rc without both `Cannot select` and `m8n16k16` is a failure for the wrong reason, not G0 passing.

---

## INBOX 113 — in-harness device-event span, before 112

This replaces 104b's unreliable profiler dependency. The event pair is recorded after `initialize` and the optional
blocking ragged-prefix H2D, immediately around `gemm.run()` on the same stream. Each row now prints two times from
the same 20 launches:

- `kernel-span-upper`: primary for ranking/MFU/MBU; includes launch scheduling/idle, so it is an upper bound on, not
  an alias for, profiler kernel-only time;
- `host-wall`: the instrumented old interval, retained only for audit and the same-clock host launch-floor marker.

The one usable asys capture is the calibration anchor, not another required tool invocation: S068's exact decode row
was 11.122 us kernel-only (20 launches, 4.3% `(max-min)/mean`) beside 20.62 us wall. The 9.50 us gap was 46% of wall.
The earlier subtract-a-guessed-floor estimate, 7.49 us / 25.5% MBU, was 48% optimistic and is retired.

Build the two quant modes separately. C1's recorded 399.74 us row was ScaleOnly; S068's 20.62/11.122 us anchor and
5,283,840-byte traffic check were ScaleZero. Tactic TileK is a row axis now -- do not restore the stale `MOE_TK=128`
build knob.

    set -euo pipefail
    cd /sim/eec/shared/junfu.qx/quactlize
    git pull --ff-only origin develop
    git submodule update --init --recursive
    echo "gate-sha=$(git rev-parse HEAD)"

    PPU_BUILD_DIR="$PWD/build_113_c1" \
      MOE_FORMATS=i4 MOE_TM_LIST=32 MOE_TN_LIST=128 MOE_WM_LIST=32 MOE_STAGES=3 \
      PPU_DEFS='LOWBIT_QMODE=1' \
      TARGET=test_lowbit_moe_bench ./build.sh
    C1=$(find build_113_c1 -type f -name test_lowbit_moe_bench -perm -u+x -print -quit)
    test -n "$C1"

    PPU_BUILD_DIR="$PWD/build_113_decode" \
      MOE_FORMATS=i4 MOE_TM_LIST=16 MOE_TN_LIST=32 MOE_WM_LIST=16 MOE_STAGES=3 \
      PPU_DEFS='' \
      TARGET=test_lowbit_moe_decode_bench ./build.sh
    DEC=$(find build_113_decode -type f -name test_lowbit_moe_decode_bench -perm -u+x -print -quit)
    test -n "$DEC"

    unset MOE_ACU MOE_ONLY MOE_ABCAST MOEG_FORCE3D MOEG_PROBE BENCH_JSONL
    MOE_REPS=1 MOE_VERBOSE=1 \
      MOE_ONLY='i4 32x128:128 w32x32 s3 bc0->0' \
      "$C1" 256 4096 512 2048 32 4 8 | tee /tmp/113_C1.log

    MOE_REPS=1 MOE_VERBOSE=1 \
      MOE_ONLY='i4 16x32:256 w16x16 s3 bc0->0' \
      "$DEC" 256 8 512 2048 32 3 8 | tee /tmp/113_S068.log

    # Same binary/config/router population, about 3.98x the distinct path bytes by widening N from 512 to 2048.
    MOE_REPS=1 MOE_VERBOSE=1 \
      MOE_ONLY='i4 16x32:256 w16x16 s3 bc0->0' \
      "$DEC" 256 8 2048 2048 32 3 8 | tee /tmp/113_N2048.log

Independent traffic checks for the printed rows are C1 ScaleOnly `318,767,104 B`, S068 ScaleZero `5,283,840 B`,
and N=K=2048 ScaleZero `21,037,056 B`. A different quant-mode banner invalidates that row's byte check.

Acceptance, in order:

1. Every non-ACU detailed candidate row names `kernel-span-upper` and `host-wall`, and prints `n=20`, min, max,
   and `spread=(max-min)/mean`. Recompute spread from the three printed values. Summary/verdict rows retain the two
   means but do not repeat the per-launch range.
2. S068's event mean must land near the independent 11.122 us anchor, not 20.62 us. Treat 9--14 us as a fault-screen
   window rather than a new performance tolerance; outside it, stop and inspect the event placement/runtime.
3. S068 and N2048 must no longer collapse to the old ~20.6/~20.7 us pair. Their event means must differ by at least
   2 us, and their `[min,max]` ranges must not overlap. Either failure means this change has not yet supplied the
   2-us-resolution ruler needed by 112.
4. C1 and S068 must each return their exact selected tag, event mean/min/max/spread, host wall, and the event-based
   MFU/MBU row. Do not substitute the still-present wall number into either metric.

Wanted back: gate SHA; both build rc/binary paths and quant banners; all three exact candidate rows plus verdict rows;
the three independent byte totals; and any hggc event error verbatim. Do not open 112 until these gates pass.

---

## Grouped/MoE Stream-K phase 2 — Min=2 plus exact 64-thread fixup cohort

This is an isolated mechanism gate. It does **not** change production grouped dispatch or any published MoE number.
Phase 1's Min=8 no-split result remains in the host oracle. The live 128-thread arms now explicitly instantiate
`MinItersPerSkUnit=2`: S068 `TileK=256` / `Kt=8` must participate in Stream-K instead of merely carrying the
scheduler name. A third ragged fixture still catches a wrong `q -> (expert,m,n)` decode that S068's one-row experts
cannot see. The exact-cohort seam now also launches the current 64-thread decode champion (`16x32:256 w16x16 s3`):
its named-barrier arrival count and FP32 workspace stripe both use the exact 64-thread CTA rather than the legacy
128-thread default.

    set -euo pipefail
    cd /sim/eec/shared/junfu.qx/quactlize
    git pull --ff-only origin develop
    git submodule update --init --recursive
    echo "gate-sha=$(git rev-parse HEAD) actlize=$(git -C third_party/actlize rev-parse HEAD)"
    git merge-base --is-ancestor c72365a HEAD
    git merge-base --is-ancestor 4129f7e HEAD

    # NO local_gates HERE. Every tier in ci/local_gates.py -- gate, lint, syntax AND boxdry -- compiles
    # against dev/fold_derivation/stub_inc, which is first on -I and exists to stand in for PPU headers on a
    # machine that has none. This box HAS them, so there is no configuration in which that tier is meaningful
    # here. Measured on ppu001 2026-08-11, same source, one variable per row:
    #
    #   stub_inc only        rc=2  hggc_fp8.h not found       <- the one hggc header with no stub
    #   SDK includes only    rc=2  /usr/include/c++/13/cmath: constexpr fpclassify without __host__
    #   both                 rc=2  crt/device_functions.h:3006: undeclared '__assert'
    #
    # The middle row is the one that settles it: all-real fails too, on gcc 13 against this CUDA, and that has
    # nothing to do with our stubs. Running the tier here produced red rows that read as "the grouped Stream-K
    # contract is broken" when the contract was never evaluated.
    #
    # So these are DEV-CONTAINER facts carried over by SHA, not box facts. Establish them there:
    #
    #   python3 ci/local_gates.py -k grouped_streamk --strict
    #   python3 ci/local_gates.py -k 'grouped Stream-K' --strict
    #   python3 ci/local_gates.py -k streamk_fixup_cohort --strict
    #
    # and record the sha they passed at. A gate result without a sha is not transferable -- on a shared
    # worktree it is a statement about a tree that is nobody's commit.

    unset PPU_A_PACK PPU_B_CHUNK PPU_B_CHUNK_BISECT PPU_MAXREG PPU_DEFS
    PPU_BUILD_DIR="$PWD/build_grouped_streamk_p2" \
      TARGET=test_moe_grouped_streamk ./build.sh | tee /tmp/grouped_streamk_p2_build.log
    GSK=$(find build_grouped_streamk_p2 -type f -name test_moe_grouped_streamk -perm -u+x -print -quit)
    test -n "$GSK"
    echo "binary=$GSK"
    timeout 180s "$GSK" | tee /tmp/grouped_streamk_p2.log

The first two host-policy lines are fixed oracles, not performance claims:

    min8 Q=128 Kt=8 W=432 heuristic_tiles=0 forced_tiles=128 forced_units=128
    min2 Q=128 Kt=8 W=432 heuristic_tiles=128 forced_tiles=128 forced_units=432

The device gate must then establish all of the following:

- router IDs are `7,11,35,77,127,128,218,224`; the 128-thread control reports `Q=16` / 14 local-lock aliases,
  while decode64 reports `Q=128` / 112 aliases; TK64/TK256 and decode64 artifact comparisons both have
  `byte_diff=0`;
- the existing nonpersistent TK64 routing control is bit-exact;
- S068/TK256 prints `Q=16 Kt=8`, `sk_units=64`, `split_tiles=16`, `peer_excess=48`, `requires_fixup=1`,
  `fixup_work_items=64`, `epilogue=16`, `separate=0`, with no missing/duplicate K tile;
- on the expected `W>=256` geometry, S068/TK64 must print `Q=16 Kt=32`, `sk_units=256`, `split_tiles=16`,
  `peer_excess=240`, `fixup_work_items=256`, and exact one-time K coverage;
- on the expected `W>=96` geometry, ragged `{0,1,17,0,33}` must print `Q=6`, `sk_units=96`, `split_tiles=6`,
  `peer_excess=90`, `fixup_work_items=96`, and exact output/coverage;
- decode64 prints `threads=64 cohort=64 Q=128 Kt=8 W=432 sk_tiles=128 sk_units=432`, scheduler/reset bytes
  `262656/512`, `uniform_oracle_supported=0`, and `PLAN-PASS`. That last zero is expected: `1024 % 432 != 0`,
  so the existing uniform peer oracle must reject this partition instead of manufacturing a peer count;
- decode64's census must report `split_tiles=128`, `epilogue=128`, `fixup_final=128`, no separate-reduction,
  q-oob, empty-expert, hole, missing-K, or duplicate-K entries, and `fixup_work_items == peer_sum`. The actual
  nonuniform `peer_sum/peer_excess` are diagnostics, not an independent expected-value oracle;
- decode64's clean launch reports `bad=0 bitdiff=0 nonfinite=0 poison_left=0`. Its nonuniform peer census must now
  surface `valid_accumulator_elements`, `logical_workspace_RW=8*valid_accumulator_elements`, the old full-tile
  logical value, unchanged allocation bytes, and `MODEL-ONLY/not-a-DRAM-counter TRAFFIC-PASS`. This is a path model,
  not an HBM counter or an MBU value;
- every C-traffic row uses the per-q valid rectangle, not `Wtile*peer_excess`: production beta-zero is
  `D + 8*sum_q((peers[q]-1)*valid_m[q]*valid_n[q])`. The correctness gate additionally reads one fp16 C plane
  because it deliberately uses nonzero beta. `old_full_tile_logical_RW` remains beside the new value only as the
  historical comparison. Measured and independently expected peer counts **and** valid-element totals must agree;
  do not reuse the old fixed full-tile byte totals, merge production-beta0 with gate semantics, or infer DRAM savings
  from the logical ratio before counters;
- every numerical arm reports `bad=0 bitdiff=0 nonfinite=0 poison_left=0`, and the final line is
  `grouped Stream-K phase 2 min2 PASS: errors=0`.

Both S068 tactics also print 20 independent `kernel-span-upper` event pairs with census disabled and the vendor
barrier tail reset before each start event. Record the reset bytes and raw median/mean/min/max/spread, but phase 2
has **no performance threshold**:
this run proves the combined scheduler and reduction seam. It does not yet route production MoE or report MBU.

Wanted back: both SHAs, all three local-gate summaries, build rc/binary path, both policy lines, active IDs/artifact
line, all decomposition/census/exact/numeric/C-traffic lines including decode64, both timing lines, final PASS, and
any compiler/runtime error verbatim. A timeout is a lock/fixup failure, not a reason to retry with a different
tactic.

Do not expect a live `requested=Heuristic` row from this isolated handle. Its `can_implement()` deliberately accepts
only `DecompositionMode::StreamK`; feeding Heuristic to `Operation::inspect()` returns an empty Plan before lowering
and is not evidence for DataParallel or SplitK. The two host-policy lines above are explicitly pre-lowering policy
oracles; the device arms are explicitly forced Stream-K.

If either TK64 arm has a lower occupancy cap, the independent oracle accepts `U=min(W,Q*Kt/2)` only when
`Q*Kt % U == 0` and `Kt % (Q*Kt/U) == 0`; then `peer_excess=U-Q`. Otherwise
`ORACLE-UNSUPPORTED/FAIL` is the intended fail-closed result, not a kernel correctness verdict.

---

## Dense Marlin scheduler — default DP / Stream-K / Marlin plus blocks-per-CU ladder

This remains a **dense-only** experiment.  Do not substitute a MoE shape, fixture, or reference number.  The first
three runs preserve the existing same-binary DP / Stream-K / Marlin-B1 comparison.  The no-flag Marlin invocation is
deliberately the B1 control: `--marlin-blocks-per-cu=1` is not passed.  Only after that control has reported the exact
instantiated kernel's `Gemm::maximum_active_blocks()` value does the script launch explicit Marlin B2/B4/B6 points.
If any requested B exceeds that runtime value, the script prints `NOT RUN` and fails before launching the explicit
ladder.  Thus six is a requested fixture point, not a hard-coded legality limit.

Every run uses one int4/gs128 artifact (`ArtifactTileK=64`), one tactic (`16x128:128 w16x32 s3 bc0->0`), the same
`M=1,N=4096,K=4096,L=1` exact fixture, and 20 distinct event pairs.  TN128 gives `Q=32<CU`, so changing B changes
only Marlin's flattened `(q,k)` stripe cohort.  DP and Stream-K remain the unchanged references.

    set -euo pipefail
    cd /sim/eec/shared/junfu.qx/quactlize
    git pull --ff-only origin develop
    git submodule update --init --recursive
    echo "gate-sha=$(git rev-parse HEAD) actlize=$(git -C third_party/actlize rev-parse HEAD)"

    unset PPU_A_PACK PPU_B_CHUNK PPU_B_CHUNK_BISECT PPU_MAXREG PPU_DEFS
    timeout 900s tools/run_dense_marlin_box.sh | tee /tmp/dense_marlin_scheduler.log

The script fails closed unless all six invocations report the exact fixture/tactic, exactly one Passed disposition,
and `n=20 distinct-event-pairs=20`.  For the fixed 72-CU box it hard-checks this lowering and the matching physical
`(G,1,1)` grid:

| Marlin point | flag | G | I | active | idle | handoffs | max peers |
|---|---|---:|---:|---:|---:|---:|---:|
| B1 control | none | 72 | 15 | 69 | 3 | 66 | 4 |
| B2 | `--marlin-blocks-per-cu=2` | 144 | 8 | 128 | 16 | 96 | 4 |
| B4 | `--marlin-blocks-per-cu=4` | 288 | 4 | 256 | 32 | 224 | 8 |
| B6 | `--marlin-blocks-per-cu=6` | 432 | 3 | 342 | 90 | 331 | 12 |

For every B, the same initialized Gemm/workspace must then complete eight additional launches with raw bitdiff zero,
stable position/value fingerprints, `same-workspace=1`, and `external-lock-reset=0`.  Stream-K alone must report
`lock-reset-before-start=1`; DP and every Marlin point must report zero.  The script also binds each timing row's
`Marlin-C peer_excess` to the pinned handoff count.  There is deliberately no performance threshold: a slower B is
still a valid and necessary result.

Exact `(q,k)` coverage and globally unique q-based lock IDs are the host-side algebraic gate, not claims reconstructed
from device timing.  `l126` proves exact-once/global-q/reverse-peer for all four ladder points (and proves implicit B1
is schedule-identical to explicit B1); `l133` retains exhaustive deployment-shape coverage for the default schedule.
The box run adds numerical output plus the real named-barrier memory-order/reset evidence.  Do not claim that a timing
row by itself measured exact-once coverage.

### ACU full-counter follow-up — one instrumented B per report

Run this only after the script above passes.  It recovers the exact preserved binary path from that log.  These are
four separate full-counter captures at `iterations=1`, matching the established ACU protocol; B1 again has no BPC
flag, while B2/B4/B6 are explicit.

    set -euo pipefail
    cd /sim/eec/shared/junfu.qx/quactlize
    MARLIN_BIN=$(sed -n 's/^\[marlin-scheduler\] binary=//p' \
      /tmp/dense_marlin_scheduler.log | tail -1)
    test -n "$MARLIN_BIN" && test -x "$MARLIN_BIN"
    ACU_DIR=$(mktemp -d /tmp/dense-marlin-bpc-acu.XXXXXX)
    test -n "$ACU_DIR" && test -d "$ACU_DIR"

    ACU_USER_ROOT=/sim/eec/shared/junfu.qx
    ACU="$ACU_USER_ROOT/asight/bin/acu"
    test -x "$ACU"
    "$ACU" -f -o "$ACU_DIR/marlin-b1.report" --set full "$MARLIN_BIN" \
      --m=1 --n=4096 --k=4096 --l=1 --g=128 --mode=1 --alpha=1 --beta=0 \
      --iterations=1 --streamk_exact_fixture --marlin
    "$ACU" -f -o "$ACU_DIR/marlin-b2.report" --set full "$MARLIN_BIN" \
      --m=1 --n=4096 --k=4096 --l=1 --g=128 --mode=1 --alpha=1 --beta=0 \
      --iterations=1 --streamk_exact_fixture --marlin --marlin-blocks-per-cu=2
    "$ACU" -f -o "$ACU_DIR/marlin-b4.report" --set full "$MARLIN_BIN" \
      --m=1 --n=4096 --k=4096 --l=1 --g=128 --mode=1 --alpha=1 --beta=0 \
      --iterations=1 --streamk_exact_fixture --marlin --marlin-blocks-per-cu=4
    "$ACU" -f -o "$ACU_DIR/marlin-b6.report" --set full "$MARLIN_BIN" \
      --m=1 --n=4096 --k=4096 --l=1 --g=128 --mode=1 --alpha=1 --beta=0 \
      --iterations=1 --streamk_exact_fixture --marlin --marlin-blocks-per-cu=6

    for B in 1 2 4 6; do
      "$ACU" --import "$ACU_DIR/marlin-b${B}.report" --csv --page details \
        > "$ACU_DIR/marlin-b${B}.details.csv"
    done
    echo "ACU reports and details CSV: $ACU_DIR"

Each `iterations=1` process still contains correctness, warmup, and eight lock-fingerprint launches; it is not a
one-kernel process.  In every details CSV, bind the counter row to the production Marlin kernel and the requested
B/grid, and exclude correctness/fingerprint/warmup rows before extracting counters.

ACU is **counter-only here**.  Its instrumentation changes runtime, so neither ACU `Duration` nor the benchmark's
stdout time under ACU is a performance number.  Take time only from the preceding `iterations=20`, 20-distinct-event
box run.  For each B, return that raw timing line beside the following CSV fields:

- kernel identity, grid/block, registers/thread, theoretical and achieved occupancy (both percent and warp/CU), and
  Block Limit Registers / Shared Mem / Warps / CU;
- Speed of Light CU, Memory, L1, L2, LLC, and DRAM throughput plus Elapsed, Active, and CU Active cycles;
- Warp State cycles/instruction for `Instruction Fetch`, `Stall AMC`, `Stall Sync`, and `Memory Dependency`;
- executed counts for `v.mma.f32.f16.m16n16k16` and `v.mov.v2s`.

The comparison table must contain, at minimum, B, the 20-event time, achieved warp/CU, scheduler handoffs, and
Instruction Fetch.  Interpret both possible outcomes: falling time with gently growing handoffs means the B1 guard
was too strict; superlinear degradation as B grows is the measured slice-count/peer-chain cost curve and supports a
lower guard.  Report the entire curve in either case, not only improving points.

The earlier baseline ACU numbers were manually transcribed from GUI screenshots.  They are useful context but are not
hard-coded expectations in this script.  If a new CSV counter contradicts a copied number, flag the copied field for
re-check before changing the model.  Return both SHAs, all decomposition/grid/traffic/fingerprint lines, each arm's
raw kernel-span median/mean/min/max/spread, all four reports and details CSVs, and any build/runtime error verbatim.
Do not substitute classic Marlin, another tactic, or a MoE fixture if any rung fails.

---

## Grouped expert identity — CLOSED LOCALLY; DO NOT RUN THIS ON THE BOX

The old morning rerun is superseded by the user's hard constraint that this
integer-addressing defect be reproduced and fixed locally.  Commit `bed75b9`
first instantiated the exact G5 CuTe pointer overload and reproduced every
retained wrong value before production changed:

- typed-int4 L slicing advanced 8192 bytes for an 8192-code stride; an explicit
  subbyte slice advanced the correct 4096-byte artifact pitch;
- that read B expert `2e` below the midpoint (1→2, 3→6);
- expert 128 began one-past the 1 MiB B allocation.  The following 128 KiB
  fp16(1/32) plane spans exactly 16 bad strides and, decoded as int4, contributes
  −44 (129→85); a later zero-filled region contributes −64 (190→126).

Commit `1c5f4e7` fixes all four noninterleaved shipping seams (ordinary, fold,
2-plane low/high) by slicing L with subbyte semantics before handing the
byte-aligned expert base to AIU.  Interleaved byte-pitch paths are intentionally
unchanged.  The local, independently anchored evidence is:

    bash dev/fold_derivation/run_l130_grouped_b_idprobe.sh
    python3 ci/check_grouped_b_idprobe_contract.py
    python3 ci/check_mixed_argument_contract.py

Do not use `tools/run_grouped_b_idprobe_box.sh` to re-diagnose this item.  The
old −44/−64 values were OOB allocation contents, not stable expert transforms,
and another device allocation layout is allowed to produce different garbage.

## A2 dense Marlin cohort capability — compile and enumerate only, do not sweep yet

Commit `f477425` (actlize `73e8884c`) replaces PRE_A2's 2/4-warp numeric
whitelist with the cooperative's structural capability: one or more complete
32-thread warps, at most 1024 threads, with the actual TiledMMA CTA, FP32
workspace stripes and named-barrier arrival count statically identical.  The
committed i4 table therefore expands from 746 to all 1772 already-legal rows.
This step validates the PPU build and selection surface only; local type/layout
proof is not a device-progress result.  Do **not** pass `--search_configs`
until the queued B2 device evidence is in.

    set -euo pipefail
    cd /sim/eec/shared/junfu.qx/quactlize
    git pull --ff-only origin develop
    git submodule update --init --recursive
    git merge-base --is-ancestor f477425 HEAD
    test "$(git -C third_party/actlize rev-parse HEAD)" = \
      73e8884c239b1ba33fe18c6825756176158b0419

    unset PPU_A_PACK PPU_B_CHUNK PPU_B_CHUNK_BISECT PPU_MAXREG PPU_DEFS
    PPU_BUILD_DIR="$PWD/build_dense_marlin_sweep_i4" \
      QUANT=int4 BENCH_GS=128 TARGET=test_lowbit_dense_marlin_sweep \
      ./build.sh | tee /tmp/dense_marlin_sweep_i4_build.log
    MARLIN_SWEEP=$(find build_dense_marlin_sweep_i4 -type f \
      -name test_lowbit_dense_marlin_sweep -perm -u+x -print -quit)
    test -n "$MARLIN_SWEEP"
    "$MARLIN_SWEEP" --list_configs | tee /tmp/dense_marlin_sweep_i4_list.log

The provenance line must say `scheduler=marlin`, `source_rows=1772`,
`eligible_rows=1772`, `filtered_rows=0`,
`cohort_capability=warp-aligned-threads-32..1024`, and retain the source table's
two real FNV hashes under `source_*_fnv1a64`.  The list must contain exactly
1772 configs and no runtime DP/Stream-K arm.  A successful build establishes
that the emitted PPU device types compile; it does not establish named-barrier
progress, numerical correctness, occupancy, or speed.  This is not a
performance result and must not be reported as one.

---

## Classic-aligned dense Marlin — 2N x 4K target and occupancy ladder

This is a new, isolated target.  It does **not** replace the measured
`test_lowbit_dense_marlin_ab` 4N x 1K/stages-3 control.  Its exact compiled
identity is ordinary int4 F1, Tile `16x128x128`, Warp `16x64x32`, stages 4,
ArtifactTileK 64: `1M x 2N x 4K`, 8 warps / 256 threads, with a 64-thread K0
output/fixup cohort.  DP and Stream-K are numerically invalid for this
four-K-cohort CTA and the binary rejects them; do not substitute either arm.

The local prerequisites are `run_l138`, `run_l139`, `run_l140`, `run_l141`,
and `run_l143`.  They prove two-source delivery, the CTA-local FP32 reduction,
the exact type, the distinct WK4 artifact plus stale-WK1 negative, and the
isolated build/CLI route.  They do not prove device progress or speed.

    set -euo pipefail
    cd /sim/eec/shared/junfu.qx/quactlize
    git pull --ff-only origin develop
    git submodule update --init --recursive
    unset PPU_A_PACK PPU_B_CHUNK PPU_B_CHUNK_BISECT PPU_MAXREG PPU_DEFS

    bash tools/run_dense_marlin_wk4_box.sh | tee /tmp/dense_marlin_wk4_box.log
    WK4_ROOT=$(sed -n 's/^\[marlin-wk4\] artifacts=//p' \
      /tmp/dense_marlin_wk4_box.log | tail -1)
    WK4_BIN=$(sed -n 's/^\[marlin-wk4\] binary=//p' \
      /tmp/dense_marlin_wk4_box.log | tail -1)
    WK4_CAP=$(sed -n 's/^\[marlin-wk4\] exact instantiated-kernel B cap=//p' \
      /tmp/dense_marlin_wk4_box.log | tail -1)
    test -n "$WK4_ROOT" && test -d "$WK4_ROOT"
    test -n "$WK4_BIN" && test -x "$WK4_BIN"
    test -n "$WK4_CAP"

The script runs B=1 without an explicit override (the default-compatibility
arm), then B in `{2,4,6}` only when `B <= Gemm::maximum_active_blocks()` for
this exact instantiated kernel.  Over-cap requested rungs must print `NOT RUN`;
the script separately asks for `cap+1` and requires the host-side exact-cap
rejection before launch.  Every supported B must report:

- the `scheduler=marlin-only topology=1Mx2Nx4K` provenance line;
- WK4 artifact `roundtrip_bad=0` (stale shipping WK1 bytes are not admissible);
- 20 independent event-pair kernel spans;
- the full decomposition/grid including `blocks_per_cu`, handoffs and 256
  threads / 8 warps per CTA;
- eight stable, raw-bit-exact same-workspace lock fingerprints with no host
  lock reset.

Only after that correctness/performance run, collect ACU counters for the B
values the runtime cap admitted.  ACU is counter-only; its instrumented time
is not a performance result.

    ACU=/sim/eec/shared/junfu.qx/asight/bin/acu
    test -x "$ACU"
    ACU_DIR=$(mktemp -d /tmp/dense-marlin-wk4-acu.XXXXXX)
    for B in 1 2 4 6; do
      if [ "$B" -gt "$WK4_CAP" ]; then
        echo "ACU NOT RUN: B=$B exceeds exact cap=$WK4_CAP"
        continue
      fi
      BFLAG=()
      [ "$B" -eq 1 ] || BFLAG=("--marlin-blocks-per-cu=$B")
      "$ACU" -f -o "$ACU_DIR/wk4-b${B}.report" --set full "$WK4_BIN" \
        --marlin --streamk_exact_fixture \
        --m=1 --n=4096 --k=4096 --l=1 --g=128 --mode=1 \
        --alpha=1 --beta=0 --iterations=1 "${BFLAG[@]}"
      "$ACU" --import "$ACU_DIR/wk4-b${B}.report" --csv --page details \
        > "$ACU_DIR/wk4-b${B}.details.csv"
    done
    echo "classic-aligned ACU artifacts: $ACU_DIR"

Bind each CSV row to the production Marlin kernel/grid; exclude initialization,
golden, warmup and eight fingerprint launches.  Return, per supported B: the
20-event median/mean/min/max/spread from the first run, achieved warp/CU,
registers/thread, block limits, handoffs, Instruction Fetch, AMC, Sync, Memory
Dependency, and executed `v.mma...` / `v.mov.v2s`.  Compare the aligned B=1
time first to the standalone classic anchor `17.8 us / 17.5% nameplate`; only
then interpret the B ladder.  If a CSV value conflicts with the manually copied
ACU dataset, request a re-check of the copied field rather than changing the
model from one transcription discrepancy.
