#!/usr/bin/env bash
# ONE RUN FOR EVERY OPEN QUESTION. Six binaries, built once, kept, then measured on one pinned configuration.
#
# WHY A SCRIPT AND NOT A COMMAND BLOCK. Every trap this task has hit is a step that is easy to drop by hand:
#   * every variant needs its OWN build tree: build.sh starts by clearing that tree, and sharing one serializes the
#     whole matrix (or, under background jobs, lets one variant erase another's objects). The product is copied out
#     of the exact path that variant's build.sh reports.
#   * a macro that never reached the device compile produces the same "no change"; build.sh prints
#     "PPU_DEFS verified on <target>'s compile command" and this script FAILS if that line is missing
#   * SPLITK_ONLY matches the whole tag, so "16x128:256" keeps every warp shape, every Stages and every slice count --
#     ~228 rows, one cold launch each. SPLITK_CFG + SPLITK_S pin exactly one
#   * same-config run-to-run spread is ~13%. The old mitigation was "every number comes from ONE run of ONE binary
#     set", which controls for BETWEEN-run drift and gives no error bar at all. The effect being chased -- the native
#     scale path measuring 12.9% slower than fp16 planes -- is SMALLER THAN THAT SPREAD, and one sample per variant
#     cannot separate the two. REPS (default 3) now repeats each pinned row and reports min and spread, so the first
#     question the perf run answers is "is this effect real", not "what is its cause"
#
#   ./run_batch.sh build     build every variant and copy it out. Each binary is CACHED against the overlay content
#                            hash plus its own defines, so a rerun with no source change compiles nothing. ONLY=a,b
#                            restricts the work to those variants and FORCE=1 ignores the cache. Editing a shared
#                            header still invalidates everything -- see build_one for why that is deliberate.
#   ./run_batch.sh check     correctness gates -- must pass before any timing is meaningful
#   ./run_batch.sh perf      the pinned rows
#   ./run_batch.sh           all three, in that order
# pipefail, because do_check pipes each executable through grep and tail: without it the pipeline reports the
# status of `tail`, the harness's own exit code is discarded, and a MISMATCH or a crash still lets `all`
# proceed to timing. The whole point of running the gates before the timings is that nothing below them
# means anything if they fail.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"                      # this lives in benchmarks/ now; the repo root is one up
# WHERE THE TARGETS LAND. This said examples/99_quactlize_w4a16_compare, which was the overlay directory
# inside the actlize submodule. #34 removed the overlay entirely and the targets now build under ppu_targets/;
# the old path simply does not exist, so this script found nothing and would have reported it as a missing
# library rather than as a stale path.
OUT="${OUT:-$HOME/ab}"
VARIANT_BUILD_ROOT="${VARIANT_BUILD_ROOT:-$OUT/.build}"
# Fifteen independent targets can compile concurrently, but each hgcc front end can consume gigabytes. One make job per
# outer worker makes the cap real instead of spawning BUILD_PARALLEL*nproc compilers. Override both knobs explicitly on
# a smaller-memory machine; BUILD_PARALLEL=1 preserves the former serial behavior.
BUILD_PARALLEL="${BUILD_PARALLEL:-15}"
BUILD_INNER_JOBS="${BUILD_INNER_JOBS:-1}"
BAND="${BAND:-64 8 2048 2048 32 3}"                 # L=64, top-k 8, N=K=2048, gs=32, decode
CFG="${CFG:-16x128:256 w16x16 s2}"                  # the pinned row; SPLITK_S below pins the slice count
BLOCKS="${BLOCKS:-10}"                               # interleaved measurement blocks; see do_perf
mkdir -p "$OUT"

# name : defines.  SK_QUANT=2 is FinegrainedScaleZero, what ships.
VARIANTS=(
  "base:SK_QUANT=2"
  "pack:SK_QUANT=2 PPU_PACKED_SCALE=1"
  # THE STORE-CONFLICT FIX: one interleaved 32-bit (scale, zero) slot instead of two 16-bit planes, so the
  # decoder's 32 lanes write 32 adjacent WORDS and hit all 32 banks once. Read against pack, never base:
  # it changes nothing base does, and pack is the only variant that has the +73,728 to remove.
  "packfuse:SK_QUANT=2 PPU_PACKED_SCALE=1 PPU_PACKED_SCALE_FUSED=1"
)

# EVERY BINARY THIS RUN IS RESPONSIBLE FOR, whether it was compiled, cached or skipped. The freshness stamp at the
# end is written only if every one of them is current, which is what makes ONLY= safe: rebuilding one variant does
# not silently bless the others, and leaving the others alone does not block the run when they are already current.
ALL_BINS=()

# THE PACKED CORRECTNESS GATES, ONE LIST, USED THREE TIMES: to build them, to require them fresh, and to RUN them.
# It is a list because the previous shape -- three literal enumerations in three functions -- lost `fuse` on the third
# one. gate_fuse was built and its freshness was demanded, and then the run loop said `for g in split swz` and never
# executed it, so a round reported five green PASS blocks while the variant the round existed for was never launched.
# A missing gate does not fail; it is simply absent, which is the hardest kind of hole to see in an output someone is
# scanning for FAIL. Adding "fuse" to the third list would have re-created the same hole for the next variant.
PACKED_GATES=(
  "pack:PPU_PACKED_SCALE=1"
  # rowC is the ONLY row where kPackedScaleOn is true, so it is the only row that exercises the fused store -- and the
  # paired-column attempt this replaces passed rowA and rowB while rowC went to bad=128/4096.
  "fuse:PPU_PACKED_SCALE=1 PPU_PACKED_SCALE_FUSED=1"
)

PACKED_GATE_BINS=()
for _pg in "${PACKED_GATES[@]}"; do PACKED_GATE_BINS+=("test_q4k_packed_gemm__gate_${_pg%%:*}"); done

# THE BINARIES ONLY, never the cache sidecars. build_one writes "<target>__<name>.built" NEXT TO each binary to
# record what it was compiled from, and it shares the prefix -- so the obvious glob returns 2N paths for N variants.
# That is how the count guard came to report "expected 8 splitk binaries, found 16" on a build where all eight were
# present and correct: a guard blocking a good run is the same failure as a guard passing a bad one, and it was MY
# per-binary cache that broke it, added after the guard and never re-read against it.
#
# Filtered by SUFFIX rather than by -x: a failed link can leave a non-executable file behind, and that file must
# still be counted so the mismatch is reported, not silently skipped into agreement.
target_bins() {  # $1 target
  ls "$OUT/$1"__* 2>/dev/null | grep -v '\.built$' || true
}

build_one() {  # $1 name  $2 defines  $3 target
  # SEPARATE STATEMENTS. `local a=$1 log=...$a...` fails under `set -u` in bash 5.1: local declares every name first
  # and only then assigns, so the self-reference is unbound. bash -n does not catch it -- only running does, which is
  # why this script is now dry-run against a stubbed build.sh before it ships.
  local name="$1" defs="$2" tgt="$3"
  local log="$OUT/build_$name.log"
  local out="$OUT/${tgt}__$name" side="$OUT/${tgt}__$name.built"
  # DON'T RECOMPILE WHAT CANNOT HAVE CHANGED. A binary is current iff it was produced from the same overlay CONTENT,
  # the same defines and the same target -- the same content question require_fresh_binaries already asks, asked per
  # binary instead of once for the set. Eight compiles for a change that touches one variant is most of the iteration
  # cost of this loop.
  #
  # BE CLEAR ABOUT WHAT THIS DOES NOT BUY. The overlay stamp covers every compiled file, so editing a shared header --
  # the collective, say -- invalidates ALL of them, even the variants whose generated code is provably identical
  # because the change sits behind a macro they do not set. That is the conservative answer and it is the right one:
  # deciding "this edit cannot affect that variant" without compiling is exactly the assumption this project keeps
  # getting wrong. What it does buy is a free rerun after no source change, and ONLY= for when you know better.
  if [ -n "${ONLY:-}" ] && ! printf '%s' " ${ONLY//,/ } " | grep -q -- " $name "; then
    printf '  %-14s %s\n' "$name" "skipped by ONLY=$ONLY"
    return 0
  fi
  local want="$OVERLAY_STAMP|$defs|$tgt"
  if [ -z "${FORCE:-}" ] && [ -x "$out" ] && [ -f "$side" ] && [ "$(head -1 "$side" 2>/dev/null)" = "$want" ]; then
    printf '  %-14s %s\n' "$name" "[cached] ${defs:-<no defines>}"
    return 0
  fi
  local was=""
  [ -x "$out" ] && was="$(md5sum < "$out" | cut -d" " -f1)"
  # REMOVED BEFORE, WRITTEN AFTER. An interrupted or failed build must not leave a sidecar claiming the stale binary
  # beside it is current -- that would be the mtime guard's bug in a new place.
  rm -f "$side"
  printf '  %-14s %s\n' "$name" "${defs:-<no defines>}"
  local rc=0 build_dir="$VARIANT_BUILD_ROOT/${tgt}__${name}"
  mkdir -p "$VARIANT_BUILD_ROOT"
  ( cd "$ROOT" && PPU_BUILD_DIR="$build_dir" JOBS="$BUILD_INNER_JOBS" PPU_DEFS="$defs" TARGET="$tgt" ./build.sh ) \
    >"$log" 2>&1 || rc=$?

  # THE BUILD'S OWN EXIT CODE FIRST. This used to jump straight to the grep below, so a build that failed at cmake or
  # at hgcc -- exiting non-zero, producing no binary -- was reported as "the macro did not reach the device compile".
  # That diagnosis names a real failure mode and was simply the wrong one, which is worse than no diagnosis: it sent
  # the reader to look at macro plumbing while the actual error sat in the log's last forty lines.
  if [ "$rc" -ne 0 ]; then
    echo "    FAILED: ./build.sh exited $rc. Last lines of $log:"
    tail -20 "$log" | sed 's/^/      /'
    return 1
  fi

  # A missing verification line means the define never reached the DEVICE compile, and every number from this binary
  # would silently be the default build's. ONLY MEANINGFUL WHEN THERE ARE DEFINES: build.sh's verification block is
  # guarded by `if [ -n "$PPU_DEFS" ]`, so a variant with none -- verify_default is one -- prints nothing and could
  # never satisfy this. Requiring it there was a guaranteed false failure in a script whose whole job is to be
  # trusted before any timing is read.
  # EVERY DEFINE, NOT ONE OF THEM. build.sh prints one verification line per define and a WARNING for each one that
  # is missing, so a single `grep -q "PPU_DEFS verified"` passes as long as ANY define landed. That is how
  # PPU_PACKED_SCALE_FUSED reached the box applying to nothing: SK_QUANT and PPU_PACKED_SCALE verified, the third did
  # not, the gate saw a verified line and let it through, and the resulting binary was pack under another name -- with
  # a passing correctness run and an acu capture identical to pack's, which is the most expensive possible way to
  # learn that a flag did nothing.
  if [ -n "$defs" ]; then
    local _d _miss=()
    for _d in $defs; do
      grep -qF "PPU_DEFS verified on $tgt's compile command: -D$_d" "$log" || _miss+=("$_d")
    done
    if [ ${#_miss[@]} -ne 0 ]; then
      echo "    FAILED: these defines did NOT reach $tgt's device compile: ${_miss[*]}"
      echo "            a binary built without them is another variant under a different name."
      grep -E "WARNING|ERROR" "$log" | head -5 | sed 's/^/      /'
      return 1
    fi
  fi
  # USE THIS JOB'S REPORTED PRODUCT, not a reconstructed default path. Each worker has a private PPU_BUILD_DIR, so
  # looking under build_ppu/ would copy an old serial build (or another worker's product) while this build succeeded.
  local built
  built="$(grep -m1 '^built: ' "$log" | cut -d' ' -f2-)"
  [ -n "$built" ] && [ -x "$built" ] || {
    echo "    FAILED: build.sh reported no executable product for $tgt (got '$built'; see $log)"; return 1; }
  cp "$built" "$out"
  local now; now="$(md5sum < "$out" | cut -d" " -f1)"
  # WORTH SAYING OUT LOUD. A rebuild that reproduces the previous bytes proves the edit was inert for THIS variant --
  # which is the fact a "only build what changed" cache would have had to assume in advance and could not.
  [ -n "$was" ] && [ "$was" = "$now" ] && printf '  %-14s %s\n' "" "(binary unchanged by this edit)"
  { printf '%s\n' "$want"; printf '%s\n' "$now"; } > "$side"
}

do_build() {
  echo "== build =="
  # ONCE. build_stamp shells out to build.sh --print-overlay and hashes every compiled file; calling it per variant
  # would run it a dozen times to answer the same question.
  OVERLAY_STAMP="$(build_stamp)"
  ALL_BINS=()
  if ! [[ "$BUILD_PARALLEL" =~ ^[1-9][0-9]*$ && "$BUILD_INNER_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "  !!! BUILD_PARALLEL and BUILD_INNER_JOBS must be positive integers (got $BUILD_PARALLEL / $BUILD_INNER_JOBS)"
    return 1
  fi
  [ -n "${ONLY:-}" ] && echo "  ONLY=$ONLY -- every other variant is left alone; the stamp below is written only if"
  [ -n "${ONLY:-}" ] && echo "  they are all still current, so a stale one blocks the run rather than passing quietly"
  [ -n "${FORCE:-}" ] && echo "  FORCE=1 -- ignoring the per-binary cache"
  local ok=1 v pg
  local -a build_names=() build_defs=() build_targets=()
  for v in "${VARIANTS[@]}"; do
    build_names+=("${v%%:*}"); build_defs+=("${v#*:}"); build_targets+=(test_moe_splitk_bench)
  done
  # The correctness gate needs its own target, packed on, plus the two variants that change the packed path.
  for pg in "${PACKED_GATES[@]}"; do
    build_names+=("gate_${pg%%:*}"); build_defs+=("${pg#*:}"); build_targets+=(test_q4k_packed_gemm)
  done
  # Register the complete responsibility set in the parent shell BEFORE workers fork. Array writes in a background
  # subshell do not come back, which otherwise makes the freshness pass bless only whichever jobs happened to be cached.
  local i
  for ((i=0; i<${#build_names[@]}; ++i)); do
    ALL_BINS+=("${build_names[i]}|${build_defs[i]}|${build_targets[i]}")
  done
  echo "  ${#build_names[@]} variants, up to $BUILD_PARALLEL concurrent build(s), $BUILD_INNER_JOBS make job(s) each"

  # Bounded waves keep this compatible with the bash version already required by the script without relying on wait -n.
  # The default is one 15-job wave; a memory-constrained override trades a little tail utilization for a hard cap.
  local start end j status
  local -a pids=() statuses=()
  for ((start=0; start<${#build_names[@]}; start+=BUILD_PARALLEL)); do
    end=$((start + BUILD_PARALLEL)); [ "$end" -gt "${#build_names[@]}" ] && end=${#build_names[@]}
    pids=(); statuses=()
    for ((i=start; i<end; ++i)); do
      status="$OUT/build_${build_names[i]}.status"
      statuses+=("$status")
      build_one "${build_names[i]}" "${build_defs[i]}" "${build_targets[i]}" >"$status" 2>&1 &
      pids+=("$!")
    done
    for ((j=0; j<${#pids[@]}; ++j)); do
      wait "${pids[j]}" || ok=0
      sed 's/^/  /' "${statuses[j]}"
    done
  done
  # THE STAMP MEANS "EVERY BINARY THIS RUN IS RESPONSIBLE FOR IS CURRENT", and with ONLY= that is no longer implied
  # by "every build_one succeeded" -- the skipped ones were never looked at. So check them all, by the same content
  # question, and refuse the stamp if any is stale. Without this, ONLY= would let do_check and do_perf run against a
  # variant compiled from code nobody has any more, which is the exact failure require_fresh_binaries exists to stop.
  local entry ename edefs etgt eside stale=()
  for entry in "${ALL_BINS[@]}"; do
    ename="${entry%%|*}"; etgt="${entry##*|}"; edefs="${entry#*|}"; edefs="${edefs%|*}"
    eside="$OUT/${etgt}__${ename}.built"
    if [ ! -x "$OUT/${etgt}__${ename}" ] || [ ! -f "$eside" ] ||
       [ "$(head -1 "$eside" 2>/dev/null)" != "$OVERLAY_STAMP|$edefs|$etgt" ]; then stale+=("$ename"); fi
  done
  if [ ${#stale[@]} -ne 0 ]; then
    echo "  !!! not current: ${stale[*]}"
    echo "      no build stamp written, so check/perf will refuse. Rerun without ONLY= (or with those names in it)."
    ok=0
  fi
  # THE STAMP IS WRITTEN AT THE VERY END, AFTER the distinct-binary check below, and that ordering is the bug this
  # comment exists for. It used to be written here, so two IDENTICAL binaries set ok=0 and printed a loud warning
  # while the stamp was already on disk -- do_perf's freshness guard then passed and timed one binary against itself,
  # which is indistinguishable from "the change does nothing". That is exactly the reading the packfuse round got.
  echo "== distinct-binary check =="
  # Two identical binaries mean an A/B that compares something with itself.
  #
  # COUNT THEM FIRST. When every build_one failed, no `cp` ran, the glob matched nothing, md5sum read stdin, uniq -d
  # produced nothing, and this printed "all six splitk binaries differ" over an empty directory -- a gate reporting
  # success about files that do not exist, immediately below eleven FAILED lines.
  local want=${#VARIANTS[@]} have
  have=$(target_bins test_moe_splitk_bench | wc -l)
  if [ "$have" -ne "$want" ]; then
    echo "  !!! expected $want splitk binaries, found $have -- nothing below this can be compared"; ok=0
  else
    local dup
    dup=$(target_bins test_moe_splitk_bench | xargs -r md5sum | awk '{print $1}' | sort | uniq -d)
    if [ -n "$dup" ]; then
      # NAME THE PAIR. "two binaries are identical" over eight files leaves the reader to diff md5s by eye, and the
      # thing they need to know is WHICH variant collapsed onto which -- that names the macro that did not apply.
      echo "  !!! IDENTICAL binaries -- the A/B against them is a binary compared with itself:"
      local h
      for h in $dup; do
        printf '        %s :' "${h:0:8}"
        target_bins test_moe_splitk_bench | xargs -r md5sum | awk -v H="$h" '$1==H {n=$2; sub(/.*__/,"",n); printf " %s", n}'
        echo
      done
      ok=0
    else echo "  all $have splitk binaries differ"; fi
  fi
  # ALWAYS SHOW THE FINGERPRINTS. A byte-neutral change -- one whose shared memory, results and counters are
  # identical by design, which is exactly what PPU_PACKED_SCALE_FUSED is -- cannot be seen in any run. The binary
  # hash is the one observable that always moves when the code does, so it is printed whether or not anything is
  # wrong, and a reader comparing two rounds can see at a glance which variants actually changed.
  echo "  binary fingerprints:"
  target_bins test_moe_splitk_bench | xargs -r md5sum |
    awk '{n=$2; sub(/.*__/,"",n); printf "        %-12s %s\n", n, substr($1,1,12)}'
  # AFTER the duplicate check, so a duplicate genuinely blocks check and perf instead of warning beside a stamp.
  if [ "$ok" -eq 1 ]; then
    printf '%s\n' "$OVERLAY_STAMP" > "$OUT/build_stamp"
    # The listing the stamp was computed from, so a future mismatch can NAME the files instead of showing
    # two hashes and advising a rebuild that may not be the fix.
    stamp_files > "$OUT/build_stamp_files" 2>/dev/null
    echo "  build stamp: $OVERLAY_STAMP"
  fi
  return $((1-ok))
}

# EVERY BINARY THIS RUN NEEDS, and it must have been produced by THIS checkout. Without the second half, do_check
# happily ran binaries left in $OUT by an earlier session: a build round in which every build_one failed -- so no `cp`
# ran at all -- still produced five green rowC MATCH lines, from binaries of unknown provenance, while the two
# variants added since that older build reported "No such file or directory". Five passes and two 127s side by side,
# and the passes were the misleading half.
# THE BINARIES MUST COME FROM THIS CHECKOUT'S SOURCES -- decided by CONTENT, not by mtime.
#
# The first version compared modification times against the newest tracked file. Two things broke it, and the second
# is fatal to the whole idea: `git ls-files` lists the submodules, and stat on a gitlink returns a directory mtime
# that build.sh moves on every run; and `git pull` REWRITES THE MTIME of every file it updates, so building and then
# pulling makes every source newer than every binary. A correct build was rejected twice, which is how a guard
# teaches people to bypass it.
#
# What the question actually is: were these binaries compiled from the sources this checkout now has? That is a
# content question, and build.sh already enumerates exactly the files that get compiled -- so hash those. Pulling a
# change to run_batch.sh or a README does not invalidate anything, because neither is in the manifest.
# THE MANIFEST IS NOT THE WHOLE COMPILED SET. build.sh --print-overlay lists the files this repo OVERLAYS onto
# actlize -- 62 of them -- and the collective this entire task edits is not among them, because it lives in the
# submodule and is compiled where it sits. So the stamp was blind to the one file being changed: binaries built
# before a collective edit passed require_fresh_binaries, and the per-binary cache added later would have gone one
# worse and printed [cached] instead of recompiling. Hash the submodule's HEAD and any modified tracked file beside
# the overlay, which covers both a submodule bump and a local edit.
# ONE SOURCE OF TRUTH FOR THE STAMP: the LIST is computed here and the stamp is its hash. They were two
# separate computations before, which is how a mismatch could be reported with nothing to show for it.
stamp_files() {
  ( cd "$ROOT" && {
      ./build.sh --print-overlay | sed 's/.*|//' | sort | xargs md5sum 2>/dev/null
      git -C third_party/actlize rev-parse HEAD 2>/dev/null
      # AGAINST HEAD, NOT ls-files -m. `git ls-files -m` compares the worktree to the INDEX, so a modification that
      # has been `git add`ed is invisible to it -- and that is precisely the state that produces the most misleading
      # possible round: the stamp does not move, every binary prints [cached], the preserved build log still shows
      # the defines verified from the PREVIOUS build, and a local type gate compiles the NEW source and passes.
      # Everything agrees and the binary is old. `diff --name-only HEAD` covers staged and unstaged alike; untracked
      # files are included because a new header dropped into the submodule compiles just as hard as an edited one.
      # BUILD OUTPUT IS NOT A SOURCE, and including it made this stamp SELF-INVALIDATING. build.sh compiles into
      # build_ppu/ -- the submodule's .gitignore has `build/`, which does not match
      # that name -- so every object file it produces is "untracked" and gets hashed. OVERLAY_STAMP is computed
      # BEFORE any compile and written to disk AFTER, so `check` immediately afterwards recomputed it over the
      # post-build tree and refused: "the compiled sources have changed since these binaries were built". Nothing
      # had changed except the build's own output, and the advice the guard printed -- rebuild -- could never
      # clear it, because rebuilding reproduces the condition.
      { git -C third_party/actlize diff --name-only HEAD 2>/dev/null
        git -C third_party/actlize ls-files -o --exclude-standard 2>/dev/null; } | sort -u |
        grep -v -E '(^|/)build[^/]*/' |
        while IFS= read -r f; do md5sum "third_party/actlize/$f" 2>/dev/null; done
    } )
}

build_stamp() { stamp_files | md5sum | cut -d' ' -f1; }

require_fresh_binaries() {
  local missing=() b stamp
  for b in "$@"; do [ -x "$OUT/$b" ] || missing+=("$b"); done
  if [ ${#missing[@]} -ne 0 ]; then
    echo "  !!! not built: ${missing[*]}"
    echo "      run './run_batch.sh build' and read its output -- a build that failed leaves the PREVIOUS run's"
    echo "      binaries in $OUT, and results from those describe code nobody is running."
    return 1
  fi
  stamp="$(build_stamp)"
  if [ ! -f "$OUT/build_stamp" ]; then
    # No stamp means the binaries predate this mechanism. Allowed, once, and said out loud -- refusing would force a
    # rebuild that proves nothing, and silently allowing is what let a previous session's binaries produce five
    # green rowC lines.
    echo "  NOTE: $OUT/build_stamp is absent, so these binaries cannot be tied to a checkout. Allowed because the"
    echo "        stamp is new; the next './run_batch.sh build' will write one."
    return 0
  fi
  if [ "$(cat "$OUT/build_stamp")" != "$stamp" ]; then
    echo "  !!! the compiled sources have changed since these binaries were built."
    echo "      stamp at build: $(cat "$OUT/build_stamp")"
    echo "      stamp now:      $stamp"
    # NAME THE FILES. Two md5s and the word "rebuild" is not a diagnosis, and it was WRONG ADVICE for the first
    # real failure this guard produced: the stamp was self-invalidating -- it hashed the build's own output -- so
    # rebuilding reproduced the mismatch every time. A guard that cannot say what moved sends the reader in a
    # circle. The per-file list is recomputed here rather than stored, so it says which inputs differ from the
    # ones this checkout has NOW; if it is empty, the difference is in the file SET, not in any file's content.
    echo "      files whose content differs from the build (recomputed against this checkout):"
    if [ -f "$OUT/build_stamp_files" ]; then
      stamp_files | diff - "$OUT/build_stamp_files" 2>/dev/null |
        grep -E '^[<>]' | sed 's/^/        /' | head -20
      local n; n=$(stamp_files | diff - "$OUT/build_stamp_files" 2>/dev/null | grep -cE '^[<>]')
      [ "${n:-0}" -gt 20 ] && echo "        ... and $((n - 20)) more"
      [ "${n:-0}" -eq 0 ] && echo "        (none -- the two stamps differ but every listed file matches; the stamp"
      [ "${n:-0}" -eq 0 ] && echo "         computation itself is unstable, which is a bug in build_stamp)"
    else
      echo "        (no per-file record; it is written by builds from this version onward)"
    fi
    echo "      rebuild ONLY if a file above is one you edited. If the list is empty or shows build artefacts,"
    echo "      the stamp is at fault and rebuilding will not clear it."
    return 1
  fi
  return 0
}

do_check() {
  echo
  echo "== correctness (nothing below matters until these pass) =="
  require_fresh_binaries \
    "${PACKED_GATE_BINS[@]}" || { echo "== aborting: the binaries are not this checkout's =="; return 1; }
  # rowC is the only row where Scale_TileK == 8, i.e. the only one on the packed path. It has been INTERMITTENT --
  # bad=128, then 724, then MATCH with no semantic source change -- and the cause is still unknown, so it is run five
  # times. A single pass proves nothing about a flaky failure.
  local fails=0 out rc g ran_gates=0
  echo "-- packed, five runs (rowC has been intermittent; the cause is NOT known)"
  for i in 1 2 3 4 5; do
    printf '   run %d: ' "$i"
    # STATUS FIRST, filtering second: piping straight into grep reports grep's status and loses the harness's.
    out=$("$OUT/test_q4k_packed_gemm__gate_pack" "$ROOT/tests/data/q4k_packed.bin" 2>&1) && rc=0 || rc=$?
    printf '%s' "$out" | grep -E "rowC|== (PASS|FAIL)" | tr '\n' ' '; echo "  [exit $rc]"
    [ "$rc" -eq 0 ] || fails=$((fails+1))
  done
  local pg
  for pg in "${PACKED_GATES[@]}"; do
    g="${pg%%:*}"; [ "$g" = pack ] && continue        # pack already ran five times above
    echo "-- packed + $g"
    out=$("$OUT/test_q4k_packed_gemm__gate_$g" "$ROOT/tests/data/q4k_packed.bin" 2>&1) && rc=0 || rc=$?
    printf '%s\n' "$out" | grep -E "rowA|rowB|rowC|== (PASS|FAIL)"; echo "   [exit $rc]"
    [ "$rc" -eq 0 ] || fails=$((fails+1))
    ran_gates=$((ran_gates+1))
  done
  # THE HOLE THIS CLOSES IS "ABSENT", NOT "FAILED". Count what ran against what exists, so a gate that stops being
  # launched says so instead of quietly not appearing.
  if [ "$ran_gates" -ne $(( ${#PACKED_GATES[@]} - 1 )) ]; then
    echo "  !!! ran $ran_gates packed gates but ${#PACKED_GATES[@]} are declared (pack runs separately)"; fails=$((fails+1))
  fi
  if [ "$fails" -ne 0 ]; then
    echo "== $fails correctness gate(s) FAILED -- the timings below would be meaningless =="; return 1
  fi
  echo "== all correctness gates passed =="
}

# THE PYTHON TIER, ON THE DEVICE. Everything above builds and runs C++ binaries; the routes, the artifact
# inverses and the twenty CUDA-core launches are reachable only from Python, through the dlopen seam. Until this
# step existed, NOTHING ran them on the box -- they were verified on a 5090 and asserted about ppu001.
#
# THREE TRAPS, one per step below.
#
#  * the build can HANG rather than fail. actlize's CMake FetchContent-clones googletest; two runs have sat at
#    index-pack with zero output for 4+ and 9+ minutes and had to be killed. A hang is worse than a failure --
#    the operator cannot tell "still compiling" from "stuck". BUILD_TIMEOUT converts it into a failure that says
#    which external fetch it died in.
#  * a GREEN RUN THAT TESTED NOTHING. quactlize falls back to a host backend when the library is absent, so the
#    whole suite can pass having exercised no device code at all. The backend is asserted BEFORE pytest runs, so
#    a fallback is a hard failure of this step and not a quiet 216/216.
#  * "the matrix is green" IS NOT A RESULT. Every status in schemes.py is a LITERAL that a person wrote. This
#    step prints it, and prints the registry cross-check that says each VALIDATED cell has a harness behind it --
#    but a passing run does not promote a cell, and the header says so, because a table printed under a green
#    test run reads as though the run produced it.
# The extension-staleness verdict, in ONE place. It is asked twice -- before rebuilding and after -- and two
# copies of this rule would be two chances for them to disagree about what "current" means.
ext_staleness_verdict() {
  PYTHONPATH="$ROOT" python3 - "$ROOT" <<'PYEOF'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
so = sorted((root / "quactlize").glob("_C*.so"))
if not so:
    print("ABSENT"); raise SystemExit
built = so[0].stat().st_mtime
newer = sorted(p.name for p in (root / "quactlize" / "csrc").rglob("*")
               if p.suffix in (".cpp", ".h", ".hpp") and p.stat().st_mtime > built)
print("STALE " + ", ".join(newer[:4]) if newer else "CURRENT")
PYEOF
}

do_pytest() {
  local timeout_s="${BUILD_TIMEOUT:-900}"
  echo
  echo "== python tier on the device =="

  # 1. the .so. build.sh prints "built: <path>"; parse that rather than re-deriving the build directory, which is
  #    the pair of facts that drifted apart the last time a path was written down in two places.
  echo "-- building libquactlize_ppu.so (timeout ${timeout_s}s)"
  local blog rc so
  blog="$OUT/quactlize_ppu.build.log"
  # PPU_PACKED_SCALE=1 IS NOT OPTIONAL FOR THIS TIER. The device symbol always exists, but without the macro the
  # packed-dense launch returns rc=34 -- so a library built without it makes the FULLY_QUANTIZED cell unrunnable
  # while every other test still passes, which reads as "that cell is fine" rather than "that cell was not built".
  timeout "$timeout_s" env PPU_DEFS="${PPU_DEFS:-PPU_PACKED_SCALE=1}" TARGET=quactlize_ppu "$ROOT/build.sh" \
    >"$blog" 2>&1 && rc=0 || rc=$?
  if [ "$rc" -eq 124 ]; then
    echo "   !!! TIMED OUT after ${timeout_s}s -- it did not fail, it hung. Last lines:"
    tail -5 "$blog" | sed 's/^/       /'
    echo "       If those name googletest/FetchContent, this is the known offline hole (task #31), not your change."
    return 1
  fi
  [ "$rc" -eq 0 ] || { echo "   !!! build failed (exit $rc):"; tail -15 "$blog" | sed 's/^/       /'; return 1; }
  so="$(grep -m1 '^built: ' "$blog" | cut -d' ' -f2-)"
  [ -n "$so" ] && [ -f "$so" ] || { echo "   !!! build.sh reported no 'built:' path, or it does not exist: '$so'"; return 1; }
  echo "   $so"

  # ASSERT THE DEFINE REACHED THE COMPILE, here, instead of discovering it as rc=34 inside a test.
  # This step passed PPU_DEFS and then trusted it. On ppu001 the library came out WITHOUT packed scale anyway and
  # the first sign was two oracles failing deep in pytest with "rc=34 means this library is not built with
  # PPU_PACKED_SCALE=1" -- an accurate message arriving several minutes after the point where it could have been
  # one line. run_batch's C++ build path has asserted this since it was written; the python one simply did not.
  local _d
  for _d in ${PPU_DEFS:-PPU_PACKED_SCALE=1}; do
    grep -qF "PPU_DEFS verified on quactlize_ppu's compile command: -D$_d" "$blog" && continue
    echo "   !!! -D$_d did NOT reach quactlize_ppu's compile command."
    # build.sh's own words, not a paraphrase of them. The first version of this grep looked for "did NOT reach"
    # while build.sh says "is NOT on <target>'s compile command" and "no build.make found for <target>" -- so the
    # one line that says WHY was filtered out by the diagnostic meant to surface it, and the operator saw a
    # refusal with no cause. Match WARNING, which is build.sh's marker for all three.
    grep -E "PPU_DEFS|PPU_EXTRA_DEFS|WARNING" "$blog" | sed 's/^/       /' | head -8
    echo "       Everything below would run against a library missing the feature, and the FULLY_QUANTIZED"
    echo "       cells would fail with rc=34 for a build reason rather than a numerical one. Refusing."
    return 1
  done
  echo "   defines verified: ${PPU_DEFS:-PPU_PACKED_SCALE=1}"

  # 2. preconditions, named individually. A bare "pytest failed to import" sends the reader into the test files
  #    when the answer is that this box has no torch.
  local missing=""
  for m in pytest torch gguf; do
    python3 -c "import $m" 2>/dev/null || missing="$missing $m"
  done
  if [ -n "$missing" ]; then
    echo "   !!! this box cannot run the python tier -- missing:$missing"
    echo "       gguf is the ORACLE, not a convenience: without it the tests have nothing independent to compare"
    echo "       against and would have to be skipped, which is the failure mode this step exists to prevent."
    return 1
  fi

  # 2b. THE SECOND LIBRARY. There are two, and the first version of this step only knew about one.
  #       libquactlize_ppu.so        the DEVICE half, dlopened at run time -- what step 1 just built
  #       quactlize's torch ext      the HOST half (preprocessing), built by setup.py -- needs no PPU
  #     gguf_backend() goes through _ops(), which is the HOST extension, so with it absent the step cannot even
  #     ask which backend it would use: the query in step 3 dies with quactlize's own "operator library is not
  #     built" ImportError and QUACTLIZE_PPU_LIB never gets a chance to matter. Observed on the box exactly that
  #     way, with the device .so sitting built at the path step 1 had just printed.
  #     Built in place rather than installed: --inplace writes into this checkout and touches no site-packages,
  #     which is the same blast radius build.sh already has. QUACTLIZE_NO_HOST_BUILD=1 turns it into a check.
  # PRESENT IS NOT THE SAME AS CURRENT, and the box proved it. This step checked only that _ops() imports, so
  # after a pull that changed gguf_prepass_ops.cpp / ppu_backend.cpp / ppu_backend.h the run went green through a
  # STALE .so -- the whole suite reported on code nobody was running, and the only thing that noticed was
  # test_layouts.py's own guard, one ERROR at the end, after everything else had already been measured against
  # last build's binary. That is the same failure shape run_batch already refuses for the C++ binaries
  # ("the compiled sources have changed since these binaries were built"); the python extension just lacked it.
  # The rule is copied from tests/test_layouts.py:373 rather than re-derived: any .cpp/.h/.hpp under csrc newer
  # than the .so means rebuild. setuptools does not track headers, which is why the header extensions are in it.
  # The verdict goes to STDOUT so it can be captured; the NAMES go with it, because "stale" without saying
  # which file is stale sends the reader to the wrong place. Nothing is discarded -- an earlier draft sent the
  # diagnostic to stderr and then captured with 2>/dev/null, which would have hidden the one useful line.
  local ext_verdict
  ext_verdict="$(ext_staleness_verdict)" || ext_verdict="ABSENT"
  case "$ext_verdict" in
    STALE*) echo "   ${ext_verdict#STALE } is newer than the built extension" ;;
  esac
  if [ "$ext_verdict" != CURRENT ] || ! PYTHONPATH="$ROOT" python3 -c 'import quactlize; quactlize._ops()' >/dev/null 2>&1; then
    if [ -n "${QUACTLIZE_NO_HOST_BUILD:-}" ]; then
      echo "   !!! quactlize's host operator extension is absent or STALE, and QUACTLIZE_NO_HOST_BUILD is set."
      echo "       Build it with:  cd $ROOT && python3 setup.py build_ext --inplace"
      return 1
    fi
    echo "-- host operator extension absent or older than csrc; rebuilding it in place (needs no PPU)"
    local hlog="$OUT/quactlize_host_ext.log"
    ( cd "$ROOT" && python3 setup.py build_ext --inplace ) >"$hlog" 2>&1 \
      || { echo "   !!! host extension build failed:"; tail -20 "$hlog" | sed 's/^/       /'; return 1; }

    # ASKING FOR A REBUILD IS NOT THE SAME AS GETTING ONE. This step detected staleness and then trusted
    # setuptools to act on it, and ppu001 ran an extension whose producer still returned the pre-073 two-tensor
    # artifact -- ValueError: not enough values to unpack (expected 3, got 2) -- from a tree whose C++ had
    # returned three for hours. Whatever the reason setuptools declined, the fix is to CHECK, then force.
    if [ "$(ext_staleness_verdict)" != CURRENT ]; then
      echo "   setuptools left it stale; removing the .so and rebuilding from scratch"
      rm -f "$ROOT"/quactlize/_C*.so
      ( cd "$ROOT" && python3 setup.py build_ext --inplace ) >>"$hlog" 2>&1 \
        || { echo "   !!! forced rebuild failed:"; tail -20 "$hlog" | sed 's/^/       /'; return 1; }
      [ "$(ext_staleness_verdict)" = CURRENT ] \
        || { echo "   !!! STILL stale after a clean rebuild -- something outside setuptools is wrong. See $hlog"
             return 1; }
    fi
    PYTHONPATH="$ROOT" python3 -c 'import quactlize; quactlize._ops()' >/dev/null 2>&1 \
      || { echo "   !!! built, but still not importable -- see $hlog"; return 1; }
    echo "   ok"
  fi

  # 3. THE BACKEND ASSERTION. Before any test runs. quactlize.gguf_backend() returns which implementation the
  #    routes will actually call; if the library did not load, it returns a host fallback and every test below
  #    would pass without touching the device.
  local backend
  backend="$(QUACTLIZE_PPU_LIB="$so" PYTHONPATH="$ROOT" python3 -c \
      'import quactlize; print(quactlize.gguf_backend())' 2>&1)" || {
    echo "   !!! could not query the backend: $backend"; return 1; }
  echo "-- backend: $backend"
  case "$backend" in
    ppu*) ;;
    *) echo "   !!! the library did not take: the routes would run on '$backend', not the device."
       echo "       A pass from here would be a pass of the HOST path with the device untested. Refusing."
       return 1 ;;
  esac

  # 4. THE DENSE INDEPENDENT ORACLE, REQUIRED AND UNSKIPPABLE. The full suite legitimately skips this test on an
  # NVIDIA development host, so "pytest passed" alone cannot prove the PPU GEMM was compared with anything. Select
  # it first, require five passes and reject any skip before allowing the broad suite to run.
  echo "-- dense PPU vs dequant-first oracle (all five formats; skips are failures here)"
  local dlog drc
  dlog="$OUT/dense_python_oracle.log"
  QUACTLIZE_PPU_LIB="$so" PYTHONPATH="$ROOT" python3 -m pytest -q -rs -s \
    "$ROOT/tests/test_gguf_routes.py::test_scale_first_dense_route_matches_dequant_first_and_rejects_fault" \
    >"$dlog" 2>&1 && drc=0 || drc=$?
  tail -15 "$dlog"
  if [ "$drc" -ne 0 ] || grep -qi "skipped" "$dlog" || ! grep -Eq '[1-9][0-9]* passed' "$dlog"; then
    echo "   !!! dense independent oracle did not execute five passing PPU cases (exit $drc)"
    return 1
  fi

  # 5. the suite
  # `-q | tail -25` was WRONG and the box proved it: a run reporting "17 failed, 201 passed, 2 errors" showed
  # none of the seventeen names, because the summary is the LAST thing pytest prints and the failure list is
  # above it. tail keeps exactly the part that says how many and drops the part that says which.
  # -rf prints the short failure summary AFTER the counts, so it survives a tail; the full output with
  # tracebacks goes to a log for anything the summary does not settle.
  # TWO PASSES, BECAUSE THE ARM A TEST RUNS AGAINST IS PART OF WHAT IT TESTS. Sixteen of the first box run's
  # seventeen failures were one cause: tests written to validate gguf_vecdot's CPU REFERENCE ARM were handed the
  # device arm, because quactlize dlopens the library whenever it is available. They are marked cpu_reference now
  # and run with the dlopen deliberately broken. Pointing QUACTLIZE_PPU_LIB at a nonexistent path is the
  # mechanism: ppu_backend::load() makes ONE attempt and caches the outcome, so a bad path fails deterministically
  # rather than depending on what is on the loader path.
  local plog prc
  local -i rc1 rc2
  echo "-- pytest pass 1/2: the CPU reference arm (device dlopen deliberately broken)"
  plog="$OUT/pytest_cpu_arm.log"
  QUACTLIZE_PPU_LIB="/nonexistent-so-the-cpu-arm-runs" PYTHONPATH="$ROOT" \
    python3 -m pytest "$ROOT/tests" -q -rfE --tb=short -m cpu_reference >"$plog" 2>&1 && rc1=0 || rc1=$?
  tail -1 "$plog" | sed 's/^/   /'
  [ "$rc1" -eq 0 ] || { echo "   -- failed ids (full tracebacks in $plog):"
    sed -n '/^=* short test summary info/,$p' "$plog" | grep -E "^(FAILED|ERROR)" | sed 's/^/     /'; }

  echo "-- pytest pass 2/2: everything else, on the device"
  plog="$OUT/pytest_full.log"
  QUACTLIZE_PPU_LIB="$so" PYTHONPATH="$ROOT" \
    python3 -m pytest "$ROOT/tests" -q -rfE --tb=short -m "not cpu_reference" >"$plog" 2>&1 && rc2=0 || rc2=$?
  tail -1 "$plog" | sed 's/^/   /'
  [ "$rc2" -eq 0 ] || { echo "   -- failed and errored ids (full tracebacks in $plog):"
    sed -n '/^=* short test summary info/,$p' "$plog" | grep -E "^(FAILED|ERROR)" | sed 's/^/     /'; }

  # THE FULLY_QUANTIZED CELL MUST RUN, for the same reason as the dense oracle below: it skips honestly when the
  # ops are not in the build, and on a box built with PPU_PACKED_SCALE=1 a skip means the cell was not exercised.
  # This is the obligation the marker's own description records -- wiring the oracle up and NOT adding it here
  # would let a skip read as a pass.
  # ONE BUILD PER FORMAT, because the device library is format-specific ON PURPOSE: a PPU_PACKED_FORMAT=2 binary
  # cannot run the Q4 packed launch (codex, 070). Testing both therefore means building both -- there is no
  # single binary that covers them, and a run that quietly tested only the default would report half a matrix as
  # if it were the whole one.
  #
  # The default build (already made above) is Q4_K. Each additional format gets its own build, and
  # QUACTLIZE_PACKED_FORMAT tells the oracles which one they are looking at so the other formats skip with a
  # reason instead of failing against a binary that was never meant to run them.
  local fqrc=0 _fmt _fmtname _label _fmtdefs _packed_fmt _fmtso _fqlog _r _b2 _d2 _LAST_FMT_BUILT=default
  # ggml type : label : extra defines.  The default build (no PPU_PACKED_FORMAT) is Q4_K.
  #   Q2_K single-plane, 20 B unit staged as five 4 B copies      PPU_PACKED_FORMAT=2
  #   Q5_K TWO-plane weight, scu16x1 -- the SAME scale unit as Q4  PPU_PACKED_FORMAT=1
  #   Q3_K two-plane, PAIRED 28 B unit spanning two superblocks     PPU_PACKED_FORMAT=3
  #   Q6_K two-plane, PAIRED 36 B unit, weight kept at TK128        PPU_PACKED_FORMAT=4
  # FIVE BUILDS IS THE COST OF FORMAT-SPECIFIC BINARIES, not an oversight. There is no library that runs more
  # than one of these, so a run that built once would report one fifth of the matrix as though it were all of it.
  for _fmt in "12:Q4_K:" "10:Q2_K:PPU_PACKED_FORMAT=2" "13:Q5_K:PPU_PACKED_FORMAT=1" \
              "11:Q3_K:PPU_PACKED_FORMAT=3" "14:Q6_K:PPU_PACKED_FORMAT=4"; do
    IFS=: read -r _fmtname _label _fmtdefs <<<"$_fmt"
    case "$_fmtname" in
      12) _packed_fmt=0 ;; 13) _packed_fmt=1 ;; 10) _packed_fmt=2 ;;
      11) _packed_fmt=3 ;; 14) _packed_fmt=4 ;;
      *) echo "   !!! no packed-format library id for ggml qtype $_fmtname"; fqrc=1; continue ;;
    esac
    _fqlog="$OUT/fully_quantized_$_label.log"
    if [ -z "$_fmtdefs" ]; then
      _fmtso="$so"                                     # the build made above
    else
      echo "-- rebuilding the device library for $_label ($_fmtdefs)"
      local _b2="$OUT/quactlize_ppu.$_label.build.log"
      timeout "$timeout_s" env PPU_DEFS="PPU_PACKED_SCALE=1 $_fmtdefs" TARGET=quactlize_ppu "$ROOT/build.sh" \
        >"$_b2" 2>&1 || { echo "   !!! $_label build failed:"; tail -12 "$_b2" | sed 's/^/       /'; fqrc=1; continue; }
      local _d2
      for _d2 in PPU_PACKED_SCALE=1 $_fmtdefs; do
        grep -qF "PPU_DEFS verified on quactlize_ppu's compile command: -D$_d2" "$_b2" && continue
        echo "   !!! -D$_d2 did NOT reach the $_label build"
        grep -E "PPU_DEFS|WARNING" "$_b2" | sed 's/^/       /' | head -6; fqrc=1
      done
      _LAST_FMT_BUILT="$_label"
      _fmtso="$(grep -m1 '^built: ' "$_b2" | cut -d' ' -f2-)"
      [ -f "$_fmtso" ] || { echo "   !!! no library from the $_label build"; fqrc=1; continue; }
    fi

    # ONLY the fully_quantized marker runs against a format-specific library, and that is not tidiness.
    # A packed build cannot also serve scale_first for the same format -- the format-1 library's Q5 TileK=256
    # type cannot accept fp16 scale-first metadata, and its scale-first ABI says so with rc=36 (codex, 075).
    # Widening this selection would turn a build's deliberate scope into a wall of failures on cells that are
    # VALIDATED and untouched.
    echo "-- fully_quantized cells, $_label (a skip here is a failure once the ops are built)"
    # Arrangement-aware tensor readers use load_format(fmt), while generic producer/BC ops still use load().
    # Name BOTH paths explicitly: setting only the base makes load_format splice `_fmtN` onto the path of the
    # library which was just built and then fail to open a file that does not exist.
    env QUACTLIZE_PPU_LIB="$so" "QUACTLIZE_PPU_LIB_FMT${_packed_fmt}=$_fmtso" \
      QUACTLIZE_PACKED_FORMAT="$_fmtname" PYTHONPATH="$ROOT" \
      python3 -m pytest "$ROOT/tests" -q -rs -m fully_quantized_dense >"$_fqlog" 2>&1 && _r=0 || _r=$?
    tail -3 "$_fqlog" | sed 's/^/   /'
    # SKIPS ARE THE DESIGN HERE, and this check predated it. Ten tests are collected -- five formats x dense and
    # grouped -- and the format gate lets exactly ONE format's two through, because no binary runs more than one.
    # So the correct shape is 2 passed / 8 skipped, and "any skip is a failure" (written before the gate existed)
    # called five green formats red. What must be distinguished is WHY something skipped:
    #     "this library is built for packed format N"  -> the gate, expected, 8 of them
    #     "are not in this build yet"                  -> the ops are missing, which is the real absence
    if grep -qi "are not in this build yet" "$_fqlog"; then
      echo "   ($_label ops not built yet -- not counted; see .coord/INBOX.md 012/013)"
    elif [ "$_r" -ne 0 ] || ! grep -Eq '(^| )6 passed' "$_fqlog"; then
      echo "   !!! the $_label fully_quantized oracles did not run to SIX passes -- dense, grouped, the merge premise, both BC GEMV arms, and BC dequant-all (exit $_r)"
      fqrc=1
    elif grep -qi "skipped" "$_fqlog" && ! grep -qi "built for packed format" "$_fqlog"; then
      echo "   !!! $_label skipped for a reason that is NOT the format gate -- read $_fqlog"
      fqrc=1
    else
      echo "   $_label: dense and grouped oracles PASSED on the device"
    fi
  done

  # RESTORE THE DEFAULT LIBRARY. build.sh writes the SAME path every time, so after the loop the .so on disk is
  # the LAST format built -- format 1, which answers rc=36 to every scale-first call. Anyone running a bare
  # pytest afterwards would watch a set of VALIDATED cells fail for a reason created by this script and gone
  # from its output. The repo's own handover warns about this shape ("two builds write the same path, so the
  # second overwrites the first"); it just had never applied to the python tier before.
  if [ "${_LAST_FMT_BUILT:-}" != "default" ]; then
    echo "-- restoring the default-format device library"
    timeout "$timeout_s" env PPU_DEFS="PPU_PACKED_SCALE=1" TARGET=quactlize_ppu "$ROOT/build.sh" \
      >"$OUT/quactlize_ppu.restore.log" 2>&1 \
      || { echo "   !!! could not restore the default library -- the tree is left on a format-specific build."
           echo "       Rebuild before trusting anything else: PPU_DEFS=PPU_PACKED_SCALE=1 TARGET=quactlize_ppu ./build.sh"
           fqrc=1; }
  fi

  # THE DEVICE-VS-CPU-ARM COMPARISON MUST NOT SKIP HERE. It skips honestly on a machine with no device, and this
  # is not one -- so a skip means the comparison it exists for did not happen, and pass 2 would go green having
  # never checked that the device arm computes the same function as the reference it is documented to match.
  echo "-- device vs its own CPU reference arm (a skip here is a failure)"
  local dlog drc
  dlog="$OUT/device_vs_cpu_arm.log"
  QUACTLIZE_PPU_LIB="$so" PYTHONPATH="$ROOT" python3 -m pytest "$ROOT/tests" -q -rs -m device_vs_cpu_arm \
    >"$dlog" 2>&1 && drc=0 || drc=$?
  tail -3 "$dlog" | sed 's/^/   /'
  if [ "$drc" -ne 0 ] || grep -qi "skipped" "$dlog" || ! grep -Eq '[1-9][0-9]* passed' "$dlog"; then
    echo "   !!! the device-vs-CPU-arm comparison did not run to a pass (exit $drc)"
    drc=1
  fi

  prc=$(( rc1 | rc2 | drc | fqrc ))

  # 6. the table, labelled as what it is
  echo
  echo "== support matrix: DECLARED, not measured by the run above =="
  echo "   Every status below is a literal in quactlize/schemes.py that a person wrote. This run does not promote"
  echo "   or demote any cell. What it CAN falsify is the line after the table."
  PYTHONPATH="$ROOT" python3 -m quactlize.schemes 2>&1 | sed 's/^/   /'
  echo
  echo "== registry cross-check: does every VALIDATED cell have a harness for THAT path =="
  PYTHONPATH="$ROOT" python3 -c '
import sys; sys.path.insert(0, "'"$ROOT"'")
from quactlize import schemes
p = schemes.check_against_registry()
print("   every VALIDATED cell has a harness behind it" if not p else "   !!! claims with no harness:")
[print("     " + x) for x in p]
sys.exit(1 if p else 0)' || prc=${prc:-1}

  [ "$prc" -eq 0 ] && echo && echo "== python tier passed on $backend ==" || echo && echo "== PYTHON TIER FAILED (pytest exit $prc) =="
  return "$prc"
}

do_perf() {
  echo
  echo "== perf: $BLOCKS interleaved blocks, $CFG S=1, band '$BAND' =="
  # WHY BLOCKS AND NOT REPETITIONS. The effect this bench exists to resolve is a few per cent, and the documented
  # same-config spread is ~13%. Three consecutive runs of one variant, then three of the next, measures each variant
  # under whatever the machine was doing during ITS window -- the two windows are not the same experiment. Blocks
  # interleave the variants and reverse their order on alternate blocks (ABBA), so a drift across the session falls
  # on every variant equally and cancels in the within-block ratio.
  #
  # EVERY OBSERVATION IS KEPT, in $OUT/perf_samples.txt as "block variant us". A summary that discards its samples
  # cannot be re-analysed, and the statistic worth reading -- the paired per-block ratio against base -- cannot be
  # recovered from a min and a max.
  #
  # THE FIRST QUESTION IS WHETHER THERE IS AN EFFECT AT ALL. Every pack-vs-base figure recorded before commit
  # 80dfeec compared two kernels computing DIFFERENT NUMBERS (see test_moe_splitk_bench.cu, "LIKE FOR LIKE"), so the
  # 12.9% that motivated this investigation is not a valid measurement and no like-for-like figure has replaced it.
  local samples="$OUT/perf_samples.txt"; : > "$samples"
  # AN ARRAY, NOT A STRING. Each VARIANT is "name:DEF1 DEF2 ..." and CONTAINS SPACES, so flattening with ${VAR[*]}
  # and iterating the result word-splits the defines into separate "variants". bash -n cannot see this.
  local b v n us out
  local -a order
  local -i i
  for b in $(seq 1 "$BLOCKS"); do
    order=("${VARIANTS[@]}")
    if [ $((b % 2)) -eq 0 ]; then                      # reverse on even blocks: ABBA, not AABB
      order=()
      for ((i=${#VARIANTS[@]}-1; i>=0; i--)); do order+=("${VARIANTS[i]}"); done
    fi
    for v in "${order[@]}"; do
      n="${v%%:*}"
      out=$(SPLITK_CFG="$CFG" SPLITK_S=1 "$OUT/test_moe_splitk_bench__$n" $BAND 2>&1 | grep -E "^  i4" | head -1)
      us=$(printf '%s' "$out" | grep -oE "[0-9]+\.[0-9]+ us" | head -1 | cut -d' ' -f1)
      if [ -z "$us" ]; then
        echo "   block $b $n: NO TIMING ROW -- $(printf '%s' "$out" | head -c 70)"; continue
      fi
      printf '%s %s %s\n' "$b" "$n" "$us" >> "$samples"
    done
    printf '   block %d/%d done\r' "$b" "$BLOCKS"
  done
  echo
  echo "   samples: $samples ($(wc -l < "$samples") observations)"
  echo
  # THE CONFIDENCE INTERVAL OF THE ESTIMATE, not the spread of the samples. The first version reported the 10th and
  # 90th percentiles of the per-block ratios and called an interval spanning 1.0 "not established". That is the wrong
  # statistic: it describes how far ONE observation scatters, which does not shrink with more blocks, so a genuine
  # 5% effect under 6% per-sample noise still "spanned 1.0" at thirty blocks while the median recovered it to 5.3%.
  # What shrinks with sqrt(n) is the uncertainty of the MEAN, and that is what decides whether an effect is real.
  #
  # Mean of the LOG ratios, with a normal-approximation interval: a ratio is multiplicative, so its errors are
  # symmetric in log space and not in linear space. exp(mean +- 1.96*sd/sqrt(n)) is the interval on the ratio.
  awk '
    { t[$1" "$2] = $3; if ($2 == "base") base[$1] = $3; v[$2] = 1; blk[$1] = 1 }
    END {
      printf "   %-11s %9s %9s   %s\n", "variant", "median us", "vs base", "95% CI on the paired ratio"
      for (name in v) {
        n = 0; sl = 0; sq = 0; su = 0
        for (b in blk) {
          k = b" "name
          if ((k in t) && (b in base) && base[b] > 0) {
            n++; lr = log(t[k]/base[b]); sl += lr; sq += lr*lr; su += t[k]
          }
        }
        if (n == 0) continue
        mu = sl/n; mean_us = su/n
        sd = (n > 1) ? sqrt((sq - n*mu*mu)/(n-1)) : 0
        se = (n > 1) ? sd/sqrt(n) : 0
        lo = exp(mu - 1.96*se); hi = exp(mu + 1.96*se)
        flag = (n < 2) ? "  <-- one block, no interval" : ((lo <= 1.0 && hi >= 1.0) ? "  <-- includes 1.0: no effect established" : "")
        printf "   %-11s %9.2f %8.1f%%   %.3f .. %.3f  (n=%d)%s\n", name, mean_us, (exp(mu)-1)*100, lo, hi, n, flag
      }
    }' "$samples" | sort -k2 -n
  cat <<'EOR'

   READ THE INTERVAL, NOT THE PERCENTAGE. A variant whose 95% interval includes 1.0 has no established difference
   from base at this block count -- raise BLOCKS, the interval narrows as sqrt(n). The paired ratio is the statistic
   because it is formed WITHIN a block, so drift across the session cancels; the mean-us column is context only.
EOR
  cat <<'EOT'

== what each difference isolates ==
   packfuse  - pack     THE STORE-CONFLICT FIX. pack adds +73,728 store conflicts over base -- 32 lanes storing
                        32 adjacent 2-byte values cover 16 of 32 banks two deep, and stores cannot broadcast.
                        Fusing makes it one 32-bit store per (n, group). Read it against PACK, not base.
                        Shared bytes do NOT change (4 KiB + 4 KiB is 8 KiB either way), so expect no
                        occupancy movement; if the time moves it is bank service and instruction count.
                        MEASURED AND ANSWERED, 2026-08-03 -- see CHECKPOINT.md 4a. It fused (tsm.st -26.83%,
                        tsm.ld -48.8%, total shared conflicts -48.30%) and the time moved +0.46%, because
                        Compute is at 38.99% and Memory at 29.87% of their roofs, so the shared work removed
                        was issuing in something else's shadow. It also PAYS ALU to save shared (v.shrl.i
                        +63.61%, v.or.i +89.16%, v.cnvt +720%) on the pipe that IS limiting. Does not ship.
                        READ BOTH `Shared Store` AND `Shared Store From Global Load`. A counter on the second
                        went 0 -> 36,864, exactly the store reduction: the fuse RELOCATED one conflict per
                        publication onto the global->shared path. Reading only the first shows a clean win.
   swz       - base       DELETED FROM THE PLAN. acu: it removed ZERO conflicts (278,528, +0.00%) while
                        Inst Executed Pipe SALU rose ~97% of peak. Kept only as the record of that.
   base      - bdqnop     what the BASELINE int4->fp16 dequant costs. Upper bound: the ablation also drops most of
                          the scale/zero LOADS, since only one fragment element stays live
   pack      - base       THE NATIVE-FORMAT TAX, AND IT IS CURRENTLY UNMEASURED. Every figure quoted before commit
                          80dfeec ("+12.9%", "+2.4%") came from a bench whose two paths computed DIFFERENT NUMBERS;
                          the two disagree with each other and their baselines differ by 17%. Nothing has replaced
                          them. This run is the first like-for-like measurement, not a re-measurement
   pack      - packnop    the packed decode's ARITHMETIC alone
   packnop   - base       its transport, the shared round trip, the explicit stores, and the barrier slot they
                          sit in. THE LEADING STRUCTURAL CANDIDATE if a tax exists at all
   (packsplit - pack) - (splitnop - packnop)
                          the placement effect with the duplicated-read cost subtracted out. packsplit - pack alone
                          cannot tell "no placement benefit" from "benefit smaller than the added read", which is
                          why splitnop exists and why the earlier "placement is dead" conclusion does not follow

   bdqnop and packnop produce DELIBERATELY WRONG numbers -- read their time, never their MATCH.

   A TIMING DIFFERENCE WITHOUT MATCHING acu COUNTERS IS NOT A STRUCTURAL ABLATION. packnop consumes only u[0], so
   the other three words' loads may be eliminated entirely; if its raw shared-load and store counts do not match
   pack's, the subtraction above is measuring a different kernel and not an ablation.
EOT
}

case "${1:-all}" in
  build) do_build ;;
  check) do_check ;;
  perf)  do_perf ;;
  pytest) do_pytest ;;
  # pytest is NOT in `all`: it builds a different target and needs torch, so a box without it would turn the
  # whole run red for a reason unrelated to the kernels. Run it explicitly.
  all)   do_build && do_check && do_perf ;;
  *)     echo "usage: $0 [build|check|perf|pytest|all]"; exit 2 ;;
esac
