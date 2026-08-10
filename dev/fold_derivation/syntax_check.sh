#!/usr/bin/env bash
# Run nvcc's FRONT END over a source locally, so a typo or a bad template instantiation does not need a round trip to
# ppu001 to be found. `nvcc -cuda` stops after the front end and inline PPU asm is an opaque string at that stage, so
# the file parses without an assembler for the target. -D__HGGCCC__ is required or CUTLASS_DEVICE degrades to host
# `inline` and every __syncthreads lands in host code.
#
# WHY BASELINE-DIFF AND NOT PATTERN FILTERING. The actlize headers emit a fixed set of complaints the real hgcc does
# not -- missing PPU intrinsics, host/device qualifiers, types the stubs do not model. Two earlier designs both failed:
#   * filtering by FILE ("only count errors attributed to the source") is blind to template instantiation failures,
#     because those report their `error:` against the library header and name the source only in the
#     "note: ... requested here" chain. That is how run_cfg<...,16,128,256,32,32,2> -- TM=16 with WM=32, so
#     warpOnM = 0 and the collective builder returns `int` -- reached the box while this script said "parses clean".
#   * filtering by PATTERN needs a list so loose it hides real errors, since the noise is large and generic
#     ("type name is not allowed", "expected a type specifier").
# So: record the noise ONCE per file into a baseline, and fail only on error signatures that are NEW.
#
#   ./syntax_check.sh --baseline <files...>   record/refresh the accepted noise
#   ./syntax_check.sh <files...>              fail on anything not in the baseline
set -u
# NOT `set -o pipefail`, deliberately: the nvcc call below is piped into grep, and a CLEAN compile makes that grep
# exit 1 because it matched nothing. pipefail would turn every clean file into a failure.
#
# But without it the pipeline's status is `sort`'s, which always succeeds -- so nvcc's own status was discarded, and
# with nvcc absent from PATH this script printed "clean (0 known-noise lines, 0 new)" and exited 0. Ten of the CI
# tier's twenty-two checks are this script. A gate that passes while compiling nothing is worse than no gate.
#
# The fix is not a better status check but a POSITIVE one: a clean verdict now requires evidence that the compiler
# ran -- it must have produced a non-empty output file. Absence of errors is not evidence of compilation.
command -v nvcc >/dev/null 2>&1 || {
  echo "syntax_check: nvcc is not on PATH -- this gate compiles, so it cannot report anything without a compiler" >&2
  exit 2
}
# The repo root, and the three directories a checkable source can live in. This used to be one directory
# up; the tree now separates kernels, tests and benchmarks, so -I has to name all of them or a harness
# fails on its own neighbour's header and the failure looks like the header being wrong.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/tests"
STUB="$(cd "$(dirname "$0")/stub_inc" && pwd)"
ACT="$(cd "$ROOT/third_party/actlize" && pwd)"
BLDIR="$(cd "$(dirname "$0")" && pwd)/syntax_baseline"
mkdir -p "$BLDIR"
RECORD=0
if [ "${1:-}" = "--baseline" ]; then RECORD=1; shift; fi
# EXTRA_DEFS lets a FLAG-ON variant get its own baseline, so a build that only breaks with a macro set is caught
# locally instead of on the box. Two box round trips were burned on errors this would have shown:
#   EXTRA_DEFS=-DPPU_B_CHUNK=1 ./syntax_check.sh --baseline <file>   then   EXTRA_DEFS=... ./syntax_check.sh <file>
EXTRA_DEFS="${EXTRA_DEFS:-}"
# GENERATED INCLUDES. A target whose sweep is generated (test_gemv_perf, test_moe_splitk_bench) includes a .inc
# that only exists in the build tree, so without this the front end stops at that include and the file is not
# checked at all. Point it at a directory holding the generated .inc:
#   GEN_INC=/path/to/gemv_units ./syntax_check.sh ../test_gemv_perf.cu
GEN_INC="${GEN_INC:-}"
_gen_flag=""
[ -n "$GEN_INC" ] && _gen_flag="-I$GEN_INC"
# NOTE the baseline file is deliberately NOT keyed on EXTRA_DEFS: a flag-on run is diffed against the flag-OFF
# baseline, so anything that appears only with the macro set shows up as NEW. Keying it would have let me baseline my
# own bugs.
FILES=${*:-"$SRC/test_fold_int2.cu"}
rc=0
for f in $FILES; do
  base=$(basename "$f")
  # signature = file + the MESSAGE, with the LINE NUMBER STRIPPED. Line numbers made the gate false-positive on
  # every edit that shifted the noise (adding 12 lines to test_fold_int2 reported 5 "NEW ERRORS" that were the same
  # 5 known ones), and a gate that cries wolf on every edit is a gate that stops being read -- which is how a real
  # error gets through. Dropping the line number costs the ability to distinguish two identical messages at
  # different lines; the count guard below covers that.
  # THE FLAGS MATTER MORE THAN THE SCRIPT. Without them the front end never instantiates the collective, so every
  # template-DEPENDENT error in the mainloop is invisible and only parse-time typos are caught:
  #   -arch=sm_80              without a real arch, __hfma2 is undeclared -> an error -> EDG stops instantiating
  #   --expt-relaxed-constexpr nvcc-only restriction on constexpr host fns in device code; clang/hgcc allow it
  #   -DPPU_FORCE_INSTANTIATE  odr-uses device_kernel<GemmKernel>, which pulls in the whole mainloop
  # Verified: with these, a static_assert planted in the 2-plane mainloop fires 32 times; without them, 0.
  #
  # THREE ERROR FORMS, READ OFF THE ACTUAL OUTPUT. The pattern was `: error`, so a source that could not even find its
# headers was reported CLEAN -- which happened to test_lowbit_moe_bench.cu the moment it started including a GENERATED
# file that is not on this gate's -I. The forms are:
#   path(line): error: ...                     EDG, the normal one
#   path(line): catastrophic error: ...        EDG, cannot open source file
#   path:line:col: fatal error: ...            the PREPROCESSOR, gcc-style -- this is the one a missing include gives
# I fixed this once by adding `catastrophic error` from memory and it still passed, because the actual line was the
# third form. All three share `: <kind>: `, so that is what is matched.
#
# Two signatures are dropped as ENVIRONMENTAL, not filtered by pattern-guessing: cute::_ and cute::product report
  # "undefined in device code" because CUTE_INLINE_CONSTANT resolves to `static constexpr` here while the box takes
  # the `static const __device__` branch. They cannot mask a real error -- a real one has a different message.
  # Emit to a real file, not /dev/null: the file is the PROOF the compiler ran. With -o /dev/null there is no
  # artifact, so "no errors" and "no compiler" produce identical output.
  _raw="$(mktemp)"; _out="$(mktemp -u)".cu.cpp
  # --error_limit=100000  NVCC STOPS AT 100 DIAGNOSTICS AND THAT SILENTLY BROKE THIS GATE. Measured 2026-08-10
  # over all 40 rows of the tier's SYNTAX list with the old flags: 34 hit the limit, produced NO artifact, and
  # were reported "clean (0 known-noise lines, 0 new)" -- because the first 100 diagnostics were all cute:: noise
  # and the filter below then emptied the list. A budget cannot be beaten by a better filter: every diagnostic
  # after the 100th is NEVER PRODUCED, so anything past it is invisible however the remainder is classified.
  # Raising the limit costs 2.7 s on the slowest row (27.1 -> 29.8 s) and takes the truncated count to 0/40.
  #
  # THE SHIM WAS TRIED AND IS WORSE, so that it does not get re-proposed: -include stub_inc/ppu_arch_shim.h
  # removes the cute:: noise at its source (5957 -> 164 errors) but the 164 that remain are actlize's inline-asm
  # constraint checks, which would all have to enter the baselines, AND it takes the rows that currently compile
  # to a full artifact from 6 to 1. Removing noise is not worth acquiring a vendor error floor.
  nvcc -std=c++17 -arch=sm_80 --expt-relaxed-constexpr -D__HGGCCC__ -DPPU_FORCE_INSTANTIATE=1 $EXTRA_DEFS $_gen_flag \
        -Xcudafe --error_limit=100000 \
        -I"$STUB" -I"$ACT/include" -I"$ACT/tools/util/include" -I"$SRC" -I"$ROOT/benchmarks" -I"$ROOT/quactlize/include" -I"$ROOT/dev" \
        -cuda -o "$_out" -x cu "$f" -Wno-deprecated-gpu-targets >"$_raw" 2>&1
  _rc=$?
  # A CLEAN VERDICT MUST BE EVIDENCE THAT THE FRONT END REACHED THE END OF THE FILE. Two forms of that evidence,
  # and one of them has to be present:
  #   the artifact          nvcc wrote the .cu.cpp, so it got all the way through;
  #   "N errors detected"   EDG prints this only after finishing the translation unit and counting.
  # "Error limit reached. Compilation terminated." is the opposite: it says diagnostics were DISCARDED, so no
  # baseline diff taken afterwards means anything. That was the old failure and it is now a refusal.
  if grep -q "Error limit reached" "$_raw"; then
    echo "$base: REFUSING -- nvcc hit its diagnostic budget, so an unknown number of errors were never emitted."
    echo "        A baseline diff over a truncated list cannot show anything NEW. Raise --error_limit."
    rm -f "$_raw" "$_out"; rc=1; continue
  fi
  if [ ! -s "$_out" ] && ! grep -qE "[0-9]+ errors? detected in the compilation of" "$_raw"; then
    echo "$base: REFUSING -- no artifact and no error count: there is no evidence the front end ran to the end."
    echo "        rc=$_rc. Absence of errors is not evidence of compilation."
    sed -n '1,3p' "$_raw" | sed 's/^/          /'
    rm -f "$_raw" "$_out"; rc=1; continue
  fi
  # THE TWO ENVIRONMENTAL SIGNATURES STAY DROPPED, and only now is that defensible. The header above argues they
  # cannot mask a real error because a real one has a different message; that was true of masking by SIMILARITY
  # and false of masking by BUDGET, which is what was actually happening. With the limit raised the filter no
  # longer decides what gets EMITTED, only what gets COUNTED -- so the argument holds as originally written.
  # Measured on the worst row: 5957 errors, of which 5957 are these two and 0 are anything else.
  sig=$(cat "$_raw" \
        | grep -E ": (error|fatal error|catastrophic error):" | grep -v 'identifier "cute::_" is undefined in device code' \
                         | grep -v 'identifier "cute::product" is undefined in device code' \
        | sed -E 's#^.*/([^/]+)#\1#; s#\(([0-9]+)\)#()#' | sort | uniq -c \
        | sed -E 's/^ +//' | sort)
  rm -f "$_raw" "$_out"
  bl="$BLDIR/$base.txt"
  # A FATAL ERROR MEANS NOTHING WAS CHECKED, so it must never become "accepted noise". The preprocessor stops at
  # the first one -- a missing generated .inc gives exactly this -- and baselining it turns the check into a
  # no-op that reports success forever. This happened: `--baseline` on a target whose sweep is generated recorded
  # `fatal error: moe_splitk_units.inc: No such file or directory` as one accepted line, so the file was never
  # parsed and the script still said "baseline recorded". Fixing the include (GEN_INC) is the answer, not
  # accepting the line.
  if printf '%s\n' "$sig" | grep -q ": fatal error:\|: catastrophic error:"; then
    echo "$base: REFUSING to proceed -- a fatal/catastrophic error means the front end STOPPED and the file was"
    echo "        not checked. Fix the include (see GEN_INC) rather than baselining it:"
    printf '%s\n' "$sig" | grep ": fatal error:\|: catastrophic error:" | head -4 | sed 's/^/          /'
    rc=1; continue
  fi
  if [ "$RECORD" = 1 ]; then
    printf '%s\n' "$sig" > "$bl"
    echo "$base: baseline recorded ($(printf '%s\n' "$sig" | grep -c . ) accepted noise lines)"
    continue
  fi
  if [ ! -f "$bl" ]; then echo "$base: NO BASELINE -- run --baseline once, then review it"; rc=1; continue; fi
  new=$(comm -13 "$bl" <(printf '%s\n' "$sig"))
  if [ -n "$new" ]; then
    echo "$base: NEW ERRORS (not in baseline)"; printf '%s\n' "$new" | head -12; rc=1
  else
    echo "$base: clean ($(grep -c . "$bl") known-noise lines, 0 new)"
  fi
done
exit $rc
