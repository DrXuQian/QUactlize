#!/usr/bin/env bash
# ONE RUN FOR EVERY OPEN QUESTION. Six binaries, built once, kept, then measured on one pinned configuration.
#
# WHY A SCRIPT AND NOT A COMMAND BLOCK. Every trap this task has hit is a step that is easy to drop by hand:
#   * build.sh `rm -rf`s the SAME output directory, so a binary must be copied out BEFORE the next build starts;
#     forget it and you compare a binary with itself, which is indistinguishable from "the change does nothing"
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
#   ./run_batch.sh build     build all six and copy them out (slow: ~6 compiles)
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
EX="${EX:-$ROOT/third_party/actlize/build_w4a16_compare/examples/99_kernels_w4a16_compare}"
OUT="${OUT:-$HOME/ab}"
BAND="${BAND:-64 8 2048 2048 32 3}"                 # L=64, top-k 8, N=K=2048, gs=32, decode
CFG="${CFG:-16x128:256 w16x16 s2}"                  # the pinned row; SPLITK_S below pins the slice count
REPS="${REPS:-3}"                                    # repetitions per variant; REPS=1 restores the old single-sample run
mkdir -p "$OUT"

# name : defines.  SK_QUANT=2 is FinegrainedScaleZero, what ships.
VARIANTS=(
  "base:SK_QUANT=2"
  "swz:SK_QUANT=2 PPU_SCALE_SWIZZLE=1"
  "bdqnop:SK_QUANT=2 PPU_B_DEQUANT_NOP=1"
  "pack:SK_QUANT=2 PPU_PACKED_SCALE=1"
  "packnop:SK_QUANT=2 PPU_PACKED_SCALE=1 PPU_PACKED_SCALE_NOP=1"
  "packsplit:SK_QUANT=2 PPU_PACKED_SCALE=1 PPU_PACKED_SPLIT_GROUPS=1"
  # splitnop prices the split's OWN added cost -- the duplicated 16 B unit read and the duplicated per-column setup --
  # without the decode arithmetic. packsplit alone could not tell "the placement benefit is zero" from "the added
  # cost exceeded it", and the difference of differences can:
  #     (packsplit - pack) - (splitnop - packnop)
  "splitnop:SK_QUANT=2 PPU_PACKED_SCALE=1 PPU_PACKED_SPLIT_GROUPS=1 PPU_PACKED_SCALE_NOP=1"
)

build_one() {  # $1 name  $2 defines  $3 target
  # SEPARATE STATEMENTS. `local a=$1 log=...$a...` fails under `set -u` in bash 5.1: local declares every name first
  # and only then assigns, so the self-reference is unbound. bash -n does not catch it -- only running does, which is
  # why this script is now dry-run against a stubbed build.sh before it ships.
  local name="$1" defs="$2" tgt="$3"
  local log="$OUT/build_$name.log"
  printf '  %-10s %s\n' "$name" "${defs:-<no defines>}"
  local rc=0
  ( cd "$ROOT" && PPU_DEFS="$defs" TARGET="$tgt" ./build.sh ) >"$log" 2>&1 || rc=$?

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
  if [ -n "$defs" ] && ! grep -q "PPU_DEFS verified on $tgt" "$log"; then
    echo "    FAILED: no 'PPU_DEFS verified' line in $log -- the macro did not reach the device compile"
    grep -E "WARNING|ERROR" "$log" | head -3 | sed 's/^/      /'
    return 1
  fi
  [ -x "$EX/$tgt" ] || { echo "    FAILED: $EX/$tgt not built (see $log)"; return 1; }
  cp "$EX/$tgt" "$OUT/${tgt}__$name"                 # BEFORE the next build deletes it
}

do_build() {
  echo "== build =="
  local ok=1
  for v in "${VARIANTS[@]}"; do build_one "${v%%:*}" "${v#*:}" test_moe_splitk_bench || ok=0; done
  # the correctness gate needs its own target, packed on, plus the two variants that change the packed path
  build_one gate_pack  "PPU_PACKED_SCALE=1"                            test_q4k_packed_gemm || ok=0
  build_one gate_split "PPU_PACKED_SCALE=1 PPU_PACKED_SPLIT_GROUPS=1"  test_q4k_packed_gemm || ok=0
  build_one gate_swz   "PPU_PACKED_SCALE=1 PPU_SCALE_SWIZZLE=1"        test_q4k_packed_gemm || ok=0
  build_one gate_swz_fp16 "PPU_SCALE_SWIZZLE=1"                        test_q4k_packed_gemm || ok=0
  build_one gate_swz_only "PPU_SCALE_SWIZZLE=1"                        test_moe_grouped_verify || ok=0
  # THE CONTROL THAT WAS MISSING. test_moe_grouped_verify died with a device assert under PPU_SCALE_SWIZZLE=1 and I
  # read that as the swizzle's doing -- but the verifier hardcodes Stages = 3 and the swizzle is gated on a power of
  # two, so it was never active in that launch. The assert sits on the COARSE scale path, reached when
  # ceil(TileK/gs) <= TileK/64, which the verifier's default gs=128 satisfies and the bench's gs=32 never does. If
  # this default build dies identically the flag is exonerated; if only the swizzle build dies, something is
  # macro-sensitive and neither explanation stands.
  build_one verify_default ""                                          test_moe_grouped_verify || ok=0
  echo "== distinct-binary check =="
  # Two identical binaries mean an A/B that compares something with itself.
  #
  # COUNT THEM FIRST. When every build_one failed, no `cp` ran, the glob matched nothing, md5sum read stdin, uniq -d
  # produced nothing, and this printed "all six splitk binaries differ" over an empty directory -- a gate reporting
  # success about files that do not exist, immediately below eleven FAILED lines.
  local want=${#VARIANTS[@]} have
  have=$(ls "$OUT"/test_moe_splitk_bench__* 2>/dev/null | wc -l)
  if [ "$have" -ne "$want" ]; then
    echo "  !!! expected $want splitk binaries, found $have -- nothing below this can be compared"; ok=0
  else
    local dup
    dup=$(md5sum "$OUT"/test_moe_splitk_bench__* | awk '{print $1}' | sort | uniq -d)
    if [ -n "$dup" ]; then echo "  !!! two splitk binaries are IDENTICAL -- the A/B is invalid"; md5sum "$OUT"/test_moe_splitk_bench__*; ok=0
    else echo "  all $have splitk binaries differ"; fi
  fi
  return $((1-ok))
}

# EVERY BINARY THIS RUN NEEDS, and it must have been produced by THIS checkout. Without the second half, do_check
# happily ran binaries left in $OUT by an earlier session: a build round in which every build_one failed -- so no `cp`
# ran at all -- still produced five green rowC MATCH lines, from binaries of unknown provenance, while the two
# variants added since that older build reported "No such file or directory". Five passes and two 127s side by side,
# and the passes were the misleading half.
require_fresh_binaries() {
  local newest missing=() stale=() b
  newest=$(cd "$ROOT" && git ls-files -z | xargs -0 stat -c %Y 2>/dev/null | sort -n | tail -1)
  for b in "$@"; do
    if [ ! -x "$OUT/$b" ]; then missing+=("$b")
    elif [ -n "$newest" ] && [ "$(stat -c %Y "$OUT/$b")" -lt "$newest" ]; then stale+=("$b"); fi
  done
  if [ ${#missing[@]} -ne 0 ]; then
    echo "  !!! not built: ${missing[*]}"
    echo "      run './run_batch.sh build' and read its output -- a build that failed leaves the PREVIOUS run's"
    echo "      binaries in $OUT, and results from those describe code nobody is running."
    return 1
  fi
  if [ ${#stale[@]} -ne 0 ]; then
    echo "  !!! older than the sources: ${stale[*]}"
    echo "      rebuild before reading anything below."
    return 1
  fi
  return 0
}

do_check() {
  echo
  echo "== correctness (nothing below matters until these pass) =="
  require_fresh_binaries \
    test_q4k_packed_gemm__gate_pack test_q4k_packed_gemm__gate_split test_q4k_packed_gemm__gate_swz \
    test_q4k_packed_gemm__gate_swz_fp16 test_moe_grouped_verify__gate_swz_only \
    test_moe_grouped_verify__verify_default || { echo "== aborting: the binaries are not this checkout's =="; return 1; }
  # rowC is the only row where Scale_TileK == 8, i.e. the only one on the packed path. It has been INTERMITTENT --
  # bad=128, then 724, then MATCH with no semantic source change -- and the cause is still unknown, so it is run five
  # times. A single pass proves nothing about a flaky failure.
  local fails=0 out rc
  echo "-- packed, five runs (rowC has been intermittent; the cause is NOT known)"
  for i in 1 2 3 4 5; do
    printf '   run %d: ' "$i"
    # STATUS FIRST, filtering second: piping straight into grep reports grep's status and loses the harness's.
    out=$("$OUT/test_q4k_packed_gemm__gate_pack" "$ROOT/tests/data/q4k_packed.bin" 2>&1) && rc=0 || rc=$?
    printf '%s' "$out" | grep -E "rowC|== (PASS|FAIL)" | tr '\n' ' '; echo "  [exit $rc]"
    [ "$rc" -eq 0 ] || fails=$((fails+1))
  done
  for g in split swz; do
    echo "-- packed + $g"
    out=$("$OUT/test_q4k_packed_gemm__gate_$g" "$ROOT/tests/data/q4k_packed.bin" 2>&1) && rc=0 || rc=$?
    printf '%s\n' "$out" | grep -E "rowA|rowB|rowC|== (PASS|FAIL)"; echo "   [exit $rc]"
    [ "$rc" -eq 0 ] || fails=$((fails+1))
  done
  # The swizzle changes ADDRESSES, not values. Any mismatch here means a view of the scale buffer that does not carry
  # it -- i.e. the kernel writes at one address and reads at another.
  echo "-- swizzle alone on rowC's fp16 planes (the swizzle IS active there: TN=128, Stages=2)"
  out=$("$OUT/test_q4k_packed_gemm__gate_swz_fp16" "$ROOT/tests/data/q4k_packed.bin" 2>&1) && rc=0 || rc=$?
  printf '%s\n' "$out" | grep -E "rowA|rowB|rowC|== (PASS|FAIL)"; echo "   [exit $rc]"
  [ "$rc" -eq 0 ] || fails=$((fails+1))
  echo "-- the assert control: default build of the same verifier, no macros at all"
  out=$("$OUT/test_moe_grouped_verify__verify_default" 8 1 2>&1) && rc=0 || rc=$?
  printf '%s\n' "$out" | tail -3; echo "   [exit $rc]  <-- if this dies too, the assert is not the swizzle's"
  echo "-- swizzle alone, on the grouped verifier (addresses change, numbers must not)"
  out=$("$OUT/test_moe_grouped_verify__gate_swz_only" 8 1 2>&1) && rc=0 || rc=$?
  printf '%s\n' "$out" | tail -6; echo "   [exit $rc]"
  [ "$rc" -eq 0 ] || fails=$((fails+1))
  # THIS GATE IS CURRENTLY VACUOUS and is labelled rather than trusted: test_moe_grouped_verify hardcodes Stages = 3
  # and the swizzle is gated on a power-of-two Stages, so PPU_SCALE_SWIZZLE changes no address in this launch. A gate
  # that measures nothing is worse than no gate once it is believed to have passed.
  if [ "$fails" -ne 0 ]; then
    echo "== $fails correctness gate(s) FAILED -- the timings below would be meaningless =="; return 1
  fi
  echo "== all correctness gates passed =="
}

do_perf() {
  echo
  echo "== perf: $REPS run(s) of one pinned row, $CFG S=1, band '$BAND' =="
  echo "   min is the statistic to read: it is the least-disturbed sample, and the spread column says whether the"
  echo "   difference between two variants is larger than the noise within either of them."
  local v n i us best worst out
  for v in "${VARIANTS[@]}"; do
    n="${v%%:*}"
    best=""; worst=""
    for i in $(seq 1 "$REPS"); do
      out=$(SPLITK_CFG="$CFG" SPLITK_S=1 "$OUT/test_moe_splitk_bench__$n" $BAND 2>&1 | grep -E "^  i4" | head -1)
      us=$(printf '%s' "$out" | grep -oE "[0-9]+\.[0-9]+ us" | head -1 | cut -d' ' -f1)
      [ -z "$us" ] && { printf -- '-- %-10s NO TIMING ROW: %s\n' "$n" "$(printf '%s' "$out" | head -c 80)"; best=""; break; }
      best=$(printf '%s\n%s\n' "$best" "$us" | grep -v '^$' | sort -g | head -1)
      worst=$(printf '%s\n%s\n' "$worst" "$us" | grep -v '^$' | sort -g | tail -1)
    done
    [ -z "$best" ] && continue
    printf -- '-- %-10s min %8s us   max %8s us   spread %s%%\n' "$n" "$best" "$worst" \
      "$(awk -v a="$best" -v b="$worst" 'BEGIN{printf "%.1f", (b-a)/a*100}')"
  done
  cat <<'EOT'

== how to read it ==
   base                     the shipped path
   swz     - base           the scale read's bank conflicts, 4-way -> 1-way (l98). Attacks the 6.6% that
                            PPU_SCALE_PREFETCH could not: SK_QUANT=0 prices the whole reload at 7.3% and
                            prefetch, which removes only the WAITING, recovered 0.7%
   base    - bdqnop         what the BASELINE int4->fp16 dequant costs. Upper bound: the ablation also drops
                            most of the scale/zero LOADS, since only one fragment element stays live
   pack    - base           the native-format tax as it stands (+12.9% at last measurement)
   pack    - packnop        the packed decode's ARITHMETIC alone
   packnop - base           its transport, its explicit stores, and the barrier slot they sit in
   packsplit - pack         eight warps decoding four groups each instead of four decoding eight. Better means the
                            publication barrier's critical path costs; no change means aggregate issue demand does

   bdqnop and packnop produce DELIBERATELY WRONG numbers -- read their time, never their MATCH.

   AND READ THE SPREAD COLUMN FIRST. If a variant's own spread is comparable to its difference from base, that
   difference has not been measured -- it has been sampled once from a distribution wide enough to contain it.
   Raise REPS before drawing any conclusion from a gap under ~15%.
EOT
}

case "${1:-all}" in
  build) do_build ;;
  check) do_check ;;
  perf)  do_perf ;;
  all)   do_build && do_check && do_perf ;;
  *)     echo "usage: $0 [build|check|perf|all]"; exit 2 ;;
esac
