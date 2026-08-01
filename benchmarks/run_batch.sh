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
BLOCKS="${BLOCKS:-10}"                               # interleaved measurement blocks; see do_perf
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
  # and iterating the result word-splits the defines into separate "variants" -- the dry-run showed blocks running
  # PPU_SCALE_SWIZZLE=1 and PPU_PACKED_SCALE=1 as if they were binaries. bash -n cannot see this.
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
   swz       - base       the scale read's bank conflicts, 4-way -> 1-way (l98)
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
  all)   do_build && do_check && do_perf ;;
  *)     echo "usage: $0 [build|check|perf|all]"; exit 2 ;;
esac
