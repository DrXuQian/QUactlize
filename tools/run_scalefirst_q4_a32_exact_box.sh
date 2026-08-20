#!/usr/bin/env bash
# Build and run only the Q4_K/A32 row that first exposed the folded-reader
# numeric defect. This is a correctness closure, not a sweep.
set -uo pipefail

classify_arm() {
  local fixture="$1" rc="$2" log="$3"
  if [ "$rc" -eq 0 ] &&
     grep -q "SF_COMPLETE status=COMPLETE shape=64x1024x5120 typed_rows=1.*fixture_mode=${fixture}.*isolation_coverage=PASS" "$log" &&
     grep -q 'raw_bad":0' "$log"; then
    printf 'PASS'
  elif [ "$rc" -ne 0 ] &&
       grep -q 'SF_FATAL .*state=RAW_FP16_MISMATCH' "$log"; then
    printf 'NUMERIC_FAIL'
  else
    printf 'INFRA_FAIL'
  fi
}

classify_locus() {
  local code_state="$1" metadata_state="$2" exact_state="$3"
  if [ "$code_state" = PASS ] && [ "$metadata_state" = PASS ] &&
     [ "$exact_state" = PASS ]; then
    printf 'CLOSED'
  elif [ "$code_state" = NUMERIC_FAIL ] && [ "$metadata_state" = PASS ]; then
    printf 'CODE_READER_OR_B_PIPELINE'
  elif [ "$code_state" = PASS ] && [ "$metadata_state" = NUMERIC_FAIL ]; then
    printf 'METADATA_LOAD_OR_APPLY'
  elif [ "$code_state" = PASS ] && [ "$metadata_state" = PASS ] &&
       [ "$exact_state" = NUMERIC_FAIL ]; then
    printf 'CODE_METADATA_INTERACTION_OR_PIPELINE'
  elif [ "$code_state" = NUMERIC_FAIL ] &&
       [ "$metadata_state" = NUMERIC_FAIL ]; then
    printf 'COMMON_PIPELINE_OR_MULTIPLE_DEFECTS'
  else
    printf 'UNREGISTERED_COMBINATION'
  fi
}

self_test() {
  local root out pass_log fail_log
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  out="${OUT:-/workspace/quactlize-q4-a32-classifier-self-test}"
  case "$out" in
    /workspace/*) ;;
    *) return 2 ;;
  esac
  mkdir -p "$out" || return 2
  pass_log="$out/pass.log"
  fail_log="$out/fail.log"
  printf '%s\n' \
    'SF_CELL {"raw_bad":0}' \
    'SF_COMPLETE status=COMPLETE shape=64x1024x5120 typed_rows=1 fixture_mode=exact isolation_coverage=PASS' \
    >"$pass_log"
  printf '%s\n' \
    'SF_FATAL symbol=x state=RAW_FP16_MISMATCH raw_bad=1' \
    >"$fail_log"
  [ "$(classify_arm exact 0 "$pass_log")" = PASS ] || return 1
  [ "$(classify_arm exact 1 "$fail_log")" = NUMERIC_FAIL ] || return 1
  [ "$(classify_arm exact 0 "$fail_log")" = INFRA_FAIL ] || return 1
  [ "$(classify_locus PASS PASS PASS)" = CLOSED ] || return 1
  [ "$(classify_locus NUMERIC_FAIL PASS NUMERIC_FAIL)" = CODE_READER_OR_B_PIPELINE ] || return 1
  [ "$(classify_locus PASS NUMERIC_FAIL NUMERIC_FAIL)" = METADATA_LOAD_OR_APPLY ] || return 1
  [ "$(classify_locus PASS PASS NUMERIC_FAIL)" = CODE_METADATA_INTERACTION_OR_PIPELINE ] || return 1
  [ "$(classify_locus NUMERIC_FAIL NUMERIC_FAIL NUMERIC_FAIL)" = COMMON_PIPELINE_OR_MULTIPLE_DEFECTS ] || return 1
  printf '[q4-a32-exact:self-test] PASS: arm and three-way locus classifier\n'
}

main() {
  local root sha short stamp out generated build binary log symbol rc fixtures
  local fixture arm state code_state metadata_state exact_state locus
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="${OUT:-/workspace/quactlize-q4-a32-exact-${short}-${stamp}}"
  case "$out" in
    /workspace/*) ;;
    *) printf '[q4-a32-exact] FAIL: OUT must be below /workspace: %s\n' "$out" >&2; return 2 ;;
  esac
  if [ -e "$out" ]; then
    printf '[q4-a32-exact] FAIL: refusing existing OUT=%s\n' "$out" >&2
    return 2
  fi
  mkdir -p "$out/generated" "$out/build" "$out/results" || return 2
  symbol=sf_q12_a32_tm64_tn64_tk128_wm16_wn32_s8_bc0
  generated="$out/generated/q12-a32-bc0-exact"

  python3 -B "$root/tools/gen_scalefirst_internal_units.py" \
    --qtype 12 --artifact-tk 32 --bchunk 0 --per-unit 1 \
    --select-symbol "$symbol" --out-dir "$generated" \
    >"$out/results/generate.log" 2>&1 || {
      tail -80 "$out/results/generate.log" >&2
      return 1
    }

  build="$out/build/q12-a32-bc0-exact"
  log="$out/results/build.log"
  (cd "$root" && PPU_BUILD_DIR="$build" PPU_ARCHS=ppu0010 \
    JOBS="${JOBS:-16}" TARGET=test_scalefirst_internal_sweep \
    SCALEFIRST_SWEEP_GENERATED_DIR="$generated" \
    SCALEFIRST_SWEEP_QTYPE=12 SCALEFIRST_SWEEP_ARTIFACT_TK=32 \
    SCALEFIRST_SWEEP_BCHUNK=0 ./build.sh) >"$log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    printf '[q4-a32-exact] FAIL: exact target build rc=%s\n' "$rc" >&2
    tail -120 "$log" >&2
    return "$rc"
  fi
  binary="$build/ppu_targets/test_scalefirst_internal_sweep"
  if [ ! -x "$binary" ]; then
    printf '[q4-a32-exact] FAIL: exact binary missing: %s\n' "$binary" >&2
    return 1
  fi
  sha256sum "$binary" >"$out/results/binary.sha256"
  printf '%s\n' "$sha" >"$out/results/git.sha"
  printf '[q4-a32-exact] sha=%s symbol=%s binary=%s\n' "$sha" "$symbol" "$binary"

  fixtures="${FIXTURES:-exact}"
  fixtures="${fixtures//,/ }"
  code_state=NOT_RUN
  metadata_state=NOT_RUN
  exact_state=NOT_RUN
  for fixture in $fixtures; do
    case "$fixture" in
      exact|code-only|metadata-only) ;;
      *)
        printf '[q4-a32-exact] FAIL: unknown FIXTURES member: %s\n' "$fixture" >&2
        return 2
        ;;
    esac
    arm="${fixture//-/_}"
    log="$out/results/${fixture}.log"
    set +e
    "$binary" --shape=64x1024x5120 --iterations="${ITERATIONS:-3}" \
      --correctness-repeats="${CORRECTNESS_REPEATS:-8}" \
      --algorithm=nonpersistent --fixture="$fixture" \
      >"$log" 2>&1
    rc=$?
    set -e
    cat "$log"
    state="$(classify_arm "$fixture" "$rc" "$log")"
    printf 'Q4_A32_ARM fixture=%s state=%s rc=%s\n' "$fixture" "$state" "$rc"
    case "$arm" in
      code_only) code_state="$state" ;;
      metadata_only) metadata_state="$state" ;;
      exact) exact_state="$state" ;;
    esac
    if [ "$state" = INFRA_FAIL ]; then
      printf '[q4-a32-exact] FAIL: %s arm was not classifiable; artifacts=%s\n' \
        "$fixture" "$out" >&2
      return 1
    fi
  done

  if [ "$code_state" != NOT_RUN ] && [ "$metadata_state" != NOT_RUN ] &&
     [ "$exact_state" != NOT_RUN ]; then
    locus="$(classify_locus "$code_state" "$metadata_state" "$exact_state")"
    printf 'Q4_A32_BISECT code_only=%s metadata_only=%s exact=%s locus=%s\n' \
      "$code_state" "$metadata_state" "$exact_state" "$locus"
    printf '[q4-a32-exact] DIAGNOSTIC_COMPLETE: three-arm locus classified; artifacts=%s\n' "$out"
    return 0
  fi

  if [ "$exact_state" = PASS ]; then
    printf '[q4-a32-exact] PASS: exact row raw-bit closed; artifacts=%s\n' "$out"
    return 0
  fi
  printf '[q4-a32-exact] FAIL: exact numeric target state=%s artifacts=%s\n' \
    "$exact_state" "$out" >&2
  return 1
}

if [ "${Q4_A32_CLASSIFIER_SELF_TEST:-0}" = 1 ]; then
  self_test
else
  main "$@"
fi
