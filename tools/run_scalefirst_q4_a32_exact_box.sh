#!/usr/bin/env bash
# Build and run only the Q4_K/A32 row that first exposed the folded-reader
# numeric defect. This is a correctness closure, not a sweep.
set -uo pipefail

classify_arm() {
  local fixture="$1" rc="$2" log="$3" expected marker_count
  case "$fixture" in
    exact) expected=0xc200 ;;
    code-only) expected=0x4000 ;;
    scale-only) expected=0x3800 ;;
    zero-only) expected=0xc200 ;;
    metadata-only) expected=0xc100 ;;
    transport-only) expected=0x4400 ;;
    *) printf 'INFRA_FAIL'; return ;;
  esac
  marker_count="$(grep -c "^SF_FIXTURE mode=${fixture} first_golden=${expected} .*roundtrip=1 exact=1 isolation=1$" "$log" || true)"
  if [ "$marker_count" -eq 1 ] && [ "$rc" -eq 0 ] &&
     grep -q "SF_COMPLETE status=COMPLETE shape=64x1024x5120 typed_rows=1.*fixture_mode=${fixture}.*isolation_coverage=PASS" "$log" &&
     grep -q 'raw_bad":0' "$log"; then
    printf 'PASS'
  elif [ "$marker_count" -eq 1 ] && [ "$rc" -ne 0 ] &&
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

legacy_signature_present() {
  local log="$1"
  grep -Eq '^SF_FATAL .*shape=64x1024x5120 .*state=RAW_FP16_MISMATCH .*raw_bad=61184 first_bad=0 want=0xc200 got=0xcc80$' "$log"
}

build_exact_variant() {
  local root="$1" generated="$2" out="$3" label="$4" defs="$5"
  local build log rc binary def
  build="$out/build/q12-a32-bc0-${label}"
  log="$out/results/build-${label}.log"
  (cd "$root" && PPU_BUILD_DIR="$build" PPU_ARCHS=ppu0010 \
    PPU_DEFS="$defs" JOBS="${JOBS:-16}" \
    TARGET=test_scalefirst_internal_sweep \
    SCALEFIRST_SWEEP_GENERATED_DIR="$generated" \
    SCALEFIRST_SWEEP_QTYPE=12 SCALEFIRST_SWEEP_ARTIFACT_TK=32 \
    SCALEFIRST_SWEEP_BCHUNK=0 ./build.sh) >"$log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    printf '[q4-a32-closure] FAIL: %s target build rc=%s\n' \
      "$label" "$rc" >&2
    tail -120 "$log" >&2
    return "$rc"
  fi
  for def in $defs; do
    if ! grep -Fq "PPU_DEFS verified on test_scalefirst_internal_sweep's compile command: -D${def}" "$log"; then
      printf '[q4-a32-closure] FAIL: %s did not prove -D%s reached the exact target\n' \
        "$label" "$def" >&2
      tail -80 "$log" >&2
      return 1
    fi
  done
  binary="$build/ppu_targets/test_scalefirst_internal_sweep"
  if [ ! -x "$binary" ]; then
    printf '[q4-a32-closure] FAIL: %s binary missing: %s\n' \
      "$label" "$binary" >&2
    return 1
  fi
  sha256sum "$binary" >"$out/results/binary-${label}.sha256"
  printf '%s\n' "$binary"
}

run_exact_variant() {
  local binary="$1" out="$2" label="$3" log rc state
  log="$out/results/exact-${label}.log"
  set +e
  "$binary" --shape=64x1024x5120 --iterations="${ITERATIONS:-1}" \
    --correctness-repeats="${CORRECTNESS_REPEATS:-8}" \
    --algorithm=nonpersistent --fixture=exact --fixture-binding \
    >"$log" 2>&1
  rc=$?
  set -e
  cat "$log"
  state="$(classify_arm exact "$rc" "$log")"
  printf '%s\n' "$state" >"$out/results/exact-${label}.state"
  printf 'Q4_A32_CLOSURE_ARM variant=%s state=%s rc=%s binary_sha256=%s\n' \
    "$label" "$state" "$rc" \
    "$(cut -d' ' -f1 "$out/results/binary-${label}.sha256")"
  [ "$state" != INFRA_FAIL ]
}

self_test() {
  local root out pass_log fail_log metadata_log bad_metadata_log
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  out="${OUT:-/workspace/quactlize-q4-a32-classifier-self-test}"
  case "$out" in
    /workspace/*) ;;
    *) return 2 ;;
  esac
  mkdir -p "$out" || return 2
  pass_log="$out/pass.log"
  fail_log="$out/fail.log"
  metadata_log="$out/metadata.log"
  bad_metadata_log="$out/bad-metadata.log"
  printf '%s\n' \
    'SF_FIXTURE mode=exact first_golden=0xc200 fixture roundtrip=1 exact=1 isolation=1' \
    'SF_CELL {"raw_bad":0}' \
    'SF_COMPLETE status=COMPLETE shape=64x1024x5120 typed_rows=1 fixture_mode=exact isolation_coverage=PASS' \
    >"$pass_log"
  printf '%s\n' \
    'SF_FIXTURE mode=exact first_golden=0xc200 fixture roundtrip=1 exact=1 isolation=1' \
    'SF_FATAL symbol=x state=RAW_FP16_MISMATCH raw_bad=1' \
    >"$fail_log"
  printf '%s\n' \
    'SF_FIXTURE mode=metadata-only first_golden=0xc100 fixture roundtrip=1 exact=1 isolation=1' \
    'SF_FATAL symbol=x state=RAW_FP16_MISMATCH raw_bad=1' \
    >"$metadata_log"
  printf '%s\n' \
    'SF_FIXTURE mode=metadata-only first_golden=0x4000 fixture roundtrip=1 exact=1 isolation=1' \
    'SF_FATAL symbol=x state=RAW_FP16_MISMATCH raw_bad=1' \
    >"$bad_metadata_log"
  [ "$(classify_arm exact 0 "$pass_log")" = PASS ] || return 1
  [ "$(classify_arm exact 1 "$fail_log")" = NUMERIC_FAIL ] || return 1
  [ "$(classify_arm exact 0 "$fail_log")" = INFRA_FAIL ] || return 1
  [ "$(classify_arm metadata-only 1 "$fail_log")" = INFRA_FAIL ] || return 1
  [ "$(classify_arm metadata-only 1 "$metadata_log")" = NUMERIC_FAIL ] || return 1
  [ "$(classify_arm metadata-only 1 "$bad_metadata_log")" = INFRA_FAIL ] || return 1
  [ "$(classify_locus PASS PASS PASS)" = CLOSED ] || return 1
  [ "$(classify_locus NUMERIC_FAIL PASS NUMERIC_FAIL)" = CODE_READER_OR_B_PIPELINE ] || return 1
  [ "$(classify_locus PASS NUMERIC_FAIL NUMERIC_FAIL)" = METADATA_LOAD_OR_APPLY ] || return 1
  [ "$(classify_locus PASS PASS NUMERIC_FAIL)" = CODE_METADATA_INTERACTION_OR_PIPELINE ] || return 1
  [ "$(classify_locus NUMERIC_FAIL NUMERIC_FAIL NUMERIC_FAIL)" = COMMON_PIPELINE_OR_MULTIPLE_DEFECTS ] || return 1
  printf '%s\n' \
    'SF_FIXTURE mode=exact first_golden=0xc200 fixture roundtrip=1 exact=1 isolation=1' \
    'SF_FATAL symbol=x shape=64x1024x5120 algorithm=NONPERSISTENT state=RAW_FP16_MISMATCH step=RAW_FP16_MISMATCH repeat=0 raw_bad=61184 first_bad=0 want=0xc200 got=0xcc80' \
    >"$out/legacy-signature.log"
  legacy_signature_present "$out/legacy-signature.log" || return 1
  sed 's/raw_bad=61184/raw_bad=61183/' "$out/legacy-signature.log" \
    >"$out/wrong-legacy-signature.log"
  ! legacy_signature_present "$out/wrong-legacy-signature.log" || return 1
  printf '[q4-a32-exact:self-test] PASS: arm/locus classifier and exact historical signature; wrong-label/golden/signature plants red\n'
}

run_coordinate_map() {
  local root="$1" binary="$2" out="$3" spec mode round log rc
  local marker_count rows_count cols_count
  local -a logs=()
  local -a specs=(
    scale-group-tag:even scale-group-tag:odd scale-group-tag:stage
    scale-n-tag:even scale-n-tag:odd scale-n-tag:stage
    zero-group-tag:even zero-group-tag:odd zero-group-tag:stage
    zero-n-tag:even zero-n-tag:odd zero-n-tag:stage
    code-k0-tag:even code-k0-tag:odd code-k0-tag:stage
    code-k1-tag:even code-k1-tag:odd code-k1-tag:stage
    code-k2-tag:even code-k2-tag:odd code-k2-tag:stage
    code-k3-tag:even code-k3-tag:odd code-k3-tag:stage
    code-n0-tag:even code-n0-tag:odd code-n0-tag:stage
    code-n1-tag:even code-n1-tag:odd code-n1-tag:stage
    code-n2-tag:even code-n2-tag:odd code-n2-tag:stage
  )
  python3 -B "$root/ci/adjudicate_q4_a32_coordinate_tags.py" --self-test ||
    return 2
  for spec in "${specs[@]}"; do
    mode="${spec%%:*}"
    round="${spec##*:}"
    log="$out/results/${mode}-${round}.log"
    set +e
    "$binary" --shape=64x1024x5120 --iterations=1 \
      --correctness-repeats=1 --algorithm=nonpersistent \
      --fixture="$mode" --tag-round="$round" --fixture-binding \
      >"$log" 2>&1
    rc=$?
    set -e
    cat "$log"
    marker_count="$(grep -Ec "^SF_FIXTURE mode=${mode} first_golden=0x[0-9a-f]{4} tag_round=${round} probe_count=64 probe_fnv=[0-9a-f]{16} .* roundtrip=1 exact=1 isolation=1$" "$log" || true)"
    rows_count="$(grep -Ec "^SF_TAG_ROWS mode=${mode} tag_round=${round} n=0 count=64 " "$log" || true)"
    cols_count="$(grep -Ec "^SF_TAG_COLS mode=${mode} tag_round=${round} m=0 count=64 " "$log" || true)"
    if [ "$marker_count" -ne 1 ] || [ "$rows_count" -ne 1 ] ||
       [ "$cols_count" -ne 1 ]; then
      printf '[q4-a32-tags] FAIL: %s marker/dump denominator=%s/%s/%s\n' \
        "$spec" "$marker_count" "$rows_count" "$cols_count" >&2
      return 1
    fi
    if [ "$rc" -eq 0 ]; then
      grep -q "^SF_COMPLETE status=COMPLETE .*fixture_mode=${mode} " "$log" || {
        printf '[q4-a32-tags] FAIL: %s returned zero without COMPLETE\n' "$spec" >&2
        return 1
      }
    else
      grep -q '^SF_FATAL .*state=RAW_FP16_MISMATCH step=RAW_FP16_MISMATCH' "$log" || {
        printf '[q4-a32-tags] FAIL: %s rc=%s was not a numeric observation\n' \
          "$spec" "$rc" >&2
        return 1
      }
    fi
    logs+=("$log")
  done
  python3 -B "$root/ci/adjudicate_q4_a32_coordinate_tags.py" \
    --out "$out/results/coordinate-map.tsv" "${logs[@]}" || return $?
  printf '[q4-a32-tags] DIAGNOSTIC_COMPLETE: exact device coordinate map captured; artifacts=%s\n' "$out"
}

run_tail_bisect() {
  local root="$1" binary="$2" out="$3" spec mode round log rc
  local marker_count rows_count cols_count
  local -a logs=()
  # scale-n is deliberately K-invariant: it proved only that the second-tile
  # output path was live.  The other three arms make old/current/next K
  # distinguishable, so a stale register and a forward overwrite cannot hide
  # behind the same value again.
  local -a specs=(
    scale-n-tag:even-next scale-n-tag:odd-next
    scale-group-tag:even-next scale-group-tag:odd-next
    zero-group-tag:even-next zero-group-tag:odd-next
    code-k1-tag:even-next code-k1-tag:odd-next
  )
  python3 -B "$root/ci/adjudicate_q4_a32_coordinate_tags.py" --self-test ||
    return 2
  for spec in "${specs[@]}"; do
    mode="${spec%%:*}"
    round="${spec##*:}"
    log="$out/results/${mode}-${round}.log"
    set +e
    "$binary" --shape=64x1024x5120 --iterations=1 \
      --correctness-repeats=1 --algorithm=nonpersistent \
      --fixture="$mode" --tag-round="$round" --fixture-binding \
      >"$log" 2>&1
    rc=$?
    set -e
    marker_count="$(grep -Ec "^SF_FIXTURE mode=${mode} first_golden=0x[0-9a-f]{4} tag_round=${round} probe_count=64 probe_fnv=[0-9a-f]{16} .* roundtrip=1 exact=1 isolation=1$" "$log" || true)"
    rows_count="$(grep -Ec "^SF_TAG_ROWS mode=${mode} tag_round=${round} n=0 count=64 " "$log" || true)"
    cols_count="$(grep -Ec "^SF_TAG_COLS mode=${mode} tag_round=${round} m=0 count=64 " "$log" || true)"
    if [ "$marker_count" -ne 1 ] || [ "$rows_count" -ne 1 ] ||
       [ "$cols_count" -ne 1 ]; then
      printf '[q4-a32-tail] FAIL: %s marker/dump denominator=%s/%s/%s\n' \
        "$spec" "$marker_count" "$rows_count" "$cols_count" >&2
      tail -80 "$log" >&2
      return 1
    fi
    if [ "$rc" -eq 0 ]; then
      grep -q "^SF_COMPLETE status=COMPLETE .*fixture_mode=${mode} " "$log" || {
        printf '[q4-a32-tail] FAIL: %s returned zero without COMPLETE\n' "$spec" >&2
        tail -80 "$log" >&2
        return 1
      }
    else
      grep -q '^SF_FATAL .*state=RAW_FP16_MISMATCH step=RAW_FP16_MISMATCH' "$log" || {
        printf '[q4-a32-tail] FAIL: %s rc=%s was not a numeric observation\n' \
          "$spec" "$rc" >&2
        tail -80 "$log" >&2
        return 1
      }
    fi
    logs+=("$log")
  done
  python3 -B "$root/ci/adjudicate_q4_a32_coordinate_tags.py" \
    --tail-bisect --out "$out/results/tail-bisect.tsv" "${logs[@]}" \
    >"$out/results/tail-bisect-verdict.log" 2>&1 || {
      cat "$out/results/tail-bisect-verdict.log" >&2
      return 2
    }
  cat "$out/results/tail-bisect-verdict.log"
  printf '[q4-a32-tail] DIAGNOSTIC_COMPLETE: shifted final-delivery slice captured; artifacts=%s\n' "$out"
}

run_closure() {
  local root="$1" generated="$2" out="$3"
  local legacy typed_old raw_first candidate label binary state
  local legacy_state typed_state raw_state candidate_state tail_count

  # A 2x2 factorial in one box invocation separates the two remaining seams:
  # destination representation (raw vs typed) and lifetime order
  # (prepare-first vs consume-first).  This costs four one-row builds but no
  # extra operator round trip, and avoids calling a passing compound patch a
  # unique root cause.
  legacy="$(build_exact_variant "$root" "$generated" "$out" legacy-both \
    'PPU_Q4_A32_LEGACY_RAW_SCATTER=1 PPU_Q4_A32_LEGACY_PREPARE_ORDER=1')" || return $?
  typed_old="$(build_exact_variant "$root" "$generated" "$out" typed-old-order \
    'PPU_Q4_A32_LEGACY_PREPARE_ORDER=1')" || return $?
  raw_first="$(build_exact_variant "$root" "$generated" "$out" raw-consume-first \
    'PPU_Q4_A32_LEGACY_RAW_SCATTER=1')" || return $?
  candidate="$(build_exact_variant "$root" "$generated" "$out" candidate '')" || return $?

  printf '[q4-a32-closure] variants=4 legacy=%s typed_old=%s raw_first=%s candidate=%s\n' \
    "$legacy" "$typed_old" "$raw_first" "$candidate"
  run_tail_bisect "$root" "$legacy" "$out" || return $?
  tail_count="$(grep -Ec '^Q4_A32_TAIL_BISECT verdict=(NEXT_TILE_FRESH|PREVIOUS_TILE_STALE|NEXT_DELIVERY_CLOBBER|MIXED) ' \
    "$out/results/tail-bisect-verdict.log" || true)"
  if [ "$tail_count" -ne 1 ]; then
    printf '[q4-a32-closure] FAIL: absolute-K tail verdict denominator=%s, want 1\n' \
      "$tail_count" >&2
    return 1
  fi

  run_exact_variant "$legacy" "$out" legacy-both || return $?
  run_exact_variant "$typed_old" "$out" typed-old-order || return $?
  run_exact_variant "$raw_first" "$out" raw-consume-first || return $?
  run_exact_variant "$candidate" "$out" candidate || return $?
  legacy_state="$(cat "$out/results/exact-legacy-both.state")"
  typed_state="$(cat "$out/results/exact-typed-old-order.state")"
  raw_state="$(cat "$out/results/exact-raw-consume-first.state")"
  candidate_state="$(cat "$out/results/exact-candidate.state")"

  if [ "$legacy_state" != NUMERIC_FAIL ] ||
     ! legacy_signature_present "$out/results/exact-legacy-both.log"; then
    printf '[q4-a32-closure] FAIL: legacy negative did not reproduce raw_bad=61184, first -3 -> -18; batch is not comparable\n' >&2
    return 1
  fi
  if [ "$candidate_state" != PASS ]; then
    printf '[q4-a32-closure] FAIL: candidate exact row is %s; a second box iteration is required\n' \
      "$candidate_state" >&2
    return 1
  fi
  printf 'Q4_A32_CLOSURE verdict=PASS legacy=%s typed_old_order=%s raw_consume_first=%s candidate=%s tail=%s\n' \
    "$legacy_state" "$typed_state" "$raw_state" "$candidate_state" \
    "$(sed -n 's/^Q4_A32_TAIL_BISECT verdict=\([^ ]*\) .*/\1/p' "$out/results/tail-bisect-verdict.log")"
  printf '[q4-a32-closure] PASS: historical signature reproduced and candidate is raw-bit exact; artifacts=%s\n' "$out"
}

main() {
  local root sha short stamp out generated build binary log symbol rc fixtures
  local fixture arm state code_state scale_state zero_state metadata_state
  local transport_state exact_state locus
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="${OUT:-/workspace/quactlize-q4-a32-exact-${short}-${stamp}}"
  case "$out" in
    /workspace/*) ;;
    *) printf '[q4-a32-exact] FAIL: OUT must be below /workspace: %s\n' "$out" >&2; return 2 ;;
  esac
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    printf '[q4-a32-exact] FAIL: ambient PPU_DEFS/PPU_EXTRA_DEFS would change the exact binary\n' >&2
    return 2
  fi
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

  if [ "${Q4_A32_CLOSURE:-0}" = 1 ]; then
    printf '%s\n' "$sha" >"$out/results/git.sha"
    run_closure "$root" "$generated" "$out"
    return $?
  fi

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

  if [ "${Q4_A32_COORDINATE_MAP:-0}" = 1 ] &&
     [ "${Q4_A32_TAIL_BISECT:-0}" = 1 ]; then
    printf '[q4-a32-exact] FAIL: coordinate map and tail bisection are mutually exclusive\n' >&2
    return 2
  fi
  if [ "${Q4_A32_TAIL_BISECT:-0}" = 1 ]; then
    run_tail_bisect "$root" "$binary" "$out"
    return $?
  fi
  if [ "${Q4_A32_COORDINATE_MAP:-0}" = 1 ]; then
    run_coordinate_map "$root" "$binary" "$out"
    return $?
  fi

  fixtures="${FIXTURES:-exact}"
  fixtures="${fixtures//,/ }"
  code_state=NOT_RUN
  scale_state=NOT_RUN
  zero_state=NOT_RUN
  metadata_state=NOT_RUN
  transport_state=NOT_RUN
  exact_state=NOT_RUN
  for fixture in $fixtures; do
    case "$fixture" in
      exact|code-only|scale-only|zero-only|metadata-only|transport-only) ;;
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
      --algorithm=nonpersistent --fixture="$fixture" --fixture-binding \
      >"$log" 2>&1
    rc=$?
    set -e
    cat "$log"
    state="$(classify_arm "$fixture" "$rc" "$log")"
    printf 'Q4_A32_ARM fixture=%s state=%s rc=%s\n' "$fixture" "$state" "$rc"
    case "$arm" in
      code_only) code_state="$state" ;;
      scale_only) scale_state="$state" ;;
      zero_only) zero_state="$state" ;;
      metadata_only) metadata_state="$state" ;;
      transport_only) transport_state="$state" ;;
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
    if [ "$transport_state" != NOT_RUN ] && [ "$scale_state" != NOT_RUN ] &&
       [ "$zero_state" != NOT_RUN ]; then
      printf 'Q4_A32_COMPONENTS transport_only=%s code_only=%s scale_only=%s zero_only=%s metadata_only=%s exact=%s\n' \
        "$transport_state" "$code_state" "$scale_state" "$zero_state" \
        "$metadata_state" "$exact_state"
    fi
    printf '[q4-a32-exact] DIAGNOSTIC_COMPLETE: component/three-arm locus classified; artifacts=%s\n' "$out"
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
