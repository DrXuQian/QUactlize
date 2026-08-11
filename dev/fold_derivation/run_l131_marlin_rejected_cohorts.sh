#!/usr/bin/env bash
# Constructive compile proof for the four cohorts excluded by the dense
# Marlin second-stage filter.  The DP positive and Marlin negative consume the
# exact same committed tactic rows.  The only semantic delta is
# DENSE_MARLIN_SWEEP=1: it bypasses the CMake cohort admission decision by
# selecting the named Marlin wrapper, while every tactic/format guard remains.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/dev/fold_derivation/l131_marlin_rejected_cohorts.cu"
SYNTAX="$ROOT/dev/fold_derivation/syntax_check.sh"

python3 "$ROOT/ci/check_dense_marlin_rejection_census.py"

dp_log="$(mktemp)"
trap 'rm -f "$dp_log" "${red_log:-}"' EXIT
set +e
bash "$SYNTAX" "$SRC" >"$dp_log" 2>&1
dp_rc=$?
set -e
if [[ $dp_rc -eq 2 || $dp_rc -eq 3 ]]; then
  echo "[L131] SKIP: syntax compiler unavailable for the ordinary-DP witness" >&2
  cat "$dp_log" >&2
  exit 3
fi
if [[ $dp_rc -ne 0 ]]; then
  echo "[L131] FAIL: ordinary DP control did not compile cleanly" >&2
  sed -n '1,20p' "$dp_log" >&2
  exit 1
fi
grep -q 'clean (0 known-noise lines, 0 new)' "$dp_log" || {
  echo "[L131] FAIL: DP control lacks a positive clean-completion witness" >&2
  cat "$dp_log" >&2
  exit 1
}
echo '[L131] ordinary-DP PASS: all four rejected representative rows compiled; same tactic/format guards'

declare -A row=(
  [1]='8,16,64,8,16,2,0'
  [8]='8,128,64,8,16,2,0'
  [16]='8,256,64,8,16,2,0'
  [32]='32,256,64,16,16,2,0'
)

for warps in 1 8 16 32; do
  red_log="$(mktemp)"
  set +e
  EXTRA_DEFS="-DDENSE_MARLIN_SWEEP=1 -DL131_ONLY_CTA_WARPS=$warps" \
    bash "$SYNTAX" "$SRC" >"$red_log" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 1 ]]; then
    echo "[L131] FAIL: cta_warps=$warps expected syntax rc=1, got $rc" >&2
    cat "$red_log" >&2
    exit 1
  fi

  # One named scheduler type; two fixup overload instantiations; four compiled
  # group-size arms for both the named kernel and generated wrapper.  These
  # counts are the raw normalized diagnostics from syntax_check.sh, not copied
  # expectations in a prose document.
  grep -Fxq '1 128-thread CTA cohorts"' "$red_log" || {
    echo "[L131] FAIL: missing scheduler-cohort diagnostic for w$warps" >&2; cat "$red_log" >&2; exit 1; }
  grep -Fxq '2 ppu_tile_scheduler_marlin.hpp(): error: static assertion failed with "Marlin cooperative derived an unsupported CTA cohort"' "$red_log" || {
    echo "[L131] FAIL: missing exact fixup diagnostic for w$warps" >&2; cat "$red_log" >&2; exit 1; }
  grep -Fxq '4 128-thread CTAs"' "$red_log" || {
    echo "[L131] FAIL: missing named-kernel diagnostic for w$warps" >&2; cat "$red_log" >&2; exit 1; }
  grep -Fxq '4 128-thread rows"' "$red_log" || {
    echo "[L131] FAIL: missing generated-wrapper diagnostic for w$warps" >&2; cat "$red_log" >&2; exit 1; }
  [[ $(grep -c '^' "$red_log") -eq 5 ]] || {
    echo "[L131] FAIL: unexpected extra normalized diagnostics for w$warps" >&2; cat "$red_log" >&2; exit 1; }

  echo "[L131] Marlin EXPECTED_RED cta_warps=$warps threads=$((warps * 32)) row=${row[$warps]} diagnostics=scheduler:1,fixup:2,kernel:4,wrapper:4"
  rm -f "$red_log"
  red_log=""
done

echo '[L131] PASS: CMake cohort guard alone was bypassed; every rejected cohort is stopped by the current Marlin implementation, not by a tactic/format guard'
