#!/usr/bin/env bash
# Constructive compile proof for the four cohorts recovered by the dense
# Marlin capability.  DP and Marlin positives consume the exact same committed
# tactic rows.  A second compile gives the production fixup a different but
# still warp-aligned/in-range explicit cohort; only the exact accumulator
# binding may reject it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/dev/fold_derivation/l131_marlin_rejected_cohorts.cu"
SYNTAX="$ROOT/dev/fold_derivation/syntax_check.sh"

python3 "$ROOT/ci/check_dense_marlin_rejection_census.py"

dp_log="$(mktemp)"
trap 'rm -f "$dp_log" "${green_log:-}" "${red_log:-}"' EXIT
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
echo '[L131] ordinary-DP PASS: all four recovered representative rows compiled; same tactic/format guards'

declare -A row=(
  [1]='8,16,64,8,16,2,0'
  [8]='8,128,64,8,16,2,0'
  [16]='8,256,64,8,16,2,0'
  [32]='32,256,64,16,16,2,0'
)
for warps in 1 8 16 32; do
  green_log="$(mktemp)"
  set +e
  EXTRA_DEFS="-DDENSE_MARLIN_SWEEP=1 -DL131_ONLY_CTA_WARPS=$warps" \
    bash "$SYNTAX" "$SRC" >"$green_log" 2>&1
  green_rc=$?
  set -e
  if [[ $green_rc -eq 2 || $green_rc -eq 3 ]]; then
    echo "[L131] SKIP: syntax compiler unavailable for Marlin cta_warps=$warps" >&2
    cat "$green_log" >&2
    exit 3
  fi
  if [[ $green_rc -ne 0 ]] ||
     ! grep -q 'clean (0 known-noise lines, 0 new)' "$green_log"; then
    echo "[L131] FAIL: Marlin cta_warps=$warps did not compile cleanly" >&2
    cat "$green_log" >&2
    exit 1
  fi
  echo "[L131] Marlin PASS cta_warps=$warps threads=$((warps * 32)) row=${row[$warps]}"
  rm -f "$green_log"
done

# One mutation is sufficient because every named wrapper consumes the same
# scheduler template.  Use the recovered 256-thread row and plant the old
# 128-thread default: both counts satisfy the broad capability, so only the
# exact accumulator-derived binding can make this compile red.
red_log="$(mktemp)"
set +e
EXTRA_DEFS="-DDENSE_MARLIN_SWEEP=1 -DL131_ONLY_CTA_WARPS=8 -DL131_WRONG_EXPLICIT_COHORT=128" \
  bash "$SYNTAX" "$SRC" >"$red_log" 2>&1
red_rc=$?
set -e
if [[ $red_rc -ne 1 ]] ||
   ! grep -q 'exact accumulator-derived CTA size"' "$red_log"; then
  echo "[L131] FAIL: wrong explicit cohort 128 for 256-thread CTA did not fail at exact binding (rc=$red_rc)" >&2
  cat "$red_log" >&2
  exit 1
fi
echo '[L131] wrong-explicit-cohort EXPECTED_RED actual=256 planted=128'
rm -f "$red_log"
red_log=""

echo '[L131] PASS: all four recovered real Marlin wrappers compile; every structurally-capable wrong explicit cohort is stopped by the exact accumulator binding'
