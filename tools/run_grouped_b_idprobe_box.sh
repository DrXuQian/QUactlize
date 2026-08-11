#!/usr/bin/env bash
# Focused morning device gate for the two exact expert-identity probes in
# test_ppu_m8n16_collective.  This intentionally does not reuse #112's
# historical aggregate runner: the latter carries G0-G4 prerequisites and log
# assertions unrelated to B addressing, and several of those assertions
# predate G5.  One target, one build, one explicit --idprobe-only invocation.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=test_ppu_m8n16_collective
ARTIFACT_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/quactlize-grouped-b-idprobe.XXXXXX")"
BUILD_ROOT="$ARTIFACT_ROOT/ppu0010"
BUILD_LOG="$ARTIFACT_ROOT/ppu0010.build.log"
RUN_LOG="$ARTIFACT_ROOT/ppu0010.idprobe.log"

fail() {
  printf '[grouped-b-idprobe] FAIL: %s\n' "$*" >&2
  printf '[grouped-b-idprobe] artifacts preserved at %s\n' "$ARTIFACT_ROOT" >&2
  exit 1
}

find_one() {
  local description="$1"
  shift
  local -a hits=()
  mapfile -t hits < <(find "$@" -print)
  if [ "${#hits[@]}" -ne 1 ]; then
    printf '[grouped-b-idprobe] %s candidates (%d):\n' \
      "$description" "${#hits[@]}" >&2
    printf '  %s\n' "${hits[@]:-<none>}" >&2
    fail "expected exactly one $description"
  fi
  printf '%s\n' "${hits[0]}"
}

printf '[grouped-b-idprobe] root-sha=%s\n' "$(git -C "$ROOT" rev-parse HEAD)"
printf '[grouped-b-idprobe] actlize-sha=%s\n' \
  "$(git -C "$ROOT/third_party/actlize" rev-parse HEAD)"
printf '[grouped-b-idprobe] artifacts=%s\n' "$ARTIFACT_ROOT"

if ! env PPU_BUILD_DIR="$BUILD_ROOT" PPU_ARCHS=ppu0010 TARGET="$TARGET" \
    "$ROOT/build.sh" 2>&1 | tee "$BUILD_LOG"; then
  fail 'ppu001 IDPROBE target build failed'
fi

BUILD_MAKE="$(find_one 'IDPROBE build.make' "$BUILD_ROOT" -type f \
  -path "*${TARGET}.dir/build.make")"
mapfile -t ARCHS < <(grep -oE -- '-arch=ppu_[0-9]+' "$BUILD_MAKE" | sort -u)
printf '[grouped-b-idprobe][arch] build.make=%s\n' "$BUILD_MAKE"
printf '[grouped-b-idprobe][arch] unique hgcc arch flags:'
printf ' %s' "${ARCHS[@]:-<none>}"
printf '\n'
if [ "${#ARCHS[@]}" -ne 1 ] || [ "${ARCHS[0]}" != '-arch=ppu_10' ]; then
  fail 'IDPROBE build must contain only -arch=ppu_10'
fi
if grep -q -- '-arch=ppu_15' "$BUILD_MAKE"; then
  fail 'IDPROBE build unexpectedly contains -arch=ppu_15'
fi

BIN="$(find_one 'IDPROBE binary' "$BUILD_ROOT" -type f \
  -name "$TARGET" -perm -u+x)"
if ! "$BIN" --idprobe-only 2>&1 | tee "$RUN_LOG"; then
  fail 'zero/B IDPROBE invocation returned nonzero'
fi

grep -q '^\[G5:IDPROBE\] PASS: 0/8 slots read an expert other than their own$' \
  "$RUN_LOG" || fail 'zero-plane active=8 arm did not pass exactly'
grep -q '^\[G5:IDPROBE\] PASS: 0/256 slots read an expert other than their own$' \
  "$RUN_LOG" || fail 'zero-plane active=256 arm did not pass exactly'
grep -q '^\[G5:B-IDPROBE\] PASS: slot-mismatches=0/8 output-bitdiff=0/256$' \
  "$RUN_LOG" || fail 'B-plane active=8 arm was not raw-fp16 bit-exact'
grep -q '^\[G5:B-IDPROBE\] PASS: slot-mismatches=0/256 output-bitdiff=0/8192$' \
  "$RUN_LOG" || fail 'B-plane active=256 arm was not raw-fp16 bit-exact'
grep -q '^== \[112:IDPROBE-ONLY\] PASS: errors=0 (zero active=8/256; B active=8/256) ==$' \
  "$RUN_LOG" || fail 'aggregate IDPROBE PASS marker is absent'

printf '\n[grouped-b-idprobe] PASS: zero and B expert identity are exact at active=8 and active=256\n'
printf '[grouped-b-idprobe] artifacts preserved at %s\n' "$ARTIFACT_ROOT"
