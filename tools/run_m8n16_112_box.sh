#!/usr/bin/env bash
# #112's device gate.  The collective result is accepted only after #111 has
# proved the ppu001 raw atom and G2's same-payload historical-index replay, so
# this script runs that prerequisite first instead of trusting a remembered
# result from another checkout.
#
# G3 checks the real ScaleZero mainloop's raw FP32 accumulator.  G4 checks the
# production grouped ptr-array epilogue at M={1,2,3,7,8}, including the exact
# same-input m16 control.  G5 is deliberately absent until #108 supplies the
# E=256/active=8 non-contiguous ragged harness.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=test_ppu_m8n16_collective
ARTIFACT_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/quactlize-m8n16-112.XXXXXX")"
BUILD_ROOT="$ARTIFACT_ROOT/ppu0010"
PREREQ_LOG="$ARTIFACT_ROOT/111-prerequisite.log"
BUILD_LOG="$ARTIFACT_ROOT/ppu0010.build.log"
RUN_LOG="$ARTIFACT_ROOT/ppu0010.run.log"

fail() {
  printf '[112] FAIL: %s\n' "$*" >&2
  printf '[112] artifacts preserved at %s\n' "$ARTIFACT_ROOT" >&2
  exit 1
}

find_one() {
  local description="$1"
  shift
  local -a hits=()
  mapfile -t hits < <(find "$@" -print)
  if [ "${#hits[@]}" -ne 1 ]; then
    printf '[112] %s candidates (%d):\n' "$description" "${#hits[@]}" >&2
    printf '  %s\n' "${hits[@]:-<none>}" >&2
    fail "expected exactly one $description"
  fi
  printf '%s\n' "${hits[0]}"
}

printf '[112] root-sha=%s\n' "$(git -C "$ROOT" rev-parse HEAD)"
printf '[112] actlize-sha=%s\n' "$(git -C "$ROOT/third_party/actlize" rev-parse HEAD)"
printf '[112] artifacts=%s\n' "$ARTIFACT_ROOT"

printf '\n== prerequisite: #111 G0/G1/G2 must pass on this checkout ==\n'
if ! "$ROOT/tools/run_m8n16_111_box.sh" 2>&1 | tee "$PREREQ_LOG"; then
  fail '#111 prerequisite failed; G3/G4 are not admissible'
fi
grep -q '^\[111\] PASS: positive arch + G1 + G2 green/red + negative arch all proved$' "$PREREQ_LOG" \
  || fail '#111 returned zero without its aggregate PASS marker'

printf '\n== G3/G4 collective build and run: ppu001 ==\n'
if ! env PPU_BUILD_DIR="$BUILD_ROOT" PPU_ARCHS=ppu0010 TARGET="$TARGET" \
    "$ROOT/build.sh" 2>&1 | tee "$BUILD_LOG"; then
  fail 'ppu001 collective build failed'
fi

BUILD_MAKE="$(find_one 'collective build.make' "$BUILD_ROOT" -type f \
    -path "*${TARGET}.dir/build.make")"
mapfile -t ARCHS < <(grep -oE -- '-arch=ppu_[0-9]+' "$BUILD_MAKE" | sort -u)
printf '[G3/G4][arch] build.make=%s\n' "$BUILD_MAKE"
printf '[G3/G4][arch] unique hgcc arch flags:'
printf ' %s' "${ARCHS[@]:-<none>}"
printf '\n'
if [ "${#ARCHS[@]}" -ne 1 ] || [ "${ARCHS[0]}" != '-arch=ppu_10' ]; then
  fail 'collective build must contain only -arch=ppu_10'
fi
if grep -q -- '-arch=ppu_15' "$BUILD_MAKE"; then
  fail 'collective build unexpectedly contains -arch=ppu_15'
fi

BIN="$(find_one 'collective gate binary' "$BUILD_ROOT" -type f \
    -name "$TARGET" -perm -u+x)"
if ! "$BIN" 2>&1 | tee "$RUN_LOG"; then
  fail 'G3/G4 numerical gate returned nonzero'
fi

grep -q '^\[G5\] BLOCKED on #108 real E=256/active=8 ragged harness; L=1 is not substituted$' "$RUN_LOG" \
  || fail 'G5 blocker marker is absent or an L=1 substitute was presented'
grep -q '^\[offline\] m8/m16 B artifacts byte-identical: 4096 physical bytes (4096 logical); roundtrip=0/8192$' "$RUN_LOG" \
  || fail 'offline m8/m16 physical artifact identity or round-trip gate did not pass exactly'
grep -Eq '^  G3 raw FP32 accum +bad=0/256 max_abs=[^ ]+ MATCH$' "$RUN_LOG" \
  || fail 'G3 raw FP32 accumulator did not match all 256 values'

for m in 1 2 3 7 8; do
  count=$((m * 32))
  grep -Eq "^  G4 m8 +M=${m} golden bad=0/${count} max_abs=[^ ]+ MATCH$" "$RUN_LOG" \
    || fail "G4 m8 M=${m} did not match the independent golden"
  grep -Eq "^  G4 m16 +M=${m} golden bad=0/${count} max_abs=[^ ]+ MATCH$" "$RUN_LOG" \
    || fail "G4 m16 control M=${m} did not match the independent golden"
  grep -q "^  G4 m8-vs-m16 M=${m} bitdiff=0/${count} MATCH$" "$RUN_LOG" \
    || fail "G4 m8/m16 same-input control M=${m} was not bit-exact"
done

grep -q '^== \[112\] PASS: errors=0 (G3/G4; G5 blocked on #108) ==$' "$RUN_LOG" \
  || fail 'aggregate G3/G4 PASS marker is absent'

printf '\n[112] PASS: #111 prerequisite + G3 raw mainloop + G4 grouped epilogue all proved\n'
printf '[112] G5 remains BLOCKED on #108; no L=1 result was counted\n'
printf '[112] artifacts preserved at %s\n' "$ARTIFACT_ROOT"
