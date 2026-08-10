#!/usr/bin/env bash
# #111's complete device gate.  One invocation creates independent ppu001 and
# ppu0015 build trees, preserves every log, and proves both directions:
#   ppu001  -> exact G1 + G2 green, including the historical-index expected red;
#   ppu0015 -> the raw atom fails specifically in m8n16k16 ISel.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=test_ppu_m8n16_gates
ARTIFACT_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/quactlize-m8n16-111.XXXXXX")"
POS_BUILD="$ARTIFACT_ROOT/ppu0010"
NEG_BUILD="$ARTIFACT_ROOT/ppu0015"
POS_LOG="$ARTIFACT_ROOT/ppu0010.build.log"
RUN_LOG="$ARTIFACT_ROOT/ppu0010.run.log"
SYMBOL_LOG="$ARTIFACT_ROOT/ppu0010.symbols.log"
NEG_LOG="$ARTIFACT_ROOT/ppu0015.expected-failure.log"

fail() {
  printf '[111] FAIL: %s\n' "$*" >&2
  printf '[111] artifacts preserved at %s\n' "$ARTIFACT_ROOT" >&2
  exit 1
}

find_one() {
  local description="$1"
  shift
  local -a hits=()
  mapfile -t hits < <(find "$@" -print)
  if [ "${#hits[@]}" -ne 1 ]; then
    printf '[111] %s candidates (%d):\n' "$description" "${#hits[@]}" >&2
    printf '  %s\n' "${hits[@]:-<none>}" >&2
    fail "expected exactly one $description"
  fi
  printf '%s\n' "${hits[0]}"
}

audit_arch() {
  local build_root="$1" want="$2" reject="$3" label="$4"
  local build_make
  build_make="$(find_one "$label build.make" "$build_root" -type f \
      -path "*${TARGET}.dir/build.make")"
  local -a archs=()
  mapfile -t archs < <(grep -oE -- '-arch=ppu_[0-9]+' "$build_make" | sort -u)
  printf '[G0][%s] build.make=%s\n' "$label" "$build_make"
  printf '[G0][%s] unique hgcc arch flags:' "$label"
  printf ' %s' "${archs[@]:-<none>}"
  printf '\n'
  if [ "${#archs[@]}" -ne 1 ] || [ "${archs[0]}" != "$want" ]; then
    fail "$label must contain only $want"
  fi
  if grep -q -- "$reject" "$build_make"; then
    fail "$label unexpectedly contains $reject"
  fi
}

printf '[111] root-sha=%s\n' "$(git -C "$ROOT" rev-parse HEAD)"
printf '[111] actlize-sha=%s\n' "$(git -C "$ROOT/third_party/actlize" rev-parse HEAD)"
printf '[111] artifacts=%s\n' "$ARTIFACT_ROOT"

if ! python3 "$ROOT/ci/check_m8n16_g2_contract.py"; then
  fail 'G2 source contract failed before the box build'
fi

printf '\n== G0 positive arch audit + G1/G2 run: ppu001 ==\n'
if ! env PPU_BUILD_DIR="$POS_BUILD" PPU_ARCHS=ppu0010 TARGET="$TARGET" \
    "$ROOT/build.sh" 2>&1 | tee "$POS_LOG"; then
  fail "ppu001 build failed; G1/G2 were not run"
fi
audit_arch "$POS_BUILD" -arch=ppu_10 -arch=ppu_15 ppu001

POS_BIN="$(find_one 'ppu001 gate binary' "$POS_BUILD" -type f \
    -name "$TARGET" -perm -u+x)"
ATOM_OBJ="$(find_one 'ppu001 G1 atom object' "$POS_BUILD" -type f \
    -name 'test_ppu_m8n16_atom_*.o')"
# The requested m8n16k16-bearing symbol is a provenance check, not an opcode
# check: its spelling comes from this test kernel.  Instruction identity is
# carried by G1 executing the raw atom and, independently, by ppu0015 naming
# the unsupported m8n16k16 intrinsic in the required ISel failure below.
if ! nm -C "$ATOM_OBJ" >"$SYMBOL_LOG"; then
  fail "nm could not inspect $ATOM_OBJ"
fi
if ! grep -q 'm8n16k16' "$SYMBOL_LOG"; then
  fail "G0 symbol audit found no m8n16k16 in $ATOM_OBJ"
fi
grep 'm8n16k16' "$SYMBOL_LOG" | sed 's/^/[G0][ppu001] provenance symbol: /'
printf '[G0][ppu001] provenance symbol contains m8n16k16: PASS\n'

if ! "$POS_BIN" 2>&1 | tee "$RUN_LOG"; then
  fail "ppu001 numerical gate returned nonzero"
fi
grep -q '^\[G1\] PASS: total_bad=0$' "$RUN_LOG" \
  || fail 'G1 did not report all atom outputs exact'
grep -q '^\[G2-control-path\] same-payload=production-x4 cube=16x64 coords=(0,0) green=get_i/get_j red=historical-nvidia-x2-provider-map$' "$RUN_LOG" \
  || fail 'G2 did not report one production payload with correct and historical index maps'
grep -q '^\[G2-green-detail\] x4_values=512 x4_bad=0 projected_changed=0/128 lower_poison_changed=128/128$' "$RUN_LOG" \
  || fail 'G2 production 16-row x4 delivery and poison checks did not pass exactly'
grep -q '^\[G2-green\] mismatches=0 PASS$' "$RUN_LOG" \
  || fail 'G2 physical-16-row x4 projection was not exactly green'
grep -q '^\[G2-negative-detail\] same_payload=x4-swzl geometry=16x64 bad_map_values=128 bad_map_bad=0 coincident_words=2/64 red_expected=124/128$' "$RUN_LOG" \
  || fail 'G2 historical NVIDIA provider map did not name exact production-x4 tags and its two reviewed coincidences'
grep -q '^\[G2-negative\] mismatches=124 EXPECTED_RED/PASS$' "$RUN_LOG" \
  || fail 'G2 historical NVIDIA-on-PPU indexing did not produce the exact required red mismatch'
grep -q '^== \[111\] PASS: G1=0 G2=0 ==$' "$RUN_LOG" \
  || fail 'aggregate G1/G2 PASS marker is absent'
printf '[G0/G1/G2][ppu001] PASS\n'

printf '\n== G0 negative arch audit: ppu0015 must fail m8n16k16 ISel ==\n'
set +e
env PPU_BUILD_DIR="$NEG_BUILD" PPU_ARCHS=ppu0015 TARGET="$TARGET" \
  "$ROOT/build.sh" 2>&1 | tee "$NEG_LOG"
neg_rc=${PIPESTATUS[0]}
set -e
if [ "$neg_rc" -eq 0 ]; then
  fail 'ppu0015 unexpectedly built the ppu001-only m8n16k16 atom'
fi
audit_arch "$NEG_BUILD" -arch=ppu_15 -arch=ppu_10 ppu0015
if ! grep -q 'Cannot select' "$NEG_LOG" || ! grep -q 'm8n16k16' "$NEG_LOG"; then
  fail 'ppu0015 failed, but not with the required m8n16k16 intrinsic-selection diagnostic'
fi
printf '[G0][ppu0015] EXPECTED_FAIL rc=%d: Cannot select ... m8n16k16\n' "$neg_rc"

printf '\n[111] PASS: positive arch + G1 + G2 green/red + negative arch all proved\n'
printf '[111] artifacts preserved at %s\n' "$ARTIFACT_ROOT"
