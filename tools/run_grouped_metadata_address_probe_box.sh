#!/usr/bin/env bash
# Isolated ppu001 G5 zero-plane address census.  A mismatch is the result, not
# a harness failure; the script fails only when one of the four observation
# layers was not exercised or the trace is internally incomplete.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=test_ppu_grouped_metadata_address
ARTIFACT_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/quactlize-g5-address.XXXXXX")"
BUILD_ROOT="$ARTIFACT_ROOT/ppu0010"
BUILD_LOG="$ARTIFACT_ROOT/ppu0010.build.log"
RUN_LOG="$ARTIFACT_ROOT/ppu0010.run.log"

fail() {
  printf '[G5:ADDR] FAIL: %s\n' "$*" >&2
  printf '[G5:ADDR] artifacts preserved at %s\n' "$ARTIFACT_ROOT" >&2
  exit 1
}

find_one() {
  local description="$1"
  shift
  local -a hits=()
  mapfile -t hits < <(find "$@" -print)
  if [ "${#hits[@]}" -ne 1 ]; then
    printf '[G5:ADDR] %s candidates (%d):\n' "$description" "${#hits[@]}" >&2
    printf '  %s\n' "${hits[@]:-<none>}" >&2
    fail "expected exactly one $description"
  fi
  printf '%s\n' "${hits[0]}"
}

printf '[G5:ADDR] root-sha=%s\n' "$(git -C "$ROOT" rev-parse HEAD)"
printf '[G5:ADDR] actlize-sha=%s\n' "$(git -C "$ROOT/third_party/actlize" rev-parse HEAD)"
printf '[G5:ADDR] artifacts=%s\n' "$ARTIFACT_ROOT"

# Do not run ci/local_gates.py here.  The box has no viable local-gate CUDA
# toolchain combination; those source/oracle facts are established in the dev
# container against the SHA printed above.
if ! env PPU_BUILD_DIR="$BUILD_ROOT" PPU_ARCHS=ppu0010 TARGET="$TARGET" \
    "$ROOT/build.sh" 2>&1 | tee "$BUILD_LOG"; then
  fail 'ppu001 metadata-address target failed to build'
fi

BUILD_MAKE="$(find_one 'metadata-address build.make' "$BUILD_ROOT" -type f \
    -path "*${TARGET}.dir/build.make")"
mapfile -t ARCHS < <(grep -oE -- '-arch=ppu_[0-9]+' "$BUILD_MAKE" | sort -u)
printf '[G5:ADDR][arch] build.make=%s\n' "$BUILD_MAKE"
printf '[G5:ADDR][arch] unique hgcc arch flags:'
printf ' %s' "${ARCHS[@]:-<none>}"
printf '\n'
if [ "${#ARCHS[@]}" -ne 1 ] || [ "${ARCHS[0]}" != '-arch=ppu_10' ]; then
  fail 'probe build must contain only -arch=ppu_10'
fi
if ! grep -q -- '-DPPU_METADATA_ADDR_PROBE=1' "$BUILD_MAKE"; then
  fail 'probe macro did not reach the hgcc device compile'
fi

BIN="$(find_one 'metadata-address binary' "$BUILD_ROOT" -type f \
    -name "$TARGET" -perm -u+x)"
if ! "$BIN" 2>&1 | tee "$RUN_LOG"; then
  fail 'address census was incomplete or internally inconsistent'
fi

grep -Eq '^\[G5:ADDR\] trace magic=0x[0-9a-f]+ version=1 shape=768/768 copy=768/768 overflow=0 config_errors=0 cta_threads=32 slots=8 tiles=4 values=8$' "$RUN_LOG" \
  || fail 'trace geometry/count handshake did not close exactly'
for expert in 127 128 129; do
  grep -Eq "^\[G5:ADDR\] expert=${expert} scheduler_ctas=1 .* seam=[A-Z0-9_]+ .*partition_experts=\\{.*\\} cp_async_experts=\\{.*\\} gz_experts=\\{.*\\} gz_addr_delta_bytes=\\{.*\\}$" "$RUN_LOG" \
    || fail "expert ${expert} did not report scheduler plus all address/copy layers"
  grep -Eq "^\[G5:ADDR\]\[shape-detail\] scheduler_e=${expert} group=[0-9]+ n=[0-9]+ explicit=0x[0-9a-f]+/0x[0-9a-f]+ gz_base=0x[0-9a-f]+ gz=0x[0-9a-f]+/0x[0-9a-f]+ gz_tag_e=[0-9]+$" "$RUN_LOG" \
    || fail "expert ${expert} did not print an actual explicit-vs-gZ address/value witness"
done
grep -q '^\[G5:ADDR\] scope=B_NOT_COVERED q==8 nulls B; this probe proves only zero-plane addressing$' "$RUN_LOG" \
  || fail 'zero-plane-only scope marker is absent'
grep -q '^\[G5:ADDR\] COMPLETE: all four observation layers were exercised for experts 127/128/129$' "$RUN_LOG" \
  || fail 'aggregate COMPLETE marker is absent'

printf '\n[G5:ADDR] PASS: diagnostic is complete (the reported seam may itself be red)\n'
printf '[G5:ADDR] artifacts preserved at %s\n' "$ARTIFACT_ROOT"
