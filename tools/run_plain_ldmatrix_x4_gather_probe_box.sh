#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHA="$(git -C "$ROOT" rev-parse HEAD)"
OUT="${OUT:-/workspace/quactlize-plain-x4-${SHA:0:7}-$(date -u +%Y%m%dT%H%M%SZ)}"
BUILD="$OUT/build"
RESULTS="$OUT/results"
mkdir -p "$RESULTS"

fail() {
  printf '[plain-x4] FAIL: %s\n' "$*" >&2
  printf 'artifacts: %s\n' "$OUT" >&2
  exit 1
}

printf '[plain-x4] sha=%s artifacts=%s\n' "$SHA" "$OUT"
env PPU_BUILD_DIR="$BUILD" PPU_ARCHS=ppu0010 \
  TARGET=plain_ldmatrix_x4_gather_probe \
  "$ROOT/build.sh" >"$RESULTS/build.log" 2>&1 \
  || { tail -120 "$RESULTS/build.log" >&2; fail 'build rc!=0'; }

mapfile -t BINS < <(find "$BUILD" -type f \
  -name plain_ldmatrix_x4_gather_probe -perm -u+x -print)
[ "${#BINS[@]}" -eq 1 ] \
  || fail "expected one probe binary, found ${#BINS[@]}"
"${BINS[0]}" >"$RESULTS/device.log" 2>&1 \
  || { cat "$RESULTS/device.log" >&2; fail 'device probe rc!=0'; }

python3 - "$RESULTS/device.log" <<'PY' | tee "$RESULTS/verdict.log"
from __future__ import annotations

import hashlib
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
pattern = re.compile(
    r"^PLAIN_X4_CELL arm=(lane-local|collapsed) lane=(\d+) "
    r"r0=([0-9a-f]{4}),([0-9a-f]{4}) "
    r"r1=([0-9a-f]{4}),([0-9a-f]{4}) "
    r"r2=([0-9a-f]{4}),([0-9a-f]{4}) "
    r"r3=([0-9a-f]{4}),([0-9a-f]{4})$",
    re.MULTILINE,
)
arms: dict[str, dict[int, list[int]]] = {"lane-local": {}, "collapsed": {}}
for match in pattern.finditer(text):
    arm, lane = match.group(1), int(match.group(2))
    if lane in arms[arm]:
        raise SystemExit(f"duplicate {arm} lane {lane}")
    arms[arm][lane] = [int(value, 16) for value in match.groups()[2:]]

for arm, rows in arms.items():
    if sorted(rows) != list(range(32)):
        raise SystemExit(f"{arm}: lane denominator differs: {sorted(rows)}")
    for lane, values in rows.items():
        for value in values:
            provider, word = value >> 8, value & 0xff
            if provider >= 32 or word >= 8:
                raise SystemExit(
                    f"{arm}: invalid tag lane={lane} value=0x{value:04x}")

def providers(rows: dict[int, list[int]]) -> set[int]:
    return {value >> 8 for values in rows.values() for value in values}

local_providers = providers(arms["lane-local"])
collapsed_providers = providers(arms["collapsed"])
if local_providers != set(range(32)):
    raise SystemExit(
        f"lane-local did not consume all provider rows: {sorted(local_providers)}")
if collapsed_providers != {0}:
    raise SystemExit(
        f"collapsed control retained nonzero providers: {sorted(collapsed_providers)}")

encoded = "\n".join(
    f"{lane}:" + ",".join(f"{value:04x}" for value in arms["lane-local"][lane])
    for lane in range(32)
)
sha = hashlib.sha256(encoded.encode()).hexdigest()
print(
    "PLAIN_X4_VERDICT verdict=LANE_LOCAL_ADDRESSES_CONFIRMED "
    f"lane_local_providers={len(local_providers)} "
    f"collapsed_providers={len(collapsed_providers)} mapping_sha256={sha}")
for lane in range(32):
    values = arms["lane-local"][lane]
    sources = ",".join(f"{value >> 8}:{value & 0xff}" for value in values)
    print(f"PLAIN_X4_MAP lane={lane} sources={sources}")
print("[plain-x4] DIAGNOSTIC_COMPLETE rows=64 controls=2")
PY

grep -q '^PLAIN_X4_VERDICT verdict=LANE_LOCAL_ADDRESSES_CONFIRMED ' \
  "$RESULTS/verdict.log" || fail 'lane-local verdict absent'
printf '[plain-x4] DIAGNOSTIC_COMPLETE artifacts=%s\n' "$OUT"
