#!/usr/bin/env bash
# One exact standalone-classic launch: ACU instruction mix + numRegs/occupancy.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLASSIC_ROOT="${CLASSIC_ROOT:-$(cd "$ROOT/.." && pwd)}"
ACU="${ACU:-/sim/eec/shared/junfu.qx/asight/bin/acu}"
SRC="$ROOT/tools/classic_marlin_156_profile.cu"
HEADER="$CLASSIC_ROOT/marlin_classic_ppu.cuh"
OUT="${CLASSIC156_OUT:-$(mktemp -d "${TMPDIR:-/tmp}/classic-marlin-156.XXXXXX")}"
BIN="$OUT/classic_marlin_156_profile"

fail() { printf '[classic-156] FAIL: %s\n' "$*" >&2; exit 1; }
[ "$#" -eq 0 ] || fail 'this runner accepts no positional arguments'
test -f "$SRC" || fail "missing profile harness: $SRC"
test -f "$HEADER" || fail "missing standalone header: $HEADER"
test -x "$ACU" || fail "ACU is not executable: $ACU"
test -z "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)" \
  || fail 'quactlize source tree must be clean'
mkdir -p "$OUT"

ROOT_SHA="$(git -C "$ROOT" rev-parse HEAD)"
NVCC="$(command -v nvcc)"
test -n "$NVCC" || fail 'nvcc/hgcc driver is unavailable'
NVCC="$(readlink -f "$NVCC")"
python3 "$ROOT/tools/probe_box_identity.py" resolve --output "$OUT/identity.json"

sha256sum "$SRC" "$HEADER" >"$OUT/source.before.sha256"
{
  printf 'root_sha=%s\nclassic_root=%s\ncompiler=%s\n' "$ROOT_SHA" "$CLASSIC_ROOT" "$NVCC"
  "$NVCC" --version
  "$ACU" --version || true
} >"$OUT/toolchain.txt" 2>&1

BUILD=("$NVCC" -O3 -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda
       -DMARLIN_STAGES=4 -DMARLIN_MAX_MB=2 -DMARLIN_MIN_BLOCKS=2
       -I "$CLASSIC_ROOT" -o "$BIN" "$SRC")
printf '%q ' "${BUILD[@]}" >"$OUT/build.command"
printf '\n' >>"$OUT/build.command"
"${BUILD[@]}" 2>&1 | tee "$OUT/build.log"
sha256sum "$SRC" "$HEADER" >"$OUT/source.after.sha256"
cmp "$OUT/source.before.sha256" "$OUT/source.after.sha256"
sha256sum "$BIN" >"$OUT/binary.sha256"

unset MARLIN_NOSPLITK MARLIN_SPLITK MARLIN_BLOCKS MARLIN_MAX_PAR \
  MARLIN_WORKSET_MB MARLIN_HBM_GBS MARLIN_GROUPSIZE
ACU_CMD=("$ACU" -f -o "$OUT/classic.report" --set full "$BIN")
printf '%q ' "${ACU_CMD[@]}" >"$OUT/acu.command"
printf '\n' >>"$OUT/acu.command"
"${ACU_CMD[@]}" 2>&1 | tee "$OUT/acu.log"
grep -Eq '^MARLIN156 launch-count=1 rc=0 sync_code=0 sync=' "$OUT/acu.log" \
  || fail 'the capture did not execute exactly one successful classic launch'
"$ACU" --import "$OUT/classic.report" --csv --page details \
  >"$OUT/classic.details.csv"
test -s "$OUT/classic.details.csv" || fail 'ACU details CSV is empty'

python3 - "$OUT/manifest.json" "$ROOT_SHA" "$CLASSIC_ROOT" <<'PY'
import hashlib, json, pathlib, sys
out, root_sha, classic_root = sys.argv[1:]
base = pathlib.Path(out).parent
files = ["identity.json", "source.before.sha256", "source.after.sha256",
         "toolchain.txt", "build.command", "build.log", "binary.sha256",
         "acu.command", "acu.log", "classic.report", "classic.details.csv"]
payload = {
    "schema": "quactlize.classic-marlin-156.v1",
    "root_sha": root_sha,
    "classic_root": classic_root,
    "shape": {"M": 1, "N": 4096, "K": 4096, "group_size": 128},
    "kernel": "Marlin<256,1,8,8,4,8>",
    "expected_vmma_per_launch": 65536,
    "launches": 1,
    "files": {},
}
for name in files:
    p = base / name
    payload["files"][name] = hashlib.sha256(p.read_bytes()).hexdigest()
pathlib.Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
sha256sum "$OUT/manifest.json" >"$OUT/manifest.sha256"

printf '[classic-156] PASS: one exact classic launch captured\n'
printf '[classic-156] result-sha=%s\n' "$ROOT_SHA"
printf '[classic-156] bundle=%s\n' "$OUT"
printf '[classic-156] return the report/details CSV; classify only Marlin<256,1,8,8,4,8>, block=256, grid=CU\n'
