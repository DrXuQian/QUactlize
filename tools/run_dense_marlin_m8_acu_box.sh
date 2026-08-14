#!/usr/bin/env bash
# Build and capture the standalone Marlin m8 extension at the exact m16
# reference N/K/WarpK/stage/artifact point.  This produces the native ACU
# report directly; it deliberately does not export or post-process CSV.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=test_lowbit_dense_marlin_m8_ab
ACU="${ACU:-$(command -v acu || true)}"
OUT="${MARLIN_M8_ACU_OUT:-/workspace/quactlize-dense-marlin-m8-acu}"

fail() { printf '[marlin-m8-acu] FAIL: %s\n' "$*" >&2; exit 1; }
[ "$#" -eq 0 ] || fail 'this runner accepts no positional arguments'
[ -n "$ACU" ] && [ -x "$ACU" ] || \
  fail 'ACU is unavailable; set ACU to the site acu executable'
[ -z "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)" ] \
  || fail 'source tree must be clean so the report names one exact SHA'
if [ -e "$OUT" ] && [ -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  fail "output directory is not empty: $OUT"
fi
mkdir -p "$OUT/build"

ROOT_SHA="$(git -C "$ROOT" rev-parse HEAD)"
ACTLIZE_SHA="$(git -C "$ROOT/third_party/actlize" rev-parse HEAD)"
git -C "$ROOT" submodule status --recursive >"$OUT/submodule-status.txt"
grep -Eq '^[+\-U]' "$OUT/submodule-status.txt" && \
  fail 'a submodule checkout differs from its recorded gitlink'

BUILD=(env PPU_BUILD_DIR="$OUT/build" PPU_ARCHS=ppu0010 TARGET="$TARGET"
       QUANT=int4 BENCH_GS=128 "$ROOT/build.sh")
printf '%q ' "${BUILD[@]}" >"$OUT/build.command"
printf '\n' >>"$OUT/build.command"
"${BUILD[@]}" 2>&1 | tee "$OUT/build.log"

mapfile -t bins < <(find "$OUT/build" -type f -name "$TARGET" -perm -u+x -print)
[ "${#bins[@]}" -eq 1 ] || fail "expected one $TARGET binary, found ${#bins[@]}"
BIN="${bins[0]}"
sha256sum "$BIN" >"$OUT/binary.sha256"

ARGS=(--marlin --streamk_exact_fixture --m=1 --n=4096 --k=4096
      --l=1 --g=128 --mode=1 --alpha=1 --beta=0 --iterations=0)
printf '%q ' "$BIN" "${ARGS[@]}" >"$OUT/correctness.command"
printf '\n' >>"$OUT/correctness.command"
"$BIN" "${ARGS[@]}" 2>&1 | tee "$OUT/correctness.log"
grep -Fq 'family=ppu-m8-extension' "$OUT/correctness.log" \
  || fail 'binary did not identify the m8 extension'
grep -Fq 'instruction=m8n16k16 tile=8x128x128 warp=8x64x32' \
  "$OUT/correctness.log" || fail 'compiled m8 topology was not printed'
grep -Fq 'stored_a_rows=1 a_copy_threads=16 a_load=plain-x2 load=cp.async' \
  "$OUT/correctness.log" || fail 'packed-row/x2 A delivery was not printed'
grep -Fq 'mapping=ppu-m8n16 coverage=exact-once' "$OUT/correctness.log" \
  || fail 'm8 output-owner map did not close'
grep -Fq 'Disposition: Passed' "$OUT/correctness.log" \
  || fail 'exact fixture did not pass'
[ "$(grep -Ec '^  \[dense marlin lock fingerprint\] repeat=[1-8]/8 raw_bitdiff=0 .* stable=1 same-workspace=1 external-lock-reset=0$' "$OUT/correctness.log" || true)" -eq 8 ] \
  || fail '8-launch lock fingerprint did not close'

REPORT="$OUT/marlin-m8-tm8-tn128-tk128-wn64-wk32.report"
ACU_CMD=("$ACU" -f -o "$REPORT" --set full "$BIN" "${ARGS[@]}")
printf '%q ' "${ACU_CMD[@]}" >"$OUT/acu.command"
printf '\n' >>"$OUT/acu.command"
"${ACU_CMD[@]}" 2>&1 | tee "$OUT/acu.log"
[ -s "$REPORT" ] || fail 'ACU produced no report'
grep -Fq 'family=ppu-m8-extension' "$OUT/acu.log" \
  || fail 'profiled process did not identify the m8 extension'
grep -Fq 'Disposition: Passed' "$OUT/acu.log" \
  || fail 'profiled exact fixture did not pass'

{
  printf 'root_sha=%s\nactlize_sha=%s\n' "$ROOT_SHA" "$ACTLIZE_SHA"
  printf 'target=%s\n' "$TARGET"
  printf 'shape=M1,N4096,K4096,L1,gs128\n'
  printf 'config=TM8,TN128,TK128,WM8,WN64,WarpK32,S4\n'
  printf 'instruction=m8n16k16\nstored_a_rows=1\n'
  printf 'a_copy_threads=16\na_copy_bytes_per_stage=256\na_load=plain-x2\n'
  printf 'acu_report=%s\n' "$REPORT"
  printf 'capture_protocol=one correctness launch plus eight same-workspace lock-fingerprint launches\n'
} >"$OUT/manifest.txt"
sha256sum "$OUT"/build.command "$OUT"/build.log "$OUT"/binary.sha256 \
  "$OUT"/correctness.command "$OUT"/correctness.log "$OUT"/acu.command \
  "$OUT"/acu.log "$OUT"/manifest.txt "$REPORT" >"$OUT/bundle.sha256"

printf '[marlin-m8-acu] PASS: exact m8 target built, passed, and was captured\n'
printf '[marlin-m8-acu] root-sha=%s binary=%s\n' "$ROOT_SHA" "$BIN"
printf '[marlin-m8-acu] report=%s\n' "$REPORT"
printf '[marlin-m8-acu] inspect the unique m8 standalone kernel symbol; no CSV was generated\n'
