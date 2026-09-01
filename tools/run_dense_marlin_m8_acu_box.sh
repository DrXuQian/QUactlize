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
grep -Eq '^[+U-]' "$OUT/submodule-status.txt" && \
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

REPORTS=()
for bpc in 1 2 3; do
  bpc_flag=()
  if [ "$bpc" -ne 1 ]; then
    bpc_flag=("--marlin-blocks-per-cu=$bpc")
  fi
  correctness_log="$OUT/correctness-bpc${bpc}.log"
  correctness_cmd="$OUT/correctness-bpc${bpc}.command"
  CORRECTNESS=("$BIN" "${ARGS[@]}" "${bpc_flag[@]}")
  printf '%q ' "${CORRECTNESS[@]}" >"$correctness_cmd"
  printf '\n' >>"$correctness_cmd"
  "${CORRECTNESS[@]}" 2>&1 | tee "$correctness_log"
  grep -Fq 'family=ppu-m8-extension' "$correctness_log" \
    || fail "binary did not identify m8 for blocks_per_cu=$bpc"
  grep -Fq 'instruction=m8n16k16 tile=8x128x128 warp=8x64x32' \
    "$correctness_log" || fail "compiled m8 topology missing for blocks_per_cu=$bpc"
  grep -Fq 'stored_a_rows=1 a_copy_threads=16 a_load=plain-x2 load=cp.async' \
    "$correctness_log" || fail "packed-row/x2 A delivery missing for blocks_per_cu=$bpc"
  grep -Fq 'mapping=ppu-m8n16 coverage=exact-once' "$correctness_log" \
    || fail "m8 output-owner map did not close for blocks_per_cu=$bpc"
  grep -Fq 'Disposition: Passed' "$correctness_log" \
    || fail "exact fixture failed for blocks_per_cu=$bpc"
  [ "$(grep -Ec '^  \[dense marlin lock fingerprint\] repeat=[1-8]/8 raw_bitdiff=0 .* stable=1 same-workspace=1 external-lock-reset=0$' "$correctness_log" || true)" -eq 8 ] \
    || fail "8-launch lock fingerprint failed for blocks_per_cu=$bpc"

  # ACU releases differ in whether -o names the final file or a basename to
  # which the tool appends .acurep.  Give it the stable basename, then require
  # exactly one of those two documented spellings.  A glob would be unsafe:
  # it could silently consume a stale/partial report from another invocation.
  report_base="$OUT/marlin-m8-tm8-tn128-tk128-wn64-wk32-bpc${bpc}.report"
  log="$OUT/acu-bpc${bpc}.log"
  cmd="$OUT/acu-bpc${bpc}.command"
  ACU_CMD=("$ACU" -f -o "$report_base" --set full "$BIN" "${ARGS[@]}"
           --marlin-profile-subject-only "${bpc_flag[@]}")
  printf '%q ' "${ACU_CMD[@]}" >"$cmd"
  printf '\n' >>"$cmd"
  "${ACU_CMD[@]}" 2>&1 | tee "$log"
  report_candidates=()
  [ -s "$report_base" ] && report_candidates+=("$report_base")
  [ -s "${report_base}.acurep" ] && report_candidates+=("${report_base}.acurep")
  [ "${#report_candidates[@]}" -eq 1 ] || \
    fail "ACU produced ${#report_candidates[@]} unambiguous report files for blocks_per_cu=$bpc (expected exactly one of $report_base or ${report_base}.acurep)"
  report="${report_candidates[0]}"
  grep -Fq 'family=ppu-m8-extension' "$log" \
    || fail "profiled process did not identify m8 for blocks_per_cu=$bpc"
  grep -Fq "[dense marlin ACU subject-only] instruction=m8n16k16 blocks_per_cu=$bpc subject_launches=1 device_reference=0 lock_fingerprints=0" "$log" \
    || fail "subject-only launch contract did not close for blocks_per_cu=$bpc"
  [ "$(grep -Fc '[dense marlin ACU subject-only]' "$log" || true)" -eq 1 ] \
    || fail "subject-only marker count is not one for blocks_per_cu=$bpc"
  ! grep -Fq 'Disposition:' "$log" \
    || fail "ACU process unexpectedly ran numerical verification for blocks_per_cu=$bpc"
  ! grep -Fq '[dense marlin lock fingerprint]' "$log" \
    || fail "ACU process unexpectedly ran lock fingerprints for blocks_per_cu=$bpc"
  REPORTS+=("$report")
done

{
  printf 'root_sha=%s\nactlize_sha=%s\n' "$ROOT_SHA" "$ACTLIZE_SHA"
  printf 'target=%s\n' "$TARGET"
  printf 'shape=M1,N4096,K4096,L1,gs128\n'
  printf 'config=TM8,TN128,TK128,WM8,WN64,WarpK32,S4\n'
  printf 'instruction=m8n16k16\nstored_a_rows=1\n'
  printf 'a_copy_threads=16\na_copy_bytes_per_stage=256\na_load=plain-x2\n'
  printf 'acu_reports=%s,%s,%s\n' "${REPORTS[0]}" "${REPORTS[1]}" "${REPORTS[2]}"
  printf 'capture_protocol=correctness process closes golden+8 fingerprints; each ACU process launches only one m8 subject and no device reference\n'
} >"$OUT/manifest.txt"
sha256sum "$OUT"/build.command "$OUT"/build.log "$OUT"/binary.sha256 \
  "$OUT"/correctness-bpc{1,2,3}.command "$OUT"/correctness-bpc{1,2,3}.log \
  "$OUT"/acu-bpc{1,2,3}.command "$OUT"/acu-bpc{1,2,3}.log \
  "$OUT"/manifest.txt "${REPORTS[@]}" >"$OUT/bundle.sha256"

printf '[marlin-m8-acu] PASS: exact m8 target built, passed, and was captured\n'
printf '[marlin-m8-acu] root-sha=%s binary=%s\n' "$ROOT_SHA" "$BIN"
printf '[marlin-m8-acu] reports=%s,%s,%s\n' "${REPORTS[0]}" "${REPORTS[1]}" "${REPORTS[2]}"
printf '[marlin-m8-acu] each report contains one m8 subject launch and no m16 GemmRef; no CSV was generated\n'
