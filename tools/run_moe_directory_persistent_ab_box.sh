#!/usr/bin/env bash
# Exact one-row A/B for the DeepGEMM-style directory scheduler port.
#
# Both arms compile the same format, resident xplane layout and quant mode and
# differ only in PPU_MOE_PERSISTENT.  The default remains the historical i4
# ScaleOnly control so an old command keeps its meaning; the multiformat runner
# below calls this script with the real GGUF ScaleFirst semantics for Q2_K..
# Q6_K.  The primary time for the persistent arm includes its one-CTA directory
# build.  A full raw-output FNV is compared before timing is interpreted, so a
# fast tile mapping cannot hide a missing or duplicated tile.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHA="$(git -C "$ROOT" rev-parse HEAD)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-/workspace/quactlize-moe-directory-ab-${SHA:0:8}-${STAMP}}"
JOBS="${JOBS:-8}"
REPS="${MOE_REPS:-3}"
FORMAT="${MOE_FORMAT:-i4}"
QMODE="${MOE_QMODE:-1}"

# The format registry owns bit widths, group size and the resident ScaleFirst
# TileK.  This shell projection additionally names the generated MoE tag and a
# representative legal tactic.  check_moe_directory_persistent.py parses both
# sources and fails if the projection drifts.
case "$FORMAT" in
  i2)
    QTYPE=10; FORMAT_NAME=Q2_K; GROUP_SIZE=16; ARTIFACT_TILEK=128
    LOW_BITS=2; HIGH_BITS=0; PLANES=1
    DEFAULT_ROW='i2 64x64:128 w64x32 s3 bc0->0'
    ;;
  q3)
    QTYPE=11; FORMAT_NAME=Q3_K; GROUP_SIZE=16; ARTIFACT_TILEK=256
    LOW_BITS=2; HIGH_BITS=1; PLANES=2
    DEFAULT_ROW='q3 64x64:256 w64x64 s3 bc0->0'
    ;;
  i4)
    QTYPE=12; FORMAT_NAME=Q4_K; GROUP_SIZE=32; ARTIFACT_TILEK=64
    LOW_BITS=4; HIGH_BITS=0; PLANES=1
    DEFAULT_ROW='i4 64x64:64 w64x32 s3 bc0->0'
    ;;
  q5)
    QTYPE=13; FORMAT_NAME=Q5_K; GROUP_SIZE=32; ARTIFACT_TILEK=256
    LOW_BITS=4; HIGH_BITS=1; PLANES=2
    DEFAULT_ROW='q5 64x64:256 w64x64 s3 bc0->0'
    ;;
  q6)
    QTYPE=14; FORMAT_NAME=Q6_K; GROUP_SIZE=16; ARTIFACT_TILEK=128
    LOW_BITS=4; HIGH_BITS=2; PLANES=2
    DEFAULT_ROW='q6 64x64:128 w64x32 s3 bc0->0'
    ;;
  *)
    printf '[moe-directory-ab] FAIL: MOE_FORMAT=%s; expected i2|q3|i4|q5|q6\n' "$FORMAT" >&2
    exit 2
    ;;
esac
case "$QMODE" in
  0) QMODE_NAME=ScaleZero ;;
  1) QMODE_NAME=ScaleOnly ;;
  *)
    printf '[moe-directory-ab] FAIL: MOE_QMODE=%s; expected 0 (ScaleZero) or 1 (ScaleOnly)\n' "$QMODE" >&2
    exit 2
    ;;
esac
MOE_ARGS="${MOE_ARGS:-64 128 2048 2048 $GROUP_SIZE 2 8}"
ROW="${MOE_ONLY:-$DEFAULT_ROW}"
read -r -a MOE_ARG_FIELDS <<<"$MOE_ARGS"
if [ "${#MOE_ARG_FIELDS[@]}" -ne 7 ] || [ "${MOE_ARG_FIELDS[4]}" != "$GROUP_SIZE" ]; then
  printf '[moe-directory-ab] FAIL: MOE_ARGS must have seven fields and gs=%s for %s: %s\n' \
    "$GROUP_SIZE" "$FORMAT_NAME" "$MOE_ARGS" >&2
  exit 2
fi
case "$ROW" in
  "$FORMAT "*) ;;
  *)
    printf '[moe-directory-ab] FAIL: MOE_ONLY row must belong to format %s: %s\n' "$FORMAT" "$ROW" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT"
printf '[moe-directory-ab] sha=%s out=%s\n' "$SHA" "$OUT"
printf '[moe-directory-ab] format=%s qtype=%s quant=%s gs=%s bits=%s+%s planes=%s layout=ScaleFirst-xplane-A%s metadata=fp16-%s row=%s args=%s\n' \
  "$FORMAT_NAME" "$QTYPE" "$QMODE_NAME" "$GROUP_SIZE" "$LOW_BITS" "$HIGH_BITS" "$PLANES" \
  "$ARTIFACT_TILEK" "$QMODE_NAME" "$ROW" "$MOE_ARGS"
printf 'format=%s\nqtype=%s\nquant=%s\ngroup_size=%s\nlow_bits=%s\nhigh_bits=%s\nplanes=%s\n' \
  "$FORMAT_NAME" "$QTYPE" "$QMODE_NAME" "$GROUP_SIZE" "$LOW_BITS" "$HIGH_BITS" "$PLANES" \
  >"$OUT/format-contract.txt"
printf 'artifact_layout=ScaleFirst-xplane\nartifact_tile_k=%s\nmetadata_layout=fp16-%s\nrow=%s\n' \
  "$ARTIFACT_TILEK" "$QMODE_NAME" "$ROW" >>"$OUT/format-contract.txt"
git -C "$ROOT" status --short >"$OUT/git-status.txt"
git -C "$ROOT/third_party/actlize" rev-parse HEAD >"$OUT/actlize-sha.txt"
# HEAD alone is not a source identity when the shared worktree is dirty.  Keep
# the complete tracked patch plus every untracked file so the binary can be
# reconstructed without committing somebody else's in-flight work.
git -C "$ROOT" diff --binary HEAD >"$OUT/tracked-worktree.patch"
git -C "$ROOT" ls-files --others --exclude-standard >"$OUT/untracked-files.txt"
if [ -s "$OUT/untracked-files.txt" ]; then
  tar -czf "$OUT/untracked-files.tar.gz" -C "$ROOT" -T "$OUT/untracked-files.txt"
else
  tar -czf "$OUT/untracked-files.tar.gz" -C "$ROOT" --files-from /dev/null
fi
(cd "$OUT" && sha256sum tracked-worktree.patch untracked-files.tar.gz \
  >source-bundle.sha256)
printf '[moe-directory-ab] source_bundle=%s\n' \
  "$(cd "$OUT" && sha256sum source-bundle.sha256 | cut -d' ' -f1)"

fail() {
  printf '[moe-directory-ab] FAIL: %s\n' "$*" >&2
  printf '[moe-directory-ab] artifacts: %s\n' "$OUT" >&2
  return 1
}

build_and_run() {
  local arm="$1" enabled="$2" dir="$OUT/$1" build="$OUT/$1/build"
  local build_log="$dir/build.log" run_log="$dir/run.log" bin
  mkdir -p "$dir" "$build"

  # The five public MOE_* filters apply to BOTH registered bands in one CMake
  # configure.  TM64/WM64 is the full-band row under test, but the bounded-M
  # decode table deliberately has no TM64 or WM64 row; restricting both axes
  # to 64 therefore makes the correctly fail-closed generator reject an empty
  # decode target before this full-band target can build.  Keep the smallest
  # legal decode sentinel (TM8/WM8) in the compile graph.  MOE_ONLY below still
  # selects exactly ROW, so neither A/B arm executes the sentinel.
  if ! (cd "$ROOT" && env \
      PPU_BUILD_DIR="$build" PPU_ARCHS=ppu0010 TARGET=test_lowbit_moe_bench \
      MOE_FORMATS="$FORMAT" MOE_TM_LIST='8;64' MOE_TN_LIST=64 MOE_WM_LIST='8;64' MOE_STAGES=3 \
      PPU_DEFS="LOWBIT_QMODE=$QMODE PPU_MOE_PERSISTENT=$enabled" JOBS="$JOBS" \
      bash build.sh) >"$build_log" 2>&1; then
    tail -120 "$build_log" >&2
    fail "$arm build failed"
    return 1
  fi
  grep -q "PPU_DEFS verified on test_lowbit_moe_bench's compile command: -DPPU_MOE_PERSISTENT=$enabled" \
    "$build_log" || { fail "$arm binary did not receive PPU_MOE_PERSISTENT=$enabled"; return 1; }
  grep -q "PPU_DEFS verified on test_lowbit_moe_bench's compile command: -DLOWBIT_QMODE=$QMODE" \
    "$build_log" || { fail "$arm binary did not receive LOWBIT_QMODE=$QMODE"; return 1; }

  mapfile -t bins < <(find "$build" -type f -name test_lowbit_moe_bench -perm -u+x -print)
  if [ "${#bins[@]}" -ne 1 ]; then
    printf '%s\n' "${bins[@]:-<none>}" >&2
    fail "$arm expected exactly one benchmark binary, found ${#bins[@]}"
    return 1
  fi
  bin="${bins[0]}"
  sha256sum "$bin" >"$dir/binary.sha256"
  printf '%s\n' "$bin" >"$dir/binary.path"

  # shellcheck disable=SC2086 -- MOE_ARGS is deliberately a seven-field CLI.
  if ! env MOE_ONLY="$ROW" MOE_OUTPUT_HASH=1 MOE_VERBOSE=1 MOE_REPS="$REPS" \
      "$bin" $MOE_ARGS >"$run_log" 2>&1; then
    tail -160 "$run_log" >&2
    fail "$arm benchmark returned nonzero"
    return 1
  fi
  local expected_scheduler=non-persistent expected_span=kernel-span-upper
  [ "$enabled" = 1 ] && expected_scheduler=persistent-directory
  [ "$enabled" = 1 ] && expected_span=scheduler-span-upper
  grep -q "^\[lowbit-moe\] scheduler=$expected_scheduler " "$run_log" || {
    fail "$arm run reported the wrong scheduler"; return 1; }
  grep -q "quant=$QMODE_NAME" "$run_log" || {
    fail "$arm run reported the wrong quant mode"; return 1; }
  grep -qF "$ROW" "$run_log" || { fail "$arm selected row never ran"; return 1; }
  ! grep -q 'DID NOT RUN\|DID NOT TIME\|ERROR:' "$run_log" || {
    fail "$arm row was refused or incomplete"; return 1; }
  [ "$(grep -c '^\[lowbit-moe output\] raw_bytes=' "$run_log")" -eq "$REPS" ] || {
    fail "$arm must emit one full-output hash per pass"; return 1; }
  [ "$(grep '^\[lowbit-moe output\]' "$run_log" | sort -u | wc -l)" -eq 1 ] || {
    fail "$arm output changed between repeated passes"; return 1; }

  grep '^\[lowbit-moe output\]' "$run_log" | sort -u >"$dir/output-hash.txt"
  grep -F "$ROW" "$run_log" | grep "$expected_span" | tail -1 >"$dir/result-line.txt"
  printf '[moe-directory-ab] arm=%s binary_sha256=%s\n' \
    "$arm" "$(cut -d' ' -f1 "$dir/binary.sha256")"
  cat "$dir/output-hash.txt" "$dir/result-line.txt"
}

build_and_run nonpersistent 0
build_and_run persistent_directory 1

base_hash="$(cat "$OUT/nonpersistent/output-hash.txt")"
subject_hash="$(cat "$OUT/persistent_directory/output-hash.txt")"
if [ "$base_hash" != "$subject_hash" ]; then
  fail "raw output differs between non-persistent and directory-persistent arms"
fi

printf '[moe-directory-ab] PASS: full D raw-bit hash identical; persistent timing includes directory build\n'
printf '[moe-directory-ab] contract: %s %s ScaleFirst-xplane-A%s fp16-%s\n' \
  "$FORMAT_NAME" "$FORMAT" "$ARTIFACT_TILEK" "$QMODE_NAME"
printf '[moe-directory-ab] baseline: '; cat "$OUT/nonpersistent/result-line.txt"
printf '[moe-directory-ab] subject:  '; cat "$OUT/persistent_directory/result-line.txt"
printf '[moe-directory-ab] artifacts: %s\n' "$OUT"
