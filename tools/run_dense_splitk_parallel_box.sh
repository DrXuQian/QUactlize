#!/usr/bin/env bash
# Build and run the M==1 packed-A dense fixed Split-K correctness canary.
# All generated files remain under /workspace.  This script never replaces the
# caller's shell or starts/stops a container; a nonzero result terminates only
# this `bash tools/run_dense_splitk_parallel_box.sh` process.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=test_lowbit_dense_splitk_parallel
STAMP="$(date -u +%Y%m%dT%H%M%S%NZ)-$$"
SHORT_SHA="$(git -C "$ROOT" rev-parse --short=12 HEAD 2>/dev/null || printf unknown)"
RAW_ARTIFACT_ROOT="${SPLITK_PARALLEL_OUT:-/workspace/quactlize-dense-splitk-parallel-${SHORT_SHA}-${STAMP}}"
RAW_BUILD_ROOT="${PPU_BUILD_DIR:-$RAW_ARTIFACT_ROOT/build}"

canonical_workspace_child() {
  local label="$1" raw="$2" resolved
  [ -n "$raw" ] || {
    printf '[splitk-box] FAIL: %s is empty\n' "$label" >&2
    return 2
  }
  resolved="$(realpath -m -- "$raw")" || {
    printf '[splitk-box] FAIL: cannot canonicalize %s=%q\n' "$label" "$raw" >&2
    return 2
  }
  case "$resolved" in
    /workspace/?*) printf '%s\n' "$resolved" ;;
    *)
      printf '[splitk-box] FAIL: %s must resolve to a strict child of /workspace, got %s\n' \
        "$label" "$resolved" >&2
      return 2
      ;;
  esac
}

ARTIFACT_ROOT="$(canonical_workspace_child SPLITK_PARALLEL_OUT "$RAW_ARTIFACT_ROOT")" || exit $?
BUILD_ROOT="$(canonical_workspace_child PPU_BUILD_DIR "$RAW_BUILD_ROOT")" || exit $?
BUILD_LOG="$ARTIFACT_ROOT/build.log"
RUN_LOG="$ARTIFACT_ROOT/run.log"
MANIFEST="$ARTIFACT_ROOT/manifest.txt"
ITERATIONS="${SPLITK_PARALLEL_ITERATIONS:-20}"

case "$BUILD_ROOT" in
  "$ARTIFACT_ROOT"/?*) ;;
  *)
    printf '[splitk-box] FAIL: PPU_BUILD_DIR must be a strict child of the run artifact root; got %s vs %s\n' \
      "$BUILD_ROOT" "$ARTIFACT_ROOT" >&2
    exit 2
    ;;
esac

if [ -e "$ARTIFACT_ROOT" ]; then
  printf '[splitk-box] FAIL: refusing to reuse existing artifact root %s\n' \
    "$ARTIFACT_ROOT" >&2
  exit 2
fi
if [ -n "${PPU_DEFS:-}" ]; then
  printf '[splitk-box] FAIL: PPU_DEFS must be empty for the fixed canary, got %q\n' \
    "$PPU_DEFS" >&2
  exit 2
fi
if [ -n "${PPU_ARCHS+x}" ] && [ "$PPU_ARCHS" != ppu0010 ]; then
  printf '[splitk-box] FAIL: PPU_ARCHS must be ppu0010, got %q\n' "$PPU_ARCHS" >&2
  exit 2
fi

mkdir -p "$ARTIFACT_ROOT" "$BUILD_ROOT"

finish() {
  local rc="$1"
  printf '[splitk-box] exit_status=%s\n' "$rc" | tee -a "$MANIFEST"
  printf '[splitk-box] artifacts=%s\n' "$ARTIFACT_ROOT"
  return "$rc"
}

{
  printf 'root_sha=%s\n' "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
  printf 'root_status_begin\n'
  git -C "$ROOT" status --short --untracked-files=all 2>/dev/null || true
  printf 'root_status_end\n'
  printf 'actlize_sha=%s\n' "$(git -C "$ROOT/third_party/actlize" rev-parse HEAD 2>/dev/null || printf unavailable)"
  printf 'target=%s\n' "$TARGET"
  printf 'iterations=%s\n' "$ITERATIONS"
  printf 'build_root=%s\n' "$BUILD_ROOT"
  printf 'measurement_scope=correctness-canary,warm-single-artifact,performance-unadjudicated\n'
  printf 'ppu_archs=ppu0010\nppu_defs=<empty>\n'
  printf 'sdk_compiler=%s\n' "$(command -v hgcc 2>/dev/null || printf unavailable)"
  hgcc --version 2>&1 | sed -n '1p' | sed 's/^/sdk_compiler_identity=/' || true
} >"$MANIFEST"

if ! [[ "$ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
  printf '[splitk-box] FAIL: SPLITK_PARALLEL_ITERATIONS must be a positive integer, got %q\n' \
    "$ITERATIONS" | tee -a "$RUN_LOG" >&2
  finish 2
  exit $?
fi

printf '[splitk-box] build target=%s artifacts=%s\n' "$TARGET" "$ARTIFACT_ROOT"
set +e
env PPU_BUILD_DIR="$BUILD_ROOT" PPU_ARCHS=ppu0010 PPU_DEFS= \
  TARGET="$TARGET" QUANT=int4 BENCH_GS=128 \
  "$ROOT/build.sh" 2>&1 | tee "$BUILD_LOG"
build_rc=${PIPESTATUS[0]}
set -e
printf 'build_exit_status=%s\n' "$build_rc" >>"$MANIFEST"
if [ "$build_rc" -ne 0 ]; then
  printf '[splitk-box] FAIL: target did not build\n' >&2
  finish "$build_rc"
  exit $?
fi

mapfile -t binaries < <(find "$BUILD_ROOT" -type f -name "$TARGET" -perm -u+x -print)
if [ "${#binaries[@]}" -ne 1 ]; then
  printf '[splitk-box] FAIL: expected exactly one binary, found %d\n' \
    "${#binaries[@]}" >&2
  printf '  %s\n' "${binaries[@]:-<none>}" >&2
  finish 2
  exit $?
fi
BIN="${binaries[0]}"
BIN_SHA="$(sha256sum "$BIN" | awk '{print $1}')"
printf 'binary=%s\nbinary_sha256=%s\n' "$BIN" "$BIN_SHA" >>"$MANIFEST"
printf '[splitk-box] binary=%s sha256=%s\n' "$BIN" "$BIN_SHA"

set +e
"$BIN" "--iterations=$ITERATIONS" 2>&1 | tee "$RUN_LOG"
run_rc=${PIPESTATUS[0]}
set -e
printf 'run_exit_status=%s\n' "$run_rc" >>"$MANIFEST"

if [ "$run_rc" -eq 0 ]; then
  rows="$(grep -Ec '^\[splitk S=(1|2|4|8)\].* -> PASS$' "$RUN_LOG" || true)"
  if [ "$rows" -ne 4 ] ||
     ! grep -Fq '[splitk] PASS: S=1 M1 packed-A provider and S=2/4/8 parallel paths share one exact fixture, repeated correctness, and raw-half fingerprint' "$RUN_LOG" ||
     ! grep -Fq '[splitk perf] UNADJUDICATED:' "$RUN_LOG"; then
    printf '[splitk-box] FAIL: rc=0 lacked the four per-S PASS rows and aggregate PASS\n' >&2
    run_rc=3
  fi
fi

if [ "$run_rc" -ne 0 ]; then
  printf '[splitk-box] FAIL: benchmark returned %d\n' "$run_rc" >&2
else
  printf '[splitk-box] PASS: S=1/2/4/8 exact repeated correctness captured; performance remains UNADJUDICATED\n'
fi
finish "$run_rc"
exit $?
