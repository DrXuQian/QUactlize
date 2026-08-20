#!/usr/bin/env bash
# Build and run only the Q4_K/A32 row that first exposed the folded-reader
# numeric defect. This is a correctness closure, not a sweep.
set -uo pipefail

main() {
  local root sha short stamp out generated build binary log symbol rc
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="${OUT:-/workspace/quactlize-q4-a32-exact-${short}-${stamp}}"
  case "$out" in
    /workspace/*) ;;
    *) printf '[q4-a32-exact] FAIL: OUT must be below /workspace: %s\n' "$out" >&2; return 2 ;;
  esac
  if [ -e "$out" ]; then
    printf '[q4-a32-exact] FAIL: refusing existing OUT=%s\n' "$out" >&2
    return 2
  fi
  mkdir -p "$out/generated" "$out/build" "$out/results" || return 2
  symbol=sf_q12_a32_tm64_tn64_tk128_wm16_wn32_s8_bc0
  generated="$out/generated/q12-a32-bc0-exact"

  python3 -B "$root/tools/gen_scalefirst_internal_units.py" \
    --qtype 12 --artifact-tk 32 --bchunk 0 --per-unit 1 \
    --select-symbol "$symbol" --out-dir "$generated" \
    >"$out/results/generate.log" 2>&1 || {
      tail -80 "$out/results/generate.log" >&2
      return 1
    }

  build="$out/build/q12-a32-bc0-exact"
  log="$out/results/build.log"
  (cd "$root" && PPU_BUILD_DIR="$build" PPU_ARCHS=ppu0010 \
    JOBS="${JOBS:-16}" TARGET=test_scalefirst_internal_sweep \
    SCALEFIRST_SWEEP_GENERATED_DIR="$generated" \
    SCALEFIRST_SWEEP_QTYPE=12 SCALEFIRST_SWEEP_ARTIFACT_TK=32 \
    SCALEFIRST_SWEEP_BCHUNK=0 ./build.sh) >"$log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    printf '[q4-a32-exact] FAIL: exact target build rc=%s\n' "$rc" >&2
    tail -120 "$log" >&2
    return "$rc"
  fi
  binary="$build/ppu_targets/test_scalefirst_internal_sweep"
  if [ ! -x "$binary" ]; then
    printf '[q4-a32-exact] FAIL: exact binary missing: %s\n' "$binary" >&2
    return 1
  fi
  sha256sum "$binary" >"$out/results/binary.sha256"
  printf '%s\n' "$sha" >"$out/results/git.sha"
  printf '[q4-a32-exact] sha=%s symbol=%s binary=%s\n' "$sha" "$symbol" "$binary"

  "$binary" --shape=64x1024x5120 --iterations="${ITERATIONS:-3}" \
    --correctness-repeats="${CORRECTNESS_REPEATS:-8}" \
    --algorithm=nonpersistent 2>&1 | tee "$out/results/exact.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    printf '[q4-a32-exact] FAIL: exact numeric target rc=%s artifacts=%s\n' \
      "$rc" "$out" >&2
    return "$rc"
  fi
  grep -q 'SF_COMPLETE status=COMPLETE shape=64x1024x5120 typed_rows=1' \
    "$out/results/exact.log" || {
      printf '[q4-a32-exact] FAIL: exact completion marker missing\n' >&2
      return 1
    }
  grep -q 'raw_bad":0' "$out/results/exact.log" || {
      printf '[q4-a32-exact] FAIL: no raw-bit exact result\n' >&2
      return 1
    }
  printf '[q4-a32-exact] PASS: exact row raw-bit closed; artifacts=%s\n' "$out"
}

main "$@"
