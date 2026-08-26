#!/usr/bin/env bash
# Exact device closure for the Q4_K packed-metadata destination contract.
# The historical filename remains stable. It builds six tactics only: WN64 x
# stages={3,4} controls plus the WN16/stage2 root-cause row, both A providers.
# The current xplane ABI admits N multiples of 256 only, so the device closure
# runs the aligned production domain. L217 covers the future N-tail contract.
set -uo pipefail

main() {
  local root workspace_root sha short stamp out jobs full generated build_dir build_log binary aligned_log
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace_root="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-q4k-tm8-wn64-closure-${short}-${stamp}-$$}")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) printf '[fq-tm8-wn64] FAIL: OUT must be a strict /workspace child: %s\n' "$out" >&2; return 2 ;;
  esac
  if [ -e "$out" ]; then
    printf '[fq-tm8-wn64] FAIL: refusing to overwrite %s\n' "$out" >&2
    return 2
  fi
  jobs="${JOBS:-16}"
  case "$jobs" in *[!0-9]*|0) printf '[fq-tm8-wn64] FAIL: JOBS must be positive\n' >&2; return 2;; esac
  mkdir -p "$out/generated/full" "$out/generated/closure" "$out/build" "$out/results" || return 2

  python3 -B "$root/tools/select_fq_tm8_wn64_closure.py" --self-test || return 2
  python3 -B "$root/tools/check_fq_tm8_wn64_closure.py" --self-test || return 2
  python3 -B "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
    --qtype 12 --artifact-tk 64 --bchunk 0 --per-unit 1 \
    --out-dir "$out/generated/full" || return 2
  full="$out/generated/full"
  generated="$out/generated/closure"
  python3 -B "$root/tools/select_fq_tm8_wn64_closure.py" \
    --source-dir "$full" --out-dir "$generated" || return 2

  build_dir="$out/build"
  build_log="$out/results/build.log"
  (cd "$root" && \
    PPU_BUILD_DIR="$build_dir" PPU_ARCHS=ppu0010 JOBS="$jobs" \
    TARGET=test_fully_quantized_internal_sweep \
    FQ_SWEEP_GENERATED_DIR="$generated" FQ_SWEEP_QTYPE=12 \
    FQ_SWEEP_ARTIFACT_TK=64 FQ_SWEEP_BCHUNK=0 \
    FQ_SWEEP_PACKED_FORMAT=0 ./build.sh) >"$build_log" 2>&1
  local build_rc=$?
  if [ "$build_rc" -ne 0 ]; then
    printf '[fq-tm8-wn64] FAIL: exact six-row target did not build rc=%d\n' "$build_rc" >&2
    tail -n 160 "$build_log" >&2
    printf '[fq-tm8-wn64] artifacts=%s\n' "$out" >&2
    return "$build_rc"
  fi
  binary="$build_dir/ppu_targets/test_fully_quantized_internal_sweep"
  if [ ! -x "$binary" ] || [ -L "$binary" ]; then
    binary="$(grep -m1 '^built: ' "$build_log" | cut -d' ' -f2-)"
  fi
  if [ -z "$binary" ] || [ ! -x "$binary" ] || [ -L "$binary" ]; then
    printf '[fq-tm8-wn64] FAIL: exact binary missing: %s\n' "$binary" >&2
    return 2
  fi

  aligned_log="$out/results/device-aligned.log"
  "$binary" --shape=1x1024x5120 --iterations="${ITERATIONS:-3}" \
    --correctness-repeats="${CORRECTNESS_REPEATS:-8192}" \
    --only-split=0 --bc-mode=skip | tee "$aligned_log"
  local run_rc=${PIPESTATUS[0]}
  if [ "$run_rc" -ne 0 ]; then
    printf '[fq-tm8-wn64] FAIL: aligned device target rc=%d artifacts=%s\n' "$run_rc" "$out" >&2
    return "$run_rc"
  fi
  python3 -B "$root/tools/check_fq_tm8_wn64_closure.py" \
    --aligned-log "$aligned_log" || return 2
  sha256sum "$binary" "$generated/manifest.json" "$aligned_log" \
    > "$out/results/authority.sha256" || return 2
  printf '[fq-tm8-wn64] PASS sha=%s rows=6 shapes=1 WN64=RETAINED WN16-S1/S2/S4=RAW-BIT tail=LOCAL-L217/CURRENT-ABI-N%%256 artifacts=%s\n' \
    "$sha" "$out"
}

main "$@"
