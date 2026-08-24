#!/usr/bin/env bash
# Exact AP0/AP1 closure for the producer timing publication-gap defect.
set -uo pipefail

main() {
  local root workspace_root sha short stamp out jobs full generated build_dir
  local build_log binary legacy_log candidate_log legacy_rc candidate_rc
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace_root="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-q4k-split-timing-${short}-${stamp}-$$}")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) printf '[fq-split-timing] FAIL: OUT must be a strict /workspace child: %s\n' "$out" >&2; return 2 ;;
  esac
  if [ -e "$out" ]; then
    printf '[fq-split-timing] FAIL: refusing to overwrite %s\n' "$out" >&2
    return 2
  fi
  jobs="${JOBS:-16}"
  case "$jobs" in *[!0-9]*|0) printf '[fq-split-timing] FAIL: JOBS must be positive\n' >&2; return 2;; esac
  mkdir -p "$out/generated/full" "$out/generated/closure" "$out/build" "$out/results" || return 2

  python3 -B "$root/ci/check_splitk_producer_timing.py" || return 2
  python3 -B "$root/tools/select_fq_split_timing_closure.py" --self-test || return 2
  python3 -B "$root/tools/check_fq_split_timing_closure.py" --self-test || return 2
  python3 -B "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
    --qtype 12 --artifact-tk 64 --bchunk 0 --tile-m-filter 8 \
    --per-unit 1 --out-dir "$out/generated/full" || return 2
  full="$out/generated/full"
  generated="$out/generated/closure"
  python3 -B "$root/tools/select_fq_split_timing_closure.py" \
    --source-dir "$full" --out-dir "$generated" || return 2

  build_dir="$out/build"
  build_log="$out/results/build.log"
  (cd "$root" && PPU_BUILD_DIR="$build_dir" PPU_ARCHS=ppu0010 JOBS="$jobs" \
    TARGET=test_fully_quantized_internal_sweep \
    FQ_SWEEP_GENERATED_DIR="$generated" FQ_SWEEP_QTYPE=12 \
    FQ_SWEEP_ARTIFACT_TK=64 FQ_SWEEP_BCHUNK=0 \
    FQ_SWEEP_PACKED_FORMAT=0 ./build.sh) >"$build_log" 2>&1
  local build_rc=$?
  if [ "$build_rc" -ne 0 ]; then
    printf '[fq-split-timing] FAIL: exact target build rc=%d\n' "$build_rc" >&2
    tail -160 "$build_log" >&2
    printf '[fq-split-timing] artifacts=%s\n' "$out" >&2
    return "$build_rc"
  fi
  binary="$build_dir/ppu_targets/test_fully_quantized_internal_sweep"
  if [ ! -x "$binary" ] || [ -L "$binary" ]; then
    binary="$(grep -m1 '^built: ' "$build_log" | cut -d' ' -f2-)"
  fi
  if [ -z "$binary" ] || [ ! -x "$binary" ] || [ -L "$binary" ]; then
    printf '[fq-split-timing] FAIL: exact binary missing: %s\n' "$binary" >&2
    return 2
  fi

  legacy_log="$out/results/legacy.log"
  candidate_log="$out/results/candidate.log"
  "$binary" --shape=1x1024x5120 --iterations="${ITERATIONS:-7}" \
    --correctness-repeats="${CORRECTNESS_REPEATS:-8}" \
    --tm8-max-m=8 --bc-mode=skip --legacy-split-timing \
    >"$legacy_log" 2>&1
  legacy_rc=$?
  "$binary" --shape=1x1024x5120 --iterations="${ITERATIONS:-7}" \
    --correctness-repeats="${CORRECTNESS_REPEATS:-8}" \
    --tm8-max-m=8 --bc-mode=skip >"$candidate_log" 2>&1
  candidate_rc=$?
  if [ "$legacy_rc" -ne 1 ] || [ "$candidate_rc" -ne 0 ]; then
    printf '[fq-split-timing] FAIL: arm rc legacy=%d candidate=%d\n' \
      "$legacy_rc" "$candidate_rc" >&2
    tail -40 "$legacy_log" >&2
    tail -40 "$candidate_log" >&2
    printf '[fq-split-timing] artifacts=%s\n' "$out" >&2
    return 2
  fi
  python3 -B "$root/tools/check_fq_split_timing_closure.py" \
    --legacy "$legacy_log" --candidate "$candidate_log" || return 2
  sha256sum "$binary" "$generated/manifest.json" "$legacy_log" \
    "$candidate_log" >"$out/results/authority.sha256" || return 2
  printf '[fq-split-timing] PASS sha=%s legacy=REPRODUCED candidate=RAW-BIT-EXACT artifacts=%s\n' \
    "$sha" "$out"
}

main "$@"
