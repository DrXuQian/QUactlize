#!/usr/bin/env bash
# One-box AP0/AP1 bisection of Q4_K Split-K partial production versus visibility.
set -uo pipefail

main() {
  local root workspace_root sha short stamp out jobs full generated build_dir
  local build_log binary direct_log probe_log direct_rc probe_rc
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace_root="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-q4k-split-workspace-${short}-${stamp}-$$}")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) printf '[fq-split-workspace] FAIL: OUT must be a strict /workspace child: %s\n' "$out" >&2; return 2 ;;
  esac
  if [ -e "$out" ]; then
    printf '[fq-split-workspace] FAIL: refusing to overwrite %s\n' "$out" >&2
    return 2
  fi
  jobs="${JOBS:-16}"
  case "$jobs" in
    *[!0-9]*|0) printf '[fq-split-workspace] FAIL: JOBS must be positive\n' >&2; return 2 ;;
  esac
  mkdir -p "$out/generated/full" "$out/generated/closure" \
    "$out/build" "$out/results" || return 2

  python3 -B "$root/tools/select_fq_split_timing_closure.py" --self-test || return 2
  python3 -B "$root/tools/check_fq_split_workspace_probe.py" --self-test || return 2
  python3 -B "$root/ci/check_fq_split_workspace_probe.py" || return 2
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
    printf '[fq-split-workspace] FAIL: exact target build rc=%d\n' "$build_rc" >&2
    tail -160 "$build_log" >&2
    printf '[fq-split-workspace] artifacts=%s\n' "$out" >&2
    return "$build_rc"
  fi
  binary="$build_dir/ppu_targets/test_fully_quantized_internal_sweep"
  if [ ! -x "$binary" ] || [ -L "$binary" ]; then
    binary="$(grep -m1 '^built: ' "$build_log" | cut -d' ' -f2-)"
  fi
  if [ -z "$binary" ] || [ ! -x "$binary" ] || [ -L "$binary" ]; then
    printf '[fq-split-workspace] FAIL: exact binary missing: %s\n' "$binary" >&2
    return 2
  fi

  direct_log="$out/results/direct.log"
  probe_log="$out/results/workspace-probe.log"
  "$binary" --shape=1x1024x5120 --iterations=1 \
    --correctness-repeats="${DIRECT_REPEATS:-64}" \
    --tm8-max-m=8 --bc-mode=skip >"$direct_log" 2>&1
  direct_rc=$?
  "$binary" --shape=1x1024x5120 --iterations=1 \
    --correctness-repeats="${PROBE_REPEATS:-16}" \
    --tm8-max-m=8 --bc-mode=skip --split-workspace-probe \
    >"$probe_log" 2>&1
  probe_rc=$?
  if { [ "$direct_rc" -ne 0 ] && [ "$direct_rc" -ne 1 ]; } || \
      [ "$probe_rc" -ne 0 ]; then
    printf '[fq-split-workspace] FAIL: arm rc direct=%d probe=%d\n' \
      "$direct_rc" "$probe_rc" >&2
    tail -60 "$direct_log" >&2
    tail -80 "$probe_log" >&2
    printf '[fq-split-workspace] artifacts=%s\n' "$out" >&2
    return 2
  fi
  python3 -B "$root/tools/check_fq_split_workspace_probe.py" \
    --direct "$direct_log" --probe "$probe_log" || {
      printf '[fq-split-workspace] artifacts=%s\n' "$out" >&2
      return 2
    }
  sha256sum "$binary" "$generated/manifest.json" "$direct_log" \
    "$probe_log" >"$out/results/authority.sha256" || return 2
  printf '[fq-split-workspace] PASS sha=%s direct_rc=%d probe_rc=%d artifacts=%s\n' \
    "$sha" "$direct_rc" "$probe_rc" "$out"
}

main "$@"
