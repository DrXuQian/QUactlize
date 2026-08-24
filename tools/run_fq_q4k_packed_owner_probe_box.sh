#!/usr/bin/env bash
# One new device build: compare owner-only packed metadata publication against
# the already-captured exact legacy workspace probe.
set -uo pipefail

main() {
  local root workspace_root legacy sha short stamp out jobs full generated
  local build_dir build_log binary direct_log probe_log direct_rc probe_rc
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace_root="$(realpath -e /workspace)" || return 2
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    printf '[fq-packed-owner] FAIL: ambient PPU_DEFS/PPU_EXTRA_DEFS changes the exact arm\n' >&2
    return 2
  fi
  if [ -z "${LEGACY_ARTIFACT:-}" ]; then
    printf '[fq-packed-owner] FAIL: LEGACY_ARTIFACT must name the prior 43fb02b workspace-probe artifact\n' >&2
    return 2
  fi
  legacy="$(realpath -e -- "$LEGACY_ARTIFACT")" || return 2
  case "$legacy" in
    "$workspace_root"/*) ;;
    *) printf '[fq-packed-owner] FAIL: legacy artifact must be a strict /workspace child: %s\n' "$legacy" >&2; return 2 ;;
  esac
  for required in results/direct.log results/workspace-probe.log results/authority.sha256; do
    if [ ! -f "$legacy/$required" ] || [ -L "$legacy/$required" ]; then
      printf '[fq-packed-owner] FAIL: legacy authority missing/symlinked: %s\n' "$legacy/$required" >&2
      return 2
    fi
  done
  sha256sum -c "$legacy/results/authority.sha256" >/dev/null || {
    printf '[fq-packed-owner] FAIL: legacy artifact hash authority changed\n' >&2
    return 2
  }

  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-q4k-packed-owner-${short}-${stamp}-$$}")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) printf '[fq-packed-owner] FAIL: OUT must be a strict /workspace child: %s\n' "$out" >&2; return 2 ;;
  esac
  if [ -e "$out" ]; then
    printf '[fq-packed-owner] FAIL: refusing to overwrite %s\n' "$out" >&2
    return 2
  fi
  jobs="${JOBS:-16}"
  case "$jobs" in
    *[!0-9]*|0) printf '[fq-packed-owner] FAIL: JOBS must be positive\n' >&2; return 2 ;;
  esac
  mkdir -p "$out/generated/full" "$out/generated/closure" \
    "$out/build" "$out/results" || return 2

  python3 -B "$root/tools/select_fq_split_timing_closure.py" --self-test || return 2
  python3 -B "$root/tools/check_fq_packed_owner_candidate.py" --self-test || return 2
  python3 -B "$root/ci/check_fq_packed_owner_candidate.py" || return 2
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
    PPU_DEFS="PPU_PACKED_METADATA_OWNER_ONLY=1" \
    TARGET=test_fully_quantized_internal_sweep \
    FQ_SWEEP_GENERATED_DIR="$generated" FQ_SWEEP_QTYPE=12 \
    FQ_SWEEP_ARTIFACT_TK=64 FQ_SWEEP_BCHUNK=0 \
    FQ_SWEEP_PACKED_FORMAT=0 ./build.sh) >"$build_log" 2>&1
  local build_rc=$?
  if [ "$build_rc" -ne 0 ]; then
    printf '[fq-packed-owner] FAIL: candidate build rc=%d\n' "$build_rc" >&2
    tail -160 "$build_log" >&2
    printf '[fq-packed-owner] artifacts=%s\n' "$out" >&2
    return "$build_rc"
  fi
  grep -Fq \
    "PPU_DEFS verified on test_fully_quantized_internal_sweep's compile command: -DPPU_PACKED_METADATA_OWNER_ONLY=1" \
    "$build_log" || {
      printf '[fq-packed-owner] FAIL: candidate define missing from target compile command\n' >&2
      return 2
    }
  binary="$build_dir/ppu_targets/test_fully_quantized_internal_sweep"
  if [ ! -x "$binary" ] || [ -L "$binary" ]; then
    binary="$(grep -m1 '^built: ' "$build_log" | cut -d' ' -f2-)"
  fi
  if [ -z "$binary" ] || [ ! -x "$binary" ] || [ -L "$binary" ]; then
    printf '[fq-packed-owner] FAIL: candidate binary missing: %s\n' "$binary" >&2
    return 2
  fi

  direct_log="$out/results/candidate-direct.log"
  probe_log="$out/results/candidate-workspace-probe.log"
  "$binary" --shape=1x1024x5120 --iterations=1 \
    --correctness-repeats="${DIRECT_REPEATS:-128}" \
    --tm8-max-m=8 --bc-mode=skip >"$direct_log" 2>&1
  direct_rc=$?
  "$binary" --shape=1x1024x5120 --iterations=1 \
    --correctness-repeats="${PROBE_REPEATS:-128}" \
    --tm8-max-m=8 --bc-mode=skip --split-workspace-probe \
    >"$probe_log" 2>&1
  probe_rc=$?
  if { [ "$direct_rc" -ne 0 ] && [ "$direct_rc" -ne 1 ]; } || \
      [ "$probe_rc" -ne 0 ]; then
    printf '[fq-packed-owner] FAIL: candidate arm rc direct=%d probe=%d\n' \
      "$direct_rc" "$probe_rc" >&2
    tail -80 "$direct_log" >&2
    tail -100 "$probe_log" >&2
    printf '[fq-packed-owner] artifacts=%s\n' "$out" >&2
    return 2
  fi
  python3 -B "$root/tools/check_fq_packed_owner_candidate.py" \
    --legacy-direct "$legacy/results/direct.log" \
    --legacy-probe "$legacy/results/workspace-probe.log" \
    --candidate-direct "$direct_log" \
    --candidate-probe "$probe_log" | tee "$out/results/verdict.log" || return 2
  sha256sum "$binary" "$generated/manifest.json" "$direct_log" \
    "$probe_log" "$out/results/verdict.log" \
    >"$out/results/candidate-authority.sha256" || return 2
  printf '%s\n' "$legacy" >"$out/results/legacy-artifact.path" || return 2
  sha256sum "$legacy/results/authority.sha256" \
    >"$out/results/legacy-authority-file.sha256" || return 2
  printf '[fq-packed-owner] DIAGNOSTIC_COMPLETE sha=%s candidate_direct_rc=%d artifacts=%s\n' \
    "$sha" "$direct_rc" "$out"
}

main "$@"
