#!/usr/bin/env bash
# First device closure for the canonical Q4_K K-pack4-transposed checkpoint.
# It builds and launches exactly one production S=1 tactic.  Split-K and the
# wider tactic denominator are deliberately gated on this raw-bit closure.
set -uo pipefail

main() {
  local root workspace_root sha short stamp out jobs full generated
  local build_dir build_log target_make binary device_log build_rc run_rc
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace_root="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-q4k-kpack4-${short}-${stamp}-$$}")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) printf '[fq-q4k-kpack4] FAIL: OUT must be a strict /workspace child: %s\n' "$out" >&2; return 2 ;;
  esac
  if [ -e "$out" ]; then
    printf '[fq-q4k-kpack4] FAIL: refusing to overwrite %s\n' "$out" >&2
    return 2
  fi
  jobs="${JOBS:-16}"
  case "$jobs" in
    *[!0-9]*|0) printf '[fq-q4k-kpack4] FAIL: JOBS must be positive\n' >&2; return 2 ;;
  esac
  mkdir -p "$out/generated/full" "$out/generated/closure" \
    "$out/build" "$out/results" || return 2

  python3 -B "$root/tools/select_fq_q4k_kpack4_closure.py" --self-test || return 2
  python3 -B "$root/tools/check_fq_q4k_kpack4_closure.py" --self-test || return 2

  # The established A64/TM8 graph is the tactic-space authority.  K-pack4
  # changes the artifact identity only after the exact row is selected.
  python3 -B "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
    --qtype 12 --artifact-tk 64 --bchunk 0 --per-unit 1 \
    --tile-m-filter 8 --out-dir "$out/generated/full" || return 2
  full="$out/generated/full"
  generated="$out/generated/closure"
  python3 -B "$root/tools/select_fq_q4k_kpack4_closure.py" \
    --source-dir "$full" --out-dir "$generated" || return 2

  build_dir="$out/build"
  build_log="$out/results/build.log"
  (cd "$root" && \
    PPU_BUILD_DIR="$build_dir" PPU_ARCHS=ppu0010 JOBS="$jobs" \
    TARGET=test_fully_quantized_internal_sweep \
    FQ_SWEEP_GENERATED_DIR="$generated" FQ_SWEEP_QTYPE=12 \
    FQ_SWEEP_ARTIFACT_TK=0 FQ_SWEEP_BCHUNK=0 \
    FQ_SWEEP_PACKED_FORMAT=0 FQ_SWEEP_WEIGHT_LAYOUT=1 \
    ./build.sh) >"$build_log" 2>&1
  build_rc=$?
  if [ "$build_rc" -ne 0 ]; then
    printf '[fq-q4k-kpack4] FAIL: exact one-row target did not build rc=%d\n' "$build_rc" >&2
    tail -n 180 "$build_log" >&2
    printf '[fq-q4k-kpack4] artifacts=%s\n' "$out" >&2
    return "$build_rc"
  fi
  target_make="$(find "$build_dir" -type f \
    -path '*test_fully_quantized_internal_sweep.dir/build.make' \
    -print -quit 2>/dev/null)"
  if ! grep -Fqx '[build.sh] FQ_SWEEP_WEIGHT_LAYOUT=1' "$build_log" ||
     ! grep -F 'FullyQuantized internal sweep: q=12 A=0 bc=0 format=0 layout=1 units=1' \
       "$build_dir/cmake.log" >/dev/null ||
     ! grep -Eq '^FQ_SWEEP_WEIGHT_LAYOUT(:[^=]*)?=1$' \
       "$build_dir/CMakeCache.txt" ||
     [ -z "$target_make" ] ||
     ! grep -Eq -- '(^|[[:space:]])-DFQ_SWEEP_WEIGHT_LAYOUT=1([[:space:]]|$)' \
       "$target_make"; then
    printf '[fq-q4k-kpack4] FAIL: weight-layout build ABI did not reach build.sh/CMake/target\n' >&2
    grep -E 'FQ_SWEEP_WEIGHT_LAYOUT|FullyQuantized internal sweep:' \
      "$build_log" "$build_dir/cmake.log" "$build_dir/CMakeCache.txt" \
      "$target_make" 2>/dev/null >&2 || true
    printf '[fq-q4k-kpack4] artifacts=%s\n' "$out" >&2
    return 2
  fi
  binary="$build_dir/ppu_targets/test_fully_quantized_internal_sweep"
  if [ ! -x "$binary" ] || [ -L "$binary" ]; then
    binary="$(grep -m1 '^built: ' "$build_log" | cut -d' ' -f2-)"
  fi
  if [ -z "$binary" ] || [ ! -x "$binary" ] || [ -L "$binary" ]; then
    printf '[fq-q4k-kpack4] FAIL: exact binary missing: %s\n' "$binary" >&2
    return 2
  fi

  device_log="$out/results/device-s1.log"
  "$binary" --shape=1x1024x5120 \
    --iterations="${ITERATIONS:-3}" \
    --correctness-repeats="${CORRECTNESS_REPEATS:-64}" \
    --only-split=1 --bc-mode=skip | tee "$device_log"
  run_rc=${PIPESTATUS[0]}
  if [ "$run_rc" -ne 0 ]; then
    printf '[fq-q4k-kpack4] FAIL: S1 device closure rc=%d artifacts=%s\n' \
      "$run_rc" "$out" >&2
    return "$run_rc"
  fi
  python3 -B "$root/tools/check_fq_q4k_kpack4_closure.py" \
    --log "$device_log" || return 2
  sha256sum "$binary" "$generated/manifest.json" \
    "$generated/fq_tc_registry.inc" "$device_log" \
    >"$out/results/authority.sha256" || return 2
  printf '[fq-q4k-kpack4] PASS sha=%s layout=q4-kpack4-transpose-v1 mapping=0x51344b5034540001 tactics=1 cells=1 S1=RAW-BIT artifacts=%s\n' \
    "$sha" "$out"
}

main "$@"
