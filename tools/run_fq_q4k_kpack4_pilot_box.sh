#!/usr/bin/env bash
# Conservative one-shape performance pilot for canonical Q4_K K-pack4.
# Build the complete native 72-row TM8 graph once, then prune at runtime:
#   S1 screen -> all-scheduler screen -> seven-sample confirmation.
set -uo pipefail

main() {
  local root workspace_root sha short stamp out jobs per_unit policy generated
  local manifest build_dir build_log target_make binary binary_sha units
  local screen_log scheduler_log confirm_log rc
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace_root="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-q4k-kpack4-pilot-${short}-${stamp}-$$}")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) printf '[fq-q4k-kpack4-pilot] FAIL: OUT must be a strict /workspace child: %s\n' "$out" >&2; return 2 ;;
  esac
  if [ -e "$out" ]; then
    printf '[fq-q4k-kpack4-pilot] FAIL: refusing to overwrite %s\n' "$out" >&2
    return 2
  fi
  jobs="${JOBS:-16}"
  per_unit="${FQ_CONFIGS_PER_UNIT:-4}"
  case "$jobs" in
    *[!0-9]*|0) printf '[fq-q4k-kpack4-pilot] FAIL: JOBS must be positive\n' >&2; return 2 ;;
  esac
  case "$per_unit" in
    *[!0-9]*|0) printf '[fq-q4k-kpack4-pilot] FAIL: FQ_CONFIGS_PER_UNIT must be positive\n' >&2; return 2 ;;
  esac
  policy="$root/benchmarks/fq_q4k_decode_real_shapes_policy.json"
  generated="$out/generated"
  manifest="$generated/manifest.json"
  build_dir="$out/build"
  mkdir -p "$generated" "$build_dir" "$out/results" || return 2

  python3 -B "$root/ci/check_fq_q4k_kpack4_generator.py" || return 2
  python3 -B "$root/tools/analyze_fq_q4k_kpack4_pilot.py" self-test \
    --policy "$policy" || return 2
  python3 -B "$root/tools/check_fq_q4k_kpack4_closure.py" --self-test || return 2

  python3 -B "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
    --qtype 12 --artifact-tk 0 --bchunk 0 --weight-layout q4-kpack4 \
    --tile-m-filter 8 --per-unit "$per_unit" --out-dir "$generated" || return 2
  python3 -B - "$manifest" <<'PY' || return 2
import json,sys
value=json.load(open(sys.argv[1]))
assert value["identity"] == {
 "qtype":12,"format":"Q4_K","artifact_tile_k":0,"bchunk":0,
 "tile_m_filter":8,"weight_layout":"q4-kpack4"}
assert value["denominator"]["typed_rows"] == 72
assert value["denominator"]["source_typed_rows"] == 846
assert value["weight_mapping"]["mapping_id"] == "0x51344b5034540001"
assert value["weight_mapping"]["artifact_tile_k_is_not_an_axis"] is True
PY
  units="$(python3 -B -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["units"]))' "$manifest")" || return 2

  git -C "$root" diff --binary --no-ext-diff HEAD > "$out/source.patch" || return 2
  {
    printf '%s\n' "$sha"
    git -C "$root/third_party/actlize" rev-parse HEAD
    sha256sum \
      "$policy" \
      "$root/benchmarks/test_fully_quantized_internal_sweep.cu" \
      "$root/benchmarks/fully_quantized_splitk_producer_bench.hpp" \
      "$root/benchmarks/fully_quantized_splitk_producer_unit.inc" \
      "$root/quactlize/include/q4_kpack4_offline.hpp" \
      "$root/quactlize/include/fpA_intB_ppu.cuh" \
      "$root/quactlize/include/ppu_mixed_policy.hpp" \
      "$root/quactlize/include/ppu_placed_arrangement.hpp" \
      "$root/quactlize/include/actlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp" \
      "$root/quactlize/csrc/device/ppu_dense_layout.cu" \
      "$root/quactlize/csrc/fq_internal_sweep.cmake.in" \
      "$root/tools/fully_quantized_internal_matrix.py" \
      "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
      "$root/tools/analyze_fq_q4k_decode_real_shapes.py" \
      "$root/tools/analyze_fq_q4k_kpack4_pilot.py" \
      "$root/tools/run_fq_q4k_kpack4_pilot_box.sh" \
      "$root/ci/check_fq_q4k_kpack4_generator.py" \
      "$root/build.sh"
  } > "$out/source-authority.sha256" || return 2

  build_log="$out/results/build.log"
  (cd "$root" && \
    PPU_BUILD_DIR="$build_dir" PPU_ARCHS=ppu0010 JOBS="$jobs" \
    TARGET=test_fully_quantized_internal_sweep \
    FQ_SWEEP_GENERATED_DIR="$generated" FQ_SWEEP_QTYPE=12 \
    FQ_SWEEP_ARTIFACT_TK=0 FQ_SWEEP_BCHUNK=0 \
    FQ_SWEEP_PACKED_FORMAT=0 FQ_SWEEP_WEIGHT_LAYOUT=1 \
    ./build.sh) > "$build_log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    printf '[fq-q4k-kpack4-pilot] FAIL: 72-row target did not build rc=%d\n' "$rc" >&2
    tail -n 180 "$build_log" >&2
    printf '[fq-q4k-kpack4-pilot] artifacts=%s\n' "$out" >&2
    return "$rc"
  fi
  target_make="$(find "$build_dir" -type f \
    -path '*test_fully_quantized_internal_sweep.dir/build.make' \
    -print -quit 2>/dev/null)"
  if ! grep -Fqx '[build.sh] FQ_SWEEP_WEIGHT_LAYOUT=1' "$build_log" ||
     ! grep -F "FullyQuantized internal sweep: q=12 A=0 bc=0 format=0 layout=1 units=$units" \
       "$build_dir/cmake.log" >/dev/null ||
     ! grep -Eq '^FQ_SWEEP_WEIGHT_LAYOUT(:[^=]*)?=1$' \
       "$build_dir/CMakeCache.txt" ||
     [ -z "$target_make" ] ||
     ! grep -Eq -- '(^|[[:space:]])-DFQ_SWEEP_WEIGHT_LAYOUT=1([[:space:]]|$)' \
       "$target_make"; then
    printf '[fq-q4k-kpack4-pilot] FAIL: layout=1 build ABI did not reach the target\n' >&2
    grep -E 'FQ_SWEEP_WEIGHT_LAYOUT|FullyQuantized internal sweep:' \
      "$build_log" "$build_dir/cmake.log" "$build_dir/CMakeCache.txt" \
      "$target_make" 2>/dev/null >&2 || true
    printf '[fq-q4k-kpack4-pilot] artifacts=%s\n' "$out" >&2
    return 2
  fi
  binary="$build_dir/ppu_targets/test_fully_quantized_internal_sweep"
  if [ ! -x "$binary" ] || [ -L "$binary" ]; then
    binary="$(grep -m1 '^built: ' "$build_log" | cut -d' ' -f2-)"
  fi
  if [ -z "$binary" ] || [ ! -x "$binary" ] || [ -L "$binary" ]; then
    printf '[fq-q4k-kpack4-pilot] FAIL: exact binary missing: %s\n' "$binary" >&2
    return 2
  fi
  binary_sha="$(sha256sum "$binary" | awk '{print $1}')" || return 2
  printf '%s  %s\n' "$binary_sha" "$binary" > "$out/results/binary.sha256" || return 2

  screen_log="$out/results/screen.log"
  printf '[fq-q4k-kpack4-pilot] phase=screen shape=1x1024x5120 typed=72 S=1\n'
  "$binary" --shape=1x1024x5120 --iterations=2 \
    --correctness-repeats=1 --only-split=1 --tm8-max-m=8 \
    --bc-mode=all | tee "$screen_log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    printf '[fq-q4k-kpack4-pilot] FAIL: screen rc=%d artifacts=%s\n' "$rc" "$out" >&2
    return "$rc"
  fi
  python3 -B "$root/tools/analyze_fq_q4k_kpack4_pilot.py" screen \
    --manifest "$manifest" --log "$screen_log" --policy "$policy" \
    --symbols-output "$out/results/screen-symbols.txt" \
    --summary-output "$out/results/screen.json" || return 2

  scheduler_log="$out/results/scheduler.log"
  printf '[fq-q4k-kpack4-pilot] phase=scheduler shape=1x1024x5120 splits=1,2,4,8\n'
  "$binary" --shape=1x1024x5120 --iterations=1 \
    --correctness-repeats=1 --only-split=0 --tm8-max-m=8 \
    --symbols-file="$out/results/screen-symbols.txt" \
    --bc-mode=skip | tee "$scheduler_log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    printf '[fq-q4k-kpack4-pilot] FAIL: scheduler rc=%d artifacts=%s\n' "$rc" "$out" >&2
    return "$rc"
  fi
  python3 -B "$root/tools/analyze_fq_q4k_kpack4_pilot.py" scheduler \
    --manifest "$manifest" --log "$scheduler_log" --policy "$policy" \
    --screen-symbols "$out/results/screen-symbols.txt" \
    --symbols-output "$out/results/confirm-symbols.txt" \
    --summary-output "$out/results/scheduler.json" || return 2

  confirm_log="$out/results/confirm.log"
  printf '[fq-q4k-kpack4-pilot] phase=confirm shape=1x1024x5120 iterations=7\n'
  "$binary" --shape=1x1024x5120 --iterations=7 \
    --correctness-repeats=2 --only-split=0 --tm8-max-m=8 \
    --symbols-file="$out/results/confirm-symbols.txt" \
    --bc-mode=skip | tee "$confirm_log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    printf '[fq-q4k-kpack4-pilot] FAIL: confirmation rc=%d artifacts=%s\n' "$rc" "$out" >&2
    return "$rc"
  fi
  python3 -B "$root/tools/analyze_fq_q4k_kpack4_pilot.py" finalize \
    --manifest "$manifest" --log "$confirm_log" --policy "$policy" \
    --symbols "$out/results/confirm-symbols.txt" \
    --output-json "$out/results/summary.json" \
    --output-tsv "$out/results/summary.tsv" || return 2

  sha256sum "$manifest" "$binary" "$screen_log" "$scheduler_log" \
    "$confirm_log" "$out/results/screen-symbols.txt" \
    "$out/results/confirm-symbols.txt" "$out/results/summary.json" \
    > "$out/results/authority.sha256" || return 2
  sed -n '1,8p' "$out/results/summary.tsv"
  printf '[fq-q4k-kpack4-pilot] PASS sha=%s layout=1 mapping=0x51344b5034540001 typed=72 artifacts=%s\n' \
    "$sha" "$out"
}

main "$@"
