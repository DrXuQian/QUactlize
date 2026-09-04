#!/usr/bin/env bash
# Cross-format correctness gate for the fragment-capped TM8/WN16 PPU
# epilogue.  Each qtype contributes exactly one canonical K-pack type to each
# of the FullyQuantized/ScaleFirst dense/grouped routes.  This is deliberately
# not a tactic sweep and its single timing sample is only the existing harness
# completion contract; no performance conclusion is drawn from it.
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SOURCE_SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
SOURCE_SHORT="${SOURCE_SHA:0:8}"
if [ -n "${OUT:-}" ]; then
  RUN_DIR="$(realpath -m -- "$OUT")"
elif [ -d /workspace ] && [ -w /workspace ]; then
  RUN_DIR="/workspace/quactlize-m8n16-cross-format-${SOURCE_SHORT}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
else
  RUN_DIR="${TMPDIR:-/tmp}/quactlize-m8n16-cross-format-${SOURCE_SHORT}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi

fail() {
  printf '[m8n16-cross-format] FAIL phase=%s artifacts=%s\n' "$1" "$RUN_DIR" >&2
  return 1
}

format_axes() {
  case "$1" in
    10) FORMAT_NAME=Q2_K; PACKED_FORMAT=2; WEIGHT_LAYOUT=2; TILE_K=128; DENSE_BEGIN=60 ;;
    11) FORMAT_NAME=Q3_K; PACKED_FORMAT=3; WEIGHT_LAYOUT=2; TILE_K=256; DENSE_BEGIN=30 ;;
    12) FORMAT_NAME=Q4_K; PACKED_FORMAT=0; WEIGHT_LAYOUT=1; TILE_K=64;  DENSE_BEGIN=60 ;;
    13) FORMAT_NAME=Q5_K; PACKED_FORMAT=1; WEIGHT_LAYOUT=2; TILE_K=256; DENSE_BEGIN=30 ;;
    14) FORMAT_NAME=Q6_K; PACKED_FORMAT=4; WEIGHT_LAYOUT=2; TILE_K=128; DENSE_BEGIN=30 ;;
    *) return 1 ;;
  esac
}

generate_format() {
  local q=$1
  format_axes "$q" || return 1
  local base="$RUN_DIR/generated/q$q"
  local sf_symbol="sf_q${q}_a0_tm8_tn64_tk${TILE_K}_wm8_wn16_s2_bc0_ap0_dn16"
  local sfg_symbol="sfg_q${q}_tm8_tn64_tk${TILE_K}_wm8_wn16_s2_ap0_dn16"
  mkdir -p "$base" || return 1

  python3 -B "$ROOT/tools/gen_fully_quantized_kpack_discovery_units.py" \
    --qtype "$q" --parent-begin "$DENSE_BEGIN" --parent-count 1 \
    --per-unit 1 --out-dir "$base/fq-dense" \
    >"$RUN_DIR/results/q$q-fq-dense.generate.log" 2>&1 || return 1
  python3 -B "$ROOT/tools/gen_fully_quantized_grouped_kpack_units.py" \
    --qtype "$q" --parent-begin 60 --parent-count 2 \
    --per-unit 2 --out-dir "$base/fq-grouped" \
    >"$RUN_DIR/results/q$q-fq-grouped.generate.log" 2>&1 || return 1
  python3 -B "$ROOT/tools/gen_scalefirst_internal_units.py" \
    --qtype "$q" --artifact-tk 0 --bchunk 0 \
    --weight-layout "$WEIGHT_LAYOUT" --select-symbol "$sf_symbol" \
    --per-unit 1 --out-dir "$base/sf-dense" \
    >"$RUN_DIR/results/q$q-sf-dense.generate.log" 2>&1 || return 1
  python3 -B "$ROOT/tools/gen_scalefirst_grouped_kpack_units.py" \
    --qtype "$q" --select-symbol "$sfg_symbol" \
    --per-unit 1 --out-dir "$base/sf-grouped" \
    >"$RUN_DIR/results/q$q-sf-grouped.generate.log" 2>&1 || return 1
  printf '[m8n16-cross-format] generated q=%s format=%s layout=%s tk=%s\n' \
    "$q" "$FORMAT_NAME" "$WEIGHT_LAYOUT" "$TILE_K"
}

build_target() {
  local q=$1 family=$2 resume=$3 target=$4 label=$5
  format_axes "$q" || return 1
  local generated="$RUN_DIR/generated/q$q"
  local build="$RUN_DIR/build/q$q/$family"
  local log="$RUN_DIR/results/q$q-$label.build.log"
  local -a family_env=()
  case "$family" in
    fq)
      family_env=(
        "FQ_SWEEP_GENERATED_DIR=$generated/fq-dense"
        "FQ_SWEEP_QTYPE=$q" FQ_SWEEP_ARTIFACT_TK=0 FQ_SWEEP_BCHUNK=0
        "FQ_SWEEP_PACKED_FORMAT=$PACKED_FORMAT"
        "FQ_SWEEP_WEIGHT_LAYOUT=$WEIGHT_LAYOUT"
        "FQ_GROUPED_KPACK_GENERATED_DIR=$generated/fq-grouped"
        "FQ_GROUPED_KPACK_QTYPE=$q"
        "FQ_GROUPED_KPACK_WEIGHT_LAYOUT=$WEIGHT_LAYOUT"
        "FQ_GROUPED_KPACK_PACKED_FORMAT=$PACKED_FORMAT")
      ;;
    sf)
      family_env=(
        "SCALEFIRST_SWEEP_GENERATED_DIR=$generated/sf-dense"
        "SCALEFIRST_SWEEP_QTYPE=$q" SCALEFIRST_SWEEP_ARTIFACT_TK=0
        SCALEFIRST_SWEEP_BCHUNK=0
        "SCALEFIRST_SWEEP_WEIGHT_LAYOUT=$WEIGHT_LAYOUT"
        "SCALEFIRST_GROUPED_KPACK_GENERATED_DIR=$generated/sf-grouped"
        "SCALEFIRST_GROUPED_KPACK_QTYPE=$q"
        "SCALEFIRST_GROUPED_KPACK_WEIGHT_LAYOUT=$WEIGHT_LAYOUT")
      ;;
    *) return 1 ;;
  esac

  # FullyQuantized and ScaleFirst dense generators intentionally emit the
  # same ppu_dense_layout source basename with different compile definitions.
  # PPUToolchain keys custom outputs by basename rather than flags, so those
  # two families must never share one CMake tree.  A family's grouped target
  # has disjoint object names and safely shares its dense configuration.
  env -u CC -u CXX -u CMAKE_GENERATOR -u CMAKE_TOOLCHAIN_FILE \
    -u FQ_SWEEP_GENERATED_DIR -u FQ_GROUPED_KPACK_GENERATED_DIR \
    -u FQ_A02_Q3_GENERATED_DIR -u FQ_KQUANT_PERF_QTYPE \
    -u SCALEFIRST_SWEEP_GENERATED_DIR \
    -u SCALEFIRST_GROUPED_KPACK_GENERATED_DIR \
    PPU_BUILD_DIR="$build" PPU_BUILD_RESUME="$resume" \
    PPU_PRESERVE_STALE_BUILD_TREES=1 \
    PPU_ARCHS=ppu0010 PPU_SDK="$PPU_SDK_ROOT" JOBS="$JOBS_PER_BUILD" \
    QUACTLIZE_KPACK_GLOBAL_PREFLIGHT_RECEIPT="$PREFLIGHT" \
    "${family_env[@]}" TARGET="$target" "$ROOT/build.sh" >"$log" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    printf '[m8n16-cross-format] build failed q=%s family=%s target=%s rc=%s log=%s\n' \
      "$q" "$family" "$target" "$rc" "$log" >&2
    tail -120 "$log" >&2
  fi
  return "$rc"
}

build_family() {
  local q=$1 family=$2
  printf '[m8n16-cross-format] build q=%s family=%s targets=2\n' "$q" "$family"
  case "$family" in
    fq)
      build_target "$q" fq 0 test_fully_quantized_internal_sweep fq-dense || return 1
      build_target "$q" fq 1 test_fully_quantized_grouped_kpack_discovery fq-grouped || return 1
      ;;
    sf)
      build_target "$q" sf 0 test_scalefirst_internal_sweep sf-dense || return 1
      build_target "$q" sf 1 test_scalefirst_grouped_kpack_discovery sf-grouped || return 1
      ;;
    *) return 1 ;;
  esac
  printf '[m8n16-cross-format] build complete q=%s family=%s\n' "$q" "$family"
}

run_route() {
  local q=$1 route=$2
  shift 2
  local log="$RUN_DIR/results/q$q-$route.run.log"
  local rc=0
  "$@" >"$log" 2>&1 || rc=$?
  printf '%d\n' "$rc" >"$RUN_DIR/results/q$q-$route.rc"
  printf '[m8n16-cross-format] run q=%s route=%s rc=%d\n' "$q" "$route" "$rc"
  grep -E '^(FQ_(SHARD|TC_CELL|SHAPE_DONE|GROUPED_KPACK_(SHARD|STRUCTURAL|CELL|COMPLETE))|SF_(SHARD|CELL|COMPLETE|GROUPED_(SHARD|SPLITK|CELL|COMPLETE))) ' \
    "$log" || true
}

run_format() {
  local q=$1
  local build="$RUN_DIR/build/q$q"
  local seed
  seed="$(printf '0x%016x' "$((0x243f6a88 + q * 0x10001))")"
  run_route "$q" fq-dense \
    "$build/fq/ppu_targets/test_fully_quantized_internal_sweep" \
    --shape=7x64x512 --shape=9x64x512 \
    --iterations=1 --correctness-repeats=7 --only-split=1 \
    --tm8-max-m=9 --bc-mode=skip --schedule-seed="$seed"
  run_route "$q" fq-grouped \
    "$build/fq/ppu_targets/test_fully_quantized_grouped_kpack_discovery" \
    --rows-file="$ROWS9" --experts=2 --n=64 --k=512 \
    --iterations=1 --warmups=1 --correctness-repeats=7 \
    --schedule-seed="$seed" --workload-key="q${q}-fq-grouped-m9" \
    --router-profile=e0-9
  run_route "$q" sf-dense \
    "$build/sf/ppu_targets/test_scalefirst_internal_sweep" \
    --shape=7x64x512 --algorithm=full-output --fixture=exact \
    --iterations=1 --correctness-repeats=7 --schedule-seed="$seed"
  run_route "$q" sf-grouped \
    "$build/sf/ppu_targets/test_scalefirst_grouped_kpack_discovery" \
    --rows-file="$ROWS9" --experts=2 --n=64 --k=512 \
    --iterations=1 --warmups=1 --correctness-repeats=7 \
    --schedule-seed="$seed" --workload-key="q${q}-sf-grouped-m9" \
    --router-profile=e0-9
}

main() {
  local default_sdk=${PPU_SDK:-${PPU_HOME:-}}
  [ -n "$SOURCE_SHA" ] || { fail source_head; return 1; }
  [ -n "$default_sdk" ] || { fail missing_ppu_sdk; return 1; }
  PPU_SDK_ROOT="$(realpath -e -- "$default_sdk" 2>/dev/null)" || {
    fail invalid_ppu_sdk; return 1;
  }
  [ -x "$PPU_SDK_ROOT/bin/hgcc" ] || { fail sdk_lacks_hgcc; return 1; }
  case "${CUDA_VISIBLE_DEVICES:-}" in
    ''|*,*|*[!0-9]*) fail one_numeric_CUDA_VISIBLE_DEVICES_required; return 1 ;;
  esac
  case "${JOBS:-16}" in
    ''|*[!0-9]*|0) fail invalid_JOBS; return 1 ;;
  esac
  JOBS_PER_BUILD=${JOBS:-16}
  if [ "$JOBS_PER_BUILD" -gt 18 ]; then
    fail JOBS_exceeds_18_per_family_build; return 1
  fi
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    fail ambient_ppu_defs; return 1
  fi
  if [ -e "$RUN_DIR" ]; then
    fail output_already_exists; return 1
  fi
  mkdir -p "$RUN_DIR"/{generated,build,inputs,results} || {
    fail create_output; return 1;
  }
  PREFLIGHT="$RUN_DIR/inputs/kpack-global-preflight.json"
  ROWS9="$RUN_DIR/inputs/e0-9.rows"
  printf '9\n0\n' >"$ROWS9"

  bash -n "$ROOT/tools/run_m8n16_cross_format_correctness_box.sh" || {
    fail runner_syntax; return 1;
  }
  python3 -B "$ROOT/ci/check_m8n16_cross_format_correctness.py" || {
    fail static_contract; return 1;
  }
  if ! python3 -B "$ROOT/tools/kpack_global_build_preflight.py" create \
      --root "$ROOT" --output "$PREFLIGHT" \
      >"$RUN_DIR/results/global-preflight.log" 2>&1; then
    tail -120 "$RUN_DIR/results/global-preflight.log" >&2
    fail global_preflight; return 1
  fi

  local q
  for q in 10 11 12 13 14; do
    if ! generate_format "$q"; then
      fail "generate_q$q"; return 1
    fi
  done

  python3 -B "$ROOT/ci/check_m8n16_cross_format_correctness.py" \
    --validate-generated-dir "$RUN_DIR" || {
      fail generated_manifest_set; return 1;
    }

  local -a pids=() labels=()
  for q in 10 11 12 13 14; do
    local family
    for family in fq sf; do
      build_family "$q" "$family" &
      pids+=("$!")
      labels+=("q$q/$family")
    done
  done
  local index rc build_bad=0
  for index in "${!pids[@]}"; do
    rc=0
    wait "${pids[$index]}" || rc=$?
    if [ "$rc" -ne 0 ]; then
      printf '[m8n16-cross-format] build worker failed route=%s rc=%s\n' \
        "${labels[$index]}" "$rc" >&2
      build_bad=1
    fi
  done
  [ "$build_bad" -eq 0 ] || { fail parallel_builds; return 1; }

  for q in 10 11 12 13 14; do
    run_format "$q"
  done

  local check_rc=0
  python3 -B "$ROOT/ci/check_m8n16_cross_format_correctness.py" \
    --validate-run-dir "$RUN_DIR" | tee "$RUN_DIR/results/check.log" || \
    check_rc=${PIPESTATUS[0]}
  [ "$check_rc" -eq 0 ] || { fail device_results; return 1; }

  local -a authority=()
  for q in 10 11 12 13 14; do
    authority+=(
      "$RUN_DIR/generated/q$q/fq-dense/manifest.json"
      "$RUN_DIR/generated/q$q/fq-grouped/manifest.json"
      "$RUN_DIR/generated/q$q/sf-dense/manifest.json"
      "$RUN_DIR/generated/q$q/sf-grouped/manifest.json"
      "$RUN_DIR/build/q$q/fq/ppu_targets/test_fully_quantized_internal_sweep"
      "$RUN_DIR/build/q$q/fq/ppu_targets/test_fully_quantized_grouped_kpack_discovery"
      "$RUN_DIR/build/q$q/sf/ppu_targets/test_scalefirst_internal_sweep"
      "$RUN_DIR/build/q$q/sf/ppu_targets/test_scalefirst_grouped_kpack_discovery")
  done
  sha256sum "${authority[@]}" "$RUN_DIR"/results/*.run.log \
    >"$RUN_DIR/results/authority.sha256" || {
      fail authority_hash; return 1;
    }
  printf 'M8N16_CROSS_FORMAT_CORRECTNESS verdict=PASS formats=5 routes=20 cells=40 measured=40 structural=0 repeats=7 out_of_scope_structural=35 source=%s actlize=%s artifacts=%s\n' \
    "$SOURCE_SHA" "$(git -C "$ROOT/third_party/actlize" rev-parse HEAD)" \
    "$RUN_DIR"
}

main "$@"
