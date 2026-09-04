#!/usr/bin/env bash
# Minimal dense S1 closure for the TM8 epilogue ownership fix.  Two fresh,
# one-parent canonical Q4 K-pack binaries exercise the exact TM8 shipping row
# and its TM16 control over the M=8 boundary.  No Split-K or broad sweep runs.
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TARGET=test_fully_quantized_internal_sweep
SOURCE_SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
SOURCE_SHORT="${SOURCE_SHA:0:8}"
if [ -n "${OUT:-}" ]; then
  RUN_DIR="$(realpath -m -- "$OUT")"
elif [ -d /workspace ] && [ -w /workspace ]; then
  RUN_DIR="/workspace/quactlize-m8-dense-epilogue-${SOURCE_SHORT}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
else
  RUN_DIR="${TMPDIR:-/tmp}/quactlize-m8-dense-epilogue-${SOURCE_SHORT}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi

fail() {
  printf '[m8-dense-epilogue-postfix] FAIL phase=%s artifacts=%s\n' "$1" "$RUN_DIR" >&2
  return 1
}

build_shard() {
  local family=$1 generated=$2 build=$3 log=$4
  printf '[m8-dense-epilogue-postfix] build family=%s generated=%s\n' \
    "$family" "$generated"
  if ! env PPU_BUILD_DIR="$build" PPU_BUILD_RESUME=0 \
      PPU_ARCHS=ppu0010 PPU_SDK="$PPU_SDK_ROOT" JOBS="$JOBS_COUNT" \
      QUACTLIZE_KPACK_GLOBAL_PREFLIGHT_RECEIPT="$PREFLIGHT" \
      TARGET="$TARGET" FQ_SWEEP_GENERATED_DIR="$generated" \
      FQ_SWEEP_QTYPE=12 FQ_SWEEP_ARTIFACT_TK=0 FQ_SWEEP_BCHUNK=0 \
      FQ_SWEEP_PACKED_FORMAT=0 FQ_SWEEP_WEIGHT_LAYOUT=1 \
      "$ROOT/build.sh" >"$log" 2>&1; then
    tail -120 "$log" >&2
    return 1
  fi
}

run_family() {
  local family=$1 binary=$2
  local log="$RUN_DIR/results/$family.run.log" rc=0
  "$binary" \
    --shape=1x64x512 --shape=8x64x512 --shape=9x64x512 \
    --shape=15x64x512 --shape=16x64x512 --shape=17x64x512 \
    --iterations=1 --correctness-repeats=7 --only-split=1 \
    --tm8-max-m=17 --bc-mode=skip >"$log" 2>&1 || rc=$?
  printf '%d\n' "$rc" >"$RUN_DIR/results/$family.rc"
  printf '[m8-dense-epilogue-postfix] family=%s rc=%d\n' "$family" "$rc"
  grep -E '^FQ_(SHARD|TC_CELL|SHAPE_DONE) ' "$log" || true
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
  JOBS_COUNT=${JOBS:-16}
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

  python3 -B "$ROOT/ci/check_m8_dense_epilogue_postfix.py" || {
    fail static_contract; return 1;
  }
  if ! python3 -B "$ROOT/tools/kpack_global_build_preflight.py" create \
      --root "$ROOT" --output "$PREFLIGHT" \
      >"$RUN_DIR/results/global-preflight.log" 2>&1; then
    tail -120 "$RUN_DIR/results/global-preflight.log" >&2
    fail global_preflight; return 1
  fi

  if ! python3 -B "$ROOT/tools/gen_fully_quantized_kpack_discovery_units.py" \
      --qtype 12 --out-dir "$RUN_DIR/generated/tm8" --per-unit 1 \
      --parent-begin 4827 --parent-count 1 \
      >"$RUN_DIR/results/generate-tm8.log" 2>&1; then
    cat "$RUN_DIR/results/generate-tm8.log" >&2
    fail generate_tm8; return 1
  fi
  if ! python3 -B "$ROOT/tools/gen_fully_quantized_kpack_discovery_units.py" \
      --qtype 12 --out-dir "$RUN_DIR/generated/tm16" --per-unit 1 \
      --parent-begin 5157 --parent-count 1 \
      >"$RUN_DIR/results/generate-tm16.log" 2>&1; then
    cat "$RUN_DIR/results/generate-tm16.log" >&2
    fail generate_tm16; return 1
  fi

  local tm8_pid tm16_pid tm8_build_rc=0 tm16_build_rc=0
  build_shard tm8 "$RUN_DIR/generated/tm8" "$RUN_DIR/build/tm8" \
    "$RUN_DIR/results/build-tm8.log" &
  tm8_pid=$!
  build_shard tm16 "$RUN_DIR/generated/tm16" "$RUN_DIR/build/tm16" \
    "$RUN_DIR/results/build-tm16.log" &
  tm16_pid=$!
  wait "$tm8_pid" || tm8_build_rc=$?
  wait "$tm16_pid" || tm16_build_rc=$?
  [ "$tm8_build_rc" -eq 0 ] || { fail build_tm8; return 1; }
  [ "$tm16_build_rc" -eq 0 ] || { fail build_tm16; return 1; }

  TM8_BIN="$RUN_DIR/build/tm8/ppu_targets/$TARGET"
  TM16_BIN="$RUN_DIR/build/tm16/ppu_targets/$TARGET"
  [ -x "$TM8_BIN" ] || { fail missing_tm8_binary; return 1; }
  [ -x "$TM16_BIN" ] || { fail missing_tm16_binary; return 1; }
  run_family tm8 "$TM8_BIN"
  run_family tm16 "$TM16_BIN"

  if ! python3 -B "$ROOT/ci/check_m8_dense_epilogue_postfix.py" \
      --validate-run-dir "$RUN_DIR"; then
    fail device_results; return 1
  fi
  printf 'FQ_M8_DENSE_EPILOGUE_POSTFIX verdict=PASS families=2 shapes=6 cells=12 repeats=7 split=S1 source=%s actlize=%s artifacts=%s\n' \
    "$SOURCE_SHA" "$(git -C "$ROOT/third_party/actlize" rev-parse HEAD)" "$RUN_DIR"
}

main "$@"
