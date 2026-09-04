#!/usr/bin/env bash
# Full-production closure for the TM8 epilogue ownership fix.  Two exact
# generated Q4 K-pack registries exercise only the former failing TM8 row and
# its TM16 control through the shipping grouped kernels.
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TARGET=test_fully_quantized_grouped_kpack_discovery
SOURCE_SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
SOURCE_SHORT="${SOURCE_SHA:0:8}"
if [ -n "${OUT:-}" ]; then
  RUN_DIR="$(realpath -m -- "$OUT")"
elif [ -d /workspace ] && [ -w /workspace ]; then
  RUN_DIR="/workspace/quactlize-m8-grouped-epilogue-${SOURCE_SHORT}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
else
  RUN_DIR="${TMPDIR:-/tmp}/quactlize-m8-grouped-epilogue-${SOURCE_SHORT}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi

fail() {
  printf '[m8-grouped-epilogue-postfix] FAIL phase=%s artifacts=%s\n' "$1" "$RUN_DIR" >&2
  return 1
}

build_shard() {
  local family=$1 generated=$2 build=$3 log=$4
  printf '[m8-grouped-epilogue-postfix] build family=%s generated=%s\n' \
    "$family" "$generated"
  if ! env PPU_BUILD_DIR="$build" PPU_BUILD_RESUME=0 \
      PPU_ARCHS=ppu0010 PPU_SDK="$PPU_SDK_ROOT" JOBS="$JOBS_COUNT" \
      QUACTLIZE_KPACK_GLOBAL_PREFLIGHT_RECEIPT="$PREFLIGHT" \
      TARGET="$TARGET" \
      FQ_GROUPED_KPACK_GENERATED_DIR="$generated" \
      FQ_GROUPED_KPACK_QTYPE=12 FQ_GROUPED_KPACK_WEIGHT_LAYOUT=1 \
      FQ_GROUPED_KPACK_PACKED_FORMAT=0 \
      "$ROOT/build.sh" >"$log" 2>&1; then
    tail -120 "$log" >&2
    return 1
  fi
}

run_arm() {
  local label=$1 family=$2 binary=$3 rows_file=$4 experts=$5 n=$6
  local profile=$7 symbol=$8 expected_algorithm=$9
  local log="$RUN_DIR/results/arms/$label.run.log"
  local rc_file="$RUN_DIR/results/arms/$label.rc"
  local rc=0
  "$binary" --rows-file="$rows_file" --experts="$experts" --n="$n" --k=512 \
    --symbol="$symbol" --workload-key="$label" --router-profile="$profile" \
    --correctness-repeats=7 --iterations=1 --warmups=1 >"$log" 2>&1 || rc=$?
  printf '%d\n' "$rc" >"$rc_file"
  printf '[m8-grouped-epilogue-postfix] arm=%s family=%s expected=%s rc=%d\n' \
    "$label" "$family" "$expected_algorithm" "$rc"
  grep -E '^FQ_GROUPED_KPACK_(SHARD|CELL|MISMATCH_MAP|COMPLETE) ' "$log" || true
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
  mkdir -p "$RUN_DIR"/{generated,build,inputs,results/arms} || {
    fail create_output; return 1;
  }
  PREFLIGHT="$RUN_DIR/inputs/kpack-global-preflight.json"

  python3 -B "$ROOT/ci/check_m8_grouped_epilogue_postfix.py" || {
    fail static_contract; return 1;
  }
  if ! python3 -B "$ROOT/tools/kpack_global_build_preflight.py" create \
      --root "$ROOT" --output "$PREFLIGHT" \
      >"$RUN_DIR/results/global-preflight.log" 2>&1; then
    tail -120 "$RUN_DIR/results/global-preflight.log" >&2
    fail global_preflight; return 1
  fi

  if ! python3 -B "$ROOT/tools/gen_fully_quantized_grouped_kpack_units.py" \
      --qtype 12 --out-dir "$RUN_DIR/generated/tm8" \
      --parent-begin 60 --parent-count 2 --per-unit 2 \
      >"$RUN_DIR/results/generate-tm8.log" 2>&1; then
    cat "$RUN_DIR/results/generate-tm8.log" >&2
    fail generate_tm8; return 1
  fi
  if ! python3 -B "$ROOT/tools/gen_fully_quantized_grouped_kpack_units.py" \
      --qtype 12 --out-dir "$RUN_DIR/generated/tm16" \
      --parent-begin 516 --parent-count 2 --per-unit 2 \
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

  E0_ROWS="$RUN_DIR/inputs/e0-9.rows"
  BOUNDARY_ROWS="$RUN_DIR/inputs/tilem-boundary.rows"
  printf '9\n0\n' >"$E0_ROWS"
  local expert value
  for ((expert = 0; expert < 256; ++expert)); do
    case "$expert" in
      0) value=15 ;; 1) value=16 ;; 2) value=17 ;;
      3) value=31 ;; 4) value=32 ;; 5) value=33 ;;
      6) value=127 ;; 7) value=128 ;; 8) value=129 ;;
      *) value=0 ;;
    esac
    printf '%d\n' "$value"
  done >"$BOUNDARY_ROWS"

  run_arm tm8-p-e0-9-n64 tm8 "$TM8_BIN" "$E0_ROWS" 2 64 e0-9 \
    fqg_q12_l1_tm8_tn64_tk64_wm8_wn16_s2_ap0_dn16_persistent GROUPED_PERSISTENT
  run_arm tm8-p-e0-9-n3072 tm8 "$TM8_BIN" "$E0_ROWS" 2 3072 e0-9 \
    fqg_q12_l1_tm8_tn64_tk64_wm8_wn16_s2_ap0_dn16_persistent GROUPED_PERSISTENT
  run_arm tm8-np-tilem-boundary-n3072 tm8 "$TM8_BIN" "$BOUNDARY_ROWS" 256 3072 tilem-boundary \
    fqg_q12_l1_tm8_tn64_tk64_wm8_wn16_s2_ap0_dn16_nonpersistent GROUPED_NONPERSISTENT
  run_arm tm16-p-e0-9-n64 tm16 "$TM16_BIN" "$E0_ROWS" 2 64 e0-9 \
    fqg_q12_l1_tm16_tn64_tk64_wm16_wn16_s2_ap0_dn16_persistent GROUPED_PERSISTENT
  run_arm tm16-p-e0-9-n3072 tm16 "$TM16_BIN" "$E0_ROWS" 2 3072 e0-9 \
    fqg_q12_l1_tm16_tn64_tk64_wm16_wn16_s2_ap0_dn16_persistent GROUPED_PERSISTENT
  run_arm tm16-np-tilem-boundary-n3072 tm16 "$TM16_BIN" "$BOUNDARY_ROWS" 256 3072 tilem-boundary \
    fqg_q12_l1_tm16_tn64_tk64_wm16_wn16_s2_ap0_dn16_nonpersistent GROUPED_NONPERSISTENT

  if ! python3 -B "$ROOT/ci/check_m8_grouped_epilogue_postfix.py" \
      --validate-run-dir "$RUN_DIR"; then
    fail device_results; return 1
  fi
  printf 'FQ_M8_GROUPED_EPILOGUE_POSTFIX verdict=PASS arms=6 repeats=7 source=%s actlize=%s artifacts=%s\n' \
    "$SOURCE_SHA" "$(git -C "$ROOT/third_party/actlize" rev-parse HEAD)" "$RUN_DIR"
}

main "$@"
