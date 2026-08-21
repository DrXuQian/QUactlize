#!/usr/bin/env bash
# Build the shipping PPU chunked-GDN library, link its standalone host
# benchmark, and measure a preregistered shape set with aggregate device
# events. The measured span includes launch idle introduced by the public ABI's
# per-call runtime/attribute checks, and is therefore an upper bound rather
# than a claim of naked kernel time. Artifacts are always kept under /workspace; no temporary directory
# or copied source authority is involved.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:---box}"
SHA="$(git -C "$ROOT" rev-parse HEAD)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-/workspace/quactlize-ppu-chunked-gdn-perf-${SHA:0:8}-${STAMP}}"
JOBS="${JOBS:-16}"
WARMUP="${WARMUP:-5}"
ITERATIONS="${ITERATIONS:-20}"
SAMPLES="${SAMPLES:-7}"
if [[ -e "$OUT" ]]; then
  if [[ ! -d "$OUT" || -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[GDN perf] FAIL: OUT must be absent or an empty directory: $OUT" >&2
    exit 2
  fi
fi
mkdir -p "$OUT"

case "$MODE" in
  --local|--box) ;;
  *)
    echo "usage: $0 [--local|--box]" >&2
    exit 2
    ;;
esac

valid_label() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] &&
    [[ "$1" != "library-build" && "$1" != "performance-summary" ]]
}

valid_shape_fields() {
  local sequences="$1" length="$2" qk_heads="$3" v_heads="$4"
  [[ "$sequences" =~ ^[1-9][0-9]*$ && "$length" =~ ^[1-9][0-9]*$ &&
     "$qk_heads" =~ ^[1-9][0-9]*$ && "$v_heads" =~ ^[1-9][0-9]*$ ]] &&
    ((v_heads % qk_heads == 0))
}

declare -a CURRENT_LOGS=()
declare -A SEEN_LABELS=()
declare -a shape_rows=()

# Validate the complete selection before either the local contract exit or an
# expensive PPU build. Thus malformed, empty and duplicate selections are
# locally falsifiable properties, not box-only surprises.
if [[ "${ACU:-0}" == 1 ]]; then
  ACU_SHAPE="${GDN_ACU_SHAPE:-2,2048,32,32,cula-like-long-grid64}"
  IFS=',' read -r sequences length qk_heads v_heads label extra <<< "$ACU_SHAPE"
  if [[ -z "${label:-}" || -n "${extra:-}" ]] || ! valid_label "$label" ||
      ! valid_shape_fields "$sequences" "$length" "$qk_heads" "$v_heads"; then
    echo "[GDN perf ACU] FAIL: GDN_ACU_SHAPE must be B,T,H,HV,label" >&2
    exit 2
  fi
else
  DEFAULT_SHAPES="1,256,32,32,short-grid32;2,256,16,32,gva-grid64;3,256,12,24,exact-fill-grid72;4,256,16,32,gva-grid128;2,2048,32,32,cula-like-long-grid64"
  if [[ -v GDN_SHAPES ]]; then
    SHAPES="$GDN_SHAPES"
  else
    SHAPES="$DEFAULT_SHAPES"
  fi
  IFS=';' read -r -a shape_rows <<< "$SHAPES"
  if [[ "${#shape_rows[@]}" -eq 0 ]]; then
    echo "[GDN perf] FAIL: GDN_SHAPES selected no rows" >&2
    exit 2
  fi
  for row in "${shape_rows[@]}"; do
    IFS=',' read -r sequences length qk_heads v_heads label extra <<< "$row"
    if [[ -z "${label:-}" || -n "${extra:-}" ]] || ! valid_label "$label" ||
        ! valid_shape_fields "$sequences" "$length" "$qk_heads" "$v_heads"; then
      echo "[GDN perf] FAIL: shape row must be B,T,H,HV,label; got '$row'" >&2
      exit 2
    fi
    if [[ -n "${SEEN_LABELS[$label]:-}" ]]; then
      echo "[GDN perf] FAIL: duplicate shape label '$label'" >&2
      exit 2
    fi
    SEEN_LABELS["$label"]=1
  done
fi

# Local syntax is intentionally a host-object check, not a fake timing path.
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -I"$ROOT/quactlize/include" \
  -isystem "$ROOT/dev/fold_derivation/stub_inc" \
  -c "$ROOT/benchmarks/test_ppu_chunked_gdn_perf.cpp" \
  -o "$OUT/test_ppu_chunked_gdn_perf.host-contract.o"
echo "[GDN perf local] PASS: standalone public-ABI benchmark compiles"

if [[ "$MODE" == "--local" ]]; then
  echo "[GDN perf device] SKIP: --local selected; timing requires a PPU box"
  echo "[GDN perf] artifacts=$OUT"
  exit 0
fi

PPU_SDK_ROOT="${PPU_SDK:-${PPU_HOME:-${PPU_SDK_SITE_DEFAULT:-/usr/local/PPU_SDK}}}"
if [[ ! -x "$PPU_SDK_ROOT/bin/hgcc" ]]; then
  echo "[GDN perf] FAIL: hgcc unavailable at $PPU_SDK_ROOT/bin/hgcc" >&2
  exit 1
fi

# Follow L205's proven source/build boundary exactly: build the shipping
# library target, then compile one host-only ABI consumer with g++.  Registering
# this .cpp as a device target would add a second source-graph seam for no gain.
PPU_BUILD_DIR="$OUT/build" TARGET=quactlize_ppu JOBS="$JOBS" \
  PPU_SDK="$PPU_SDK_ROOT" bash "$ROOT/build.sh" | tee "$OUT/library-build.log"
LIB="$(find "$OUT/build" -type f -name libquactlize_ppu.so -print -quit)"
if [[ -z "$LIB" ]]; then
  echo "[GDN perf] FAIL: shipping libquactlize_ppu.so was not produced" >&2
  exit 1
fi
LIBDIR="$(dirname "$LIB")"
BIN="$OUT/test_ppu_chunked_gdn_perf"

g++ -std=c++17 -O3 -DSWITCH_TO_HGGCRT \
  -I"$ROOT/quactlize/include" \
  -I"$PPU_SDK_ROOT/include" \
  -I"$PPU_SDK_ROOT/targets/x86_64-linux/include" \
  "$ROOT/benchmarks/test_ppu_chunked_gdn_perf.cpp" \
  -L"$LIBDIR" -lquactlize_ppu \
  -L"$PPU_SDK_ROOT/lib" -lhg_wrapper -lhggc_wrapper -lhggcrt1 -lhggc \
  -Wl,-rpath,"$LIBDIR" -Wl,-rpath,"$PPU_SDK_ROOT/lib" \
  -o "$BIN"

# DT_RUNPATH can be overridden by an inherited LD_LIBRARY_PATH.  Put the
# measured library first, then prove the dynamic loader resolves that exact
# file before accepting any timing or ACU report.
export LD_LIBRARY_PATH="$LIBDIR:$PPU_SDK_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
RESOLVED_LIB="$(ldd "$BIN" | awk '/libquactlize_ppu\.so/ {print $3; exit}')"
if [[ -z "$RESOLVED_LIB" || "$(readlink -f "$RESOLVED_LIB")" != "$(readlink -f "$LIB")" ]]; then
  echo "[GDN perf] FAIL: runtime library is '$RESOLVED_LIB', expected '$LIB'" >&2
  exit 1
fi

{
  echo "git_head=$SHA"
  echo "git_status_begin"
  git -C "$ROOT" status --porcelain=v1 --untracked-files=all
  echo "git_status_end"
  echo "source=benchmarks/test_ppu_chunked_gdn_perf.cpp"
  echo "library=$LIB"
  echo "binary=$BIN"
  echo "runtime_library=$RESOLVED_LIB"
  sha256sum \
    "$ROOT/benchmarks/test_ppu_chunked_gdn_perf.cpp" \
    "$ROOT/tools/run_ppu_chunked_gdn_perf_box.sh" \
    "$ROOT/quactlize/include/quactlize_ppu_linear_attention.h" \
    "$ROOT/quactlize/csrc/device/ppu_chunked_gdn_backend.cu" \
    "$ROOT/quactlize/include/actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_types.hpp" \
    "$ROOT/quactlize/include/actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_inverse.hpp" \
    "$ROOT/quactlize/include/actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_aiu.cuh" \
    "$ROOT/quactlize/include/actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_kernel.cuh" \
    "$ROOT/quactlize/include/actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_collective.cuh" \
    "$LIB" "$BIN"
} | tee "$OUT/binary-identity.txt"

run_shape() {
  local sequences="$1" length="$2" qk_heads="$3" v_heads="$4" label="$5"
  local log="$OUT/${label}.log"
  {
    echo "[GDN perf runner] label=$label B=$sequences T=$length H=$qk_heads HV=$v_heads"
    "$BIN" \
      --sequences="$sequences" --length="$length" \
      --qk-heads="$qk_heads" --v-heads="$v_heads" \
      --warmup="$WARMUP" --iterations="$ITERATIONS" --samples="$SAMPLES" \
      --initial-state=1 --final-state=1
  } | tee "$log"
  CURRENT_LOGS+=("$log")
}

if [[ "${ACU:-0}" == 1 ]]; then
  ACU_BIN="${ACU_BIN:-/sim/eec/shared/junfu.qx/asight/bin/acu}"
  if [[ ! -x "$ACU_BIN" ]]; then
    echo "[GDN perf ACU] FAIL: acu unavailable at $ACU_BIN" >&2
    exit 1
  fi
  # One launch, no warmup and no timing loop. The report is an instruction/
  # resource profile; it is deliberately labelled NOT_TIMING by the binary.
  REPORT="$OUT/${label}.report.acurep"
  "$ACU_BIN" -f -o "$REPORT" --set full "$BIN" \
    --sequences="$sequences" --length="$length" \
    --qk-heads="$qk_heads" --v-heads="$v_heads" \
    --initial-state=1 --final-state=1 --acu \
    | tee "$OUT/${label}.acu.log"
  if [[ ! -s "$REPORT" ]]; then
    echo "[GDN perf ACU] FAIL: acu produced no nonempty report at $REPORT" >&2
    exit 1
  fi
  echo "[GDN perf ACU] report=$REPORT timing=NOT_TIMING"
  echo "[GDN perf] PASS; artifacts=$OUT"
  exit 0
fi

# B,T,H,HV,label. These five rows separate short-sequence wave count (32, 64,
# exactly 72, 128 CTAs), GVA 1:2 from 1:1, and a cuLA-like long sequence.
for row in "${shape_rows[@]}"; do
  IFS=',' read -r sequences length qk_heads v_heads label extra <<< "$row"
  run_shape "$sequences" "$length" "$qk_heads" "$v_heads" "$label"
done

grep -h '^\[GDN perf runner\]\|^\[GDN perf config\]\|^\[GDN perf\] protocol=' "${CURRENT_LOGS[@]}" \
  | tee "$OUT/performance-summary.log"
echo "[GDN perf] PASS; artifacts=$OUT"
