#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${QUACTLIZE_L206_OUT:-/workspace/quactlize-l206-chunked-gdn-global-dot}
mkdir -p "$OUT"

NVCC=${NVCC:-nvcc}
command -v "$NVCC" >/dev/null || {
  echo '[l206] SKIP: nvcc is required for the local __HGGCCC__ type oracle'
  exit 3
}

common=(
  -std=c++17 -arch=sm_120 --expt-relaxed-constexpr
  -D__HGGCCC__=1
  -I"$ROOT/quactlize/include"
  -I"$ROOT/third_party/actlize/include"
  -I"$ROOT/dev/fold_derivation/stub_inc"
  "$ROOT/dev/fold_derivation/l206_chunked_gdn_global_dot_ownership.cu"
)

"$NVCC" "${common[@]}" -o "$OUT/l206" >"$OUT/positive.log" 2>&1
"$OUT/l206" | tee "$OUT/run.log"
grep -Fq \
  '[l206] PASS: source=__HGGCCC__-CollectiveBuilder::TiledMma threads=128 coordinate_stride=1 tile=64x64 visits=4096 holes=0 duplicate_coordinates=0 duplicate_visits=0 oob=0 min=1 max=1' \
  "$OUT/run.log" || {
    echo '[l206] FAIL: positive ownership witness changed' >&2
    exit 1
  }

for plant in THREAD_COUNT COORDINATE_STRIDE; do
  log="$OUT/negative-${plant,,}.log"
  binary="$OUT/negative-${plant,,}"
  "$NVCC" "${common[@]}" -D"L206_PLANT_${plant}=1" \
      -o "$binary" >"$log" 2>&1
  if "$binary" >>"$log" 2>&1; then
    echo "[l206] FAIL: ${plant} negative unexpectedly passed" >&2
    exit 1
  fi
  grep -Fq \
    '[l206] FAIL: source=__HGGCCC__-CollectiveBuilder::TiledMma' \
    "$log" || {
      echo "[l206] FAIL: ${plant} negative failed for the wrong reason" >&2
      exit 1
    }
  echo "[l206 negative] ${plant}=EXPECTED_RED/PASS"
done

echo "[l206] PASS: production accumulator map exact-once; two negatives red; artifacts=$OUT"
