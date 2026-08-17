#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${QUACTLIZE_L204_OUT:-/workspace/quactlize-l204-chunked-gdn-device}
mkdir -p "$OUT"

NVCC=${NVCC:-nvcc}
command -v "$NVCC" >/dev/null || {
  echo '[l204] SKIP: nvcc is required to instantiate the portable CUDA device body'
  exit 3
}

common=(
  -std=c++17 -arch=sm_120 --expt-relaxed-constexpr
  -I"$ROOT/quactlize/include"
  -I"$ROOT/third_party/actlize/include"
  -I"$ROOT/dev/fold_derivation/stub_inc"
  "$ROOT/dev/fold_derivation/l204_chunked_gdn_device_compile.cu"
)

"$NVCC" "${common[@]}" -o "$OUT/l204" >"$OUT/positive.log" 2>&1
"$OUT/l204" | tee "$OUT/run.log"
grep -Fq '[l204] PASS: device-body=INSTANTIATED' "$OUT/run.log" || {
  echo '[l204] FAIL: positive binary lost its exact device-body witness' >&2
  exit 1
}

for plant in CHUNK HEAD; do
  if "$NVCC" "${common[@]}" -D"L204_PLANT_${plant}=1" \
      -o "$OUT/negative-${plant,,}" >"$OUT/negative-${plant,,}.log" 2>&1; then
    echo "[l204] FAIL: ${plant} negative unexpectedly compiled" >&2
    exit 1
  fi
  grep -Fq 'this collective is the fixed C64/D128 implementation' \
      "$OUT/negative-${plant,,}.log" || {
    echo "[l204] FAIL: ${plant} negative failed for the wrong reason" >&2
    exit 1
  }
  echo "[l204 negative] ${plant}=EXPECTED_RED/PASS"
done

echo "[l204] PASS: exact device type compiled; C/head negatives red; artifacts=$OUT"
