#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${QUACTLIZE_L207_OUT:-/workspace/quactlize-l207-chunked-gdn-resident-mma}
mkdir -p "$OUT"

NVCC=${NVCC:-nvcc}
command -v "$NVCC" >/dev/null || {
  echo '[l207] SKIP: nvcc is required for the local __HGGCCC__ type oracle'
  exit 3
}

common=(
  -std=c++17 -arch=sm_120 --expt-relaxed-constexpr
  -D__HGGCCC__=1
  -I"$ROOT/quactlize/include"
  -I"$ROOT/third_party/actlize/include"
  -I"$ROOT/dev/fold_derivation/stub_inc"
  "$ROOT/dev/fold_derivation/l207_chunked_gdn_resident_mma_ownership.cu"
)

"$NVCC" "${common[@]}" -o "$OUT/l207" >"$OUT/positive-build.log" 2>&1
"$OUT/l207" | tee "$OUT/positive-run.log"
grep -Fq \
  '[l207] PASS: source=production-__HGGCCC__-CollectiveBuilder::TiledMma tile=64x64x64 warp=32x32x64 threads=128 resident=A@V A=(visits=8192,holes=0,dup_coord=0,dup_visits=0,oob=0,min=2,max=2,coord_bad=0,map=29a6a34b79ebcb25,anchor=29a6a34b79ebcb25) B=(visits=8192,holes=0,dup_coord=0,dup_visits=0,oob=0,min=2,max=2,coord_bad=0,map=70eb86e99be9bb25,anchor=70eb86e99be9bb25) C=(visits=4096,holes=0,dup_coord=0,dup_visits=0,oob=0,min=1,max=1,coord_bad=0,map=ee9011938bb9c325,anchor=ee9011938bb9c325) anchor=public-PPU0010-atom+2Mx2N-warp-topology' \
  "$OUT/positive-run.log" || {
    echo '[l207] FAIL: positive resident-fragment ownership witness changed' >&2
    exit 1
  }

for plant in B_TRANSPOSE COORDINATE_STRIDE THREAD_COUNT; do
  log="$OUT/negative-${plant,,}.log"
  binary="$OUT/negative-${plant,,}"
  "$NVCC" "${common[@]}" -D"L207_PLANT_${plant}=1" \
      -o "$binary" >"$log" 2>&1
  if "$binary" >>"$log" 2>&1; then
    echo "[l207] FAIL: ${plant} negative unexpectedly passed" >&2
    exit 1
  fi
  grep -Fq \
    '[l207] FAIL: source=production-__HGGCCC__-CollectiveBuilder::TiledMma' \
    "$log" || {
      echo "[l207] FAIL: ${plant} negative failed for the wrong reason" >&2
      exit 1
    }
  echo "[l207 negative] ${plant}=EXPECTED_RED/PASS"
done

echo "[l207] PASS: production resident A/B/C maps anchored; three negatives red; artifacts=$OUT"
