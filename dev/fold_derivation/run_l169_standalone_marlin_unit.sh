#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
source_file="$repo/dev/fold_derivation/l169_standalone_marlin_unit.cu"
defs='-DDENSE_MARLIN_WK4_AB=1 -DDENSE_MARLIN_AB=1 -DDENSE_STREAMK_AB=1 -DBENCH_GS=128 -DBENCH_TSK=64 -DDENSE_AB_BITS=4 -DDENSE_AB_ARTIFACT_TK=64 -DDENSE_AB_TM=16 -DDENSE_AB_TN=128 -DDENSE_AB_TK=128 -DDENSE_AB_WM=16 -DDENSE_AB_WN=64 -DDENSE_AB_WARP_K=32 -DDENSE_AB_ST=4 -DDENSE_AB_BC=0 -DTILE_M=16 -DTILE_N=128 -DWARP_M=16 -DWARP_N=64 -DSTAGES=4'

command -v nvcc >/dev/null 2>&1 || {
  echo '[l169] FAIL: nvcc is required for the generated-unit compile oracle' >&2
  exit 1
}

tmp="$(mktemp -d)"
raw="$tmp/nvcc.log"
out="$tmp/unit.cu.cpp"
set +e
# shellcheck disable=SC2086  # defs is the intentional compile-definition list.
nvcc -std=c++17 -arch=sm_80 --expt-relaxed-constexpr \
  -D__HGGCCC__ -DPPU_FORCE_INSTANTIATE=1 $defs \
  -Xcudafe --error_limit=100000 \
  -I"$repo/dev/fold_derivation/stub_inc" \
  -I"$repo/third_party/actlize/include" \
  -I"$repo/third_party/actlize/tools/util/include" \
  -I"$repo/tests" -I"$repo/benchmarks" -I"$repo/quactlize/include" \
  -I"$repo/dev" -cuda -o "$out" -x cu "$source_file" \
  -Wno-deprecated-gpu-targets >"$raw" 2>&1
nvcc_rc=$?
set -e

# Stock nvcc cannot model two PPU-only inline constants in the actlize stack.
# Those are an explicit environmental floor, not a baseline: every other
# compiler diagnostic is a real failure, and completion still requires EDG's
# final error-count line plus an instantiated Marlin device_kernel note.
unexpected="$tmp/unexpected.log"
grep -E ': (error|fatal error|catastrophic error):' "$raw" \
  | grep -v 'identifier "cute::_" is undefined in device code' \
  | grep -v 'identifier "cute::product" is undefined in device code' \
  >"$unexpected" || true
if [[ -s "$unexpected" ]]; then
  echo "[l169] FAIL: generated standalone unit has non-environmental diagnostics (nvcc rc=$nvcc_rc)" >&2
  sed -n '1,20p' "$unexpected" >&2
  exit 1
fi
grep -Eq '[0-9]+ errors? detected in the compilation of' "$raw" || {
  echo '[l169] FAIL: compiler did not prove it reached the end of the unit' >&2
  exit 1
}
grep -q 'device_kernel<Operator>.*MarlinKernelPPU' "$raw" || {
  echo '[l169] FAIL: generated unit did not instantiate the standalone device kernel' >&2
  exit 1
}

echo '[l169] PASS: generated-unit shape instantiates standalone Marlin collective/scheduler/kernel; only the two explicit nvcc/PPU environmental diagnostics remain'
