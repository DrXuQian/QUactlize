#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L127_OUT:-/tmp/quactlize_l127}"
mkdir -p "${out}"

nvcc -std=c++17 -x cu -arch=sm_80 -w \
  -I "${repo}/dev/fold_derivation/stub_inc" \
  -I "${repo}/third_party/actlize/include" \
  -I "${repo}/third_party/actlize/tools/util/include" \
  -I "${repo}/quactlize/include" \
  -o "${out}/l127_metadata_stride" \
  "${repo}/dev/fold_derivation/l127_metadata_stride.cu"

"${out}/l127_metadata_stride"
