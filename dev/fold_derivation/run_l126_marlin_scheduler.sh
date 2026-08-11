#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L126_OUT:-/tmp/quactlize_l126}"
mkdir -p "${out}"

inc=(-I "${repo}/dev/fold_derivation/stub_inc"
     -I "${repo}/third_party/actlize/include"
     -I "${repo}/third_party/actlize/tools/util/include"
     -I "${repo}/quactlize/include" -I "${repo}/tests" -I "${repo}/benchmarks")
nvcc -std=c++17 -x cu -arch=sm_80 -w "${inc[@]}" \
  -D__HGGCCC__ --expt-relaxed-constexpr -DL126_TYPE_ONLY=1 \
  -o "${out}/types" "${repo}/dev/fold_derivation/l126_marlin_scheduler.cu"
nvcc -std=c++17 -x cu -arch=sm_80 -w "${inc[@]}" \
  -o "${out}/scheduler" "${repo}/dev/fold_derivation/l126_marlin_scheduler.cu"
"${out}/types"
"${out}/scheduler"
