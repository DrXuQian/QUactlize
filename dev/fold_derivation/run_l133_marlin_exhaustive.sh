#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
manifest="${1:?usage: run_l133_marlin_exhaustive.sh manifest.tsv}"
out="${QUACTLIZE_L133_OUT:-/tmp/quactlize_l133}"
mkdir -p "${out}"

inc=()
if [[ -n "${L133_CORE_OVERRIDE:-}" ]]; then
  inc+=(-I "${L133_CORE_OVERRIDE}")
fi
inc+=(-I "${repo}/dev/fold_derivation/stub_inc"
     -I "${repo}/third_party/actlize/include"
     -I "${repo}/third_party/actlize/tools/util/include"
     -I "${repo}/quactlize/include")

nvcc -std=c++17 -x cu -arch=sm_80 -w "${inc[@]}" \
  -o "${out}/oracle" "${repo}/dev/fold_derivation/l133_marlin_exhaustive.cu"
expected_rows="$(wc -l < "${manifest}")"
"${out}/oracle" "${manifest}" "${expected_rows}"
