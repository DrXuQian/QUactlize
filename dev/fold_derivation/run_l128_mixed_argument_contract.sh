#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L128_OUT:-/tmp/quactlize_l128}"
mkdir -p "${out}"

nvcc -std=c++17 -x cu -arch=sm_80 -w \
  -I "${repo}/dev/fold_derivation/stub_inc" \
  -I "${repo}/third_party/actlize/include" \
  -I "${repo}/third_party/actlize/tools/util/include" \
  -I "${repo}/quactlize/include" \
  -o "${out}/l128_mixed_argument_contract" \
  "${repo}/dev/fold_derivation/l128_mixed_argument_contract.cu"

"${out}/l128_mixed_argument_contract"
