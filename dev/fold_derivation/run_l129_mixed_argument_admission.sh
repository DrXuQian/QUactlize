#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L129_OUT:-/tmp/quactlize_l129}"
mkdir -p "${out}"

nvcc -std=c++17 -x cu -arch=sm_80 -w \
  -I "${repo}/dev/fold_derivation/stub_inc" \
  -I "${repo}/third_party/actlize/include" \
  -I "${repo}/third_party/actlize/tools/util/include" \
  -I "${repo}/quactlize/include" \
  -o "${out}/l129_mixed_argument_admission" \
  "${repo}/dev/fold_derivation/l129_mixed_argument_admission.cu"

"${out}/l129_mixed_argument_admission"
