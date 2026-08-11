#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L124_OUT:-/tmp/quactlize_l124}"
mkdir -p "${out}"
python3 "${repo}/dev/fold_derivation/gen_l124_cases.py" "${repo}" "${out}/l124_cases.inc"
nvcc -std=c++17 -x cu -arch=sm_80 -w \
  -I "${out}" \
  -I "${repo}/dev/fold_derivation/stub_inc" \
  -I "${repo}/third_party/actlize/include" \
  -I "${repo}/third_party/actlize/tools/util/include" \
  -I "${repo}/quactlize/include" \
  -o "${out}/l124_fp32_residue_mask" \
  "${repo}/dev/fold_derivation/l124_fp32_residue_mask.cu"
"${out}/l124_fp32_residue_mask"
