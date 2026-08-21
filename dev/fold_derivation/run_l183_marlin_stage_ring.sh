#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L183_OUT:-/workspace/quactlize-l183-marlin-stage-ring}"
cxx="${CXX:-g++}"

mkdir -p "${out}"
"${cxx}" -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  "${repo}/dev/fold_derivation/l183_marlin_stage_ring.cpp" \
  -o "${out}/l183_marlin_stage_ring"

args=(
  "--production=${repo}/quactlize/include/actlize_extensions/cutlass/gemm/collective/marlin_collective_ppu.hpp"
  "--classic=${repo}/../marlin_classic_ppu.cuh"
)
"${out}/l183_marlin_stage_ring" "${args[@]}"

plants=(preload-short wrong-ring-slot wrong-refill-predicate)
for plant in "${plants[@]}"; do
  log="${out}/${plant}.log"
  set +e
  "${out}/l183_marlin_stage_ring" "${args[@]}" \
    "--plant=${plant}" >"${log}" 2>&1
  rc=$?
  set -e
  if [[ ${rc} -ne 1 ]] ||
     ! grep -Fq "[l183:red] plant=${plant} caught=1" "${log}"; then
    cat "${log}" >&2
    echo "[l183:runner] FAIL: ${plant} did not produce its named causal RED" >&2
    exit 1
  fi
  cat "${log}"
done

echo '[l183:runner] positive=160/160_PASS negative_controls=3/3_RED result=PASS'
