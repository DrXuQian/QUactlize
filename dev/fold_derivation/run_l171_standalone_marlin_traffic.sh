#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
marlin_root="$(cd "${repo}/.." && pwd)"
out="${QUACTLIZE_L171_OUT:-/tmp/quactlize_l171}"
mkdir -p "${out}"

flags=(-std=c++17 -x cu -arch=sm_80 -w --expt-relaxed-constexpr -D__HGGCCC__
       -I "${repo}/dev/fold_derivation/stub_inc"
       -I "${repo}/third_party/actlize/include"
       -I "${repo}/third_party/actlize/tools/util/include"
       -I "${repo}/quactlize/include")
nvcc "${flags[@]}" -o "${out}/l171" \
  "${repo}/dev/fold_derivation/l171_standalone_marlin_traffic.cpp"

args=(
  "--collective=${repo}/quactlize/include/actlize_extensions/cutlass/gemm/collective/marlin_collective_ppu.hpp"
  "--kernel=${repo}/quactlize/include/actlize_extensions/cutlass/gemm/kernel/marlin_kernel_ppu.hpp"
  "--awesome=${marlin_root}/ref/awesome-cute/gemm/marlin_gemm/marlin_cute_trait.h"
  "--classic=${marlin_root}/marlin_classic_ppu.cuh"
)
"${out}/l171" "${args[@]}"

for plant in a-per-k-cohort b-two-source-duplicate stage-ring-refill scale-all-threads; do
  log="${out}/${plant}.log"
  set +e
  "${out}/l171" "${args[@]}" "--plant=${plant}" >"${log}" 2>&1
  rc=$?
  set -e
  if [[ ${rc} -ne 1 ]] ||
     ! grep -Fq "[l171:red] plant=${plant} caught=1" "${log}"; then
    cat "${log}" >&2
    echo "[l171] FAIL: ${plant} did not produce its named causal RED" >&2
    exit 1
  fi
done

echo '[l171:runner] positive=PASS negative_controls=4/4_RED result=PASS'
