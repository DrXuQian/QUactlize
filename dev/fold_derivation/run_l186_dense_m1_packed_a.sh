#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L186_OUT:-/workspace/quactlize-l186-dense-m1-packed-a}"
mkdir -p "${out}"

type_src="${repo}/dev/fold_derivation/l186_dense_m1_packed_a.cu"
geometry_src="${repo}/dev/fold_derivation/l186_dense_m1_packed_a_geometry.cu"
common=(-std=c++17 -arch=sm_80 -w
        -I "${repo}/dev/fold_derivation/stub_inc"
        -I "${repo}/third_party/actlize/include"
        -I "${repo}/third_party/actlize/tools/util/include"
        -I "${repo}/quactlize/include")
type_flags=("${common[@]}" --expt-relaxed-constexpr -D__HGGCCC__
            -I "${repo}/tests" -I "${repo}/benchmarks")

nvcc "${type_flags[@]}" -o "${out}/types" "${type_src}"
"${out}/types"
# The seven-cell matrix includes scale-first and arrangement-aware fully-quantized types. Compile the exact Q4 and
# Q2 packed-metadata build modes too: a default-only type proof would never instantiate the metadata branch used by
# the FQ ABI even though the A provider itself compiled.
nvcc "${type_flags[@]}" -DPPU_PACKED_SCALE=1 -DPPU_PACKED_FORMAT=0 \
  -o "${out}/types-q4-packed" "${type_src}"
"${out}/types-q4-packed"
nvcc "${type_flags[@]}" -DPPU_PACKED_SCALE=1 -DPPU_PACKED_FORMAT=2 \
  -o "${out}/types-q2-packed" "${type_src}"
"${out}/types-q2-packed"
nvcc "${common[@]}" -x cu -o "${out}/geometry" "${geometry_src}"
"${out}/geometry"

# Structural RED controls use the same two oracles. A zero-row provider and a logical m16 substitute must both be
# rejected at compile time; a one-element destination mutation must be observed by exhaustive collision coverage.
# Source plants have their own RED spelling: the checker returns nonzero only after naming the violated contract;
# a plant that merely makes the checker print "plant escaped" returns success and this runner rejects it below.
set +e
nvcc "${type_flags[@]}" -DL186_PACK_ROWS=0 -o "${out}/rows0" "${type_src}" \
  >"${out}/rows0.log" 2>&1
rows0_rc=$?
nvcc "${type_flags[@]}" -DL186_TILE_M=16 -o "${out}/m16" "${type_src}" \
  >"${out}/m16.log" 2>&1
m16_rc=$?
nvcc "${common[@]}" -x cu -DL186_BAD_DESTINATION_DELTA=1 \
  -o "${out}/bad_destination" "${geometry_src}" >"${out}/bad_destination_build.log" 2>&1
bad_build_rc=$?
bad_run_rc=99
if [[ ${bad_build_rc} -eq 0 ]]; then
  "${out}/bad_destination" >"${out}/bad_destination.log" 2>&1
  bad_run_rc=$?
fi
nvcc "${common[@]}" -x cu -DL186_BAD_SLICE_SWAP=1 \
  -o "${out}/bad_slice_swap" "${geometry_src}" >"${out}/bad_slice_swap_build.log" 2>&1
swap_build_rc=$?
swap_run_rc=99
if [[ ${swap_build_rc} -eq 0 ]]; then
  "${out}/bad_slice_swap" >"${out}/bad_slice_swap.log" 2>&1
  swap_run_rc=$?
fi
nvcc "${common[@]}" -x cu -DL186_BAD_LOGICAL_X2_WORD_DELTA=1 \
  -o "${out}/bad_logical_x2" "${geometry_src}" >"${out}/bad_logical_x2_build.log" 2>&1
x2_build_rc=$?
x2_run_rc=99
if [[ ${x2_build_rc} -eq 0 ]]; then
  "${out}/bad_logical_x2" >"${out}/bad_logical_x2.log" 2>&1
  x2_run_rc=$?
fi
set -e

if [[ ${rows0_rc} -eq 0 || ${m16_rc} -eq 0 ||
      ${bad_build_rc} -ne 0 || ${bad_run_rc} -eq 0 ||
      ${swap_build_rc} -ne 0 || ${swap_run_rc} -eq 0 ||
      ${x2_build_rc} -ne 0 || ${x2_run_rc} -eq 0 ]]; then
  echo "[l186] FAIL RED controls rows0=${rows0_rc} m16=${m16_rc} " \
       "bad_build=${bad_build_rc} bad_run=${bad_run_rc} " \
       "swap_build=${swap_build_rc} swap_run=${swap_run_rc} " \
       "x2_build=${x2_build_rc} x2_run=${x2_run_rc}" >&2
  exit 1
fi

python3 "${repo}/ci/check_dense_m1_packed_a.py"
for plant in missing-m1-guard default-type-wrapped query-launch-diverged coverage-denominator physical-stage-pitch \
             packed-route-marked-universal; do
  if python3 "${repo}/ci/check_dense_m1_packed_a.py" --plant "${plant}" \
      >"${out}/plant-${plant}.log" 2>&1; then
    echo "[l186] FAIL source plant escaped: ${plant}" >&2
    exit 1
  fi
done

echo "[l186] PASS: 7 production Q2/Q4 cells + writer/independent-reader geometry; " \
     "logical-x2 scalar map exact; rows0/m16/destination/slice-swap/x2-word " \
     "and 6 source plants RED; output=${out}"
