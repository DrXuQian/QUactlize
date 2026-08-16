#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
base="${QUACTLIZE_L198_OUT:-/workspace/quactlize-l198-dense-splitk-oneplane}"
out="${base}/run-$$"
mkdir -p "${out}"

host_source="${repo}/dev/fold_derivation/l198_dense_splitk_oneplane.cpp"
host_binary="${out}/l198-dense-splitk-oneplane-host"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wno-comment -pedantic \
  -I "${repo}/quactlize/include" -I "${repo}/benchmarks" \
  "${host_source}" -o "${host_binary}" >"${out}/host-build.log" 2>&1 || {
  echo '[l198:runner] FAIL: exhaustive host oracle did not compile' >&2
  tail -n 120 "${out}/host-build.log" >&2
  exit 1
}

"${host_binary}" | tee "${out}/host-run.log"
grep -Fqx \
  '[l198] PASS tables=3 rows=4790 modes=2 cells=38320 admitted=33004 inadmissible_pipeline_depth=5316 per_split=9580/9384/8088/5952 formats=12908/14556/5540 bc=22956/10048 descriptors=8187072 qk=78703616 raw_bit_checks=231028 resident_payload_invariant=33004 partial=FP32' \
  "${out}/host-run.log" || {
  echo '[l198:runner] FAIL: exhaustive host denominator drifted' >&2
  exit 1
}

for plant in bits mode artifact fold bchunk partial; do
  set +e
  "${host_binary}" --plant "${plant}" \
    >"${out}/host-plant-${plant}.log" 2>&1
  rc=$?
  set -e
  cat "${out}/host-plant-${plant}.log"
  if [[ "${rc}" -ne 1 ]] ||
      ! grep -Fqx \
        "[l198:plant] EXPECTED_RED name=${plant} planted=1 errors=1" \
        "${out}/host-plant-${plant}.log"; then
    echo "[l198:runner] FAIL: host plant=${plant} was not exactly one RED" >&2
    exit 1
  fi
done

command -v nvcc >/dev/null 2>&1 || {
  echo '[l198:runner] FAIL: nvcc is required for shipping type binding' >&2
  exit 1
}

type_source="${repo}/dev/fold_derivation/l198_dense_splitk_oneplane_types.cu"
type_flags=(
  -std=c++17 -arch=sm_80 --expt-relaxed-constexpr -D__HGGCCC__
  -I "${repo}/dev/fold_derivation/stub_inc"
  -I "${repo}/third_party/actlize/include"
  -I "${repo}/third_party/actlize/tools/util/include"
  -I "${repo}/quactlize/include"
  -Wno-deprecated-gpu-targets
  -diag-suppress 549,177,20012,3357,20014
)

for bc in 0 1; do
  type_binary="${out}/l198-dense-splitk-oneplane-type-bc${bc}"
  nvcc "${type_flags[@]}" -DPPU_B_CHUNK="${bc}" \
    "${type_source}" -o "${type_binary}" \
    >"${out}/type-bc${bc}-build.log" 2>&1 || {
    echo "[l198:runner] FAIL: type matrix bc=${bc} did not compile" >&2
    tail -n 160 "${out}/type-bc${bc}-build.log" >&2
    exit 1
  }
  "${type_binary}" | tee "${out}/type-bc${bc}-run.log"
  if [[ "${bc}" -eq 0 ]]; then
    format_cells=6
    bits='4/2/1'
  else
    format_cells=4
    bits='2/1'
  fi
  grep -Fqx \
    "[l198:type] PASS bc=${bc} format_cells=${format_cells} bits=${bits} modes=ScaleOnly/ScaleZero S1=EXACT_SHIPPING_TYPE SGT1=EXACT_COLLECTIVE partial=FP32 completion=SEPARATE artifact_reader=EXACT arguments=SO:null+/-nonnull- SZ:nonnull+/null- static_gs=16/32:match+/mismatch- workspace_s8=131072 atom_i2=0 atom_i1=0" \
    "${out}/type-bc${bc}-run.log" || {
    echo "[l198:runner] FAIL: type/argument matrix bc=${bc} drifted" >&2
    exit 1
  }
done

for plant in BITS MODE ARTIFACT BCHUNK; do
  case "${plant}" in
    BITS) marker='L198_FORMAT_BITS_SEAM' ;;
    MODE) marker='L198_FORMAT_MODE_SEAM' ;;
    ARTIFACT) marker='L198_FORMAT_ARTIFACT_SEAM' ;;
    BCHUNK) marker='L198_FORMAT_BCHUNK_SEAM' ;;
  esac
  set +e
  nvcc "${type_flags[@]}" -DPPU_B_CHUNK=0 \
    -D"L198_PLANT_${plant}"=1 "${type_source}" \
    -o "${out}/type-plant-${plant,,}" \
    >"${out}/type-plant-${plant,,}.log" 2>&1
  rc=$?
  set -e
  if [[ "${rc}" -eq 0 ]] ||
      ! grep -Fq "static assertion failed with \"${marker}\"" \
        "${out}/type-plant-${plant,,}.log"; then
    echo "[l198:runner] FAIL: compile plant=${plant,,} escaped ${marker}" >&2
    tail -n 160 "${out}/type-plant-${plant,,}.log" >&2
    exit 1
  fi
  echo "[l198:type-plant] EXPECTED_RED name=${plant,,} marker=${marker}"
done

echo "[l198:runner] PASS host_cells=38320 admitted=33004 host_plants=6 type_buckets=2 format_plants=4 static_gs=16/32 artifacts=${out}"
