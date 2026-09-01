#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L148_OUT:-/tmp/quactlize_l148_dense_shipping_tm8}"
mkdir -p "${out}"
src="${repo}/dev/fold_derivation/l148_dense_shipping_tm8_compiled.cu"
flags=(-std=c++17 -arch=sm_80 -w --expt-relaxed-constexpr -D__HGGCCC__
       -I "${repo}/dev/fold_derivation/stub_inc"
       -I "${repo}/third_party/actlize/include"
       -I "${repo}/third_party/actlize/tools/util/include"
       -I "${repo}/quactlize/include"
       -I "${repo}/tests" -I "${repo}/benchmarks")

combined="${out}/compiled_cells.txt"
: > "${combined}"

nvcc "${flags[@]}" -DL148_SCALE_FIRST=1 -o "${out}/scale" "${src}"
"${out}/scale" | tee -a "${combined}"
for format in 0 1 2 3 4; do
  nvcc "${flags[@]}" -DPPU_PACKED_SCALE=1 -DPPU_PACKED_FORMAT="${format}" \
    -o "${out}/fq_${format}" "${src}"
  "${out}/fq_${format}" | tee -a "${combined}"
done

cells="$(grep -c '^compiled format=' "${combined}")"
legal="$(grep -c ' verdict=LEGAL' "${combined}")"
illegal="$(grep -c ' verdict=ILLEGAL' "${combined}")"
if [[ "${cells}" -ne 60 || "${legal}" -ne 51 || "${illegal}" -ne 9 ]]; then
  echo "[l148] FAIL: exact compiled census cells=${cells} legal=${legal} illegal=${illegal}" >&2
  exit 1
fi

expected_invalid=(
  "Q2_K|fully-quantized|8x128:8x32:s12"
  "Q3_K|scale-first|8x128:8x32:s12"
  "Q3_K|fully-quantized|8x128:8x32:s12"
  "Q4_K|fully-quantized|8x128:8x32:s12"
  "Q5_K|scale-first|8x128:8x32:s8"
  "Q5_K|scale-first|8x128:8x32:s12"
  "Q5_K|fully-quantized|8x128:8x32:s8"
  "Q5_K|fully-quantized|8x128:8x32:s12"
  "Q6_K|fully-quantized|8x128:8x32:s12"
)
actual_invalid="$({
  awk '/^compiled format=/ && / verdict=ILLEGAL/ {
    split($2, f, "="); split($3, m, "="); split($5, c, "=");
    print f[2] "|" m[2] "|" c[2]
  }' "${combined}"
} | sort)"
expected_sorted="$(printf '%s\n' "${expected_invalid[@]}" | sort)"
if [[ "${actual_invalid}" != "${expected_sorted}" ]]; then
  echo "[l148] FAIL exact illegal-cell identity drifted" >&2
  diff -u <(printf '%s\n' "${expected_sorted}") <(printf '%s\n' "${actual_invalid}") || true
  exit 1
fi
if ! grep -Fq 'format=Q5_K mode=scale-first tk=256 config=8x128:8x32:s8 shared_bytes=262160 verdict=ILLEGAL' "${combined}" ||
   ! grep -Fq 'format=Q6_K mode=fully-quantized tk=128 config=8x128:8x32:s12 shared_bytes=301056 verdict=ILLEGAL' "${combined}"; then
  echo "[l148] FAIL: exact boundary witnesses (zero-array padding / packed staging) drifted" >&2
  exit 1
fi
echo "[l148] PASS exact compiled census cells=${cells} legal=${legal} illegal=${illegal}; all 9 identities pinned; output=${combined}"
