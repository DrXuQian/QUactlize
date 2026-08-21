#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L186_OUT:-/workspace/quactlize-l186-marlin-accumulator-n}"
mkdir -p "${out}"

command -v nvcc >/dev/null 2>&1 || {
  echo '[l186] FAIL: nvcc is required for the compile-only type oracle' >&2
  exit 1
}

source_file="${repo}/dev/fold_derivation/l186_marlin_accumulator_n_axis.cu"
common=(
  -std=c++17 -x cu -arch=sm_80 -w --expt-relaxed-constexpr
  -I "${repo}/dev/fold_derivation/stub_inc"
  -I "${repo}/third_party/actlize/include"
  -I "${repo}/third_party/actlize/tools/util/include"
  -I "${repo}/quactlize/include"
)

nvcc "${common[@]}" -o "${out}/l186" "${source_file}"
"${out}/l186" | tee "${out}/positive.log"
grep -Fq \
  'L186 shipping=m8:64B/4,m16:128B/4 wide=m8:128B/8,m16:256B/8 WN64-type-identity=1 native-fragments=1 result=PASS' \
  "${out}/positive.log"

# Negative 1: keeping the same bytes but routing N=4 through a different
# aggregate type must fail.  This distinguishes type identity from sizeof.
rel=actlize_extensions/cutlass/gemm/collective/marlin_mma_ppu.hpp
identity_overlay="${out}/identity-overlay"
mkdir -p "${identity_overlay}/$(dirname "${rel}")"
cp "${repo}/quactlize/include/${rel}" "${identity_overlay}/${rel}"
python3 - "${identity_overlay}/${rel}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = "NBlocks == 4,\n    MarlinAccumulatorFor<InstructionM>,"
new = "false,\n    MarlinAccumulatorFor<InstructionM>,"
if text.count(old) != 1:
    raise SystemExit("L186 identity plant seam is not unique")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
PY
set +e
nvcc -I "${identity_overlay}" "${common[@]}" \
  -o "${out}/identity-red" "${source_file}" \
  >"${out}/identity-red.log" 2>&1
identity_rc=$?
set -e
if [[ ${identity_rc} -eq 0 ]] ||
   ! grep -Fq 'L186_WN64_EXACT_TYPE_IDENTITY' "${out}/identity-red.log"; then
  sed -n '1,80p' "${out}/identity-red.log" >&2
  echo '[l186] FAIL: WN64 type-identity plant did not turn RED' >&2
  exit 1
fi

# Negative 2: lose one WN128 native fragment without changing any other seam.
# The compile oracle must name the exact eight-fragment contract.
count_overlay="${out}/count-overlay"
mkdir -p "${count_overlay}/$(dirname "${rel}")"
cp "${repo}/quactlize/include/${rel}" "${count_overlay}/${rel}"
python3 - "${count_overlay}/${rel}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = "Fragment fragments[NBlocks];"
new = "Fragment fragments[NBlocks - 1];"
if text.count(old) != 1:
    raise SystemExit("L186 fragment-count plant seam is not unique")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
PY
set +e
nvcc -I "${count_overlay}" "${common[@]}" \
  -o "${out}/count-red" "${source_file}" \
  >"${out}/count-red.log" 2>&1
count_rc=$?
set -e
if [[ ${count_rc} -eq 0 ]] ||
   ! grep -Fq 'L186_WN128_EIGHT_NATIVE_N16_FRAGMENTS' "${out}/count-red.log"; then
  sed -n '1,80p' "${out}/count-red.log" >&2
  echo '[l186] FAIL: WN128 fragment-count plant did not turn RED' >&2
  exit 1
fi

echo '[l186:runner] positive=WN64-exact-type+WN128-eight-native negative=identity+count_RED result=PASS'
