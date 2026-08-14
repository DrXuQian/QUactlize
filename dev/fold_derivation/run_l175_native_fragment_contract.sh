#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
oracle="${repo}/dev/fold_derivation/l175_native_fragment_contract.py"
source_file="${repo}/dev/fold_derivation/l175_native_fragment_contract.cu"
out="${QUACTLIZE_L175_OUT:-/workspace/quactlize-l175-native-fragment}"
mkdir -p "${out}"

python3 "${oracle}" | tee "${out}/source-positive.log"

command -v nvcc >/dev/null 2>&1 || {
  echo '[l175] FAIL: nvcc is required for the compile-only type oracle' >&2
  exit 1
}

common=(
  -std=c++17 -x cu -arch=sm_80 -w --expt-relaxed-constexpr
  -I "${repo}/dev/fold_derivation/stub_inc"
  -I "${repo}/third_party/actlize/include"
  -I "${repo}/third_party/actlize/tools/util/include"
  -I "${repo}/quactlize/include"
)
nvcc "${common[@]}" -o "${out}/l175" "${source_file}"
"${out}/l175" | tee "${out}/type-positive.log"
grep -Fq \
  'L175 fragment_bytes=32 fragment_values=8 accumulator_bytes=128 accumulator_fragments=4 m8_fragment_bytes=16 m8_fragment_values=4 m8_accumulator_bytes=64 m8_accumulator_fragments=4 m16_a_bytes=16 m8_a_bytes=8 standard_layout=1 trivial=1 result=PASS' \
  "${out}/type-positive.log"

plants=(generic-fragment whole-accum-reinterpret wrong-4x8-layout wrong-m8-a-registers)
for plant in "${plants[@]}"; do
  set +e
  python3 "${oracle}" --plant="${plant}" >"${out}/${plant}.log" 2>&1
  rc=$?
  set -e
  if [[ ${rc} -ne 1 ]] ||
     ! grep -Fq "[l175:red] plant=${plant} caught=1" "${out}/${plant}.log"; then
    cat "${out}/${plant}.log" >&2
    echo "[l175] FAIL: ${plant} did not produce its named causal RED" >&2
    exit 1
  fi
done

# A dimension transposition preserves 128 bytes, so compile the real type
# assertions through an include overlay.  This proves sizeof alone cannot make
# the wrong 8x4 register association green.
tmp="${out}/wrong-layout-work"
mkdir -p "${tmp}"
rel=quactlize_extensions/cutlass/gemm/collective/marlin_mma_ppu.hpp
mkdir -p "${tmp}/overlay/$(dirname "${rel}")"
cp "${repo}/quactlize/include/${rel}" "${tmp}/overlay/${rel}"
python3 - "${tmp}/overlay/${rel}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
if text.count("float value[8];") != 1 or text.count("FragmentC fragments[4];") != 1:
    raise SystemExit("L175 wrong-layout compile plant seam drifted")
text = text.replace("float value[8];", "float value[4];", 1)
text = text.replace("FragmentC fragments[4];", "FragmentC fragments[8];", 1)
path.write_text(text, encoding="utf-8")
PY
set +e
nvcc -I "${tmp}/overlay" "${common[@]}" \
  -o "${out}/wrong-layout" "${source_file}" \
  >"${out}/wrong-layout-compile.log" 2>&1
wrong_rc=$?
set -e
if [[ ${wrong_rc} -eq 0 ]]; then
  echo '[l175] FAIL: size-preserving 8x4 layout compiled unexpectedly' >&2
  exit 1
fi
grep -Fq 'L175_FRAGMENT_C_EIGHT_FLOATS' "${out}/wrong-layout-compile.log" || {
  sed -n '1,60p' "${out}/wrong-layout-compile.log" >&2
  echo '[l175] FAIL: wrong-layout compile did not reach FragmentC dimension assertion' >&2
  exit 1
}
grep -Fq 'L175_ACCUMULATOR_FOUR_NATIVE_FRAGMENTS' "${out}/wrong-layout-compile.log" || {
  sed -n '1,60p' "${out}/wrong-layout-compile.log" >&2
  echo '[l175] FAIL: wrong-layout compile did not reach accumulator dimension assertion' >&2
  exit 1
}

# Preserve the independent classic acc_i/acc_j and exact raw 4->2->1 oracle.
# Do not infer this from the new type assertions: shape and semantics are
# separate facts.
QUACTLIZE_L139_STANDALONE_OUT="${tmp}/l139-out" \
  bash "${repo}/dev/fold_derivation/run_l139_standalone_fragment_oracle.sh" \
  >"${out}/l139.log" 2>&1
grep -Fq \
  'generic_vs_classic=6144 compact_vs_classic=8128' \
  "${out}/l139.log"
grep -Fq \
  'L139 PASS: four classic acc_i/acc_j cohorts are exhaustive; K0=64x32 and production 4->2->1 reduction are raw-bit exact' \
  "${out}/l139.log"

echo '[l175:runner] positive=source+compile+L139 negative_controls=4/4_RED wrong-layout-compile=RED result=PASS'
