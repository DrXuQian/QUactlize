#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="$(mktemp -d /tmp/quactlize-l178.XXXXXX)"
trap 'rm -rf "${out}"' EXIT

flags=(-std=c++17 -x cu -arch=sm_80 -w --expt-relaxed-constexpr -D__HGGCCC__
       -I "${repo}/dev/fold_derivation/stub_inc"
       -I "${repo}/third_party/actlize/include"
       -I "${repo}/third_party/actlize/tools/util/include"
       -I "${repo}/quactlize/include")
nvcc "${flags[@]}" -o "${out}/l178" \
  "${repo}/dev/fold_derivation/l178_marlin_state_hoist.cpp"

reference_args=(
  "--classic=${repo}/../marlin_classic_ppu.cuh"
  "--awesome=${repo}/../ref/awesome-cute/gemm/marlin_gemm/marlin_cute_trait.h"
)
"${out}/l178" "${reference_args[@]}"

semantic_plants=(
  b-pitch-codes
  b-inner-n-cohort
  b-k-missing-wk
  scale-k-byte-half
  a-k-missing-step
  q-local-not-global
  tight-a-smem
  a-predicate-all-threads
  scale-predicate-cohort
)
for plant in "${semantic_plants[@]}"; do
  log="${out}/${plant}.log"
  set +e
  "${out}/l178" "${reference_args[@]}" --plant="${plant}" >"${log}" 2>&1
  rc=$?
  set -e
  if [[ ${rc} -ne 1 ]] ||
     ! grep -Fq "[l178:red] plant=${plant} caught=1" "${log}"; then
    cat "${log}" >&2
    echo "[l178] FAIL: ${plant} did not produce its named causal RED" >&2
    exit 1
  fi
done

python3 "${repo}/ci/check_l178_marlin_state_hoist.py" --source-only
source_plants=(runtime-topology shared-per-segment integer-segment-offset)
for plant in "${source_plants[@]}"; do
  log="${out}/${plant}.source.log"
  set +e
  python3 "${repo}/ci/check_l178_marlin_state_hoist.py" \
    --source-only --plant="${plant}" >"${log}" 2>&1
  rc=$?
  set -e
  if [[ ${rc} -ne 1 ]] ||
     ! grep -Fq "[l178-source:red] plant=${plant} caught=1" "${log}"; then
    cat "${log}" >&2
    echo "[l178] FAIL: ${plant} did not produce its named source RED" >&2
    exit 1
  fi
done

echo "[l178:runner] positive=exhaustive+source negative_controls=12/12_RED result=PASS"
