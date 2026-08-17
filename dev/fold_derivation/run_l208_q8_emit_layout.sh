#!/usr/bin/env bash
set -Eeuo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
base="${QUACTLIZE_L208_OUT:-/workspace/quactlize-l208-q8-emit-layout}"
out="${base}/run-$$"
mkdir -p "${out}"

source_file="${repo}/dev/fold_derivation/l208_q8_emit_layout.cu"
flags=(
  -std=c++17 -O2 -x cu -arch=sm_80 --expt-relaxed-constexpr -w
  -I "${repo}/dev/fold_derivation/stub_inc"
  -I "${repo}/third_party/actlize/include"
  -I "${repo}/third_party/actlize/tools/util/include"
  -I "${repo}/quactlize/include"
  -I "${repo}/benchmarks"
)

nvcc "${flags[@]}" "${source_file}" -o "${out}/l208-positive" \
  >"${out}/positive-build.log" 2>&1 || {
  echo "[l208-runner] FAIL: positive oracle did not compile" >&2
  tail -n 160 "${out}/positive-build.log" >&2
  exit 1
}
"${out}/l208-positive" | tee "${out}/positive-run.log"
grep -Fqx \
  '[l208 emit] inputs=16 mismatch=0 holes=0 duplicates=0 map=0,2,1,3,8,10,9,11,4,6,5,7,12,14,13,15' \
  "${out}/positive-run.log" || {
  echo "[l208-runner] FAIL: actlize int8x16 emission denominator or map drifted" >&2
  exit 1
}
grep -Fqx \
  '[l208] PASS emit=PASS placement=PASS q8_value=PASS' \
  "${out}/positive-run.log" || {
  echo "[l208-runner] FAIL: positive oracle did not close all three properties" >&2
  exit 1
}
grep -Fqx \
  '[l208 placement] candidates=18 canonical=A32/F1 fixture=128x256 unset=0 out_of_range=0 holes=0 duplicates=0 roundtrip_bad=0/589824 byte_diff=0/589824' \
  "${out}/positive-run.log" || {
  echo "[l208-runner] FAIL: the 18-row Q8 family is not one exact A32/F1 resident layout" >&2
  exit 1
}

nvcc "${flags[@]}" -DL208_PLANT_WRONG_PERM=1 "${source_file}" -o "${out}/l208-wrong-perm" \
  >"${out}/wrong-perm-build.log" 2>&1 || {
  echo "[l208-runner] FAIL: wrong-permutation plant did not compile" >&2
  tail -n 160 "${out}/wrong-perm-build.log" >&2
  exit 1
}
set +e
"${out}/l208-wrong-perm" >"${out}/wrong-perm-run.log" 2>&1
rc=$?
set -e
cat "${out}/wrong-perm-run.log"
if [[ "${rc}" -ne 1 ]] ||
   ! grep -Fqx '[l208 emit] inputs=16 mismatch=1 holes=1 duplicates=1 map=0,3,1,3,8,10,9,11,4,6,5,7,12,14,13,15' \
      "${out}/wrong-perm-run.log" ||
   ! grep -Fqx '[l208] FAIL emit=FAIL placement=PASS q8_value=PASS' \
      "${out}/wrong-perm-run.log"; then
  echo "[l208-runner] FAIL: one-bit permutation plant did not produce the exact expected RED" >&2
  exit 1
fi

# Candidate-denominator negative: remove one tuple from the shared authority
# and compile L208 against that planted copy.  The oracle's independently
# fixed denominator of 18 must turn red; otherwise benchmark and oracle could
# shrink together and still claim complete coverage.
awk '
  /^PREFILL_Q8_CANDIDATE\(/ { ++candidate; if (candidate == 18) next }
  { print }
' "${repo}/benchmarks/prefill_q8_candidates.inc" >"${out}/prefill_q8_candidates.inc"
nvcc -I "${out}" "${flags[@]}" "${source_file}" -o "${out}/l208-missing-candidate" \
  >"${out}/missing-candidate-build.log" 2>&1 || {
  echo "[l208-runner] FAIL: missing-candidate plant did not compile" >&2
  tail -n 160 "${out}/missing-candidate-build.log" >&2
  exit 1
}
set +e
"${out}/l208-missing-candidate" >"${out}/missing-candidate-run.log" 2>&1
rc=$?
set -e
tail -n 4 "${out}/missing-candidate-run.log"
if [[ "${rc}" -ne 1 ]] ||
   ! grep -Fqx \
      '[l208 placement] candidates=17 canonical=A32/F1 fixture=128x256 unset=0 out_of_range=0 holes=0 duplicates=0 roundtrip_bad=0/557056 byte_diff=0/557056' \
      "${out}/missing-candidate-run.log" ||
   ! grep -Fqx '[l208] FAIL emit=PASS placement=FAIL q8_value=PASS' \
      "${out}/missing-candidate-run.log"; then
  echo "[l208-runner] FAIL: dropping one shared candidate did not produce the exact expected RED" >&2
  exit 1
fi

echo "[l208-runner] PASS emission=16/16 placement=18x(exact-once+roundtrip+A32-byte-identical) q8_value=1024/1024 wrong_perm=EXPECTED_RED missing_candidate=EXPECTED_RED artifacts=${out}"
