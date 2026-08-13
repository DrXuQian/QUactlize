#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${QUACTLIZE_L173_OUT:-}" ]]; then
  out="${QUACTLIZE_L173_OUT}"
else
  out="$(mktemp -d /tmp/quactlize-l173.XXXXXX)"
fi
mkdir -p "${out}"

flags=(-std=c++17 -x cu -arch=sm_80 -w --expt-relaxed-constexpr -D__HGGCCC__
       -I "${repo}/dev/fold_derivation/stub_inc"
       -I "${repo}/third_party/actlize/include"
       -I "${repo}/third_party/actlize/tools/util/include"
       -I "${repo}/quactlize/include")
nvcc "${flags[@]}" -o "${out}/l173" \
  "${repo}/dev/fold_derivation/l173_marlin_cta_state.cpp"
"${out}/l173"

plants=(init-per-segment init-before-valid stale-rebase drop-b-q drop-scale-q
        drop-a-k drop-b-k drop-scale-k tight-a-swizzle)
for plant in "${plants[@]}"; do
  log="${out}/${plant}.log"
  set +e
  "${out}/l173" --plant="${plant}" >"${log}" 2>&1
  rc=$?
  set -e
  if [[ ${rc} -ne 1 ]] ||
     ! grep -Fq "[l173:red] plant=${plant} caught=1" "${log}"; then
    cat "${log}" >&2
    echo "[l173] FAIL: ${plant} did not produce its named causal RED" >&2
    exit 1
  fi
done

echo "[l173:runner] positive=PASS negative_controls=${#plants[@]}/${#plants[@]}_RED result=PASS"
