#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L139_STANDALONE_OUT:-/workspace/quactlize-l139-standalone-fragment}"
mkdir -p "${out}"

python3 "${repo}/dev/fold_derivation/check_l139_standalone_fragment.py"

nvcc -std=c++17 -x cu -arch=sm_80 -w --expt-relaxed-constexpr \
  -I "${repo}/dev/fold_derivation/stub_inc" \
  -I "${repo}/third_party/actlize/include" \
  -I "${repo}/third_party/actlize/tools/util/include" \
  -I "${repo}/quactlize/include" \
  -o "${out}/l139" \
  "${repo}/dev/fold_derivation/l139_marlin_warpk_reduce.cu"

"${out}/l139" | tee "${out}/positive.log"
grep -Fq \
  'L139 PASS: four classic acc_i/acc_j cohorts are exhaustive; K0=64x32 and production 4->2->1 reduction are raw-bit exact' \
  "${out}/positive.log"
grep -Fq \
  'generic_vs_classic=6144 compact_vs_classic=8128' \
  "${out}/positive.log"

faults=(
  generic-layout
  compact-layout
  omit-cohort
  flat-reduction
  all-cohorts-write
  duplicate-k0-owner
)
for fault in "${faults[@]}"; do
  set +e
  "${out}/l139" --fault="${fault}" >"${out}/${fault}.log" 2>&1
  rc=$?
  set -e
  if [[ ${rc} -ne 2 ]]; then
    cat "${out}/${fault}.log" >&2
    echo "[l139:runner] FAIL: ${fault} expected RED rc=2, got ${rc}" >&2
    exit 1
  fi
  grep -Fq "L139 EXPECTED-RED fault=${fault}" "${out}/${fault}.log"
done

# These are exact signatures, not merely "some difference".  They ensure the
# two tempting bijections remain independently observable.
grep -Fq \
  'fault=generic-layout raw={association:1536 cadence:0}' \
  "${out}/generic-layout.log"
grep -Fq \
  'fault=compact-layout raw={association:2032 cadence:0}' \
  "${out}/compact-layout.log"
grep -Fq \
  'fault=flat-reduction raw={association:0 cadence:2048}' \
  "${out}/flat-reduction.log"

echo '[l139:runner] PASS: production alias + exhaustive 4-cohort geometry + raw 4->2->1; six causal plants RED'
