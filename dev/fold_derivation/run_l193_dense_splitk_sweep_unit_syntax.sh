#!/usr/bin/env bash
# Compile the real L192-generated dense Split-K sweep main and representative
# wrapper units through nvcc's complete CUDA front end.  Stock nvcc cannot
# lower this PPU CuTe tree: the only accepted diagnostics in the unmodified
# arm are the two measured host-environment cute::_/cute::product failures.
# Everything else is a real failure.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
base="${QUACTLIZE_L193_OUT:-/workspace/quactlize-l193-dense-splitk-sweep-unit-syntax}"
mkdir -p "${base}"
out="${base}/run-$$"
mkdir "${out}"

command -v nvcc >/dev/null 2>&1 || {
  echo '[l193] FAIL: nvcc is required for the generated-unit front-end gate' >&2
  exit 1
}

# Generate the sources by executing the production CMake block.  Do not
# maintain a second row list or substitute hand-written TUs here.
l192_root="${out}/l192"
QUACTLIZE_L192_OUT="${l192_root}" \
  bash "${repo}/dev/fold_derivation/run_l192_dense_splitk_sweep_generator.sh" \
  >"${out}/l192.log" 2>&1 || {
    echo '[l193] FAIL: L192 production generator prerequisite failed' >&2
    sed -n '1,120p' "${out}/l192.log" >&2
    exit 1
  }

mapfile -t registries < <(
  find "${l192_root}" -type f -name dense_splitk_sweep_configs.inc -print
)
if [[ "${#registries[@]}" -ne 1 ]]; then
  echo "[l193] FAIL: expected one generated registry, found ${#registries[@]}" >&2
  exit 1
fi
gen="$(dirname "${registries[0]}")"
units="${gen}/dense_splitk_sweep_units"
main="${gen}/test_lowbit_dense_splitk_sweep_main.cu"
if [[ ! -s "${main}" ]] || \
   [[ "$(find "${units}" -maxdepth 1 -type f -name 'dense_splitk_sweep_unit_*.cu' ! -name '*.in' | wc -l)" -ne 51 ]]; then
  echo '[l193] FAIL: L192 output is not the exact one-main/51-unit graph' >&2
  exit 1
fi

flags=(
  -std=c++17 -arch=sm_80 --expt-relaxed-constexpr
  -D__HGGCCC__ -DPPU_FORCE_INSTANTIATE=1
  -Xcudafe --error_limit=100000
  -I "${repo}/dev/fold_derivation/stub_inc"
  -I "${repo}/third_party/actlize/include"
  -I "${repo}/third_party/actlize/tools/util/include"
  -I "${repo}/tests"
  -I "${repo}/benchmarks"
  -I "${repo}/quactlize/include"
  -I "${repo}/dev"
  -I "${gen}"
  -I "${units}"
  -cuda -x cu -Wno-deprecated-gpu-targets
)

diagnostic_lines() {
  grep -E ': (error|fatal error|catastrophic error):' "$1" || true
}

require_complete_frontend() {
  local label="$1" log="$2" artifact="$3" rc="$4"
  if grep -Fq 'Error limit reached' "${log}"; then
    echo "[l193] FAIL: ${label} exhausted nvcc's diagnostic budget" >&2
    return 1
  fi
  if [[ ! -s "${artifact}" ]] && \
     ! grep -qE '^[0-9]+ errors? detected in the compilation of ' "${log}"; then
    echo "[l193] FAIL: ${label} produced neither an artifact nor a final EDG error count (rc=${rc})" >&2
    sed -n '1,20p' "${log}" >&2
    return 1
  fi
}

compile_exact() {
  local label="$1" source="$2" expected_rows="$3"
  local log="${out}/${label}.log" artifact="${out}/${label}.cu.cpp"
  if [[ "${expected_rows}" -gt 0 ]] && \
     [[ "$(grep -Ec '^  X\(dense_splitk_cfg_' "${source}" || true)" -ne "${expected_rows}" ]]; then
    echo "[l193] FAIL: ${label} generated batch does not contain ${expected_rows} authority rows" >&2
    return 1
  fi
  set +e
  nvcc "${flags[@]}" -o "${artifact}" "${source}" >"${log}" 2>&1
  local rc=$?
  set -e
  require_complete_frontend "${label}" "${log}" "${artifact}" "${rc}"

  local all_errors="${out}/${label}.errors"
  local unexpected="${out}/${label}.unexpected"
  diagnostic_lines "${log}" >"${all_errors}"
  grep -Fv 'identifier "cute::_" is undefined in device code' "${all_errors}" \
    | grep -Fv 'identifier "cute::product" is undefined in device code' \
    >"${unexpected}" || true
  if [[ -s "${unexpected}" ]]; then
    echo "[l193] FAIL: ${label} has an unexpected front-end diagnostic" >&2
    sed -n '1,40p' "${unexpected}" >&2
    return 1
  fi

  local observed reported
  observed="$(wc -l <"${all_errors}")"
  if [[ "${observed}" -eq 0 ]]; then
    if [[ "${rc}" -ne 0 ]] || [[ ! -s "${artifact}" ]]; then
      echo "[l193] FAIL: ${label} had no diagnostics but did not compile (rc=${rc})" >&2
      return 1
    fi
  else
    reported="$(sed -nE 's/^([0-9]+) errors? detected in the compilation of .*/\1/p' "${log}" | tail -1)"
    if [[ "${rc}" -eq 0 ]] || [[ -z "${reported}" ]] || \
       [[ "${reported}" -ne "${observed}" ]]; then
      echo "[l193] FAIL: ${label} diagnostic accounting is incomplete: rc=${rc} observed=${observed} reported=${reported:-none}" >&2
      return 1
    fi
  fi

  if [[ "${expected_rows}" -gt 0 ]]; then
    local run_rows prepared_rows
    run_rows="$(grep -Fc 'instantiation of "__nv_bool dense_splitk_sweep::run_row<' "${log}" || true)"
    prepared_rows="$(grep -Fc 'PreparedOnePlaneLauncher<' "${log}" || true)"
    # Once stock nvcc encounters the known CuTe environment defect, EDG emits
    # four instantiation stacks for the first concrete wrapper rather than one
    # stack per batch row.  Batch cardinality is checked directly above; here
    # the honest claim is that this representative TU reached a real generated
    # run_row and its prepared handle, not that diagnostics enumerate all rows.
    if [[ "${run_rows}" -lt 1 ]] || [[ "${prepared_rows}" -lt 1 ]]; then
      echo "[l193] FAIL: ${label} never reached a generated run_row/prepared handle: run_row=${run_rows} prepared=${prepared_rows}" >&2
      return 1
    fi
  fi
  echo "[l193] ${label}: COMPLETE known-environment-errors=${observed} generated-rows=${expected_rows}"
}

# Main proves the generated registry/runner TU reaches the complete front end.
# Units 0/10/11/50 bind the wrapper body at the minimum boundary, the
# TK128->TK256 transition, deep TK256 pipeline depths, and the final one-row
# maximum-TN/TK/WN batch.  These are production-generated sources, not models.
compile_exact main "${main}" 0
compile_exact exact-warm-tn64 \
  "${repo}/benchmarks/dense_splitk_exact_warm_ab_tn64.cu" 0
compile_exact exact-warm-tn128 \
  "${repo}/benchmarks/dense_splitk_exact_warm_ab_tn128.cu" 0
for spec in 0:4 10:4 11:4 50:1; do
  index="${spec%%:*}"
  rows="${spec##*:}"
  compile_exact "unit-${index}" \
    "${units}/dense_splitk_sweep_unit_${index}.cu" "${rows}"
done

# A controlled overlay changes only PreparedOnePlaneLauncher::run().  The
# exact generated unit-0 must instantiate both calls once for each of its four
# rows.  This is stronger than merely seeing the Prepared type in a stack: it
# proves the producer and reducer run body is reached.  The overlay is wholly
# under /workspace and never edits the shipping header.
overlay="${out}/marker-overlay"
mkdir "${overlay}"
cp "${repo}/quactlize/include/dense_splitk_parallel_ppu.cuh" \
   "${overlay}/dense_splitk_parallel_ppu.cuh"
python3 - "${overlay}/dense_splitk_parallel_ppu.cuh" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
needle = '''  cutlass::Status run(hggcStream_t stream = nullptr) {
    if (!initialized_) return cutlass::Status::kErrorInvalidProblem;'''
if text.count(needle) != 1:
    raise SystemExit("L193 PreparedOnePlaneLauncher::run seam is not unique")
plant = needle + '''
    static_assert(sizeof(SplitGemm) == 0,
                  "L193_PREPARED_PRODUCER_BODY_INSTANTIATED");
    static_assert(sizeof(Reduction) == 0,
                  "L193_PREPARED_REDUCER_BODY_INSTANTIATED");'''
path.write_text(text.replace(needle, plant, 1), encoding="utf-8")
PY

marker_log="${out}/body-marker.log"
marker_artifact="${out}/body-marker.cu.cpp"
set +e
nvcc -I "${overlay}" "${flags[@]}" -o "${marker_artifact}" \
  "${units}/dense_splitk_sweep_unit_0.cu" >"${marker_log}" 2>&1
marker_rc=$?
set -e
require_complete_frontend body-marker "${marker_log}" "${marker_artifact}" "${marker_rc}"

producer_count="$(grep -Fc 'error: static assertion failed with "L193_PREPARED_PRODUCER_BODY_INSTANTIATED"' "${marker_log}" || true)"
reducer_count="$(grep -Fc 'error: static assertion failed with "L193_PREPARED_REDUCER_BODY_INSTANTIATED"' "${marker_log}" || true)"
marker_errors="${out}/body-marker.errors"
marker_unexpected="${out}/body-marker.unexpected"
diagnostic_lines "${marker_log}" >"${marker_errors}"
grep -Fv 'identifier "cute::_" is undefined in device code' "${marker_errors}" \
  | grep -Fv 'identifier "cute::product" is undefined in device code' \
  | grep -Fv 'error: static assertion failed with "L193_PREPARED_PRODUCER_BODY_INSTANTIATED"' \
  | grep -Fv 'error: static assertion failed with "L193_PREPARED_REDUCER_BODY_INSTANTIATED"' \
  >"${marker_unexpected}" || true
if [[ "${marker_rc}" -eq 0 ]] || [[ "${producer_count}" -ne 4 ]] || \
   [[ "${reducer_count}" -ne 4 ]] || [[ -s "${marker_unexpected}" ]]; then
  echo "[l193] FAIL: controlled run-body witness drifted: producer=${producer_count} reducer=${reducer_count} rc=${marker_rc}" >&2
  sed -n '1,40p' "${marker_unexpected}" >&2
  exit 1
fi

echo "[l193] PASS: real L192 main + exact warm A/B TUs + boundary units completed nvcc -cuda; only known CuTe environment errors; producer/reducer run bodies=4/4; artifacts=${out}"
