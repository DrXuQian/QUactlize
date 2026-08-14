#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tmp="$(mktemp -d /tmp/quactlize-l179.XXXXXX)"
trap 'rm -rf "${tmp}"' EXIT

host_flags=(-std=c++17 -x cu -arch=sm_80 -w --expt-relaxed-constexpr
  -I "${repo}/dev/fold_derivation/stub_inc"
  -I "${repo}/third_party/actlize/include"
  -I "${repo}/third_party/actlize/tools/util/include"
  -I "${repo}/quactlize/include")
geometry_source="${repo}/dev/fold_derivation/l179_marlin_checked_lowering.cpp"
nvcc "${host_flags[@]}" -o "${tmp}/geometry" "${geometry_source}"
"${tmp}/geometry"

# A direct census control proves the range/exact-once oracle itself can fail.
set +e
"${tmp}/geometry" --plant=col-plus-one >"${tmp}/col-plus-one.log" 2>&1
rc=$?
set -e
[[ ${rc} -eq 1 ]] &&
  grep -Fq '[l179:red] plant=col-plus-one caught=1 reason=output cohort escaped its global q tile' \
    "${tmp}/col-plus-one.log" || {
  cat "${tmp}/col-plus-one.log" >&2
  echo '[l179] FAIL: direct coordinate census control was not rejected' >&2
  exit 1
}

# Production-causal arithmetic controls: change one real guard in a temporary
# include overlay, rebuild the same oracle, and require the adjacent first-
# invalid production boundary to become accepted.
collective_rel=quactlize_extensions/cutlass/gemm/collective/marlin_collective_ppu.hpp
collective_src="${repo}/quactlize/include/${collective_rel}"
for spec in \
  'drop-b-k-product:production b_k_delta*K boundary was removed' \
  'drop-b-delta-materialization:production b_k_delta materialization boundary was removed'; do
  plant="${spec%%:*}"
  witness="${spec#*:}"
  overlay="${tmp}/overlay-${plant}"
  probe="${overlay}/${collective_rel}"
  mkdir -p "$(dirname "${probe}")"
  cp "${collective_src}" "${probe}"
  python3 - "${probe}" "${plant}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
plant = sys.argv[2]
text = path.read_text()
if plant == "drop-b-k-product":
    needle = "mul_fits_int(b_k_delta, last_k_tile) &&"
    replacement = "true &&  // L179 removed b_k_delta*K guard"
elif plant == "drop-b-delta-materialization":
    needle = "if (!mul_fits_int(b_global_stride, KBlocks)) {"
    replacement = "if (false) {  // L179 removed b_k_delta materialization guard"
else:
    raise SystemExit(f"unknown plant {plant}")
if text.count(needle) != 1:
    raise SystemExit(f"L179 plant seam is not unique: {plant}")
path.write_text(text.replace(needle, replacement, 1))
PY
  nvcc -I "${overlay}" "${host_flags[@]}" -o "${tmp}/${plant}" "${geometry_source}"
  set +e
  "${tmp}/${plant}" --plant="${plant}" >"${tmp}/${plant}.log" 2>&1
  rc=$?
  set -e
  [[ ${rc} -eq 1 ]] && grep -Fq "[l179:red] plant=${plant} caught=1 reason=${witness}" \
    "${tmp}/${plant}.log" || {
    cat "${tmp}/${plant}.log" >&2
    echo "[l179] FAIL: production arithmetic plant ${plant} missed its exact boundary" >&2
    exit 1
  }
done

# The map is also a production seam.  Mutate each helper independently in an
# overlay; the exhaustive census must reject it for a coordinate reason, not
# merely because a named plant was supplied.
map_rel=quactlize_extensions/cutlass/gemm/kernel/marlin_output_map_ppu.hpp
map_src="${repo}/quactlize/include/${map_rel}"
for plant in output-row output-n-base output-col-offset; do
  overlay="${tmp}/overlay-${plant}"
  probe="${overlay}/${map_rel}"
  mkdir -p "$(dirname "${probe}")"
  cp "${map_src}" "${probe}"
  python3 - "${probe}" "${plant}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
plant = sys.argv[2]
text = path.read_text()
changes = {
    "output-row": ("return lane / 4 +", "return lane / 8 +"),
    "output-n-base": ("n_tile * 8 +", "n_tile * 4 +"),
    "output-col-offset": ("((value % 4) << 2)", "((value % 4) << 1)"),
}
needle, replacement = changes[plant]
if text.count(needle) != 1:
    raise SystemExit(f"L179 output-map plant seam is not unique: {plant}")
path.write_text(text.replace(needle, replacement, 1))
PY
  nvcc -I "${overlay}" "${host_flags[@]}" -o "${tmp}/${plant}" "${geometry_source}"
  set +e
  "${tmp}/${plant}" --plant="${plant}" >"${tmp}/${plant}.log" 2>&1
  rc=$?
  set -e
  [[ ${rc} -eq 1 ]] && grep -Fq "[l179:red] plant=${plant} caught=1" \
    "${tmp}/${plant}.log" &&
    ! grep -Fq 'named coordinate plant missed its invariant' "${tmp}/${plant}.log" || {
    cat "${tmp}/${plant}.log" >&2
    echo "[l179] FAIL: production output-map plant ${plant} was not causally rejected" >&2
    exit 1
  }
done

python3 "${repo}/ci/check_l179_marlin_checked_lowering.py"
for plant in device-valid unchecked-lowering noncanonical-batch-stride col-guard unchecked-workspace raw-adapter; do
  set +e
  python3 "${repo}/ci/check_l179_marlin_checked_lowering.py" --plant="${plant}" \
    >"${tmp}/${plant}.source.log" 2>&1
  rc=$?
  set -e
  [[ ${rc} -eq 1 ]] && grep -Fq "[l179-source:red] plant=${plant} caught=1" "${tmp}/${plant}.source.log" || {
    cat "${tmp}/${plant}.source.log" >&2
    echo "[l179] FAIL: source plant ${plant} was not rejected" >&2
    exit 1
  }
done

unit_flags=(-std=c++17 -arch=sm_80 --expt-relaxed-constexpr -D__HGGCCC__
  -I "${repo}/dev/fold_derivation/stub_inc"
  -I "${repo}/third_party/actlize/include"
  -I "${repo}/third_party/actlize/tools/util/include"
  -I "${repo}/quactlize/include" -x cu
  "${repo}/dev/fold_derivation/l179_marlin_checked_handle.cu"
  -Wno-deprecated-gpu-targets)
nvcc "${unit_flags[@]}" -o "${tmp}/handle" >"${tmp}/positive.log" 2>&1 || {
  sed -n '1,80p' "${tmp}/positive.log" >&2
  echo '[l179] FAIL: owned handle lifecycle fixture did not compile/link' >&2
  exit 1
}
"${tmp}/handle"

for spec in \
  'L179_PLANT_UPDATE:deleted function' \
  'L179_PLANT_RAW_PARAMS_RUN:deleted function' \
  'L179_PLANT_PARAMS_ACCESS:deleted function' \
  'L179_PLANT_UPCAST:L179_FORBIDS_RAW_ADAPTER_UPCAST' \
  'L179_PLANT_PARAMS_CALL:deleted function' \
  'L179_PLANT_PARAMS_GRID:deleted function'; do
  macro="${spec%%:*}"
  witness="${spec#*:}"
  set +e
  nvcc "${unit_flags[@]}" -D"${macro}" -o "${tmp}/${macro}" \
    >"${tmp}/${macro}.log" 2>&1
  rc=$?
  set -e
  if [[ ${rc} -eq 0 ]] || ! grep -Fq "${witness}" "${tmp}/${macro}.log"; then
    sed -n '1,60p' "${tmp}/${macro}.log" >&2
    echo "[l179] FAIL: unsafe API ${macro} did not fail with its expected witness" >&2
    exit 1
  fi
done

echo '[l179:runner] positive=geometry+source+handle-lifecycle negative_controls=17/17_RED result=PASS'
