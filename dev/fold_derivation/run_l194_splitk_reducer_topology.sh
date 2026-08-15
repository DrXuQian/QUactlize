#!/usr/bin/env bash
set -u

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L194_OUT:-/workspace/quactlize-l194-splitk-reducer-topology}"
mkdir -p "${out}"

nvcc="${NVCC:-$(command -v nvcc 2>/dev/null || true)}"
nvidia_smi="${NVIDIA_SMI:-$(command -v nvidia-smi 2>/dev/null || true)}"
if [[ -z "${nvcc}" || -z "${nvidia_smi}" ]]; then
  echo "[l194] SKIP: nvcc and a visible NVIDIA device are required"
  exit 3
fi
if ! "${nvidia_smi}" -L >"${out}/devices.log" 2>&1 ||
   [[ ! -s "${out}/devices.log" ]]; then
  echo "[l194] SKIP: no NVIDIA device is visible"
  exit 3
fi
capability="$("${nvidia_smi}" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]')"
if [[ ! "${capability}" =~ ^[0-9]+\.[0-9]+$ ]]; then
  echo "[l194] SKIP: could not derive compute capability"
  exit 3
fi
arch="sm_${capability/./}"
source="${repo}/dev/fold_derivation/l194_splitk_reducer_topology.cu"
binary="${out}/l194_splitk_reducer_topology"
flags=(
  -std=c++17 -O3 -arch="${arch}" --expt-relaxed-constexpr -D__HGGCCC__
  -I "${repo}/dev/fold_derivation/stub_inc"
  -I "${repo}/quactlize/include"
  -I "${repo}/third_party/actlize/include"
)

"${nvcc}" "${flags[@]}" "${source}" -o "${binary}" >"${out}/build.log" 2>&1
build_rc=$?
if [[ ${build_rc} -ne 0 ]]; then
  echo "[l194] FAIL: topology benchmark did not compile for ${arch}"
  tail -n 60 "${out}/build.log"
  exit 1
fi
"${binary}" | tee "${out}/run.log"
run_rc=${PIPESTATUS[0]}
if [[ ${run_rc} -ne 0 ]] ||
   ! grep -Fq '[l194] PASS: legacy, 12 vector topology cases and 3 production-fast cases are raw-bit exact' "${out}/run.log"; then
  echo "[l194] FAIL: live topology sweep did not close"
  exit 1
fi

# Named RED control: the production S=4 dispatcher must actually select the
# four-plane body.  Re-routing only that case to S=2 must reproduce a numeric
# failure; a generic-only benchmark would incorrectly remain green here.
plant_root="${out}/plant"
plant_header="${plant_root}/quactlize_extensions/cutlass/gemm/device/ppu_mixed_input_splitk_parallel.hpp"
mkdir -p "$(dirname "${plant_header}")"
plant_source="${repo}/quactlize/include/quactlize_extensions/cutlass/gemm/device/ppu_mixed_input_splitk_parallel.hpp"
plant_anchor='case 4: return launch<KernelS4>(stream);'
plant_anchor_count="$(grep -Foc "${plant_anchor}" "${plant_source}")"
if [[ "${plant_anchor_count}" != 1 ]]; then
  echo "[l194] FAIL: S4-dispatch plant anchor count=${plant_anchor_count}, expected=1"
  exit 1
fi
sed \
  's/case 4: return launch<KernelS4>(stream);/case 4: return launch<KernelS2>(stream);/' \
  "${plant_source}" \
  >"${plant_header}"
if cmp -s "${plant_header}" "${plant_source}"; then
  echo "[l194] FAIL: S4-dispatch negative control was not planted"
  exit 1
fi
plant_binary="${out}/l194_splitk_reducer_topology_s4_dispatch_red"
"${nvcc}" -I "${plant_root}" "${flags[@]}" "${source}" \
  -o "${plant_binary}" >"${out}/plant-build.log" 2>&1
plant_build_rc=$?
if [[ ${plant_build_rc} -ne 0 ]]; then
  echo "[l194] FAIL: S4-dispatch negative control did not compile"
  tail -n 60 "${out}/plant-build.log"
  exit 1
fi
set +e
"${plant_binary}" >"${out}/plant-run.log" 2>&1
plant_run_rc=$?
set -e
if [[ ${plant_run_rc} -eq 0 ]] ||
   ! grep -Eq 'L194_PRODUCTION_FAST S=4 .* bad=[1-9][0-9]*' \
     "${out}/plant-run.log"; then
  echo "[l194] FAIL: S4-dispatch negative control did not produce the named RED"
  tail -n 40 "${out}/plant-run.log"
  exit 1
fi
echo "[l194] RED/PASS: S4->S2 dispatcher plant produced numeric failure"
echo "[l194] PASS: ${arch} local topology sweep; artifacts=${out}"
