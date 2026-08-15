#!/usr/bin/env bash
set -u

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${QUACTLIZE_L195_OUT:-/workspace/quactlize-l195-splitk-reducer-saturated}"
mkdir -p "${out}"

nvcc="${NVCC:-$(command -v nvcc 2>/dev/null || true)}"
nvidia_smi="${NVIDIA_SMI:-$(command -v nvidia-smi 2>/dev/null || true)}"
if [[ -z "${nvcc}" || -z "${nvidia_smi}" ]]; then
  echo "[l195] SKIP: nvcc and a visible NVIDIA device are required"
  exit 3
fi
capability="$("${nvidia_smi}" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]')"
if [[ ! "${capability}" =~ ^[0-9]+\.[0-9]+$ ]]; then
  echo "[l195] SKIP: could not derive compute capability"
  exit 3
fi
arch="sm_${capability/./}"
source="${repo}/dev/fold_derivation/l195_splitk_reducer_saturated.cu"
binary="${out}/l195_splitk_reducer_saturated"
flags=(
  -std=c++17 -O3 -arch="${arch}" --expt-relaxed-constexpr -D__HGGCCC__
  -I "${repo}/dev/fold_derivation/stub_inc"
  -I "${repo}/quactlize/include"
  -I "${repo}/third_party/actlize/include"
)
"${nvcc}" "${flags[@]}" "${source}" -o "${binary}" >"${out}/build.log" 2>&1
if [[ $? -ne 0 ]]; then
  echo "[l195] FAIL: saturated reducer benchmark did not compile for ${arch}"
  tail -n 60 "${out}/build.log"
  exit 1
fi
"${binary}" | tee "${out}/run.log"
rc=${PIPESTATUS[0]}
if [[ ${rc} -ne 0 ]] ||
   ! grep -Fq '[l195] PASS: all production vector widths are exact; performance is diagnostic' "${out}/run.log"; then
  echo "[l195] FAIL: saturated reducer diagnostic did not close"
  exit 1
fi
echo "[l195] PASS: ${arch}; artifacts=${out}"
