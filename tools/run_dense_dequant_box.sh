#!/usr/bin/env bash
# Freeze this bounded dequant experiment; no model/provider build or install.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
PYTHON=${PYTHON:-python3}
GPU=${GPU:-${CUDA_VISIBLE_DEVICES:-1}}
[[ ${GPU} =~ ^[0-9]+$ ]] || { echo 'ERROR: choose exactly one GPU ordinal.' >&2; exit 1; }
[[ ${IDLE_CONFIRMED:-0} == 1 ]] || { echo 'ERROR: check the assigned GPU is idle, then set IDLE_CONFIRMED=1.' >&2; exit 1; }
[[ ${PROFILE_ACU:-0} =~ ^[01]$ ]] || { echo 'ERROR: PROFILE_ACU must be 0 or 1.' >&2; exit 1; }
[[ ${Q8_EXCHANGE:-0} =~ ^[01]$ ]] || { echo 'ERROR: Q8_EXCHANGE must be 0 or 1.' >&2; exit 1; }
[[ ${Q8_FAST_TAIL:-0} =~ ^[01]$ ]] || { echo 'ERROR: Q8_FAST_TAIL must be 0 or 1.' >&2; exit 1; }
[[ ${Q8_EXCHANGE:-0}${Q8_FAST_TAIL:-0} != 11 ]] || { echo 'ERROR: choose one Q8 experiment.' >&2; exit 1; }
[[ -z ${RUN:-} ]] || { echo 'ERROR: unset RUN; use RESUME_RUN to retry a retained run.' >&2; exit 1; }
PPU_SDK=$(realpath -e -- "${PPU_SDK:-/usr/local/PPU_SDK}")
test -x "${PPU_SDK}/bin/hgcc"
export CUDA_VISIBLE_DEVICES=${GPU} PPU_SDK
export PATH="${PPU_SDK}/bin:${PATH}"
export LD_LIBRARY_PATH="${PPU_SDK}/lib:${PPU_SDK}/targets/x86_64-linux/lib:${PPU_SDK}/CUDA_SDK/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"${PYTHON}" -c 'import numpy,torch;from gguf.quants import dequantize'
args=(--confirm-idle --sdk "${PPU_SDK}" --l2-bytes "${L2_BYTES:-67108864}")
if [[ ${Q8_EXCHANGE:-0} == 1 ]]; then args+=(--q8-exchange); fi
if [[ ${Q8_FAST_TAIL:-0} == 1 ]]; then args+=(--q8-fast-tail); fi
[[ -z ${POINTS:-} ]] || args+=(--points "${POINTS}")
if [[ -n ${RESUME_RUN:-} ]]; then
    RUN=$(realpath -e -- "${RESUME_RUN}")
    test -f "${RUN}/source/tools/run_dense_dequant.py"
    test -f "${RUN}/results/build/receipt.json"
    args+=(--resume)
else
    test -d "${OUTPUT_ROOT:-/workspace}"
    REV=$(git -C "${ROOT}" rev-parse --verify HEAD)
    ACTLIZE_REV=$(git -C "${ROOT}" rev-parse "${REV}:third_party/actlize")
    [[ ${REV} =~ ^[0-9a-f]{40}$ && ${ACTLIZE_REV} =~ ^[0-9a-f]{40}$ ]] || exit 1
    if ! git -C "${ROOT}/third_party/actlize" cat-file -e "${ACTLIZE_REV}^{commit}"; then
        echo "ERROR: local actlize must contain ${ACTLIZE_REV}; no nested network fetch attempted." >&2
        exit 1
    fi
    RUN=$(mktemp -d "${OUTPUT_ROOT:-/workspace}/kpack-dense-dequant.XXXXXXXX")
    [[ -n ${RUN} && -d ${RUN} && ! -L ${RUN} ]] || exit 1
    mkdir "${RUN}/source"
    git -C "${ROOT}" archive "${REV}" | tar -x -C "${RUN}/source"
    mkdir -p "${RUN}/source/third_party/actlize"
    git -C "${ROOT}/third_party/actlize" archive "${ACTLIZE_REV}" | tar -x -C "${RUN}/source/third_party/actlize"
    export DENSE_DEQUANT_SOURCE_SHA=${REV} DENSE_DEQUANT_ACTLIZE_SHA=${ACTLIZE_REV}
fi
if [[ ${PROFILE_ACU:-0} == 1 ]]; then
    test -x "${ACU:-${PPU_SDK}/asight/bin/acu}"
    args+=(--acu "${ACU:-${PPU_SDK}/asight/bin/acu}")
    [[ -z ${PROFILE_POINTS:-} ]] || args+=(--profile-points "${PROFILE_POINTS}")
fi
[[ -z ${PPU_SMI:-} ]] || args+=(--smi "${PPU_SMI}")
echo "DENSE_DEQUANT_BOX run=${RUN} gpu=${GPU} q8_exchange=${Q8_EXCHANGE:-0} q8_fast_tail=${Q8_FAST_TAIL:-0} gemm_calls=0 kernel_default_changed=0"
set +e
"${PYTHON}" -u "${RUN}/source/tools/run_dense_dequant.py" --output "${RUN}/results" \
    "${args[@]}" 2>&1 | tee "${RUN}/console-$(date +%s%N).log"
rc=${PIPESTATUS[0]}
set -e
echo "runner_rc=${rc} run=${RUN}"
echo "DENSE_DEQUANT_RESUME RESUME_RUN=${RUN} GPU=${GPU} Q8_EXCHANGE=${Q8_EXCHANGE:-0} Q8_FAST_TAIL=${Q8_FAST_TAIL:-0} PROFILE_ACU=${PROFILE_ACU:-0} IDLE_CONFIRMED=1 bash tools/run_dense_dequant_box.sh"
exit "${rc}"
