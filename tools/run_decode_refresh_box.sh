#!/usr/bin/env bash
# Frozen small-M experiment; no production writes, ACU, or automatic submodule fetch.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
PYTHON=${PYTHON:-python3}
args=()
if [[ -n ${RESUME_RUN:-} ]]; then
    RUN=$(realpath -e -- "${RESUME_RUN}")
    [[ -n ${RUN} && -d ${RUN}/source && -f ${RUN}/results/plan.json ]] || {
        echo 'ERROR: RESUME_RUN must contain the frozen source and results/plan.json.' >&2
        exit 1
    }
    SOURCE=${RUN}/source
    if [[ -f ${RUN}/results/inputs.json ]]; then
        GPUS=${GPUS:-$("${PYTHON}" -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["cards"]))' "${RUN}/results/inputs.json")}
    fi
    PHASE=${PHASE:-$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["phase"])' "${RUN}/results/plan.json")}
    args+=(--resume)
else
    [[ -z ${RUN:-} ]] || { echo 'ERROR: unset RUN or use RESUME_RUN.' >&2; exit 1; }
    test -d "${OUTPUT_ROOT:-/workspace}"
    RUN=$(mktemp -d "${OUTPUT_ROOT:-/workspace}/kpack-decode-refresh.XXXXXXXX")
    [[ -n ${RUN} && -d ${RUN} && ! -L ${RUN} ]] || exit 1
    SOURCE=${RUN}/source
    mkdir "${SOURCE}"
    REV=$(git -C "${ROOT}" rev-parse --verify HEAD)
    ACTLIZE_REV=$(git -C "${ROOT}" rev-parse "${REV}:third_party/actlize")
    [[ ${REV} =~ ^[0-9a-f]{40}$ && ${ACTLIZE_REV} =~ ^[0-9a-f]{40}$ ]] || exit 1
    if ! git -C "${ROOT}/third_party/actlize" cat-file -e "${ACTLIZE_REV}^{commit}"; then
        echo "ERROR: local actlize lacks pinned ${ACTLIZE_REV}; no network/submodule checkout attempted." >&2
        exit 1
    fi
    git -C "${ROOT}" archive "${REV}" | tar -x -C "${SOURCE}"
    mkdir -p "${SOURCE}/third_party/actlize"
    git -C "${ROOT}/third_party/actlize" archive "${ACTLIZE_REV}" | tar -x -C "${SOURCE}/third_party/actlize"
fi
test -f "${SOURCE}/tools/run_decode_refresh.py"
if [[ ${PLAN_ONLY:-0} == 1 ]]; then
    args+=(--plan-only)
else
    [[ ${IDLE_CONFIRMED:-0} == 1 ]] || {
        echo 'ERROR: set IDLE_CONFIRMED=1 only after checking every requested card is otherwise idle.' >&2
        exit 1
    }
    PPU_SDK=${PPU_SDK:-/usr/local/PPU_SDK}
    test -x "${PPU_SDK}/bin/hgcc"
    "${PYTHON}" -c 'import numpy; from gguf.quants import dequantize'
    args+=(--confirm-idle --sdk "${PPU_SDK}")
    [[ ${BUILD_ONLY:-0} != 1 ]] || args+=(--build-only)
    [[ -z ${PPU_SMI:-} ]] || args+=(--smi "${PPU_SMI}")
    export PATH="${PPU_SDK}/bin:${PATH}"
    export LD_LIBRARY_PATH="${PPU_SDK}/lib:${PPU_SDK}/targets/x86_64-linux/lib:${PPU_SDK}/CUDA_SDK/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
GPUS=${GPUS:-1,2,3,4,5,6,7}
PHASE=${PHASE:-all}
echo "DECODE_REFRESH_BOX run=${RUN} devices=${GPUS} phase=${PHASE} clock_GHz=${EXPECTED_CE_GHZ:-1.7} acu=0"
set +e
"${PYTHON}" -u "${SOURCE}/tools/run_decode_refresh.py" \
    --output "${RUN}/results" --devices "${GPUS}" --phase "${PHASE}" \
    --l2-bytes "${L2_BYTES:-67108864}" --jobs "${JOBS:-192}" \
    --expected-ce-ghz "${EXPECTED_CE_GHZ:-1.7}" "${args[@]}" \
    2>&1 | tee "${RUN}/console-$(date +%s%N).log"
rc=${PIPESTATUS[0]}
set -e
echo "DECODE_REFRESH_RESUME RESUME_RUN=${RUN} IDLE_CONFIRMED=1 bash tools/run_decode_refresh_box.sh"
echo "runner_rc=${rc}"
exit "${rc}"
