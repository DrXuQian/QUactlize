#!/usr/bin/env bash
# Freeze committed source and local pinned actlize; no network submodule fetch.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
PYTHON=${PYTHON:-python3}
args=()
if [[ -n ${RESUME_RUN:-} ]]; then
    [[ -z ${RUN:-} ]] || { echo 'ERROR: unset RUN when using RESUME_RUN.' >&2; exit 1; }
    RUN=$(realpath -e -- "${RESUME_RUN}")
    [[ -n ${RUN} && -d ${RUN}/source && -f ${RUN}/results/finalists.json ]] || {
        echo 'ERROR: RESUME_RUN needs frozen source and results/finalists.json.' >&2
        exit 1
    }
    SOURCE=${RUN}/source
    GPUS=${GPUS:-$("${PYTHON}" -c 'import json,sys;print(",".join(json.load(open(sys.argv[1]))["cards"]))' \
        "${RUN}/results/finalists.json")}
    if [[ -f ${RUN}/results/run-options.json ]]; then
        PREFILL_REFRESH_SCOPE=${PREFILL_REFRESH_SCOPE:-$("${PYTHON}" -c \
            'import json,sys;print(json.load(open(sys.argv[1]))["scope"])' "${RUN}/results/run-options.json")}
        REUSE_BUILD=${REUSE_BUILD:-$("${PYTHON}" -c \
            'import json,sys;print(json.load(open(sys.argv[1]))["reuse_build"] or "")' "${RUN}/results/run-options.json")}
    fi
    args+=(--resume)
else
    [[ -z ${RUN:-} ]] || { echo 'ERROR: unset RUN or use RESUME_RUN.' >&2; exit 1; }
    test -d "${OUTPUT_ROOT:-/workspace}"
    REV=$(git -C "${ROOT}" rev-parse --verify HEAD)
    ACTLIZE_REV=$(git -C "${ROOT}" rev-parse "${REV}:third_party/actlize")
    [[ ${REV} =~ ^[0-9a-f]{40}$ && ${ACTLIZE_REV} =~ ^[0-9a-f]{40}$ ]] || exit 1
    if ! git -C "${ROOT}/third_party/actlize" cat-file -e "${ACTLIZE_REV}^{commit}"; then
        echo "ERROR: local actlize lacks pinned ${ACTLIZE_REV}; no network checkout attempted." >&2
        exit 1
    fi
    RUN=$(mktemp -d "${OUTPUT_ROOT:-/workspace}/kpack-prefill-refresh.XXXXXXXX")
    [[ -n ${RUN} && -d ${RUN} && ! -L ${RUN} ]] || exit 1
    SOURCE=${RUN}/source
    mkdir "${SOURCE}"
    git -C "${ROOT}" archive "${REV}" | tar -x -C "${SOURCE}"
    mkdir -p "${SOURCE}/third_party/actlize"
    git -C "${ROOT}/third_party/actlize" archive "${ACTLIZE_REV}" | tar -x -C "${SOURCE}/third_party/actlize"
fi
test -f "${SOURCE}/tools/run_prefill_refresh.py"
PREFILL_REFRESH_SCOPE=${PREFILL_REFRESH_SCOPE:-full}
[[ ${PREFILL_REFRESH_SCOPE} == full || ${PREFILL_REFRESH_SCOPE} == reconfirm ]] || {
    echo 'ERROR: PREFILL_REFRESH_SCOPE must be full or reconfirm.' >&2; exit 1;
}
# Old frozen runs have no --scope option. Their original full resume remains
# valid; switching a completed full run into a retest requires a NEW run.
if [[ -z ${RESUME_RUN:-} || -f ${RUN}/results/run-options.json ]]; then
    args+=(--scope "${PREFILL_REFRESH_SCOPE}")
else
    [[ ${PREFILL_REFRESH_SCOPE} == full && -z ${REUSE_BUILD:-} ]] || {
        echo 'ERROR: create a fresh reconfirm run; do not change an old frozen run.' >&2; exit 1;
    }
fi
[[ -z ${REUSE_BUILD:-} ]] || args+=(--reuse-build "${REUSE_BUILD}")
if [[ ${PLAN_ONLY:-0} == 1 ]]; then
    args+=(--plan-only)
else
    [[ ${IDLE_CONFIRMED:-0} == 1 ]] || {
        echo 'ERROR: set IDLE_CONFIRMED=1 only after checking every assigned card is idle.' >&2
        exit 1
    }
    PPU_SDK=${PPU_SDK:-/usr/local/PPU_SDK}
    test -x "${PPU_SDK}/bin/hgcc"
    args+=(--confirm-idle --sdk "${PPU_SDK}")
    [[ ${BUILD_ONLY:-0} != 1 ]] || args+=(--build-only)
    [[ -z ${PPU_SMI:-} ]] || args+=(--smi "${PPU_SMI}")
    export PATH="${PPU_SDK}/bin:${PATH}"
    export LD_LIBRARY_PATH="${PPU_SDK}/lib:${PPU_SDK}/targets/x86_64-linux/lib:${PPU_SDK}/CUDA_SDK/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    "${PYTHON}" -c 'import numpy,torch;from gguf.quants import dequantize'
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ ${PREFILL_REFRESH_SCOPE} == reconfirm ]]; then
    GPUS=${GPUS:-1,2,3,4,5}
else
    GPUS=${GPUS:-1,2,3,4,5,6,7}
fi
echo "PREFILL_REFRESH_BOX run=${RUN} scope=${PREFILL_REFRESH_SCOPE} devices=${GPUS} clock_MHz=1700 actual_clock=ADVISORY cuBLAS_graph=0 asys=0 acu=0"
set +e
"${PYTHON}" -u "${SOURCE}/tools/run_prefill_refresh.py" \
    --output "${RUN}/results" --devices "${GPUS}" \
    --l2-bytes "${L2_BYTES:-67108864}" --jobs "${JOBS:-192}" \
    --expected-ce-ghz "${EXPECTED_CE_GHZ:-1.7}" "${args[@]}" \
    2>&1 | tee "${RUN}/console-$(date +%s%N).log"
rc=${PIPESTATUS[0]}
set -e
echo "PREFILL_REFRESH_RESUME RESUME_RUN=${RUN} IDLE_CONFIRMED=1 bash tools/run_prefill_refresh_box.sh"
echo "runner_rc=${rc}"
exit "${rc}"
