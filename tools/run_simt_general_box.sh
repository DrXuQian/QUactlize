#!/usr/bin/env bash
# Frozen source and local gitlinks only. Never fetch nested HTTPS submodules.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ ${IDLE_CONFIRMED:-0} == 1 ]] || { echo 'ERROR: confirm an otherwise idle GPU.' >&2; exit 1; }
PPU_SDK=${PPU_SDK:-/usr/local/PPU_SDK}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
[[ ${CUDA_VISIBLE_DEVICES} =~ ^[0-9]+$ ]] || { echo 'ERROR: choose exactly one GPU.' >&2; exit 1; }
test -x "${PPU_SDK}/bin/hgcc"
"${PYTHON:-python3}" -c 'import numpy; from gguf.quants import dequantize'
args=()
if [[ -n ${RESUME_RUN:-} ]]; then
    RUN=$(realpath -e -- "${RESUME_RUN}")
    [[ -d ${RUN}/source && -f ${RUN}/results/plan.json && ! -L ${RUN} ]] || exit 1
    SOURCE=${RUN}/source
    # Resume the frozen implementation, not today's checkout.
    test -f "${SOURCE}/tools/run_simt_general.py"
    args+=(--resume)
else
    [[ -z ${RUN:-} ]] || { echo 'ERROR: unset RUN or use RESUME_RUN.' >&2; exit 1; }
    test -d "${OUTPUT_ROOT:-/workspace}"
    RUN=$(mktemp -d "${OUTPUT_ROOT:-/workspace}/kpack-simt-general.XXXXXXXX")
    [[ -n ${RUN} && -d ${RUN} && ! -L ${RUN} ]] || exit 1
    SOURCE=${RUN}/source
    mkdir "${SOURCE}"
    REV=$(git -C "${ROOT}" rev-parse --verify HEAD)
    ACTLIZE_REV=$(git -C "${ROOT}" rev-parse "${REV}:third_party/actlize")
    [[ ${REV} =~ ^[0-9a-f]{40}$ && ${ACTLIZE_REV} =~ ^[0-9a-f]{40}$ ]] || exit 1
    if ! git -C "${ROOT}/third_party/actlize" cat-file -e "${ACTLIZE_REV}^{commit}"; then
        echo "ERROR: local actlize checkout must contain pinned commit ${ACTLIZE_REV}; no nested network checkout attempted." >&2
        exit 1
    fi
    git -C "${ROOT}" archive "${REV}" | tar -x -C "${SOURCE}"
    mkdir -p "${SOURCE}/third_party/actlize"
    git -C "${ROOT}/third_party/actlize" archive "${ACTLIZE_REV}" | tar -x -C "${SOURCE}/third_party/actlize"
fi
if [[ ${PROFILE_ACU:-1} == 1 ]]; then
    test -x "${ACU:-${PPU_SDK}/asight/bin/acu}"
    args+=(--acu "${ACU:-${PPU_SDK}/asight/bin/acu}")
fi
echo "GENERAL_BOX run=${RUN} gpu=${CUDA_VISIBLE_DEVICES} points=${POINTS:-all} production_changed=0"
set +e
"${PYTHON:-python3}" -u "${SOURCE}/tools/run_simt_general.py" \
    --sdk "${PPU_SDK}" --output "${RUN}/results" --points "${POINTS:-all}" \
    --l2-bytes "${L2_BYTES:-67108864}" --confirm-idle "${args[@]}" 2>&1 | tee "${RUN}/console-$(date +%s%N).log"
rc=${PIPESTATUS[0]}
set -e
echo "GENERAL_BOX_RESUME RESUME_RUN=${RUN} IDLE_CONFIRMED=1 bash tools/run_simt_general_box.sh"
echo "runner_rc=${rc}"
exit "${rc}"
