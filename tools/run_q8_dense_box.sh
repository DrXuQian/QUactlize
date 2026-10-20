#!/usr/bin/env bash
# Frozen local gitlinks, no production install and no nested network checkout.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ ${IDLE_CONFIRMED:-0} == 1 ]] || { echo 'ERROR: confirm one otherwise idle GPU.' >&2; exit 1; }
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
    test -f "${SOURCE}/tools/run_q8_dense.py"
    args+=(--resume)
else
    [[ -z ${RUN:-} ]] || { echo 'ERROR: unset RUN or use RESUME_RUN.' >&2; exit 1; }
    test -d "${OUTPUT_ROOT:-/workspace}"
    RUN=$(mktemp -d "${OUTPUT_ROOT:-/workspace}/kpack-q8-dense.XXXXXXXX")
    [[ -n ${RUN} && -d ${RUN} && ! -L ${RUN} ]] || exit 1
    SOURCE=${RUN}/source
    mkdir "${SOURCE}"
    REV=$(git -C "${ROOT}" rev-parse --verify HEAD)
    ACTLIZE_REV=$(git -C "${ROOT}" rev-parse "${REV}:third_party/actlize")
    [[ ${REV} =~ ^[0-9a-f]{40}$ && ${ACTLIZE_REV} =~ ^[0-9a-f]{40}$ ]] || exit 1
    if ! git -C "${ROOT}/third_party/actlize" cat-file -e "${ACTLIZE_REV}^{commit}"; then
        echo "ERROR: local actlize checkout must contain ${ACTLIZE_REV}; no nested network fetch attempted." >&2
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
echo "Q8_DENSE_BOX run=${RUN} gpu=${CUDA_VISIBLE_DEVICES} expected_ce_ghz=${EXPECTED_CE_GHZ:-1.7} production_changed=0"
set +e
"${PYTHON:-python3}" -u "${SOURCE}/tools/run_q8_dense.py" \
    --sdk "${PPU_SDK}" --output "${RUN}/results" --l2-bytes "${L2_BYTES:-67108864}" \
    --expected-ce-ghz "${EXPECTED_CE_GHZ:-1.7}" --confirm-idle "${args[@]}" 2>&1 | tee "${RUN}/console-$(date +%s%N).log"
rc=${PIPESTATUS[0]}
set -e
echo "Q8_DENSE_RESUME RESUME_RUN=${RUN} IDLE_CONFIRMED=1 EXPECTED_CE_GHZ=${EXPECTED_CE_GHZ:-1.7} bash tools/run_q8_dense_box.sh"
echo "runner_rc=${rc}"
exit "${rc}"
