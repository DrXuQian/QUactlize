#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ ${IDLE_CONFIRMED:-0} == 1 ]] || { echo 'ERROR: confirm one otherwise idle GPU.' >&2; exit 1; }
[[ -z ${RUN:-} ]] || { echo 'ERROR: unset RUN; keep old results unchanged.' >&2; exit 1; }
PPU_SDK=${PPU_SDK:-/usr/local/PPU_SDK}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
[[ ${CUDA_VISIBLE_DEVICES} =~ ^[0-9]+$ ]] || { echo 'ERROR: select exactly one GPU.' >&2; exit 1; }
test -f "${PPU_SDK}/bin/hgcc"
test -x "${ACU:-${PPU_SDK}/asight/bin/acu}"
"${PYTHON:-python3}" -c 'import numpy; from gguf.quants import dequantize'
test -d "${OUTPUT_ROOT:-/workspace}"
RUN=$(mktemp -d "${OUTPUT_ROOT:-/workspace}/kpack-q6-schedule.XXXXXXXX")
[[ -n ${RUN} && -d ${RUN} && ! -L ${RUN} ]] || exit 1
SOURCE=${RUN}/source
mkdir "${SOURCE}"
REV=$(git -C "${ROOT}" rev-parse --verify HEAD)
ACTLIZE_REV=$(git -C "${ROOT}" rev-parse "${REV}:third_party/actlize")
[[ ${REV} =~ ^[0-9a-f]{40}$ && ${ACTLIZE_REV} =~ ^[0-9a-f]{40}$ ]] || exit 1
git -C "${ROOT}/third_party/actlize" cat-file -e "${ACTLIZE_REV}^{commit}"
git -C "${ROOT}" archive "${REV}" | tar -x -C "${SOURCE}"
mkdir -p "${SOURCE}/third_party/actlize"
git -C "${ROOT}/third_party/actlize" archive "${ACTLIZE_REV}" | tar -x -C "${SOURCE}/third_party/actlize"
echo "Q6_SCHEDULE_BOX run=${RUN} source=${REV} gpu=${CUDA_VISIBLE_DEVICES} points=${POINTS:-0} score=CE_CYCLES profiles=18"
set +e
"${PYTHON:-python3}" -u "${SOURCE}/tools/run_q6_schedule.py" \
    --sdk "${PPU_SDK}" --output "${RUN}/results" --l2-bytes "${L2_BYTES:-67108864}" \
    --points "${POINTS:-0}" --confirm-idle --acu "${ACU:-${PPU_SDK}/asight/bin/acu}" 2>&1 | tee "${RUN}/console.log"
rc=${PIPESTATUS[0]}
set -e
echo "results=${RUN}/results/review.tgz"
echo "runner_rc=${rc}"
exit "${rc}"
