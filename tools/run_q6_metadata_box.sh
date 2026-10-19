#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ ${IDLE_CONFIRMED:-0} == 1 ]] || { echo 'ERROR: confirm one otherwise idle GPU.' >&2; exit 1; }
[[ -z ${RUN:-} ]] || { echo 'ERROR: unset RUN; use a fresh bounded experiment directory.' >&2; exit 1; }
PPU_SDK=${PPU_SDK:-/usr/local/PPU_SDK}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
[[ ${CUDA_VISIBLE_DEVICES} =~ ^[0-9]+$ ]] || { echo 'ERROR: select exactly one GPU.' >&2; exit 1; }
test -f "${PPU_SDK}/bin/hgcc"
"${PYTHON:-python3}" -c 'import numpy; from gguf.quants import dequantize'
test -d "${OUTPUT_ROOT:-/workspace}"
RUN=$(mktemp -d "${OUTPUT_ROOT:-/workspace}/kpack-q6-metadata.XXXXXXXX")
[[ -d ${RUN} && ! -L ${RUN} ]] || exit 1
SOURCE=${RUN}/source
mkdir "${SOURCE}"
REV=$(git -C "${ROOT}" rev-parse --verify HEAD)
ACTLIZE_REV=$(git -C "${ROOT}" rev-parse "${REV}:third_party/actlize")
[[ ${REV} =~ ^[0-9a-f]{40}$ && ${ACTLIZE_REV} =~ ^[0-9a-f]{40}$ ]] || exit 1
git -C "${ROOT}/third_party/actlize" cat-file -e "${ACTLIZE_REV}^{commit}"
git -C "${ROOT}" archive "${REV}" | tar -x -C "${SOURCE}"
mkdir -p "${SOURCE}/third_party/actlize"
git -C "${ROOT}/third_party/actlize" archive "${ACTLIZE_REV}" | tar -x -C "${SOURCE}/third_party/actlize"
args=()
if [[ ${PROFILE_ACU:-1} == 1 ]]; then
    test -x "${ACU:-${PPU_SDK}/asight/bin/acu}"
    args+=(--acu "${ACU:-${PPU_SDK}/asight/bin/acu}")
fi
echo "Q6_METADATA_START run=${RUN} source=${REV} gpu=${CUDA_VISIBLE_DEVICES}"
echo '8 points, one unchanged config, old/new Q6 only; no native comparison, no caller rebuild.'
set +e
"${PYTHON:-python3}" -u "${SOURCE}/tools/run_q6_metadata.py" \
    --sdk "${PPU_SDK}" --output "${RUN}/results" --l2-bytes "${L2_BYTES:-67108864}" \
    --confirm-idle "${args[@]}" 2>&1 | tee "${RUN}/console.log"
rc=${PIPESTATUS[0]}
set -e
echo "results=${RUN}/results/review.tgz"
echo "runner_rc=${rc}"
exit "${rc}"
