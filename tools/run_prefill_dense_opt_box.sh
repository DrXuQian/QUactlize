#!/usr/bin/env bash
# Explicit candidate gate; never updates production selection or llama.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
PYTHON=${PYTHON:-python3}
GPU=${GPU:-1}
[[ ${GPU} =~ ^[0-9]+$ && ${IDLE_CONFIRMED:-0} == 1 ]] || {
    echo 'ERROR: choose one idle GPU and set IDLE_CONFIRMED=1.' >&2; exit 1;
}
[[ -z ${RUN:-} ]] || { echo 'ERROR: unset RUN; new candidate experiment only.' >&2; exit 1; }
PPU_SDK=$(realpath -e -- "${PPU_SDK:-/usr/local/PPU_SDK}")
test -x "${PPU_SDK}/bin/hgcc"
test -x "${PPU_SDK}/asight/bin/acu"
"${PYTHON}" -c 'import numpy,torch;from gguf.quants import dequantize'
REV=$(git -C "${ROOT}" rev-parse --verify HEAD)
ACT_REV=$(git -C "${ROOT}" rev-parse "${REV}:third_party/actlize")
[[ ${REV} =~ ^[0-9a-f]{40}$ && ${ACT_REV} =~ ^[0-9a-f]{40}$ ]]
git -C "${ROOT}/third_party/actlize" cat-file -e "${ACT_REV}^{commit}"
test -d "${OUTPUT_ROOT:-/workspace}"
RUN=$(mktemp -d "${OUTPUT_ROOT:-/workspace}/prefill-dense-opt.XXXXXXXX")
[[ -n ${RUN} && -d ${RUN} && ! -L ${RUN} ]]
mkdir "${RUN}/source"
git -C "${ROOT}" archive "${REV}" | tar -x -C "${RUN}/source"
mkdir -p "${RUN}/source/third_party/actlize"
git -C "${ROOT}/third_party/actlize" archive "${ACT_REV}" | tar -x -C "${RUN}/source/third_party/actlize"
export PATH="${PPU_SDK}/bin:${PATH}"
export LD_LIBRARY_PATH="${PPU_SDK}/lib:${PPU_SDK}/targets/x86_64-linux/lib:${PPU_SDK}/CUDA_SDK/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "DENSE_OPT_BOX run=${RUN} gpu=${GPU} shapes=7 arms=11 clock=1700 actual_clock=ADVISORY production_changed=0"
set +e
"${PYTHON}" -u "${RUN}/source/tools/run_prefill_dense_opt.py" --sdk "${PPU_SDK}" \
    --output "${RUN}/results" --device "${GPU}" --l2-bytes "${L2_BYTES:-67108864}" \
    --jobs "${JOBS:-4}" --confirm-idle 2>&1 | tee "${RUN}/console.log"
rc=${PIPESTATUS[0]}
set -e
echo "runner_rc=${rc} review=${RUN}/results/review-*.tgz"
echo "ACU=${RUN}/results/profile-*.acurep"
exit "${rc}"
