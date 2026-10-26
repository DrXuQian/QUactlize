#!/usr/bin/env bash
# Bounded 10-point recovery; no event timings, production writes or network fetch.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
PYTHON=${PYTHON:-python3}
args=()
if [[ ${PLAN_ONLY:-0} != 1 ]]; then
    [[ ${IDLE_CONFIRMED:-0} == 1 ]] || {
        echo 'ERROR: confirm every assigned GPU is otherwise idle with IDLE_CONFIRMED=1.' >&2
        exit 1
    }
    PPU_SDK=${PPU_SDK:-/usr/local/PPU_SDK}
    test -x "${PPU_SDK}/bin/hgcc"
    test -x "${ASYS:-${PPU_SDK}/asight/bin/asys}"
    "${PYTHON}" -c 'import numpy; from gguf.quants import dequantize'
    args+=(--confirm-idle --sdk "${PPU_SDK}")
    [[ -z ${ASYS:-} ]] || args+=(--asys "${ASYS}")
    [[ -z ${PPU_SMI:-} ]] || args+=(--smi "${PPU_SMI}")
    export PATH="${PPU_SDK}/bin:${PATH}"
    export LD_LIBRARY_PATH="${PPU_SDK}/lib:${PPU_SDK}/targets/x86_64-linux/lib:${PPU_SDK}/CUDA_SDK/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
    args+=(--plan-only)
fi
if [[ -n ${RESUME_RUN:-} ]]; then
    RUN=$(realpath -e -- "${RESUME_RUN}")
    [[ -n ${RUN} && -d ${RUN}/source && -f ${RUN}/source/.quactlize-source.json && -f ${RUN}/results/plan.json ]] || {
        echo 'ERROR: resume requires the original frozen source and Asys results/plan.json.' >&2
        exit 1
    }
    SOURCE=${RUN}/source
    if [[ -f ${RUN}/results/inputs.json ]]; then
        GPUS=${GPUS:-$("${PYTHON}" -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["cards"]))' "${RUN}/results/inputs.json")}
    fi
    args+=(--resume)
else
    [[ -z ${RUN:-} ]] || { echo 'ERROR: unset RUN; use RESUME_RUN for an existing experiment.' >&2; exit 1; }
    REV=$(git -C "${ROOT}" rev-parse --verify HEAD)
    TREE=$(git -C "${ROOT}" rev-parse "${REV}^{tree}")
    ACTLIZE_REV=$(git -C "${ROOT}" rev-parse "${REV}:third_party/actlize")
    [[ ${REV} =~ ^[0-9a-f]{40}$ && ${TREE} =~ ^[0-9a-f]{40}$ && ${ACTLIZE_REV} =~ ^[0-9a-f]{40}$ ]] || exit 1
    git -C "${ROOT}" cat-file -e "${REV}:tools/run_smallm_regressions_asys.py"
    ACTLIZE_ROOT=$(realpath -e -- "${ROOT}/third_party/actlize")
    [[ $(git -C "${ACTLIZE_ROOT}" rev-parse --show-toplevel) == "${ACTLIZE_ROOT}" ]] || {
        echo 'ERROR: third_party/actlize must be an initialized local Git checkout.' >&2
        exit 1
    }
    if ! git -C "${ROOT}/third_party/actlize" cat-file -e "${ACTLIZE_REV}^{commit}"; then
        echo "ERROR: local actlize lacks pinned ${ACTLIZE_REV}; no HTTPS/submodule fetch attempted." >&2
        exit 1
    fi
    test -d "${OUTPUT_ROOT:-/workspace}"
    RUN=$(mktemp -d "${OUTPUT_ROOT:-/workspace}/kpack-smallm-asys.XXXXXXXX")
    [[ -n ${RUN} && -d ${RUN} && ! -L ${RUN} ]] || exit 1
    SOURCE=${RUN}/source
    mkdir "${SOURCE}"
    git -C "${ROOT}" archive "${REV}" | tar -x -C "${SOURCE}"
    mkdir -p "${SOURCE}/third_party/actlize"
    git -C "${ROOT}/third_party/actlize" archive "${ACTLIZE_REV}" | tar -x -C "${SOURCE}/third_party/actlize"
    "${PYTHON}" - "${SOURCE}" "${REV}" "${TREE}" "${ACTLIZE_REV}" <<'PY'
import json,sys
from pathlib import Path
root,commit,tree,actlize=sys.argv[1:]
with (Path(root)/'.quactlize-source.json').open('x') as f:
    json.dump(dict(quactlize_commit=commit,quactlize_tree=tree,actlize_commit=actlize,
                   snapshot='COMMITTED_ARCHIVE'),f,sort_keys=True)
PY
fi
test -f "${SOURCE}/tools/run_smallm_regressions_asys.py"
SOURCE_COMMIT=$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["quactlize_commit"])' "${SOURCE}/.quactlize-source.json")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
GPUS=${GPUS:-0,1,2,3}
echo "SMALLM_ASYS_BOX run=${RUN} devices=${GPUS} points=10 commit=${SOURCE_COMMIT} clock_request_MHz=1700 actual_clock=ADVISORY production_changed=0"
set +e
"${PYTHON}" -u "${SOURCE}/tools/run_smallm_regressions_asys.py" \
    --output "${RUN}/results" --devices "${GPUS}" --jobs "${JOBS:-8}" \
    --l2-bytes "${L2_BYTES:-67108864}" "${args[@]}" \
    2>&1 | tee "${RUN}/console-$(date +%s%N).log"
rc=${PIPESTATUS[0]}
set -e
echo "SMALLM_ASYS_RESUME RESUME_RUN=${RUN} IDLE_CONFIRMED=1 bash tools/run_smallm_regressions_asys_box.sh"
echo "Asys=${RUN}/results/gpu-*/attempts/*/trace.asysrep"
echo "runner_rc=${rc}; upload only results/review-lite-*.tar.xz; original traces stay on Box."
echo 'Current Docker shell is preserved.'
exit "${rc}"
