#!/usr/bin/env bash
# Generic GGUF/TP sweep handoff. No model registry, network fetch or product writes.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
PYTHON=${PYTHON:-python3}
stage=sweep
case ${1:-} in plan|prepare|sweep|export) stage=$1; shift ;; esac
for arg in "$@"; do
    case ${arg} in --output|--output=*|--resume)
        echo 'ERROR: this wrapper owns --output; use RESUME_RUN for an existing run.' >&2; exit 1 ;;
    esac
done
if [[ ${stage} == sweep ]]; then
    [[ ${IDLE_CONFIRMED:-0} == 1 ]] || {
        echo 'ERROR: confirm the assigned GPUs are idle with IDLE_CONFIRMED=1.' >&2; exit 1;
    }
    PPU_SDK=${PPU_SDK:-/usr/local/PPU_SDK}
    test -x "${PPU_SDK}/bin/hgcc"
    test -x "${ASYS:-${PPU_SDK}/asight/bin/asys}"
    export PATH="${PPU_SDK}/bin:${PATH}"
    export LD_LIBRARY_PATH="${PPU_SDK}/lib:${PPU_SDK}/targets/x86_64-linux/lib:${PPU_SDK}/CUDA_SDK/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    "${PYTHON}" -c 'import numpy, torch; from gguf.quants import dequantize'
fi
args=()
if [[ -n ${RESUME_RUN:-} ]]; then
    TUNE_RUN=$(realpath -e -- "${RESUME_RUN}")
    test -f "${TUNE_RUN}/source/.quactlize-source.json"
    test -f "${TUNE_RUN}/results/inventory.json"
    if [[ -f ${TUNE_RUN}/results/runs/coordinator.json ]]; then
        GPUS=${GPUS:-$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["devices"])' "${TUNE_RUN}/results/runs/coordinator.json")}
    fi
    args+=(--resume)
else
    [[ ${stage} != export ]] || { echo 'ERROR: export requires RESUME_RUN.' >&2; exit 1; }
    REV=$(git -C "${ROOT}" rev-parse --verify HEAD)
    TREE=$(git -C "${ROOT}" rev-parse "${REV}^{tree}")
    ACTLIZE_REV=$(git -C "${ROOT}" rev-parse "${REV}:third_party/actlize")
    [[ ${REV} =~ ^[0-9a-f]{40}$ && ${TREE} =~ ^[0-9a-f]{40}$ && ${ACTLIZE_REV} =~ ^[0-9a-f]{40}$ ]] || exit 1
    git -C "${ROOT}" cat-file -e "${REV}:tools/tune_model.py"
    ACTLIZE_ROOT=$(realpath -e -- "${ACTLIZE_ROOT:-${ROOT}/third_party/actlize}")
    [[ $(git -C "${ACTLIZE_ROOT}" rev-parse --show-toplevel) == "${ACTLIZE_ROOT}" ]] || {
        echo 'ERROR: actlize must be an initialized local Git checkout.' >&2; exit 1;
    }
    if ! git -C "${ACTLIZE_ROOT}" cat-file -e "${ACTLIZE_REV}^{commit}"; then
        echo "ERROR: local actlize lacks pinned ${ACTLIZE_REV}; no submodule/network update attempted." >&2
        exit 1
    fi
    TUNE_PARENT=${OUTPUT_ROOT:-/workspace}
    test -d "${TUNE_PARENT}"
    TUNE_RUN=$(mktemp -d "${TUNE_PARENT}/kpack-model-tune.XXXXXXXX")
    [[ -n ${TUNE_RUN} && -d ${TUNE_RUN} && ! -L ${TUNE_RUN} ]] || exit 1
    mkdir "${TUNE_RUN}/source"
    git -C "${ROOT}" archive "${REV}" | tar -x -C "${TUNE_RUN}/source"
    mkdir -p "${TUNE_RUN}/source/third_party/actlize"
    git -C "${ACTLIZE_ROOT}" archive "${ACTLIZE_REV}" | tar -x -C "${TUNE_RUN}/source/third_party/actlize"
    "${PYTHON}" - "${TUNE_RUN}/source/.quactlize-source.json" "${REV}" "${TREE}" "${ACTLIZE_REV}" <<'PY'
import json,sys
path,commit,tree,actlize=sys.argv[1:]
with open(path,'x') as out:
    json.dump(dict(quactlize_commit=commit,quactlize_tree=tree,actlize_commit=actlize,
                   snapshot='COMMITTED_ARCHIVE'),out,sort_keys=True)
PY
fi
SOURCE=${TUNE_RUN}/source
test -f "${SOURCE}/tools/tune_model.py"
args+=(--output "${TUNE_RUN}/results" --devices "${GPUS:-0,1,2,3}" --jobs "${JOBS:-8}")
[[ -z ${LLAMA_CI_DIR:-} ]] || args+=(--llama-source "${LLAMA_CI_DIR}")
if [[ ${stage} == sweep ]]; then
    args+=(--sdk "${PPU_SDK}" --l2-bytes "${L2_BYTES:-67108864}" --confirm-idle)
    [[ -z ${ASYS:-} ]] || args+=(--asys "${ASYS}")
    [[ -z ${PPU_SMI:-} ]] || args+=(--smi "${PPU_SMI}")
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "MODEL_TUNING_BOX run=${TUNE_RUN} stage=${stage} devices=${GPUS:-0,1,2,3} production_changed=0"
set +e
"${PYTHON}" -u "${SOURCE}/tools/tune_model.py" "${stage}" "${args[@]}" "$@" \
    2>&1 | tee "${TUNE_RUN}/console-$(date +%s%N).log"
rc=${PIPESTATUS[0]}
set -e
(cd "${SOURCE}" && "${PYTHON}" -m dev.model_tuning.review "${TUNE_RUN}")
echo "Resume: RESUME_RUN=${TUNE_RUN} IDLE_CONFIRMED=1 bash tools/tune_model_box.sh"
echo "runner_rc=${rc}; original Asys reports/cache remain under ${TUNE_RUN}/results/runs/"
echo 'Current Docker shell is preserved.'
exit "${rc}"
